"""The suite/dataset file format: what a person may write, and what it becomes."""

import json

import pytest
import yaml

from kavalai.eval import (
    Case,
    Dataset,
    EvaluatorSpec,
    ExperimentResult,
    Suite,
)
from kavalai.eval.models import CaseResult, CaseVerdict, Score, Totals


def test_evaluator_spec_accepts_both_shorthand_forms():
    """``- no_error`` and ``- {type: contains, text: x}`` are both valid YAML."""
    assert EvaluatorSpec.coerce("no_error").type == "no_error"
    assert EvaluatorSpec.coerce("no_error").options == {}

    spec = EvaluatorSpec.coerce({"type": "contains", "text": "60 days"})
    assert spec.type == "contains"
    assert spec.options == {"text": "60 days"}

    # An existing spec passes through unchanged.
    assert EvaluatorSpec.coerce(spec) is spec


def test_bare_strings_are_coerced_before_validation(tmp_path):
    """The shorthand has to survive model validation, not just post-init."""
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "s",
                "dataset": "cases.yaml",
                "evaluators": ["no_error", {"type": "contains", "text": "x"}],
                "slices": {"a": {"evaluators": ["no_error"], "min_pass_rate": 1.0}},
            }
        )
    )
    suite = Suite.from_yaml(path)
    assert [e.type for e in suite.evaluators] == ["no_error", "contains"]
    assert [e.type for e in suite.slices["a"].evaluators] == ["no_error"]


def test_suite_resolves_every_path_against_its_own_directory(tmp_path):
    """A suite is a directory you can copy anywhere."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "suite.yaml").write_text(
        yaml.safe_dump(
            {"name": "s", "dataset": "cases/qa.yaml", "baseline": "baseline.json"}
        )
    )
    suite = Suite.from_yaml(tmp_path / "eval" / "suite.yaml")
    assert suite.dataset_paths() == [tmp_path / "eval" / "cases" / "qa.yaml"]
    assert suite.baseline_path() == tmp_path / "eval" / "baseline.json"
    assert suite.result_path("pr-1") == tmp_path / "eval" / "results" / "pr-1.json"
    # An absolute path is left alone.
    assert suite.resolve("/tmp/x.yaml").as_posix() == "/tmp/x.yaml"


def test_dataset_round_trips_through_yaml(tmp_path):
    dataset = Dataset(
        name="d",
        evaluators=["no_error"],
        cases=[Case(name="a", inputs={"q": 1}, slice="direct")],
    )
    path = tmp_path / "d.yaml"
    dataset.to_yaml(path)
    loaded = Dataset.from_yaml(path)
    assert loaded.cases[0].name == "a"
    assert loaded.slices() == {"direct"}


def test_dataset_name_defaults_to_the_file_name(tmp_path):
    path = tmp_path / "orders.yaml"
    path.write_text(yaml.safe_dump({"cases": []}))
    assert Dataset.from_yaml(path).name == "orders"


def test_merging_several_datasets_keeps_their_own_evaluators(tmp_path):
    for name, evaluator in (("a", "no_error"), ("b", "output_not_empty")):
        (tmp_path / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {"evaluators": [evaluator], "cases": [{"name": name, "inputs": {}}]}
            )
        )
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"name": "s", "dataset": ["a.yaml", "b.yaml"]})
    )
    dataset = Suite.from_yaml(tmp_path / "suite.yaml").load_dataset()
    assert [c.name for c in dataset.cases] == ["a", "b"]
    assert [e.type for e in dataset.cases[0].evaluators] == ["no_error"]
    assert [e.type for e in dataset.cases[1].evaluators] == ["output_not_empty"]


def test_evaluators_for_layers_suite_dataset_slice_and_case(tmp_path):
    """Most general first, so a case's own evaluators can rely on the rest."""
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "s",
                "dataset": "d.yaml",
                "evaluators": ["no_error"],
                "slices": {"direct": {"evaluators": ["output_not_empty"]}},
            }
        )
    )
    suite = Suite.from_yaml(tmp_path / "suite.yaml")
    dataset = Dataset(name="d", evaluators=["regex"])
    case = Case(name="c", inputs={}, slice="direct", evaluators=["contains"])

    assert [e.type for e in suite.evaluators_for(case, dataset)] == [
        "no_error",
        "regex",
        "output_not_empty",
        "contains",
    ]


def test_score_boolean_helper():
    passed = Score.boolean("n", True)
    assert (passed.value, passed.passed) == (1.0, True)
    failed = Score.boolean("n", False, reason="because")
    assert (failed.value, failed.passed, failed.reason) == (0.0, False, "because")


def test_measured_scores_are_not_assertions():
    """``passed=None`` means measured, not asserted — it must not fail a case."""
    result = CaseResult(case="c", scores=[Score(name="tokens", value=42.0)])
    assert result.failed_scores == []


def test_experiment_result_round_trips_to_json(tmp_path):
    result = ExperimentResult(
        suite="s",
        tag="t",
        totals=Totals(cases=1, passed=1, pass_rate=1.0),
        verdicts=[CaseVerdict(case="a", status="passed", passes=1, total=1)],
    )
    path = tmp_path / "r.json"
    result.to_json(path)
    assert json.loads(path.read_text())["suite"] == "s"

    loaded = ExperimentResult.from_json(path)
    assert loaded.status_of("a") == "passed"
    assert loaded.status_of("nope") is None


def test_a_verdict_is_flaky_only_between_all_and_nothing():
    assert CaseVerdict(case="a", passes=1, total=3).flaky is True
    assert CaseVerdict(case="a", passes=3, total=3).flaky is False
    assert CaseVerdict(case="a", passes=0, total=3).flaky is False


def test_suite_without_a_baseline_file_loads_none(tmp_path):
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"name": "s", "dataset": "d.yaml"})
    )
    assert Suite.from_yaml(tmp_path / "suite.yaml").load_baseline() is None
