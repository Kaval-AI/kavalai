"""The experiment: repeats, concurrency, aggregation, the gate and the diff."""

import json

import pytest
import yaml

from kavalai.eval import (
    Case,
    Experiment,
    ExperimentResult,
    RunRecord,
    Suite,
    assert_suite_passes,
    diff_against,
    external_id_for,
)
from kavalai.eval.models import CaseResult, CaseVerdict, Score, Totals
from kavalai.eval.runner import BudgetExceeded, _verdict, load_setup
from kavalai.eval.targets import Target


class ScriptedTarget(Target):
    """Returns a canned record per case name, counting how often it was asked."""

    observes_trajectory = True

    def __init__(self, outputs: dict, fail: set = frozenset()):
        self.outputs = outputs
        self.fail = fail
        self.calls: list[tuple[str, str | None]] = []

    async def setup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def run(self, case: Case, external_id=None) -> RunRecord:
        self.calls.append((case.name, external_id))
        if case.name in self.fail:
            return RunRecord(status="failed", error="boom")
        return RunRecord(output={"agent_response": self.outputs.get(case.name, "hi")})

    def describe(self) -> dict:
        return {"kind": "scripted"}


def write_suite(tmp_path, cases, **suite) -> Suite:
    (tmp_path / "cases.yaml").write_text(
        yaml.safe_dump({"name": "d", "cases": cases}), encoding="utf-8"
    )
    suite.setdefault("name", "s")
    suite.setdefault("dataset", "cases.yaml")
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite), encoding="utf-8")
    return Suite.from_yaml(tmp_path / "suite.yaml")


CASES = [
    {
        "name": "good",
        "slice": "a",
        "inputs": {},
        "evaluators": [{"type": "contains", "text": "hi"}],
    },
    {
        "name": "bad",
        "slice": "a",
        "inputs": {},
        "evaluators": [{"type": "contains", "text": "nope"}],
    },
]


async def test_a_run_grades_every_case_and_aggregates(tmp_path):
    suite = write_suite(tmp_path, CASES, evaluators=["no_error"])
    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()

    assert result.totals.cases == 2
    assert (result.totals.passed, result.totals.failed) == (1, 1)
    assert result.totals.pass_rate == 0.5
    assert result.status_of("bad") == "failed"
    assert [s.name for s in result.slices] == ["a"]
    assert result.target == {"kind": "scripted"}


async def test_an_errored_run_is_not_a_failing_grade(tmp_path):
    """ "The harness broke" and "the workflow is wrong" call for different people."""
    suite = write_suite(tmp_path, CASES, evaluators=["no_error"])
    target = ScriptedTarget({}, fail={"good"})
    result = await Experiment(suite, tag="t", target=target).run()

    assert result.totals.errors == 1
    assert result.status_of("good") == "error"
    assert any("errored" in reason for reason in result.gate.reasons)


async def test_repeats_vote_and_report_flakiness():
    """A judged case failing one of three is flaky, not a blocker."""
    runs = [
        CaseResult(case="c", repeat=0, status="passed"),
        CaseResult(case="c", repeat=1, status="failed"),
        CaseResult(case="c", repeat=2, status="passed"),
    ]
    verdict = _verdict("c", runs)
    assert verdict.status == "flaky" and verdict.flaky is True

    assert (
        _verdict("c", [r.model_copy(update={"status": "passed"}) for r in runs]).status
        == "passed"
    )
    assert (
        _verdict("c", [r.model_copy(update={"status": "failed"}) for r in runs]).status
        == "failed"
    )
    # A majority cannot vote away a run that broke.
    with_error = runs[:2] + [CaseResult(case="c", repeat=2, status="error")]
    assert _verdict("c", with_error).status == "error"
    # A tie is not a majority.
    assert _verdict("c", runs[:2]).status == "failed"


async def test_repeats_run_the_case_that_many_times(tmp_path):
    suite = write_suite(tmp_path, CASES[:1], repeats=3)
    target = ScriptedTarget({})
    await Experiment(suite, tag="t", target=target).run()
    assert len(target.calls) == 3


async def test_external_ids_are_only_minted_when_asked_for(tmp_path):
    suite = write_suite(tmp_path, CASES[:1])
    target = ScriptedTarget({})
    await Experiment(suite, tag="t", target=target).run()
    assert target.calls == [("good", None)]

    target = ScriptedTarget({})
    await Experiment(suite, tag="pr-1", target=target, persist_sessions=True).run()
    assert target.calls == [("good", "eval:s:pr-1:good:0")]


def test_the_external_id_format_is_a_documented_prefix():
    assert (
        external_id_for("bakery", "pr-412", "vague", 0) == "eval:bakery:pr-412:vague:0"
    )


async def test_slice_thresholds_are_checked_independently(tmp_path):
    suite = write_suite(
        tmp_path,
        CASES,
        evaluators=["no_error"],
        slices={"a": {"min_pass_rate": 1.0}},
        gate={"min_pass_rate": 0.0},
    )
    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
    assert result.slices[0].ok is False
    assert result.gate.passed is False
    assert "slice 'a'" in result.gate.reasons[0]


async def test_a_required_evaluator_fails_the_run_outright(tmp_path):
    suite = write_suite(
        tmp_path,
        CASES,
        evaluators=["no_error"],
        gate={"min_pass_rate": 0.0, "required_evaluators": ["contains"]},
    )
    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
    assert result.gate.passed is False
    assert "required evaluator 'contains' failed on: bad" in result.gate.reasons


async def test_a_regression_against_the_baseline_fails_the_gate(tmp_path):
    suite = write_suite(
        tmp_path, CASES, evaluators=["no_error"], gate={"min_pass_rate": 0.0}
    )
    baseline = ExperimentResult(
        suite="s",
        tag="baseline",
        verdicts=[
            CaseVerdict(case="good", status="passed"),
            CaseVerdict(case="bad", status="passed"),
        ],
    )
    baseline.to_json(tmp_path / "baseline.json")

    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
    assert result.gate.regressions == ["bad"]
    assert result.gate.passed is False


async def test_a_newly_passing_case_is_reported_but_does_not_fail(tmp_path):
    suite = write_suite(
        tmp_path, CASES[:1], evaluators=["no_error"], gate={"min_pass_rate": 0.0}
    )
    ExperimentResult(
        suite="s", tag="b", verdicts=[CaseVerdict(case="good", status="failed")]
    ).to_json(tmp_path / "baseline.json")

    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
    assert result.gate.fixes == ["good"]
    assert result.gate.passed is True


def test_diff_ignores_cases_the_baseline_never_saw():
    baseline = ExperimentResult(
        suite="s", tag="b", verdicts=[CaseVerdict(case="old", status="passed")]
    )
    current = ExperimentResult(
        suite="s",
        tag="c",
        verdicts=[
            CaseVerdict(case="old", status="failed"),
            CaseVerdict(case="brand_new", status="failed"),
        ],
    )
    assert diff_against(baseline, current) == (["old"], [])


async def test_the_token_budget_stops_the_run_rather_than_billing_on(tmp_path):
    class Expensive(ScriptedTarget):
        async def run(self, case, external_id=None):
            from kavalai.eval.targets import ModelCallRecord

            await super().run(case, external_id)
            return RunRecord(
                output={"agent_response": "hi"},
                model_calls=[ModelCallRecord(total_tokens=1000)],
            )

    cases = [{"name": f"c{i}", "inputs": {}} for i in range(6)]
    suite = write_suite(
        tmp_path,
        cases,
        concurrency=1,
        evaluators=["no_error"],
        gate={"max_tokens": 2000},
    )
    with pytest.raises(BudgetExceeded, match="ceiling"):
        await Experiment(suite, tag="t", target=Expensive({})).run()


async def test_an_empty_suite_says_where_it_looked(tmp_path):
    suite = write_suite(tmp_path, [])
    with pytest.raises(ValueError, match="has no cases"):
        await Experiment(suite, tag="t", target=ScriptedTarget({})).run()


async def test_an_evaluator_that_raises_fails_its_case_and_says_why(tmp_path):
    """A broken evaluator must never be scored as a pass."""
    from kavalai.eval import Evaluator, evaluator
    from kavalai.eval.evaluators.base import REGISTRY

    @evaluator("test_only_explodes", replace=True)
    class Explodes(Evaluator):
        async def score(self, case, record):
            raise RuntimeError("kaboom")

    try:
        suite = write_suite(
            tmp_path, [{"name": "c", "inputs": {}}], evaluators=["test_only_explodes"]
        )
        result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
        assert result.status_of("c") == "failed"
        assert "kaboom" in result.results[0].scores[0].reason
    finally:
        del REGISTRY["test_only_explodes"]


class Blind(ScriptedTarget):
    """A target that cannot see a trajectory — a deployed agent behind HTTP."""

    observes_trajectory = False


async def test_a_blind_target_refuses_to_run_trajectory_assertions(tmp_path):
    """Said once, up front, rather than as 128 identical failures.

    The worst outcome available is a gate that reports green because it saw
    nothing; the second worst is burying the one line the operator needs to
    read under a wall of red. So: refuse to start, and name the assertions.
    """
    suite = write_suite(
        tmp_path,
        [{"name": "c", "inputs": {}}],
        evaluators=[{"type": "node_visited", "node": "x"}],
    )
    with pytest.raises(ValueError, match="cannot observe one"):
        await Experiment(suite, tag="t", target=Blind({})).run()


async def test_dropping_them_is_explicit_and_reported(tmp_path):
    suite = write_suite(
        tmp_path,
        [{"name": "c", "inputs": {}}],
        evaluators=["no_error", {"type": "node_visited", "node": "x"}],
    )
    result = await Experiment(
        suite, tag="t", target=Blind({}), skip_trajectory_evaluators=True
    ).run()

    assert result.status_of("c") == "passed"
    # Never silently: the report says what did not run.
    assert any("node_visited" in note for note in result.notes)
    assert [s.name for s in result.results[0].scores] == ["no_error"]


async def test_an_evaluation_error_mid_run_still_fails_its_case(tmp_path):
    """A trajectory evaluator reached at run time refuses, and says why."""
    from kavalai.eval.evaluators.base import EvaluationError, build_evaluator

    evaluator = build_evaluator({"type": "node_visited", "node": "x"})
    with pytest.raises(EvaluationError, match="does not produce one"):
        await evaluator.score(Case(name="c", inputs={}), RunRecord())


async def test_assert_suite_passes_reports_every_failing_case(tmp_path):
    suite = write_suite(
        tmp_path, CASES, evaluators=["no_error"], gate={"min_pass_rate": 1.0}
    )
    result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()

    with pytest.raises(AssertionError) as excinfo:
        assert_suite_passes(result)
    assert "bad" in str(excinfo.value)

    result.gate.passed = True
    assert_suite_passes(result)


def test_load_setup_imports_a_module_by_path(tmp_path):
    module = tmp_path / "setup_module.py"
    module.write_text("MARKER = 42\n")
    assert load_setup(module).MARKER == 42

    with pytest.raises(FileNotFoundError):
        load_setup(tmp_path / "missing.py")


async def test_the_suite_setup_module_is_imported_before_the_run(tmp_path):
    (tmp_path / "prepare.py").write_text(
        "from kavalai.eval import Evaluator, Score, evaluator\n"
        "@evaluator('test_only_from_setup', replace=True)\n"
        "class FromSetup(Evaluator):\n"
        "    async def score(self, case, record):\n"
        "        return Score.boolean(self.name, True)\n"
    )
    suite = write_suite(
        tmp_path,
        [{"name": "c", "inputs": {}}],
        setup="prepare.py",
        evaluators=["test_only_from_setup"],
    )
    from kavalai.eval.evaluators.base import REGISTRY

    try:
        result = await Experiment(suite, tag="t", target=ScriptedTarget({})).run()
        assert result.status_of("c") == "passed"
    finally:
        REGISTRY.pop("test_only_from_setup", None)
