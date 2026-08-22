"""The command line: the only place in `kavalai.eval` that reads the environment."""

import json

import pytest
import yaml

from kavalai.eval import ExperimentResult
from kavalai.eval.cli import (
    EXIT_ERROR,
    EXIT_GATE_FAILED,
    EXIT_OK,
    build_parser,
    expand_env,
    main,
)
from kavalai.eval.models import CaseVerdict, Totals

WORKFLOW = {
    "name": "wf",
    "description": "d",
    "llm_model": "openai/fake",
    "data_types": {
        "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
        "output": {
            "type": "object",
            "properties": {"agent_response": {"type": "string"}},
        },
    },
    "nodes": [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "hi",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ],
}


def write_fixtures(path, chat_texts: list[str], response: str) -> None:
    """Pre-record the model responses this suite will need.

    The suite runs with ``--fixtures``, which is the product's own keyless
    path — no provider, and no monkeypatching of a module other tests share.
    """
    import json

    from kavalai.llm_clients.base_client import ChatHistory, ChatMessage
    from kavalai.eval.fixtures import fixture_key

    entries = {}
    for text in chat_texts:
        history = ChatHistory(messages=[ChatMessage(role="system", content=text)])
        key = fixture_key("openai/fake", history)
        entries[key] = {"model": "openai/fake", "prompt": text, "response": response}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@pytest.fixture
def suite_dir(tmp_path):
    """A complete, self-contained suite that needs no provider."""
    (tmp_path / "wf.yaml").write_text(yaml.safe_dump(WORKFLOW))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    write_fixtures(
        eval_dir / "fixtures" / "llm.json",
        ["hi"],
        json.dumps({"agent_response": "hi there"}),
    )
    (eval_dir / "cases.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "d",
                "cases": [
                    {
                        "name": "ok",
                        "inputs": {"user_message": "hi"},
                        "evaluators": [{"type": "contains", "text": "hi there"}],
                    },
                ],
            }
        )
    )
    (eval_dir / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "s",
                "dataset": "cases.yaml",
                "target": {"kind": "engine", "workflow": "../wf.yaml"},
                "evaluators": ["no_error"],
                "gate": {"min_pass_rate": 1.0},
            }
        )
    )
    return eval_dir


def test_the_suite_path_alone_means_run(suite_dir, capsys):
    """`kavalai-eval path/to/suite.yaml` is the command people actually type."""
    assert main([str(suite_dir / "suite.yaml"), "--tag", "t", "--fixtures"]) == EXIT_OK
    assert (suite_dir / "results" / "t.json").exists()
    assert (suite_dir / "results" / "t.junit.xml").exists()


def test_a_failing_gate_exits_one(suite_dir):
    path = suite_dir / "suite.yaml"
    data = yaml.safe_load(path.read_text())
    data["cases"] = None
    data["evaluators"] = ["no_error", {"type": "contains", "text": "never appears"}]
    path.write_text(yaml.safe_dump({k: v for k, v in data.items() if v is not None}))
    assert main([str(path), "--tag", "t", "--fixtures"]) == EXIT_GATE_FAILED


def test_a_broken_suite_exits_two_rather_than_pretending_to_pass(tmp_path):
    """CI has to be able to tell "the harness broke" from "the workflow is wrong"."""
    broken = tmp_path / "suite.yaml"
    broken.write_text(yaml.safe_dump({"name": "s", "dataset": "missing.yaml"}))
    assert main([str(broken)]) == EXIT_ERROR


def test_the_comment_file_is_written_when_asked(suite_dir, tmp_path):
    comment = tmp_path / "comment.md"
    main(
        [
            str(suite_dir / "suite.yaml"),
            "--tag",
            "t",
            "--fixtures",
            "--comment",
            str(comment),
        ]
    )
    assert "passed the gate" in comment.read_text()


def test_diff_and_accept_move_the_baseline(tmp_path, capsys):
    result = ExperimentResult(
        suite="s",
        tag="t",
        totals=Totals(cases=1, passed=1, pass_rate=1.0),
        verdicts=[CaseVerdict(case="a", status="passed", passes=1, total=1)],
    )
    result_path = tmp_path / "r.json"
    result.to_json(result_path)
    baseline = tmp_path / "baseline.json"
    result.to_json(baseline)

    assert main(["diff", str(baseline), str(result_path)]) == EXIT_OK
    assert "no change" in capsys.readouterr().out

    target = tmp_path / "new-baseline.json"
    assert main(["accept", str(result_path), "-o", str(target)]) == EXIT_OK
    assert json.loads(target.read_text())["suite"] == "s"
    # Accepting a baseline is accepting new behaviour, and says so.
    assert "Commit it" in capsys.readouterr().out


def test_accept_writes_to_the_suites_own_baseline(suite_dir, tmp_path):
    result = ExperimentResult(suite="s", tag="t")
    result_path = tmp_path / "r.json"
    result.to_json(result_path)
    main(["accept", str(result_path), "--suite", str(suite_dir / "suite.yaml")])
    assert (suite_dir / "baseline.json").exists()


def test_evaluators_lists_what_is_available(capsys):
    assert main(["evaluators"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "no_error" in out and "tool_not_called" in out


def test_env_expansion_happens_in_the_cli_not_the_library(monkeypatch):
    monkeypatch.setenv("TEST_ONLY_URL", "http://agent:8000")
    assert expand_env("${TEST_ONLY_URL}/run") == "http://agent:8000/run"
    assert expand_env(None) is None
    with pytest.raises(SystemExit, match="TEST_ONLY_MISSING"):
        expand_env("${TEST_ONLY_MISSING}")


def test_persist_sessions_without_a_database_says_so(suite_dir, monkeypatch):
    monkeypatch.delenv("KAVALAI_DB_URI", raising=False)
    with pytest.raises(SystemExit, match="needs a database"):
        main([str(suite_dir / "suite.yaml"), "--persist-sessions", "--fixtures"])


def test_no_command_prints_help(capsys):
    assert main([]) == EXIT_ERROR
    assert "usage" in capsys.readouterr().out


def test_the_persona_entry_point_targets_the_persona_command():
    parser = build_parser()
    args = parser.parse_args(["persona", "p.yaml", "--workflow", "w.yaml"])
    assert args.persona == "p.yaml" and args.workflow == "w.yaml"


def test_a_persona_without_a_target_says_which_flag_is_missing(tmp_path):
    persona = tmp_path / "p.yaml"
    persona.write_text(yaml.safe_dump({"goal": "g"}))
    with pytest.raises(SystemExit, match="--workflow"):
        main(["persona", str(persona), "-v"])
