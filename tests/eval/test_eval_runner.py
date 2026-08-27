"""The YAML suite: what it accepts, how it runs, and what the CLI returns."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kavalai.eval import (
    EvalCase,
    EvalResult,
    EvalSuite,
    eval_runner,
    load_suite,
    run_suite,
)
from kavalai.eval.eval_runner import (
    EXIT_ERROR,
    EXIT_FAILED,
    EXIT_PASSED,
    format_result,
    format_summary,
    main,
    resolve_base_url,
)
from kavalai.eval.judge_evaluator import JudgeVerdict
from tests.eval.conftest import FakeJudge, agent_transport

REPO = Path(__file__).resolve().parent.parent.parent

SUITE = {
    "name": "demo",
    "base_url": "http://testserver",
    "cases": [
        {
            "name": "greets",
            "input": {"user_message": "Hi"},
            "expected": {"agent_response": {"contains": "hello"}},
        },
        {
            "name": "polite",
            "type": "judge",
            "input": {"user_message": "Hi"},
            "expected": "The answer greets the user back.",
        },
    ],
}


@pytest.fixture
def suite_file(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(SUITE), encoding="utf-8")
    return path


@pytest.fixture
def fake_judge(monkeypatch):
    """Every JudgeEvaluator in the run grades with this stand-in model."""
    judge = FakeJudge(verdict=JudgeVerdict(passed=True))
    monkeypatch.setattr("kavalai.eval.judge_evaluator.make_client", lambda model: judge)
    return judge


def test_a_suite_file_is_read_into_cases(suite_file):
    suite = load_suite(suite_file)

    assert suite.name == "demo"
    assert [c.name for c in suite.cases] == ["greets", "polite"]
    assert suite.cases[0].type == "simple"
    assert suite.cases[1].type == "judge"


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """A silently ignored key is a case that never ran."""
    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump({"name": "d", "cases": [], "evaluators": ["no_error"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="evaluators"):
        load_suite(path)


def test_a_judged_case_needs_a_criterion_in_words():
    with pytest.raises(ValidationError, match="plain-language criterion"):
        EvalCase(name="c", type="judge", input={}, expected={"a": 1})

    with pytest.raises(ValidationError, match="plain-language criterion"):
        EvalCase(name="c", type="judge", input={})


def test_a_simple_case_cannot_expect_a_sentence():
    with pytest.raises(ValidationError, match="type: judge"):
        EvalCase(name="c", input={}, expected="It answers politely.")


def test_a_simple_case_may_expect_nothing_at_all():
    """That still asserts the agent answered without erroring."""
    assert EvalCase(name="c", input={"user_message": "Hi"}).expected is None


async def test_run_suite_runs_every_case_in_order(suite_file, fake_judge):
    seen = []

    results = await run_suite(
        load_suite(suite_file),
        transport=agent_transport(),
        on_result=seen.append,
    )

    assert [r.name for r in results] == ["greets", "polite"]
    assert all(r.passed for r in results)
    assert [r.name for r in seen] == ["greets", "polite"]
    assert len(fake_judge.prompts) == 1


async def test_a_failing_case_does_not_stop_the_rest(fake_judge):
    suite = EvalSuite(
        name="demo",
        cases=[
            EvalCase(
                name="wrong",
                input={"user_message": "Hi"},
                expected={"agent_response": "Goodbye"},
            ),
            EvalCase(
                name="right",
                input={"user_message": "Hi"},
                expected={"agent_response": {"contains": "hello"}},
            ),
        ],
    )

    results = await run_suite(
        suite, base_url="http://testserver", transport=agent_transport()
    )

    assert [r.passed for r in results] == [False, True]


async def test_the_caller_can_override_where_the_agent_lives():
    """--host/--port beat the base_url written in the file."""
    requests = []
    suite = EvalSuite(
        name="demo",
        base_url="http://from-the-file",
        cases=[EvalCase(name="c", input={"user_message": "Hi"})],
    )

    await run_suite(
        suite,
        base_url="http://from-the-caller",
        transport=agent_transport(requests=requests),
    )

    assert {r.url.host for r in requests} == {"from-the-caller"}


async def test_the_file_says_where_the_agent_lives_when_the_caller_does_not():
    requests = []
    suite = EvalSuite(
        name="demo",
        base_url="http://from-the-file",
        cases=[EvalCase(name="c", input={"user_message": "Hi"})],
    )

    await run_suite(suite, transport=agent_transport(requests=requests))

    assert {r.url.host for r in requests} == {"from-the-file"}


async def test_the_judge_model_can_be_overridden(monkeypatch):
    models = []
    monkeypatch.setattr(
        "kavalai.eval.judge_evaluator.make_client",
        lambda model: (
            models.append(model) or FakeJudge(verdict=JudgeVerdict(passed=True))
        ),
    )
    suite = EvalSuite(
        name="demo",
        judge_model="openai/from-the-file",
        cases=[
            EvalCase(
                name="c",
                type="judge",
                input={"user_message": "Hi"},
                expected="It answers.",
            )
        ],
    )

    await run_suite(
        suite,
        base_url="http://testserver",
        judge_model="openai/from-the-caller",
        transport=agent_transport(),
    )

    assert models == ["openai/from-the-caller"]


def test_results_are_reported_one_line_each():
    passed = EvalResult(name="a", passed=True)
    failed = EvalResult(name="b", passed=False, reason="no greeting")

    assert format_result(passed) == "PASS  a"
    assert format_result(failed).startswith("FAIL  b")
    assert "no greeting" in format_result(failed)
    assert format_summary([passed, failed]) == "1/2 passed"


def test_host_and_port_become_a_base_url():
    assert resolve_base_url("agents.example.com", 8000) == (
        "http://agents.example.com:8000"
    )
    assert resolve_base_url(None, 25000) == "http://localhost:25000"
    assert resolve_base_url("example.com", None) == "http://example.com:10000"
    # Neither given: the suite's own base_url still applies.
    assert resolve_base_url(None, None) is None


@pytest.fixture
def cli_transport(monkeypatch):
    """Let ``main`` run for real, but against the stand-in agent."""
    real_run_suite = eval_runner.run_suite

    async def run_against_the_stand_in(suite, **kwargs):
        return await real_run_suite(suite, transport=agent_transport(), **kwargs)

    monkeypatch.setattr(eval_runner, "run_suite", run_against_the_stand_in)


def test_the_cli_returns_zero_when_every_case_passes(
    suite_file, fake_judge, cli_transport, capsys
):
    code = main([str(suite_file), "--host", "testserver"])

    out = capsys.readouterr().out
    assert code == EXIT_PASSED
    assert "2/2 passed" in out
    assert "PASS  greets" in out
    assert "demo: 2 cases against http://testserver:10000" in out


def test_the_cli_returns_one_when_a_case_fails(tmp_path, cli_transport, capsys):
    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "base_url": "http://testserver",
                "cases": [
                    {
                        "name": "wrong",
                        "input": {"user_message": "Hi"},
                        "expected": {"agent_response": "Goodbye"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main([str(path)])

    out = capsys.readouterr().out
    assert code == EXIT_FAILED
    assert "0/1 passed" in out
    assert "FAIL  wrong" in out


def test_the_cli_returns_two_when_the_suite_itself_is_broken(tmp_path, capsys):
    """CI has to tell a broken suite from a wrong agent."""
    path = tmp_path / "cases.yaml"
    path.write_text("name: demo\ncases: not-a-list\n", encoding="utf-8")

    assert main([str(path)]) == EXIT_ERROR
    assert "Cannot run" in capsys.readouterr().err


def test_the_cli_returns_two_when_the_run_breaks(suite_file, monkeypatch, capsys):
    async def explode(*args, **kwargs):
        raise RuntimeError("the transport went away")

    monkeypatch.setattr("kavalai.eval.eval_runner.run_suite", explode)

    assert main([str(suite_file)]) == EXIT_ERROR
    assert "the transport went away" in capsys.readouterr().err


def test_basic_auth_is_split_off_the_command_line(suite_file, monkeypatch):
    seen = {}

    async def capture(suite, **kwargs):
        seen.update(kwargs)
        return [EvalResult(name="c", passed=True)]

    monkeypatch.setattr("kavalai.eval.eval_runner.run_suite", capture)

    main([str(suite_file), "--auth", "user:secret", "--port", "25000"])

    assert seen["username"] == "user"
    assert seen["password"] == "secret"
    assert seen["base_url"] == "http://localhost:25000"


def test_no_auth_flag_means_no_credentials(suite_file, monkeypatch):
    seen = {}

    async def capture(suite, **kwargs):
        seen.update(kwargs)
        return [EvalResult(name="c", passed=True)]

    monkeypatch.setattr("kavalai.eval.eval_runner.run_suite", capture)

    main([str(suite_file)])

    assert seen["username"] is None and seen["password"] is None


def test_the_shipped_green_village_cases_are_a_valid_suite():
    suite = load_suite(REPO / "examples" / "green_village" / "eval_cases.yaml")

    assert suite.cases
    assert {c.type for c in suite.cases} == {"simple", "judge"}
