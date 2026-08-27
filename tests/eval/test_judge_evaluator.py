"""The judged evaluator: the prompt it builds, and the verdicts it reports."""

import pytest

from kavalai.eval import JudgeEvaluator, JudgeVerdict
from kavalai.eval.judge_evaluator import DEFAULT_JUDGE_MODEL
from tests.eval.conftest import FakeJudge, agent_transport


def make_evaluator(verdict=None, error=None, transport=None, **kwargs):
    judge = FakeJudge(verdict=verdict, error=error)
    evaluator = JudgeEvaluator(
        "http://testserver",
        transport=transport or agent_transport(),
        llm_client=judge,
        **kwargs,
    )
    return evaluator, judge


async def test_a_passing_verdict_carries_no_reason():
    evaluator, _ = make_evaluator(JudgeVerdict(passed=True))

    result = await evaluator.evaluate(
        {"user_message": "Hi"}, "The answer greets the user.", name="greeting"
    )

    assert result.passed and result.reason == ""
    assert result.output == {"agent_response": "Hello world", "used_ids": None}


async def test_a_failing_verdict_carries_the_judges_reason():
    evaluator, _ = make_evaluator(
        JudgeVerdict(passed=False, reason="It never answers the question.")
    )

    result = await evaluator.evaluate({"user_message": "Hi"}, "It answers.")

    assert not result.passed
    assert result.reason == "It never answers the question."


async def test_a_failing_verdict_without_a_reason_still_says_something():
    evaluator, _ = make_evaluator(JudgeVerdict(passed=False))

    result = await evaluator.evaluate({"user_message": "Hi"}, "It answers.")

    assert not result.passed and result.reason


async def test_the_prompt_carries_the_input_the_output_and_the_criterion():
    evaluator, judge = make_evaluator(JudgeVerdict(passed=True))

    await evaluator.evaluate({"user_message": "Hi"}, "The answer greets back.")

    (prompt,) = judge.prompts
    assert "user_message" in prompt and "Hi" in prompt
    assert "Hello world" in prompt
    assert "The answer greets back." in prompt


async def test_a_judged_case_without_a_criterion_is_refused():
    """Judging against nothing would pass on any answer at all."""
    evaluator, _ = make_evaluator(JudgeVerdict(passed=True))

    with pytest.raises(ValueError, match="no criterion"):
        await evaluator.evaluate({"user_message": "Hi"}, None, name="empty")


async def test_a_failing_agent_is_reported_before_the_judge_is_called():
    evaluator, judge = make_evaluator(
        JudgeVerdict(passed=True), transport=agent_transport(status_code=500)
    )

    result = await evaluator.evaluate({"user_message": "Hi"}, "It answers.")

    assert not result.passed
    assert "the agent run failed" in result.reason
    assert judge.prompts == []


async def test_a_failing_judge_fails_the_case_and_keeps_the_output():
    evaluator, _ = make_evaluator(error=RuntimeError("no api key"))

    result = await evaluator.evaluate({"user_message": "Hi"}, "It answers.")

    assert not result.passed
    assert "the judge failed: no api key" in result.reason
    assert result.output == {"agent_response": "Hello world", "used_ids": None}


def test_the_judging_model_is_only_built_when_it_is_needed(monkeypatch):
    """A suite of literal cases must not need a provider key."""
    built = []

    def fake_make_client(model):
        built.append(model)
        return FakeJudge()

    monkeypatch.setattr("kavalai.eval.judge_evaluator.make_client", fake_make_client)
    evaluator = JudgeEvaluator("http://testserver")
    assert built == []

    assert evaluator.llm_client is evaluator.llm_client
    assert built == [DEFAULT_JUDGE_MODEL]


def test_the_model_can_be_chosen(monkeypatch):
    built = []
    monkeypatch.setattr(
        "kavalai.eval.judge_evaluator.make_client",
        lambda model: built.append(model) or FakeJudge(),
    )

    assert JudgeEvaluator("http://testserver", model="anthropic/claude").llm_client

    assert built == ["anthropic/claude"]


async def test_the_prompt_can_be_replaced():
    evaluator, judge = make_evaluator(
        JudgeVerdict(passed=True), prompt="{criterion} // {inputs} // {output}"
    )

    await evaluator.evaluate({"user_message": "Hi"}, "Greets back.")

    assert judge.prompts[0].startswith("Greets back. //")
