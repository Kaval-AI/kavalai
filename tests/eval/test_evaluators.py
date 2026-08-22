"""Every evaluator, against a known-good and a known-bad input.

The evaluation harness gates deploys and is itself untested code unless
something like this exists. An evaluator with an inverted condition that
returns ``passed=True`` for everything is *worse* than having no gate, because
it manufactures confidence — so the meta-test at the bottom insists that every
registered evaluator has been shown to fail at least once.
"""

import pytest

from kavalai.eval import Case, RunRecord, Score, Trajectory, build_evaluator
from kavalai.eval.evaluators.base import (
    EvaluationError,
    Evaluator,
    REGISTRY,
    _value_at,
    evaluator,
    known_evaluators,
)
from kavalai.workflow.tasklog import TaskRecord

#: Every evaluator this module has proved can fail. The meta-test compares it
#: with the registry, so adding an evaluator without a failing case breaks CI.
PROVEN_TO_FAIL: set[str] = set()


async def verdict(spec, case, record) -> Score:
    """Score one record and remember whether the evaluator can fail."""
    result = await build_evaluator(spec).score(case, record)
    score = result[0] if isinstance(result, list) else result
    if score.passed is False:
        PROVEN_TO_FAIL.add(score.name)
    return score


def rec(**kwargs) -> RunRecord:
    kwargs.setdefault("output", {"agent_response": "The pond is 1.2 metres deep."})
    return RunRecord(**kwargs)


def traj(*records: TaskRecord) -> Trajectory:
    return Trajectory(records=list(records))


CASE = Case(name="c", inputs={"user_message": "how deep?"})


# ------------------------------------------------------------- deterministic
async def test_no_error():
    assert (await verdict("no_error", CASE, rec())).passed is True
    score = await verdict("no_error", CASE, rec(status="failed", error="boom"))
    assert score.passed is False and score.reason == "boom"


async def test_equals_expected():
    case = Case(name="c", inputs={}, expected={"agent_response": "yes"})
    assert (
        await verdict("equals_expected", case, rec(output={"agent_response": "yes"}))
    ).passed
    score = await verdict("equals_expected", case, rec(output={"agent_response": "no"}))
    assert score.passed is False and "expected" in score.reason


async def test_field_equals():
    spec = {"type": "field_equals", "path": "order.items[0].quantity", "value": 4}
    good = rec(output={"order": {"items": [{"quantity": 4}]}})
    bad = rec(output={"order": {"items": [{"quantity": 3}]}})
    assert (await verdict(spec, CASE, good)).passed is True
    assert (await verdict(spec, CASE, bad)).passed is False


async def test_field_equals_reports_a_missing_path_rather_than_raising():
    spec = {"type": "field_equals", "path": "order.total", "value": 1}
    score = await verdict(spec, CASE, rec(output={}))
    assert score.passed is False and "None" in score.reason


async def test_json_subset_ignores_extra_fields():
    case = Case(name="c", inputs={}, expected={"a": 1})
    assert (
        await verdict("json_subset", case, rec(output={"a": 1, "b": 2}))
    ).passed is True
    score = await verdict("json_subset", case, rec(output={"a": 2}))
    assert score.passed is False and "expected 1" in score.reason


@pytest.mark.parametrize(
    "expected,actual,ok",
    [
        ({"a": {"b": 1}}, {"a": {"b": 1}}, True),
        ({"a": {"b": 1}}, {"a": {"b": 2}}, False),
        ({"a": {"b": 1}}, {"a": 5}, False),
        ({"a": [1, 2]}, {"a": [1, 2, 3]}, True),
        ({"a": [1, 2]}, {"a": [1]}, False),
        ({"a": [1]}, {"a": "no"}, False),
        ({"a": 1}, {}, False),
    ],
)
async def test_json_subset_nesting(expected, actual, ok):
    case = Case(name="c", inputs={}, expected=expected)
    assert (await verdict("json_subset", case, rec(output=actual))).passed is ok


async def test_json_subset_without_an_expectation_is_measured_not_asserted():
    score = await verdict("json_subset", CASE, rec())
    assert score.passed is None


async def test_contains_is_case_insensitive_by_default():
    assert (
        await verdict({"type": "contains", "text": "1.2"}, CASE, rec())
    ).passed is True
    assert (
        await verdict({"type": "contains", "text": "POND"}, CASE, rec())
    ).passed is True
    strict = {"type": "contains", "text": "POND", "case_sensitive": True}
    assert (await verdict(strict, CASE, rec())).passed is False


async def test_contains_falls_back_to_the_case_expectation():
    case = Case(name="c", inputs={}, expected={"contains": "1.2"})
    assert (await verdict("contains", case, rec())).passed is True
    assert (await verdict("contains", Case(name="c", inputs={}), rec())).passed is None


async def test_not_contains():
    spec = {"type": "not_contains", "text": "4 metres"}
    assert (await verdict(spec, CASE, rec())).passed is True
    bad = rec(output={"agent_response": "It is 4 metres deep."})
    assert (await verdict(spec, CASE, bad)).passed is False


async def test_regex():
    assert (await verdict({"type": "regex", "pattern": r"\d\.\d"}, CASE, rec())).passed
    assert (
        await verdict({"type": "regex", "pattern": "^nope"}, CASE, rec())
    ).passed is False
    insensitive = {"type": "regex", "pattern": "POND", "flags": "i"}
    assert (await verdict(insensitive, CASE, rec())).passed is True


async def test_no_digits():
    refusal = rec(output={"agent_response": "The facts do not say."})
    assert (await verdict("no_digits", CASE, refusal)).passed is True
    assert (await verdict("no_digits", CASE, rec())).passed is False


async def test_latency_under_carries_the_measurement():
    spec = {"type": "latency_under", "seconds": 2}
    assert (await verdict(spec, CASE, rec(duration_seconds=1.0))).passed is True
    score = await verdict(spec, CASE, rec(duration_seconds=6.4))
    assert score.passed is False and score.reason == "6.4s > 2.0s"
    assert score.value == 6.4


async def test_tokens_under_carries_the_measurement():
    from kavalai.eval.targets import ModelCallRecord

    calls = [ModelCallRecord(total_tokens=4812)]
    spec = {"type": "tokens_under", "n": 3000}
    score = await verdict(spec, CASE, rec(model_calls=calls))
    assert score.passed is False and score.reason == "4,812 tokens > 3,000"
    assert (await verdict(spec, CASE, rec())).passed is True


def test_tokens_under_accepts_the_per_case_spelling():
    assert build_evaluator({"type": "tokens_under", "per_case": 10}).limit == 10
    with pytest.raises(ValueError):
        build_evaluator({"type": "tokens_under"})


async def test_output_not_empty():
    assert (await verdict("output_not_empty", CASE, rec())).passed is True
    assert (await verdict("output_not_empty", CASE, rec(output=None))).passed is False


async def test_always_fails_is_a_canary():
    """One case per suite behind this; if it ever passes, the harness is broken."""
    assert (await verdict("always_fails", CASE, rec())).passed is False


# ---------------------------------------------------------------- trajectory
def nodes():
    return traj(
        TaskRecord(name="begin", node_type="start", seq=0),
        TaskRecord(
            name="retrieve",
            node_type="rag_query",
            seq=1,
            output=[{"source_id": "fact-09"}, {"source_id": "fact-01"}],
        ),
        TaskRecord(
            name="route",
            node_type="switch",
            seq=2,
            inputs={"expr": "parsed.intent", "value": "order"},
            output={"taken": "validate", "matched": True},
        ),
        TaskRecord(
            name="validate",
            node_type="function",
            seq=3,
            tool_uri="python://validate_order",
            output={"missing_fields": []},
        ),
        TaskRecord(name="agent", node_type="agent", seq=4),
        TaskRecord(
            name="crawl",
            node_type="tool_call",
            seq=5,
            parent_task_name="agent",
            tool_uri="python://crawl",
            inputs={"args": {"url": "https://x"}, "step": 0},
        ),
        TaskRecord(name="finish", node_type="end", seq=6),
    )


async def test_trajectory_evaluators_error_rather_than_pass_blind():
    """The worst failure mode is a gate that reports green because it saw nothing."""
    with pytest.raises(EvaluationError, match="does not produce one"):
        await build_evaluator({"type": "node_visited", "node": "x"}).score(CASE, rec())


async def test_node_visited_and_not_visited():
    record = rec(trajectory=nodes())
    assert (
        await verdict({"type": "node_visited", "node": "validate"}, CASE, record)
    ).passed
    score = await verdict({"type": "node_visited", "node": "store"}, CASE, record)
    assert score.passed is False and "never ran" in score.reason
    assert (
        await verdict({"type": "node_not_visited", "node": "store"}, CASE, record)
    ).passed
    assert (
        await verdict({"type": "node_not_visited", "node": "validate"}, CASE, record)
    ).passed is False


async def test_branch_taken_reports_the_value_it_routed_on():
    record = rec(trajectory=nodes())
    spec = {"type": "branch_taken", "node": "route", "target": "validate"}
    assert (await verdict(spec, CASE, record)).passed is True

    wrong = {"type": "branch_taken", "node": "route", "target": "reply_other"}
    score = await verdict(wrong, CASE, record)
    assert score.passed is False and "'order'" in score.reason

    absent = {"type": "branch_taken", "node": "nope", "target": "x"}
    assert (await verdict(absent, CASE, record)).passed is False


async def test_branch_taken_reads_the_expectation_from_the_case():
    record = rec(trajectory=nodes())
    case = Case(name="c", inputs={}, expected={"branch": "validate"})
    assert (await verdict("branch_taken", case, record)).passed is True
    assert (await verdict("branch_taken", CASE, record)).passed is None


async def test_switch_matched_catches_a_label_outside_the_enum():
    good = rec(trajectory=nodes())
    assert (await verdict("switch_matched", CASE, good)).passed is True

    fell_through = traj(
        TaskRecord(
            name="route",
            node_type="switch",
            seq=0,
            inputs={"expr": "intent", "value": "Refund "},
            output={"taken": "default_node", "matched": False},
        )
    )
    score = await verdict("switch_matched", CASE, rec(trajectory=fell_through))
    assert score.passed is False and "Refund " in score.reason


async def test_tool_called_finds_function_nodes_and_agent_choices_alike():
    record = rec(trajectory=nodes())
    for uri in ("python://validate_order", "python://crawl"):
        assert (await verdict({"type": "tool_called", "uri": uri}, CASE, record)).passed
    missing = {"type": "tool_called", "uri": "python://store_order"}
    assert (await verdict(missing, CASE, record)).passed is False


async def test_tool_called_can_require_an_exact_count():
    record = rec(trajectory=nodes())
    spec = {"type": "tool_called", "uri": "python://crawl", "times": 2}
    score = await verdict(spec, CASE, record)
    assert score.passed is False and "called 1x" in score.reason


async def test_tool_not_called_is_the_safety_assertion():
    record = rec(trajectory=nodes())
    assert (
        await verdict(
            {"type": "tool_not_called", "uri": "python://refund"}, CASE, record
        )
    ).passed is True
    assert (
        await verdict(
            {"type": "tool_not_called", "uri": "python://crawl"}, CASE, record
        )
    ).passed is False


async def test_tool_call_order():
    record = rec(trajectory=nodes())
    good = {
        "type": "tool_call_order",
        "uris": ["python://validate_order", "python://crawl"],
    }
    assert (await verdict(good, CASE, record)).passed is True
    bad = {
        "type": "tool_call_order",
        "uris": ["python://crawl", "python://validate_order"],
    }
    assert (await verdict(bad, CASE, record)).passed is False


async def test_tool_args_match():
    record = rec(trajectory=nodes())
    good = {
        "type": "tool_args_match",
        "uri": "python://crawl",
        "path": "url",
        "value": "https://x",
    }
    assert (await verdict(good, CASE, record)).passed is True
    bad = {**good, "value": "https://y"}
    assert (await verdict(bad, CASE, record)).passed is False
    never = {**good, "uri": "python://nope"}
    assert (await verdict(never, CASE, record)).passed is False


async def test_max_agent_steps():
    record = rec(trajectory=nodes())
    assert (
        await verdict({"type": "max_agent_steps", "n": 3}, CASE, record)
    ).passed is True
    assert (
        await verdict({"type": "max_agent_steps", "n": 0}, CASE, record)
    ).passed is False


async def test_retrieval_hit_at_k_defaults_to_what_the_node_returned():
    """A `k` stricter than the node's top_k reports a miss on a right answer."""
    record = rec(trajectory=nodes())
    case = Case(name="c", inputs={}, expected={"source_ids": ["fact-01"]})
    assert (await verdict({"type": "retrieval_hit_at_k"}, case, record)).passed is True

    score = await verdict({"type": "retrieval_hit_at_k", "k": 1}, case, record)
    assert score.passed is False and "top 1" in score.reason

    missing = Case(name="c", inputs={}, expected={"source_ids": ["fact-99"]})
    assert (
        await verdict({"type": "retrieval_hit_at_k"}, missing, record)
    ).passed is False


async def test_retrieval_hit_at_k_without_an_expectation_or_a_node():
    record = rec(trajectory=nodes())
    assert (await verdict({"type": "retrieval_hit_at_k"}, CASE, record)).passed is None

    case = Case(name="c", inputs={}, expected={"source_id": "fact-01"})
    empty = rec(trajectory=traj(TaskRecord(name="other", node_type="llm", seq=0)))
    score = await verdict({"type": "retrieval_hit_at_k"}, case, empty)
    assert score.passed is False and "never ran" in score.reason


async def test_groundedness_catches_a_fabricated_citation():
    good = rec(output={"used_ids": ["fact-09"]}, trajectory=nodes())
    assert (await verdict("groundedness", CASE, good)).passed is True

    bad = rec(output={"used_ids": ["fact-77"]}, trajectory=nodes())
    score = await verdict("groundedness", CASE, bad)
    assert score.passed is False and "fact-77" in score.reason


# -------------------------------------------------------------- conversation
def chat(*turns) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t}
        for i, t in enumerate(turns)
    ]


async def test_goal_achieved_uses_the_personas_own_verdict():
    met = rec(chat=chat("hi", "there"), meta={"goal_achieved": True})
    assert (await verdict("goal_achieved", CASE, met)).passed is True

    unmet = rec(
        chat=chat("hi", "there"), meta={"goal_achieved": False, "user_turns": 4}
    )
    score = await verdict("goal_achieved", CASE, unmet)
    assert score.passed is False and "4 turns" in score.reason


async def test_conversation_evaluators_refuse_a_single_shot_run():
    score = await verdict("goal_achieved", CASE, rec())
    assert score.passed is None and "persona run" in score.reason


async def test_turns_to_resolution():
    record = rec(chat=chat("a", "b", "c", "d"))
    assert (
        await verdict({"type": "turns_to_resolution", "max": 2}, CASE, record)
    ).passed
    score = await verdict({"type": "turns_to_resolution", "max": 1}, CASE, record)
    assert score.passed is False and "2 turns > 1" in score.reason


async def test_no_repeated_question():
    fine = rec(chat=chat("hi", "How many would you like?", "two", "Which day?"))
    assert (await verdict("no_repeated_question", CASE, fine)).passed is True

    repeated = rec(
        chat=chat("hi", "How many would you like?", "dunno", "How many would you like?")
    )
    score = await verdict("no_repeated_question", CASE, repeated)
    assert score.passed is False and "asked twice" in score.reason


# --------------------------------------------------------------- the registry
def test_registering_a_duplicate_name_is_refused():
    @evaluator("test_only_duplicate")
    class First(Evaluator):
        async def score(self, case, record):
            return Score.boolean(self.name, True)

    with pytest.raises(ValueError, match="already registered"):

        @evaluator("test_only_duplicate")
        class Second(Evaluator):
            async def score(self, case, record):
                return Score.boolean(self.name, True)

    @evaluator("test_only_duplicate", replace=True)
    class Third(Evaluator):
        async def score(self, case, record):
            return Score.boolean(self.name, True)

    assert REGISTRY["test_only_duplicate"] is Third
    del REGISTRY["test_only_duplicate"]


def test_an_unknown_evaluator_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="Unknown evaluator 'nope'"):
        build_evaluator("nope")


def test_bad_options_say_which_evaluator_rejected_them():
    with pytest.raises(ValueError, match="rejected its options"):
        build_evaluator({"type": "contains", "nonsense": 1})


@pytest.mark.parametrize(
    "data,path,expected",
    [
        ({"a": {"b": 1}}, "a.b", 1),
        ({"a": [{"b": 2}]}, "a[0].b", 2),
        ({"a": [1, 2]}, "a[-1]", 2),
        ({"a": [1]}, "a[5]", None),
        ({"a": 1}, "a.b", None),
        ({}, "", {}),
    ],
)
def test_value_at(data, path, expected):
    assert _value_at(data, path) == expected


def test_every_evaluator_has_been_shown_to_fail():
    """The meta-test: an evaluator nobody proved can fail is a silent gate.

    Judged evaluators are excluded — they need a provider, and their failure
    path is the shared :class:`LLMJudge` one, which is covered separately.
    """
    needs_a_model = {
        "llm_judge",
        "refuses",
        "semantic_similarity",
        "stayed_on_topic",
        "resisted_injection",
        "persona_satisfaction",
    }
    untested = set(known_evaluators()) - PROVEN_TO_FAIL - needs_a_model
    assert not untested, (
        f"These evaluators were never shown to fail: {sorted(untested)}. "
        "An evaluator that cannot fail manufactures confidence."
    )
