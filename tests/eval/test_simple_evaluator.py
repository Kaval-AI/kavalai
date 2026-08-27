"""The literal evaluator: matchers, and one round trip to a stand-in agent."""

import json

import pytest

from kavalai.eval import SimpleEvaluator, check_output
from kavalai.eval.simple_evaluator import check_field, is_matcher_spec
from tests.eval.conftest import agent_transport


def test_a_bare_value_is_an_equality_check():
    assert check_field("f", "yes", "yes") == []
    assert check_field("f", "yes", "no") == ["f: expected 'no', got 'yes'"]


def test_a_dict_of_matcher_names_is_read_as_matchers():
    assert is_matcher_spec({"contains": "x"})
    assert not is_matcher_spec({"answer": "x"})
    assert not is_matcher_spec({})


def test_a_dict_of_other_keys_is_compared_whole():
    # An agent whose output field is itself an object is compared for
    # equality, not mistaken for a matcher mapping.
    assert check_field("f", {"answer": 1}, {"answer": 1}) == []
    assert len(check_field("f", {"answer": 1}, {"answer": 2})) == 1


def test_contains_is_substring_for_text_and_case_insensitive():
    assert check_field("f", "The Marsh Marigold", {"contains": "marsh marigold"}) == []
    assert check_field("f", "the daisy", {"contains": ["marigold"]}) == [
        "f: 'the daisy' is missing ['marigold']"
    ]


def test_contains_is_membership_for_a_list():
    assert check_field("f", ["fact-00"], {"contains": "fact-00"}) == []
    assert len(check_field("f", ["fact-01"], {"contains": "fact-00"})) == 1


def test_contains_fails_on_a_value_that_holds_nothing():
    assert len(check_field("f", 104, {"contains": "104"})) == 1


def test_not_contains():
    assert check_field("f", "all good", {"not_contains": "error"}) == []
    assert len(check_field("f", "an error", {"not_contains": ["error"]})) == 1


def test_regex():
    assert check_field("f", "1,847 books", {"regex": "1,?847"}) == []
    assert len(check_field("f", "1 847 books", {"regex": "1,?847"})) == 1


def test_one_of():
    assert check_field("f", "yes", {"one_of": ["yes", "no"]}) == []
    assert len(check_field("f", "maybe", {"one_of": ["yes", "no"]})) == 1


def test_several_matchers_on_one_field_all_have_to_hold():
    failures = check_field("f", "104 residents", {"contains": "104", "regex": "^5"})
    assert len(failures) == 1


def test_a_field_the_agent_did_not_answer_is_a_failure():
    assert check_output({"agent_response": "hi"}, {"used_ids": ["a"]}) == [
        "used_ids: the agent's output has no such field"
    ]


def test_fields_the_expectation_ignores_are_not_checked():
    assert check_output({"a": 1, "b": 2}, {"a": 1}) == []
    assert check_output({"a": 1}, None) == []


async def test_evaluate_passes_and_reports_the_agent_output(transport):
    evaluator = SimpleEvaluator("http://testserver", transport=transport)

    result = await evaluator.evaluate(
        {"user_message": "Hi"},
        {"agent_response": {"contains": "hello"}},
        name="greeting",
    )

    assert result.passed and bool(result) and result.reason == ""
    assert result.name == "greeting"
    assert result.output == {"agent_response": "Hello world", "used_ids": None}
    assert result.inputs == {"user_message": "Hi"}


async def test_evaluate_fails_with_the_reason(transport):
    evaluator = SimpleEvaluator("http://testserver", transport=transport)

    result = await evaluator.evaluate(
        {"user_message": "Hi"}, {"agent_response": {"contains": "goodbye"}}
    )

    assert not result.passed
    assert "goodbye" in result.reason


async def test_each_case_runs_in_its_own_session(transport):
    """The evaluator must not carry one case's conversation into the next."""
    requests = []
    evaluator = SimpleEvaluator(
        "http://testserver",
        transport=agent_transport(
            reply={"agent_response": "Hello world"}, requests=requests
        ),
    )
    evaluator.client.session_id = "left-over"

    await evaluator.evaluate({"user_message": "Hi"})

    runs = [r for r in requests if r.url.path == "/run_agent"]
    assert json.loads(runs[0].content)["session_id"] is None


async def test_the_case_name_is_sent_as_the_external_id():
    requests = []
    evaluator = SimpleEvaluator(
        "http://testserver", transport=agent_transport(requests=requests)
    )

    await evaluator.evaluate({"user_message": "Hi"}, name="greeting")

    runs = [r for r in requests if r.url.path == "/run_agent"]
    assert json.loads(runs[0].content)["external_id"] == "eval:greeting"


async def test_the_tag_names_the_run_inside_the_external_id():
    """What lets a data scientist tell one run's sessions from another's."""
    requests = []
    evaluator = SimpleEvaluator(
        "http://testserver",
        tag="variant-b",
        transport=agent_transport(requests=requests),
    )

    await evaluator.evaluate({"user_message": "Hi"}, name="greeting")

    runs = [r for r in requests if r.url.path == "/run_agent"]
    assert json.loads(runs[0].content)["external_id"] == "eval:variant-b:greeting"


async def test_input_is_validated_against_the_agents_own_input_type(transport):
    """A mistyped input field fails the case, and says which one."""
    evaluator = SimpleEvaluator("http://testserver", transport=transport)

    result = await evaluator.evaluate({"user_mesage": "Hi"})

    assert not result.passed
    assert "the agent run failed" in result.reason


async def test_a_server_error_fails_the_case_instead_of_raising():
    evaluator = SimpleEvaluator(
        "http://testserver", transport=agent_transport(status_code=500)
    )

    result = await evaluator.evaluate({"user_message": "Hi"}, name="boom")

    assert not result.passed
    assert "the agent run failed" in result.reason
    assert result.output is None


async def test_basic_auth_is_sent_when_configured():
    requests = []
    evaluator = SimpleEvaluator(
        "http://testserver",
        username="user",
        password="secret",
        transport=agent_transport(requests=requests),
    )

    await evaluator.evaluate({"user_message": "Hi"})

    assert "authorization" in requests[-1].headers


async def test_run_agent_returns_the_typed_output(transport):
    """The base call is usable on its own: send an input, get the output."""
    evaluator = SimpleEvaluator("http://testserver", transport=transport)

    output = await evaluator.run_agent({"user_message": "Hi"})

    assert output.agent_response == "Hello world"


async def test_the_base_class_grades_nothing_by_itself(transport):
    from kavalai.eval import AgentEvaluator

    with pytest.raises(NotImplementedError):
        await AgentEvaluator("http://testserver", transport=transport).evaluate({})
