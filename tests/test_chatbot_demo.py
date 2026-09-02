"""Regression test for the interactive chatbot demo on the docs landing page.

The demo's single source is ``docs/_includes/chatbot-demo.html`` (pulled into
``docs/index.rst`` and driven by the chat widget). This extracts the
``CHATBOT_CODE`` it ships and runs the workflow through the real engine with a
fake LLM, so the demo — and its structured ``Reply`` with ``choices`` — cannot
silently break.
"""

import re
from pathlib import Path

import pytest

from kavalai.workflow import WorkflowEngine
from kavalai.llm_clients.base_client import BaseLlmClient

DEMO_HTML = (
    Path(__file__).resolve().parents[1] / "docs" / "_includes" / "chatbot-demo.html"
)
MODEL = "Qwen2.5-1.5B-Instruct-q4f32_1-MLC"


def _chatbot_code() -> str:
    html = DEMO_HTML.read_text(encoding="utf-8")
    match = re.search(r"var CHATBOT_CODE =\s*`([^`]*)`", html)
    assert match, "Could not find CHATBOT_CODE in the demo partial"
    return match.group(1)


class _FakeClient(BaseLlmClient):
    def __init__(self, *a, **k):
        super().__init__()

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        reply = response_model(
            agent_response="Hi! How can I help?",
            choices=["Tell me a joke", "What can you do?", "Goodbye"],
        )
        await value_streamer.stream_partial(reply.model_dump_json())
        await value_streamer.stream_complete()


@pytest.fixture
def workflow():
    ns = {"KAVAL_BROWSER_MODEL": MODEL}
    exec(_chatbot_code(), ns)  # noqa: S102 - trusted, repo-owned demo code
    wf = ns["workflow"]
    wf.client_factory = lambda *a, **k: _FakeClient()
    return wf


def test_demo_builds_a_workflow(workflow):
    assert isinstance(workflow, WorkflowEngine)


def test_input_and_output_are_the_two_models(workflow):
    assert set(workflow.get_data_type("input").model_fields) == {"message"}
    assert set(workflow.get_data_type("output").model_fields) == {
        "agent_response",
        "choices",
    }


async def test_chatbot_returns_structured_reply_with_choices(workflow):
    # The chat widget sends each turn as {message: ...} (inputKey="message").
    state = await workflow.run({"message": "hello"})
    assert state.output_data["agent_response"] == "Hi! How can I help?"
    assert state.output_data["choices"] == [
        "Tell me a joke",
        "What can you do?",
        "Goodbye",
    ]
    assert state.trace == ["start", "reply", "end"]


def test_demo_uses_browser_model_and_choices_field():
    code = _chatbot_code()
    assert "browser/{KAVAL_BROWSER_MODEL}" in code
    assert "choices: list[str]" in code


def test_demo_has_memory_and_prompt_examples():
    from kavalai.agent_service import AgentService

    # The chatbot is built with an agent service, which (with use_history on
    # by default) is what gives it memory.
    ns = {"KAVAL_BROWSER_MODEL": MODEL}
    exec(_chatbot_code(), ns)  # noqa: S102 - trusted, repo-owned demo code
    assert isinstance(ns["workflow"].agent_service, AgentService)
    # The prompt carries at least one few-shot example for the tiny model.
    assert "Example:" in _chatbot_code()


async def test_chatbot_remembers_earlier_turns(workflow):
    """Persistence + use_history + a reused external id (what the chat widget
    passes) means each turn sees the conversation so far."""
    seen = []

    class _Recorder(_FakeClient):
        async def _run_chat_completions(self, chat_history, response_model, streamer):
            seen.append([m.content for m in chat_history.messages])
            await super()._run_chat_completions(chat_history, response_model, streamer)

    workflow.client_factory = lambda *a, **k: _Recorder()
    await workflow.run({"message": "my name is Tim"}, external_id="s")
    await workflow.run({"message": "what is my name?"}, external_id="s")

    # Turn 2's prompt includes turn 1's message — the bot has memory.
    turn_two = " ".join(c or "" for c in seen[1])
    assert "my name is Tim" in turn_two
