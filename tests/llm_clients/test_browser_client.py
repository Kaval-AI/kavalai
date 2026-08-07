import json
import sys
import types

import pytest
from pydantic import BaseModel

from kavalai.llm_clients import browser_client
from kavalai.llm_clients.browser_client import BrowserLLMClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
)


class Answer(BaseModel):
    answer: str


class CapturingReceiver(ModelStatsReceiver):
    """Collects every ModelCallStat the client reports."""

    def __init__(self):
        self.stats: list[ModelCallStat] = []

    def receive_model_stats(self, stats: ModelCallStat):
        self.stats.append(stats)


class FakeBridge:
    """Stand-in for ``window.kavalBrowserLLM`` exposed by the page."""

    def __init__(self, result: str | None = None, raise_exc: Exception | None = None):
        self._result = result
        self._raise_exc = raise_exc
        self.last_request: str | None = None

    async def chat(self, request_json: str) -> str:
        self.last_request = request_json
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


def _install_bridge(monkeypatch, bridge: FakeBridge | None, *, pyodide: bool = True):
    """Pretend we run under Pyodide and inject a fake ``js`` module."""
    monkeypatch.setattr(browser_client, "is_pyodide", lambda: pyodide)
    fake_js = types.ModuleType("js")
    if bridge is not None:
        fake_js.kavalBrowserLLM = bridge
    monkeypatch.setitem(sys.modules, "js", fake_js)


async def _drain(streamer):
    contents = []
    async for content in streamer:
        contents.append(content)
    return contents


@pytest.mark.asyncio
async def test_chat_completions_streams_content_and_reports_stats(monkeypatch):
    bridge = FakeBridge(
        result=json.dumps(
            {
                "content": "red, blue, yellow",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
            }
        )
    )
    _install_bridge(monkeypatch, bridge)

    receiver = CapturingReceiver()
    client = BrowserLLMClient(
        "Llama-3.2-1B-Instruct-q4f32_1-MLC", model_stats_receiver=receiver
    )
    chat_history = ChatHistory(
        messages=[
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="Three primary colours?"),
        ]
    )

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    contents = await _drain(streamer)

    assert contents[-1].type == "complete"
    assert contents[-1].value == "red, blue, yellow"

    # The request handed to the bridge is a JSON string with our messages/params.
    sent = json.loads(bridge.last_request)
    assert sent["model"] == "Llama-3.2-1B-Instruct-q4f32_1-MLC"
    assert sent["messages"][0] == {"role": "system", "content": "You are terse."}
    # Sampling params default to None and are only sent when explicitly set.
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "response_format" not in sent

    assert len(receiver.stats) == 1
    stat = receiver.stats[0]
    assert stat.model == "browser/Llama-3.2-1B-Instruct-q4f32_1-MLC"
    assert stat.prompt_tokens == 7
    assert stat.completion_tokens == 5
    assert stat.total_tokens == 12
    assert stat.response_data == "red, blue, yellow"


@pytest.mark.asyncio
async def test_structured_output_passes_schema_and_parses(monkeypatch):
    bridge = FakeBridge(
        result=json.dumps({"content": '{"answer": "Paris"}', "usage": {}})
    )
    _install_bridge(monkeypatch, bridge)

    client = BrowserLLMClient("Qwen2.5-0.5B-Instruct-q4f16_1-MLC")
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Capital of France?")]
    )

    result = await client.chat_completions(
        chat_history=chat_history, response_model=Answer
    )

    assert isinstance(result, Answer)
    assert result.answer == "Paris"

    sent = json.loads(bridge.last_request)
    assert sent["response_format"]["type"] == "json_object"
    schema = sent["response_format"]["schema"]
    assert isinstance(schema, dict)
    assert "answer" in schema["properties"]


@pytest.mark.asyncio
async def test_prompt_wraps_system_message_and_returns_content(monkeypatch):
    bridge = FakeBridge(result=json.dumps({"content": "42", "usage": {}}))
    _install_bridge(monkeypatch, bridge)

    client = BrowserLLMClient("model-x")

    result = await client.prompt("What is six times seven?")

    assert result == "42"

    # prompt() builds a lone system message; the browser client relabels it to
    # ``user`` so WebLLM (which rejects a trailing system message) accepts it.
    sent = json.loads(bridge.last_request)
    assert sent["messages"] == [{"role": "user", "content": "What is six times seven?"}]
    assert "response_format" not in sent


@pytest.mark.asyncio
async def test_prompt_with_response_model_parses(monkeypatch):
    bridge = FakeBridge(
        result=json.dumps({"content": '{"answer": "Paris"}', "usage": {}})
    )
    _install_bridge(monkeypatch, bridge)

    client = BrowserLLMClient("model-x")

    result = await client.prompt("Capital of France?", response_model=Answer)

    assert isinstance(result, Answer)
    assert result.answer == "Paris"

    sent = json.loads(bridge.last_request)
    assert sent["messages"] == [{"role": "user", "content": "Capital of France?"}]
    assert sent["response_format"]["type"] == "json_object"
    assert "answer" in sent["response_format"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_usage_total_falls_back_to_sum(monkeypatch):
    # No total_tokens in usage: the client computes prompt + completion.
    bridge = FakeBridge(
        result=json.dumps(
            {"content": "ok", "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        )
    )
    _install_bridge(monkeypatch, bridge)

    receiver = CapturingReceiver()
    client = BrowserLLMClient(
        "gemma-2-2b-it-q4f16_1-MLC", model_stats_receiver=receiver
    )
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    await client.chat_completions(chat_history=chat_history)

    assert receiver.stats[0].total_tokens == 7


@pytest.mark.asyncio
async def test_missing_usage_defaults_to_zero(monkeypatch):
    bridge = FakeBridge(result=json.dumps({"content": "ok"}))
    _install_bridge(monkeypatch, bridge)

    receiver = CapturingReceiver()
    client = BrowserLLMClient("model-x", model_stats_receiver=receiver)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    await client.chat_completions(chat_history=chat_history)

    stat = receiver.stats[0]
    assert stat.prompt_tokens == 0
    assert stat.completion_tokens == 0
    assert stat.total_tokens == 0


@pytest.mark.asyncio
async def test_bridge_error_payload_raises(monkeypatch):
    bridge = FakeBridge(result=json.dumps({"error": "WebGPU not available"}))
    _install_bridge(monkeypatch, bridge)

    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    with pytest.raises(RuntimeError, match="WebGPU not available"):
        await _drain(streamer)


@pytest.mark.asyncio
async def test_bridge_exception_is_wrapped(monkeypatch):
    bridge = FakeBridge(raise_exc=ValueError("boom"))
    _install_bridge(monkeypatch, bridge)

    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    with pytest.raises(RuntimeError, match="In-browser LLM call failed"):
        await _drain(streamer)


@pytest.mark.asyncio
async def test_raises_outside_pyodide(monkeypatch):
    _install_bridge(monkeypatch, None, pyodide=False)

    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    with pytest.raises(RuntimeError, match="only works inside a Pyodide"):
        await _drain(streamer)


@pytest.mark.asyncio
async def test_raises_when_bridge_absent(monkeypatch):
    _install_bridge(monkeypatch, None, pyodide=True)

    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    with pytest.raises(RuntimeError, match="No in-browser LLM engine found"):
        await _drain(streamer)


def test_get_bridge_raises_outside_pyodide(monkeypatch):
    monkeypatch.setattr(browser_client, "is_pyodide", lambda: False)
    client = BrowserLLMClient("model-x")
    with pytest.raises(LlmClientException, match="only works inside a Pyodide"):
        client._get_bridge()


def test_build_request_relabels_trailing_system_message():
    # WebLLM rejects a conversation ending on a system message, so a lone
    # system message is relabeled to ``user``.
    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(
        messages=[ChatMessage(role="system", content="Answer briefly.")]
    )

    request = client._build_request(chat_history, None)

    assert request["messages"] == [{"role": "user", "content": "Answer briefly."}]


def test_build_request_keeps_leading_system_message():
    # A system message followed by a user turn is left untouched.
    client = BrowserLLMClient("model-x")
    chat_history = ChatHistory(
        messages=[
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="Capital of Estonia?"),
        ]
    )

    request = client._build_request(chat_history, None)

    assert request["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Capital of Estonia?"},
    ]


def test_build_request_omits_none_params():
    params = LlmClientParameters(temperature=None, top_p=None)
    client = BrowserLLMClient("model-x", llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(content="hi")])

    request = client._build_request(chat_history, None)

    # Roles default to "user" when unset; sampling params omitted when None.
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in request
    assert "top_p" not in request
    assert "response_format" not in request


def test_build_request_includes_explicit_sampling_params():
    params = LlmClientParameters(temperature=0.4, top_p=0.9)
    client = BrowserLLMClient("model-x", llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(content="hi")])

    request = client._build_request(chat_history, None)

    assert request["temperature"] == 0.4
    assert request["top_p"] == 0.9
