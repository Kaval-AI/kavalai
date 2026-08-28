import json
import os

import pytest
import httpx
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
    ModelStatsReceiver,
)
from kavalai.llm_clients.ollama_client import OllamaClient


class SimpleResponse(BaseModel):
    answer: str


def is_ollama_running():
    host = os.getenv("OLLAMA_HOST", "localhost:11434")
    url = f"http://{host}/api/tags" if "://" not in host else f"{host}/api/tags"
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not is_ollama_running(),
    reason="OLLAMA_HOST not set or Ollama service not reachable",
)

USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hello'")])


class FakeOllamaClient:
    """Stands in for ``ollama.AsyncClient``: records the call, replays chunks."""

    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.call_kwargs = None

    async def chat(self, **kwargs):
        self.call_kwargs = kwargs
        if self.error is not None:
            raise self.error

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


class CollectingStats(ModelStatsReceiver):
    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


def make_client(chunks=None, error=None, parameters=None, stats_receiver=None):
    """An OllamaClient talking to a fake Ollama instead of a live server."""
    client = OllamaClient(
        model="llama3.2:1b",
        llm_client_parameters=parameters,
        model_stats_receiver=stats_receiver,
    )
    client.client = FakeOllamaClient(chunks=chunks, error=error)
    return client


# Unit tests (no Ollama server required)


@pytest.mark.asyncio
async def test_streams_deltas_and_reports_token_stats():
    stats_receiver = CollectingStats()
    client = make_client(
        chunks=[
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}},
            {"done": True, "prompt_eval_count": 7, "eval_count": 3},
        ],
        stats_receiver=stats_receiver,
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    contents = [content async for content in streamer]

    assert [c.value for c in contents] == ["Hel", "Hello", "Hello"]
    assert contents[-1].type == "complete"

    (stats,) = stats_receiver.stats
    assert stats.model == "ollama/llama3.2:1b"
    assert (stats.prompt_tokens, stats.completion_tokens, stats.total_tokens) == (
        7,
        3,
        10,
    )
    assert stats.response_data == "Hello"


@pytest.mark.asyncio
async def test_sampling_parameters_are_sent_as_options():
    client = make_client(
        chunks=[{"message": {"content": "hi"}}, {"done": True}],
        parameters=LlmClientParameters(temperature=0.25, top_p=0.9),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    assert client.client.call_kwargs["options"] == {"temperature": 0.25, "top_p": 0.9}
    assert client.client.call_kwargs["stream"] is True
    assert "format" not in client.client.call_kwargs


@pytest.mark.asyncio
async def test_unset_parameters_send_no_options():
    client = make_client(chunks=[{"message": {"content": "hi"}}, {"done": True}])

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    assert client.client.call_kwargs["options"] == {}
    assert client.client.call_kwargs["messages"] == [
        {"role": "user", "content": "Say 'Hello'"}
    ]


@pytest.mark.asyncio
async def test_response_model_requests_the_json_schema():
    """The response model's schema is sent, not the legacy `format="json"`.

    Plain JSON mode only constrains the output to *valid* JSON, so a small
    model can satisfy it with an object of an entirely different shape.
    """
    payload = '{"answer": "Paris"}'
    client = make_client(
        chunks=[{"message": {"content": payload}}, {"done": True}],
    )

    result = await client.chat_completions(
        chat_history=USER_HISTORY, response_model=SimpleResponse
    )

    assert client.client.call_kwargs["format"] == SimpleResponse.model_json_schema()
    assert result == SimpleResponse(answer="Paris")


@pytest.mark.asyncio
async def test_provider_error_surfaces_on_the_stream():
    client = make_client(error=ValueError("ollama exploded"))

    with pytest.raises(RuntimeError, match="ollama exploded"):
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        async for _ in streamer:
            pass


def test_host_and_timeout_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = OllamaClient(model="llama3.2:1b")
    assert client.host == "http://localhost:11434"
    assert client.timeout == 30.0


def test_ollama_parameters():
    params = LlmClientParameters(temperature=0.0, top_p=1.0, timeout_seconds=45.0)
    client = OllamaClient(model="llama3.2:1b", llm_client_parameters=params)

    assert client.timeout == 45.0
    assert client.model == "llama3.2:1b"


# Integration tests (require a reachable Ollama)


@pytest.fixture
def model_name():
    return "llama3.2:1b"


@pytest.fixture
def ollama_client(model_name):
    return OllamaClient(model=model_name)


@requires_ollama
@pytest.mark.asyncio
async def test_ollama_chat_completions(ollama_client):
    streamer = await ollama_client.stream_chat_completions(chat_history=USER_HISTORY)

    contents = [content async for content in streamer]

    assert len(contents) >= 2
    assert any(c.type == "partial" for c in contents)
    assert contents[-1].type == "complete"
    assert "Hello" in contents[-1].value


@requires_ollama
@pytest.mark.asyncio
async def test_ollama_structured_output(ollama_client):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user",
                content="What is the capital of France? Respond in JSON format with field 'answer'.",
            )
        ]
    )

    streamer = await ollama_client.stream_chat_completions(
        chat_history=chat_history, response_model=SimpleResponse
    )

    contents = [content async for content in streamer]

    assert contents[-1].type == "complete"
    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]


@pytest.mark.asyncio
async def test_system_only_history_is_sent_as_a_user_turn():
    """Ollama passes roles through, so the normalisation has to happen first."""
    client = make_client(chunks=[{"message": {"content": "hi"}}, {"done": True}])

    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(
            messages=[ChatMessage(role="system", content="Say 'Hello'")]
        )
    )
    [_ async for _ in streamer]

    assert client.client.call_kwargs["messages"] == [
        {"role": "user", "content": "Say 'Hello'"}
    ]
