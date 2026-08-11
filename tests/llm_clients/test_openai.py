import os
import pytest
import json
from pydantic import BaseModel

from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.kwargs_mapper import is_openai_reasoning_model
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
    ModelStatsReceiver,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)


class SimpleResponse(BaseModel):
    answer: str


# Supported OpenAI models usable with the Responses API, as of June 2026.
# Sourced from the OpenAI models documentation
# (https://platform.openai.com/docs/models).
#
# Reasoning models (GPT-5 family + o-series) reject sampling params such as
# temperature/top_p; the remaining models still accept them.
OPENAI_REASONING_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "o3",
    "o4-mini",
]

OPENAI_SAMPLING_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]

SUPPORTED_OPENAI_MODELS = OPENAI_REASONING_MODELS + OPENAI_SAMPLING_MODELS

USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hello'")])


# ---------------------------------------------------------------------------
# Real Responses API events, so the client's isinstance dispatch is exercised
# for real instead of against mock stand-ins.
# ---------------------------------------------------------------------------


def text_delta(delta: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        content_index=0,
        delta=delta,
        item_id="item-1",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        type="response.output_text.delta",
    )


def refusal_delta(delta: str) -> ResponseRefusalDeltaEvent:
    return ResponseRefusalDeltaEvent(
        content_index=0,
        delta=delta,
        item_id="item-1",
        output_index=0,
        sequence_number=1,
        type="response.refusal.delta",
    )


def error_event(message: str) -> ResponseErrorEvent:
    return ResponseErrorEvent(message=message, sequence_number=1, type="error")


def completed(input_tokens: int = 10, output_tokens: int = 5) -> ResponseCompletedEvent:
    usage = ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    response = Response(
        id="resp-1",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )
    return ResponseCompletedEvent(
        response=response, sequence_number=2, type="response.completed"
    )


class FakeStream:
    """The async context manager returned by ``responses.stream(...)``."""

    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        async def iterator():
            for event in self.events:
                yield event

        return iterator()

    async def __aexit__(self, *exc_info):
        return False


class FakeResponses:
    def __init__(self, events):
        self.events = events
        self.call_kwargs = None

    def stream(self, **kwargs):
        self.call_kwargs = kwargs
        return FakeStream(self.events)


class FakeOpenAI:
    """Stands in for ``AsyncOpenAI`` with just the surface the client uses."""

    def __init__(self, events, timeout=None):
        self.responses = FakeResponses(events)
        self.timeout = timeout


class CollectingStats(ModelStatsReceiver):
    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


def make_client(*events, model="gpt-4o-mini", parameters=None, stats_receiver=None):
    """An OpenAIClient whose transport replays ``events``."""
    client = OpenAIClient(
        model=model,
        llm_client_parameters=parameters,
        model_stats_receiver=stats_receiver,
        api_key="fake",
    )
    client.client = FakeOpenAI(events, timeout=client.client.timeout)
    return client


def sent_kwargs(client):
    return client.client.responses.call_kwargs


# ---------------------------------------------------------------------------
# Unit tests (no API key required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_streams_text_deltas():
    stats_receiver = CollectingStats()
    client = make_client(
        text_delta("Hel"), text_delta("lo"), completed(), stats_receiver=stats_receiver
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    contents = [content async for content in streamer]

    assert [c.value for c in contents] == ["Hel", "Hello", "Hello"]
    assert contents[-1].type == "complete"

    (stats,) = stats_receiver.stats
    assert stats.model == "openai/gpt-4o-mini"
    assert (stats.prompt_tokens, stats.completion_tokens, stats.total_tokens) == (
        10,
        5,
        15,
    )


@pytest.mark.asyncio
async def test_structured_output_is_validated():
    client = make_client(text_delta('{"answer": "Paris"}'), completed())

    result = await client.chat_completions(
        chat_history=USER_HISTORY, response_model=SimpleResponse
    )

    assert result == SimpleResponse(answer="Paris")
    assert sent_kwargs(client)["text_format"] is SimpleResponse


@pytest.mark.asyncio
async def test_refusal_deltas_are_streamed_like_text():
    client = make_client(
        refusal_delta("I'm sorry, "), refusal_delta("I can't help."), completed()
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    contents = [content async for content in streamer]

    assert contents[-1].value == "I'm sorry, I can't help."


@pytest.mark.asyncio
async def test_stream_error_event_fails_the_stream():
    client = make_client(text_delta("partial"), error_event("upstream exploded"))

    with pytest.raises(RuntimeError, match="OpenAI Stream Error"):
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        async for _ in streamer:
            pass


@pytest.mark.asyncio
async def test_sampling_parameters_are_sent_for_sampling_models():
    client = make_client(
        text_delta("Hi"),
        completed(),
        parameters=LlmClientParameters(temperature=0.0, top_p=1.0),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    assert sent_kwargs(client)["temperature"] == 0.0
    assert sent_kwargs(client)["top_p"] == 1.0


@pytest.mark.asyncio
async def test_service_tier_and_reasoning_effort_are_forwarded():
    client = make_client(
        text_delta("Hi"),
        completed(),
        model="gpt-5.5",
        parameters=LlmClientParameters(service_tier="priority", reasoning_effort="low"),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    call_kwargs = sent_kwargs(client)
    assert call_kwargs["service_tier"] == "priority"
    # The Responses API nests effort under `reasoning`.
    assert call_kwargs["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_reasoning_model_omits_sampling_params():
    # GPT-5 family reasoning models reject top_p/temperature on the Responses API.
    client = make_client(
        text_delta("Hi"),
        completed(),
        model="gpt-5.5",
        parameters=LlmClientParameters(temperature=0.0, top_p=1.0),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    call_kwargs = sent_kwargs(client)
    assert "top_p" not in call_kwargs
    assert "temperature" not in call_kwargs
    assert call_kwargs["model"] == "gpt-5.5"


def test_timeout_comes_from_the_client_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="priority", timeout_seconds=45.0
    )
    client = OpenAIClient(
        model="gpt-4o-mini", llm_client_parameters=params, api_key="fake"
    )

    assert client.client.timeout == 45.0


# ---------------------------------------------------------------------------
# Integration tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_OPENAI_MODELS)
async def test_openai_supported_models_integration(model):
    """Hit the real Responses API for every supported model with parameters set.

    The same parameter set is sent for the whole lineup: reasoning models reject
    sampling params (temperature/top_p) and instead take reasoning_effort, while
    the other models take the sampling params. The client maps these per-model,
    so a real call must succeed for every supported model.
    """
    params = LlmClientParameters(temperature=0.0, top_p=1.0, timeout_seconds=60.0)
    if is_openai_reasoning_model(model):
        params.reasoning_effort = "low"

    client = OpenAIClient(model=model, llm_client_parameters=params)
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Reply with the single word: Hello")]
    )

    streamer = await client.stream_chat_completions(chat_history=chat_history)

    contents = [content async for content in streamer]

    assert contents[-1].type == "complete"
    assert contents[-1].value.strip() != ""


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_OPENAI_MODELS)
async def test_openai_supported_models_structured_output(model):
    """Structured output (response_model) works against every supported model."""
    params = LlmClientParameters(timeout_seconds=60.0)
    if is_openai_reasoning_model(model):
        params.reasoning_effort = "low"

    client = OpenAIClient(model=model, llm_client_parameters=params)
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user",
                content="What is the capital of France? Respond in JSON.",
            )
        ]
    )

    streamer = await client.stream_chat_completions(
        chat_history=chat_history, response_model=SimpleResponse
    )

    contents = [content async for content in streamer]

    assert contents[-1].type == "complete"
    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]
