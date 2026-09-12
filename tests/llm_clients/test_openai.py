import os
import pytest
import json
from pydantic import BaseModel

from kavalai.llm_clients.openai_client import OpenAIClient
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    ModelStatsReceiver,
    OutputTruncatedError,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseIncompleteEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from tests.llm_clients.truncation_cases import (
    StatsCollector,
    streamed_text_over_the_cap_raises,
    structured_answer_fits_the_cap,
    structured_answer_over_the_cap_raises,
)


class SimpleResponse(BaseModel):
    answer: str


USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hello'")])


# Real Responses API events, so the client's isinstance dispatch is exercised
# for real instead of against mock stand-ins.


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


def usage(input_tokens=10, output_tokens=5, reasoning_tokens=0) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def response(**fields) -> Response:
    return Response(
        id="resp-1",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        **fields,
    )


def completed(input_tokens: int = 10, output_tokens: int = 5) -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        response=response(usage=usage(input_tokens, output_tokens)),
        sequence_number=2,
        type="response.completed",
    )


def incomplete(reason="max_output_tokens", **counts) -> ResponseIncompleteEvent:
    """The event a response cut short ends with, in place of ``completed``."""
    details = IncompleteDetails(reason=reason) if reason else None
    return ResponseIncompleteEvent(
        response=response(
            status="incomplete", incomplete_details=details, usage=usage(**counts)
        ),
        sequence_number=2,
        type="response.incomplete",
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
        self.calls = 0

    def stream(self, **kwargs):
        self.call_kwargs = kwargs
        self.calls += 1
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


# Unit tests (no API key required)


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
    # Sent as `text.format`, so the SDK does not validate a truncated answer
    # before the stream says why it stopped.
    assert "text_format" not in sent_kwargs(client)
    text_format = sent_kwargs(client)["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"]["required"] == ["answer"]


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
async def test_sampling_parameters_are_sent():
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
async def test_the_output_cap_is_sent_only_when_set():
    unset = make_client(text_delta("Hi"), completed())
    await unset.chat_completions(chat_history=USER_HISTORY)
    assert "max_output_tokens" not in sent_kwargs(unset)

    capped = make_client(
        text_delta("Hi"),
        completed(),
        parameters=LlmClientParameters(max_output_tokens=300),
    )
    await capped.chat_completions(chat_history=USER_HISTORY)
    assert sent_kwargs(capped)["max_output_tokens"] == 300


@pytest.mark.asyncio
async def test_a_response_cut_at_the_cap_raises_and_is_recorded():
    stats_receiver = CollectingStats()
    client = make_client(
        text_delta('{"answer": "Pa'),
        incomplete(input_tokens=12, output_tokens=40, reasoning_tokens=30),
        parameters=LlmClientParameters(max_output_tokens=40),
        stats_receiver=stats_receiver,
    )

    with pytest.raises(OutputTruncatedError) as caught:
        await client.chat_completions(
            chat_history=USER_HISTORY, response_model=SimpleResponse
        )

    error = caught.value
    assert (error.reason, error.max_output_tokens) == ("max_output_tokens", 40)
    assert error.partial_output == '{"answer": "Pa'
    assert "40 output tokens, 30 of them reasoning" in str(error)
    assert client.client.responses.calls == 1
    (stat,) = stats_receiver.stats
    assert (stat.prompt_tokens, stat.completion_tokens) == (12, 40)
    assert stat.reasoning_tokens == 30
    assert stat.response_code is None


@pytest.mark.asyncio
async def test_a_response_incomplete_for_another_reason_raises():
    client = make_client(text_delta("I can"), incomplete(reason="content_filter"))

    with pytest.raises(LlmClientException) as caught:
        await client.chat_completions(chat_history=USER_HISTORY)

    assert not isinstance(caught.value, OutputTruncatedError)
    assert "incomplete response (reason 'content_filter')" in str(caught.value)


@pytest.mark.asyncio
async def test_an_incomplete_response_without_details_still_raises():
    client = make_client(incomplete(reason=None))

    with pytest.raises(LlmClientException, match="reason 'unknown'"):
        await client.chat_completions(chat_history=USER_HISTORY)


@pytest.mark.asyncio
async def test_a_response_without_usage_records_zero_tokens():
    stats_receiver = CollectingStats()
    event = ResponseCompletedEvent(
        response=response(usage=None), sequence_number=2, type="response.completed"
    )
    client = make_client(text_delta("Hi"), event, stats_receiver=stats_receiver)

    assert await client.chat_completions(chat_history=USER_HISTORY) == "Hi"
    (stat,) = stats_receiver.stats
    assert (stat.prompt_tokens, stat.completion_tokens) == (0, 0)


def test_timeout_comes_from_the_client_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="priority", timeout_seconds=45.0
    )
    client = OpenAIClient(
        model="gpt-4o-mini", llm_client_parameters=params, api_key="fake"
    )

    assert client.client.timeout == 45.0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_structured_output_against_the_real_api():
    """One real call: streaming, parameter mapping and structured output.

    Deselected by default (see ``addopts`` in ``pyproject.toml``); CI runs it
    with ``-m integration``. One model is enough — which models exist and
    which parameters they accept is the provider's business, and a call with
    an unsupported parameter fails with the provider's own error.
    """
    params = LlmClientParameters(reasoning_effort="low", timeout_seconds=60.0)
    client = OpenAIClient(model="gpt-5-mini", llm_client_parameters=params)
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
    assert "Paris" in json.loads(contents[-1].value)["answer"]


openai_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def real_client(stats, **parameters):
    return OpenAIClient(
        model="gpt-5-mini",
        llm_client_parameters=LlmClientParameters(timeout_seconds=60.0, **parameters),
        model_stats_receiver=stats,
    )


@pytest.mark.integration
@openai_key
async def test_openai_structured_answer_within_a_generous_cap():
    stats = StatsCollector()
    client = real_client(stats, reasoning_effort="low", max_output_tokens=1024)
    await structured_answer_fits_the_cap(client, stats, cap=1024)


@pytest.mark.integration
@openai_key
async def test_openai_structured_answer_over_the_cap_raises():
    """``minimal`` effort, so the cap is spent on visible (partial) JSON."""
    stats = StatsCollector()
    client = real_client(stats, reasoning_effort="minimal", max_output_tokens=40)
    error = await structured_answer_over_the_cap_raises(client, stats, cap=40)
    assert error.reason == "max_output_tokens"


@pytest.mark.integration
@openai_key
async def test_openai_streamed_text_over_the_cap_raises():
    stats = StatsCollector()
    client = real_client(stats, reasoning_effort="minimal", max_output_tokens=40)
    await streamed_text_over_the_cap_raises(client, stats, cap=40)


@pytest.mark.integration
@openai_key
async def test_openai_reasoning_can_spend_the_whole_cap():
    """On a reasoning model the cap includes reasoning, so the visible output
    can be empty — still reported as a truncation, not an empty answer."""
    stats = StatsCollector()
    client = real_client(stats, reasoning_effort="low", max_output_tokens=16)
    error = await streamed_text_over_the_cap_raises(client, stats, cap=16)
    assert error.partial_output == ""
    assert "No visible output was produced" in str(error)
