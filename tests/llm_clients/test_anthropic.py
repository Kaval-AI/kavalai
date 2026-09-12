import os
import pytest
import json
from unittest.mock import MagicMock
from pydantic import BaseModel

from kavalai.llm_clients.anthropic_client import (
    AnthropicClient,
    DEFAULT_MAX_TOKENS,
    convert_messages,
    forbid_additional_properties,
)
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    ModelStatsReceiver,
    OutputTruncatedError,
)
from tests.llm_clients.truncation_cases import (
    StatsCollector,
    streamed_text_over_the_cap_raises,
    structured_answer_fits_the_cap,
    structured_answer_over_the_cap_raises,
)


class SimpleResponse(BaseModel):
    answer: str


class FakeMessageStream:
    """Async context manager mimicking the SDK's MessageStreamManager."""

    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()

    async def get_final_message(self):
        return self._final_message


def make_text_event(text):
    return MagicMock(type="text", text=text)


def make_final_message(stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return MagicMock(
        stop_reason=stop_reason,
        usage=MagicMock(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def mock_stream(client, events, final_message):
    """Replace the client's SDK instance with a mock streaming the given events."""
    client.client = MagicMock()
    client.client.messages.stream = MagicMock(
        return_value=FakeMessageStream(events, final_message)
    )
    return client.client.messages.stream


@pytest.fixture
def anthropicclient():
    return AnthropicClient(
        model="claude-haiku-4-5", api_key=os.getenv("ANTHROPIC_API_KEY", "fake")
    )


@pytest.mark.asyncio
async def test_anthropic_chat_completions(anthropicclient):
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )
    mock_stream(
        anthropicclient,
        [make_text_event("Hel"), make_text_event("lo")],
        make_final_message(),
    )

    streamer = await anthropicclient.stream_chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert len(contents) >= 2
    assert any(c.type == "partial" for c in contents)
    assert contents[-1].type == "complete"
    assert "Hello" in contents[-1].value


@pytest.mark.asyncio
async def test_anthropic_structured_output(anthropicclient):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(
                role="user", content="What is the capital of France? Respond in JSON."
            )
        ]
    )
    stream_mock = mock_stream(
        anthropicclient,
        [make_text_event('{"answer": "Paris"}')],
        make_final_message(),
    )

    streamer = await anthropicclient.stream_chat_completions(
        chat_history=chat_history, response_model=SimpleResponse
    )

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]

    call_kwargs = stream_mock.call_args.kwargs
    output_format = call_kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False
    assert "answer" in output_format["schema"]["properties"]


@pytest.mark.asyncio
async def test_anthropic_parameters():
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, service_tier="auto", timeout_seconds=45.0
    )
    client = AnthropicClient(
        model="claude-haiku-4-5", llm_client_parameters=params, api_key="fake"
    )

    assert client.client.timeout == 45.0

    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])
    stream_mock = mock_stream(client, [make_text_event("Hi")], make_final_message())

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    async for content in streamer:
        if content.type == "complete":
            assert "Hi" in content.value

    call_kwargs = stream_mock.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    # The Messages API rejects a request carrying both sampling parameters, so
    # top_p is dropped when temperature is also set.
    assert "top_p" not in call_kwargs
    assert call_kwargs["service_tier"] == "auto"
    assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_anthropic_top_p_sent_when_temperature_unset():
    params = LlmClientParameters(top_p=0.9)
    client = AnthropicClient(
        model="claude-haiku-4-5", llm_client_parameters=params, api_key="fake"
    )

    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])
    stream_mock = mock_stream(client, [make_text_event("Hi")], make_final_message())

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    async for _ in streamer:
        pass

    call_kwargs = stream_mock.call_args.kwargs
    assert call_kwargs["top_p"] == 0.9
    assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_anthropic_default_parameters_omit_sampling_params():
    # temperature/top_p default to None and must not be sent unless set —
    # current Claude models reject them on the Messages API.
    params = LlmClientParameters(reasoning_effort="high")
    client = AnthropicClient(
        model="claude-opus-5", llm_client_parameters=params, api_key="fake"
    )

    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])
    stream_mock = mock_stream(client, [make_text_event("Hi")], make_final_message())

    streamer = await client.stream_chat_completions(chat_history=chat_history)
    async for _ in streamer:
        pass

    call_kwargs = stream_mock.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs
    assert call_kwargs["model"] == "claude-opus-5"
    assert call_kwargs["output_config"]["effort"] == "high"


@pytest.mark.asyncio
async def test_anthropic_system_message_extraction(anthropicclient):
    chat_history = ChatHistory(
        messages=[
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="Say 'Hi'"),
        ]
    )
    stream_mock = mock_stream(
        anthropicclient, [make_text_event("Hi")], make_final_message()
    )

    streamer = await anthropicclient.stream_chat_completions(chat_history=chat_history)
    async for _ in streamer:
        pass

    call_kwargs = stream_mock.call_args.kwargs
    assert call_kwargs["system"] == "You are terse."
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say 'Hi'"}]


@pytest.mark.asyncio
async def test_anthropic_system_only_prompt(anthropicclient):
    # BaseLlmClient.prompt() sends a single system message; the Messages API
    # requires a non-empty messages list, so it becomes the user turn.
    stream_mock = mock_stream(
        anthropicclient, [make_text_event("Hi")], make_final_message()
    )

    result = await anthropicclient.prompt("Say 'Hi'")

    assert "Hi" in result
    call_kwargs = stream_mock.call_args.kwargs
    assert "system" not in call_kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say 'Hi'"}]


@pytest.mark.asyncio
async def test_anthropic_refusal_raises(anthropicclient):
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])
    mock_stream(anthropicclient, [], make_final_message(stop_reason="refusal"))

    streamer = await anthropicclient.stream_chat_completions(chat_history=chat_history)
    with pytest.raises(RuntimeError, match="refused"):
        async for _ in streamer:
            pass


class CollectingStats(ModelStatsReceiver):
    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


HI = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hi'")])


@pytest.mark.asyncio
async def test_the_output_cap_replaces_the_required_default():
    client = AnthropicClient(
        model="claude-haiku-4-5",
        llm_client_parameters=LlmClientParameters(max_output_tokens=300),
        api_key="fake",
    )
    stream_mock = mock_stream(client, [make_text_event("Hi")], make_final_message())

    await client.chat_completions(chat_history=HI)

    assert stream_mock.call_args.kwargs["max_tokens"] == 300


@pytest.mark.asyncio
async def test_a_stop_at_max_tokens_raises_and_is_recorded():
    stats_receiver = CollectingStats()
    client = AnthropicClient(
        model="claude-haiku-4-5",
        llm_client_parameters=LlmClientParameters(max_output_tokens=16),
        model_stats_receiver=stats_receiver,
        api_key="fake",
    )
    mock_stream(
        client,
        [make_text_event('{"answer": "Pa')],
        make_final_message(stop_reason="max_tokens", input_tokens=20, output_tokens=16),
    )

    with pytest.raises(OutputTruncatedError) as caught:
        await client.chat_completions(chat_history=HI, response_model=SimpleResponse)

    error = caught.value
    assert (error.reason, error.max_output_tokens) == ("max_tokens", 16)
    assert error.partial_output == '{"answer": "Pa'
    (stat,) = stats_receiver.stats
    assert (stat.prompt_tokens, stat.completion_tokens) == (20, 16)
    assert stat.response_code is None


@pytest.mark.asyncio
async def test_a_stop_at_the_required_default_names_it(anthropicclient):
    mock_stream(anthropicclient, [], make_final_message(stop_reason="max_tokens"))

    with pytest.raises(OutputTruncatedError) as caught:
        await anthropicclient.chat_completions(chat_history=HI)

    assert caught.value.max_output_tokens == DEFAULT_MAX_TOKENS
    assert f"output cap of {DEFAULT_MAX_TOKENS} tokens" in str(caught.value)


@pytest.mark.asyncio
async def test_an_exhausted_context_window_raises(anthropicclient):
    mock_stream(
        anthropicclient,
        [make_text_event("Once")],
        make_final_message(stop_reason="model_context_window_exceeded"),
    )

    with pytest.raises(LlmClientException, match="context window") as caught:
        await anthropicclient.chat_completions(chat_history=HI)

    assert not isinstance(caught.value, OutputTruncatedError)


@pytest.mark.asyncio
async def test_anthropic_model_stats(anthropicclient):
    received = []
    anthropicclient.model_stats_receiver = MagicMock()
    anthropicclient.model_stats_receiver.receive_model_stats = received.append

    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Say 'Hello'")]
    )
    mock_stream(
        anthropicclient,
        [make_text_event("Hello")],
        make_final_message(input_tokens=12, output_tokens=7),
    )

    streamer = await anthropicclient.stream_chat_completions(chat_history=chat_history)
    async for _ in streamer:
        pass

    assert len(received) == 1
    stats = received[0]
    assert stats.model == "anthropic/claude-haiku-4-5"
    assert stats.prompt_tokens == 12
    assert stats.completion_tokens == 7
    assert stats.total_tokens == 19
    assert stats.response_data == "Hello"


def test_convert_messages_roles_and_system_join():
    system, messages = convert_messages(
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": None},
        ]
    )
    assert system == "A\nB"
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "result"},
    ]


def test_convert_messages_empty_history():
    system, messages = convert_messages([])
    assert system is None
    assert messages == [{"role": "user", "content": "..."}]


def test_forbid_additional_properties_nested():
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "items_field": {"type": "array", "items": {"type": "object"}},
            "union": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
        "$defs": {"Sub": {"type": "object", "properties": {}}},
        "allOf": [{"type": "object"}],
    }
    forbid_additional_properties(schema)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["nested"]["additionalProperties"] is False
    assert schema["properties"]["items_field"]["items"]["additionalProperties"] is False
    assert schema["properties"]["union"]["anyOf"][0]["additionalProperties"] is False
    assert "additionalProperties" not in schema["properties"]["union"]["anyOf"][1]
    assert schema["$defs"]["Sub"]["additionalProperties"] is False
    assert schema["allOf"][0]["additionalProperties"] is False


def test_forbid_additional_properties_non_dict():
    # Must not raise on non-dict input.
    forbid_additional_properties("not-a-schema")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
async def test_anthropic_structured_output_against_the_real_api():
    """One real call: streaming, parameter mapping and structured output.

    Deselected by default (see ``addopts`` in ``pyproject.toml``); CI runs it
    with ``-m integration``. One model is enough — which models exist and
    which parameters they accept is the provider's business, and a call with
    an unsupported parameter fails with the provider's own error.
    """
    params = LlmClientParameters(reasoning_effort="low", timeout_seconds=60.0)
    client = AnthropicClient(model="claude-sonnet-5", llm_client_parameters=params)
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


anthropic_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


def real_client(stats, **parameters):
    return AnthropicClient(
        model="claude-sonnet-5",
        llm_client_parameters=LlmClientParameters(timeout_seconds=60.0, **parameters),
        model_stats_receiver=stats,
    )


@pytest.mark.integration
@anthropic_key
async def test_anthropic_structured_answer_within_a_generous_cap():
    stats = StatsCollector()
    client = real_client(stats, reasoning_effort="low", max_output_tokens=1024)
    await structured_answer_fits_the_cap(client, stats, cap=1024)


@pytest.mark.integration
@anthropic_key
async def test_anthropic_structured_answer_over_the_cap_raises():
    stats = StatsCollector()
    client = real_client(stats, max_output_tokens=40)
    error = await structured_answer_over_the_cap_raises(client, stats, cap=40)
    assert error.reason == "max_tokens"


@pytest.mark.integration
@anthropic_key
async def test_anthropic_streamed_text_over_the_cap_raises():
    stats = StatsCollector()
    client = real_client(stats, max_output_tokens=40)
    await streamed_text_over_the_cap_raises(client, stats, cap=40)
