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
    LlmClientParameters,
)


class SimpleResponse(BaseModel):
    answer: str


# Supported Anthropic models, as of August 2026. Claude 4.7+ and the 5 family
# reject sampling params such as temperature/top_p; older models still accept
# them. The client sends sampling params only when explicitly set, so the
# integration tests set them only for the older models.
ANTHROPIC_NO_SAMPLING_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
]

ANTHROPIC_SAMPLING_MODELS = [
    "claude-haiku-4-5",
]

SUPPORTED_ANTHROPIC_MODELS = ANTHROPIC_NO_SAMPLING_MODELS + ANTHROPIC_SAMPLING_MODELS


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
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens),
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


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
@pytest.mark.parametrize("model", SUPPORTED_ANTHROPIC_MODELS)
async def test_anthropic_supported_models_integration(model):
    """Hit the real Messages API for every supported model with parameters set.

    Current models reject sampling params (temperature/top_p) and instead take
    effort, while older models still accept the sampling params — so each part
    of the lineup gets the parameters it supports, and a real call must succeed
    for every supported model.
    """
    params = LlmClientParameters(timeout_seconds=60.0)
    if model in ANTHROPIC_SAMPLING_MODELS:
        params.temperature = 0.0
        params.top_p = 1.0
    else:
        params.reasoning_effort = "low"

    client = AnthropicClient(model=model, llm_client_parameters=params)
    chat_history = ChatHistory(
        messages=[ChatMessage(role="user", content="Reply with the single word: Hello")]
    )

    streamer = await client.stream_chat_completions(chat_history=chat_history)

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    assert contents[-1].value.strip() != ""


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
@pytest.mark.parametrize("model", SUPPORTED_ANTHROPIC_MODELS)
async def test_anthropic_supported_models_structured_output(model):
    """Structured output (response_model) works against every supported model."""
    params = LlmClientParameters(timeout_seconds=60.0)
    if model not in ANTHROPIC_SAMPLING_MODELS:
        params.reasoning_effort = "low"

    client = AnthropicClient(model=model, llm_client_parameters=params)
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

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[-1].type == "complete"
    data = json.loads(contents[-1].value)
    assert "Paris" in data["answer"]
