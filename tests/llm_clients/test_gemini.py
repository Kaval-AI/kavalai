import os
import json
import pytest
from google.genai import types
from pydantic import BaseModel

from kavalai.llm_clients.gemini_client import (
    GeminiClient,
    convert_messages,
    remove_additional_properties,
)
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    ModelStatsReceiver,
)


class SimpleResponse(BaseModel):
    answer: str


USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hello'")])


def text_chunk(*texts, thoughts=(), prompt_tokens=None, completion_tokens=None):
    """A generate_content_stream chunk built from real Gemini types."""
    parts = [types.Part(text=t, thought=True) for t in thoughts]
    parts += [types.Part(text=t) for t in texts]
    usage = None
    if prompt_tokens is not None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=completion_tokens
        )
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=usage,
    )


def empty_chunk():
    """A chunk with no candidates, as Gemini emits for filtered content."""
    return types.GenerateContentResponse(candidates=[])


class FakeModels:
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.call_kwargs = None

    async def generate_content_stream(self, **kwargs):
        self.call_kwargs = kwargs

        async def stream():
            if self.error is not None:
                raise self.error
            for chunk in self.chunks:
                yield chunk

        return stream()


class FakeGenaiClient:
    """Stands in for ``genai.Client`` with only the surface the client uses."""

    def __init__(self, chunks, error=None):
        self.aio = types_namespace = type("Aio", (), {})()
        types_namespace.models = FakeModels(chunks, error=error)


class CollectingStats(ModelStatsReceiver):
    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


def make_client(chunks=(), error=None, parameters=None, stats_receiver=None):
    """A GeminiClient whose transport replays ``chunks``."""
    client = GeminiClient(
        model="gemini-3.5-flash",
        llm_client_parameters=parameters,
        model_stats_receiver=stats_receiver,
        api_key="fake",
    )
    client.client = FakeGenaiClient(list(chunks), error=error)
    return client


def sent_config(client):
    return client.client.aio.models.call_kwargs["config"]


# ---------------------------------------------------------------------------
# Unit tests (no API key required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streams_text_and_reports_stats():
    stats_receiver = CollectingStats()
    client = make_client(
        chunks=[
            text_chunk("Hel"),
            text_chunk("lo", prompt_tokens=4, completion_tokens=2),
            empty_chunk(),
        ],
        stats_receiver=stats_receiver,
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    contents = [content async for content in streamer]

    assert [c.value for c in contents] == ["Hel", "Hello", "Hello"]
    (stats,) = stats_receiver.stats
    assert stats.model == "gemini/gemini-3.5-flash"
    assert (stats.prompt_tokens, stats.completion_tokens) == (4, 2)


@pytest.mark.asyncio
async def test_sampling_parameters_and_service_tier_are_mapped():
    client = make_client(
        chunks=[text_chunk("hi")],
        parameters=LlmClientParameters(
            temperature=0.25, top_p=0.9, service_tier="priority"
        ),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    config = sent_config(client)
    assert config.temperature == 0.25
    assert config.top_p == 0.9
    assert config.service_tier == types.ServiceTier.PRIORITY


@pytest.mark.asyncio
async def test_unknown_service_tier_is_ignored():
    client = make_client(
        chunks=[text_chunk("hi")],
        parameters=LlmClientParameters(service_tier="turbo"),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    [_ async for _ in streamer]

    assert sent_config(client).service_tier is None


@pytest.mark.asyncio
async def test_reasoning_effort_streams_thoughts_separately():
    client = make_client(
        chunks=[text_chunk("answer", thoughts=["pondering"])],
        parameters=LlmClientParameters(reasoning_effort="low"),
    )

    streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
    contents = [content async for content in streamer]

    assert sent_config(client).thinking_config.include_thoughts is True
    by_name = {(c.name, c.type): c.value for c in contents}
    assert by_name[("thought", "partial")] == "pondering"
    assert by_name[("response", "complete")] == "answer"


@pytest.mark.asyncio
async def test_system_message_becomes_a_system_instruction():
    client = make_client(chunks=[text_chunk("hi")])
    history = ChatHistory(
        messages=[
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="Hello?"),
        ]
    )

    streamer = await client.stream_chat_completions(chat_history=history)
    [_ async for _ in streamer]

    assert sent_config(client).system_instruction == "Be terse."


@pytest.mark.asyncio
async def test_response_model_sets_a_json_schema_without_additional_properties():
    client = make_client(chunks=[text_chunk('{"answer": "Paris"}')])

    result = await client.chat_completions(
        chat_history=USER_HISTORY, response_model=SimpleResponse
    )

    config = sent_config(client)
    assert config.response_mime_type == "application/json"
    assert "additionalProperties" not in json.dumps(config.response_schema)
    assert result == SimpleResponse(answer="Paris")


@pytest.mark.asyncio
async def test_stream_error_propagates():
    client = make_client(error=ValueError("gemini exploded"))

    with pytest.raises(RuntimeError, match="gemini exploded"):
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        async for _ in streamer:
            pass


def test_missing_api_key_raises_llm_client_exception(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LlmClientException):
        GeminiClient(model="gemini-3.5-flash")


# ---------------------------------------------------------------------------
# convert_messages / remove_additional_properties
# ---------------------------------------------------------------------------


def test_convert_messages_joins_system_messages_and_maps_roles():
    system_instruction, contents = convert_messages(
        [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )

    assert system_instruction == "First.\nSecond."
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "Hi"


def test_convert_messages_reads_text_items_from_list_content():
    _, contents = convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image", "url": "ignored"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
    )

    assert [p.text for p in contents[0].parts] == ["part one", "part two"]


def test_convert_messages_promotes_a_lone_system_message_to_the_user_turn():
    # Gemini requires a non-empty contents list, so a system-only history is
    # sent as the single user turn instead.
    system_instruction, contents = convert_messages(
        [{"role": "system", "content": "Only a system prompt."}]
    )

    assert system_instruction is None
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Only a system prompt."


def test_convert_messages_falls_back_to_a_placeholder_turn():
    system_instruction, contents = convert_messages([])

    assert system_instruction is None
    assert contents[0].parts[0].text == "..."


def test_remove_additional_properties_recurses_through_the_whole_schema():
    schema = {
        "additionalProperties": False,
        "properties": {"nested": {"additionalProperties": False}},
        "items": {"additionalProperties": False},
        "anyOf": [{"additionalProperties": False}],
        "$defs": {"Inner": {"additionalProperties": False}},
        "definitions": {"Legacy": {"additionalProperties": False}},
    }

    remove_additional_properties(schema)

    assert "additionalProperties" not in json.dumps(schema)


def test_remove_additional_properties_ignores_non_dict_input():
    # Reached via list schemas whose "items" is a bare type name.
    remove_additional_properties("string")


# Supported Gemini models usable with the generateContent API, as of June 2026.
# Sourced from the Gemini API models documentation
# (https://ai.google.dev/gemini-api/docs/models). Gemini 1.5 and 2.0 are retired.
SUPPORTED_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_GEMINI_MODELS)
async def test_gemini_supported_models_integration(model):
    """Hit the real generateContent API for every supported model with params set.

    Sends temperature/top_p (universally supported) plus reasoning_effort, which
    the client maps to a thinking config; a real call must succeed for every model.
    """
    params = LlmClientParameters(
        temperature=0.0, top_p=1.0, reasoning_effort="low", timeout_seconds=60.0
    )
    client = GeminiClient(model=model, llm_client_parameters=params)
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
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.parametrize("model", SUPPORTED_GEMINI_MODELS)
async def test_gemini_supported_models_structured_output(model):
    """Structured output (response_schema) works against every supported model."""
    params = LlmClientParameters(timeout_seconds=60.0)
    client = GeminiClient(model=model, llm_client_parameters=params)
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
