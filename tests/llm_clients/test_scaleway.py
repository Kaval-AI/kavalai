import os
from unittest.mock import AsyncMock

import pytest
from openai.types import CompletionUsage, CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import (
    CompletionTokensDetails,
    PromptTokensDetails,
)
from openai.types.create_embedding_response import Usage
from pydantic import BaseModel

import kavalai
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    OutputTruncatedError,
)
from kavalai.llm_clients.embeddings import make_embedding_client
from kavalai.llm_clients.scaleway_client import (
    ScalewayClient,
    ScalewayEmbeddingClient,
    scaleway_api_key,
    scaleway_base_url,
)
from kavalai.workflow.clients import make_client as make_registered_client
from tests.llm_clients.truncation_cases import (
    StatsCollector,
    streamed_text_over_the_cap_raises,
    structured_answer_fits_the_cap,
    structured_answer_over_the_cap_raises,
)


class SimpleResponse(BaseModel):
    answer: str


USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="Say 'Hello'")])


@pytest.fixture(autouse=True)
def scaleway_environment(request, monkeypatch):
    """A key for the constructor, and no project id from the shell.

    The integration tests keep the real environment.
    """
    if "integration" in request.keywords:
        return
    monkeypatch.setenv("SCW_SECRET_KEY", "scw-test-key")
    monkeypatch.delenv("SCW_DEFAULT_PROJECT_ID", raising=False)


# Real Chat Completions chunks, as the OpenAI SDK parses them from the stream.


def completion_chunk(choices, usage=None):
    return ChatCompletionChunk(
        id="chunk-1",
        created=0,
        model="gpt-oss-120b",
        object="chat.completion.chunk",
        choices=choices,
        usage=usage,
    )


def chunk(content, finish_reason=None):
    return completion_chunk(
        [
            Choice(
                index=0,
                delta=ChoiceDelta(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def usage_chunk(prompt_tokens=7, completion_tokens=3, cached=None, reasoning=None):
    """The final chunk ``stream_options.include_usage`` adds: no choices."""
    return completion_chunk(
        [],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=(
                None if cached is None else PromptTokensDetails(cached_tokens=cached)
            ),
            completion_tokens_details=(
                None
                if reasoning is None
                else CompletionTokensDetails(reasoning_tokens=reasoning)
            ),
        ),
    )


class FakeCompletions:
    """Stands in for ``client.chat.completions``: records the call, replays chunks."""

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.call_kwargs = None

    async def create(self, **kwargs):
        self.call_kwargs = kwargs
        if self.error is not None:
            raise self.error

        async def stream():
            for item in self.chunks:
                yield item

        return stream()


def make_client(*chunks, error=None, parameters=None, stats_receiver=None):
    """A ScalewayClient whose SDK client replays ``chunks``."""
    client = ScalewayClient(
        model="gpt-oss-120b",
        llm_client_parameters=parameters,
        model_stats_receiver=stats_receiver,
    )
    completions = FakeCompletions(list(chunks), error=error)
    client.client.chat.completions.create = completions.create
    client.fake = completions
    return client


async def drain(client, chat_history=USER_HISTORY):
    streamer = await client.stream_chat_completions(chat_history=chat_history)
    return [content async for content in streamer]


@pytest.mark.asyncio
async def test_streams_deltas_and_reports_token_stats():
    stats_receiver = StatsCollector()
    client = make_client(
        chunk("Hel"),
        chunk("lo", finish_reason="stop"),
        usage_chunk(prompt_tokens=7, completion_tokens=3),
        stats_receiver=stats_receiver,
    )

    contents = await drain(client)

    assert [c.value for c in contents] == ["Hel", "Hello", "Hello"]
    assert contents[-1].type == "complete"
    (stats,) = stats_receiver.calls
    assert stats.model == "scaleway/gpt-oss-120b"
    assert (stats.prompt_tokens, stats.completion_tokens, stats.total_tokens) == (
        7,
        3,
        10,
    )
    assert (stats.cached_prompt_tokens, stats.reasoning_tokens) == (None, None)
    assert stats.response_data == "Hello"
    assert stats.response_code == 200


@pytest.mark.asyncio
async def test_cached_and_reasoning_tokens_are_recorded_when_reported():
    stats_receiver = StatsCollector()
    client = make_client(
        chunk("hi", finish_reason="stop"),
        usage_chunk(prompt_tokens=20, completion_tokens=50, cached=12, reasoning=35),
        stats_receiver=stats_receiver,
    )

    await drain(client)

    (stats,) = stats_receiver.calls
    assert (stats.cached_prompt_tokens, stats.reasoning_tokens) == (12, 35)


@pytest.mark.asyncio
async def test_a_stream_without_usage_records_zero_tokens():
    stats_receiver = StatsCollector()
    client = make_client(
        chunk("hi", finish_reason="stop"), stats_receiver=stats_receiver
    )

    await drain(client)

    (stats,) = stats_receiver.calls
    assert (stats.prompt_tokens, stats.completion_tokens) == (0, 0)


@pytest.mark.asyncio
async def test_unset_parameters_are_not_sent():
    client = make_client(chunk("hi", finish_reason="stop"))

    await drain(client)

    assert client.fake.call_kwargs == {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "Say 'Hello'"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.asyncio
async def test_parameters_are_sent_under_the_chat_completions_names():
    client = make_client(
        chunk("hi", finish_reason="stop"),
        parameters=LlmClientParameters(
            temperature=0.25,
            top_p=0.9,
            service_tier="flex",
            reasoning_effort="low",
            max_output_tokens=512,
        ),
    )

    await drain(client)

    sent = client.fake.call_kwargs
    assert sent["temperature"] == 0.25
    assert sent["top_p"] == 0.9
    assert sent["service_tier"] == "flex"
    assert sent["reasoning_effort"] == "low"
    assert sent["max_completion_tokens"] == 512
    assert "max_tokens" not in sent


@pytest.mark.asyncio
async def test_structured_output_is_requested_as_a_json_schema_and_validated():
    client = make_client(
        chunk('{"answer": '), chunk('"Hello"}', finish_reason="stop"), usage_chunk()
    )

    result = await client.chat_completions(
        chat_history=USER_HISTORY, response_model=SimpleResponse
    )

    assert result == SimpleResponse(answer="Hello")
    assert client.fake.call_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "SimpleResponse",
            "schema": SimpleResponse.model_json_schema(),
        },
    }


@pytest.mark.asyncio
async def test_a_system_only_history_gains_a_user_turn():
    client = make_client(chunk("hi", finish_reason="stop"))
    history = ChatHistory(messages=[ChatMessage(role="system", content="Be brief.")])

    await drain(client, history)

    assert client.fake.call_kwargs["messages"][-1]["role"] == "user"


@pytest.mark.asyncio
async def test_a_response_cut_at_the_cap_raises_and_is_recorded():
    stats_receiver = StatsCollector()
    client = make_client(
        chunk("Once upon"),
        chunk(" a", finish_reason="length"),
        usage_chunk(prompt_tokens=9, completion_tokens=40, reasoning=10),
        parameters=LlmClientParameters(max_output_tokens=40),
        stats_receiver=stats_receiver,
    )

    with pytest.raises(OutputTruncatedError) as caught:
        await drain(client)

    error = caught.value
    assert error.reason == "length"
    assert error.max_output_tokens == 40
    assert error.partial_output == "Once upon a"
    (stats,) = stats_receiver.calls
    assert stats.response_code is None
    assert (stats.prompt_tokens, stats.completion_tokens) == (9, 40)
    assert stats.reasoning_tokens == 10


@pytest.mark.asyncio
async def test_api_error_propagates_through_the_streamer():
    client = make_client(error=ValueError("scaleway exploded"))

    with pytest.raises(RuntimeError, match="scaleway exploded"):
        await drain(client)


DEPLOYMENT = "https://deployment.ifr.fr-par.scaleway.com/v1"


@pytest.mark.parametrize(
    "env_project, base_url, project_id, expected",
    [
        (None, None, None, "https://api.scaleway.ai/v1"),
        ("env-project", None, None, "https://api.scaleway.ai/env-project/v1"),
        ("env-project", None, "given", "https://api.scaleway.ai/given/v1"),
        ("env-project", DEPLOYMENT, "given", DEPLOYMENT),
    ],
)
def test_the_endpoint(monkeypatch, env_project, base_url, project_id, expected):
    if env_project:
        monkeypatch.setenv("SCW_DEFAULT_PROJECT_ID", env_project)

    assert scaleway_base_url(base_url, project_id) == expected


def test_the_client_is_built_with_the_resolved_key_and_endpoint():
    from_environment = ScalewayClient(model="gpt-oss-120b")
    from_arguments = ScalewayClient(
        model="gpt-oss-120b", api_key="given-key", project_id="given"
    )

    assert from_environment.client.api_key == "scw-test-key"
    assert str(from_environment.client.base_url) == "https://api.scaleway.ai/v1/"
    assert from_arguments.client.api_key == "given-key"
    assert str(from_arguments.client.base_url) == "https://api.scaleway.ai/given/v1/"


def test_a_missing_key_is_refused_rather_than_replaced_by_the_openai_key(
    monkeypatch,
):
    monkeypatch.delenv("SCW_SECRET_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    with pytest.raises(LlmClientException, match="SCW_SECRET_KEY"):
        scaleway_api_key()
    with pytest.raises(LlmClientException, match="SCW_SECRET_KEY"):
        ScalewayClient(model="gpt-oss-120b")
    with pytest.raises(LlmClientException, match="SCW_SECRET_KEY"):
        ScalewayEmbeddingClient("qwen3-embedding-8b")


def test_timeout_comes_from_the_client_parameters():
    client = ScalewayClient(
        model="gpt-oss-120b",
        llm_client_parameters=LlmClientParameters(timeout_seconds=12),
    )

    assert client.client.timeout == 12


def test_the_provider_is_registered_and_exported():
    client = make_registered_client("scaleway/gpt-oss-120b")

    assert isinstance(client, ScalewayClient)
    assert client.model == "gpt-oss-120b"
    assert kavalai.ScalewayClient is ScalewayClient
    assert "scaleway" in kavalai.registered_llm_providers()
    assert "scaleway" in kavalai.registered_embedding_providers()


# Embeddings


def embedding_response(*vectors, tokens=7):
    return CreateEmbeddingResponse(
        data=[
            Embedding(embedding=vector, index=i, object="embedding")
            for i, vector in enumerate(vectors)
        ],
        model="qwen3-embedding-8b",
        object="list",
        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
    )


def embedding_client_returning(response, **init):
    client = make_embedding_client("scaleway/qwen3-embedding-8b", **init)
    client.client.embeddings.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_embeddings_are_recorded_under_the_scaleway_prefix():
    client = embedding_client_returning(embedding_response([0.1, 0.2], [0.3, 0.4]))

    embeddings, stats = await client.compute_embeddings(["a", "b"])

    assert isinstance(client, ScalewayEmbeddingClient)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert stats.model == "scaleway/qwen3-embedding-8b"
    assert (stats.batch_size, stats.total_tokens) == (2, 7)
    sent = client.client.embeddings.create.call_args.kwargs
    assert sent["model"] == "qwen3-embedding-8b"
    assert "dimensions" not in sent


@pytest.mark.asyncio
async def test_embedding_dimensions_are_sent_when_set():
    client = embedding_client_returning(embedding_response([0.1]), dimensions=256)

    await client.compute_embeddings(["a"])

    assert client.client.embeddings.create.call_args.kwargs["dimensions"] == 256


def test_the_embedding_client_uses_the_scaleway_key_and_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    client = ScalewayEmbeddingClient("qwen3-embedding-8b", project_id="p")

    assert client.client.api_key == "scw-test-key"
    assert str(client.client.base_url) == "https://api.scaleway.ai/p/v1/"


# Integration tests (require SCW_SECRET_KEY)

scaleway_key = pytest.mark.skipif(
    not os.getenv("SCW_SECRET_KEY"), reason="SCW_SECRET_KEY not set"
)
SMOKE_MODEL = "mistral-small-3.2-24b-instruct-2506"


def real_client(stats, **parameters):
    return ScalewayClient(
        model=SMOKE_MODEL,
        llm_client_parameters=LlmClientParameters(timeout_seconds=60, **parameters),
        model_stats_receiver=stats,
    )


@pytest.mark.integration
@scaleway_key
@pytest.mark.asyncio
async def test_scaleway_structured_answer_within_a_generous_cap():
    stats = StatsCollector()
    await structured_answer_fits_the_cap(
        real_client(stats, max_output_tokens=1024), stats
    )


@pytest.mark.integration
@scaleway_key
@pytest.mark.asyncio
async def test_scaleway_structured_answer_over_the_cap_raises():
    stats = StatsCollector()
    await structured_answer_over_the_cap_raises(
        real_client(stats, max_output_tokens=40), stats
    )


@pytest.mark.integration
@scaleway_key
@pytest.mark.asyncio
async def test_scaleway_streamed_text_over_the_cap_raises():
    stats = StatsCollector()
    await streamed_text_over_the_cap_raises(
        real_client(stats, max_output_tokens=40), stats
    )


@pytest.mark.integration
@scaleway_key
@pytest.mark.asyncio
async def test_scaleway_embeddings_against_the_real_api():
    client = ScalewayEmbeddingClient("qwen3-embedding-8b", dimensions=256)

    embeddings, stats = await client.compute_embeddings(["tere", "hello world"])

    assert [len(vector) for vector in embeddings] == [256, 256]
    assert stats.model == "scaleway/qwen3-embedding-8b"
    assert stats.total_tokens > 0
