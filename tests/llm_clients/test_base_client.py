import pytest
from unittest.mock import patch

from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
    ModelStatsLogger,
    ModelStatsReceiver,
    ModelCallStat,
)
import openai


class ScriptedClient(BaseLlmClient):
    """A client whose completion is driven by a small script.

    The script is called as ``script(attempt, streamer)`` for each attempt, so
    a test writes what the provider does instead of patching a private method.
    """

    def __init__(self, script=None, **kwargs):
        super().__init__(**kwargs)
        self.script = script
        self.attempts = 0

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        self.attempts += 1
        if self.script is not None:
            await self.script(self.attempts, streamer)


async def stream_text(streamer, *chunks, name="response"):
    """Stream ``chunks`` on ``name`` and complete it."""
    value_streamer = streamer.get_value_streamer(name)
    for chunk in chunks:
        await value_streamer.stream_partial(chunk)
    await value_streamer.stream_complete()


def no_backoff():
    """Skip the retry sleep so retry tests don't wait on real backoff."""
    return patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None)


USER_HISTORY = ChatHistory(messages=[ChatMessage(role="user", content="test")])


@pytest.mark.asyncio
async def test_retry_logic():
    async def flaky(attempt, streamer):
        if attempt == 1:
            raise openai.APIConnectionError(request=None)
        await stream_text(streamer, "Success")

    client = ScriptedClient(
        flaky, llm_client_parameters=LlmClientParameters(timeout_seconds=2.0)
    )

    with no_backoff():
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        contents = [content async for content in streamer]

    assert client.attempts == 2
    assert contents[-1].value == "Success"
    assert contents[-1].type == "complete"


@pytest.mark.asyncio
async def test_non_retriable_error():
    async def boom(attempt, streamer):
        raise ValueError("Non-retriable error")

    client = ScriptedClient(
        boom, llm_client_parameters=LlmClientParameters(timeout_seconds=1.0)
    )

    # The error reaches the consumer as a RuntimeError from the streamer.
    with pytest.raises(RuntimeError, match="Non-retriable error"):
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        async for _ in streamer:
            pass

    assert client.attempts == 1


@pytest.mark.asyncio
async def test_retry_emits_restart_chunk():
    """A retried attempt announces itself with a restart chunk so streaming
    consumers can discard the failed attempt's partials."""

    async def flaky(attempt, streamer):
        value_streamer = streamer.get_value_streamer("response")
        if attempt == 1:
            await value_streamer.stream_partial("garbage")
            raise openai.APIConnectionError(request=None)
        await value_streamer.stream_partial("Success")
        await value_streamer.stream_complete()

    client = ScriptedClient(
        flaky, llm_client_parameters=LlmClientParameters(timeout_seconds=2.0)
    )

    with no_backoff():
        streamer = await client.stream_chat_completions(chat_history=USER_HISTORY)
        contents = [content async for content in streamer]

    types = [c.type for c in contents]
    assert types == ["partial", "restart", "partial", "complete"]
    assert "attempt 1" in contents[1].value
    # Without reset_active() the stale registration would prevent termination.
    assert contents[-1].value == "Success"


@pytest.mark.asyncio
async def test_stream_timeout_defaults_to_twice_llm_timeout():
    client = ScriptedClient(
        llm_client_parameters=LlmClientParameters(timeout_seconds=7.0)
    )
    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(messages=[])
    )
    assert streamer._timeout_seconds == 14.0


@pytest.mark.asyncio
async def test_stream_timeout_explicit_override():
    client = ScriptedClient(
        llm_client_parameters=LlmClientParameters(
            timeout_seconds=7.0, stream_timeout_seconds=42.0
        )
    )
    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(messages=[])
    )
    assert streamer._timeout_seconds == 42.0


@pytest.mark.asyncio
async def test_stream_delta_passed_to_streamer():
    async def two_chunks(attempt, streamer):
        await stream_text(streamer, "a", "b")

    client = ScriptedClient(two_chunks)
    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(messages=[]), stream_delta=True
    )
    contents = [content async for content in streamer]

    # Delta mode: raw deltas, and the complete chunk carries no value.
    assert [c.value for c in contents] == ["a", "b", None]


@pytest.mark.asyncio
async def test_chat_completions_returns_the_completed_text():
    async def answer(attempt, streamer):
        await stream_text(streamer, "Hello", ", world!")

    client = ScriptedClient(answer)
    # Partials accumulate, so the completed value is the whole answer.
    result = await client.chat_completions(chat_history=USER_HISTORY)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_chat_completions_validates_against_the_response_model():
    class City(BaseModel):
        name: str
        country: str

    async def answer(attempt, streamer):
        await stream_text(streamer, '{"name": "Tallinn", "country": "Estonia"}')

    client = ScriptedClient(answer)
    result = await client.chat_completions(
        chat_history=USER_HISTORY, response_model=City
    )
    assert result == City(name="Tallinn", country="Estonia")


@pytest.mark.asyncio
async def test_prompt_wraps_the_message_as_a_system_turn():
    seen = {}

    async def capture(attempt, streamer):
        await stream_text(streamer, "answered")

    class CapturingClient(ScriptedClient):
        async def _run_chat_completions(self, chat_history, response_model, streamer):
            seen["messages"] = chat_history.messages
            await super()._run_chat_completions(chat_history, response_model, streamer)

    client = CapturingClient(capture)
    assert await client.prompt("What is Tallinn?") == "answered"
    assert [(m.role, m.content) for m in seen["messages"]] == [
        ("system", "What is Tallinn?")
    ]


@pytest.mark.asyncio
async def test_stream_prompt_returns_a_streamer_for_a_single_message():
    async def answer(attempt, streamer):
        await stream_text(streamer, "one", " two")

    client = ScriptedClient(answer)
    streamer = await client.stream_prompt("Count to two.")
    contents = [content async for content in streamer]

    assert [c.value for c in contents] == ["one", "one two", "one two"]
    assert contents[-1].type == "complete"


@pytest.mark.asyncio
async def test_base_run_chat_completions_must_be_implemented():
    client = BaseLlmClient()
    with pytest.raises(NotImplementedError):
        await client._run_chat_completions(
            chat_history=ChatHistory(messages=[]), response_model=None, streamer=None
        )


def test_model_stats_receiver_must_be_implemented():
    with pytest.raises(NotImplementedError):
        ModelStatsReceiver().receive_model_stats(
            ModelCallStat(call_type="llm", model="gpt-4")
        )


def test_model_stats_logger():
    with patch("kavalai.llm_clients.base_client.logger") as mock_logger:
        stats = ModelCallStat(
            call_type="llm",
            model="gpt-4",
            total_tokens=100,
            duration_seconds=1.5,
        )

        # Test default format
        logger_instance = ModelStatsLogger()
        logger_instance.receive_model_stats(stats)
        mock_logger.info.assert_called_with("Model stats (gpt-4): 100 tokens, 1.50s")

        # Test custom format
        logger_instance = ModelStatsLogger(format_str="Custom: {total_tokens}")
        logger_instance.receive_model_stats(stats)
        mock_logger.info.assert_called_with("Custom: 100")


def test_ensure_user_turn_promotes_a_system_only_history():
    """Workflow `llm` nodes render everything into one system message.

    Providers disagree about a request with no user turn — Anthropic rejects an
    empty message list, and llama3.2 answers with a literal `assistant\\n\\n`
    prefix — so the system content becomes the user turn.
    """
    from kavalai.llm_clients.base_client import ChatMessage, ensure_user_turn

    result = ensure_user_turn(
        [
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="system", content="Answer in Estonian."),
        ]
    )

    assert [(m.role, m.content) for m in result] == [
        ("user", "You are terse.\nAnswer in Estonian.")
    ]


def test_ensure_user_turn_leaves_a_real_conversation_alone():
    from kavalai.llm_clients.base_client import ChatMessage, ensure_user_turn

    messages = [
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="Hello?"),
    ]

    assert ensure_user_turn(messages) is messages


def test_error_status_code_reads_each_provider_shape():
    from kavalai.llm_clients.base_client import error_status_code

    class WithStatus(Exception):
        status_code = 429

    class WithCode(Exception):
        code = 503

    class WithResponse(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 500})()

    assert error_status_code(WithStatus()) == 429
    assert error_status_code(WithCode()) == 503
    assert error_status_code(WithResponse()) == 500
    assert error_status_code(ValueError("no status here")) is None


@pytest.mark.asyncio
async def test_failed_calls_are_recorded_with_their_status_code():
    """A provider outage must leave a trace, not just a healthy-looking table."""
    from kavalai.llm_clients.base_client import (
        BaseLlmClient,
        ChatHistory,
        ChatMessage,
        ModelStatsReceiver,
    )

    class Collector(ModelStatsReceiver):
        def __init__(self):
            self.stats = []

        def receive_model_stats(self, stats):
            self.stats.append(stats)

    class RateLimited(Exception):
        status_code = 429

    class AlwaysFailing(BaseLlmClient):
        provider = "fake"

        def __init__(self, receiver):
            super().__init__(None, receiver)
            self.model = "m"

        async def _run_chat_completions(self, chat_history, response_model, streamer):
            raise RateLimited("slow down")

    collector = Collector()
    client = AlwaysFailing(collector)

    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(messages=[ChatMessage(role="user", content="hi")])
    )
    with pytest.raises(RuntimeError):
        async for _ in streamer:
            pass

    assert collector.stats, "a failed call must still be recorded"
    stat = collector.stats[-1]
    assert stat.response_code == 429
    assert stat.model == "fake/m"
    assert stat.total_tokens is None
    assert "slow down" in stat.response_data
