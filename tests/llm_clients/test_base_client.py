import pytest
from unittest.mock import patch
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
    ModelStatsLogger,
    ModelCallStat,
)
import openai


class MockClient(BaseLlmClient):
    async def _run_chat_completions(self, chat_history, response_model, streamer):
        # This will be mocked in tests
        pass


@pytest.mark.asyncio
async def test_retry_logic():
    # Set a short timeout for the streamer to fail fast if it doesn't work
    params = LlmClientParameters(timeout_seconds=2.0)
    client = MockClient(llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="test")])

    # Mock _run_chat_completions to fail once with a retriable error and then succeed
    call_count = 0

    async def side_effect(chat_history, response_model, streamer):
        nonlocal call_count
        call_count += 1
        print(f"Call count: {call_count}")
        if call_count == 1:
            raise openai.APIConnectionError(request=None)

        # On second call, succeed
        value_streamer = streamer.get_value_streamer("response")
        await value_streamer.stream_partial("Success")
        await value_streamer.stream_complete()

    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        # We need to lower the retry delay for testing
        with patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None):
            streamer = await client.stream_chat_completions(chat_history=chat_history)

            contents = []
            async for content in streamer:
                contents.append(content)

            assert call_count == 2
            assert contents[-1].value == "Success"
            assert contents[-1].type == "complete"


@pytest.mark.asyncio
async def test_non_retriable_error():
    params = LlmClientParameters(timeout_seconds=1.0)
    client = MockClient(llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="test")])

    async def side_effect(chat_history, response_model, streamer):
        raise ValueError("Non-retriable error")

    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        # Should raise RuntimeError from the streamer
        with pytest.raises(RuntimeError, match="Non-retriable error"):
            streamer = await client.stream_chat_completions(chat_history=chat_history)
            async for _ in streamer:
                pass


@pytest.mark.asyncio
async def test_retry_emits_restart_chunk():
    """A retried attempt announces itself with a restart chunk so streaming
    consumers can discard the failed attempt's partials."""
    params = LlmClientParameters(timeout_seconds=2.0)
    client = MockClient(llm_client_parameters=params)
    chat_history = ChatHistory(messages=[ChatMessage(role="user", content="test")])

    call_count = 0

    async def side_effect(chat_history, response_model, streamer):
        nonlocal call_count
        call_count += 1
        value_streamer = streamer.get_value_streamer("response")
        if call_count == 1:
            await value_streamer.stream_partial("garbage")
            raise openai.APIConnectionError(request=None)
        await value_streamer.stream_partial("Success")
        await value_streamer.stream_complete()

    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        with patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None):
            streamer = await client.stream_chat_completions(chat_history=chat_history)
            contents = [content async for content in streamer]

    types = [c.type for c in contents]
    assert types == ["partial", "restart", "partial", "complete"]
    assert "attempt 1" in contents[1].value
    # Without reset_active() the stale registration would prevent termination.
    assert contents[-1].value == "Success"


@pytest.mark.asyncio
async def test_stream_timeout_defaults_to_twice_llm_timeout():
    client = MockClient(llm_client_parameters=LlmClientParameters(timeout_seconds=7.0))
    streamer = await client.stream_chat_completions(
        chat_history=ChatHistory(messages=[])
    )
    assert streamer._timeout_seconds == 14.0


@pytest.mark.asyncio
async def test_stream_timeout_explicit_override():
    client = MockClient(
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
    async def side_effect(chat_history, response_model, streamer):
        value_streamer = streamer.get_value_streamer("response")
        await value_streamer.stream_partial("a")
        await value_streamer.stream_partial("b")
        await value_streamer.stream_complete()

    client = MockClient()
    with patch.object(MockClient, "_run_chat_completions", side_effect=side_effect):
        streamer = await client.stream_chat_completions(
            chat_history=ChatHistory(messages=[]), stream_delta=True
        )
        contents = [content async for content in streamer]

    # Delta mode: raw deltas, and the complete chunk carries no value.
    assert [c.value for c in contents] == ["a", "b", None]


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
