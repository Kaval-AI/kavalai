import pytest
from unittest.mock import patch

import openai

from kavalai.llm_clients.with_retry import with_retry


class FakeGeminiError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def make_flaky(failures, exc_factory):
    """A coroutine failing ``failures`` times before returning "ok"."""
    calls = {"count": 0}

    async def func():
        calls["count"] += 1
        if calls["count"] <= failures:
            raise exc_factory()
        return "ok"

    return func, calls


@pytest.mark.asyncio
async def test_on_retry_called_per_retriable_attempt():
    func, calls = make_flaky(2, lambda: openai.APIConnectionError(request=None))
    retries = []

    async def on_retry(attempt, error):
        retries.append((attempt, type(error)))

    with patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None):
        result = await with_retry(func, on_retry=on_retry)

    assert result == "ok"
    assert calls["count"] == 3
    assert retries == [
        (1, openai.APIConnectionError),
        (2, openai.APIConnectionError),
    ]


@pytest.mark.asyncio
async def test_no_on_retry_after_final_attempt():
    func, calls = make_flaky(10, lambda: openai.APIConnectionError(request=None))
    retries = []

    async def on_retry(attempt, error):
        retries.append(attempt)

    with patch("kavalai.llm_clients.with_retry.asyncio.sleep", return_value=None):
        with pytest.raises(openai.APIConnectionError):
            await with_retry(func, max_retries=2, on_retry=on_retry)

    assert calls["count"] == 3
    # The exhausted final attempt does not announce another retry.
    assert retries == [1, 2]


@pytest.mark.asyncio
async def test_gemini_client_error_retries_with_on_retry():
    func, calls = make_flaky(1, lambda: FakeGeminiError("throttled", status=429))
    retries = []

    async def on_retry(attempt, error):
        retries.append((attempt, str(error)))

    with patch(
        "kavalai.llm_clients.with_retry._gemini_client_error",
        return_value=FakeGeminiError,
    ):
        result = await with_retry(func, on_retry=on_retry)

    assert result == "ok"
    assert calls["count"] == 2
    assert retries == [(1, "throttled")]


@pytest.mark.asyncio
async def test_gemini_404_not_retried():
    func, calls = make_flaky(1, lambda: FakeGeminiError("nope", status=404))
    retries = []

    async def on_retry(attempt, error):
        retries.append(attempt)

    with patch(
        "kavalai.llm_clients.with_retry._gemini_client_error",
        return_value=FakeGeminiError,
    ):
        with pytest.raises(FakeGeminiError):
            await with_retry(func, on_retry=on_retry)

    assert calls["count"] == 1
    assert retries == []


@pytest.mark.asyncio
async def test_gemini_404_in_message_not_retried():
    func, calls = make_flaky(1, lambda: FakeGeminiError("error 404 not found"))
    with patch(
        "kavalai.llm_clients.with_retry._gemini_client_error",
        return_value=FakeGeminiError,
    ):
        with pytest.raises(FakeGeminiError):
            await with_retry(func)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_non_retriable_raises_immediately():
    func, calls = make_flaky(1, lambda: ValueError("bad"))
    with pytest.raises(ValueError):
        await with_retry(func)
    assert calls["count"] == 1
