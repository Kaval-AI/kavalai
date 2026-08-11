import json

import pytest
from unittest.mock import patch

import anthropic
import openai
from google.genai import errors as genai_errors

from kavalai.llm_clients.with_retry import (
    _gemini_client_error,
    _import_attrs,
    _retriable_exceptions,
    with_retry,
)


def test_import_attrs_returns_attributes_of_an_installed_module():
    assert _import_attrs("json", "JSONDecodeError") == [json.JSONDecodeError]


def test_import_attrs_returns_empty_list_for_a_missing_module():
    # The provider SDKs are optional extras: an absent one contributes nothing.
    assert _import_attrs("kavalai_no_such_provider_sdk", "RateLimitError") == []


def test_retriable_exceptions_collect_every_installed_sdk():
    exceptions = _retriable_exceptions()
    assert openai.RateLimitError in exceptions
    assert genai_errors.ServerError in exceptions
    assert anthropic.RateLimitError in exceptions


def test_gemini_client_error_resolves_to_the_sdk_type():
    assert _gemini_client_error() is genai_errors.ClientError


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
