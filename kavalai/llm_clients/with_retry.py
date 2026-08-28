"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import importlib
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class _NeverRaised(Exception):
    """Sentinel exception used as an ``except`` target for an absent optional SDK.

    It is never raised, so an ``except _NeverRaised`` clause is effectively a
    no-op when the corresponding provider package (``openai`` / ``google-genai``)
    is not installed.
    """


def _import_attrs(module_name: str, *attrs: str) -> list:
    """Return the named attributes of ``module_name``, or ``[]`` if it is absent.

    The provider SDKs are optional extras, so a package that is not installed
    simply contributes nothing instead of raising.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return []
    return [getattr(module, attr) for attr in attrs]


def _retriable_exceptions() -> tuple:
    """Collect retriable exception types from whichever LLM SDKs are installed.

    ``openai``, ``google-genai`` and ``anthropic`` are optional extras; when a
    package is absent its exception types simply do not contribute to the
    retry set.
    """
    exceptions = [
        *_import_attrs(
            "openai",
            "RateLimitError",
            "InternalServerError",
            "APIConnectionError",
            "APITimeoutError",
            "LengthFinishReasonError",
        ),
        *_import_attrs("google.genai.errors", "ServerError", "ClientError"),
        *_import_attrs(
            "anthropic",
            "RateLimitError",
            "InternalServerError",
            "APIConnectionError",
            "APITimeoutError",
        ),
    ]
    return tuple(exceptions) or (_NeverRaised,)


def _gemini_client_error() -> type:
    """Return the Gemini ``ClientError`` type, or a sentinel if google-genai is absent."""
    found = _import_attrs("google.genai.errors", "ClientError")
    return found[0] if found else _NeverRaised


def _is_rate_limited(error: Exception) -> bool:
    return getattr(error, "status", None) == 429 or "429" in str(error)


async def with_retry(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None,
    **kwargs,
) -> Any:
    """
    Exponential backoff retry wrapper for LLM client calls.
    Retries only on the transient error types of the OpenAI, Gemini and
    Anthropic SDKs; anything else (auth errors, programming errors) is raised
    at once.

    The provider SDKs are optional extras and are imported lazily, so this
    wrapper also works in lightweight / pyodide installs where none is present.

    :param func: The function to call.
    :param max_retries: Maximum number of retries.
    :param base_delay: Initial delay in seconds.
    :param max_delay: Maximum delay in seconds.
    :param on_retry: Optional async callback awaited before each retry attempt
        with ``(attempt_number, exception)``; used by streaming callers to
        signal a restart to consumers.
    :return: The result of the function call.
    """
    retriable_exceptions = _retriable_exceptions()
    gemini_client_error = _gemini_client_error()

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (*retriable_exceptions, gemini_client_error) as e:
            # Gemini reports rate limits as ``ClientError`` (429), the same
            # class as bad requests, auth failures and unknown models, which
            # never recover and are raised at once.
            if isinstance(e, gemini_client_error) and not _is_rate_limited(e):
                raise
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"LLM call to {args[0] if args else 'unknown'} failed with {type(e).__name__}: {str(e)}. "
                f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            if on_retry is not None:
                await on_retry(attempt + 1, e)
            await asyncio.sleep(delay)

    raise last_exception
