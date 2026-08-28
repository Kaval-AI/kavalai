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
import json
import time
from typing import Any, Optional, Type, Literal

from pydantic import BaseModel
from loguru import logger

from kavalai.llm_clients.streamer import Streamer
from kavalai.llm_clients.with_retry import with_retry


class LlmClientParameters(BaseModel):
    """Optional per-call LLM parameters.

    Sampling parameters default to ``None`` and are only sent to a provider
    when explicitly set, so each provider's own defaults apply otherwise
    (some models, e.g. recent Claude models, reject sampling params outright).
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None
    timeout_seconds: Optional[float] = 30.0
    # Inter-chunk inactivity timeout for streaming consumers. When unset,
    # ``stream_chat_completions`` uses 2 x timeout_seconds so the stream survives
    # one timed-out attempt plus its retry backoff (timeout_seconds alone
    # would kill retries).
    stream_timeout_seconds: Optional[float] = None


class ChatMessage(BaseModel):
    """Standard chat completion message."""

    role: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


def ensure_user_turn(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Guarantee the history contains a turn a model can answer.

    Workflow ``llm`` nodes render their whole prompt into a single ``system``
    message, and :meth:`BaseLlmClient.prompt` does the same. A chat request
    with no user turn is unusual and providers disagree about it: Anthropic's
    Messages API rejects an empty ``messages`` list outright, while small local
    models answer it literally — ``llama3.2`` prefixes its reply with a literal
    "assistant" line.

    So the convention is: if nothing but system content is present, that content
    becomes the user turn. Histories that already contain a user or assistant
    message are returned untouched.
    """
    if any(msg.role not in (None, "system") for msg in messages):
        return messages

    content = "\n".join(msg.content for msg in messages if msg.content)
    if not content:
        return messages
    return [ChatMessage(role="user", content=content)]


class ModelCallStat(BaseModel):
    call_type: Literal["llm", "embedding"]
    model: Optional[str] = None
    request_data: Optional[str] = None
    response_data: Optional[str] = None
    response_code: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Subsets of the two counts above, when the provider reports them.
    cached_prompt_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    batch_size: Optional[int] = None
    duration_seconds: Optional[float] = None


class ModelStatsReceiver:
    def receive_model_stats(self, stats: ModelCallStat):
        raise NotImplementedError("You must implement this in the subclass.")


class ModelStatsLogger(ModelStatsReceiver):
    """Logs model call statistics using a configurable format."""

    def __init__(self, format_str: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            format_str: Optional python format string.
                        Default: "Model stats ({model}): {total_tokens} tokens, {duration_seconds:.2f}s"
        """
        self.format_str = (
            format_str
            or "Model stats ({model}): {total_tokens} tokens, {duration_seconds:.2f}s"
        )

    def receive_model_stats(self, stats: ModelCallStat):
        logger.info(self.format_str.format(**stats.model_dump()))


def error_status_code(error: Exception) -> Optional[int]:
    """Best-effort HTTP status from a provider SDK exception.

    Every provider raises its own error type, but all of them carry the status
    somewhere: ``status_code`` (OpenAI, Anthropic, Ollama), ``code`` (Gemini),
    or a nested response object.
    """
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


class BaseLlmClient:
    #: Provider prefix used when recording model calls (``openai/gpt-4o``).
    provider: str = ""

    def __init__(
        self,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
    ):
        self.parameters = llm_client_parameters or LlmClientParameters()
        self.model_stats_receiver = (
            ModelStatsLogger() if model_stats_receiver is None else model_stats_receiver
        )

    @property
    def timeout_seconds(self) -> float:
        """Per-request timeout, 30 s unless ``LlmClientParameters`` sets one."""
        return self.parameters.timeout_seconds or 30.0

    async def stream_chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
        stream_delta: bool = False,
    ) -> Streamer:
        """
        Execute a chat completion and return a Streamer.

        Args:
            chat_history: The history of messages.
            response_model: Optional Pydantic model for structured output.
            stream_delta: When True, partial chunks carry only the newly
                generated text and consumers reassemble; otherwise each partial
                carries the full accumulated (safe-parsed) content so far.

        Returns:
            A Streamer instance that will yield the completion events.
        """
        # The inactivity timeout must outlast a single timed-out attempt plus
        # its retry backoff; retries reset the clock via the restart chunk.
        timeout = self.parameters.stream_timeout_seconds or 2 * self.timeout_seconds
        streamer = Streamer(stream_delta=stream_delta, timeout_seconds=timeout)

        started = time.perf_counter()

        async def _on_retry(attempt: int, error: Exception):
            # The next attempt re-registers its value streamers and re-sends
            # content from scratch; tell consumers to discard what they have.
            streamer.reset_active()
            # Each failed attempt gets its own row, so a 429 storm is visible
            # in the Model Calls table rather than only in the logs.
            await self._record_failed_call(error, time.perf_counter() - started)
            await streamer.stream_restart(f"attempt {attempt}: {error}")

        async def _run():
            try:
                await with_retry(
                    self._run_chat_completions,
                    chat_history=chat_history,
                    response_model=response_model,
                    streamer=streamer,
                    on_retry=_on_retry,
                )
            except Exception as e:
                await self._record_failed_call(e, time.perf_counter() - started)
                await streamer.stream_error(e)

        # Start the completion process in the background with retry
        asyncio.create_task(_run())

        return streamer

    async def chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
    ):
        streamer = await self.stream_chat_completions(
            chat_history=chat_history, response_model=response_model
        )
        # The streamer only stops iterating once a 'complete' chunk has arrived,
        # so the loop always returns; falling through yields None implicitly.
        async for chunk in streamer:
            if chunk.type == "complete":
                if response_model:
                    return response_model.model_validate_json(chunk.value)
                return chunk.value

    async def stream_prompt(
        self, system_message: str, response_model: Optional[Type[BaseModel]] = None
    ) -> Streamer:
        history = ChatHistory(
            messages=[ChatMessage(role="system", content=system_message)]
        )
        return await self.stream_chat_completions(
            chat_history=history, response_model=response_model
        )

    async def prompt(
        self, system_message: str, response_model: Optional[Type[BaseModel]] = None
    ):
        history = ChatHistory(
            messages=[ChatMessage(role="system", content=system_message)]
        )
        return await self.chat_completions(
            chat_history=history, response_model=response_model
        )

    @classmethod
    def from_model(
        cls,
        model: str,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
        **defaults,
    ) -> "BaseLlmClient":
        """Construct this client for a resolved model name.

        The registry calls this rather than the constructor, so a client whose
        ``__init__`` does not take the shape below can say so once here instead
        of making every registration wrap it in a lambda. ``defaults`` are the
        keyword arguments bound at registration (a base URL, a host, a key).

        Args:
            model: Model name with the provider prefix already removed.
            parameters: Optional per-call sampling/timeout parameters.
            stats_receiver: Where model-call statistics are reported. Passing
                it through is what puts a custom client's usage in the
                backoffice alongside the built-in providers.
            **defaults: Extra constructor arguments bound at registration.
        """
        return cls(
            model,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
            **defaults,
        )

    async def _send_model_call_stats(self, stats: ModelCallStat):
        """Subclasses should use this method to report model stats."""
        if self.model_stats_receiver is not None:
            self.model_stats_receiver.receive_model_stats(stats)

    def stat_model_name(self) -> str:
        """The ``provider/model`` name this client records its calls under."""
        model = getattr(self, "model", "") or ""
        return f"{self.provider}/{model}" if self.provider else model

    async def _record_failed_call(self, error: Exception, duration: float) -> None:
        """Record an attempt that never produced a response.

        Successful calls were the only ones ever written, so the Model Calls
        table showed a suspiciously healthy service: a provider outage or a
        rate-limit storm left no trace at all. There are no token counts to
        report for a failed attempt, so the row carries the status code and the
        error text.
        """
        await self._send_model_call_stats(
            ModelCallStat(
                call_type="llm",
                model=self.stat_model_name(),
                response_code=error_status_code(error),
                response_data=str(error),
                duration_seconds=duration,
            )
        )

    async def _record_completed_call(
        self,
        *,
        request_data: Any,
        response_data: str,
        started: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
        cached_prompt_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> None:
        """Record a successful completion.

        ``request_data`` is serialised with ``json.dumps(default=str)``, so
        provider SDK objects (Pydantic models, enums) in the call kwargs are
        stored as their string form rather than failing the record.
        """
        await self._send_model_call_stats(
            ModelCallStat(
                call_type="llm",
                model=self.stat_model_name(),
                request_data=json.dumps(request_data, default=str),
                response_data=response_data,
                duration_seconds=time.perf_counter() - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(
                    prompt_tokens + completion_tokens
                    if total_tokens is None
                    else total_tokens
                ),
                cached_prompt_tokens=cached_prompt_tokens,
                reasoning_tokens=reasoning_tokens,
                response_code=200,
            )
        )

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """Perform the provider call and push chunks into ``streamer``.

        Runs as the background task started by :meth:`stream_chat_completions`;
        subclasses override it.
        """
        raise NotImplementedError("Subclasses must implement _run_chat_completions")


class LlmClientException(RuntimeError):
    pass
