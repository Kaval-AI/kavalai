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
from typing import Optional, Type, Literal

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


class ChatMessage(BaseModel):
    """Standard chat completion message."""

    role: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class ModelCallStat(BaseModel):
    call_type: Literal["llm", "embedding"]
    model: Optional[str] = None
    request_data: Optional[str] = None
    response_data: Optional[str] = None
    response_code: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
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


class BaseLlmClient:
    def __init__(
        self,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
    ):
        if not llm_client_parameters:
            llm_client_parameters = LlmClientParameters()
        self.parameters = llm_client_parameters
        self.streamer = None
        self.model_stats_receiver = model_stats_receiver
        if self.model_stats_receiver is None:
            self.model_stats_receiver = ModelStatsLogger()

    async def stream_chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Streamer:
        """
        Execute a chat completion and return a Streamer.

        Args:
            chat_history: The history of messages.
            response_model: Optional Pydantic model for structured output.

        Returns:
            A Streamer instance that will yield the completion events.
        """
        timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            timeout = self.parameters.timeout_seconds

        streamer = Streamer(timeout_seconds=timeout)

        async def _run():
            try:
                await with_retry(
                    self._run_chat_completions,
                    chat_history=chat_history,
                    response_model=response_model,
                    streamer=streamer,
                )
            except Exception as e:
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
        async for chunk in streamer:
            if chunk.type == "complete":
                if response_model:
                    return response_model.model_validate_json(chunk.value)
                return chunk.value
        return None

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

    async def _send_model_call_stats(self, stats: ModelCallStat):
        """Subclasses should use this method to report model stats."""
        if self.model_stats_receiver is not None:
            self.model_stats_receiver.receive_model_stats(stats)

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual LLM API call and stream results.
        This method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _run_chat_completions")


class LlmClientException(RuntimeError):
    pass


class BaseEmbeddingClient:
    pass
