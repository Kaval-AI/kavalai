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

import os
import time
from typing import Any, Optional, Type

from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
)
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ensure_user_turn,
    BaseLlmClient,
    ChatHistory,
    LlmClientException,
    LlmClientParameters,
    ModelStatsReceiver,
)
from kavalai.llm_clients.streamer import Streamer


class OpenAIClient(BaseLlmClient):
    """
    OpenAI LLM client implementation using the Responses API and Streamer.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            model: The OpenAI model name (e.g., 'gpt-4o').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            model_stats_receiver: Optional receiver for model call statistics.
            api_key: Optional API key (falls back to OPENAI_API_KEY env var).
            base_url: Optional base URL for the API.
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url

        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds
        )

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual OpenAI API call and stream results.
        """
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in ensure_user_turn(chat_history.messages)
        ]
        call_kwargs = {"model": self.model, "input": messages}

        params = self.parameters
        # Sampling parameters go out exactly as set. Which models accept them
        # changes with every release, so no model list is kept here: a model
        # that rejects a parameter fails the call with the provider's own
        # "Unsupported parameter" error rather than having it dropped silently.
        if params.temperature is not None:
            call_kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            call_kwargs["top_p"] = params.top_p
        if params.service_tier is not None:
            call_kwargs["service_tier"] = params.service_tier
        if params.reasoning_effort is not None:
            # The Responses API takes reasoning effort nested under
            # `reasoning`, not as a top-level `reasoning_effort` kwarg
            # (that is the Chat Completions form).
            call_kwargs["reasoning"] = {"effort": params.reasoning_effort}
        if params.max_output_tokens is not None:
            call_kwargs["max_output_tokens"] = params.max_output_tokens

        if response_model:
            # The schema goes out as `text.format` rather than `text_format`.
            # Given `text_format`, the SDK validates the text itself when it
            # ends, so a truncated answer would fail there as a validation
            # error before the stream reports why it stopped.
            call_kwargs["text"] = {"format": type_to_text_format_param(response_model)}

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        incomplete_reason = None
        full_response = ""

        async with self.client.responses.stream(**call_kwargs) as stream:
            async for event in stream:
                if isinstance(
                    event, (ResponseTextDeltaEvent, ResponseRefusalDeltaEvent)
                ):
                    full_response += event.delta
                    await value_streamer.stream_partial(event.delta)
                elif isinstance(event, ResponseErrorEvent):
                    # ResponseErrorEvent carries `message`/`code`, not `error`.
                    raise RuntimeError(f"OpenAI Stream Error: {event.message}")
                elif isinstance(
                    event, (ResponseCompletedEvent, ResponseIncompleteEvent)
                ):
                    if event.response.usage is not None:
                        usage = usage_counts(event.response.usage)
                    if isinstance(event, ResponseIncompleteEvent):
                        details = event.response.incomplete_details
                        incomplete_reason = getattr(details, "reason", None) or (
                            "unknown"
                        )

        if incomplete_reason == "max_output_tokens":
            raise self._output_truncated(
                incomplete_reason, full_response, request_data=call_kwargs, **usage
            )
        if incomplete_reason is not None:
            raise LlmClientException(
                f"OpenAI model '{self.model}' returned an incomplete response "
                f"(reason '{incomplete_reason}'); the partial output is not "
                "returned."
            )

        await value_streamer.stream_complete()
        await self._record_completed_call(
            request_data=call_kwargs,
            response_data=full_response,
            started=start_time,
            **usage,
        )


def usage_counts(usage: Any) -> dict:
    """The token counts of a Responses API ``usage``, named as a model call stat.

    Cached input is billed at a fraction of fresh input, and reasoning tokens
    are output the caller never sees — both are subsets of the two main counts.
    """
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "cached_prompt_tokens": getattr(input_details, "cached_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
    }
