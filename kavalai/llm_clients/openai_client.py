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
import json
from typing import Optional, Type

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
)
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ensure_user_turn,
    BaseLlmClient,
    ChatHistory,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
)
from kavalai.llm_clients.streamer import Streamer
from kavalai.llm_clients.kwargs_mapper import is_openai_reasoning_model


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

        timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            timeout = self.parameters.timeout_seconds

        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=timeout
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

        messages = []
        for msg in ensure_user_turn(chat_history.messages):
            message_dict = {"role": msg.role, "content": msg.content}
            messages.append(message_dict)

        call_kwargs = {
            "model": self.model,
            "input": messages,
        }

        if self.parameters:
            # Reasoning models (GPT-5 family, o-series) reject sampling params
            # such as top_p/temperature on the Responses API.
            sampling_supported = not is_openai_reasoning_model(self.model)
            if sampling_supported and self.parameters.temperature is not None:
                call_kwargs["temperature"] = self.parameters.temperature
            if sampling_supported and self.parameters.top_p is not None:
                call_kwargs["top_p"] = self.parameters.top_p
            if self.parameters.service_tier is not None:
                call_kwargs["service_tier"] = self.parameters.service_tier
            if self.parameters.reasoning_effort is not None:
                # The Responses API takes reasoning effort nested under
                # `reasoning`, not as a top-level `reasoning_effort` kwarg
                # (that is the Chat Completions form).
                call_kwargs["reasoning"] = {"effort": self.parameters.reasoning_effort}

        if response_model:
            call_kwargs["text_format"] = response_model

        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = None
        reasoning_tokens = None
        full_response = ""

        async with self.client.responses.stream(**call_kwargs) as stream:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    full_response += event.delta
                    await value_streamer.stream_partial(event.delta)
                elif isinstance(event, ResponseRefusalDeltaEvent):
                    full_response += event.delta
                    await value_streamer.stream_partial(event.delta)
                elif isinstance(event, ResponseErrorEvent):
                    # We raise here to let the background task fail
                    # ResponseErrorEvent carries `message`/`code`, not `error`.
                    raise RuntimeError(f"OpenAI Stream Error: {event.message}")
                elif isinstance(event, ResponseCompletedEvent):
                    usage = event.response.usage
                    prompt_tokens = usage.input_tokens
                    completion_tokens = usage.output_tokens
                    # Cached input is billed at a fraction of fresh input, and
                    # reasoning tokens are output the caller never sees — both
                    # are subsets of the counts above.
                    input_details = getattr(usage, "input_tokens_details", None)
                    cached_prompt_tokens = getattr(input_details, "cached_tokens", None)
                    output_details = getattr(usage, "output_tokens_details", None)
                    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

        await value_streamer.stream_complete()

        duration = time.perf_counter() - start_time
        stats = ModelCallStat(
            call_type="llm",
            model=self.stat_model_name(),
            request_data=json.dumps(call_kwargs, default=str),
            response_data=full_response,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            response_code=200,
        )
        await self._send_model_call_stats(stats)
