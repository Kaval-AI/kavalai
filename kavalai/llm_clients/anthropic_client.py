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
from typing import Any, Dict, List, Optional, Tuple, Type

from anthropic import AsyncAnthropic
from loguru import logger
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    LlmClientException,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
)
from kavalai.llm_clients.streamer import Streamer

# The Messages API requires max_tokens on every request. It is only an output
# ceiling (nothing is pre-allocated) and responses stream, so a generous
# default is safe.
DEFAULT_MAX_TOKENS = 64000


class AnthropicClient(BaseLlmClient):
    """
    Anthropic (Claude) LLM client implementation using the Messages API and
    Streamer.
    """

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Anthropic client.

        Args:
            model: The Anthropic model name (e.g., 'claude-opus-5').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            model_stats_receiver: Optional receiver for model call statistics.
            api_key: Optional API key (falls back to ANTHROPIC_API_KEY env var).
            base_url: Optional base URL for the API.
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url

        timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            timeout = self.parameters.timeout_seconds

        self.client = AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=timeout
        )

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual Anthropic API call and stream results.
        """
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        system_prompt, messages = convert_messages(
            [msg.model_dump() for msg in chat_history.messages]
        )

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        output_config: Dict[str, Any] = {}
        if self.parameters:
            if self.parameters.temperature is not None:
                call_kwargs["temperature"] = self.parameters.temperature
            if self.parameters.top_p is not None:
                # The Messages API rejects a request carrying both sampling
                # parameters, so send only temperature when both are set.
                if "temperature" in call_kwargs:
                    logger.warning(
                        f"Anthropic model '{self.model}' accepts only one of "
                        "temperature and top_p; sending temperature and "
                        "ignoring top_p."
                    )
                else:
                    call_kwargs["top_p"] = self.parameters.top_p
            if self.parameters.service_tier is not None:
                call_kwargs["service_tier"] = self.parameters.service_tier
            if self.parameters.reasoning_effort is not None:
                output_config["effort"] = self.parameters.reasoning_effort

        if response_model:
            schema = response_model.model_json_schema()
            forbid_additional_properties(schema)
            output_config["format"] = {"type": "json_schema", "schema": schema}

        if output_config:
            call_kwargs["output_config"] = output_config

        prompt_tokens = 0
        completion_tokens = 0
        full_response = ""

        async with self.client.messages.stream(**call_kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    full_response += event.text
                    await value_streamer.stream_partial(event.text)
            final_message = await stream.get_final_message()

        if final_message.stop_reason == "refusal":
            raise LlmClientException(
                f"Anthropic model '{self.model}' refused to generate a response."
            )
        if final_message.usage:
            prompt_tokens = final_message.usage.input_tokens
            completion_tokens = final_message.usage.output_tokens

        await value_streamer.stream_complete()

        duration = time.perf_counter() - start_time
        stats = ModelCallStat(
            call_type="llm",
            model=f"anthropic/{self.model}",
            request_data=json.dumps(call_kwargs, default=str),
            response_data=full_response,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        await self._send_model_call_stats(stats)


def convert_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split chat messages into the system prompt and Anthropic message turns.

    The Messages API takes the system prompt as a top-level ``system``
    parameter and requires a non-empty ``messages`` list, so a system-only
    history (e.g. from :meth:`BaseLlmClient.prompt`) is sent as the single
    user turn instead.
    """
    system_prompt = None
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if content is None:
            continue

        if role == "system":
            if system_prompt:
                system_prompt += "\n" + content
            else:
                system_prompt = content
            continue

        anthropic_role = "user" if role == "user" else "assistant"
        converted.append({"role": anthropic_role, "content": content})

    if not converted:
        converted.append({"role": "user", "content": system_prompt or "..."})
        system_prompt = None

    return system_prompt, converted


def forbid_additional_properties(schema: Dict[str, Any]) -> None:
    """
    Recursively set 'additionalProperties: false' on all object schemas.
    Anthropic structured outputs require closed object schemas.
    """
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False

    # Recursively process nested objects
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            forbid_additional_properties(prop_schema)

    # Handle arrays
    if "items" in schema:
        forbid_additional_properties(schema["items"])

    # Handle allOf, anyOf, oneOf
    for key in ["allOf", "anyOf", "oneOf"]:
        if key in schema:
            for sub_schema in schema[key]:
                forbid_additional_properties(sub_schema)

    # Handle $defs or definitions (where nested models are stored)
    for key in ["$defs", "definitions"]:
        if key in schema:
            for def_schema in schema[key].values():
                forbid_additional_properties(def_schema)
