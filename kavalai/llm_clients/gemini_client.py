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
from typing import Any, Dict, List, Optional, Tuple, Type

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ensure_user_turn,
    BaseLlmClient,
    LlmClientException,
    ChatHistory,
    LlmClientParameters,
    ModelStatsReceiver,
)
from kavalai.llm_clients.common import walk_json_schema
from kavalai.llm_clients.streamer import Streamer

_SERVICE_TIERS = {
    "priority": types.ServiceTier.PRIORITY,
    "standard": types.ServiceTier.STANDARD,
    "flex": types.ServiceTier.FLEX,
}


class GeminiClient(BaseLlmClient):
    """
    Gemini LLM client implementation using the Streamer.
    """

    provider = "gemini"

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Gemini client.

        Args:
            model: The Gemini model name (e.g., 'gemini-1.5-flash').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            model_stats_receiver: Optional receiver for model call statistics.
            api_key: Optional API key (falls back to GEMINI_API_KEY env var).
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        try:
            self.client = genai.Client(api_key=self.api_key)
        except ValueError as e:
            raise LlmClientException(str(e)) from e

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual Gemini API call and stream results.
        """
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        system_instruction, contents = convert_messages(
            [msg.model_dump() for msg in ensure_user_turn(chat_history.messages)]
        )

        config_kwargs = {}
        params = self.parameters
        if params.temperature is not None:
            config_kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            config_kwargs["top_p"] = params.top_p
        if params.service_tier is not None:
            service_tier = params.service_tier.lower()
            if service_tier in _SERVICE_TIERS:
                config_kwargs["service_tier"] = _SERVICE_TIERS[service_tier]
                logger.info(f"Gemini Service Tier: {service_tier.upper()}")
        if params.reasoning_effort is not None:
            # Gemini has no effort levels here; any setting turns on streamed
            # thoughts, delivered on the "thought" value streamer below.
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
            )

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if response_model:
            config_kwargs["response_mime_type"] = "application/json"
            schema = response_model.model_json_schema()
            remove_additional_properties(schema)
            config_kwargs["response_schema"] = schema

        config = types.GenerateContentConfig(**config_kwargs)

        thought_streamer = None
        if config_kwargs.get("thinking_config"):
            thought_streamer = streamer.get_value_streamer("thought")

        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = None
        reasoning_tokens = None
        full_response = ""

        try:
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.thought:
                                if thought_streamer:
                                    await thought_streamer.stream_partial(part.text)
                            elif part.text:
                                full_response += part.text
                                await value_streamer.stream_partial(part.text)
                if chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    prompt_tokens = usage.prompt_token_count
                    completion_tokens = usage.candidates_token_count
                    # Context cache hits and "thinking" tokens, when reported.
                    cached_prompt_tokens = getattr(
                        usage, "cached_content_token_count", None
                    )
                    reasoning_tokens = getattr(usage, "thoughts_token_count", None)
        except Exception as e:
            logger.error(f"Gemini Stream Error: {e}")
            raise

        await value_streamer.stream_complete()
        if thought_streamer:
            await thought_streamer.stream_complete()

        await self._record_completed_call(
            request_data={
                "model": self.model,
                "contents": [str(c) for c in contents],
                "config": config_kwargs,
            },
            response_data=full_response,
            started=start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            reasoning_tokens=reasoning_tokens,
        )


def convert_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[types.Content]]:
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if system_instruction:
                system_instruction += "\n" + content
            else:
                system_instruction = content
            continue

        # Convert role to Gemini format (user or model)
        gemini_role = "user" if role == "user" else "model"

        parts = []
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item.get("text")))

        contents.append(types.Content(role=gemini_role, parts=parts))

    if not contents:
        if system_instruction:
            contents.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text=system_instruction)]
                )
            )
            system_instruction = None
        else:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text="...")])
            )

    return system_instruction, contents


def remove_additional_properties(schema: Dict[str, Any]) -> None:
    """Strip ``additionalProperties`` from every object schema, in place.

    Gemini's response schema does not accept the field.
    """
    walk_json_schema(schema, lambda node: node.pop("additionalProperties", None))
