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

import ollama
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


class OllamaClient(BaseLlmClient):
    """
    Ollama LLM client implementation using the Streamer.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the Ollama client.

        Args:
            model: The Ollama model name (e.g., 'llama3').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            model_stats_receiver: Optional receiver for model call statistics.
            host: Optional Ollama host (falls back to OLLAMA_HOST env var).
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        self.timeout = 30.0
        if self.parameters and self.parameters.timeout_seconds:
            self.timeout = self.parameters.timeout_seconds

        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout)

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """
        Background task to handle the actual Ollama API call and stream results.
        """
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        messages = []
        for msg in ensure_user_turn(chat_history.messages):
            message_dict = {"role": msg.role, "content": msg.content}
            messages.append(message_dict)

        options = {}
        if self.parameters:
            if self.parameters.temperature is not None:
                options["temperature"] = self.parameters.temperature
            if self.parameters.top_p is not None:
                options["top_p"] = self.parameters.top_p

        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        if response_model:
            # Ollama takes a full JSON Schema here (since v0.5). The older
            # `format="json"` only asks for *some* valid JSON, which small
            # models happily satisfy with an object of the wrong shape.
            call_kwargs["format"] = response_model.model_json_schema()

        prompt_tokens = 0
        completion_tokens = 0
        full_response = ""

        # Errors propagate: the caller's background task turns them into an
        # error chunk on the stream as with the other provider clients.
        async for chunk in await self.client.chat(**call_kwargs):
            if "message" in chunk and "content" in chunk["message"]:
                delta = chunk["message"]["content"]
                full_response += delta
                await value_streamer.stream_partial(delta)

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)

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
            response_code=200,
        )
        await self._send_model_call_stats(stats)
