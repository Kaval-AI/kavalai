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
from typing import Optional, Type

import ollama
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    ensure_user_turn,
    BaseLlmClient,
    ChatHistory,
    LlmClientParameters,
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
        self.timeout = self.timeout_seconds
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

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in ensure_user_turn(chat_history.messages)
        ]

        params = self.parameters
        options = {}
        if params.temperature is not None:
            options["temperature"] = params.temperature
        if params.top_p is not None:
            options["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            options["num_predict"] = params.max_output_tokens

        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if params.reasoning_effort is not None:
            # `think` takes a boolean or, on models that grade it, "low",
            # "medium" or "high". "none" is the one effort name that means
            # off; any other value goes out as given for Ollama to judge.
            call_kwargs["think"] = (
                False if params.reasoning_effort == "none" else params.reasoning_effort
            )

        if response_model:
            # Ollama takes a full JSON Schema here (since v0.5). The older
            # `format="json"` only asks for *some* valid JSON, which small
            # models happily satisfy with an object of the wrong shape.
            call_kwargs["format"] = response_model.model_json_schema()

        prompt_tokens = 0
        completion_tokens = 0
        done_reason = None
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
                done_reason = chunk.get("done_reason")

        if done_reason == "length":
            raise self._output_truncated(
                done_reason,
                full_response,
                request_data=call_kwargs,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        await value_streamer.stream_complete()
        await self._record_completed_call(
            request_data=call_kwargs,
            response_data=full_response,
            started=start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
