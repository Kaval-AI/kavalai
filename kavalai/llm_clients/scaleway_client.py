"""Chat client for Scaleway Generative APIs.

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

Scaleway serves open-weight models behind an OpenAI-compatible API, so the
client here uses the ``openai`` SDK pointed at Scaleway. It speaks Chat
Completions rather than the Responses API the ``openai/`` provider uses: Chat
Completions is the interface every Scaleway chat model supports. The embedding
client, :class:`~kavalai.llm_clients.embeddings.ScalewayEmbeddingClient`,
resolves its key and endpoint through the two functions below.

``SCALEWAY_API_URL`` is the serverless endpoint; :func:`scaleway_base_url`
decides which endpoint a client calls.
"""

import os
import time
from typing import Any, Optional, Type

from openai import AsyncOpenAI
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

SCALEWAY_API_URL = "https://api.scaleway.ai"


def scaleway_api_key(api_key: Optional[str] = None) -> str:
    """The Scaleway secret key: the argument, else ``SCW_SECRET_KEY``.

    A missing key is refused here. The ``openai`` SDK would otherwise fall
    back to ``OPENAI_API_KEY`` and send an OpenAI credential to Scaleway.
    """
    key = api_key or os.getenv("SCW_SECRET_KEY")
    if not key:
        raise LlmClientException(
            "No Scaleway API key: pass api_key= or set SCW_SECRET_KEY to an "
            "IAM secret key allowed to use Generative APIs."
        )
    return key


def scaleway_base_url(
    base_url: Optional[str] = None, project_id: Optional[str] = None
) -> str:
    """The endpoint to call: ``base_url``, else the serverless API.

    A serverless request is billed to the default project of the key's
    organisation unless a project id is placed in the path: ``project_id``,
    else ``SCW_DEFAULT_PROJECT_ID``. An explicit ``base_url`` — the endpoint
    of a dedicated deployment — is used as given.
    """
    if base_url:
        return base_url
    project = project_id or os.getenv("SCW_DEFAULT_PROJECT_ID")
    if project:
        return f"{SCALEWAY_API_URL}/{project}/v1"
    return f"{SCALEWAY_API_URL}/v1"


class ScalewayClient(BaseLlmClient):
    """Scaleway chat models through the Chat Completions API.

    Structured output is requested as ``response_format`` with a
    ``json_schema``. ``max_output_tokens`` is sent as
    ``max_completion_tokens`` and ``reasoning_effort`` under its own name; the
    efforts a model accepts differ per model and are not checked here.
    Reasoning text a model streams separately from its answer is not part of
    the response, though its tokens are counted.
    """

    provider = "scaleway"

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Scaleway client.

        Args:
            model: The Scaleway model name (e.g., 'gpt-oss-120b').
            llm_client_parameters: Optional parameters like temperature, top_p, etc.
            model_stats_receiver: Optional receiver for model call statistics.
            api_key: Optional secret key (falls back to SCW_SECRET_KEY).
            base_url: Optional endpoint; see :func:`scaleway_base_url`.
            project_id: Optional billing project; see :func:`scaleway_base_url`.
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.client = AsyncOpenAI(
            api_key=scaleway_api_key(api_key),
            base_url=scaleway_base_url(base_url, project_id),
            timeout=self.timeout_seconds,
        )

    def _call_kwargs(
        self, chat_history: ChatHistory, response_model: Optional[Type[BaseModel]]
    ) -> dict[str, Any]:
        """The Chat Completions request for one call."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in ensure_user_turn(chat_history.messages)
            ],
            "stream": True,
            # Without this the stream carries no token counts at all.
            "stream_options": {"include_usage": True},
        }

        params = self.parameters
        if params.temperature is not None:
            call_kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            call_kwargs["top_p"] = params.top_p
        if params.service_tier is not None:
            call_kwargs["service_tier"] = params.service_tier
        if params.reasoning_effort is not None:
            call_kwargs["reasoning_effort"] = params.reasoning_effort
        if params.max_output_tokens is not None:
            call_kwargs["max_completion_tokens"] = params.max_output_tokens

        if response_model:
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }
        return call_kwargs

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ) -> None:
        """
        Background task to handle the actual Scaleway API call and stream results.
        """
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        call_kwargs = self._call_kwargs(chat_history, response_model)

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason = None
        full_response = ""

        async for chunk in await self.client.chat.completions.create(**call_kwargs):
            if chunk.usage is not None:
                usage = chat_usage_counts(chunk.usage)
            for choice in chunk.choices:
                if choice.delta.content:
                    full_response += choice.delta.content
                    await value_streamer.stream_partial(choice.delta.content)
                finish_reason = choice.finish_reason or finish_reason

        if finish_reason == "length":
            raise self._output_truncated(
                finish_reason, full_response, request_data=call_kwargs, **usage
            )

        await value_streamer.stream_complete()
        await self._record_completed_call(
            request_data=call_kwargs,
            response_data=full_response,
            started=start_time,
            **usage,
        )


def chat_usage_counts(usage: Any) -> dict[str, Any]:
    """The token counts of a Chat Completions ``usage``, named as a model call stat.

    The detail objects are optional in the API and absent for most models, in
    which case the cached and reasoning counts stay ``None``.
    """
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_prompt_tokens": getattr(prompt_details, "cached_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
    }
