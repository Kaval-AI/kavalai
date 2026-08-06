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

from typing import Any, Optional

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    LlmClientParameters,
    ModelStatsReceiver,
)

# Known LlmClientParameters fields that may be supplied via a node's llm_kwargs.
_PARAM_FIELDS = set(LlmClientParameters.model_fields.keys())


def build_parameters(llm_kwargs: Optional[dict[str, Any]]) -> LlmClientParameters:
    """Build :class:`LlmClientParameters` from a node's ``llm_kwargs``.

    Recognised keys (temperature, top_p, reasoning_effort, service_tier,
    timeout_seconds) are mapped onto the parameters model; unknown keys are
    ignored so authors can keep provider-specific extras without breaking.
    """
    kwargs = llm_kwargs or {}
    known = {k: v for k, v in kwargs.items() if k in _PARAM_FIELDS}
    return LlmClientParameters(**known)


def make_client(
    model: str,
    parameters: Optional[LlmClientParameters] = None,
    stats_receiver: Optional[ModelStatsReceiver] = None,
) -> BaseLlmClient:
    """Construct a v2 LLM client from a ``provider/model`` string.

    Supported providers: ``openai``, ``gemini``, ``anthropic``, ``ollama``,
    ``browser``.
    The ``browser`` provider runs inference client-side via a WebLLM bridge
    (Pyodide only) and needs no API key — see
    :class:`~kavalai.llm_clients.browser_client.BrowserLLMClient`.
    """
    if "/" not in model:
        raise ValueError(f"Model must be in 'provider/model' form, got '{model}'.")
    provider, model_name = model.split("/", maxsplit=1)

    if provider == "openai":
        from kavalai.llm_clients.openai_client import OpenAIClient

        return OpenAIClient(
            model_name,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
        )
    if provider == "gemini":
        from kavalai.llm_clients.gemini_client import GeminiClient

        return GeminiClient(
            model_name,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
        )
    if provider == "anthropic":
        from kavalai.llm_clients.anthropic_client import AnthropicClient

        return AnthropicClient(
            model_name,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
        )
    if provider == "ollama":
        from kavalai.llm_clients.ollama_client import OllamaClient

        return OllamaClient(
            model_name,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
        )
    if provider == "browser":
        from kavalai.llm_clients.browser_client import BrowserLLMClient

        return BrowserLLMClient(
            model_name,
            llm_client_parameters=parameters,
            model_stats_receiver=stats_receiver,
        )
    raise ValueError(f"Unsupported LLM provider: '{provider}'.")
