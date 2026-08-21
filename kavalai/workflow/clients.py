"""Build LLM clients from ``provider/model`` identifiers.

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

from kavalai.llm_clients import registry
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
    **kwargs: Any,
) -> BaseLlmClient:
    """Construct an LLM client from a ``provider/model`` string.

    The provider is the part before the first ``/``; the remainder is the model
    name. Built-in providers are ``openai``, ``gemini``, ``anthropic``,
    ``ollama`` and ``browser``. The ``browser`` provider runs inference
    client-side via a WebLLM bridge (Pyodide only) and needs no API key --- see
    :class:`~kavalai.llm_clients.browser_client.BrowserLLMClient`.

    Any provider registered with :func:`~kavalai.register_llm_provider`
    resolves here too, and an exact ``provider/model`` registration takes
    precedence over the provider-wide one, so a single model can be pinned to
    its own client class.

    Args:
        model: The ``provider/model`` identifier.
        parameters: Optional per-call sampling and timeout parameters.
        stats_receiver: Where the client reports model-call statistics.
        **kwargs: Extra constructor arguments, overriding any bound at
            registration.

    Raises:
        ValueError: ``model`` has no provider prefix.
        RegistryError: No provider is registered for the prefix.
    """
    if "/" not in model:
        raise ValueError(f"Model must be in 'provider/model' form, got '{model}'.")
    _, model_name = model.split("/", maxsplit=1)
    return registry.llm_providers.build(
        model, model_name, parameters, stats_receiver, **kwargs
    )
