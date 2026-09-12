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

``OUTPUT_CAP_NAMES`` are the names other libraries and providers give the
output cap; an ``llm_kwargs`` key spelled so is pointed at
``max_output_tokens``.
"""

from difflib import get_close_matches
from typing import Any, Optional

from kavalai.llm_clients import registry
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    LlmClientParameters,
    ModelStatsReceiver,
)
from kavalai.workflow.models import WorkflowException

OUTPUT_CAP_NAMES = frozenset(
    {"max_tokens", "max_completion_tokens", "num_predict", "max_new_tokens"}
)


def build_parameters(llm_kwargs: Optional[dict[str, Any]]) -> LlmClientParameters:
    """Build :class:`LlmClientParameters` from a node's ``llm_kwargs``.

    Every key must be a field of :class:`LlmClientParameters`. An unknown key
    raises rather than being dropped: a misspelled ``temperature`` or a
    provider's own name for the output cap would otherwise leave the call
    running on defaults the author believes they changed.

    Raises:
        WorkflowException: A key is not a parameter; the message names it,
            suggests the closest parameter and lists the valid ones.
    """
    kwargs = llm_kwargs or {}
    valid = LlmClientParameters.model_fields
    unknown = sorted(key for key in kwargs if key not in valid)
    if unknown:
        raise WorkflowException(unknown_keys_message(unknown, sorted(valid)))
    return LlmClientParameters(**kwargs)


def unknown_keys_message(unknown: list[str], valid: list[str]) -> str:
    """The error text for ``llm_kwargs`` keys that are not parameters."""
    described = []
    for key in unknown:
        if key in OUTPUT_CAP_NAMES:
            suggestion = "max_output_tokens"
        else:
            close = get_close_matches(key, valid, n=1)
            suggestion = close[0] if close else None
        described.append(
            f"'{key}' (did you mean '{suggestion}'?)" if suggestion else f"'{key}'"
        )
    noun = "key" if len(unknown) == 1 else "keys"
    return (
        f"Unknown llm_kwargs {noun} {', '.join(described)}. "
        f"Valid keys: {', '.join(valid)}."
    )


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
