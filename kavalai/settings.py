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

Environment readers shared by the entry points.

Library code takes its configuration as arguments; only the processes —
``python -m kavalai.server``, ``kavalai-eval`` — read the environment, and
the two things they read in the same shape live here so the names are spelled
in one place. Nothing else imports this module.
"""

import os
from typing import Optional

from kavalai.normalizer import Normalizer, set_default_normalizer

# ``KAVALAI_LLM_<NAME>`` -> ``llm_kwargs`` key, with the parser that turns the
# string into the value :class:`~kavalai.LlmClientParameters` expects.
LLM_PARAMETER_VARIABLES: dict[str, tuple[str, type]] = {
    "KAVALAI_LLM_TEMPERATURE": ("temperature", float),
    "KAVALAI_LLM_TOP_P": ("top_p", float),
    "KAVALAI_LLM_REASONING_EFFORT": ("reasoning_effort", str),
    "KAVALAI_LLM_SERVICE_TIER": ("service_tier", str),
    "KAVALAI_LLM_TIMEOUT_SECONDS": ("timeout_seconds", float),
    "KAVALAI_LLM_STREAM_TIMEOUT_SECONDS": ("stream_timeout_seconds", float),
}


def llm_parameters_from_env(environ: Optional[dict] = None) -> dict:
    """The fleet-wide ``llm_kwargs`` the ``KAVALAI_LLM_*`` variables describe.

    An unset or empty variable is left out, so the provider's own default
    applies; a value that does not parse is an error at start-up rather than
    at the first model call.
    """
    environ = os.environ if environ is None else environ
    parameters = {}
    for variable, (key, parse) in LLM_PARAMETER_VARIABLES.items():
        raw = environ.get(variable, "")
        if raw == "":
            continue
        try:
            parameters[key] = parse(raw)
        except ValueError as e:
            raise ValueError(f"{variable}={raw!r} is not a {parse.__name__}") from e
    return parameters


def apply_normalizer_from_env(environ: Optional[dict] = None) -> Optional[Normalizer]:
    """Install the normalizer ``KAVALAI_EMBEDDING_NORMALIZER_YAML`` points at.

    Returns the normalizer, or ``None`` when the variable is unset — the
    built-in L2 default then stays in force.
    """
    environ = os.environ if environ is None else environ
    path = environ.get("KAVALAI_EMBEDDING_NORMALIZER_YAML", "")
    if not path:
        return None
    normalizer = Normalizer.load_from_yaml(path)
    set_default_normalizer(normalizer)
    return normalizer
