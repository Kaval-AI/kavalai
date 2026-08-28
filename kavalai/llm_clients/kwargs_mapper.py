"""
Utilities to map user-friendly llm_kwargs into provider-specific parameters.

This allows writing workflow YAML with common options like:

llm_kwargs:
  reasoning: high
  temperature: 0.2
  max_tokens: 512

and have them mapped correctly for different providers (OpenAI, Gemini,
Anthropic, Ollama).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


class LLMKWargsMapper:
    """Map common/user-friendly kwargs to provider-specific ones.

    Rules (initial set):
    - temperature/top_p: passed-through for both providers
    - max_tokens -> max_output_tokens (both providers)
    - stop vs stop_sequences:
        * OpenAI Responses API uses `stop`
        * Gemini uses `stop_sequences`
    - reasoning (common):
        * OpenAI -> reasoning_effort (low|medium|high)
        * Gemini -> thinking_level (low|medium|high) OR thinking_budget (int)
    - Provider-specific passthrough and filtering to avoid invalid keys.
    """

    # Minimal safe key allow-lists per provider to avoid passing unsupported params.
    _OPENAI_KEYS: set[str] = {
        "temperature",
        "top_p",
        "max_output_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        # Reasoning
        "reasoning_effort",
        # Service Tier
        "service_tier",
        # Streaming/structured handled by clients; we won't filter them here if present
        "response_format",
    }

    _GEMINI_KEYS: set[str] = {
        "temperature",
        "top_p",
        "top_k",
        "candidate_count",
        "max_output_tokens",
        "stop_sequences",
        # Reasoning/thinking
        "thinking_budget",
        "thinking_level",
        "service_tier",
        # Structured output/system
        "response_mime_type",
        "response_schema",
        "system_instruction",
    }

    _ANTHROPIC_KEYS: set[str] = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop_sequences",
        # Reasoning
        "effort",
        # Service Tier
        "service_tier",
    }

    _OLLAMA_KEYS: set[str] = {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop",
        "num_ctx",
        "num_predict",
        "repeat_penalty",
        "seed",
        "format",
    }

    @staticmethod
    def _pop_any(d: Dict[str, Any], keys: Iterable[str]) -> Any | None:
        for k in keys:
            if k in d:
                return d.pop(k)
        return None

    @classmethod
    def map(cls, provider: str, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        provider = (provider or "").lower()
        out: Dict[str, Any] = dict(kwargs or {})

        # Normalize common synonyms
        if "max_output_tokens" not in out and "max_tokens" in out:
            out["max_output_tokens"] = out.pop("max_tokens")

        if provider == "openai":
            return cls._map_openai(out)
        elif provider == "gemini":
            return cls._map_gemini(out)
        elif provider == "anthropic":
            return cls._map_anthropic(out)
        elif provider == "ollama":
            return cls._map_ollama(out)
        else:
            # Unknown provider: return original kwargs
            return out

    @classmethod
    def _map_openai(cls, out: Dict[str, Any]) -> Dict[str, Any]:
        # Convert stop_sequences -> stop
        if "stop" not in out and "stop_sequences" in out:
            out["stop"] = out.pop("stop_sequences")

        # Convert priority -> service_tier
        # priority: high -> service_tier: priority (priority processing)
        # priority: normal -> service_tier: default (default processing)
        # priority: low -> service_tier: flex (slow/flex processing)
        if "service_tier" not in out and "priority" in out:
            val = out.pop("priority")
            if val == "high":
                out["service_tier"] = "priority"
            elif val == "low":
                out["service_tier"] = "flex"
            elif val == "normal":
                out["service_tier"] = "default"

        # Convert reasoning -> reasoning_effort
        if "reasoning_effort" not in out and "reasoning" in out:
            val = out.pop("reasoning")
            if isinstance(val, dict):
                effort = val.get("effort") or val.get("level")
                if effort is not None:
                    out["reasoning_effort"] = str(effort).lower()
            elif isinstance(val, str):
                out["reasoning_effort"] = val.lower()

        # Remove Gemini-only keys that could break the API call
        for k in [
            "top_k",
            "candidate_count",
            "stop_sequences",
            "thinking_budget",
            "response_mime_type",
            "response_schema",
            "system_instruction",
        ]:
            out.pop(k, None)

        # Filter to allowed keys (but keep unknowns to avoid breaking mocks/tests)
        # We'll conservatively return only the known safe keys plus any keys that
        # start with 'x_' (user custom) to keep PlanningAgent flexibility.
        filtered = {
            k: v
            for k, v in out.items()
            if k in cls._OPENAI_KEYS or k.startswith("x_") or k in ("stream_delta",)
        }
        # If keys were all filtered out (e.g., custom usage), return original
        return filtered or out

    @classmethod
    def _map_gemini(cls, out: Dict[str, Any]) -> Dict[str, Any]:
        # Convert stop -> stop_sequences
        if "stop_sequences" not in out and "stop" in out:
            stop = out.pop("stop")
            if isinstance(stop, str):
                out["stop_sequences"] = [stop]
            else:
                out["stop_sequences"] = stop

        # Convert priority -> service_tier
        # priority: high -> service_tier: priority (priority processing)
        # priority: normal -> no service_tier set (default processing)
        # priority: low -> service_tier: flex (slow/flex processing)
        if "service_tier" not in out and "priority" in out:
            val = out.pop("priority")
            if val == "high":
                out["service_tier"] = "priority"
            elif val == "low":
                out["service_tier"] = "flex"
            # For "normal", we don't set service_tier (None = default)

        # Convert reasoning -> thinking_level/thinking_budget
        if "reasoning" in out and "thinking_budget" not in out:
            val = out.pop("reasoning")
            if isinstance(val, int):
                out["thinking_budget"] = val
            elif isinstance(val, str):
                out["thinking_level"] = val.lower()
            elif isinstance(val, dict):
                # support {level: str} or {effort: str} or {budget: int}
                if "budget" in val and val["budget"] is not None:
                    try:
                        out["thinking_budget"] = int(val["budget"])  # type: ignore[arg-type]
                    except Exception:
                        pass
                else:
                    level = val.get("level") or val.get("effort")
                    if level is not None:
                        out["thinking_level"] = str(level).lower()

        # Remove OpenAI-only keys
        for k in [
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "reasoning_effort",
        ]:
            out.pop(k, None)

        # Filter to allowed keys (keep x_* custom as above)
        filtered = {
            k: v
            for k, v in out.items()
            if k in cls._GEMINI_KEYS or k.startswith("x_") or k in ("stream_delta",)
        }
        return filtered or out

    @classmethod
    def _map_anthropic(cls, out: Dict[str, Any]) -> Dict[str, Any]:
        # Convert max_output_tokens (normalized in map()) -> max_tokens
        if "max_tokens" not in out and "max_output_tokens" in out:
            out["max_tokens"] = out.pop("max_output_tokens")

        # Convert stop -> stop_sequences
        if "stop_sequences" not in out and "stop" in out:
            stop = out.pop("stop")
            if isinstance(stop, str):
                out["stop_sequences"] = [stop]
            else:
                out["stop_sequences"] = stop

        # Convert priority -> service_tier
        # priority: high -> service_tier: auto (use priority capacity if available)
        # priority: normal/low -> no service_tier set (standard processing)
        if "service_tier" not in out and "priority" in out:
            val = out.pop("priority")
            if val == "high":
                out["service_tier"] = "auto"

        # Convert reasoning / reasoning_effort -> effort
        if "effort" not in out and "reasoning" in out:
            val = out.pop("reasoning")
            if isinstance(val, dict):
                effort = val.get("effort") or val.get("level")
                if effort is not None:
                    out["effort"] = str(effort).lower()
            elif isinstance(val, str):
                out["effort"] = val.lower()
        if "effort" not in out and "reasoning_effort" in out:
            out["effort"] = str(out.pop("reasoning_effort")).lower()

        # Remove OpenAI/Gemini/Ollama-only keys
        for k in [
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
            "reasoning_effort",
            "candidate_count",
            "thinking_budget",
            "thinking_level",
            "response_mime_type",
            "response_schema",
            "system_instruction",
            "num_ctx",
            "num_predict",
            "repeat_penalty",
            "format",
        ]:
            out.pop(k, None)

        # Filter to allowed keys (keep x_* custom as above)
        filtered = {
            k: v
            for k, v in out.items()
            if k in cls._ANTHROPIC_KEYS or k.startswith("x_") or k in ("stream_delta",)
        }
        return filtered or out

    @classmethod
    def _map_ollama(cls, out: Dict[str, Any]) -> Dict[str, Any]:
        # Convert stop_sequences -> stop
        if "stop" not in out and "stop_sequences" in out:
            out["stop"] = out.pop("stop_sequences")

        # Convert max_output_tokens -> num_predict
        if "num_predict" not in out and "max_output_tokens" in out:
            out["num_predict"] = out.pop("max_output_tokens")

        # Remove OpenAI/Gemini-specific keys
        for k in [
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "reasoning_effort",
            "thinking_budget",
            "thinking_level",
            "response_mime_type",
            "response_schema",
            "system_instruction",
            "candidate_count",
        ]:
            out.pop(k, None)

        # Filter to allowed keys (keep x_* custom as above)
        filtered = {
            k: v
            for k, v in out.items()
            if k in cls._OLLAMA_KEYS or k.startswith("x_") or k in ("stream_delta",)
        }
        return filtered or out
