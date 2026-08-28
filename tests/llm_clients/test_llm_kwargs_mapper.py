from kavalai.llm_clients.kwargs_mapper import (
    LLMKWargsMapper,
)


def test_openai_reasoning_and_stops_and_max_tokens_mapping():
    kwargs = {
        "reasoning": "high",
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "max_tokens": 100,
    }
    mapped = LLMKWargsMapper.map("openai", "gpt-4o", kwargs)

    assert mapped.get("reasoning_effort") == "high"
    assert mapped.get("temperature") == 0.2
    assert mapped.get("stop") == ["END"]
    assert mapped.get("max_output_tokens") == 100
    assert "stop_sequences" not in mapped
    assert "max_tokens" not in mapped


def test_openai_keeps_sampling_params_for_every_model():
    # No model list decides what is sent: a model that rejects a parameter
    # fails the call with the provider's error instead of having it dropped.
    kwargs = {"top_p": 0.2, "temperature": 0.7}
    mapped = LLMKWargsMapper.map("openai", "gpt-5.5", kwargs)
    assert mapped.get("top_p") == 0.2
    assert mapped.get("temperature") == 0.7


def test_gemini_reasoning_level_and_stops_and_max_tokens_mapping():
    kwargs = {
        "reasoning": "medium",
        "temperature": 0.7,
        "stop": "#END",
        "max_tokens": 64,
        "presence_penalty": 0.1,  # should be stripped for gemini
    }
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash", kwargs)

    assert mapped.get("thinking_level") == "medium"
    assert mapped.get("temperature") == 0.7
    assert mapped.get("stop_sequences") == ["#END"]
    assert mapped.get("max_output_tokens") == 64
    assert "presence_penalty" not in mapped


def test_gemini_reasoning_budget_mapping():
    kwargs = {"reasoning": 300}
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash-thinking", kwargs)
    assert mapped.get("thinking_budget") == 300


def test_anthropic_reasoning_and_stops_and_max_tokens_mapping():
    kwargs = {
        "reasoning": "high",
        "temperature": 0.2,
        "stop": "#END",
        "max_tokens": 100,
        "presence_penalty": 0.1,  # should be stripped for anthropic
    }
    mapped = LLMKWargsMapper.map("anthropic", "claude-haiku-4-5", kwargs)

    assert mapped.get("effort") == "high"
    assert mapped.get("temperature") == 0.2
    assert mapped.get("stop_sequences") == ["#END"]
    assert mapped.get("max_tokens") == 100
    assert "presence_penalty" not in mapped
    assert "stop" not in mapped


def test_anthropic_stop_list_mapping():
    mapped = LLMKWargsMapper.map(
        "anthropic", "claude-haiku-4-5", {"stop": ["END", "STOP"]}
    )
    assert mapped.get("stop_sequences") == ["END", "STOP"]


def test_anthropic_reasoning_dict_and_effort_synonyms():
    mapped = LLMKWargsMapper.map(
        "anthropic", "claude-haiku-4-5", {"reasoning": {"effort": "LOW"}}
    )
    assert mapped.get("effort") == "low"

    mapped = LLMKWargsMapper.map(
        "anthropic", "claude-haiku-4-5", {"reasoning_effort": "Medium"}
    )
    assert mapped.get("effort") == "medium"


def test_anthropic_keeps_sampling_params():
    # Sampling params pass through untouched; models that reject them will
    # fail loudly at the API rather than have them silently dropped.
    kwargs = {"top_p": 0.2, "temperature": 0.7, "top_k": 40}
    mapped = LLMKWargsMapper.map("anthropic", "claude-opus-5", kwargs)
    assert mapped.get("top_p") == 0.2
    assert mapped.get("temperature") == 0.7
    assert mapped.get("top_k") == 40


def test_passthrough_existing_specific_keys():
    # If user already provided provider-specific key, do not override
    oa_kwargs = {"reasoning_effort": "low"}
    oa_mapped = LLMKWargsMapper.map("openai", "gpt-4o", oa_kwargs)
    assert oa_mapped.get("reasoning_effort") == "low"

    ge_kwargs = {"thinking_budget": 500}
    ge_mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash-thinking", ge_kwargs)
    assert ge_mapped.get("thinking_budget") == 500


def test_pop_any_returns_the_first_present_key():
    d = {"b": 2, "c": 3}
    assert LLMKWargsMapper._pop_any(d, ["a", "b", "c"]) == 2
    assert d == {"c": 3}
    # Nothing matched: the dict is untouched.
    assert LLMKWargsMapper._pop_any(d, ["x", "y"]) is None
    assert d == {"c": 3}


def test_unknown_provider_passes_kwargs_through_unchanged():
    kwargs = {"whatever": 1, "max_tokens": 5}
    mapped = LLMKWargsMapper.map("mystery", "some-model", kwargs)
    # max_tokens is still normalised, everything else is left alone.
    assert mapped == {"whatever": 1, "max_output_tokens": 5}


def test_openai_reasoning_dict_uses_effort_or_level():
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"reasoning": {"effort": "HIGH"}}).get(
            "reasoning_effort"
        )
        == "high"
    )
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"reasoning": {"level": "Low"}}).get(
            "reasoning_effort"
        )
        == "low"
    )
    # A dict carrying neither key contributes nothing.
    assert "reasoning_effort" not in LLMKWargsMapper.map(
        "openai", "gpt-4o", {"reasoning": {"budget": 100}}
    )


def test_gemini_stop_string_becomes_a_list():
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"stop": "END"})
    assert mapped["stop_sequences"] == ["END"]


def test_gemini_stop_list_is_passed_through():
    mapped = LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"stop": ["A", "B"]})
    assert mapped["stop_sequences"] == ["A", "B"]


def test_gemini_reasoning_dict_variants():
    budget = LLMKWargsMapper.map(
        "gemini", "gemini-2.0-flash", {"reasoning": {"budget": "512"}}
    )
    assert budget["thinking_budget"] == 512

    level = LLMKWargsMapper.map(
        "gemini", "gemini-2.0-flash", {"reasoning": {"effort": "HIGH"}}
    )
    assert level["thinking_level"] == "high"

    # An unusable budget is dropped rather than raising.
    bad = LLMKWargsMapper.map(
        "gemini", "gemini-2.0-flash", {"reasoning": {"budget": "not-a-number"}}
    )
    assert "thinking_budget" not in bad


def test_ollama_mapping_renames_and_filters_keys():
    mapped = LLMKWargsMapper.map(
        "ollama",
        "llama3",
        {
            "stop_sequences": ["END"],
            "max_tokens": 128,
            "temperature": 0.5,
            "reasoning_effort": "high",  # OpenAI-only, dropped
            "x_custom": "kept",
        },
    )

    assert mapped["stop"] == ["END"]
    assert mapped["num_predict"] == 128
    assert mapped["temperature"] == 0.5
    assert mapped["x_custom"] == "kept"
    assert "reasoning_effort" not in mapped


def test_ollama_mapping_keeps_originals_when_nothing_is_recognised():
    # The filter must not empty the kwargs entirely.
    mapped = LLMKWargsMapper.map("ollama", "llama3", {"unknown_key": 1})
    assert mapped == {"unknown_key": 1}


def test_priority_mapping():
    # OpenAI
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "high"}).get(
            "service_tier"
        )
        == "priority"
    )
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "low"}).get("service_tier")
        == "flex"
    )
    assert (
        LLMKWargsMapper.map("openai", "gpt-4o", {"priority": "normal"}).get(
            "service_tier"
        )
        == "default"
    )

    # Anthropic
    assert (
        LLMKWargsMapper.map("anthropic", "claude-haiku-4-5", {"priority": "high"}).get(
            "service_tier"
        )
        == "auto"
    )
    assert (
        LLMKWargsMapper.map("anthropic", "claude-haiku-4-5", {"priority": "low"}).get(
            "service_tier"
        )
        is None
    )  # standard processing (no service_tier set)
    assert (
        LLMKWargsMapper.map(
            "anthropic", "claude-haiku-4-5", {"priority": "normal"}
        ).get("service_tier")
        is None
    )

    # Gemini
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "high"}).get(
            "service_tier"
        )
        == "priority"
    )
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "low"}).get(
            "service_tier"
        )
        == "flex"
    )
    assert (
        LLMKWargsMapper.map("gemini", "gemini-2.0-flash", {"priority": "normal"}).get(
            "service_tier"
        )
        is None
    )  # default for gemini is None
