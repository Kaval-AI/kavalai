from kavalai.llm_clients.kwargs_mapper import LLMKWargsMapper


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


def test_openai_reasoning_model_strips_sampling_params():
    # GPT-5 family and o-series reject top_p/temperature on the Responses API.
    for model in ("gpt-5.5", "gpt-5", "gpt-5-mini", "o1", "o3-mini", "o4-mini"):
        kwargs = {
            "top_p": 0.2,
            "temperature": 0.7,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "logit_bias": {"50256": -100},
            "reasoning": "high",
            "max_tokens": 100,
        }
        mapped = LLMKWargsMapper.map("openai", model, kwargs)

        assert "top_p" not in mapped
        assert "temperature" not in mapped
        assert "presence_penalty" not in mapped
        assert "frequency_penalty" not in mapped
        assert "logit_bias" not in mapped
        # Reasoning/other params still map through.
        assert mapped.get("reasoning_effort") == "high"
        assert mapped.get("max_output_tokens") == 100


def test_openai_non_reasoning_model_keeps_sampling_params():
    # Standard chat models (e.g. gpt-4o) still accept top_p/temperature.
    kwargs = {"top_p": 0.2, "temperature": 0.7}
    mapped = LLMKWargsMapper.map("openai", "gpt-4o", kwargs)
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
