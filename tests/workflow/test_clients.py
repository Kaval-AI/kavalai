import pytest

from kavalai.workflow import clients
from kavalai.workflow.models import WorkflowException
from kavalai.llm_clients.anthropic_client import AnthropicClient
from kavalai.llm_clients.base_client import LlmClientParameters
from kavalai.llm_clients.browser_client import BrowserLLMClient
from kavalai.llm_clients.gemini_client import GeminiClient
from kavalai.llm_clients.ollama_client import OllamaClient
from kavalai.llm_clients.openai_client import OpenAIClient


def test_build_parameters_maps_every_known_key():
    params = clients.build_parameters(
        {"temperature": 0.3, "top_p": 0.9, "max_output_tokens": 512}
    )
    assert isinstance(params, LlmClientParameters)
    assert params.temperature == 0.3
    assert params.top_p == 0.9
    assert params.max_output_tokens == 512


@pytest.mark.parametrize(
    "key", ["max_tokens", "max_completion_tokens", "num_predict", "max_new_tokens"]
)
def test_a_provider_name_for_the_cap_points_at_max_output_tokens(key):
    with pytest.raises(WorkflowException) as caught:
        clients.build_parameters({key: 256})

    assert str(caught.value).startswith(
        f"Unknown llm_kwargs key '{key}' (did you mean 'max_output_tokens'?). "
        "Valid keys: max_output_tokens, reasoning_effort, service_tier, "
        "stream_timeout_seconds, temperature, timeout_seconds, top_p."
    )


def test_a_misspelled_key_suggests_the_closest_parameter():
    with pytest.raises(
        WorkflowException, match="'temprature' \\(did you mean 'temperature'\\?\\)"
    ):
        clients.build_parameters({"temprature": 0.2})


def test_unknown_keys_are_all_named_and_unrelated_ones_get_no_hint():
    with pytest.raises(WorkflowException) as caught:
        clients.build_parameters({"seed": 7, "max_tokens": 9, "temperature": 0})

    assert str(caught.value).startswith(
        "Unknown llm_kwargs keys 'max_tokens' (did you mean "
        "'max_output_tokens'?), 'seed'. Valid keys: "
    )


def test_build_parameters_none():
    params = clients.build_parameters(None)
    assert isinstance(params, LlmClientParameters)


def test_make_client_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = clients.make_client("openai/gpt-4o")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"


def test_make_client_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = clients.make_client("gemini/gemini-2.0", clients.build_parameters({}))
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.0"


def test_make_client_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = clients.make_client("anthropic/claude-opus-5")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-opus-5"


def test_make_client_ollama():
    client = clients.make_client("ollama/llama3")
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3"


def test_make_client_browser():
    client = clients.make_client("browser/Llama-3.2-1B-Instruct-q4f32_1-MLC")
    assert isinstance(client, BrowserLLMClient)
    assert client.model == "Llama-3.2-1B-Instruct-q4f32_1-MLC"


def test_make_client_requires_provider():
    with pytest.raises(ValueError, match="provider/model"):
        clients.make_client("gpt-4o")


def test_make_client_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        clients.make_client("mistral/mistral-large")
