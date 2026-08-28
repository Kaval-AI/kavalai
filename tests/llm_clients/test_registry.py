"""Tests for the LLM / embedding / RAG backend registries."""

import sys

import pytest

from kavalai.llm_clients import registry
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    LlmClientParameters,
    ModelStatsLogger,
)
from kavalai.llm_clients.embeddings import BaseEmbeddingClient, make_embedding_client
from kavalai.workflow.clients import make_client


class FakeLlmClient(BaseLlmClient):
    """Records what the registry handed it."""

    provider = "fake"

    def __init__(
        self,
        model,
        llm_client_parameters=None,
        model_stats_receiver=None,
        base_url=None,
    ):
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model
        self.base_url = base_url

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        raise NotImplementedError


class OddConstructorClient(FakeLlmClient):
    """A client whose ``__init__`` does not take the standard shape."""

    def __init__(self, everything):
        super().__init__(everything)
        self.everything = everything

    @classmethod
    def from_model(cls, model, parameters=None, stats_receiver=None, **defaults):
        return cls(f"{model}!")


class FakeEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model, host=None):
        super().__init__(model)
        self.host = host


@pytest.fixture
def clean_registry():
    """Undo every registration a test makes, whatever it registers."""
    before = {
        reg: set(reg.names())
        for reg in (
            registry.llm_providers,
            registry.embedding_providers,
            registry.rag_services,
        )
    }
    yield
    for reg, names in before.items():
        for name in set(reg.names()) - names:
            reg.unregister(name)


def test_register_a_class(clean_registry):
    registry.register_llm_provider("fake", FakeLlmClient)

    client = make_client("fake/model-x")

    assert isinstance(client, FakeLlmClient)
    assert client.model == "model-x"


def test_register_a_dotted_path(clean_registry):
    registry.register_llm_provider(
        "dotted", "tests.llm_clients.test_registry.FakeLlmClient"
    )

    client = make_client("dotted/model-x")

    assert isinstance(client, FakeLlmClient)
    assert client.model == "model-x"


def test_class_and_dotted_path_produce_the_same_client(clean_registry):
    registry.register_llm_provider("a", FakeLlmClient)
    registry.register_llm_provider("b", "tests.llm_clients.test_registry.FakeLlmClient")

    assert type(make_client("a/m")) is type(make_client("b/m"))


def test_register_a_callable(clean_registry):
    registry.register_llm_provider(
        "callable", lambda model, params, stats: FakeLlmClient(f"via-callable/{model}")
    )

    assert make_client("callable/m").model == "via-callable/m"


def test_from_model_override_is_honoured(clean_registry):
    registry.register_llm_provider("odd", OddConstructorClient)

    assert make_client("odd/m").everything == "m!"


def test_registration_defaults_reach_the_constructor(clean_registry):
    registry.register_llm_provider(
        "eu", FakeLlmClient, base_url="https://eu.example/v1"
    )

    assert make_client("eu/m").base_url == "https://eu.example/v1"


def test_caller_kwargs_override_registration_defaults(clean_registry):
    registry.register_llm_provider("eu", FakeLlmClient, base_url="https://eu/v1")

    client = make_client("eu/m", base_url="https://override/v1")

    assert client.base_url == "https://override/v1"


def test_parameters_and_stats_receiver_are_forwarded(clean_registry):
    registry.register_llm_provider("fake", FakeLlmClient)
    parameters = LlmClientParameters(temperature=0.25)
    receiver = ModelStatsLogger()

    client = make_client("fake/m", parameters, receiver)

    assert client.parameters.temperature == 0.25
    assert client.model_stats_receiver is receiver


def test_exact_model_key_wins_over_the_provider_key(clean_registry):
    registry.register_llm_provider("acme", FakeLlmClient)
    registry.register_llm_provider(
        "acme/special", lambda model, params, stats: FakeLlmClient("pinned")
    )

    assert make_client("acme/special").model == "pinned"
    assert make_client("acme/ordinary").model == "ordinary"


def test_model_name_keeps_slashes_after_the_provider(clean_registry):
    registry.register_embedding_provider("fake", FakeEmbeddingClient)

    client = make_embedding_client("fake/BAAI/bge-small-en-v1.5")

    assert client.model == "BAAI/bge-small-en-v1.5"


def test_unknown_name_lists_what_was_tried_and_what_exists():
    with pytest.raises(registry.RegistryError) as error:
        make_client("nosuch/model")

    message = str(error.value)
    assert "'nosuch/model' and 'nosuch'" in message
    assert "openai" in message
    assert "register_llm_provider()" in message


def test_registry_error_is_a_value_error():
    # make_client raised ValueError for a bad name before the registries
    # existed; callers catch that.
    with pytest.raises(ValueError):
        make_client("nosuch/model")


def test_duplicate_registration_raises(clean_registry):
    registry.register_llm_provider("dup", FakeLlmClient)

    with pytest.raises(registry.RegistryError, match="already registered"):
        registry.register_llm_provider("dup", FakeLlmClient)


def test_replace_overrides_and_warns(clean_registry, caplog):
    registry.register_llm_provider("openai", FakeLlmClient, replace=True)

    assert isinstance(make_client("openai/gpt-4o"), FakeLlmClient)
    assert any(
        "openai" in record.message and "re-registered" in record.message
        for record in caplog.records
    ), caplog.records


def test_unregister_restores_the_builtin(clean_registry):
    registry.register_llm_provider("openai", FakeLlmClient, replace=True)
    registry.llm_providers.unregister("openai")
    registry.llm_providers.register(
        "openai", "kavalai.llm_clients.openai_client.OpenAIClient"
    )

    assert "openai" in registry.registered_llm_providers()


def test_unregister_ignores_unknown_names():
    registry.llm_providers.unregister("never-registered")


@pytest.mark.parametrize("name", ["", "/leading", "trailing/"])
def test_invalid_names_are_rejected(name, clean_registry):
    with pytest.raises(registry.RegistryError):
        registry.register_llm_provider(name, FakeLlmClient)


def test_bad_registration_default_names_the_registration(clean_registry):
    registry.register_llm_provider("typo", FakeLlmClient, base_yrl="oops")

    with pytest.raises(registry.RegistryError) as error:
        make_client("typo/m")

    assert "typo" in str(error.value)
    assert "base_yrl" in str(error.value)


def test_listing_builtins_imports_no_sdk():
    """``import kavalai`` must work where no provider SDK is installed."""
    names = (
        registry.registered_llm_providers()
        + registry.registered_embedding_providers()
        + registry.registered_rag_services()
    )

    assert "openai" in names
    # The registry module itself must not have dragged any SDK in. Other tests
    # in the session will have imported them, so check the registry's own view.
    assert registry.llm_providers._resolved.keys() <= {"openai", "anthropic"}


def test_builtin_targets_are_dotted_strings_not_classes():
    # A class here would mean an eager import at module scope.
    for target in registry._BUILTIN_LLM_PROVIDERS.values():
        assert isinstance(target, str)
    for target in registry._BUILTIN_EMBEDDING_PROVIDERS.values():
        assert isinstance(target, str)
    for target in registry._BUILTIN_RAG_SERVICES.values():
        assert isinstance(target, str)


def test_lazy_clients_table_is_derived_from_the_registry():
    import kavalai

    for class_name, (module_path, _) in kavalai._LAZY_CLIENTS.items():
        assert f"{module_path}.{class_name}" in (
            registry._BUILTIN_LLM_PROVIDERS.values()
        )


def test_verify_raises_on_a_bad_dotted_path(clean_registry):
    registry.register_llm_provider("broken", "no.such.module.Class")

    with pytest.raises(registry.RegistryError) as error:
        registry.verify_registrations()

    assert "broken" in str(error.value)
    assert "no.such.module" in str(error.value)


def test_verify_reports_a_missing_attribute(clean_registry):
    registry.register_llm_provider(
        "missing", "tests.llm_clients.test_registry.NoSuchClass"
    )

    with pytest.raises(registry.RegistryError, match="has no attribute"):
        registry.verify_registrations()


def test_verify_rejects_a_path_without_a_module(clean_registry):
    registry.register_llm_provider("flat", "JustAName")

    with pytest.raises(registry.RegistryError, match="not a dotted path"):
        registry.verify_registrations()


def test_verify_skips_builtins(clean_registry, monkeypatch):
    """An install with one SDK must not fail start-up over another's."""
    monkeypatch.setitem(sys.modules, "openai", None)
    registry.verify_registrations()


def test_validate_imports_immediately(clean_registry):
    with pytest.raises(registry.RegistryError):
        registry.register_llm_provider("eager", "no.such.module.Class", validate=True)


def test_embedding_provider_registration(clean_registry):
    registry.register_embedding_provider("fake", FakeEmbeddingClient, host="h")

    client = make_embedding_client("fake/m")

    assert isinstance(client, FakeEmbeddingClient)
    assert (client.model, client.host) == ("m", "h")


def test_rag_service_registration_binds_everything_as_defaults(clean_registry):
    class FakeRagService:
        def __init__(self, filename=None, model=None):
            self.filename = filename
            self.model = model

    registry.register_rag_service(
        "docs", FakeRagService, filename="x.db", model="fastembed/m"
    )

    service = registry.rag_services.build("docs")

    assert (service.filename, service.model) == ("x.db", "fastembed/m")


def test_registries_are_independent(clean_registry):
    registry.register_llm_provider("shared", FakeLlmClient)
    registry.register_embedding_provider("shared", FakeEmbeddingClient)

    assert "shared" in registry.registered_llm_providers()
    assert "shared" in registry.registered_embedding_providers()
    assert "shared" not in registry.registered_rag_services()
