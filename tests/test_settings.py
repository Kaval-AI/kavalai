"""``kavalai.settings`` — the environment readers the entry points share."""

import pytest

from kavalai.normalizer import (
    Normalizer,
    get_default_normalizer,
    set_default_normalizer,
)
from kavalai.settings import (
    LLM_PARAMETER_VARIABLES,
    apply_normalizer_from_env,
    llm_parameters_from_env,
)
from kavalai.workflow.clients import build_parameters


def test_every_llm_variable_maps_onto_a_client_parameter():
    """A ``KAVALAI_LLM_*`` name the parameters model cannot carry is a lie."""
    fields = set(build_parameters({}).model_fields)
    assert {key for key, _ in LLM_PARAMETER_VARIABLES.values()} <= fields


def test_unset_and_empty_variables_are_left_out():
    assert llm_parameters_from_env({}) == {}
    assert llm_parameters_from_env({"KAVALAI_LLM_TEMPERATURE": ""}) == {}


def test_values_are_parsed_to_the_parameter_type():
    parameters = llm_parameters_from_env(
        {
            "KAVALAI_LLM_TEMPERATURE": "0.2",
            "KAVALAI_LLM_TOP_P": "0.9",
            "KAVALAI_LLM_REASONING_EFFORT": "low",
            "KAVALAI_LLM_SERVICE_TIER": "flex",
            "KAVALAI_LLM_TIMEOUT_SECONDS": "45",
            "KAVALAI_LLM_STREAM_TIMEOUT_SECONDS": "90",
        }
    )
    assert parameters == {
        "temperature": 0.2,
        "top_p": 0.9,
        "reasoning_effort": "low",
        "service_tier": "flex",
        "timeout_seconds": 45.0,
        "stream_timeout_seconds": 90.0,
    }
    assert build_parameters(parameters).temperature == 0.2


def test_a_value_that_does_not_parse_fails_at_start_up():
    with pytest.raises(
        ValueError, match="KAVALAI_LLM_TEMPERATURE='warm' is not a float"
    ):
        llm_parameters_from_env({"KAVALAI_LLM_TEMPERATURE": "warm"})


def test_the_process_environment_is_the_default(monkeypatch):
    monkeypatch.setenv("KAVALAI_LLM_TOP_P", "0.5")
    assert llm_parameters_from_env()["top_p"] == 0.5


@pytest.fixture
def clean_default_normalizer():
    import kavalai.normalizer

    original = kavalai.normalizer._default_normalizer
    set_default_normalizer(None)
    yield
    kavalai.normalizer._default_normalizer = original


def test_normalizer_unset_leaves_the_default(clean_default_normalizer):
    assert apply_normalizer_from_env({}) is None
    assert get_default_normalizer().l2 is True


def test_normalizer_is_installed_from_the_named_file(
    tmp_path, clean_default_normalizer
):
    path = tmp_path / "normalizer.yaml"
    Normalizer(l1=True, l2=False, center=True, center_vector=[0.1, 0.2]).save_to_yaml(
        str(path)
    )
    installed = apply_normalizer_from_env(
        {"KAVALAI_EMBEDDING_NORMALIZER_YAML": str(path)}
    )
    assert installed is get_default_normalizer()
    assert installed.l1 is True
    assert installed.center_enabled is True


def test_normalizer_from_the_process_environment(
    tmp_path, monkeypatch, clean_default_normalizer
):
    path = tmp_path / "normalizer.yaml"
    Normalizer(l2=True).save_to_yaml(str(path))
    monkeypatch.setenv("KAVALAI_EMBEDDING_NORMALIZER_YAML", str(path))
    assert apply_normalizer_from_env() is get_default_normalizer()
