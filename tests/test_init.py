"""Tests for the lazy provider-client imports exposed by ``kavalai``."""

import pytest

import kavalai
from kavalai.llm_clients.anthropic_client import AnthropicClient


def test_lazy_client_resolves_to_the_real_class():
    assert kavalai.AnthropicClient is AnthropicClient


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchThing'"):
        kavalai.NoSuchThing


def test_missing_optional_dependency_names_the_package_and_the_extra(monkeypatch):
    # Simulate a provider whose SDK (and therefore whose client module) is absent.
    monkeypatch.setitem(
        kavalai._LAZY_CLIENTS,
        "GhostClient",
        ("kavalai.llm_clients.no_such_client", "ghost"),
    )
    with pytest.raises(
        ImportError, match=r"'ghost' package.+pip install kavalai\[common\]"
    ):
        kavalai.GhostClient


def test_dir_lists_the_public_api():
    listed = dir(kavalai)
    assert listed == sorted(listed)
    assert "WorkflowEngine" in listed
