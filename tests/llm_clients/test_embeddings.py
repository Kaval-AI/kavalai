import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kavalai.db import ModelCallStat
from kavalai.llm_clients import browser_client
from kavalai.llm_clients import embeddings as emb
from kavalai.llm_clients.base_client import LlmClientException
from kavalai.llm_clients.embeddings import (
    BaseEmbeddingClient,
    BrowserEmbeddingClient,
    FastEmbedClient,
    GeminiEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAIEmbeddingClient,
    make_embedding_client,
)
from kavalai.normalizer import Normalizer


# ------------------------------------------------------------------- factory
def test_make_embedding_client_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(
        make_embedding_client("openai/text-embedding-3-small"), OpenAIEmbeddingClient
    )
    assert isinstance(
        make_embedding_client("gemini/text-embedding-004"), GeminiEmbeddingClient
    )
    assert isinstance(
        make_embedding_client("ollama/nomic-embed-text"), OllamaEmbeddingClient
    )
    fc = make_embedding_client("fastembed/BAAI/bge-small-en-v1.5")
    assert isinstance(fc, FastEmbedClient)
    # The provider is split off but the rest of the path is kept intact.
    assert fc.model == "BAAI/bge-small-en-v1.5"
    bc = make_embedding_client("browser/snowflake-arctic-embed-m-q0f32-MLC-b4")
    assert isinstance(bc, BrowserEmbeddingClient)
    assert bc.model == "snowflake-arctic-embed-m-q0f32-MLC-b4"


def test_make_embedding_client_errors():
    with pytest.raises(ValueError, match="provider/model"):
        make_embedding_client("no-slash")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        make_embedding_client("anthropic/x")


def test_base_client_not_implemented():
    import asyncio

    with pytest.raises(NotImplementedError):
        asyncio.run(BaseEmbeddingClient("m").compute_embeddings(["a"]))


# ---------------------------------------------------------------- providers
async def test_openai_embeddings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = OpenAIEmbeddingClient("text-embedding-3-small")
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(total_tokens=7),
        model_dump=lambda: {"ok": True},
    )
    client.client = MagicMock()
    client.client.embeddings.create = AsyncMock(return_value=response)

    vectors, stats = await client.compute_embeddings(["hello"])
    assert vectors == [[0.1, 0.2]]
    assert isinstance(stats, ModelCallStat)
    assert stats.call_type == "embedding"
    assert stats.model == "openai/text-embedding-3-small"
    assert stats.total_tokens == 7


def test_maybe_normalize_passes_through_when_disabled():
    vectors = [[3.0, 4.0]]
    assert emb._maybe_normalize(vectors, normalize=False, normalizer=None) is vectors


def test_maybe_normalize_falls_back_to_the_default_normalizer():
    # No normalizer supplied: the process-wide default is used.
    result = emb._maybe_normalize([[3.0, 4.0]], normalize=True, normalizer=None)
    assert len(result[0]) == 2


async def test_openai_embeddings_normalized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = OpenAIEmbeddingClient("text-embedding-3-small")
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[3.0, 4.0])],
        usage=SimpleNamespace(total_tokens=1),
        model_dump=lambda: {},
    )
    client.client = MagicMock()
    client.client.embeddings.create = AsyncMock(return_value=response)

    vectors, _ = await client.compute_embeddings(
        ["x"], normalize=True, normalizer=Normalizer(l2=True)
    )
    # L2-normalised [3,4] -> [0.6, 0.8]
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)


async def test_gemini_embeddings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = GeminiEmbeddingClient("text-embedding-004")
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.6])])
    client.client = MagicMock()
    client.client.aio.models.embed_content = AsyncMock(return_value=response)
    # Avoid importing the real google types in compute_embeddings.
    monkeypatch.setattr(
        "google.genai.types.EmbedContentConfig", lambda **kw: kw, raising=False
    )

    vectors, stats = await client.compute_embeddings(["hi"])
    assert vectors == [[0.5, 0.6]]
    assert stats.model == "gemini/text-embedding-004"


async def test_ollama_embeddings():
    client = OllamaEmbeddingClient("nomic-embed-text")
    client.client = MagicMock()
    client.client.embed = AsyncMock(
        return_value={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
    )
    vectors, stats = await client.compute_embeddings(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]  # one per input text
    assert stats.total_tokens == 8
    assert stats.model == "ollama/nomic-embed-text"


async def test_fastembed_embeddings(monkeypatch):
    client = FastEmbedClient("BAAI/bge-small-en-v1.5")

    class _Vec:
        def tolist(self):
            return [0.9, 0.1]

    fake_model = MagicMock()
    fake_model.embed = lambda texts, **kw: [_Vec() for _ in texts]
    monkeypatch.setattr(client, "_get_model", lambda: fake_model)

    vectors, stats = await client.compute_embeddings(["a"])
    assert vectors == [[0.9, 0.1]]
    assert stats.call_type == "embedding"
    assert stats.model == "fastembed/BAAI/bge-small-en-v1.5"


def test_fastembed_get_model_lazy(monkeypatch):
    # _get_model builds a TextEmbedding lazily and caches it.
    created = []

    class _Fake:
        def __init__(self, **kw):
            created.append(kw)

    monkeypatch.setattr(emb, "FastEmbedClient", FastEmbedClient)
    monkeypatch.setitem(
        __import__("sys").modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=_Fake),
    )
    client = FastEmbedClient("m", cache_dir="/tmp", threads=2)
    first = client._get_model()
    second = client._get_model()
    assert first is second
    assert created and created[0]["model_name"] == "m"


# ------------------------------------------------------------- browser bridge
class FakeEmbedBridge:
    """Stand-in for ``window.kavalBrowserLLM`` exposing an ``embed`` function."""

    def __init__(self, result=None, raise_exc=None):
        self._result = result
        self._raise_exc = raise_exc
        self.last_request = None

    async def embed(self, request_json):
        self.last_request = request_json
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


def _install_bridge(monkeypatch, bridge, *, pyodide=True):
    """Pretend we run under Pyodide and inject a fake ``js`` module.

    ``get_browser_bridge`` resolves ``is_pyodide`` and the ``js`` module in the
    ``browser_client`` namespace, so that is where we patch.
    """
    monkeypatch.setattr(browser_client, "is_pyodide", lambda: pyodide)
    fake_js = types.ModuleType("js")
    if bridge is not None:
        fake_js.kavalBrowserLLM = bridge
    monkeypatch.setitem(sys.modules, "js", fake_js)


async def test_browser_embeddings(monkeypatch):
    bridge = FakeEmbedBridge(
        result=json.dumps(
            {
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        )
    )
    _install_bridge(monkeypatch, bridge)

    client = BrowserEmbeddingClient("snowflake-arctic-embed-m-q0f32-MLC-b4")
    vectors, stats = await client.compute_embeddings(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert isinstance(stats, ModelCallStat)
    assert stats.call_type == "embedding"
    assert stats.model == "browser/snowflake-arctic-embed-m-q0f32-MLC-b4"
    assert stats.batch_size == 2
    assert stats.total_tokens == 5

    # The request handed to the bridge carries the model and the texts.
    sent = json.loads(bridge.last_request)
    assert sent == {
        "model": "snowflake-arctic-embed-m-q0f32-MLC-b4",
        "input": ["hello", "world"],
    }


async def test_browser_embeddings_normalized(monkeypatch):
    bridge = FakeEmbedBridge(
        result=json.dumps({"embeddings": [[3.0, 4.0]], "usage": {}})
    )
    _install_bridge(monkeypatch, bridge)

    client = BrowserEmbeddingClient("model-x")
    vectors, _ = await client.compute_embeddings(
        ["x"], normalize=True, normalizer=Normalizer(l2=True)
    )

    # L2-normalised [3,4] -> [0.6, 0.8]
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)


async def test_browser_embeddings_usage_falls_back_to_prompt_tokens(monkeypatch):
    # No total_tokens: the client falls back to prompt_tokens.
    bridge = FakeEmbedBridge(
        result=json.dumps({"embeddings": [[0.0]], "usage": {"prompt_tokens": 9}})
    )
    _install_bridge(monkeypatch, bridge)

    _, stats = await BrowserEmbeddingClient("m").compute_embeddings(["a"])
    assert stats.total_tokens == 9


async def test_browser_embeddings_missing_usage_defaults_to_zero(monkeypatch):
    bridge = FakeEmbedBridge(result=json.dumps({"embeddings": [[0.0]]}))
    _install_bridge(monkeypatch, bridge)

    _, stats = await BrowserEmbeddingClient("m").compute_embeddings(["a"])
    assert stats.total_tokens == 0


async def test_browser_embeddings_error_payload_raises(monkeypatch):
    bridge = FakeEmbedBridge(result=json.dumps({"error": "WebGPU not available"}))
    _install_bridge(monkeypatch, bridge)

    with pytest.raises(LlmClientException, match="WebGPU not available"):
        await BrowserEmbeddingClient("m").compute_embeddings(["a"])


async def test_browser_embeddings_bridge_exception_is_wrapped(monkeypatch):
    bridge = FakeEmbedBridge(raise_exc=ValueError("boom"))
    _install_bridge(monkeypatch, bridge)

    with pytest.raises(LlmClientException, match="In-browser embedding call failed"):
        await BrowserEmbeddingClient("m").compute_embeddings(["a"])


async def test_browser_embeddings_raises_outside_pyodide(monkeypatch):
    _install_bridge(monkeypatch, None, pyodide=False)

    with pytest.raises(LlmClientException, match="only works inside a Pyodide"):
        await BrowserEmbeddingClient("m").compute_embeddings(["a"])


async def test_browser_embeddings_raises_when_bridge_absent(monkeypatch):
    _install_bridge(monkeypatch, None, pyodide=True)

    with pytest.raises(LlmClientException, match="No in-browser LLM engine found"):
        await BrowserEmbeddingClient("m").compute_embeddings(["a"])
