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
"""

import json
import os
import time
from typing import List, Optional, Tuple

from kavalai.db import ModelCallStat
from kavalai.llm_clients.base_client import LlmClientException
from kavalai.llm_clients.common import create_model_call_stat, get_model_name
from kavalai.normalizer import Normalizer, get_default_normalizer

Embeddings = List[List[float]]


class BaseEmbeddingClient:
    """Common interface for v2 embedding clients.

    The model name is bound at construction (the factory splits the
    ``provider/model`` string), so ``compute_embeddings`` only takes the texts.
    Implementations return the embeddings plus a database-ready
    :class:`~kavalai.db.ModelCallStat` (the ORM row) so callers such as
    :class:`~kavalai.rag.postgres.PostgresRagService` can persist usage directly.
    """

    def __init__(self, model: str):
        self.model = model

    @classmethod
    def from_model(cls, model: str, **defaults) -> "BaseEmbeddingClient":
        """Construct this client for a resolved model name.

        The registry calls this rather than the constructor, because embedding
        clients do not share one: they variously take ``api_key``, ``host``,
        ``base_url``, ``cache_dir`` or nothing at all. ``defaults`` are the
        keyword arguments bound at registration.

        Args:
            model: Model name with the provider prefix already removed.
            **defaults: Extra constructor arguments bound at registration.
        """
        return cls(model, **defaults)

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        raise NotImplementedError("Subclasses must implement compute_embeddings.")


def _maybe_normalize(
    embeddings: Embeddings, normalize: bool, normalizer: Optional[Normalizer]
) -> Embeddings:
    if not normalize:
        return embeddings
    if normalizer is None:
        normalizer = get_default_normalizer()
    return normalizer.transform(embeddings)


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embeddings (e.g. ``text-embedding-3-small``)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        super().__init__(model)
        from openai import AsyncOpenAI

        self.timeout = timeout
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
        )

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        start_time = time.perf_counter()
        response = await self.client.embeddings.create(
            input=texts, model=self.model, timeout=self.timeout, **kwargs
        )
        duration = time.perf_counter() - start_time

        embeddings = _maybe_normalize(
            [data.embedding for data in response.data], normalize, normalizer
        )
        total_tokens = response.usage.total_tokens if response.usage else 0

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"openai/{self.model}",
            duration_seconds=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
            response_data=response.model_dump()
            if hasattr(response, "model_dump")
            else response,
        )
        return embeddings, stats


class GeminiEmbeddingClient(BaseEmbeddingClient):
    """Google Gemini embeddings."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        from google import genai

        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        from google.genai import types

        start_time = time.perf_counter()
        model_name = get_model_name(self.model)
        response = await self.client.aio.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(**kwargs),
        )
        duration = time.perf_counter() - start_time

        embeddings = _maybe_normalize(
            [embedding.values for embedding in response.embeddings],
            normalize,
            normalizer,
        )
        stats = create_model_call_stat(
            call_type="embedding",
            model=f"gemini/{model_name}",
            duration_seconds=duration,
            batch_size=len(texts),
            total_tokens=0,
        )
        return embeddings, stats


class OllamaEmbeddingClient(BaseEmbeddingClient):
    """Ollama (local) embeddings."""

    def __init__(self, model: str, host: Optional[str] = None, timeout: float = 30.0):
        super().__init__(model)
        import ollama

        self.client = ollama.AsyncClient(
            host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout=timeout,
        )

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        start_time = time.perf_counter()
        model_name = get_model_name(self.model)

        embeddings: Embeddings = []
        total_prompt_tokens = 0
        for text in texts:
            response = await self.client.embed(model=model_name, input=text, **kwargs)
            embeddings.extend(response.get("embeddings", []))
            total_prompt_tokens += response.get("prompt_eval_count", 0)

        embeddings = _maybe_normalize(embeddings, normalize, normalizer)
        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"ollama/{model_name}",
            duration_seconds=duration,
            batch_size=len(texts),
            total_tokens=total_prompt_tokens,
        )
        return embeddings, stats


class FastEmbedClient(BaseEmbeddingClient):
    """Local embeddings via FastEmbed / ONNX Runtime (no API key)."""

    def __init__(
        self,
        model: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.cache_dir = cache_dir
        self.threads = threads
        self.init_kwargs = kwargs
        self._embedding_model = None

    @classmethod
    def from_model(cls, model: str, **defaults) -> "FastEmbedClient":
        """Fall back to ``FASTEMBED_CACHE_DIR`` / ``FASTEMBED_THREADS``.

        The other embedding clients read their credential inside ``__init__``;
        FastEmbed needs no credential but does take two environment-tunable
        knobs, so the same fallback lives here instead.
        """
        threads = defaults.pop("threads", None) or os.getenv("FASTEMBED_THREADS")
        cache_dir = defaults.pop("cache_dir", None) or os.getenv("FASTEMBED_CACHE_DIR")
        return cls(
            model,
            cache_dir=cache_dir,
            threads=int(threads) if threads else None,
            **defaults,
        )

    def _get_model(self):
        if self._embedding_model is None:
            from fastembed import TextEmbedding

            self._embedding_model = TextEmbedding(
                model_name=self.model,
                cache_dir=self.cache_dir,
                threads=self.threads,
                **self.init_kwargs,
            )
        return self._embedding_model

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        start_time = time.perf_counter()
        embeddings = [e.tolist() for e in self._get_model().embed(texts, **kwargs)]
        embeddings = _maybe_normalize(embeddings, normalize, normalizer)
        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"fastembed/{self.model}",
            duration_seconds=duration,
            batch_size=len(texts),
            total_tokens=None,  # FastEmbed does not expose token counts.
        )
        return embeddings, stats


class BrowserEmbeddingClient(BaseEmbeddingClient):
    """In-browser embeddings via the WebLLM bridge (Pyodide only, no API key).

    Mirrors :class:`~kavalai.llm_clients.browser_client.BrowserLLMClient`:
    inference happens inside the page through ``window.kavalBrowserLLM``, here
    via its async ``embed`` function::

        window.kavalBrowserLLM.embed(requestJson) -> Promise<resultJson>

    where ``requestJson`` is a JSON string of ``{model, input}`` (``input`` is
    the list of texts) and ``resultJson`` is a JSON string of either
    ``{embeddings, usage}`` or ``{error}``. The model is downloaded once and
    cached by the browser — no API key, no provider account, no CORS.

    Use it through ``make_embedding_client("browser/<model-id>")``; ``<model-id>``
    is passed verbatim to the bridge (e.g. a WebLLM embedding id like
    ``snowflake-arctic-embed-m-q0f32-MLC-b4``).
    """

    async def compute_embeddings(
        self,
        texts: List[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> Tuple[Embeddings, ModelCallStat]:
        from kavalai.llm_clients.browser_client import get_browser_bridge

        start_time = time.perf_counter()
        bridge = get_browser_bridge()
        request = {"model": self.model, "input": list(texts)}

        try:
            # ``bridge.embed`` resolves to a JS Promise; awaiting it in Pyodide
            # yields the resolved JSON string.
            raw = await bridge.embed(json.dumps(request))
        except Exception as exc:  # JsException or anything the bridge throws.
            raise LlmClientException(
                f"In-browser embedding call failed: {exc}"
            ) from exc

        data = json.loads(raw)
        if data.get("error"):
            raise LlmClientException(f"In-browser embedding error: {data['error']}")

        embeddings = _maybe_normalize(
            data.get("embeddings") or [], normalize, normalizer
        )
        usage = data.get("usage") or {}
        total_tokens = usage.get("total_tokens") or usage.get("prompt_tokens") or 0
        duration = time.perf_counter() - start_time

        stats = create_model_call_stat(
            call_type="embedding",
            model=f"browser/{self.model}",
            duration_seconds=duration,
            batch_size=len(texts),
            total_tokens=total_tokens,
        )
        return embeddings, stats


def make_embedding_client(model: str, **kwargs) -> BaseEmbeddingClient:
    """Construct an embedding client from a ``provider/model`` string.

    The provider is the part before the first ``/``; the remainder is the model
    name and may itself contain slashes
    (``fastembed/BAAI/bge-small-en-v1.5``). Built-in providers are ``openai``,
    ``gemini``, ``ollama``, ``fastembed`` and ``browser``; ``browser`` runs
    client-side via a WebLLM bridge (Pyodide only) and needs no API key.

    Any provider registered with
    :func:`~kavalai.register_embedding_provider` resolves here too, and an
    exact ``provider/model`` registration takes precedence over the
    provider-wide one.

    Args:
        model: The ``provider/model`` identifier.
        **kwargs: Extra constructor arguments, overriding any bound at
            registration.

    Raises:
        ValueError: ``model`` has no provider prefix.
        RegistryError: No provider is registered for the prefix.
    """
    from kavalai.llm_clients import registry

    if "/" not in model:
        raise ValueError(f"Embedding model must be 'provider/model', got '{model}'.")
    _, model_name = model.split("/", maxsplit=1)
    return registry.embedding_providers.build(model, model_name, **kwargs)
