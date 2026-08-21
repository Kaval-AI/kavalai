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

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional, Union
from uuid import UUID

from pydantic import BaseModel

from kavalai.normalizer import Normalizer, get_default_normalizer


class RagServiceResult(BaseModel):
    """
    Represents a single result from a RAG query.

    Attributes:
        id (UUID): Unique identifier of the indexed item. Kaval.AI mints these,
            and backends must be able to store one: a store whose identifiers
            are integers or bounded strings has to accept a UUID (as text if
            necessary) and give the same value back.
        model (str): The embedding model used for this item.
        collection_name (str): The name of the collection this item belongs to.
        source_id (str): An external identifier for the source of this item.
        content (Optional[str]): The original text content that was indexed.
        embedding_size (int): The dimension of the embedding vector.
        rag_metadata (dict): Additional metadata associated with the item.
        similarity (float): How close the item is to the query, **higher is
            better**, comparable across backends for the same embedding model.
            Backends index with cosine distance and report ``1.0 - distance``,
            so a perfect match scores ``1.0``. A backend whose store reports a
            metric-dependent score must normalise to this convention rather
            than passing its own number through.
        created_at (Optional[datetime]): Timestamp when the item was created.
        updated_at (Optional[datetime]): Timestamp when the item was last updated.
        query_index (Optional[int]): Index of the query in batch queries (for query_batch results).
    """

    id: UUID
    model: str
    collection_name: str
    source_id: str
    content: Optional[str] = None
    embedding_size: int
    rag_metadata: dict
    similarity: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    query_index: Optional[int] = None


class BaseRagService(ABC):
    """
    Interface for RAG (Retrieval-Augmented Generation) storage backends.

    A RAG service indexes text documents as embeddings and answers similarity
    queries against them. The interface is generic; the implementations are
    deliberately not. ``PostgresRagService`` pushes similarity into pgvector
    and carries a dozen methods this interface never mentions, and that is the
    intended shape --- a backend is free to expose whatever its store does well,
    as long as the declared surface below behaves identically everywhere.

    The surface comes in three tiers:

    **Required.** :meth:`index`, :meth:`index_batch`, :meth:`query`,
    :meth:`query_batch`, :meth:`delete` and :meth:`delete_by_source_id` are
    abstract. Every backend implements all six.

    **Optional.** :meth:`count_entries` and :meth:`iter_entries` raise
    ``NotImplementedError`` by default. Ask :meth:`supports` before calling
    them rather than catching the exception.

    **Defaulted.** :meth:`compute_similarity_matrix` and
    :meth:`learn_normalizer` work on every backend as written, and may be
    overridden with an exact or cheaper implementation.

    ``tests/rag/test_conformance.py`` runs the declared contract against every
    backend, so "implements the interface correctly" is checked rather than
    assumed.
    """

    #: Optional methods this backend actually implements. Subclasses that
    #: override ``count_entries`` or ``iter_entries`` list them here.
    capabilities: frozenset = frozenset()

    def supports(self, capability: str) -> bool:
        """Whether an optional method is implemented by this backend.

        ``count_entries`` and ``iter_entries`` are optional, so a caller that
        needs one (the backoffice embedding projector needs ``iter_entries``)
        should ask rather than catch ``NotImplementedError``.

        Args:
            capability: An optional method name, e.g. ``"iter_entries"``.

        Returns:
            True if this backend implements it.
        """
        return capability in self.capabilities

    # Number of candidates fetched per source by the default (query_batch-based)
    # compute_similarity_matrix implementation. Sources with more indexed items
    # than this may yield approximate "avg" aggregates.
    similarity_matrix_candidates_per_source: int = 100

    @abstractmethod
    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
        source_id: str = "default",
    ) -> dict:
        """
        Index a single text blob with metadata.

        Args:
            text (str): The text content to index.
            source_metadata (Optional[dict]): Metadata to associate with the text.
            collection_name (str): Name of the collection. Defaults to "default".
            source_id (str): Source identifier. Defaults to "default".

        Returns:
            dict: The created entry. It always carries ``id``, ``model``,
            ``collection_name``, ``source_id``, ``content``,
            ``embedding_size``, ``rag_metadata``, ``created_at`` and
            ``updated_at``; a backend may add its own keys (Postgres also
            returns ``embedding``). The guaranteed keys are part of the
            contract, not a backend detail.
        """

    @abstractmethod
    async def index_batch(
        self,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        collection_name: str = "default",
    ) -> list[dict]:
        """
        Index multiple text items in a single batch.

        Batch indexing can be significantly more efficient than repeated
        :meth:`index` calls with certain backends.

        Args:
            texts (list[str]): List of text strings to index.
            metadata_list (list[dict]): List of metadata dictionaries for each text.
            source_ids (Optional[list[str]]): Optional list of source identifiers.
                                              If not provided, "default" is used.
            collection_name (str): Name of the collection to add items to. Defaults to "default".

        Returns:
            list[dict]: The created entries, each shaped as in :meth:`index`.

        Raises:
            ValueError: If the lengths of texts, metadata_list, or source_ids do not match.
        """

    @abstractmethod
    async def query(
        self,
        text: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        keep_best: bool = False,
        include_content: bool = True,
    ) -> list[RagServiceResult]:
        """
        Query the indexed items for similarities to the input text.

        Args:
            text (str): The query text.
            top_k (int): Number of top results to return. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            source_ids (Optional[list[str]]): If provided, filter by source identifiers.
            keep_best (bool): If True, only the best result per source_id is
                returned. Useful when a single source is split into multiple
                indexed items. Backends whose store cannot group server-side
                over-fetch and reduce in the client, which makes the result
                approximate when a source has more indexed items than the
                over-fetch window --- the same caveat
                :meth:`compute_similarity_matrix` carries.
            include_content (bool): When False, ``content`` is omitted from the
                results. Everything else is unchanged. Free on a local
                database; on a remote store it avoids transferring text that
                the caller is only going to discard.

        Returns:
            list[RagServiceResult]: Results ordered by descending
            ``similarity``.
        """

    @abstractmethod
    async def query_batch(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        include_content: bool = True,
    ) -> list[list[RagServiceResult]]:
        """
        Query the indexed items for similarities to multiple input texts.

        Batch querying can be significantly more efficient than repeated
        :meth:`query` calls with certain backends.

        Args:
            texts (list[str]): List of query texts to search for.
            top_k (int): Number of top results to return per query. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            source_ids (Optional[list[str]]): If provided, filter by source identifiers.
            include_content (bool): When False, ``content`` is omitted from the
                results. See :meth:`query`.

        Returns:
            list[list[RagServiceResult]]: A list of result lists, where each inner list contains
                                          the top_k results for the corresponding query text.
        """

    @abstractmethod
    async def delete(
        self, item_id: UUID, collection_name: Optional[str] = None
    ) -> None:
        """
        Delete a single indexed item by its identifier.

        Args:
            item_id (UUID): Identifier of the indexed item to delete.
            collection_name (Optional[str]): Collection the item belongs to.
                Backends that store collections separately search all
                collections when omitted.
        """

    @abstractmethod
    async def delete_by_source_id(
        self,
        collection_name: str,
        source_id: Union[str, list[str]],
    ) -> None:
        """
        Delete all items in a collection that match the given source identifier(s).

        Args:
            collection_name (str): The name of the collection.
            source_id (Union[str, list[str]]): A source identifier, or a list of them.
        """

    async def count_entries(self, collection_name: str) -> int:
        """Number of entries in a collection (0 if it doesn't exist)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support count_entries"
        )

    def iter_entries(
        self, collection_name: str, batch_size: int = 500
    ) -> "AsyncIterator[dict]":
        """
        Iterate all entries of a collection (including embeddings).

        Yields dicts with keys: id, source_id, content, embedding,
        rag_metadata, created_at, updated_at. Used for bulk export (e.g. the
        backoffice embedding projector) so no caller needs to touch backend
        storage directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support iter_entries"
        )

    async def compute_similarity_matrix(
        self,
        texts: list[str],
        source_ids: list[str],
        method: str = "min",
        collection_name: str = "default",
    ) -> list[list[float]]:
        """
        Compute a similarity matrix between multiple texts and multiple source identifiers.

        Default implementation built on :meth:`query_batch`: it retrieves up to
        ``similarity_matrix_candidates_per_source`` candidates per source and
        aggregates similarities per source_id. Sources with more indexed items
        than that may yield approximate "avg" aggregates; backends can override
        this with an exact implementation.

        Args:
            texts (list[str]): List of query texts (rows in the matrix).
            source_ids (list[str]): List of source identifiers to compare against (columns in the matrix).
            method (str): Aggregate method to use when multiple items exist for a source_id.
                          "min" (default) uses the shortest distance (highest similarity).
                          "avg" uses the average distance.

        Returns:
            list[list[float]]: A 2D matrix where matrix[i][j] is the similarity between
                               texts[i] and source_ids[j]. Missing sources score 0.0.
        """
        if not texts or not source_ids:
            return [[0.0 for _ in source_ids] for _ in texts]

        top_k = len(source_ids) * self.similarity_matrix_candidates_per_source
        batch_results = await self.query_batch(
            texts=texts,
            top_k=top_k,
            collection_name=collection_name,
            source_ids=source_ids,
        )

        source_id_to_idx = {sid: i for i, sid in enumerate(source_ids)}
        matrix = [[0.0 for _ in range(len(source_ids))] for _ in range(len(texts))]

        for t_idx, results in enumerate(batch_results):
            similarities_by_source: dict[str, list[float]] = {}
            for result in results:
                similarities_by_source.setdefault(result.source_id, []).append(
                    result.similarity
                )
            for sid, sims in similarities_by_source.items():
                s_idx = source_id_to_idx.get(sid)
                if s_idx is None:
                    continue
                # similarity = 1 - distance, so min distance == max similarity and
                # the average distance maps to the average similarity.
                matrix[t_idx][s_idx] = (
                    max(sims) if method == "min" else sum(sims) / len(sims)
                )

        return matrix

    async def learn_normalizer(
        self, collection_name: Optional[str] = None
    ) -> Normalizer:
        """
        Learn a normalizer from the indexed data.

        Default implementation returns the process-wide default normalizer;
        backends with access to the stored embeddings should override this to
        learn (e.g.) a centering vector from the index.

        Args:
            collection_name (Optional[str]): If provided, learn only from this collection.

        Returns:
            Normalizer: The learned (or default) normalizer.
        """
        return get_default_normalizer()
