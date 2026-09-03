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

The storage model the SQL-backed RAG services share.

Both :class:`~kavalai.rag.postgres.PostgresRagService` and
:class:`~kavalai.rag.sqllite.SqliteRagService` keep a registry table,
``rag_collections``, with one row per collection — its name, the table its
vectors live in, the embedding model, the embedding dimension and a
``schema_version`` — and one table per collection holding ``id``,
``source_id``, ``content``, ``embedding``, ``metadata`` and two timestamps.
Only the column types differ between the two databases.

:class:`CollectionRagService` holds everything that follows from that model
and is the same on every backend: collection naming, lazy provisioning on
first index, the dimension check, the in-code schema upgrades, the public
browse methods the backoffice explorer relies on, and the mapping of rows
onto :class:`~kavalai.rag.base.RagServiceResult`. A backend implements the
statements — a couple of dozen small hooks, each one SQL statement — and
nothing else.

``RAG_COLLECTION_SCHEMA_VERSION`` is the version of the per-collection table
layout. Bump it when the layout changes and register an upgrade step in the
backend's ``collection_upgrades``, which maps a *from_version* to an async
callable ``(connection, service, collection_info) -> None`` that brings a
collection from ``from_version`` to ``from_version + 1``.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, AsyncContextManager, AsyncIterator, Callable, Optional, Union
from uuid import UUID, uuid4

from loguru import logger

from kavalai.llm_clients.embeddings import make_embedding_client
from kavalai.normalizer import Normalizer
from kavalai.rag.base import BaseRagService, RagServiceResult

RAG_COLLECTION_SCHEMA_VERSION = 1


class CollectionInfo:
    """Registry entry for one RAG collection."""

    def __init__(
        self,
        name: str,
        table_name: str,
        model: str,
        embedding_size: int,
        schema_version: int,
    ):
        self.name = name
        self.table_name = table_name
        self.model = model
        self.embedding_size = embedding_size
        self.schema_version = schema_version


class CollectionRagService(BaseRagService):
    """A RAG service over registered per-collection tables.

    The public surface is :class:`~kavalai.rag.base.BaseRagService` plus the
    browse methods a management interface needs — :meth:`list_collections`,
    :meth:`get_stats`, :meth:`create_collection`, :meth:`drop_collection` and
    :meth:`get_embeddings_by_ids`. All of them work without an embedding
    model; ``model`` is needed only to index or query, so a service opened
    for inspection is constructed with ``model=None``.

    ``collection_name=None`` means ``"default"`` everywhere: with one table
    per collection there is no cross-collection search.

    Subclasses implement the ``_``-prefixed hooks. ``_connection`` yields
    whatever the backend talks to (an async session, a DBAPI connection); the
    other hooks receive it back and run one statement each. Rows cross the
    boundary as plain dicts with ``id`` a :class:`~uuid.UUID`, ``metadata`` a
    dict and the timestamps :class:`~datetime.datetime` objects, so the base
    never sees a backend's storage types.
    """

    capabilities = frozenset({"count_entries", "iter_entries"})

    REGISTRY_TABLE = "rag_collections"

    collection_schema_version: int = RAG_COLLECTION_SCHEMA_VERSION
    collection_upgrades: dict[int, Callable] = {}

    def __init__(
        self,
        model: Optional[str] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self._embedding_client = None
        self._registry_ready = False
        self._collections: dict[str, CollectionInfo] = {}

    @property
    def embedding_client(self):
        """Embedding client, created on first use.

        Resolved lazily so that a caller holding a service can substitute the
        embedding side before any provider is looked up.
        """
        if self._embedding_client is None:
            if not self.model:
                raise ValueError(
                    f"This {type(self).__name__} was created without an "
                    f"embedding model; indexing and querying require one."
                )
            self._embedding_client = make_embedding_client(self.model)
        return self._embedding_client

    @embedding_client.setter
    def embedding_client(self, client) -> None:
        self._embedding_client = client

    @staticmethod
    def table_name_for_collection(collection_name: str) -> str:
        """Deterministic, SQL-safe table name for a collection.

        A sanitized slug keeps the name readable; a short hash of the exact
        collection name guarantees uniqueness across names that sanitize to
        the same slug.
        """
        slug = re.sub(r"[^a-z0-9_]+", "_", collection_name.lower()).strip("_")[:32]
        digest = hashlib.sha1(  # nosec B324 - naming, not security
            collection_name.encode("utf-8")
        ).hexdigest()[:8]
        return f"rag_c_{slug}_{digest}" if slug else f"rag_c_{digest}"

    # Backend hooks

    def _connection(self) -> AsyncContextManager[Any]:
        """Yield the handle every other hook receives."""
        raise NotImplementedError

    async def _commit(self, conn) -> None:
        """End the current transaction on ``conn``."""
        raise NotImplementedError

    async def _record_stats(self, conn, stats) -> None:
        """Persist an embedding call's usage, where the backend has a place for it."""

    async def _create_registry(self, conn) -> None:
        """Create the registry table (and whatever it depends on) if missing."""
        raise NotImplementedError

    async def _fetch_registry_row(self, conn, name: str) -> Optional[CollectionInfo]:
        raise NotImplementedError

    async def _list_registry(self, conn) -> list[CollectionInfo]:
        """Every registry row, ordered by name."""
        raise NotImplementedError

    async def _insert_registry_row(self, conn, info: CollectionInfo) -> None:
        raise NotImplementedError

    async def _set_registry_version(self, conn, name: str, version: int) -> None:
        raise NotImplementedError

    async def _delete_registry_row(self, conn, name: str) -> None:
        raise NotImplementedError

    async def _create_collection_table(self, conn, info: CollectionInfo) -> None:
        """Create the collection's table and its indexes."""
        raise NotImplementedError

    async def _drop_collection_table(self, conn, info: CollectionInfo) -> None:
        raise NotImplementedError

    async def _prepare_collection(self, conn, info: CollectionInfo) -> None:
        """Per-connection set-up a backend needs before touching a collection."""

    async def _count_rows(self, conn, info: CollectionInfo) -> int:
        raise NotImplementedError

    async def _insert_rows(self, conn, info: CollectionInfo, rows: list[dict]) -> None:
        raise NotImplementedError

    async def _delete_row(self, conn, info: CollectionInfo, item_id: UUID) -> None:
        raise NotImplementedError

    async def _delete_rows_by_source_ids(
        self, conn, info: CollectionInfo, source_ids: list[str]
    ) -> None:
        raise NotImplementedError

    def _iter_rows(
        self, conn, info: CollectionInfo, batch_size: int
    ) -> AsyncIterator[dict]:
        """Every row of the collection, embeddings included, in ``id`` order."""
        raise NotImplementedError

    async def _fetch_embeddings(
        self, conn, info: CollectionInfo, ids: list[UUID]
    ) -> dict[UUID, list[float]]:
        raise NotImplementedError

    async def _scan(
        self,
        conn,
        info: CollectionInfo,
        embeddings: list[list[float]],
        top_k: int,
        source_ids: Optional[list[str]],
        keep_best: bool,
    ) -> list[list[dict]]:
        """The nearest rows per query embedding, each with a cosine ``distance``."""
        raise NotImplementedError

    # Registry

    async def _ensure_registry(self, conn) -> None:
        if self._registry_ready:
            return
        await self._create_registry(conn)
        await self._commit(conn)
        self._registry_ready = True

    async def _load_collection(
        self, conn, collection_name: str
    ) -> Optional[CollectionInfo]:
        """Fetch a collection's registry entry (cached), upgrading if stale."""
        cached = self._collections.get(collection_name)
        if cached is not None:
            return cached

        await self._ensure_registry(conn)
        info = await self._fetch_registry_row(conn, collection_name)
        if info is None:
            return None
        await self._upgrade_collection(conn, info)
        await self._prepare_collection(conn, info)
        self._collections[collection_name] = info
        return info

    async def _upgrade_collection(self, conn, info: CollectionInfo) -> None:
        """Bring a collection table up to ``collection_schema_version``."""
        if info.schema_version > self.collection_schema_version:
            raise ValueError(
                f"Collection '{info.name}' has schema_version "
                f"{info.schema_version}, newer than this library supports "
                f"({self.collection_schema_version}). Upgrade kavalai."
            )
        while info.schema_version < self.collection_schema_version:
            upgrade = self.collection_upgrades.get(info.schema_version)
            if upgrade is None:
                raise ValueError(
                    f"No upgrade step registered from collection schema_version "
                    f"{info.schema_version} (collection '{info.name}')."
                )
            logger.info(
                f"Upgrading RAG collection '{info.name}' from schema_version "
                f"{info.schema_version} to {info.schema_version + 1}."
            )
            await upgrade(conn, self, info)
            info.schema_version += 1
            await self._set_registry_version(conn, info.name, info.schema_version)
            await self._commit(conn)

    async def _ensure_collection(
        self, conn, collection_name: str, embedding_size: int
    ) -> CollectionInfo:
        """Get a collection, creating its table on first use."""
        info = await self._load_collection(conn, collection_name)
        if info is not None:
            if info.embedding_size != embedding_size:
                raise ValueError(
                    f"Collection '{collection_name}' stores "
                    f"{info.embedding_size}-dimensional embeddings; got "
                    f"{embedding_size}."
                )
            return info

        if not self.model:
            raise ValueError(
                f"This {type(self).__name__} was created without an embedding "
                f"model; creating collections requires one."
            )

        info = CollectionInfo(
            name=collection_name,
            table_name=self.table_name_for_collection(collection_name),
            model=self.model,
            embedding_size=embedding_size,
            schema_version=self.collection_schema_version,
        )
        await self._create_collection_table(conn, info)
        await self._insert_registry_row(conn, info)
        await self._commit(conn)
        await self._prepare_collection(conn, info)
        logger.info(
            f"Provisioned RAG collection '{collection_name}' "
            f"(table {info.table_name}, dim {embedding_size})."
        )
        self._collections[collection_name] = info
        return info

    async def create_collection(
        self, collection_name: str, embedding_size: int
    ) -> None:
        """Explicitly provision a collection with a known embedding dimension."""
        async with self._connection() as conn:
            await self._ensure_collection(conn, collection_name, embedding_size)

    async def drop_collection(self, collection_name: str) -> None:
        """Drop a collection: its table and registry entry."""
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                return
            await self._drop_collection_table(conn, info)
            await self._delete_registry_row(conn, collection_name)
            await self._commit(conn)
        self._collections.pop(collection_name, None)
        logger.info(f"Dropped RAG collection '{collection_name}'.")

    async def list_collections(self) -> list[dict]:
        """List registered collections with entry counts."""
        async with self._connection() as conn:
            await self._ensure_registry(conn)
            collections = []
            for info in await self._list_registry(conn):
                collections.append(
                    {
                        "name": info.name,
                        "model": info.model,
                        "embedding_size": info.embedding_size,
                        "schema_version": info.schema_version,
                        "count": await self._count_rows(conn, info),
                    }
                )
            await self._commit(conn)
            return collections

    async def get_stats(self) -> dict:
        """Aggregate stats across collections (for e.g. the backoffice)."""
        collections = await self.list_collections()
        return {
            "total_entries": sum(c["count"] for c in collections),
            "total_collections": len(collections),
            "collections": [c["name"] for c in collections],
        }

    # Indexing

    async def _compute_embeddings(self, conn, texts: list[str]) -> list[list[float]]:
        embeddings, stats = await self.embedding_client.compute_embeddings(
            texts=texts, normalizer=self.normalizer
        )
        await self._record_stats(conn, stats)
        return embeddings

    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
        source_id: str = "default",
    ) -> dict:
        """Index a single text blob with metadata. Returns the created row dict."""
        return (
            await self.index_batch(
                texts=[text],
                metadata_list=[source_metadata or {}],
                collection_name=collection_name,
                source_ids=[source_id],
            )
        )[0]

    async def index_batch(
        self,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        collection_name: str = "default",
    ) -> list[dict]:
        """
        Index multiple text items in a single batch.

        The collection is provisioned on first use, taking its embedding
        dimension from the computed embeddings.

        Returns:
            list[dict]: Created rows (id, model, collection_name, source_id,
                content, embedding_size, embedding, rag_metadata, created_at,
                updated_at).
        """
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        if source_ids and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

        async with self._connection() as conn:
            embeddings = await self._compute_embeddings(conn, texts)
            dim = len(embeddings[0])
            info = await self._ensure_collection(conn, collection_name, dim)

            now = datetime.now(timezone.utc)
            source_ids = source_ids or ["default"] * len(texts)
            rows = [
                {
                    "id": uuid4(),
                    "model": info.model,
                    "collection_name": collection_name,
                    "source_id": source_id,
                    "content": content,
                    "embedding_size": dim,
                    "embedding": list(emb),
                    "rag_metadata": meta,
                    "created_at": now,
                    "updated_at": now,
                }
                for content, meta, emb, source_id in zip(
                    texts, metadata_list, embeddings, source_ids
                )
            ]
            await self._insert_rows(conn, info, rows)
            await self._commit(conn)
            return rows

    # Deletion

    async def delete(
        self, item_id: UUID, collection_name: Optional[str] = None
    ) -> None:
        """
        Delete a single indexed item by its identifier.

        Args:
            item_id (UUID): Identifier of the indexed item to delete.
            collection_name (Optional[str]): Collection the item belongs to.
                If omitted, all registered collections are searched.
        """
        async with self._connection() as conn:
            if collection_name is not None:
                info = await self._load_collection(conn, collection_name)
                infos = [info] if info else []
            else:
                await self._ensure_registry(conn)
                infos = [
                    await self._load_collection(conn, entry.name)
                    for entry in await self._list_registry(conn)
                ]
            for info in infos:
                await self._delete_row(conn, info, item_id)
            await self._commit(conn)

    async def delete_by_source_id(
        self,
        collection_name: str,
        source_id: Union[str, list[str]],
    ) -> None:
        """Delete all items in a collection matching the source identifier(s)."""
        source_ids = [source_id] if isinstance(source_id, str) else source_id
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                return
            await self._delete_rows_by_source_ids(conn, info, source_ids)
            await self._commit(conn)

    # Querying

    def _map_row(
        self,
        row: dict,
        info: CollectionInfo,
        query_index: Optional[int] = None,
        include_content: bool = True,
    ) -> RagServiceResult:
        distance = row.get("distance")
        return RagServiceResult(
            id=row["id"],
            model=info.model,
            collection_name=info.name,
            source_id=row["source_id"],
            content=row["content"] if include_content else None,
            embedding_size=info.embedding_size,
            rag_metadata=row.get("metadata") or {},
            similarity=1.0 - float(distance) if distance is not None else 0.0,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            query_index=query_index,
        )

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
        Query one collection for similarities to the input text.

        ``collection_name`` defaults to ``"default"`` — with table-per-collection
        storage there is no cross-collection search; query each collection
        explicitly if needed.
        """
        results = await self.query_batch(
            texts=[text],
            top_k=top_k,
            collection_name=collection_name,
            source_ids=source_ids,
            keep_best=keep_best,
            include_content=include_content,
        )
        out = results[0]
        for item in out:
            item.query_index = None
        return out

    async def query_batch(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        keep_best: bool = False,
        include_content: bool = True,
    ) -> list[list[RagServiceResult]]:
        """
        Query one collection for similarities to multiple input texts.

        The embeddings for all query texts are computed in a single call.
        """
        if not texts:
            return []
        collection_name = collection_name or "default"

        async with self._connection() as conn:
            embeddings = await self._compute_embeddings(conn, texts)
            info = await self._load_collection(conn, collection_name)
            if info is None:
                await self._commit(conn)
                return [[] for _ in texts]

            batches = await self._scan(
                conn, info, embeddings, top_k, source_ids, keep_best
            )
            await self._commit(conn)
            return [
                [self._map_row(row, info, index, include_content) for row in rows]
                for index, rows in enumerate(batches)
            ]

    # Bulk export

    async def count_entries(self, collection_name: str) -> int:
        """Number of entries in a collection (0 if it doesn't exist)."""
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            count = 0 if info is None else await self._count_rows(conn, info)
            # Close the read transaction: shared-session factories (e.g. the
            # backoffice) would otherwise pin a pooled connection.
            await self._commit(conn)
            return count

    async def iter_entries(
        self, collection_name: str, batch_size: int = 500
    ) -> AsyncIterator[dict]:
        """
        Iterate all entries of a collection (including embeddings) in stable
        ``id`` order. Yields dicts with keys: id, source_id, content,
        embedding, rag_metadata, created_at, updated_at.
        """
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                await self._commit(conn)
                return
            async for row in self._iter_rows(conn, info, batch_size):
                yield row
            await self._commit(conn)

    async def get_embeddings_by_ids(
        self, collection_name: str, ids: list[UUID]
    ) -> dict[UUID, list[float]]:
        """Fetch embeddings for specific entry ids within a collection."""
        if not ids:
            return {}
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            found = (
                {} if info is None else await self._fetch_embeddings(conn, info, ids)
            )
            await self._commit(conn)
            return found
