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
and is the same on every backend: collection naming, provisioning, the
dimension check, the in-code schema upgrades, the public browse methods the
backoffice explorer relies on, and the mapping of rows onto
:class:`~kavalai.rag.base.RagServiceResult`. A backend implements the
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

MetadataValue = Union[str, int, float, bool]


class CollectionInfo:
    """Registry entry for one RAG collection.

    ``vector_type`` is the column type of the collection's embeddings —
    ``"vector"`` (32-bit floats) or, on PostgreSQL, ``"halfvec"`` (16-bit).
    It is read from the table itself rather than stored in the registry.
    """

    def __init__(
        self,
        name: str,
        table_name: str,
        model: str,
        embedding_size: int,
        schema_version: int,
        vector_type: str = "vector",
    ):
        self.name = name
        self.table_name = table_name
        self.model = model
        self.embedding_size = embedding_size
        self.schema_version = schema_version
        self.vector_type = vector_type


def validate_metadata_match(match: dict) -> None:
    """Check a metadata match is a non-empty mapping of top-level scalars.

    Equality on top-level keys is what both backends can serve the same way —
    ``metadata @> :match`` on PostgreSQL, ``json_extract`` on SQLite — so it
    is all the interface promises.
    """
    if not isinstance(match, dict) or not match:
        raise ValueError("A metadata match must be a non-empty dict.")
    for key, value in match.items():
        if not isinstance(key, str) or not key or '"' in key:
            raise ValueError(
                f"Metadata match key {key!r} must be a non-empty string without "
                f"double quotes."
            )
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"Metadata match value for {key!r} must be a string, number or "
                f"boolean, not {type(value).__name__}."
            )


class CollectionRagService(BaseRagService):
    """A RAG service over registered per-collection tables.

    The public surface is :class:`~kavalai.rag.base.BaseRagService` plus the
    methods a management interface needs — :meth:`list_collections`,
    :meth:`get_stats`, :meth:`create_collection`, :meth:`drop_collection`,
    :meth:`ensure_registry` and :meth:`get_embeddings_by_ids`.

    **The registry's model is authoritative.** Indexing into or querying an
    existing collection embeds with the model recorded for that collection,
    so one service serves collections of different embedding models, and a
    query can never be embedded with a model the collection was not built
    with. ``model`` is the model *new* collections are created with; a
    service without one browses, queries and indexes existing collections but
    cannot create one. One embedding client is kept per model.

    ``collection_name=None`` means ``"default"`` everywhere: with one table
    per collection there is no cross-collection search.

    Args:
        model: Embedding model for collections this service creates.
        normalizer: Normaliser applied on both the index and the query side.
        provision: When ``False`` the service issues no DDL: it never creates
            the registry, a collection table or the vector extension, and
            indexing into a missing collection raises. Collections are then
            created by :meth:`create_collection` and :meth:`ensure_registry`
            from a role that may, which lets the runtime role hold data
            privileges only.
        stats_receiver: Receives every embedding call's statistics unless a
            call passes its own. ``None`` records nothing.
        vector_type: Column type for collections this service creates. Only
            the types in ``vector_types`` are accepted.

    Subclasses implement the ``_``-prefixed hooks. ``_connection`` yields
    whatever the backend talks to (an async session, a DBAPI connection); the
    other hooks receive it back and run one statement each. Rows cross the
    boundary as plain dicts with ``id`` a :class:`~uuid.UUID`, ``metadata`` a
    dict and the timestamps :class:`~datetime.datetime` objects, so the base
    never sees a backend's storage types.
    """

    capabilities = frozenset(
        {"count_entries", "iter_entries", "delete_by_metadata", "replace"}
    )

    REGISTRY_TABLE = "rag_collections"

    collection_schema_version: int = RAG_COLLECTION_SCHEMA_VERSION
    collection_upgrades: dict[int, Callable] = {}
    vector_types: frozenset = frozenset({"vector"})

    def __init__(
        self,
        model: Optional[str] = None,
        normalizer: Optional[Normalizer] = None,
        *,
        provision: bool = True,
        stats_receiver: Any = None,
        vector_type: str = "vector",
    ):
        if vector_type not in self.vector_types:
            raise ValueError(
                f"{type(self).__name__} stores {sorted(self.vector_types)} "
                f"embeddings, not {vector_type!r}."
            )
        self.model = model
        self.normalizer = normalizer
        self.provision = provision
        self.stats_receiver = stats_receiver
        self.vector_type = vector_type
        self._embedding_clients: dict[str, Any] = {}
        self._embedding_override = None
        self._registry_ready = False
        self._collections: dict[str, CollectionInfo] = {}

    @property
    def embedding_client(self):
        """The embedding client for ``model``, created on first use.

        Assigning a client substitutes it for every model this service
        embeds with — how a test replaces the provider before any lookup.
        """
        if self._embedding_override is not None:
            return self._embedding_override
        return self._client_for(self._require_model("embedding text"))

    @embedding_client.setter
    def embedding_client(self, client) -> None:
        self._embedding_override = client

    def _client_for(self, model: str):
        if self._embedding_override is not None:
            return self._embedding_override
        client = self._embedding_clients.get(model)
        if client is None:
            client = make_embedding_client(model)
            self._embedding_clients[model] = client
        return client

    def _require_model(self, action: str) -> str:
        if not self.model:
            raise ValueError(
                f"This {type(self).__name__} was created without an embedding "
                f"model; {action} requires one."
            )
        return self.model

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

    async def _rollback(self, conn) -> None:
        """Abandon the current transaction on ``conn``."""
        raise NotImplementedError

    async def _registry_exists(self, conn) -> bool:
        """Whether the registry table exists. Must not create it."""
        raise NotImplementedError

    async def _create_registry(self, conn) -> None:
        """Create the registry table (and whatever it depends on) if missing."""
        raise NotImplementedError

    async def _fetch_registry_row(self, conn, name: str) -> Optional[CollectionInfo]:
        raise NotImplementedError

    async def _list_registry(self, conn) -> list[CollectionInfo]:
        """Every registry row, ordered by name."""
        raise NotImplementedError

    async def _insert_registry_row(self, conn, info: CollectionInfo) -> None:
        """Insert the row unless one with the same name exists."""
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

    async def _delete_rows(
        self, conn, info: CollectionInfo, item_ids: list[UUID]
    ) -> None:
        raise NotImplementedError

    async def _delete_rows_by_source_ids(
        self, conn, info: CollectionInfo, source_ids: list[str]
    ) -> None:
        raise NotImplementedError

    async def _delete_rows_by_metadata(
        self, conn, info: CollectionInfo, match: dict[str, MetadataValue]
    ) -> None:
        """Delete the rows whose metadata has every key of ``match`` equal."""
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

    async def _registry_available(self, conn) -> bool:
        """Whether the registry exists, without creating it."""
        if not self._registry_ready and await self._registry_exists(conn):
            self._registry_ready = True
        return self._registry_ready

    async def _ensure_registry(self, conn) -> None:
        """Create the registry if it is missing. Callers decide whether DDL is allowed."""
        if await self._registry_available(conn):
            return
        await self._create_registry(conn)
        await self._commit(conn)
        self._registry_ready = True

    async def ensure_registry(self) -> None:
        """Create the registry — and on PostgreSQL the vector extension.

        Issues DDL whatever ``provision`` says: this is the call a migration
        job or an administrator makes, so that services running with
        ``provision=False`` find the registry in place.
        """
        async with self._connection() as conn:
            await self._ensure_registry(conn)

    async def _load_collection(
        self, conn, collection_name: str
    ) -> Optional[CollectionInfo]:
        """Fetch a collection's registry entry (cached), upgrading if stale.

        Never creates anything: a missing registry reads as a missing
        collection.
        """
        cached = self._collections.get(collection_name)
        if cached is not None:
            return cached

        if not await self._registry_available(conn):
            return None
        info = await self._fetch_registry_row(conn, collection_name)
        if info is None:
            return None
        await self._upgrade_collection(conn, info)
        await self._prepare_collection(conn, info)
        if self.model and info.model != self.model:
            logger.warning(
                f"RAG collection '{collection_name}' was built with "
                f"{info.model}; it is embedded with that model, not with this "
                f"service's {self.model}."
            )
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
        if info.schema_version < self.collection_schema_version and not self.provision:
            raise RuntimeError(
                f"Collection '{info.name}' needs an upgrade from schema_version "
                f"{info.schema_version}, and this {type(self).__name__} was "
                f"created with provision=False. Open it once with a service "
                f"that may issue DDL."
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
        self,
        conn,
        collection_name: str,
        embedding_size: int,
        model: str,
        *,
        vector_type: Optional[str] = None,
    ) -> CollectionInfo:
        """Get a collection, creating its table if it is missing.

        Issues DDL whatever ``provision`` says; callers that must not create
        a collection check before they get here.
        """
        info = await self._load_collection(conn, collection_name)
        if info is not None:
            self._check_dimension(info, embedding_size)
            return info

        await self._ensure_registry(conn)

        info = CollectionInfo(
            name=collection_name,
            table_name=self.table_name_for_collection(collection_name),
            model=model,
            embedding_size=embedding_size,
            schema_version=self.collection_schema_version,
            vector_type=vector_type or self.vector_type,
        )
        await self._create_collection_table(conn, info)
        await self._insert_registry_row(conn, info)
        # Two processes may provision one name at once; the registry keeps
        # whichever row arrived first, and a loser with another dimension or
        # model must not carry on with its own idea of the collection.
        stored = await self._fetch_registry_row(conn, collection_name)
        await self._commit(conn)
        if stored.embedding_size != embedding_size or stored.model != model:
            raise ValueError(
                f"RAG collection '{collection_name}' was created concurrently "
                f"with {stored.model} ({stored.embedding_size} dimensions); "
                f"this call wanted {model} ({embedding_size} dimensions)."
            )
        await self._prepare_collection(conn, stored)
        logger.info(
            f"Provisioned RAG collection '{collection_name}' "
            f"(table {stored.table_name}, dim {embedding_size}, "
            f"{stored.vector_type})."
        )
        self._collections[collection_name] = stored
        return stored

    @staticmethod
    def _check_dimension(info: CollectionInfo, embedding_size: int) -> None:
        if info.embedding_size != embedding_size:
            raise ValueError(
                f"Collection '{info.name}' stores {info.embedding_size}-"
                f"dimensional embeddings; got {embedding_size}."
            )

    async def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        *,
        model: Optional[str] = None,
        vector_type: Optional[str] = None,
    ) -> None:
        """Explicitly provision a collection with a known embedding dimension.

        Issues DDL whatever ``provision`` says.

        Args:
            model: Embedding model recorded for the collection. Defaults to
                the service's ``model``.
            vector_type: Column type. Defaults to the service's
                ``vector_type``.
        """
        model = model or self._require_model("creating a collection")
        if vector_type is not None and vector_type not in self.vector_types:
            raise ValueError(
                f"{type(self).__name__} stores {sorted(self.vector_types)} "
                f"embeddings, not {vector_type!r}."
            )
        async with self._connection() as conn:
            await self._ensure_collection(
                conn,
                collection_name,
                embedding_size,
                model,
                vector_type=vector_type,
            )

    async def drop_collection(self, collection_name: str) -> None:
        """Drop a collection: its table and registry entry."""
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                await self._commit(conn)
                return
            await self._drop_collection_table(conn, info)
            await self._delete_registry_row(conn, collection_name)
            await self._commit(conn)
        self._collections.pop(collection_name, None)
        logger.info(f"Dropped RAG collection '{collection_name}'.")

    async def list_collections(self) -> list[dict]:
        """List registered collections with entry counts.

        Returns an empty list when the registry does not exist; listing never
        creates it.
        """
        async with self._connection() as conn:
            if not await self._registry_available(conn):
                await self._commit(conn)
                return []
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

    # Embedding

    async def _compute_embeddings(
        self, texts: list[str], model: str, stats_receiver: Any = None
    ) -> list[list[float]]:
        """Embed ``texts`` with ``model`` and report the call's statistics.

        Normalisation is opt-in: only a service constructed with a normaliser
        normalises, and then on both the index and the query side. The call is
        reported to ``stats_receiver`` or, failing that, the service's own;
        the service never writes statistics anywhere itself.
        """
        embeddings, stats = await self._client_for(model).compute_embeddings(
            texts=texts,
            normalize=self.normalizer is not None,
            normalizer=self.normalizer,
        )
        receiver = stats_receiver or self.stats_receiver
        if receiver is not None and stats is not None:
            receiver.receive_model_stats(stats)
        return embeddings

    @staticmethod
    def _check_lengths(
        texts: list[str], metadata_list: list[dict], source_ids: Optional[list[str]]
    ) -> None:
        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )
        if source_ids is not None and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

    @staticmethod
    def _build_rows(
        info: CollectionInfo,
        texts: list[str],
        metadata_list: list[dict],
        embeddings: list[list[float]],
        source_ids: Optional[list[str]],
    ) -> list[dict]:
        now = datetime.now(timezone.utc)
        source_ids = source_ids if source_ids is not None else ["default"] * len(texts)
        return [
            {
                "id": uuid4(),
                "model": info.model,
                "collection_name": info.name,
                "source_id": source_id,
                "content": content,
                "embedding_size": info.embedding_size,
                "embedding": list(emb),
                "rag_metadata": meta,
                "created_at": now,
                "updated_at": now,
            }
            for content, meta, emb, source_id in zip(
                texts, metadata_list, embeddings, source_ids, strict=True
            )
        ]

    async def _collection_for_writing(
        self, conn, collection_name: str, texts: list[str], stats_receiver: Any
    ) -> tuple[Optional[CollectionInfo], list[list[float]]]:
        """Embed ``texts`` for a collection, creating the collection if needed.

        The registry is read and the read transaction ended before the
        embedding call, so no transaction is held open across the network. A
        collection that cannot be created is refused before anything is
        embedded, and an empty ``texts`` needs neither a collection nor a
        model.
        """
        info = await self._load_collection(conn, collection_name)
        await self._commit(conn)
        if not texts:
            return info, []
        if info is None and not self.provision:
            raise RuntimeError(
                f"RAG collection '{collection_name}' does not exist, and this "
                f"{type(self).__name__} was created with provision=False, so it "
                f"issues no DDL. Create it with create_collection() from a role "
                f"that may."
            )
        model = info.model if info else self._require_model("creating a collection")
        embeddings = await self._compute_embeddings(texts, model, stats_receiver)
        dim = len(embeddings[0])
        if info is None:
            info = await self._ensure_collection(conn, collection_name, dim, model)
        else:
            self._check_dimension(info, dim)
        return info, embeddings

    # Indexing

    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
        source_id: str = "default",
        *,
        stats_receiver: Any = None,
    ) -> dict:
        """Index a single text blob with metadata. Returns the created row dict."""
        return (
            await self.index_batch(
                texts=[text],
                metadata_list=[source_metadata or {}],
                collection_name=collection_name,
                source_ids=[source_id],
                stats_receiver=stats_receiver,
            )
        )[0]

    async def index_batch(
        self,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        collection_name: str = "default",
        *,
        stats_receiver: Any = None,
    ) -> list[dict]:
        """
        Index multiple text items in a single batch.

        A missing collection is provisioned on first use, taking its embedding
        dimension from the computed embeddings and its model from the
        service's ``model``; an existing one is embedded with its own model.

        Args:
            stats_receiver: Receives the embedding call's statistics, instead
                of the service's own receiver.

        Returns:
            list[dict]: Created rows (id, model, collection_name, source_id,
                content, embedding_size, embedding, rag_metadata, created_at,
                updated_at).
        """
        if not texts:
            return []
        self._check_lengths(texts, metadata_list, source_ids)

        async with self._connection() as conn:
            info, embeddings = await self._collection_for_writing(
                conn, collection_name, texts, stats_receiver
            )
            rows = self._build_rows(info, texts, metadata_list, embeddings, source_ids)
            await self._insert_rows(conn, info, rows)
            await self._commit(conn)
            return rows

    async def replace(
        self,
        collection_name: str,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        *,
        match: Optional[dict[str, MetadataValue]] = None,
        source_id: Optional[str] = None,
        stats_receiver: Any = None,
    ) -> list[dict]:
        """Replace the rows selected by ``match`` or ``source_id`` with new ones.

        The texts are embedded first, outside any transaction; the old rows
        are then deleted and the new ones inserted in one transaction, so a
        failure leaves the old rows in place rather than a document half
        indexed. Re-indexing one page of a site is
        ``replace(name, chunks, metas, match={"page_id": page})``. An empty
        ``texts`` deletes the selected rows.

        Args:
            match: Rows whose metadata has every key of ``match`` equal.
            source_id: Rows with this source identifier.
            stats_receiver: Receives the embedding call's statistics.

        Returns:
            The created rows, as :meth:`index_batch` returns them.
        """
        if match is None and source_id is None:
            raise ValueError("replace() needs match= or source_id= to select rows.")
        if match is not None:
            validate_metadata_match(match)
        self._check_lengths(texts, metadata_list, source_ids)

        async with self._connection() as conn:
            info, embeddings = await self._collection_for_writing(
                conn, collection_name, texts, stats_receiver
            )
            if info is None:
                return []
            rows = self._build_rows(info, texts, metadata_list, embeddings, source_ids)
            try:
                if match is not None:
                    await self._delete_rows_by_metadata(conn, info, match)
                if source_id is not None:
                    await self._delete_rows_by_source_ids(conn, info, [source_id])
                if rows:
                    await self._insert_rows(conn, info, rows)
                await self._commit(conn)
            except BaseException:
                await self._rollback(conn)
                raise
            return rows

    # Deletion

    async def _collections_for(
        self, conn, collection_name: Optional[str]
    ) -> list[CollectionInfo]:
        """The named collection, or every registered one when ``None``."""
        if collection_name is not None:
            info = await self._load_collection(conn, collection_name)
            return [info] if info else []
        if not await self._registry_available(conn):
            return []
        return [
            await self._load_collection(conn, entry.name)
            for entry in await self._list_registry(conn)
        ]

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
        await self.delete_many([item_id], collection_name=collection_name)

    async def delete_many(
        self, ids: list[UUID], collection_name: Optional[str] = None
    ) -> None:
        """Delete several items by identifier, in one statement per collection.

        Args:
            ids: Identifiers of the items to delete.
            collection_name: Collection the items belong to. If omitted, all
                registered collections are searched.
        """
        if not ids:
            return
        async with self._connection() as conn:
            for info in await self._collections_for(conn, collection_name):
                await self._delete_rows(conn, info, list(ids))
            await self._commit(conn)

    async def delete_by_source_id(
        self,
        collection_name: str,
        source_id: Union[str, list[str]],
    ) -> None:
        """Delete all items in a collection matching the source identifier(s)."""
        source_ids = [source_id] if isinstance(source_id, str) else source_id
        if not source_ids:
            return
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                await self._commit(conn)
                return
            await self._delete_rows_by_source_ids(conn, info, source_ids)
            await self._commit(conn)

    async def delete_by_metadata(
        self, collection_name: str, match: dict[str, MetadataValue]
    ) -> None:
        """Delete the items whose metadata has every key of ``match`` equal.

        ``match`` holds top-level keys and scalar values:
        ``{"page_id": "p-17"}`` deletes one page's chunks when each chunk was
        indexed with its page in the metadata.
        """
        validate_metadata_match(match)
        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            if info is None:
                await self._commit(conn)
                return
            await self._delete_rows_by_metadata(conn, info, match)
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
        *,
        min_similarity: Optional[float] = None,
        stats_receiver: Any = None,
    ) -> list[RagServiceResult]:
        """
        Query one collection for similarities to the input text.

        ``collection_name`` defaults to ``"default"`` — with table-per-collection
        storage there is no cross-collection search; query each collection
        explicitly if needed. See :meth:`query_batch` for the other arguments.
        """
        results = await self.query_batch(
            texts=[text],
            top_k=top_k,
            collection_name=collection_name,
            source_ids=source_ids,
            keep_best=keep_best,
            include_content=include_content,
            min_similarity=min_similarity,
            stats_receiver=stats_receiver,
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
        *,
        min_similarity: Optional[float] = None,
        stats_receiver: Any = None,
    ) -> list[list[RagServiceResult]]:
        """
        Query one collection for similarities to multiple input texts.

        The embeddings for all query texts are computed in a single call, with
        the model the collection was built with. Nothing is embedded when the
        answer is known to be empty: an empty ``source_ids`` or a collection
        that does not exist.

        Args:
            source_ids: ``None`` searches the whole collection; a list
                restricts the search to those sources, and an empty list
                matches nothing.
            min_similarity: Drop results whose ``similarity`` is lower. The
                filter applies after ``top_k``, so fewer than ``top_k``
                results may come back.
            stats_receiver: Receives the embedding call's statistics, instead
                of the service's own receiver.
        """
        if not texts:
            return []
        if source_ids is not None and not source_ids:
            return [[] for _ in texts]
        collection_name = collection_name or "default"

        async with self._connection() as conn:
            info = await self._load_collection(conn, collection_name)
            await self._commit(conn)
            if info is None:
                return [[] for _ in texts]
            embeddings = await self._compute_embeddings(
                texts, info.model, stats_receiver
            )
            batches = await self._scan(
                conn, info, embeddings, top_k, source_ids, keep_best
            )
            await self._commit(conn)

        results = [
            [self._map_row(row, info, index, include_content) for row in rows]
            for index, rows in enumerate(batches)
        ]
        if min_similarity is not None:
            results = [
                [hit for hit in hits if hit.similarity >= min_similarity]
                for hits in results
            ]
        return results

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
