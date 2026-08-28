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

PostgreSQL (pgvector) backed RAG service — **self-provisioning**.

This backend owns its schema entirely; no Alembic migration set covers it.
It maintains a small registry table (``rag_collections``) plus one table per
collection with a *typed* ``vector(N)`` column, a real HNSW index for the
collection's exact dimension, and a GIN index on the metadata column.
Dropping a collection is ``DROP TABLE``. The registry row carries a
``schema_version`` used for in-code upgrades of collection tables.

All SQL here is raw and therefore bypasses ``schema_translate_map`` — every
statement qualifies the configured schema explicitly.

``RAG_COLLECTION_SCHEMA_VERSION`` is the version of the per-collection table
layout. Bump it when the layout changes and register an upgrade step in
``_COLLECTION_UPGRADES``, which maps a *from_version* to an async callable
``(session, service, collection_info) -> None`` that brings a collection from
``from_version`` to ``from_version + 1``.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional, Union, AsyncContextManager
from uuid import UUID, uuid4

from loguru import logger

# Imported under an alias because several methods take a ``text`` parameter.
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.db import Agent, db_manager
from kavalai.llm_clients.embeddings import make_embedding_client
from kavalai.normalizer import Normalizer
from kavalai.rag.base import BaseRagService, RagServiceResult

RAG_COLLECTION_SCHEMA_VERSION = 1

_COLLECTION_UPGRADES: dict[int, Callable] = {}


def _vector_literal(embedding) -> str:
    """Render an embedding as the ``[x, y, ...]`` text pgvector casts from."""
    return str(list(embedding))


def _parse_vector(value) -> list[float]:
    """Parse a vector read back as ``[x,y,...]`` text; lists pass through."""
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",")]
    return list(value)


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


class PostgresRagService(BaseRagService):
    """
    PostgreSQL (pgvector) backed RAG service with backend-owned DDL.

    Each collection lives in its own table (typed ``vector(N)`` column,
    per-collection HNSW + GIN indexes) registered in ``rag_collections``.
    Collections are provisioned lazily on first index (the embedding dimension
    is taken from the first batch) or explicitly via :meth:`create_collection`.

    All operations are scoped to a single collection (``"default"`` unless
    specified) — ``collection_name`` is a logical handle in the
    :class:`BaseRagService` interface; here it maps to a table.
    """

    capabilities = frozenset({"count_entries", "iter_entries"})

    REGISTRY_TABLE = "rag_collections"

    def __init__(
        self,
        session_maker: Union[
            async_sessionmaker[AsyncSession],
            Callable[[], AsyncContextManager[AsyncSession]],
        ],
        model: Optional[str] = None,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
        schema: Optional[str] = None,
    ):
        """
        Initialize the PostgresRagService.

        Args:
            session_maker: Async session maker or a factory that returns an async
                context manager for the session.
            model (Optional[str]): The embedding model to use. May be ``None``
                for export/stats-only usage (anything that computes embeddings
                will then fail).
            agent (Optional[Agent]): Optional Agent object to associate with this service.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.
            schema (Optional[str]): Schema the RAG tables live in. All backend SQL is
                raw and qualifies this schema explicitly. ``None`` uses the
                connection default (Postgres: ``public``).
        """
        self.session_maker = session_maker
        self.model = model
        self.agent = agent
        self.normalizer = normalizer
        self.schema = schema
        self._embedding_client = None
        self._registry_ready = False
        self._collections: dict[str, CollectionInfo] = {}

    @property
    def embedding_client(self):
        """Embedding client, created lazily so model-less usage works."""
        if self._embedding_client is None:
            if not self.model:
                raise ValueError(
                    "This PostgresRagService was created without an embedding "
                    "model; indexing and querying require one."
                )
            self._embedding_client = make_embedding_client(self.model)
        return self._embedding_client

    @embedding_client.setter
    def embedding_client(self, client) -> None:
        self._embedding_client = client

    @classmethod
    def from_uri(
        cls,
        uri: str,
        model: str,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
        schema: Optional[str] = None,
    ) -> "PostgresRagService":
        """Create a PostgresRagService from a database URI."""
        session_maker = db_manager.get_sessionmaker(uri=uri, schema=schema)
        return cls(session_maker, model, agent, normalizer, schema=schema)

    @classmethod
    def from_session_maker(
        cls,
        session_maker: async_sessionmaker[AsyncSession],
        model: str,
        agent: Optional[Agent] = None,
        normalizer: Optional[Normalizer] = None,
        schema: Optional[str] = None,
    ) -> "PostgresRagService":
        """Create a PostgresRagService from a session maker."""
        return cls(session_maker, model, agent, normalizer, schema=schema)

    # Naming / SQL helpers

    def _qualified(self, table_name: str) -> str:
        """Schema-qualified, quoted table reference for raw SQL."""
        if self.schema:
            return f'"{self.schema}"."{table_name}"'
        return f'"{table_name}"'

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

    # Provisioning

    async def _ensure_registry(self, session: AsyncSession) -> None:
        """Create the pgvector extension and the registry table if needed."""
        if self._registry_ready:
            return
        await session.execute(
            sql_text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        )
        await session.execute(
            sql_text(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qualified(self.REGISTRY_TABLE)} (
                    name TEXT PRIMARY KEY,
                    table_name TEXT UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    embedding_size INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        await session.commit()
        self._registry_ready = True

    async def _load_collection(
        self, session: AsyncSession, collection_name: str
    ) -> Optional[CollectionInfo]:
        """Fetch a collection's registry entry (cached), upgrading if stale."""
        cached = self._collections.get(collection_name)
        if cached is not None:
            return cached

        await self._ensure_registry(session)
        row = (
            await session.execute(
                sql_text(
                    f"SELECT name, table_name, model, embedding_size, schema_version "
                    f"FROM {self._qualified(self.REGISTRY_TABLE)} WHERE name = :name"
                ).bindparams(name=collection_name)
            )
        ).first()
        if row is None:
            return None

        info = CollectionInfo(
            name=row.name,
            table_name=row.table_name,
            model=row.model,
            embedding_size=row.embedding_size,
            schema_version=row.schema_version,
        )
        await self._upgrade_collection(session, info)
        self._collections[collection_name] = info
        return info

    async def _upgrade_collection(
        self, session: AsyncSession, info: CollectionInfo
    ) -> None:
        """Bring a collection table up to ``RAG_COLLECTION_SCHEMA_VERSION``."""
        if info.schema_version > RAG_COLLECTION_SCHEMA_VERSION:
            raise ValueError(
                f"Collection '{info.name}' has schema_version "
                f"{info.schema_version}, newer than this library supports "
                f"({RAG_COLLECTION_SCHEMA_VERSION}). Upgrade kavalai."
            )
        while info.schema_version < RAG_COLLECTION_SCHEMA_VERSION:
            upgrade = _COLLECTION_UPGRADES.get(info.schema_version)
            if upgrade is None:
                raise ValueError(
                    f"No upgrade step registered from collection schema_version "
                    f"{info.schema_version} (collection '{info.name}')."
                )
            logger.info(
                f"Upgrading RAG collection '{info.name}' from schema_version "
                f"{info.schema_version} to {info.schema_version + 1}."
            )
            await upgrade(session, self, info)
            info.schema_version += 1
            await session.execute(
                sql_text(
                    f"UPDATE {self._qualified(self.REGISTRY_TABLE)} "
                    f"SET schema_version = :version WHERE name = :name"
                ).bindparams(version=info.schema_version, name=info.name)
            )
            await session.commit()

    async def _ensure_collection(
        self, session: AsyncSession, collection_name: str, embedding_size: int
    ) -> CollectionInfo:
        """Get a collection, creating its table on first use."""
        info = await self._load_collection(session, collection_name)
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
                "This PostgresRagService was created without an embedding "
                "model; creating collections requires one."
            )

        table_name = self.table_name_for_collection(collection_name)
        qualified = self._qualified(table_name)
        await session.execute(
            sql_text(
                f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    id UUID PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    content TEXT,
                    embedding public.vector({embedding_size}),
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
        )
        await session.execute(
            sql_text(
                f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_source_id" '
                f"ON {qualified} (source_id)"
            )
        )
        await session.execute(
            sql_text(
                f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_metadata" '
                f"ON {qualified} USING gin (metadata)"
            )
        )
        await session.execute(
            sql_text(
                f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_embedding" '
                f"ON {qualified} USING hnsw (embedding public.vector_cosine_ops)"
            )
        )
        await session.execute(
            sql_text(
                f"INSERT INTO {self._qualified(self.REGISTRY_TABLE)} "
                f"(name, table_name, model, embedding_size, schema_version) "
                f"VALUES (:name, :table_name, :model, :embedding_size, :version) "
                f"ON CONFLICT (name) DO NOTHING"
            ).bindparams(
                name=collection_name,
                table_name=table_name,
                model=self.model,
                embedding_size=embedding_size,
                version=RAG_COLLECTION_SCHEMA_VERSION,
            )
        )
        await session.commit()
        logger.info(
            f"Provisioned RAG collection '{collection_name}' "
            f"(table {table_name}, dim {embedding_size})."
        )
        info = CollectionInfo(
            name=collection_name,
            table_name=table_name,
            model=self.model,
            embedding_size=embedding_size,
            schema_version=RAG_COLLECTION_SCHEMA_VERSION,
        )
        self._collections[collection_name] = info
        return info

    async def create_collection(
        self, collection_name: str, embedding_size: int
    ) -> None:
        """Explicitly provision a collection with a known embedding dimension."""
        async with self.session_maker() as session:
            await self._ensure_collection(session, collection_name, embedding_size)

    async def drop_collection(self, collection_name: str) -> None:
        """Drop a collection: its table and registry entry."""
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                return
            await session.execute(
                sql_text(f"DROP TABLE IF EXISTS {self._qualified(info.table_name)}")
            )
            await session.execute(
                sql_text(
                    f"DELETE FROM {self._qualified(self.REGISTRY_TABLE)} "
                    f"WHERE name = :name"
                ).bindparams(name=collection_name)
            )
            await session.commit()
        self._collections.pop(collection_name, None)
        logger.info(f"Dropped RAG collection '{collection_name}'.")

    async def list_collections(self) -> list[dict]:
        """List registered collections with entry counts."""
        async with self.session_maker() as session:
            await self._ensure_registry(session)
            rows = (
                await session.execute(
                    sql_text(
                        f"SELECT name, table_name, model, embedding_size, "
                        f"schema_version FROM "
                        f"{self._qualified(self.REGISTRY_TABLE)} ORDER BY name"
                    )
                )
            ).all()
            collections = []
            for row in rows:
                count = (
                    await session.execute(
                        sql_text(
                            f"SELECT count(*) FROM {self._qualified(row.table_name)}"
                        )
                    )
                ).scalar()
                collections.append(
                    {
                        "name": row.name,
                        "model": row.model,
                        "embedding_size": row.embedding_size,
                        "schema_version": row.schema_version,
                        "count": count or 0,
                    }
                )
            await session.commit()
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
                content, embedding_size, rag_metadata, created_at, updated_at).
        """
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        if source_ids and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

        async with self.session_maker() as session:
            embeddings, stats = await self.embedding_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            dim = len(embeddings[0])
            info = await self._ensure_collection(session, collection_name, dim)

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

            insert_sql = sql_text(
                f"INSERT INTO {self._qualified(info.table_name)} "
                f"(id, source_id, content, embedding, metadata, created_at, updated_at) "
                f"VALUES (:id, :source_id, :content, CAST(:embedding AS vector), "
                f"CAST(:metadata AS jsonb), :created_at, :updated_at)"
            )
            await session.execute(
                insert_sql,
                [
                    {
                        "id": row["id"],
                        "source_id": row["source_id"],
                        "content": row["content"],
                        "embedding": _vector_literal(row["embedding"]),
                        "metadata": json.dumps(row["rag_metadata"]),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ],
            )
            await session.commit()
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
        async with self.session_maker() as session:
            if collection_name is not None:
                info = await self._load_collection(session, collection_name)
                infos = [info] if info else []
            else:
                infos = [
                    await self._load_collection(session, c["name"])
                    for c in await self.list_collections()
                ]
            for info in infos:
                await session.execute(
                    sql_text(
                        f"DELETE FROM {self._qualified(info.table_name)} WHERE id = :id"
                    ).bindparams(id=item_id)
                )
            await session.commit()

    async def delete_by_source_id(
        self,
        collection_name: str,
        source_id: Union[str, list[str]],
    ) -> None:
        """Delete all items in a collection matching the source identifier(s)."""
        source_ids = [source_id] if isinstance(source_id, str) else source_id
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                return
            await session.execute(
                sql_text(
                    f"DELETE FROM {self._qualified(info.table_name)} "
                    f"WHERE source_id = ANY(:source_ids)"
                ).bindparams(source_ids=source_ids)
            )
            await session.commit()

    # Querying

    def _map_row(
        self,
        row,
        info: CollectionInfo,
        query_index=None,
        include_content: bool = True,
    ) -> RagServiceResult:
        return RagServiceResult(
            id=row.id,
            model=info.model,
            collection_name=info.name,
            source_id=row.source_id,
            content=row.content if include_content else None,
            embedding_size=info.embedding_size,
            rag_metadata=row.metadata or {},
            similarity=1.0 - float(row.distance) if row.distance is not None else 0.0,
            created_at=row.created_at,
            updated_at=row.updated_at,
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
        Query one collection for similarities to multiple input texts in a
        single database call (CROSS JOIN LATERAL over an unnested vector array).
        """
        if not texts:
            return []
        collection_name = collection_name or "default"

        async with self.session_maker() as session:
            embeddings, stats = await self.embedding_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()  # persist embedding stats
                return [[] for _ in texts]

            cte_sql, params = self._build_batch_query_cte(
                info=info,
                embeddings=embeddings,
                top_k=top_k,
                source_ids=source_ids,
                keep_best=keep_best,
            )
            query_sql = (
                f"WITH {cte_sql} SELECT * FROM rag_results "
                f"ORDER BY query_idx ASC, distance ASC"
            )
            result = await session.execute(sql_text(query_sql).bindparams(**params))
            rows = result.all()
            await session.commit()

            results_by_query: dict[int, list[RagServiceResult]] = {
                i: [] for i in range(len(texts))
            }
            for row in rows:
                idx = int(row.query_idx) - 1
                results_by_query[idx].append(
                    self._map_row(row, info, idx, include_content)
                )
            return [results_by_query[i] for i in range(len(texts))]

    def build_batch_query_cte(
        self,
        embeddings: list[list[float]],
        top_k: int,
        collection_name: Optional[str] = None,
        source_filter_sql: Optional[str] = None,
        keep_best: bool = False,
        info: Optional[CollectionInfo] = None,
    ) -> tuple[str, dict]:
        """
        Build a ``rag_results`` CTE for batch vector search, embeddable in
        larger queries. Requires the collection to exist (pass ``info`` or a
        ``collection_name`` previously loaded via any service call).
        """
        if info is None:
            info = self._collections.get(collection_name or "default")
        if info is None:
            raise ValueError(
                f"Unknown RAG collection {collection_name or 'default'!r}; "
                f"load or create it first."
            )
        return self._build_batch_query_cte(
            info=info,
            embeddings=embeddings,
            top_k=top_k,
            source_filter_sql=source_filter_sql,
            keep_best=keep_best,
        )

    def _build_batch_query_cte(
        self,
        info: CollectionInfo,
        embeddings: list[list[float]],
        top_k: int,
        source_ids: Optional[list[str]] = None,
        source_filter_sql: Optional[str] = None,
        keep_best: bool = False,
    ) -> tuple[str, dict]:
        params: dict = {"top_k": top_k}
        vector_parts = []
        for i, embedding in enumerate(embeddings):
            params[f"vector_{i}"] = _vector_literal(embedding)
            vector_parts.append(f"CAST(:vector_{i} AS public.vector)")
        vector_array = f"ARRAY[{', '.join(vector_parts)}]"

        where_clauses = ["TRUE"]
        if source_ids:
            where_clauses.append("rag_index.source_id = ANY(:source_ids)")
            params["source_ids"] = source_ids
        if source_filter_sql:
            where_clauses.append(f"({source_filter_sql})")
        where_clause = " AND ".join(where_clauses)

        # For keep_best, scan a wider window than top_k so deduplication by
        # source_id still leaves up to top_k distinct sources.
        inner_limit = ":scan_k" if keep_best else ":top_k"
        if keep_best:
            params["scan_k"] = max(top_k * 10, 100)

        # The collection table is aliased as ``rag_index`` so caller-supplied
        # filters (source_filter_sql) can reference a stable name.
        lateral = f"""
            SELECT
                id, source_id, content, metadata, created_at, updated_at,
                (embedding <=> v.query_vector) as distance
            FROM {self._qualified(info.table_name)} AS rag_index
            WHERE {where_clause}
            ORDER BY (embedding <=> v.query_vector) ASC
            LIMIT {inner_limit}
        """

        if keep_best:
            # DISTINCT ON keeps the best chunk per (query, source); the ranked
            # outer query then caps at top_k distinct sources per query.
            cte_sql = f"""rag_results AS (
                SELECT * FROM (
                    SELECT
                        dedup.*,
                        row_number() OVER (
                            PARTITION BY dedup.query_idx
                            ORDER BY dedup.distance ASC
                        ) AS rn
                    FROM (
                        SELECT DISTINCT ON (v.query_idx, results.source_id)
                            results.*,
                            (1.0 - results.distance) as similarity,
                            v.query_idx
                        FROM unnest({vector_array}) WITH ORDINALITY AS v(query_vector, query_idx)
                        CROSS JOIN LATERAL ({lateral}) AS results
                        ORDER BY v.query_idx ASC, results.source_id, results.distance ASC
                    ) AS dedup
                ) AS ranked
                WHERE ranked.rn <= :top_k
            )"""
        else:
            cte_sql = f"""rag_results AS (
                SELECT
                    results.*,
                    (1.0 - results.distance) as similarity,
                    v.query_idx
                FROM unnest({vector_array}) WITH ORDINALITY AS v(query_vector, query_idx)
                CROSS JOIN LATERAL ({lateral}) AS results
                ORDER BY v.query_idx ASC, results.distance ASC
            )"""
        return cte_sql, params

    async def batch_query_with_join(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        join_table: Optional[str] = None,
        join_condition: Optional[str] = None,
        join_columns: Optional[list[str]] = None,
        additional_where: Optional[str] = None,
        keep_best: bool = False,
    ) -> list[list[dict]]:
        """
        Query one collection and join with another table in a single SQL query.

        ``join_condition`` references the CTE alias ``r`` (e.g.
        ``"p.id::text = r.source_id"``); ``additional_where`` filters the joined
        table. Inside the vector-search CTE the collection table is aliased as
        ``rag_index`` for source filters.
        """
        if not texts:
            return []
        collection_name = collection_name or "default"

        async with self.session_maker() as session:
            embeddings, stats = await self.embedding_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()
                return [[] for _ in texts]

            source_filter = None
            if join_table and additional_where:
                source_filter = f"""
                    EXISTS (
                        SELECT 1 FROM {join_table}
                        WHERE {join_condition.replace("r.source_id", "rag_index.source_id")}
                        AND {additional_where}
                    )
                """

            cte_sql, params = self._build_batch_query_cte(
                info=info,
                embeddings=embeddings,
                top_k=top_k,
                source_filter_sql=source_filter,
                keep_best=keep_best,
            )

            select_columns = [
                "r.id",
                "r.source_id",
                "r.content",
                "r.similarity",
                "r.query_idx",
            ]
            if join_columns:
                select_columns.extend(join_columns)
            select_clause = ", ".join(select_columns)

            join_sql = ""
            if join_table and join_condition:
                join_sql = f"JOIN {join_table} ON {join_condition}"
            query_sql = f"""
                WITH {cte_sql}
                SELECT {select_clause}
                FROM rag_results r
                {join_sql}
                ORDER BY r.query_idx, r.similarity DESC
            """

            result = await session.execute(sql_text(query_sql).bindparams(**params))
            rows = result.all()
            await session.commit()

            results_by_query: dict[int, list[dict]] = {i: [] for i in range(len(texts))}
            for row in rows:
                row_dict = dict(row._mapping)
                query_idx = row_dict.pop("query_idx") - 1
                results_by_query[query_idx].append(row_dict)
            return [results_by_query[i] for i in range(len(texts))]

    async def compute_similarity_matrix(
        self,
        texts: list[str],
        source_ids: list[str],
        method: str = "min",
        collection_name: str = "default",
    ) -> list[list[float]]:
        """
        Compute a similarity matrix between texts and source identifiers within
        one collection, in a single database query.
        """
        if not texts or not source_ids:
            return [[0.0 for _ in source_ids] for _ in texts]

        async with self.session_maker() as session:
            embeddings, stats = await self.embedding_client.compute_embeddings(
                texts=texts,
                normalizer=self.normalizer,
            )
            session.add(stats)

            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()
                return [[0.0 for _ in source_ids] for _ in texts]

            agg = "min" if method == "min" else "avg"
            params: dict = {"source_ids": source_ids}
            dist_cols = []
            for i, emb in enumerate(embeddings):
                params[f"vector_{i}"] = _vector_literal(emb)
                dist_cols.append(
                    f"{agg}(embedding <=> CAST(:vector_{i} AS public.vector)) "
                    f"AS dist_{i}"
                )
            query_sql = (
                f"SELECT source_id, {', '.join(dist_cols)} "
                f"FROM {self._qualified(info.table_name)} "
                f"WHERE source_id = ANY(:source_ids) GROUP BY source_id"
            )
            rows = (
                await session.execute(sql_text(query_sql).bindparams(**params))
            ).all()
            await session.commit()

            source_id_to_idx = {sid: i for i, sid in enumerate(source_ids)}
            matrix = [[0.0 for _ in source_ids] for _ in texts]
            for row in rows:
                s_idx = source_id_to_idx.get(row.source_id)
                if s_idx is None:
                    continue
                for t_idx in range(len(texts)):
                    dist = getattr(row, f"dist_{t_idx}")
                    matrix[t_idx][s_idx] = (
                        1.0 - float(dist) if dist is not None else 0.0
                    )
            return matrix

    async def learn_normalizer(
        self, collection_name: Optional[str] = None
    ) -> Normalizer:
        """Learn a centering normalizer from one collection's embeddings."""
        collection_name = collection_name or "default"
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                raise Exception("No embeddings found in RAG index.")
            mean_vector = (
                await session.execute(
                    sql_text(
                        f"SELECT avg(embedding) FROM {self._qualified(info.table_name)}"
                    )
                )
            ).scalar()
            if mean_vector is None:
                raise Exception("No embeddings found in RAG index.")
            await session.commit()
            return Normalizer(center_vector=_parse_vector(mean_vector))

    # Bulk export

    async def iter_entries(
        self, collection_name: str, batch_size: int = 500
    ) -> AsyncIterator[dict]:
        """
        Iterate all entries of a collection (including embeddings) in stable
        ``id`` order using keyset pagination. Yields dicts with keys:
        id, source_id, content, embedding, rag_metadata, created_at, updated_at.
        """
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()
                return
            params: dict = {"batch_size": batch_size}
            while True:
                where = "WHERE id > :last_id" if "last_id" in params else ""
                query_sql = (
                    f"SELECT id, source_id, content, embedding::text AS embedding, "
                    f"metadata, created_at, updated_at "
                    f"FROM {self._qualified(info.table_name)} {where} "
                    f"ORDER BY id ASC LIMIT :batch_size"
                )
                rows = (
                    await session.execute(sql_text(query_sql).bindparams(**params))
                ).all()
                if not rows:
                    # Close the read transaction (see count_entries).
                    await session.commit()
                    return
                for row in rows:
                    yield {
                        "id": row.id,
                        "source_id": row.source_id,
                        "content": row.content,
                        "embedding": _parse_vector(row.embedding),
                        "rag_metadata": row.metadata or {},
                        "created_at": row.created_at,
                        "updated_at": row.updated_at,
                    }
                params["last_id"] = rows[-1].id

    async def count_entries(self, collection_name: str) -> int:
        """Number of entries in a collection (0 if it doesn't exist)."""
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()
                return 0
            count = (
                await session.execute(
                    sql_text(f"SELECT count(*) FROM {self._qualified(info.table_name)}")
                )
            ).scalar() or 0
            # Close the read transaction: shared-session factories (e.g. the
            # backoffice) would otherwise pin a pooled connection.
            await session.commit()
            return count

    async def get_embeddings_by_ids(
        self, collection_name: str, ids: list[UUID]
    ) -> dict[UUID, list[float]]:
        """Fetch embeddings for specific entry ids within a collection."""
        if not ids:
            return {}
        async with self.session_maker() as session:
            info = await self._load_collection(session, collection_name)
            if info is None:
                await session.commit()
                return {}
            rows = (
                await session.execute(
                    sql_text(
                        f"SELECT id, embedding::text AS embedding "
                        f"FROM {self._qualified(info.table_name)} "
                        f"WHERE id = ANY(:ids)"
                    ).bindparams(ids=ids)
                )
            ).all()
            await session.commit()
            return {row.id: _parse_vector(row.embedding) for row in rows}
