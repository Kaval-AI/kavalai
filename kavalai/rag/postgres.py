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

PostgreSQL (pgvector) backed RAG service.

This backend owns its schema; no Alembic migration set covers it. The storage
model is the one described in :mod:`kavalai.rag.collections`: a
``rag_collections`` registry plus one table per collection with a typed
``vector(N)`` or ``halfvec(N)`` column, an HNSW index for the collection's
exact dimension, and a GIN index on the metadata column. Dropping a collection
is ``DROP TABLE``. With ``provision=False`` the service issues no DDL and the
tables are created by :meth:`~CollectionRagService.ensure_registry` and
:meth:`~CollectionRagService.create_collection` from a role that may.

All SQL here is raw and therefore bypasses ``schema_translate_map`` — every
statement qualifies the configured schema explicitly, and the vector types are
qualified with ``public``, where the extension is created.

``HALFVEC_MIN_VERSION`` and ``ITERATIVE_SCAN_MIN_VERSION`` are the pgvector
releases that introduced ``halfvec`` and iterative index scans.
``_COLLECTION_UPGRADES`` maps a *from_version* to an async callable
``(session, service, collection_info) -> None`` that brings a collection from
``from_version`` to ``from_version + 1``; see
``RAG_COLLECTION_SCHEMA_VERSION`` in :mod:`kavalai.rag.collections`.
"""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Callable, Optional, Union
from uuid import UUID

# Imported under an alias because several methods take a ``text`` parameter.
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.db import db_manager
from kavalai.normalizer import Normalizer
from kavalai.rag.collections import (
    RAG_COLLECTION_SCHEMA_VERSION,
    CollectionInfo,
    CollectionRagService,
    MetadataValue,
)

__all__ = [
    "CollectionInfo",
    "PostgresRagService",
    "RAG_COLLECTION_SCHEMA_VERSION",
]

_COLLECTION_UPGRADES: dict[int, Callable] = {}

HALFVEC_MIN_VERSION = (0, 7)
ITERATIVE_SCAN_MIN_VERSION = (0, 8)


def _vector_literal(embedding) -> str:
    """Render an embedding as the ``[x, y, ...]`` text pgvector casts from."""
    return str(list(embedding))


def _parse_vector(value) -> list[float]:
    """Parse a vector read back as ``[x,y,...]`` text; lists pass through."""
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",")]
    return list(value)


def _parse_version(value: Optional[str]) -> tuple[int, ...]:
    """``"0.8.0"`` -> ``(0, 8, 0)``; anything unparsable is ``(0,)``."""
    parts = []
    for part in (value or "").split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts) or (0,)


class PostgresRagService(CollectionRagService):
    """
    PostgreSQL (pgvector) backed RAG service with backend-owned DDL.

    Each collection lives in its own table (typed ``vector(N)`` or
    ``halfvec(N)`` column, per-collection HNSW and GIN indexes) registered in
    ``rag_collections``. Collections are provisioned on first index (the
    embedding dimension is taken from the first batch) or explicitly via
    :meth:`~kavalai.rag.collections.CollectionRagService.create_collection`.

    ``vector_type="halfvec"`` stores new collections' embeddings as 16-bit
    floats: half the memory, and HNSW indexes for up to 4,000 dimensions
    instead of 2,000, so a 3,072-dimension model can be indexed at all. It
    needs pgvector 0.7 or later. An existing collection keeps the type it was
    created with.

    A query restricted by ``source_ids`` enables pgvector's iterative index
    scans (0.8 or later) for its transaction. Without them an HNSW search
    filters its candidates after the search, so a filter whose rows lie away
    from the query can return fewer than ``top_k`` rows, or none.
    """

    collection_upgrades = _COLLECTION_UPGRADES
    vector_types = frozenset({"vector", "halfvec"})

    def __init__(
        self,
        session_maker: Union[
            async_sessionmaker[AsyncSession],
            Callable[[], AsyncContextManager[AsyncSession]],
        ],
        model: Optional[str] = None,
        *,
        normalizer: Optional[Normalizer] = None,
        schema: Optional[str] = None,
        provision: bool = True,
        stats_receiver: Any = None,
        vector_type: str = "vector",
    ):
        """
        Initialize the PostgresRagService.

        Args:
            session_maker: Async session maker or a factory that returns an async
                context manager for the session.
            model (Optional[str]): Embedding model for collections this service
                creates. Existing collections are embedded with their own.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.
            schema (Optional[str]): Schema the RAG tables live in. All backend SQL is
                raw and qualifies this schema explicitly. ``None`` uses the
                connection default (Postgres: ``public``).
            provision (bool): Whether the service may create the registry and
                collection tables itself. See
                :class:`~kavalai.rag.collections.CollectionRagService`.
            stats_receiver: Receives each embedding call's statistics.
            vector_type (str): ``"vector"`` or ``"halfvec"`` for new collections.
        """
        super().__init__(
            model=model,
            normalizer=normalizer,
            provision=provision,
            stats_receiver=stats_receiver,
            vector_type=vector_type,
        )
        self.session_maker = session_maker
        self.schema = schema
        self._pgvector_version: Optional[tuple[int, ...]] = None

    @classmethod
    def from_uri(
        cls,
        uri: str,
        model: Optional[str] = None,
        *,
        schema: Optional[str] = None,
        **options: Any,
    ) -> "PostgresRagService":
        """Create a PostgresRagService from a database URI.

        ``options`` are the constructor's keyword arguments.
        """
        session_maker = db_manager.get_sessionmaker(uri=uri, schema=schema)
        return cls(session_maker, model, schema=schema, **options)

    @classmethod
    def from_session_maker(
        cls,
        session_maker: async_sessionmaker[AsyncSession],
        model: Optional[str] = None,
        **options: Any,
    ) -> "PostgresRagService":
        """Create a PostgresRagService from a session maker."""
        return cls(session_maker, model, **options)

    def _qualified(self, table_name: str) -> str:
        """Schema-qualified, quoted table reference for raw SQL."""
        if self.schema:
            return f'"{self.schema}"."{table_name}"'
        return f'"{table_name}"'

    async def _pgvector(self, session: AsyncSession) -> tuple[int, ...]:
        """The installed pgvector version, read once per service."""
        if self._pgvector_version is None:
            version = (
                await session.execute(
                    sql_text(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                )
            ).scalar()
            self._pgvector_version = _parse_version(version)
        return self._pgvector_version

    # Hooks

    @asynccontextmanager
    async def _connection(self):
        async with self.session_maker() as session:
            yield session

    async def _commit(self, session: AsyncSession) -> None:
        await session.commit()

    async def _rollback(self, session: AsyncSession) -> None:
        await session.rollback()

    async def _registry_exists(self, session: AsyncSession) -> bool:
        return (
            await session.execute(
                sql_text("SELECT to_regclass(:name) IS NOT NULL").bindparams(
                    name=self._qualified(self.REGISTRY_TABLE)
                )
            )
        ).scalar()

    async def _create_registry(self, session: AsyncSession) -> None:
        """Create the pgvector extension and the registry table if needed."""
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

    @staticmethod
    def _info_from_row(row) -> CollectionInfo:
        return CollectionInfo(
            name=row.name,
            table_name=row.table_name,
            model=row.model,
            embedding_size=row.embedding_size,
            schema_version=row.schema_version,
        )

    async def _fetch_registry_row(
        self, session: AsyncSession, name: str
    ) -> Optional[CollectionInfo]:
        row = (
            await session.execute(
                sql_text(
                    f"SELECT name, table_name, model, embedding_size, schema_version "
                    f"FROM {self._qualified(self.REGISTRY_TABLE)} WHERE name = :name"
                ).bindparams(name=name)
            )
        ).first()
        return None if row is None else self._info_from_row(row)

    async def _list_registry(self, session: AsyncSession) -> list[CollectionInfo]:
        rows = (
            await session.execute(
                sql_text(
                    f"SELECT name, table_name, model, embedding_size, "
                    f"schema_version FROM "
                    f"{self._qualified(self.REGISTRY_TABLE)} ORDER BY name"
                )
            )
        ).all()
        return [self._info_from_row(row) for row in rows]

    async def _insert_registry_row(
        self, session: AsyncSession, info: CollectionInfo
    ) -> None:
        await session.execute(
            sql_text(
                f"INSERT INTO {self._qualified(self.REGISTRY_TABLE)} "
                f"(name, table_name, model, embedding_size, schema_version) "
                f"VALUES (:name, :table_name, :model, :embedding_size, :version) "
                f"ON CONFLICT (name) DO NOTHING"
            ).bindparams(
                name=info.name,
                table_name=info.table_name,
                model=info.model,
                embedding_size=info.embedding_size,
                version=info.schema_version,
            )
        )

    async def _set_registry_version(
        self, session: AsyncSession, name: str, version: int
    ) -> None:
        await session.execute(
            sql_text(
                f"UPDATE {self._qualified(self.REGISTRY_TABLE)} "
                f"SET schema_version = :version WHERE name = :name"
            ).bindparams(version=version, name=name)
        )

    async def _delete_registry_row(self, session: AsyncSession, name: str) -> None:
        await session.execute(
            sql_text(
                f"DELETE FROM {self._qualified(self.REGISTRY_TABLE)} WHERE name = :name"
            ).bindparams(name=name)
        )

    async def _create_collection_table(
        self, session: AsyncSession, info: CollectionInfo
    ) -> None:
        if info.vector_type == "halfvec":
            version = await self._pgvector(session)
            if version < HALFVEC_MIN_VERSION:
                raise ValueError(
                    f"halfvec collections need pgvector "
                    f"{'.'.join(map(str, HALFVEC_MIN_VERSION))} or later; the "
                    f"database has {'.'.join(map(str, version))}."
                )
        table_name = info.table_name
        qualified = self._qualified(table_name)
        vector_type = f"public.{info.vector_type}"
        await session.execute(
            sql_text(
                f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    id UUID PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    content TEXT,
                    embedding {vector_type}({info.embedding_size}),
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
                f"ON {qualified} USING hnsw "
                f"(embedding public.{info.vector_type}_cosine_ops)"
            )
        )

    async def _drop_collection_table(
        self, session: AsyncSession, info: CollectionInfo
    ) -> None:
        await session.execute(
            sql_text(f"DROP TABLE IF EXISTS {self._qualified(info.table_name)}")
        )

    async def _prepare_collection(
        self, session: AsyncSession, info: CollectionInfo
    ) -> None:
        """Read the collection's vector type from its embedding column."""
        column_type = (
            await session.execute(
                sql_text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
                ).bindparams(table=self._qualified(info.table_name))
            )
        ).scalar()
        info.vector_type = (
            "halfvec" if column_type and "halfvec" in column_type else "vector"
        )

    async def _count_rows(self, session: AsyncSession, info: CollectionInfo) -> int:
        return (
            await session.execute(
                sql_text(f"SELECT count(*) FROM {self._qualified(info.table_name)}")
            )
        ).scalar() or 0

    async def _insert_rows(
        self, session: AsyncSession, info: CollectionInfo, rows: list[dict]
    ) -> None:
        insert_sql = sql_text(
            f"INSERT INTO {self._qualified(info.table_name)} "
            f"(id, source_id, content, embedding, metadata, created_at, updated_at) "
            f"VALUES (:id, :source_id, :content, "
            f"CAST(:embedding AS public.{info.vector_type}), "
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

    async def _delete_rows(
        self, session: AsyncSession, info: CollectionInfo, item_ids: list[UUID]
    ) -> None:
        await session.execute(
            sql_text(
                f"DELETE FROM {self._qualified(info.table_name)} WHERE id = ANY(:ids)"
            ).bindparams(ids=item_ids)
        )

    async def _delete_rows_by_source_ids(
        self, session: AsyncSession, info: CollectionInfo, source_ids: list[str]
    ) -> None:
        await session.execute(
            sql_text(
                f"DELETE FROM {self._qualified(info.table_name)} "
                f"WHERE source_id = ANY(:source_ids)"
            ).bindparams(source_ids=source_ids)
        )

    async def _delete_rows_by_metadata(
        self,
        session: AsyncSession,
        info: CollectionInfo,
        match: dict[str, MetadataValue],
    ) -> None:
        """``metadata @> :match``, served by the collection's GIN index."""
        await session.execute(
            sql_text(
                f"DELETE FROM {self._qualified(info.table_name)} "
                f"WHERE metadata @> CAST(:match AS jsonb)"
            ).bindparams(match=json.dumps(match))
        )

    async def _iter_rows(
        self, session: AsyncSession, info: CollectionInfo, batch_size: int
    ) -> AsyncIterator[dict]:
        """Keyset pagination over ``id``."""
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

    async def _fetch_embeddings(
        self, session: AsyncSession, info: CollectionInfo, ids: list[UUID]
    ) -> dict[UUID, list[float]]:
        rows = (
            await session.execute(
                sql_text(
                    f"SELECT id, embedding::text AS embedding "
                    f"FROM {self._qualified(info.table_name)} "
                    f"WHERE id = ANY(:ids)"
                ).bindparams(ids=ids)
            )
        ).all()
        return {row.id: _parse_vector(row.embedding) for row in rows}

    async def _enable_filtered_scan(self, session: AsyncSession) -> None:
        """Let HNSW keep scanning until a filtered query has its rows.

        ``SET LOCAL`` lasts for the current transaction only, so it affects the
        query that follows and nothing else on a pooled connection.
        """
        if await self._pgvector(session) >= ITERATIVE_SCAN_MIN_VERSION:
            await session.execute(
                sql_text("SET LOCAL hnsw.iterative_scan = relaxed_order")
            )

    async def _scan(
        self,
        session: AsyncSession,
        info: CollectionInfo,
        embeddings: list[list[float]],
        top_k: int,
        source_ids: Optional[list[str]],
        keep_best: bool,
    ) -> list[list[dict]]:
        """One query for the whole batch (CROSS JOIN LATERAL over the vectors)."""
        cte_sql, params = self._build_batch_query_cte(
            info=info,
            embeddings=embeddings,
            top_k=top_k,
            source_ids=source_ids,
            keep_best=keep_best,
        )
        if source_ids:
            await self._enable_filtered_scan(session)
        query_sql = (
            f"WITH {cte_sql} SELECT * FROM rag_results "
            f"ORDER BY query_idx ASC, distance ASC"
        )
        rows = (await session.execute(sql_text(query_sql).bindparams(**params))).all()
        batches: list[list[dict]] = [[] for _ in embeddings]
        for row in rows:
            batches[int(row.query_idx) - 1].append(
                {
                    "id": row.id,
                    "source_id": row.source_id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "distance": row.distance,
                }
            )
        return batches

    # Postgres-only queries

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
            vector_parts.append(f"CAST(:vector_{i} AS public.{info.vector_type})")
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
        *,
        stats_receiver: Any = None,
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
            info = await self._load_collection(session, collection_name)
            await session.commit()
            if info is None:
                return [[] for _ in texts]
            embeddings = await self._compute_embeddings(
                texts, info.model, stats_receiver
            )

            source_filter = None
            if join_table and additional_where:
                source_filter = f"""
                    EXISTS (
                        SELECT 1 FROM {join_table}
                        WHERE {join_condition.replace("r.source_id", "rag_index.source_id")}
                        AND {additional_where}
                    )
                """
                await self._enable_filtered_scan(session)

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
            info = await self._load_collection(session, collection_name)
            await session.commit()
            if info is None:
                return [[0.0 for _ in source_ids] for _ in texts]
            embeddings = await self._compute_embeddings(texts, info.model)

            agg = "min" if method == "min" else "avg"
            params: dict = {"source_ids": source_ids}
            dist_cols = []
            for i, emb in enumerate(embeddings):
                params[f"vector_{i}"] = _vector_literal(emb)
                dist_cols.append(
                    f"{agg}(embedding <=> CAST(:vector_{i} AS "
                    f"public.{info.vector_type})) AS dist_{i}"
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
                        f"SELECT avg(embedding)::text "
                        f"FROM {self._qualified(info.table_name)}"
                    )
                )
            ).scalar()
            if mean_vector is None:
                raise Exception("No embeddings found in RAG index.")
            await session.commit()
            return Normalizer(center_vector=_parse_vector(mean_vector))
