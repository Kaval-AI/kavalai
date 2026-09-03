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

SQLite (sqlite-vector) backed RAG service.

The storage model is the one described in :mod:`kavalai.rag.collections` —
a ``rag_collections`` registry and one table per collection — in a single
ordinary SQLite file. Ids are TEXT UUIDs, metadata is JSON text, timestamps
are ISO-8601 text and embeddings are FLOAT32 blobs (``vector_as_f32``)
searched through the ``sqlite-vector`` extension under cosine distance, so
similarity scores match the Postgres backend (``similarity = 1 - distance``).

``_COLLECTION_UPGRADES`` maps a *from_version* to an async callable
``(connection, service, collection_info) -> None``; see
``RAG_COLLECTION_SCHEMA_VERSION`` in :mod:`kavalai.rag.collections`.
"""

import json
import os
import sqlite3
import struct
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional
from uuid import UUID

from loguru import logger

from kavalai.normalizer import Normalizer
from kavalai.rag.collections import CollectionInfo, CollectionRagService

_COLLECTION_UPGRADES: dict[int, Callable] = {}

LEGACY_TABLE = "rag_index"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _decode_f32_blob(blob) -> "list[float] | None":
    """Decode a sqlite-vector f32 BLOB back into a list of floats."""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        raw = bytes(blob)
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if isinstance(blob, str):
        return json.loads(blob)
    return list(blob)


class SqliteRagService(CollectionRagService):
    """
    SQLite backed RAG service using the sqlite-vector extension
    (https://github.com/sqliteai/sqlite-vector).

    No server: the whole index — registry and collection tables — lives in
    one file that can be built offline and copied wherever it is needed,
    ``":memory:"`` included. Each collection carries its own embedding
    dimension, as on Postgres.

    All methods run the SQLite work synchronously on the calling loop — no
    worker threads — so the service also works under Pyodide / WebAssembly.
    Unlike :class:`~kavalai.rag.postgres.PostgresRagService`, embedding usage
    stats are logged but not persisted (the index file has no stats table).
    """

    collection_upgrades = _COLLECTION_UPGRADES

    def __init__(
        self,
        filename: str,
        model: Optional[str] = None,
        auto_create: bool = True,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initialize the SqliteRagService.

        Args:
            filename (str): Path of the SQLite database file (":memory:" for an
                            in-memory index).
            model (Optional[str]): The embedding model to use (e.g.
                "openai/text-embedding-3-small"). May be ``None`` to browse,
                count, export or drop collections; indexing and querying
                then fail.
            auto_create (bool): If True (default), create the file and the
                                registry when they do not exist. If False,
                                raise when the file or the registry is missing.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.

        Raises:
            FileNotFoundError: If the file is missing and auto_create is False.
            ValueError: If the registry is missing and auto_create is False,
                        or the file holds the single-table layout of
                        kavalai 1.0, which this version does not read.
        """
        super().__init__(model=model, normalizer=normalizer)
        self.filename = filename

        in_memory = filename == ":memory:"
        if not auto_create and not in_memory and not os.path.exists(filename):
            raise FileNotFoundError(
                f"RAG index file {filename!r} does not exist (auto_create=False)."
            )

        self._conn = sqlite3.connect(filename)
        self._conn.row_factory = sqlite3.Row
        self._load_vector_extension()

        has_registry = self._table_exists(self.REGISTRY_TABLE)
        if not has_registry and self._table_exists(LEGACY_TABLE):
            self._conn.close()
            raise ValueError(
                f"RAG index file {filename!r} uses the single-table layout of "
                f"kavalai 1.0, which this version does not read. Rebuild the "
                f"index."
            )
        if not has_registry and not auto_create:
            self._conn.close()
            raise ValueError(
                f"Table {self.REGISTRY_TABLE!r} does not exist in {filename!r} "
                f"(auto_create=False)."
            )

    def _load_vector_extension(self) -> None:
        """Load the sqlite-vector extension unless it is already built in (WASM)."""
        try:
            self._conn.execute("SELECT vector_version()")
            return  # Statically linked (e.g. sqliteai's WASM build).
        except sqlite3.OperationalError:
            pass

        try:
            from importlib.resources import files

            extension_path = str(files("sqlite_vector.binaries") / "vector")
        except ModuleNotFoundError as e:
            raise ImportError(
                "SqliteRagService requires the sqlite-vector extension. "
                "Install it with: pip install sqliteai-vector (or the kavalai[common] extra)."
            ) from e

        self._conn.enable_load_extension(True)
        try:
            self._conn.load_extension(extension_path)
        finally:
            self._conn.enable_load_extension(False)

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # Hooks

    @asynccontextmanager
    async def _connection(self):
        yield self._conn

    async def _commit(self, conn: sqlite3.Connection) -> None:
        conn.commit()

    async def _record_stats(self, conn, stats) -> None:
        logger.debug(
            f"SqliteRagService embedded {stats.batch_size} texts with "
            f"{self.model} ({stats.total_tokens} tokens)"
        )

    async def _create_registry(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.REGISTRY_TABLE} (
                name TEXT PRIMARY KEY,
                table_name TEXT UNIQUE NOT NULL,
                model TEXT NOT NULL,
                embedding_size INTEGER NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _info_from_row(row: sqlite3.Row) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            table_name=row["table_name"],
            model=row["model"],
            embedding_size=row["embedding_size"],
            schema_version=row["schema_version"],
        )

    async def _fetch_registry_row(
        self, conn: sqlite3.Connection, name: str
    ) -> Optional[CollectionInfo]:
        row = conn.execute(
            f"SELECT name, table_name, model, embedding_size, schema_version "
            f"FROM {self.REGISTRY_TABLE} WHERE name = ?",
            (name,),
        ).fetchone()
        return None if row is None else self._info_from_row(row)

    async def _list_registry(self, conn: sqlite3.Connection) -> list[CollectionInfo]:
        rows = conn.execute(
            f"SELECT name, table_name, model, embedding_size, schema_version "
            f"FROM {self.REGISTRY_TABLE} ORDER BY name"
        ).fetchall()
        return [self._info_from_row(row) for row in rows]

    async def _insert_registry_row(
        self, conn: sqlite3.Connection, info: CollectionInfo
    ) -> None:
        conn.execute(
            f"INSERT OR IGNORE INTO {self.REGISTRY_TABLE} "
            f"(name, table_name, model, embedding_size, schema_version, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            (
                info.name,
                info.table_name,
                info.model,
                info.embedding_size,
                info.schema_version,
                _utc_now_iso(),
            ),
        )

    async def _set_registry_version(
        self, conn: sqlite3.Connection, name: str, version: int
    ) -> None:
        conn.execute(
            f"UPDATE {self.REGISTRY_TABLE} SET schema_version = ? WHERE name = ?",
            (version, name),
        )

    async def _delete_registry_row(self, conn: sqlite3.Connection, name: str) -> None:
        conn.execute(f"DELETE FROM {self.REGISTRY_TABLE} WHERE name = ?", (name,))

    async def _create_collection_table(
        self, conn: sqlite3.Connection, info: CollectionInfo
    ) -> None:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {info.table_name} (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                content TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_{info.table_name}_source_id
                ON {info.table_name} (source_id);
            """
        )

    async def _drop_collection_table(
        self, conn: sqlite3.Connection, info: CollectionInfo
    ) -> None:
        conn.execute(f"DROP TABLE IF EXISTS {info.table_name}")

    async def _prepare_collection(
        self, conn: sqlite3.Connection, info: CollectionInfo
    ) -> None:
        """Register the embedding column with the vector extension.

        ``vector_init`` must run once per connection and table; the
        collection cache makes this exactly once per service instance.
        """
        conn.execute(
            f"SELECT vector_init('{info.table_name}', 'embedding', ?)",
            (f"type=FLOAT32,dimension={info.embedding_size},distance=COSINE",),
        )

    async def _count_rows(self, conn: sqlite3.Connection, info: CollectionInfo) -> int:
        row = conn.execute(f"SELECT count(*) FROM {info.table_name}").fetchone()
        return row[0] if row else 0

    async def _insert_rows(
        self, conn: sqlite3.Connection, info: CollectionInfo, rows: list[dict]
    ) -> None:
        conn.executemany(
            f"""
            INSERT INTO {info.table_name}
                (id, source_id, content, embedding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, vector_as_f32(?), ?, ?, ?)
            """,
            [
                (
                    str(row["id"]),
                    row["source_id"],
                    row["content"],
                    json.dumps(row["embedding"]),
                    json.dumps(row["rag_metadata"]),
                    row["created_at"].isoformat(),
                    row["updated_at"].isoformat(),
                )
                for row in rows
            ],
        )

    async def _delete_row(
        self, conn: sqlite3.Connection, info: CollectionInfo, item_id: UUID
    ) -> None:
        conn.execute(f"DELETE FROM {info.table_name} WHERE id = ?", (str(item_id),))

    async def _delete_rows_by_source_ids(
        self, conn: sqlite3.Connection, info: CollectionInfo, source_ids: list[str]
    ) -> None:
        placeholders = ", ".join("?" for _ in source_ids)
        conn.execute(
            f"DELETE FROM {info.table_name} WHERE source_id IN ({placeholders})",
            tuple(source_ids),
        )

    async def _iter_rows(
        self, conn: sqlite3.Connection, info: CollectionInfo, batch_size: int
    ) -> AsyncIterator[dict]:
        cursor = conn.execute(
            f"SELECT id, source_id, content, embedding, metadata, created_at, "
            f"updated_at FROM {info.table_name} ORDER BY id"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            for row in rows:
                yield {
                    "id": UUID(row["id"]),
                    "source_id": row["source_id"],
                    "content": row["content"],
                    "embedding": _decode_f32_blob(row["embedding"]),
                    "rag_metadata": json.loads(row["metadata"])
                    if row["metadata"]
                    else {},
                    "created_at": _parse_timestamp(row["created_at"]),
                    "updated_at": _parse_timestamp(row["updated_at"]),
                }

    async def _fetch_embeddings(
        self, conn: sqlite3.Connection, info: CollectionInfo, ids: list[UUID]
    ) -> dict[UUID, list[float]]:
        placeholders = ", ".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, embedding FROM {info.table_name} WHERE id IN ({placeholders})",
            tuple(str(item_id) for item_id in ids),
        ).fetchall()
        return {UUID(row["id"]): _decode_f32_blob(row["embedding"]) for row in rows}

    async def _scan(
        self,
        conn: sqlite3.Connection,
        info: CollectionInfo,
        embeddings: list[list[float]],
        top_k: int,
        source_ids: Optional[list[str]],
        keep_best: bool,
    ) -> list[list[dict]]:
        """One vector scan per query embedding."""
        sql, filter_params = self._build_scan_sql(
            info, top_k=top_k, source_ids=source_ids, keep_best=keep_best
        )
        batches = []
        for embedding in embeddings:
            rows = conn.execute(sql, (json.dumps(embedding), *filter_params)).fetchall()
            batches.append(
                [
                    {
                        "id": UUID(row["id"]),
                        "source_id": row["source_id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else {},
                        "created_at": _parse_timestamp(row["created_at"]),
                        "updated_at": _parse_timestamp(row["updated_at"]),
                        "distance": row["distance"],
                    }
                    for row in rows
                ]
            )
        return batches

    @staticmethod
    def _build_scan_sql(
        info: CollectionInfo,
        top_k: int,
        source_ids: Optional[list[str]],
        keep_best: bool,
    ) -> tuple[str, list]:
        """
        Build the vector scan query.

        ``vector_full_scan`` is used without ``k`` so filters are applied to the
        distances of *all* rows before ``LIMIT`` — passing ``k`` to the scan
        would drop matches when the k nearest overall fall outside the filter.
        """
        where_clauses = ["1 = 1"]
        params: list = []

        if source_ids:
            placeholders = ", ".join("?" for _ in source_ids)
            where_clauses.append(f"t.source_id IN ({placeholders})")
            params.extend(source_ids)

        where_clause = " AND ".join(where_clauses)

        if keep_best:
            # SQLite's bare-column MIN() picks the row with the smallest
            # distance per source_id group.
            distance_col = "MIN(v.distance) AS distance"
            group_by = "GROUP BY t.source_id"
        else:
            distance_col = "v.distance AS distance"
            group_by = ""

        sql = f"""
            SELECT
                t.id, t.source_id, t.content, t.metadata, t.created_at,
                t.updated_at, {distance_col}
            FROM vector_full_scan('{info.table_name}', 'embedding', vector_as_f32(?)) AS v
            JOIN {info.table_name} t ON t.rowid = v.rowid
            WHERE {where_clause}
            {group_by}
            ORDER BY distance ASC
            LIMIT ?
        """
        params.append(top_k)
        return sql, params
