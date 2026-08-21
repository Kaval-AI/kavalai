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
import struct
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Union
from uuid import UUID, uuid4

from loguru import logger

from kavalai.llm_clients.embeddings import make_embedding_client
from kavalai.normalizer import Normalizer
from kavalai.rag.base import BaseRagService, RagServiceResult

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


class SqliteRagService(BaseRagService):
    """
    SQLite backed RAG service using the sqlite-vector extension
    (https://github.com/sqliteai/sqlite-vector).

    The whole index lives in a single ordinary SQLite file with one table, so
    it can be pre-compiled offline and shipped to a website: the same file is
    readable in the browser with SQLite AI's WASM build
    (``@sqliteai/sqlite-wasm``), which has the vector extension enabled — the
    same setup the WebLLM playground uses. To keep that portable, ids are TEXT
    UUIDs, metadata is JSON text and embeddings are FLOAT32 blobs
    (``vector_as_f32``). Cosine distance is used, so similarity scores match
    the Postgres backend (``similarity = 1 - distance``).

    All methods run the SQLite work synchronously on the calling loop — no
    worker threads — so the service also works under Pyodide / WebAssembly.
    Unlike :class:`~kavalai.rag.postgres.PostgresRagService`, embedding usage
    stats are logged but not persisted (the index file has no stats table).
    """

    #: Both optional methods of the interface are implemented here.
    capabilities = frozenset({"count_entries", "iter_entries"})

    def __init__(
        self,
        filename: str,
        model: str,
        table_name: str = "rag_index",
        auto_create: bool = True,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initialize the SqliteRagService.

        Args:
            filename (str): Path of the SQLite database file (":memory:" for an
                            in-memory index).
            model (str): The name of the embedding model to use (e.g., "openai/text-embedding-3-small").
            table_name (str): Name of the index table inside the database file.
                              Defaults to "rag_index".
            auto_create (bool): If True (default), create the file and the table
                                when they do not exist. If False, raise an error
                                when the file or the table is missing.
            normalizer (Optional[Normalizer]): Optional normalizer to use for embeddings.

        Raises:
            ValueError: If table_name is not a valid identifier, or the table is
                        missing and auto_create is False.
            FileNotFoundError: If the file is missing and auto_create is False.
        """
        if not _VALID_IDENTIFIER.match(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")

        self.filename = filename
        self.table_name = table_name
        self.model = model
        self.normalizer = normalizer
        self._embedding_client = None
        self._dimension: Optional[int] = None
        self._vector_initialized = False

        in_memory = filename == ":memory:"
        if not auto_create and not in_memory and not os.path.exists(filename):
            raise FileNotFoundError(
                f"RAG index file {filename!r} does not exist (auto_create=False)."
            )

        self._conn = sqlite3.connect(filename)
        self._conn.row_factory = sqlite3.Row
        self._load_vector_extension()

        if not self._table_exists():
            if not auto_create:
                self._conn.close()
                raise ValueError(
                    f"Table {table_name!r} does not exist in {filename!r} (auto_create=False)."
                )
            self._create_table()

    @property
    def embedding_client(self):
        """Embedding client, created on first use.

        Resolved lazily, as ``PostgresRagService`` does: building it in
        ``__init__`` meant an unregistered embedding provider raised before the
        caller ever held the service, leaving no chance to assign one.
        """
        if self._embedding_client is None:
            self._embedding_client = make_embedding_client(self.model)
        return self._embedding_client

    @embedding_client.setter
    def embedding_client(self, client) -> None:
        self._embedding_client = client

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

    def _table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (self.table_name,),
        ).fetchone()
        return row is not None

    def _create_table(self) -> None:
        """Create the index table; mirrors the Postgres rag_index migration."""
        self._conn.executescript(
            f"""
            CREATE TABLE {self.table_name} (
                id TEXT PRIMARY KEY,
                model TEXT,
                collection_name TEXT NOT NULL,
                source_id TEXT NOT NULL,
                content TEXT,
                embedding_size INTEGER NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE INDEX idx_{self.table_name}_collection ON {self.table_name} (collection_name);
            CREATE INDEX idx_{self.table_name}_source_id ON {self.table_name} (source_id);
            """
        )
        self._conn.commit()

    def _ensure_vector_ready(self, dimension: int) -> None:
        """
        Register the embedding column with the vector extension.

        ``vector_init`` must run once per connection. When the file already
        holds an index built with a different dimension, ``vector_init`` itself
        rejects the mismatch via the config persisted in the database.
        """
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(
                f"Embedding dimension {dimension} does not match the index "
                f"dimension {self._dimension} of table {self.table_name!r}."
            )

        if self._vector_initialized:
            return

        self._conn.execute(
            f"SELECT vector_init('{self.table_name}', 'embedding', ?)",
            (f"type=FLOAT32,dimension={self._dimension},distance=COSINE",),
        )
        self._vector_initialized = True

    async def _compute_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings; usage stats are logged, not persisted."""
        embeddings, stats = await self.embedding_client.compute_embeddings(
            texts=texts,
            normalizer=self.normalizer,
        )
        if stats is not None:
            logger.debug(
                f"SqliteRagService embedded {len(texts)} texts with {self.model} "
                f"({getattr(stats, 'total_tokens', None)} tokens)"
            )
        return embeddings

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
            dict: The created index row (id, model, collection_name, source_id,
                  content, embedding_size, rag_metadata, created_at, updated_at).
        """
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

        Args:
            texts (list[str]): List of text strings to index.
            metadata_list (list[dict]): List of metadata dictionaries for each text.
            source_ids (Optional[list[str]]): Optional list of source identifiers.
                                              If not provided, "default" is used.
            collection_name (str): Name of the collection to add items to. Defaults to "default".

        Returns:
            list[dict]: List of created index rows (see :meth:`index`).

        Raises:
            ValueError: If the lengths of texts, metadata_list, or source_ids do
                        not match, or the embedding dimension does not match the index.
        """
        if not texts:
            return []

        if len(texts) != len(metadata_list):
            raise ValueError(
                "The number of texts and metadata dictionaries must be the same."
            )

        if source_ids and len(texts) != len(source_ids):
            raise ValueError("The number of texts and source_ids must be the same.")

        embeddings = await self._compute_embeddings(texts)
        dim = len(embeddings[0])
        # The scan functions silently skip rows whose blob size does not match
        # the registered dimension, so enforce consistency up front.
        self._ensure_vector_ready(dim)

        now = _utc_now_iso()
        rows = []
        for i, (content, meta, emb) in enumerate(zip(texts, metadata_list, embeddings)):
            rows.append(
                {
                    "id": str(uuid4()),
                    "model": self.model,
                    "collection_name": collection_name,
                    "source_id": source_ids[i] if source_ids else "default",
                    "content": content,
                    "embedding_size": dim,
                    "rag_metadata": meta,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        self._conn.executemany(
            f"""
            INSERT INTO {self.table_name}
                (id, model, collection_name, source_id, content, embedding_size,
                 embedding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, vector_as_f32(?), ?, ?, ?)
            """,
            [
                (
                    row["id"],
                    row["model"],
                    row["collection_name"],
                    row["source_id"],
                    row["content"],
                    row["embedding_size"],
                    json.dumps(emb),
                    json.dumps(row["rag_metadata"]),
                    row["created_at"],
                    row["updated_at"],
                )
                for row, emb in zip(rows, embeddings)
            ],
        )
        self._conn.commit()
        return rows

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
            keep_best (bool): If True, only the best result per source_id is returned.
                             Useful when a single source is split into multiple indexed items.

        Returns:
            list[RagServiceResult]: List of results with similarity scores.
        """
        results = await self.query_batch(
            texts=[text],
            top_k=top_k,
            collection_name=collection_name,
            source_ids=source_ids,
            keep_best=keep_best,
            include_content=include_content,
        )
        for result in results[0]:
            result.query_index = None
        return results[0]

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
        Query the indexed items for similarities to multiple input texts.

        The embeddings for all query texts are computed in a single call; each
        text is then answered with one vector scan.

        Args:
            texts (list[str]): List of query texts to search for.
            top_k (int): Number of top results to return per query. Defaults to 5.
            collection_name (Optional[str]): If provided, filter by collection name.
            source_ids (Optional[list[str]]): If provided, filter by source identifiers.
            keep_best (bool): If True, only the best result per source_id is returned per query.

        Returns:
            list[list[RagServiceResult]]: A list of result lists, where each inner list contains
                                          the top_k results for the corresponding query text.
        """
        if not texts:
            return []

        embeddings = await self._compute_embeddings(texts)
        self._ensure_vector_ready(len(embeddings[0]))

        sql, filter_params = self._build_scan_sql(
            top_k=top_k,
            collection_name=collection_name,
            source_ids=source_ids,
            keep_best=keep_best,
        )

        batch_results = []
        for query_index, embedding in enumerate(embeddings):
            rows = self._conn.execute(
                sql, (json.dumps(embedding), *filter_params)
            ).fetchall()
            batch_results.append(
                [
                    self._map_row_to_result(row, query_index, include_content)
                    for row in rows
                ]
            )
        return batch_results

    def _build_scan_sql(
        self,
        top_k: int,
        collection_name: Optional[str],
        source_ids: Optional[list[str]],
        keep_best: bool,
    ) -> tuple[str, list]:
        """
        Build the vector scan query.

        ``vector_full_scan`` is used without ``k`` so filters are applied to the
        distances of *all* rows before ``LIMIT`` — passing ``k`` to the scan
        would drop matches when the k nearest overall fall outside the filter.
        """
        where_clauses = ["t.model = ?"]
        params: list = [self.model]

        if collection_name:
            where_clauses.append("t.collection_name = ?")
            params.append(collection_name)

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
                t.id, t.model, t.collection_name, t.source_id, t.content,
                t.embedding_size, t.metadata, t.created_at, t.updated_at,
                {distance_col}
            FROM vector_full_scan('{self.table_name}', 'embedding', vector_as_f32(?)) AS v
            JOIN {self.table_name} t ON t.rowid = v.rowid
            WHERE {where_clause}
            {group_by}
            ORDER BY distance ASC
            LIMIT ?
        """
        params.append(top_k)
        return sql, params

    def _map_row_to_result(
        self,
        row: sqlite3.Row,
        query_index: Optional[int],
        include_content: bool = True,
    ) -> RagServiceResult:
        """Map a database row to a RagServiceResult."""
        return RagServiceResult(
            id=UUID(row["id"]),
            model=row["model"],
            collection_name=row["collection_name"],
            source_id=row["source_id"],
            content=row["content"] if include_content else None,
            embedding_size=row["embedding_size"],
            rag_metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            similarity=1.0 - float(row["distance"]),
            created_at=_parse_timestamp(row["created_at"]),
            updated_at=_parse_timestamp(row["updated_at"]),
            query_index=query_index,
        )

    async def delete(
        self, item_id: UUID, collection_name: Optional[str] = None
    ) -> None:
        """
        Delete a single indexed item by its identifier.

        Args:
            item_id (UUID): Identifier of the indexed item to delete.
            collection_name (Optional[str]): Ignored — all collections share
                one table in this backend, and ids are globally unique.
        """
        self._conn.execute(
            f"DELETE FROM {self.table_name} WHERE id = ?", (str(item_id),)
        )
        self._conn.commit()

    async def count_entries(self, collection_name: str) -> int:
        """Number of entries in a collection (0 if it doesn't exist)."""
        if not self._table_exists():
            return 0
        row = self._conn.execute(
            f"SELECT count(*) FROM {self.table_name} WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return row[0] if row else 0

    async def iter_entries(self, collection_name: str, batch_size: int = 500):
        """Iterate all entries of a collection (including embeddings)."""
        if not self._table_exists():
            return
        cursor = self._conn.execute(
            f"""
            SELECT id, source_id, content, embedding,
                   metadata, created_at, updated_at
            FROM {self.table_name} WHERE collection_name = ? ORDER BY id
            """,
            (collection_name,),
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
        source_ids = [source_id] if isinstance(source_id, str) else source_id
        placeholders = ", ".join("?" for _ in source_ids)
        self._conn.execute(
            f"DELETE FROM {self.table_name} "
            f"WHERE collection_name = ? AND source_id IN ({placeholders})",
            (collection_name, *source_ids),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
