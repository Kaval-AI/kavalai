import sqlite3
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kavalai.rag import RagServiceResult, SqliteRagService

MODEL = "fake/embedding-model"

# Deterministic 3-dim embeddings; cosine distance is used, so direction is
# what matters.
VECTORS = {
    "apple": [1.0, 0.0, 0.0],
    "apple pie": [0.95, 0.05, 0.0],
    "banana": [0.0, 1.0, 0.0],
    "cherry": [0.0, 0.0, 1.0],
    "fruit": [0.7, 0.7, 0.1],
}


def make_fake_embedding_client(vectors=VECTORS):
    async def compute_embeddings(texts, normalizer=None):
        return [vectors[t] for t in texts], MagicMock(total_tokens=len(texts))

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=compute_embeddings)
    return client


@pytest.fixture
def service_factory(tmp_path):
    """Create SqliteRagService instances with a mocked embedding client."""
    created = []
    with patch("kavalai.rag.collections.make_embedding_client") as mock_make_client:
        mock_make_client.side_effect = lambda model: make_fake_embedding_client()

        def factory(filename=None, model=MODEL, **kwargs):
            service = SqliteRagService(
                filename or str(tmp_path / "rag.db"), model=model, **kwargs
            )
            created.append(service)
            return service

        yield factory
    for service in created:
        service.close()


async def index_fruits(service, collection_name="fruits"):
    return await service.index_batch(
        texts=["apple", "apple pie", "banana", "cherry"],
        metadata_list=[{"kind": "pome"}, {"kind": "pastry"}, {}, {}],
        source_ids=["sid_apple", "sid_apple", "sid_banana", "sid_cherry"],
        collection_name=collection_name,
    )


@pytest.mark.asyncio
async def test_auto_create_and_roundtrip(service_factory, tmp_path):
    service = service_factory()

    items = await index_fruits(service)
    assert len(items) == 4
    assert items[0]["content"] == "apple"
    assert items[0]["collection_name"] == "fruits"
    assert items[0]["source_id"] == "sid_apple"
    assert items[0]["rag_metadata"] == {"kind": "pome"}
    assert items[0]["embedding_size"] == 3
    assert items[0]["model"] == MODEL

    results = await service.query("apple", top_k=2, collection_name="fruits")
    assert len(results) == 2
    assert isinstance(results[0], RagServiceResult)
    assert results[0].content == "apple"
    assert results[1].content == "apple pie"
    assert results[0].similarity > results[1].similarity > 0.9
    assert results[0].rag_metadata == {"kind": "pome"}
    assert isinstance(results[0].id, uuid.UUID)
    assert isinstance(results[0].created_at, datetime)
    assert results[0].query_index is None

    # The index file exists on disk (auto-created)
    assert (tmp_path / "rag.db").exists()


@pytest.mark.asyncio
async def test_reopen_existing_file(service_factory, tmp_path):
    filename = str(tmp_path / "prebuilt.db")
    service = service_factory(filename)
    await index_fruits(service)
    service.close()

    # Reopen the pre-compiled index without auto-create and query it directly.
    reopened = service_factory(filename, auto_create=False)
    results = await reopened.query("banana", top_k=1, collection_name="fruits")
    assert len(results) == 1
    assert results[0].content == "banana"


def test_auto_create_false_missing_file(service_factory, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        service_factory(str(tmp_path / "missing.db"), auto_create=False)


def test_auto_create_false_missing_table(service_factory, tmp_path):
    filename = str(tmp_path / "existing.db")
    conn = sqlite3.connect(filename)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="'rag_collections' does not exist"):
        service_factory(filename, auto_create=False)


def test_legacy_single_table_layout_is_refused(service_factory, tmp_path):
    """A file built by kavalai 1.0 (one ``rag_index`` table) is not read.

    Refusing with a message beats opening it as an empty index: the registry
    would be created next to the old table and every query would come back
    empty with no hint why.
    """
    filename = str(tmp_path / "old.db")
    conn = sqlite3.connect(filename)
    conn.execute("CREATE TABLE rag_index (id TEXT PRIMARY KEY, collection_name TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="kavalai 1.0.*Rebuild the index"):
        service_factory(filename)


@pytest.mark.asyncio
async def test_one_table_per_collection_in_the_registry(service_factory, tmp_path):
    """The Postgres shape: a ``rag_collections`` row and a table per collection."""
    filename = str(tmp_path / "shape.db")
    service = service_factory(filename)
    await service.index(text="apple", collection_name="fruits")
    await service.index(text="banana", collection_name="more fruits")

    conn = sqlite3.connect(filename)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    registry = conn.execute(
        "SELECT name, table_name, model, embedding_size FROM rag_collections "
        "ORDER BY name"
    ).fetchall()
    conn.close()

    assert "rag_collections" in tables
    assert {
        SqliteRagService.table_name_for_collection(n) for n in ("fruits", "more fruits")
    } <= tables
    assert registry == [
        ("fruits", SqliteRagService.table_name_for_collection("fruits"), MODEL, 3),
        (
            "more fruits",
            SqliteRagService.table_name_for_collection("more fruits"),
            MODEL,
            3,
        ),
    ]


@pytest.mark.asyncio
async def test_collections_have_their_own_dimension(service_factory):
    """Each collection registers its own ``vector_init`` dimension."""
    service = service_factory()
    await index_fruits(service)

    service.embedding_client = make_fake_embedding_client(
        {"weird": [0.1, 0.2, 0.3, 0.4], "odd": [0.4, 0.3, 0.2, 0.1]}
    )
    await service.index_batch(
        texts=["weird", "odd"], metadata_list=[{}, {}], collection_name="four"
    )

    assert [
        r.content for r in await service.query("weird", collection_name="four")
    ] == [
        "weird",
        "odd",
    ]
    assert [c["embedding_size"] for c in await service.list_collections()] == [4, 3]


@pytest.mark.asyncio
async def test_index_single(service_factory):
    service = service_factory()
    item = await service.index(
        text="apple", source_metadata={"key": "val"}, source_id="s1"
    )
    assert item["content"] == "apple"
    assert item["rag_metadata"] == {"key": "val"}
    assert item["collection_name"] == "default"
    assert item["source_id"] == "s1"


@pytest.mark.asyncio
async def test_index_batch_validation(service_factory):
    service = service_factory()

    assert await service.index_batch(texts=[], metadata_list=[]) == []

    with pytest.raises(
        ValueError,
        match="The number of texts and metadata dictionaries must be the same.",
    ):
        await service.index_batch(texts=["apple"], metadata_list=[])

    with pytest.raises(
        ValueError, match="The number of texts and source_ids must be the same."
    ):
        await service.index_batch(
            texts=["apple"], metadata_list=[{}], source_ids=["1", "2"]
        )


@pytest.mark.asyncio
async def test_dimension_mismatch(service_factory):
    service = service_factory()
    await index_fruits(service)

    # Later embeddings with a different dimension must be rejected, since the
    # scan would silently skip them otherwise.
    four_dim = make_fake_embedding_client({"weird": [0.1, 0.2, 0.3, 0.4]})
    service.embedding_client = four_dim
    with pytest.raises(ValueError, match="3-dimensional embeddings; got 4"):
        await service.index_batch(
            texts=["weird"], metadata_list=[{}], collection_name="fruits"
        )


@pytest.mark.asyncio
async def test_query_filters(service_factory):
    service = service_factory()
    await index_fruits(service, collection_name="fruits")
    await service.index(text="cherry", collection_name="other", source_id="sid_other")

    # Collection filter
    results = await service.query("cherry", top_k=10, collection_name="other")
    assert len(results) == 1
    assert results[0].source_id == "sid_other"

    # Source id filter
    results = await service.query(
        "fruit",
        top_k=10,
        collection_name="fruits",
        source_ids=["sid_apple", "sid_cherry"],
    )
    assert {r.source_id for r in results} == {"sid_apple", "sid_cherry"}
    assert len(results) == 3  # sid_apple has two chunks

    # top_k limits results
    results = await service.query("fruit", top_k=2, collection_name="fruits")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_results_carry_the_collection_model(service_factory, tmp_path):
    """The model is a property of the collection, recorded when it is created."""
    filename = str(tmp_path / "models.db")
    service_a = service_factory(filename, model="fake/model-a")
    await service_a.index(text="apple", source_id="sid_a", collection_name="c")

    browsing = service_factory(filename, model=None)
    assert [c["model"] for c in await browsing.list_collections()] == ["fake/model-a"]
    with pytest.raises(ValueError, match="without an embedding model"):
        await browsing.query("apple", collection_name="c")

    results = await service_a.query("apple", top_k=10, collection_name="c")
    assert [r.model for r in results] == ["fake/model-a"]


@pytest.mark.asyncio
async def test_query_keep_best(service_factory):
    service = service_factory()
    await index_fruits(service)

    results = await service.query(
        "apple", top_k=10, collection_name="fruits", keep_best=True
    )
    # One result per source_id, and for sid_apple the closest chunk wins.
    assert len(results) == 3
    source_ids = [r.source_id for r in results]
    assert len(source_ids) == len(set(source_ids))
    best_apple = next(r for r in results if r.source_id == "sid_apple")
    assert best_apple.content == "apple"


@pytest.mark.asyncio
async def test_query_empty_index(service_factory):
    service = service_factory()
    assert await service.query("apple", top_k=5) == []


@pytest.mark.asyncio
async def test_query_batch(service_factory):
    service = service_factory()
    await index_fruits(service)

    results = await service.query_batch(
        texts=["apple", "banana"], top_k=1, collection_name="fruits"
    )
    assert len(results) == 2
    assert results[0][0].content == "apple"
    assert results[0][0].query_index == 0
    assert results[1][0].content == "banana"
    assert results[1][0].query_index == 1

    # Embeddings for the whole batch are computed in one call
    service.embedding_client.compute_embeddings.assert_awaited_with(
        texts=["apple", "banana"], normalizer=None
    )

    assert await service.query_batch(texts=[]) == []


@pytest.mark.asyncio
async def test_delete(service_factory):
    service = service_factory()
    items = await index_fruits(service)

    await service.delete(items[0]["id"])

    results = await service.query("apple", top_k=10, collection_name="fruits")
    assert items[0]["id"] not in {r.id for r in results}
    assert len(results) == 3


@pytest.mark.asyncio
async def test_delete_by_source_id(service_factory):
    service = service_factory()
    await index_fruits(service)

    # Single source id string
    await service.delete_by_source_id("fruits", "sid_apple")
    results = await service.query("fruit", top_k=10, collection_name="fruits")
    assert {r.source_id for r in results} == {"sid_banana", "sid_cherry"}

    # List of source ids
    await service.delete_by_source_id("fruits", ["sid_banana", "sid_cherry"])
    assert await service.query("fruit", top_k=10, collection_name="fruits") == []


def test_extension_already_loaded(service_factory, tmp_path):
    """When the connection already has the extension (e.g. WASM builds with it
    statically linked), no load attempt is made."""
    from importlib.resources import files

    extension_path = str(files("sqlite_vector.binaries") / "vector")

    original_connect = sqlite3.connect

    def connect_preloaded(filename):
        conn = original_connect(filename)
        conn.enable_load_extension(True)
        conn.load_extension(extension_path)
        conn.enable_load_extension(False)
        return conn

    # Break the package lookup: if the early-return path were not taken, the
    # constructor would raise ImportError instead of reusing the loaded extension.
    with patch("kavalai.rag.sqllite.sqlite3.connect", side_effect=connect_preloaded):
        with patch("importlib.resources.files", side_effect=ModuleNotFoundError):
            service = service_factory(str(tmp_path / "preloaded.db"))
    assert service._conn.execute("SELECT vector_version()").fetchone()


def test_extension_package_missing(service_factory, tmp_path):
    with patch("importlib.resources.files", side_effect=ModuleNotFoundError):
        with pytest.raises(ImportError, match="sqliteai-vector"):
            service_factory(str(tmp_path / "noext.db"))


@pytest.mark.asyncio
async def test_inherited_compute_similarity_matrix(service_factory):
    """The base-class default implementation works against this backend."""
    service = service_factory()
    await index_fruits(service)

    matrix = await service.compute_similarity_matrix(
        texts=["apple", "banana"],
        source_ids=["sid_apple", "sid_banana", "sid_missing"],
        collection_name="fruits",
    )

    assert len(matrix) == 2
    assert matrix[0][0] > 0.99  # apple vs sid_apple
    assert matrix[0][0] > matrix[0][1]  # apple closer to sid_apple than sid_banana
    assert matrix[1][1] > 0.99  # banana vs sid_banana
    assert matrix[0][2] == 0.0  # missing source
