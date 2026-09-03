"""The contract every ``BaseRagService`` backend must satisfy.

A generic interface is only generic if the implementations agree about what it
means, and agreement is not something a docstring can enforce. Everything the
interface *declares* is checked here against every backend, so a new one is a
known quantity rather than an adventure.

Backends are free to do more than this -- ``PostgresRagService`` has a dozen
methods the interface never mentions. This suite is about the declared surface
only.

Embeddings are deterministic and injected, so nothing here calls a provider:
the vectors below are chosen so the expected ranking is obvious by inspection.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from kavalai.llm_clients.common import create_model_call_stat
from kavalai.rag import PostgresRagService, RagServiceResult, SqliteRagService

# Four-dimensional unit-ish vectors. Cosine distance is what every backend
# indexes with, so direction is all that matters.
VECTORS = {
    "apple": [1.0, 0.0, 0.0, 0.0],
    "apple pie": [0.95, 0.05, 0.0, 0.0],
    "apple tart": [0.9, 0.1, 0.0, 0.0],
    "banana": [0.0, 1.0, 0.0, 0.0],
    "cherry": [0.0, 0.0, 1.0, 0.0],
    "durian": [0.0, 0.0, 0.0, 1.0],
}

CORPUS = [
    ("apple", "apples"),
    ("apple pie", "apples"),
    ("apple tart", "apples"),
    ("banana", "bananas"),
    ("cherry", "cherries"),
]

GUARANTEED_INDEX_KEYS = {
    "id",
    "model",
    "collection_name",
    "source_id",
    "content",
    "embedding_size",
    "rag_metadata",
    "created_at",
    "updated_at",
}


def fake_embedding_client():
    """An embedding client with fixed vectors, so rankings are predictable."""

    async def compute_embeddings(texts, *args, **kwargs):
        # A real ModelCallStat, not a mock: PostgresRagService persists what it
        # is handed, so a mock here would fail inside SQLAlchemy rather than in
        # the assertion, and the contract does include reporting usage.
        stats = create_model_call_stat(
            call_type="embedding",
            model="fake/embedding-model",
            duration_seconds=0.0,
            batch_size=len(texts),
            total_tokens=len(texts),
        )
        return [VECTORS[t] for t in texts], stats

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=compute_embeddings)
    return client


@pytest.fixture(params=["sqlite", "postgres"])
def rag_service(request, tmp_path):
    """Every backend, each with the same injected embedding client.

    Assigning ``embedding_client`` is itself part of the contract: a caller
    holding a service must be able to substitute the embedding side without the
    constructor having already resolved a provider.
    """
    backend = request.param
    if backend == "sqlite":
        service = SqliteRagService(
            str(tmp_path / "conformance.db"), model="fake/embedding-model"
        )
    else:
        # Requested lazily so a sqlite-only run does not start a container.
        config = request.getfixturevalue("agents_db_config")
        request.getfixturevalue("migrated_agents_db")
        service = PostgresRagService.from_uri(
            config["uri"], "fake/embedding-model", schema=config["schema"]
        )
    service.embedding_client = fake_embedding_client()
    return service


@pytest.fixture
def collection(request):
    """A collection name unique to the test, so backends can share a database."""
    return f"conf_{abs(hash(request.node.name)) % 10**8}"


@pytest.fixture
async def populated(rag_service, collection):
    """``CORPUS`` indexed into ``collection``."""
    await rag_service.index_batch(
        texts=[text for text, _ in CORPUS],
        metadata_list=[{"fruit": source} for _, source in CORPUS],
        source_ids=[source for _, source in CORPUS],
        collection_name=collection,
    )
    return rag_service


pytestmark = pytest.mark.asyncio


async def test_index_returns_the_guaranteed_keys(rag_service, collection):
    entry = await rag_service.index(
        "apple", {"fruit": "apples"}, collection_name=collection, source_id="apples"
    )

    assert GUARANTEED_INDEX_KEYS <= set(entry)
    assert entry["content"] == "apple"
    assert entry["collection_name"] == collection
    assert entry["source_id"] == "apples"
    assert entry["rag_metadata"] == {"fruit": "apples"}
    assert entry["embedding_size"] == len(VECTORS["apple"])


async def test_index_mints_a_uuid(rag_service, collection):
    entry = await rag_service.index("apple", collection_name=collection)

    # Backends whose store uses integers or bounded strings must still round
    # trip the UUID Kaval.AI minted.
    assert UUID(str(entry["id"]))


async def test_index_batch_returns_one_entry_per_text(rag_service, collection):
    entries = await rag_service.index_batch(
        texts=["apple", "banana"],
        metadata_list=[{}, {}],
        source_ids=["a", "b"],
        collection_name=collection,
    )

    assert [e["content"] for e in entries] == ["apple", "banana"]


async def test_query_ranks_by_descending_similarity(populated, collection):
    hits = await populated.query("apple", top_k=5, collection_name=collection)

    similarities = [hit.similarity for hit in hits]
    assert similarities == sorted(similarities, reverse=True)
    assert hits[0].content == "apple"
    assert isinstance(hits[0], RagServiceResult)


async def test_similarity_is_higher_is_better_and_bounded(populated, collection):
    hits = await populated.query("apple", top_k=5, collection_name=collection)

    by_content = {hit.content: hit.similarity for hit in hits}
    # An exact match scores ~1.0; an orthogonal vector scores ~0.0.
    assert by_content["apple"] == pytest.approx(1.0, abs=1e-3)
    assert by_content["banana"] == pytest.approx(0.0, abs=1e-3)
    assert by_content["apple"] > by_content["apple pie"] > by_content["banana"]
    for similarity in by_content.values():
        assert -1.0001 <= similarity <= 1.0001


async def test_query_honours_top_k(populated, collection):
    assert len(await populated.query("apple", top_k=2, collection_name=collection)) == 2
    assert len(await populated.query("apple", top_k=4, collection_name=collection)) == 4


async def test_query_filters_by_source_ids(populated, collection):
    hits = await populated.query(
        "apple", top_k=5, collection_name=collection, source_ids=["bananas"]
    )

    assert {hit.source_id for hit in hits} == {"bananas"}


async def test_keep_best_returns_one_hit_per_source(populated, collection):
    hits = await populated.query(
        "apple", top_k=5, collection_name=collection, keep_best=True
    )

    source_ids = [hit.source_id for hit in hits]
    assert len(source_ids) == len(set(source_ids))
    # Of the three "apples" items, the exact match is the one kept.
    best = next(hit for hit in hits if hit.source_id == "apples")
    assert best.content == "apple"


async def test_collections_are_isolated(rag_service, collection):
    await rag_service.index("apple", collection_name=collection, source_id="a")
    await rag_service.index(
        "banana", collection_name=f"{collection}_other", source_id="b"
    )

    hits = await rag_service.query("apple", top_k=5, collection_name=collection)

    assert [hit.content for hit in hits] == ["apple"]


async def test_query_batch_returns_one_list_per_text(populated, collection):
    batches = await populated.query_batch(
        texts=["apple", "banana"], top_k=2, collection_name=collection
    )

    assert len(batches) == 2
    assert batches[0][0].content == "apple"
    assert batches[1][0].content == "banana"


async def test_query_batch_with_no_texts_returns_nothing(rag_service, collection):
    assert await rag_service.query_batch(texts=[], collection_name=collection) == []


async def test_delete_removes_one_item(populated, collection):
    hits = await populated.query("banana", top_k=1, collection_name=collection)

    await populated.delete(hits[0].id, collection_name=collection)

    remaining = await populated.query("banana", top_k=5, collection_name=collection)
    assert "banana" not in [hit.content for hit in remaining]


async def test_delete_by_source_id_accepts_a_string(populated, collection):
    await populated.delete_by_source_id(collection, "apples")

    hits = await populated.query("apple", top_k=5, collection_name=collection)
    assert "apples" not in {hit.source_id for hit in hits}


async def test_delete_by_source_id_accepts_a_list(populated, collection):
    await populated.delete_by_source_id(collection, ["apples", "bananas"])

    hits = await populated.query("apple", top_k=5, collection_name=collection)
    assert {hit.source_id for hit in hits} == {"cherries"}


async def test_delete_of_an_unknown_id_is_not_an_error(populated, collection):
    await populated.delete(uuid4(), collection_name=collection)


async def test_include_content_false_drops_only_the_content(populated, collection):
    with_content = await populated.query("apple", top_k=3, collection_name=collection)
    without = await populated.query(
        "apple", top_k=3, collection_name=collection, include_content=False
    )

    assert [hit.content for hit in without] == [None, None, None]
    assert [hit.source_id for hit in without] == [hit.source_id for hit in with_content]
    assert [hit.similarity for hit in without] == [
        hit.similarity for hit in with_content
    ]
    assert [hit.id for hit in without] == [hit.id for hit in with_content]


async def test_include_content_false_applies_to_query_batch(populated, collection):
    batches = await populated.query_batch(
        texts=["apple"], top_k=2, collection_name=collection, include_content=False
    )

    assert all(hit.content is None for hit in batches[0])


async def test_optional_methods_either_work_or_say_they_do_not(populated, collection):
    """A backend cannot quietly half-implement the optional tier."""
    if populated.supports("count_entries"):
        assert await populated.count_entries(collection) == len(CORPUS)
    else:
        with pytest.raises(NotImplementedError):
            await populated.count_entries(collection)

    if populated.supports("iter_entries"):
        entries = [entry async for entry in populated.iter_entries(collection)]
        assert len(entries) == len(CORPUS)
        assert all(entry["embedding"] for entry in entries)
    else:
        with pytest.raises(NotImplementedError):
            [entry async for entry in populated.iter_entries(collection)]


async def test_supports_is_false_for_an_unknown_capability(rag_service):
    assert rag_service.supports("teleportation") is False


async def test_compute_similarity_matrix_shape_and_ordering(populated, collection):
    matrix = await populated.compute_similarity_matrix(
        texts=["apple", "banana"],
        source_ids=["apples", "bananas", "cherries"],
        collection_name=collection,
    )

    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)
    # Each query is closest to its own source.
    assert matrix[0][0] == max(matrix[0])
    assert matrix[1][1] == max(matrix[1])


async def test_compute_similarity_matrix_with_no_input(populated, collection):
    assert (
        await populated.compute_similarity_matrix(
            texts=[], source_ids=["apples"], collection_name=collection
        )
        == []
    )


async def test_learn_normalizer_returns_a_normalizer(populated, collection):
    normalizer = await populated.learn_normalizer(collection_name=collection)

    assert hasattr(normalizer, "transform")


async def test_embedding_client_is_injectable(rag_service):
    """The property is how a caller substitutes a custom embedding provider."""
    sentinel = fake_embedding_client()
    rag_service.embedding_client = sentinel

    assert rag_service.embedding_client is sentinel


def test_unresolvable_embedding_provider_does_not_break_construction(tmp_path):
    """Resolution is lazy, so a caller still gets an object to inject into."""
    service = SqliteRagService(
        str(tmp_path / "lazy.db"), model="nosuchprovider/whatever"
    )

    service.embedding_client = fake_embedding_client()

    assert service.embedding_client is not None


# The registry-backed shape both SQL backends share: browse methods that work
# without an embedding model, and one table per collection.


async def test_list_collections_reports_model_dimension_and_count(
    populated, collection
):
    entry = next(
        c for c in await populated.list_collections() if c["name"] == collection
    )

    assert entry["model"] == "fake/embedding-model"
    assert entry["embedding_size"] == 4
    assert entry["count"] == len(CORPUS)
    assert entry["schema_version"] >= 1


async def test_get_stats_aggregates_the_registry(populated, collection):
    stats = await populated.get_stats()

    assert collection in stats["collections"]
    assert stats["total_collections"] == len(stats["collections"])
    assert stats["total_entries"] >= len(CORPUS)


async def test_create_and_drop_collection(rag_service, collection):
    await rag_service.create_collection(collection, embedding_size=4)
    assert collection in {c["name"] for c in await rag_service.list_collections()}
    assert await rag_service.count_entries(collection) == 0

    with pytest.raises(ValueError, match="4-dimensional"):
        await rag_service.create_collection(collection, embedding_size=3)

    await rag_service.drop_collection(collection)
    assert collection not in {c["name"] for c in await rag_service.list_collections()}
    await rag_service.drop_collection(collection)  # dropping twice is not an error


async def test_get_embeddings_by_ids(populated, collection):
    entries = [entry async for entry in populated.iter_entries(collection)]
    wanted = [entries[0]["id"], entries[-1]["id"]]

    by_id = await populated.get_embeddings_by_ids(collection, wanted)

    assert set(by_id) == set(wanted)
    assert by_id[entries[0]["id"]] == pytest.approx(entries[0]["embedding"], abs=1e-6)
    assert await populated.get_embeddings_by_ids(collection, []) == {}
    assert await populated.get_embeddings_by_ids("no_such_collection", wanted) == {}


async def test_a_service_without_a_model_can_browse_but_not_embed(
    populated, collection, request, tmp_path
):
    """The backoffice opens indexes it did not build, so the model is optional."""
    if isinstance(populated, SqliteRagService):
        browsing = SqliteRagService(populated.filename, model=None)
    else:
        browsing = PostgresRagService(populated.session_maker, schema=populated.schema)

    assert await browsing.count_entries(collection) == len(CORPUS)
    assert collection in {c["name"] for c in await browsing.list_collections()}
    with pytest.raises(ValueError, match="without an embedding model"):
        await browsing.query("apple", collection_name=collection)
    with pytest.raises(ValueError, match="without an embedding model"):
        await browsing.create_collection(f"{collection}_new", embedding_size=4)


async def test_no_collection_name_means_default(rag_service):
    """Both backends read ``None`` as the ``"default"`` collection."""
    await rag_service.index("apple", collection_name="default", source_id="a")
    await rag_service.index("banana", collection_name="elsewhere", source_id="b")

    hits = await rag_service.query("apple", top_k=5)

    assert [hit.content for hit in hits] == ["apple"]
    await rag_service.delete_by_source_id("default", "a")
    await rag_service.drop_collection("elsewhere")


async def test_querying_an_unknown_collection_is_empty(rag_service):
    assert await rag_service.query("apple", collection_name="never_created") == []
    assert await rag_service.count_entries("never_created") == 0
    assert [e async for e in rag_service.iter_entries("never_created")] == []


def test_rag_service_from_uri_picks_the_backend(tmp_path):
    from kavalai.rag import rag_service_from_uri

    sqlite = rag_service_from_uri(f"sqlite:///{tmp_path / 'idx.db'}")
    assert isinstance(sqlite, SqliteRagService)
    assert sqlite.model is None

    postgres = rag_service_from_uri(
        "postgresql://u:p@localhost:1/db", model="fake/embedding-model", schema="s"
    )
    assert isinstance(postgres, PostgresRagService)
    assert postgres.schema == "s" and postgres.model == "fake/embedding-model"
