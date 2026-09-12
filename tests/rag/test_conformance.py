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

import sqlite3
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text as sql_text

from kavalai.llm_clients.common import create_model_call_stat
from kavalai.normalizer import Normalizer
from kavalai.rag import (
    CollectionInfo,
    PostgresRagService,
    RagServiceResult,
    SqliteRagService,
)

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
    """An embedding client with fixed vectors, so rankings are predictable.

    It honours ``normalize``/``normalizer`` the way the real clients do, so
    the contract that a service's normalizer reaches the embedding side can
    be checked without a provider.
    """

    async def compute_embeddings(
        texts, *args, normalize=False, normalizer=None, **kwargs
    ):
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
        embeddings = [VECTORS[t] for t in texts]
        if normalize and normalizer is not None:
            embeddings = normalizer.transform(embeddings)
        return embeddings, stats

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


@pytest.fixture
def same_database(rag_service):
    """A factory of further services on ``rag_service``'s database.

    Two services over one database are how a model change, a DML-only runtime
    role or a concurrent provisioner look from the inside.
    """

    def factory(model="fake/embedding-model", *, inject=True, **options):
        if isinstance(rag_service, SqliteRagService):
            service = SqliteRagService(rag_service.filename, model=model, **options)
        else:
            service = PostgresRagService(
                rag_service.session_maker, model, schema=rag_service.schema, **options
            )
        if inject:
            service.embedding_client = fake_embedding_client()
        return service

    return factory


@pytest.fixture
async def fresh_database(rag_service, request, tmp_path):
    """A factory of services on a database the registry was never created in.

    SQLite gets a new file; Postgres a new, empty schema, so a missing
    registry is really missing rather than merely unused.
    """
    schema = f"fresh_{abs(hash(request.node.name)) % 10**8}"
    if isinstance(rag_service, PostgresRagService):
        async with rag_service.session_maker() as session:
            await session.execute(sql_text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await session.execute(sql_text(f'CREATE SCHEMA "{schema}"'))
            await session.commit()

    def factory(model="fake/embedding-model", **options):
        if isinstance(rag_service, SqliteRagService):
            service = SqliteRagService(
                str(tmp_path / "fresh.db"), model=model, **options
            )
        else:
            service = PostgresRagService(
                rag_service.session_maker, model, schema=schema, **options
            )
        service.embedding_client = fake_embedding_client()
        return service

    yield factory

    if isinstance(rag_service, PostgresRagService):
        async with rag_service.session_maker() as session:
            await session.execute(sql_text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await session.commit()


async def registry_exists(service) -> bool:
    """Whether ``rag_collections`` exists, asked of the database directly."""
    if isinstance(service, SqliteRagService):
        conn = sqlite3.connect(service.filename)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'rag_collections'"
            ).fetchone()
        finally:
            conn.close()
        return row is not None
    async with service.session_maker() as session:
        return (
            await session.execute(
                sql_text("SELECT to_regclass(:name) IS NOT NULL").bindparams(
                    name=f'"{service.schema}"."rag_collections"'
                )
            )
        ).scalar()


class StatsCollector:
    """A model-stats receiver that keeps what it is given."""

    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


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


async def test_a_service_without_a_model_uses_the_collection_model(
    populated, collection, request, tmp_path
):
    """The backoffice opens indexes it did not build, so the model is optional.

    Existing collections are embedded with the model the registry records for
    them; only creating a collection needs a model of the service's own.
    """
    if isinstance(populated, SqliteRagService):
        browsing = SqliteRagService(populated.filename, model=None)
    else:
        browsing = PostgresRagService(populated.session_maker, schema=populated.schema)
    browsing.embedding_client = fake_embedding_client()

    assert await browsing.count_entries(collection) == len(CORPUS)
    assert collection in {c["name"] for c in await browsing.list_collections()}
    hits = await browsing.query("apple", top_k=1, collection_name=collection)
    assert [hit.model for hit in hits] == [populated.model]
    with pytest.raises(ValueError, match="without an embedding model"):
        await browsing.create_collection(f"{collection}_new", embedding_size=4)
    with pytest.raises(ValueError, match="without an embedding model"):
        await browsing.index("apple", collection_name=f"{collection}_new")


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


class RecordingNormalizer(Normalizer):
    """A normalizer that only records what it was asked to transform."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def transform(self, embeddings):
        self.calls += 1
        return [list(vector) for vector in embeddings]


async def test_a_service_normalizer_reaches_both_indexing_and_querying(
    rag_service, collection
):
    """A normalizer given to a service is applied — it used to be handed to
    the client without ``normalize=True``, which every client treats as
    "do nothing"."""
    normalizer = RecordingNormalizer()
    rag_service.normalizer = normalizer

    await rag_service.index_batch(
        texts=["apple", "banana"],
        metadata_list=[{}, {}],
        source_ids=["a", "b"],
        collection_name=collection,
    )
    assert normalizer.calls == 1

    await rag_service.query("apple", collection_name=collection, top_k=1)
    assert normalizer.calls == 2


async def test_without_a_normalizer_nothing_is_normalized(rag_service, collection):
    await rag_service.index("apple", {}, collection_name=collection, source_id="a")
    _, kwargs = rag_service.embedding_client.compute_embeddings.call_args
    assert kwargs["normalize"] is False and kwargs["normalizer"] is None


# An empty filter matches nothing, and a known-empty answer costs no embedding.


async def test_an_empty_source_ids_matches_nothing_without_embedding(
    populated, collection
):
    """``[]`` is a filter that excludes everything, never "no filter".

    A pre-filter that found no candidates must not widen into a search of the
    whole collection, and the answer is known before any embedding is paid.
    """
    client = populated.embedding_client
    client.compute_embeddings.reset_mock()

    assert (
        await populated.query("apple", collection_name=collection, source_ids=[]) == []
    )
    assert await populated.query_batch(
        texts=["apple", "banana"], collection_name=collection, source_ids=[]
    ) == [[], []]
    client.compute_embeddings.assert_not_awaited()

    everything = await populated.query(
        "apple", top_k=10, collection_name=collection, source_ids=None
    )
    assert len(everything) == len(CORPUS)


async def test_a_missing_collection_is_answered_without_embedding(rag_service):
    client = rag_service.embedding_client
    client.compute_embeddings.reset_mock()

    assert await rag_service.query("apple", collection_name="never_created") == []
    assert await rag_service.query_batch(
        texts=["apple", "banana"], collection_name="never_created"
    ) == [[], []]
    client.compute_embeddings.assert_not_awaited()


async def test_index_batch_with_empty_source_ids_is_a_length_error(
    rag_service, collection
):
    with pytest.raises(ValueError, match="texts and source_ids"):
        await rag_service.index_batch(
            texts=["apple"],
            metadata_list=[{}],
            source_ids=[],
            collection_name=collection,
        )


# Batch and metadata deletion.


async def test_delete_many_removes_exactly_the_given_ids(populated, collection):
    entries = [entry async for entry in populated.iter_entries(collection)]
    apples = [e["id"] for e in entries if e["source_id"] == "apples"]

    await populated.delete_many(apples, collection_name=collection)

    remaining = [entry async for entry in populated.iter_entries(collection)]
    assert {e["source_id"] for e in remaining} == {"bananas", "cherries"}
    await populated.delete_many([], collection_name=collection)
    await populated.delete_many([uuid4()], collection_name=collection)
    assert await populated.count_entries(collection) == 2


async def test_delete_many_without_a_collection_searches_every_one(fresh_database):
    service = fresh_database()
    assert await service.delete_many([uuid4()]) is None  # no registry yet

    first = await service.index("apple", collection_name="one", source_id="a")
    second = await service.index("banana", collection_name="two", source_id="b")
    await service.index("cherry", collection_name="two", source_id="c")

    await service.delete_many([first["id"], second["id"]])

    assert await service.count_entries("one") == 0
    assert await service.count_entries("two") == 1


METADATA_ROWS = [
    ("apple", {"page": "p1", "n": 1, "x": 1.5, "flag": True}),
    ("apple pie", {"page": "p1", "n": 2, "x": 2.5, "flag": False}),
    ("apple tart", {"page": "p2", "n": 3, "x": 3.5, "flag": True}),
    ("banana", {"page": "p2", "n": "1", "x": 1, "flag": 1}),
    ("cherry", {"page": "p3", "nested": {"page": "p1"}}),
]


@pytest.fixture
async def with_metadata(rag_service, collection):
    """Rows whose metadata mixes strings, integers, floats and booleans."""
    await rag_service.index_batch(
        texts=[text for text, _ in METADATA_ROWS],
        metadata_list=[meta for _, meta in METADATA_ROWS],
        source_ids=[text for text, _ in METADATA_ROWS],
        collection_name=collection,
    )
    return rag_service


async def contents(service, collection) -> set[str]:
    return {entry["content"] async for entry in service.iter_entries(collection)}


@pytest.mark.parametrize(
    "match, deleted",
    [
        ({"page": "p1"}, {"apple", "apple pie"}),
        ({"page": "p2", "n": 3}, {"apple tart"}),
        ({"n": 2}, {"apple pie"}),
        ({"x": 2.5}, {"apple pie"}),
        ({"x": 1}, {"banana"}),
        ({"x": 1.0}, {"banana"}),
        ({"flag": False}, {"apple pie"}),
        ({"page": "nowhere"}, set()),
    ],
)
async def test_delete_by_metadata_matches_top_level_scalars(
    with_metadata, collection, match, deleted
):
    before = await contents(with_metadata, collection)

    await with_metadata.delete_by_metadata(collection, match)

    assert before - await contents(with_metadata, collection) == deleted


async def test_delete_by_metadata_keeps_booleans_and_numbers_apart(
    with_metadata, collection
):
    """JSON ``true`` is not the number 1, and ``"1"`` is not 1, on any backend."""
    await with_metadata.delete_by_metadata(collection, {"flag": True})
    assert "banana" in await contents(with_metadata, collection)  # flag is 1

    await with_metadata.delete_by_metadata(collection, {"n": 1})
    assert "banana" in await contents(with_metadata, collection)  # n is "1"
    assert await contents(with_metadata, collection) == {
        "apple pie",
        "banana",
        "cherry",
    }

    await with_metadata.delete_by_metadata(collection, {"flag": 1})
    assert "banana" not in await contents(with_metadata, collection)


@pytest.mark.parametrize(
    "match, message",
    [
        ({}, "non-empty dict"),
        (["page"], "non-empty dict"),
        ({1: "a"}, "non-empty string"),
        ({"": "a"}, "non-empty string"),
        ({'pa"ge': "a"}, "without double quotes"),
        ({"page": ["p1"]}, "string, number or boolean, not list"),
        ({"page": {"id": "p1"}}, "string, number or boolean, not dict"),
        ({"page": None}, "string, number or boolean, not NoneType"),
    ],
)
async def test_delete_by_metadata_refuses_what_it_cannot_match(
    populated, collection, match, message
):
    with pytest.raises(ValueError, match=message):
        await populated.delete_by_metadata(collection, match)
    assert await populated.count_entries(collection) == len(CORPUS)


async def test_delete_by_metadata_on_a_missing_collection_is_not_an_error(
    rag_service,
):
    await rag_service.delete_by_metadata("never_created", {"page": "p1"})


async def test_the_deletion_tier_is_advertised(rag_service):
    assert rag_service.supports("delete_by_metadata")
    assert rag_service.supports("replace")


# Replace: embed first, then delete and insert in one transaction.


async def test_replace_by_match(populated, collection):
    rows = await populated.replace(
        collection,
        ["durian"],
        [{"fruit": "apples"}],
        match={"fruit": "apples"},
    )

    assert [row["content"] for row in rows] == ["durian"]
    assert GUARANTEED_INDEX_KEYS <= set(rows[0])
    assert await contents(populated, collection) == {"durian", "banana", "cherry"}


async def test_replace_by_source_id(populated, collection):
    await populated.replace(
        collection,
        ["banana", "cherry"],
        [{}, {}],
        source_ids=["bananas", "bananas"],
        source_id="bananas",
    )

    entries = [entry async for entry in populated.iter_entries(collection)]
    by_source = {}
    for entry in entries:
        by_source.setdefault(entry["source_id"], []).append(entry["content"])
    assert sorted(by_source["bananas"]) == ["banana", "cherry"]
    assert len(by_source["apples"]) == 3


async def test_replace_with_match_and_source_id_deletes_both(populated, collection):
    await populated.replace(
        collection,
        ["durian"],
        [{}],
        source_ids=["durians"],
        match={"fruit": "apples"},
        source_id="cherries",
    )

    assert await contents(populated, collection) == {"banana", "durian"}


async def test_replace_with_no_texts_deletes_the_selection_without_embedding(
    populated, collection
):
    client = populated.embedding_client
    client.compute_embeddings.reset_mock()

    assert await populated.replace(collection, [], [], source_id="apples") == []

    client.compute_embeddings.assert_not_awaited()
    assert await contents(populated, collection) == {"banana", "cherry"}


async def test_replace_into_a_missing_collection(rag_service, collection):
    assert await rag_service.replace(collection, [], [], source_id="a") == []

    await rag_service.replace(
        collection, ["apple"], [{"page": "p1"}], match={"page": "p1"}
    )

    assert await contents(rag_service, collection) == {"apple"}


async def test_replace_needs_a_valid_selection(populated, collection):
    with pytest.raises(ValueError, match="needs match= or source_id="):
        await populated.replace(collection, ["apple"], [{}])
    with pytest.raises(ValueError, match="non-empty dict"):
        await populated.replace(collection, ["apple"], [{}], match={})
    with pytest.raises(ValueError, match="texts and metadata"):
        await populated.replace(collection, ["apple"], [], source_id="apples")
    with pytest.raises(ValueError, match="texts and source_ids"):
        await populated.replace(
            collection, ["apple"], [{}], source_ids=[], source_id="apples"
        )
    assert await populated.count_entries(collection) == len(CORPUS)


async def test_a_failed_replace_leaves_the_old_rows(populated, collection, monkeypatch):
    """The delete and the insert commit together or not at all."""

    async def failing_insert(conn, info, rows):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(populated, "_insert_rows", failing_insert)

    with pytest.raises(RuntimeError, match="insert failed"):
        await populated.replace(collection, ["durian"], [{}], match={"fruit": "apples"})

    monkeypatch.undo()
    assert await contents(populated, collection) == {text for text, _ in CORPUS}
    # The service is still usable afterwards: nothing was left half-committed.
    await populated.replace(collection, ["durian"], [{}], match={"fruit": "apples"})
    assert await contents(populated, collection) == {"durian", "banana", "cherry"}


# Similarity threshold.


async def test_min_similarity_drops_weaker_hits(populated, collection):
    hits = await populated.query(
        "apple", top_k=5, collection_name=collection, min_similarity=0.5
    )

    assert [hit.content for hit in hits] == ["apple", "apple pie", "apple tart"]
    assert all(hit.similarity >= 0.5 for hit in hits)


async def test_min_similarity_applies_to_every_query_of_a_batch(populated, collection):
    batches = await populated.query_batch(
        texts=["apple", "durian"],
        top_k=5,
        collection_name=collection,
        min_similarity=0.5,
    )

    assert len(batches[0]) == 3
    assert batches[1] == []  # durian is orthogonal to everything indexed


# Embedding statistics go to a receiver, never into a table of the service's.


async def test_a_per_call_stats_receiver_gets_every_embedding_call(
    rag_service, collection
):
    collector = StatsCollector()

    await rag_service.index(
        "apple", collection_name=collection, stats_receiver=collector
    )
    await rag_service.index_batch(
        texts=["banana", "cherry"],
        metadata_list=[{}, {}],
        collection_name=collection,
        stats_receiver=collector,
    )
    await rag_service.query(
        "apple", collection_name=collection, stats_receiver=collector
    )
    await rag_service.query_batch(
        texts=["apple", "banana"], collection_name=collection, stats_receiver=collector
    )
    await rag_service.replace(
        collection, ["durian"], [{}], source_id="x", stats_receiver=collector
    )

    assert [stat.batch_size for stat in collector.stats] == [1, 2, 1, 2, 1]
    assert {stat.call_type for stat in collector.stats} == {"embedding"}


async def test_the_constructor_receiver_is_the_default(same_database, collection):
    default = StatsCollector()
    service = same_database(stats_receiver=default)
    await service.index("apple", collection_name=collection)
    assert len(default.stats) == 1

    per_call = StatsCollector()
    await service.query("apple", collection_name=collection, stats_receiver=per_call)
    assert (len(default.stats), len(per_call.stats)) == (1, 1)


async def test_without_a_receiver_nothing_is_reported(rag_service, collection):
    assert rag_service.stats_receiver is None
    await rag_service.index("apple", collection_name=collection)
    hits = await rag_service.query("apple", collection_name=collection)
    assert [hit.content for hit in hits] == ["apple"]


# The registry's model is authoritative.


@pytest.fixture
def requested_models(monkeypatch):
    """Record every model an embedding client is built for."""
    requested = []

    def make_client(model):
        requested.append(model)
        return fake_embedding_client()

    monkeypatch.setattr("kavalai.rag.collections.make_embedding_client", make_client)
    return requested


async def test_an_existing_collection_is_embedded_with_its_recorded_model(
    populated, collection, same_database, requested_models, caplog
):
    other = same_database("fake/model-b", inject=False)

    hits = await other.query("apple", top_k=1, collection_name=collection)
    await other.query("banana", top_k=1, collection_name=collection)
    await other.index("cherry", collection_name=collection)

    assert requested_models == ["fake/embedding-model"]
    assert [hit.model for hit in hits] == ["fake/embedding-model"]
    assert (
        f"RAG collection '{collection}' was built with fake/embedding-model; it "
        f"is embedded with that model, not with this service's fake/model-b."
    ) in caplog.text

    await other.index("apple", collection_name=f"{collection}_b")
    assert requested_models == ["fake/embedding-model", "fake/model-b"]
    listed = {c["name"]: c["model"] for c in await other.list_collections()}
    assert listed[f"{collection}_b"] == "fake/model-b"
    await other.drop_collection(f"{collection}_b")


async def test_the_embedding_client_property_follows_the_service_model(
    same_database, requested_models
):
    service = same_database("fake/model-b", inject=False)
    assert service.embedding_client is service.embedding_client
    assert requested_models == ["fake/model-b"]

    modelless = same_database(None, inject=False)
    with pytest.raises(ValueError, match="without an embedding model"):
        _ = modelless.embedding_client


async def test_an_assigned_client_serves_every_model(
    populated, collection, same_database, requested_models
):
    other = same_database("fake/model-b")
    await other.query("apple", collection_name=collection)
    await other.index("apple", collection_name=f"{collection}_b")

    assert requested_models == []
    assert other.embedding_client.compute_embeddings.await_count == 2
    await other.drop_collection(f"{collection}_b")


async def test_create_collection_records_the_model_it_is_given(
    same_database, collection
):
    modelless = same_database(None)
    await modelless.create_collection(
        collection, embedding_size=4, model="fake/explicit-model"
    )

    listed = {c["name"]: c["model"] for c in await modelless.list_collections()}
    assert listed[collection] == "fake/explicit-model"
    await modelless.drop_collection(collection)


async def test_a_concurrently_created_collection_is_not_overridden(
    rag_service, collection, monkeypatch
):
    """The registry keeps the first row; the loser must not use its own idea."""
    original = rag_service._create_collection_table

    async def create_while_another_process_registers(conn, info):
        await original(conn, info)
        await rag_service._insert_registry_row(
            conn,
            CollectionInfo(
                name=info.name,
                table_name=info.table_name,
                model="fake/other-model",
                embedding_size=info.embedding_size,
                schema_version=info.schema_version,
            ),
        )

    monkeypatch.setattr(
        rag_service, "_create_collection_table", create_while_another_process_registers
    )

    with pytest.raises(ValueError, match="created concurrently with fake/other-model"):
        await rag_service.index("apple", collection_name=collection)

    monkeypatch.undo()
    assert collection not in rag_service._collections
    listed = {c["name"]: c["model"] for c in await rag_service.list_collections()}
    assert listed[collection] == "fake/other-model"
    await rag_service.drop_collection(collection)


async def test_a_concurrent_creation_with_the_same_shape_is_accepted(
    rag_service, collection, monkeypatch
):
    original = rag_service._create_collection_table

    async def create_while_another_process_registers(conn, info):
        await original(conn, info)
        await rag_service._insert_registry_row(conn, info)

    monkeypatch.setattr(
        rag_service, "_create_collection_table", create_while_another_process_registers
    )

    await rag_service.index("apple", collection_name=collection)

    assert await rag_service.count_entries(collection) == 1


# Provisioning.


async def test_read_paths_never_create_the_registry(fresh_database):
    """A read that performs DDL is a bug whatever the roles, provision or not."""
    service = fresh_database()

    assert await service.list_collections() == []
    assert await service.get_stats() == {
        "total_entries": 0,
        "total_collections": 0,
        "collections": [],
    }
    assert await service.count_entries("default") == 0
    assert await service.query("apple") == []
    assert [e async for e in service.iter_entries("default")] == []
    assert await service.get_embeddings_by_ids("default", [uuid4()]) == {}
    await service.delete(uuid4())
    await service.delete_by_source_id("default", "a")
    await service.delete_by_metadata("default", {"page": "p1"})
    await service.drop_collection("default")

    assert await registry_exists(service) is False


async def test_without_provisioning_the_service_issues_no_ddl(fresh_database):
    service = fresh_database(provision=False)
    client = service.embedding_client

    assert await service.list_collections() == []
    assert await service.query("apple", collection_name="docs") == []
    with pytest.raises(RuntimeError, match=r"provision=False.*create_collection\(\)"):
        await service.index("apple", collection_name="docs")
    with pytest.raises(RuntimeError, match="provision=False"):
        await service.replace("docs", ["apple"], [{}], source_id="a")
    assert await service.replace("docs", [], [], source_id="a") == []
    # The refusal is known before the texts are embedded, so none are.
    client.compute_embeddings.assert_not_awaited()
    assert await registry_exists(service) is False

    modelless = fresh_database(None, provision=False)
    with pytest.raises(RuntimeError, match=r"create_collection\(\)"):
        await modelless.index("apple", collection_name="docs")
    assert await modelless.replace("docs", [], [], match={"page": "p1"}) == []

    await service.ensure_registry()
    assert await registry_exists(service) is True
    assert await service.list_collections() == []
    with pytest.raises(RuntimeError, match="does not exist"):
        await service.index("apple", collection_name="docs")

    await service.create_collection("docs", embedding_size=4)
    await service.index("apple", collection_name="docs")
    assert await service.count_entries("docs") == 1


async def test_ensure_registry_is_idempotent(fresh_database):
    service = fresh_database(provision=False)
    await service.ensure_registry()
    await service.ensure_registry()
    assert await registry_exists(service) is True


async def test_a_dml_only_service_uses_what_an_owner_provisioned(
    fresh_database,
):
    owner = fresh_database()
    await owner.ensure_registry()
    await owner.create_collection("docs", embedding_size=4)

    runtime = fresh_database(provision=False)
    await runtime.index_batch(
        texts=["apple", "banana"], metadata_list=[{}, {}], collection_name="docs"
    )
    hits = await runtime.query("apple", top_k=1, collection_name="docs")
    assert [hit.content for hit in hits] == ["apple"]
    await runtime.delete_by_metadata("docs", {"page": "none"})
    await runtime.replace("docs", ["cherry"], [{}], source_id="default")
    assert await runtime.count_entries("docs") == 1


async def test_without_provisioning_a_stale_collection_is_not_upgraded(
    same_database, collection
):
    owner = same_database()
    await owner.create_collection(collection, embedding_size=4)
    async with owner._connection() as conn:
        await owner._set_registry_version(conn, collection, 0)
        await owner._commit(conn)

    runtime = same_database(provision=False)
    with pytest.raises(RuntimeError, match="needs an upgrade from schema_version 0"):
        await runtime.count_entries(collection)

    async with owner._connection() as conn:
        await owner._set_registry_version(conn, collection, 1)
        await owner._commit(conn)
    await owner.drop_collection(collection)


# Vector types.


async def test_an_unknown_vector_type_is_refused(
    rag_service, same_database, collection
):
    with pytest.raises(ValueError, match="not 'float8'"):
        same_database(vector_type="float8")
    with pytest.raises(ValueError, match="not 'float8'"):
        await rag_service.create_collection(
            collection, embedding_size=4, vector_type="float8"
        )


def test_rag_service_from_uri_forwards_options(tmp_path):
    from kavalai.rag import rag_service_from_uri

    receiver = StatsCollector()
    sqlite = rag_service_from_uri(
        f"sqlite:///{tmp_path / 'idx.db'}", provision=False, stats_receiver=receiver
    )
    assert (sqlite.provision, sqlite.stats_receiver) == (False, receiver)

    postgres = rag_service_from_uri(
        "postgresql://u:p@localhost:1/db", vector_type="halfvec", provision=False
    )
    assert (postgres.vector_type, postgres.provision) == ("halfvec", False)


async def test_delete_by_source_id_with_an_empty_list_deletes_nothing(
    populated, collection
):
    await populated.delete_by_source_id(collection, [])
    assert await populated.count_entries(collection) == len(CORPUS)


async def test_a_collection_without_an_upgrade_path_is_refused(
    same_database, collection
):
    owner = same_database()
    await owner.create_collection(collection, embedding_size=4)
    async with owner._connection() as conn:
        await owner._set_registry_version(conn, collection, 0)
        await owner._commit(conn)

    with pytest.raises(ValueError, match="No upgrade step registered"):
        await same_database().count_entries(collection)

    async with owner._connection() as conn:
        await owner._set_registry_version(conn, collection, 1)
        await owner._commit(conn)
    await owner.drop_collection(collection)


async def test_create_collection_on_an_existing_collection_is_idempotent(
    populated, collection
):
    """An owner job re-runs it after an upgrade; it must not fail or reset."""
    await populated.create_collection(collection, embedding_size=4)
    assert await populated.count_entries(collection) == len(CORPUS)
