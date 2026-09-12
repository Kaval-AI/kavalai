"""Postgres-specific behaviour of ``PostgresRagService``.

The declared ``BaseRagService`` contract is exercised against this backend in
``test_conformance.py``; what remains here is what only the Postgres backend
does -- the table-per-collection registry, the batch CTE, joins against
foreign tables, ``method=`` on the similarity matrix -- plus the purely mocked
normalizer tests.

Every test that talks to the container injects a deterministic embedding
client, so the suite never reaches a provider and runs without an API key.
"""

import hashlib
import math
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from kavalai.db import ModelCallStat, db_manager
from kavalai.llm_clients.common import create_model_call_stat
from kavalai.normalizer import Normalizer
from kavalai.rag import PostgresRagService

EMBEDDING_MODEL = "fake/embedding-model"
EMBEDDING_DIM = 32


def fake_embedding(text_value: str) -> list[float]:
    """A unit bag-of-words vector for any text.

    Each lower-cased word is hashed onto one of the dimensions, so identical
    texts embed identically, texts sharing a word are close, and unrelated
    texts are nearly orthogonal. Dimension 0 is a small constant so a text
    made only of unseen words still has a non-zero vector: pgvector's cosine
    distance to a zero vector is NaN, which would turn a "no match" query into
    a database error rather than an empty result.
    """
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = 0.1
    for word in text_value.lower().split():
        digest = hashlib.sha1(word.encode()).digest()
        index = 1 + int.from_bytes(digest[:4], "big") % (EMBEDDING_DIM - 1)
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def fake_embedding_client():
    """An embedding client that embeds any text without a network call."""

    async def compute_embeddings(texts, *args, **kwargs):
        # A real ModelCallStat, not a mock: the service persists what it is
        # handed, so a mock would fail inside SQLAlchemy.
        stats = create_model_call_stat(
            call_type="embedding",
            model=EMBEDDING_MODEL,
            duration_seconds=0.0,
            batch_size=len(texts),
            total_tokens=len(texts),
        )
        return [fake_embedding(t) for t in texts], stats

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=compute_embeddings)
    return client


@pytest.fixture
def embedding_model():
    return EMBEDDING_MODEL


@pytest.fixture
def service(agents_db_config, migrated_agents_db, embedding_model):
    """A service on the shared test database with the fake embedding client."""
    svc = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    svc.embedding_client = fake_embedding_client()
    return svc


@pytest.fixture
def collection(request):
    """A collection name unique to the test, as the database is shared."""
    return f"pg_{abs(hash(request.node.name)) % 10**8}"


def test_fake_embedding_is_deterministic_and_word_based():
    assert fake_embedding("apple") == fake_embedding("Apple")
    assert len(fake_embedding("anything")) == EMBEDDING_DIM
    assert _dot(fake_embedding("apple"), fake_embedding("banana")) < 0.05
    assert _dot(fake_embedding("apple"), fake_embedding("apple pie")) > 0.6


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.asyncio
async def test_rag_service_with_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    # Mock session maker
    mock_session_maker = MagicMock()
    mock_session_maker.return_value.__aenter__.return_value = mock_session

    model = "openai/text-embedding-3-small"
    normalizer = Normalizer(l2=True)

    # We need to mock LLMClient's compute_embeddings in PostgresRagService
    with patch("kavalai.rag.collections.make_embedding_client") as mock_llm_client_cls:
        mock_llm_client = mock_llm_client_cls.return_value
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_llm_client.compute_embeddings = AsyncMock(
            return_value=([[0.1, 0.2, 0.3]], mock_stats)
        )

        # Initialize PostgresRagService with normalizer
        # Pass a session instead of URI to avoid real DB engine creation
        @asynccontextmanager
        async def session_factory():
            yield mock_session

        service = PostgresRagService(
            session_maker=session_factory, model=model, normalizer=normalizer
        )
        # Seed the collection registry cache: these tests only verify that the
        # normalizer is forwarded to compute_embeddings, not provisioning.
        from kavalai.rag.postgres import CollectionInfo

        service._registry_ready = True
        for name in ("test_coll", "default"):
            service._collections[name] = CollectionInfo(
                name=name,
                table_name=f"rag_c_{name}",
                model=model,
                embedding_size=3,
                schema_version=1,
            )

        assert service.normalizer == normalizer

        # 1. Test index_batch
        await service.index_batch(
            texts=["test"], metadata_list=[{}], collection_name="test_coll"
        )

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["test"], normalize=True, normalizer=normalizer
        )

        # 2. Test query
        mock_llm_client.compute_embeddings.reset_mock()
        await service.query("test query")

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["test query"], normalize=True, normalizer=normalizer
        )

        # 3. Test compute_similarity_matrix
        mock_llm_client.compute_embeddings.reset_mock()
        await service.compute_similarity_matrix(
            texts=["t1"], source_ids=["s1"], collection_name="default"
        )

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["t1"], normalize=True, normalizer=normalizer
        )


@pytest.mark.asyncio
async def test_rag_service_without_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    model = "openai/text-embedding-3-small"

    with patch("kavalai.rag.collections.make_embedding_client") as mock_llm_client_cls:
        mock_llm_client = mock_llm_client_cls.return_value
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_llm_client.compute_embeddings = AsyncMock(
            return_value=([[0.1, 0.2, 0.3]], mock_stats)
        )

        # Initialize PostgresRagService without normalizer
        @asynccontextmanager
        async def session_factory():
            yield mock_session

        service = PostgresRagService(session_maker=session_factory, model=model)
        from kavalai.rag.postgres import CollectionInfo

        service._registry_ready = True
        service._collections["default"] = CollectionInfo(
            name="default",
            table_name="rag_c_default",
            model=model,
            embedding_size=3,
            schema_version=1,
        )

        assert service.normalizer is None

        await service.query("test query")

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["test query"], normalize=False, normalizer=None
        )


@pytest.mark.asyncio
async def test_rag_service_keep_best_duplicates(service, collection):
    """keep_best survives ties: identical rows for one source yield one hit.

    The Postgres query uses DISTINCT ON over the source id, and a tie on the
    distance must not leak a second row.
    """
    texts = ["Tesla Model 3 is an electric car", "Tesla Model 3 is an electric car"]
    await service.index_batch(
        texts=texts,
        metadata_list=[{"brand": "Tesla"}, {"brand": "Tesla"}],
        collection_name=collection,
        source_ids=["tesla_3", "tesla_3"],
    )

    results = await service.query(
        "Tesla electric car", top_k=10, collection_name=collection, keep_best=True
    )

    assert len(results) == 1
    assert results[0].source_id == "tesla_3"


@pytest.mark.asyncio
async def test_rag_service_top_k_with_keep_best(service, collection):
    """top_k counts sources after keep_best, not rows before it.

    Applying LIMIT before the DISTINCT ON would return fewer sources than
    asked for whenever one source has several close rows.
    """
    texts = []
    source_ids = []
    for i in range(1, 6):
        texts.extend([f"content {i} a", f"content {i} b"])
        source_ids.extend([f"source_{i}", f"source_{i}"])

    await service.index_batch(
        texts=texts,
        metadata_list=[{} for _ in texts],
        collection_name=collection,
        source_ids=source_ids,
    )

    top_k = 3
    results = await service.query(
        "content", top_k=top_k, collection_name=collection, keep_best=True
    )
    assert len(results) == top_k
    assert len({r.source_id for r in results}) == top_k

    results_1 = await service.query(
        "content", top_k=1, collection_name=collection, keep_best=True
    )
    assert len(results_1) == 1


@pytest.mark.asyncio
async def test_compute_similarity_matrix_methods(service, collection):
    """``method="min"`` takes the closest row per source, ``"avg"`` the mean."""
    texts = ["apple", "apple pie", "banana", "banana bread", "cherry"]
    source_ids = ["fruit_1", "fruit_1", "fruit_2", "fruit_2", "fruit_3"]
    await service.index_batch(
        texts=texts,
        metadata_list=[{}] * len(texts),
        collection_name=collection,
        source_ids=source_ids,
    )

    queries = ["apple", "banana"]
    target_source_ids = ["fruit_1", "fruit_2", "fruit_3", "fruit_nonexistent"]

    matrix_min = await service.compute_similarity_matrix(
        texts=queries,
        source_ids=target_source_ids,
        method="min",
        collection_name=collection,
    )
    assert len(matrix_min) == 2
    assert all(len(row) == 4 for row in matrix_min)
    # "apple" is indexed verbatim under fruit_1, so its best row is an exact hit.
    assert matrix_min[0][0] > 0.9
    assert matrix_min[0][0] > matrix_min[0][1]
    # A source with no rows scores zero rather than being omitted.
    assert matrix_min[0][3] == 0.0

    matrix_avg = await service.compute_similarity_matrix(
        texts=queries,
        source_ids=target_source_ids,
        method="avg",
        collection_name=collection,
    )
    assert len(matrix_avg) == 2
    assert all(len(row) == 4 for row in matrix_avg)
    # fruit_1 also holds "apple pie", which drags the mean below the best row.
    assert matrix_avg[0][0] < matrix_min[0][0]


@pytest.mark.asyncio
async def test_rag_service_batch_query_with_join(service, collection):
    """The batch CTE joins a foreign table and honours its WHERE clause."""
    products_table = f"products_{collection}"
    async with service.session_maker() as session:
        await session.execute(text(f"DROP TABLE IF EXISTS {products_table}"))
        await session.execute(
            text(
                f"""
                CREATE TABLE {products_table} (
                    id UUID PRIMARY KEY,
                    name VARCHAR,
                    category VARCHAR,
                    price FLOAT,
                    in_stock BOOLEAN
                )
                """
            )
        )
        product_data = [
            (uuid.uuid4(), "Wireless Headphones", "electronics", 99.99, True),
            (uuid.uuid4(), "Laptop Stand", "electronics", 49.99, True),
            (uuid.uuid4(), "Office Chair", "furniture", 299.99, True),
            (uuid.uuid4(), "Desk Lamp", "electronics", 29.99, False),
            (uuid.uuid4(), "USB Cable", "electronics", 9.99, True),
        ]
        for product_id, name, category, price, in_stock in product_data:
            await session.execute(
                text(
                    f"INSERT INTO {products_table} "
                    "(id, name, category, price, in_stock) "
                    "VALUES (:id, :name, :category, :price, :in_stock)"
                ),
                {
                    "id": product_id,
                    "name": name,
                    "category": category,
                    "price": price,
                    "in_stock": in_stock,
                },
            )
        await session.commit()

    await service.index_batch(
        texts=[p[1] for p in product_data],
        metadata_list=[{}] * len(product_data),
        collection_name=collection,
        source_ids=[str(p[0]) for p in product_data],
    )
    join = dict(
        collection_name=collection,
        join_table=f"{products_table} p",
        join_condition="p.id::text = r.source_id",
    )

    results_no_filter = await service.batch_query_with_join(
        texts=["headphones"],
        top_k=5,
        join_columns=["p.name", "p.category", "p.price", "p.in_stock"],
        **join,
    )
    assert len(results_no_filter) == 1
    assert len(results_no_filter[0]) > 0
    assert results_no_filter[0][0]["name"] == "Wireless Headphones"

    results_electronics = await service.batch_query_with_join(
        texts=["headphones", "chair"],
        top_k=5,
        join_columns=["p.name", "p.category", "p.price"],
        additional_where="p.category = 'electronics'",
        **join,
    )
    assert len(results_electronics) == 2
    assert all(r["category"] == "electronics" for r in results_electronics[0])
    assert all(r["category"] == "electronics" for r in results_electronics[1])
    assert "Office Chair" not in [r["name"] for r in results_electronics[1]]

    results_in_stock = await service.batch_query_with_join(
        texts=["lamp"],
        top_k=5,
        join_columns=["p.name", "p.in_stock"],
        additional_where="p.in_stock = true",
        **join,
    )
    assert len(results_in_stock) == 1
    assert "Desk Lamp" not in [r["name"] for r in results_in_stock[0]]

    results_combined = await service.batch_query_with_join(
        texts=["electronics"],
        top_k=5,
        join_columns=["p.name", "p.category", "p.price", "p.in_stock"],
        additional_where=(
            "p.category = 'electronics' AND p.in_stock = true AND p.price < 100"
        ),
        **join,
    )
    assert len(results_combined) == 1
    assert len(results_combined[0]) > 0
    for result in results_combined[0]:
        assert result["category"] == "electronics"
        assert result["in_stock"] is True
        assert result["price"] < 100

    results_empty = await service.batch_query_with_join(
        texts=["nonexistent product xyz"],
        top_k=5,
        join_columns=["p.name"],
        additional_where="p.category = 'nonexistent'",
        **join,
    )
    assert results_empty == [[]]

    async with service.session_maker() as session:
        await session.execute(text(f"DROP TABLE IF EXISTS {products_table}"))
        await session.commit()


@pytest.mark.asyncio
async def test_build_batch_query_cte(service, collection):
    """The CTE text names the collection table and the per-query vectors."""
    embeddings = [[0.1] * EMBEDDING_DIM, [0.2] * EMBEDDING_DIM]

    # The CTE builder resolves the collection's table, so it must exist.
    await service.create_collection(collection, embedding_size=EMBEDDING_DIM)
    try:
        await _assert_batch_query_cte(service, embeddings, collection)
    finally:
        await service.drop_collection(collection)


async def _assert_batch_query_cte(service, embeddings, collection):
    table_name = PostgresRagService.table_name_for_collection(collection)

    cte_sql, params = service.build_batch_query_cte(
        embeddings=embeddings, top_k=5, collection_name=collection
    )
    assert "rag_results AS" in cte_sql
    assert "unnest" in cte_sql.lower()
    assert "CROSS JOIN LATERAL" in cte_sql
    assert "WITH ORDINALITY" in cte_sql
    assert table_name in cte_sql
    assert params["top_k"] == 5
    assert "vector_0" in params
    assert "vector_1" in params

    cte_sql_filtered, params_filtered = service.build_batch_query_cte(
        embeddings=embeddings,
        top_k=10,
        collection_name=collection,
        source_filter_sql=(
            "EXISTS (SELECT 1 FROM products p WHERE p.id::text = rag_index.source_id)"
        ),
    )
    assert "EXISTS" in cte_sql_filtered
    assert "products p" in cte_sql_filtered
    assert params_filtered["top_k"] == 10

    cte_sql_keep_best, params_keep_best = service.build_batch_query_cte(
        embeddings=embeddings,
        top_k=5,
        collection_name=collection,
        keep_best=True,
    )
    assert "DISTINCT ON (v.query_idx, results.source_id)" in cte_sql_keep_best
    assert (
        "ORDER BY v.query_idx ASC, results.source_id, results.distance ASC"
        in cte_sql_keep_best
    )
    assert params_keep_best["top_k"] == 5


@pytest.mark.asyncio
async def test_batch_query_with_join_empty_input(service, collection):
    results = await service.batch_query_with_join(
        texts=[], top_k=5, collection_name=collection
    )
    assert results == []


@pytest.mark.asyncio
async def test_batch_query_with_join_keep_best(service, collection):
    """In the batch CTE, LIMIT applies per query inside the LATERAL join.

    Applied outside, a source with several matching rows would use up the
    limit before a second source appeared.
    """
    await service.index_batch(
        texts=["part 1 of doc", "part 2 of doc", "different doc"],
        metadata_list=[{"key": "1"}, {"key": "2"}, {"key": "3"}],
        source_ids=["source_1", "source_1", "source_2"],
        collection_name=collection,
    )

    results = await service.batch_query_with_join(
        texts=["doc"], top_k=2, collection_name=collection, keep_best=True
    )
    assert len(results) == 1
    assert len(results[0]) == 2
    assert {res["source_id"] for res in results[0]} == {"source_1", "source_2"}

    results_top1 = await service.batch_query_with_join(
        texts=["doc"], top_k=1, collection_name=collection, keep_best=True
    )
    assert len(results_top1[0]) == 1


@pytest.mark.asyncio
async def test_rag_service_index_batch_edge_cases(service, collection):
    assert (
        await service.index_batch(
            texts=[], metadata_list=[], collection_name=collection
        )
        == []
    )

    with pytest.raises(
        ValueError,
        match="The number of texts and metadata dictionaries must be the same.",
    ):
        await service.index_batch(
            texts=["a"], metadata_list=[], collection_name=collection
        )

    with pytest.raises(
        ValueError, match="The number of texts and source_ids must be the same."
    ):
        await service.index_batch(
            texts=["a"],
            metadata_list=[{}],
            source_ids=["1", "2"],
            collection_name=collection,
        )

    item = await service.index(
        text="test single", source_metadata={"key": "val"}, collection_name=collection
    )
    assert item["content"] == "test single"
    assert item["rag_metadata"] == {"key": "val"}
    assert item["collection_name"] == collection
    assert len(item["embedding"]) == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_rag_service_compute_similarity_matrix_empty(service, collection):
    assert (
        await service.compute_similarity_matrix(
            texts=[], source_ids=["1"], collection_name=collection
        )
        == []
    )
    assert await service.compute_similarity_matrix(
        texts=["a"], source_ids=[], collection_name=collection
    ) == [[]]


@pytest.mark.asyncio
async def test_rag_service_batch_query_with_join_no_table(service, collection):
    """Without a join table the batch query returns the bare index columns."""
    await service.index(text="test join", collection_name=collection, source_id="s1")

    results = await service.batch_query_with_join(
        texts=["test join"], top_k=5, collection_name=collection
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0]["content"] == "test join"
    assert "id" in results[0][0]
    assert "source_id" in results[0][0]
    assert "similarity" in results[0][0]


@pytest.mark.asyncio
async def test_collection_provisioning_and_drop(
    agents_db_config, service, embedding_model, collection
):
    """Indexing provisions a typed table + registry row; drop removes both."""
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import create_async_engine

    from kavalai.db import ensure_async_scheme

    await service.index_batch(
        texts=["a", "b"], metadata_list=[{}, {}], collection_name=collection
    )

    table_name = PostgresRagService.table_name_for_collection(collection)
    schema = agents_db_config["schema"]
    engine = create_async_engine(ensure_async_scheme(agents_db_config["uri"]))

    async def table_names() -> list[str]:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda c: inspect(c).get_table_names(schema=schema)
            )

    tables = await table_names()
    assert table_name in tables
    assert "rag_collections" in tables
    async with engine.connect() as conn:
        index_names = await conn.run_sync(
            lambda c: {i["name"] for i in inspect(c).get_indexes(table_name, schema)}
        )
    assert f"ix_{table_name}_embedding" in index_names
    assert f"ix_{table_name}_metadata" in index_names

    collections = await service.list_collections()
    entry = next(c for c in collections if c["name"] == collection)
    assert entry["model"] == embedding_model
    assert entry["embedding_size"] == EMBEDDING_DIM
    assert entry["count"] == 2

    stats = await service.get_stats()
    assert collection in stats["collections"]
    assert stats["total_entries"] >= 2

    await service.drop_collection(collection)
    assert table_name not in await table_names()
    await engine.dispose()
    assert await service.count_entries(collection) == 0


@pytest.mark.asyncio
async def test_collection_dim_mismatch(service, collection):
    await service.create_collection(collection, embedding_size=3)
    with pytest.raises(ValueError, match="3-dimensional"):
        await service.index_batch(
            texts=["a"], metadata_list=[{}], collection_name=collection
        )
    await service.drop_collection(collection)


@pytest.mark.asyncio
async def test_collection_schema_version_upgrade(
    agents_db_config, service, embedding_model, collection
):
    """Registry schema_version drives in-code upgrades of collection tables."""
    import kavalai.rag.postgres as pg_mod

    await service.create_collection(collection, embedding_size=3)

    schema = agents_db_config["schema"]
    session_maker = db_manager.get_sessionmaker(
        uri=agents_db_config["uri"], schema=schema
    )

    async def set_schema_version(version: int) -> None:
        async with session_maker() as session:
            await session.execute(
                text(
                    f'UPDATE "{schema}".rag_collections SET schema_version = :v '
                    f"WHERE name = :name"
                ).bindparams(v=version, name=collection)
            )
            await session.commit()

    await set_schema_version(0)

    upgraded = []

    async def fake_upgrade(session, svc, info):
        upgraded.append(info.name)

    # A fresh service has not loaded the registry yet, so its first use
    # triggers the load and with it the upgrade.
    fresh = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=schema
    )
    fresh.embedding_client = fake_embedding_client()
    original = dict(pg_mod._COLLECTION_UPGRADES)
    pg_mod._COLLECTION_UPGRADES[0] = fake_upgrade
    try:
        assert await fresh.count_entries(collection) == 0
    finally:
        pg_mod._COLLECTION_UPGRADES.clear()
        pg_mod._COLLECTION_UPGRADES.update(original)

    assert upgraded == [collection]
    collections = await fresh.list_collections()
    entry = next(c for c in collections if c["name"] == collection)
    assert entry["schema_version"] == pg_mod.RAG_COLLECTION_SCHEMA_VERSION

    # A version newer than the library supports is refused.
    await set_schema_version(99)
    newer = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=schema
    )
    newer.embedding_client = fake_embedding_client()
    with pytest.raises(ValueError, match="newer than this library"):
        await newer.count_entries(collection)

    await service.drop_collection(collection)


@pytest.mark.asyncio
async def test_get_embeddings_by_ids(service, collection):
    """``get_embeddings_by_ids`` is Postgres-only and keyed by the minted ids."""
    items = await service.index_batch(
        texts=["one", "two", "three"],
        metadata_list=[{"n": 1}, {"n": 2}, {"n": 3}],
        source_ids=["s1", "s2", "s3"],
        collection_name=collection,
    )

    ids = [items[0]["id"], items[2]["id"]]
    by_id = await service.get_embeddings_by_ids(collection, ids)
    assert set(by_id.keys()) == set(ids)
    assert all(len(v) == EMBEDDING_DIM for v in by_id.values())
    assert by_id[items[0]["id"]] == pytest.approx(fake_embedding("one"), abs=1e-6)

    await service.drop_collection(collection)


def test_vector_literal_and_parse_vector_round_trip():
    from kavalai.rag.postgres import _parse_vector, _vector_literal

    literal = _vector_literal((0.5, -1.0, 2.0))
    assert literal == "[0.5, -1.0, 2.0]"
    # pgvector returns the same bracketed text, without spaces.
    assert _parse_vector("[0.5,-1.0,2.0]") == [0.5, -1.0, 2.0]
    assert _parse_vector(literal) == [0.5, -1.0, 2.0]
    # A driver that already decodes the vector hands back a sequence.
    assert _parse_vector((0.5, -1.0)) == [0.5, -1.0]


def test_parse_version():
    from kavalai.rag.postgres import _parse_version

    assert _parse_version("0.8.0") == (0, 8, 0)
    assert _parse_version("0.7") == (0, 7)
    assert _parse_version("0.7.4dev") == (0, 7)
    assert _parse_version("") == (0,)
    assert _parse_version(None) == (0,)
    assert _parse_version("unknown") == (0,)


@asynccontextmanager
async def _no_session():
    yield AsyncMock()


def test_options_are_keyword_only_and_agent_is_gone():
    """The old positional ``agent`` parameter is removed, not reinterpreted."""
    with pytest.raises(TypeError):
        PostgresRagService(_no_session, EMBEDDING_MODEL, Normalizer())
    with pytest.raises(TypeError, match="agent"):
        PostgresRagService(_no_session, EMBEDDING_MODEL, agent=object())


def test_from_uri_and_from_session_maker_forward_options(agents_db_config):
    receiver = MagicMock()
    svc = PostgresRagService.from_uri(
        agents_db_config["uri"],
        EMBEDDING_MODEL,
        schema="elsewhere",
        provision=False,
        stats_receiver=receiver,
        vector_type="halfvec",
    )
    assert (svc.schema, svc.provision, svc.vector_type) == (
        "elsewhere",
        False,
        "halfvec",
    )
    assert svc.stats_receiver is receiver

    other = PostgresRagService.from_session_maker(
        svc.session_maker, EMBEDDING_MODEL, schema="s", provision=False
    )
    assert (other.session_maker, other.schema, other.provision) == (
        svc.session_maker,
        "s",
        False,
    )


@pytest.mark.asyncio
async def test_the_service_writes_no_model_call_stats(
    agents_db_config, service, collection
):
    """Statistics go to a receiver; the RAG service owns no table of them."""
    schema = agents_db_config["schema"]

    async def stat_rows() -> int:
        async with service.session_maker() as session:
            return (
                await session.execute(
                    text(f'SELECT count(*) FROM "{schema}".model_call_stats')
                )
            ).scalar()

    before = await stat_rows()
    await service.index_batch(
        texts=["apple", "banana"], metadata_list=[{}, {}], collection_name=collection
    )
    await service.query("apple", collection_name=collection)

    assert await stat_rows() == before


@pytest.mark.asyncio
async def test_pgvector_version_is_read_once(service):
    async with service.session_maker() as session:
        version = await service._pgvector(session)
    assert version >= (0, 7)
    service._pgvector_version = (9, 9)
    async with service.session_maker() as session:
        assert await service._pgvector(session) == (9, 9)


@pytest.mark.asyncio
async def test_postgres_only_paths_on_a_missing_collection(service):
    assert await service.batch_query_with_join(
        texts=["apple"], collection_name="never_created"
    ) == [[]]
    assert await service.compute_similarity_matrix(
        texts=["apple"], source_ids=["a", "b"], collection_name="never_created"
    ) == [[0.0, 0.0]]
    with pytest.raises(Exception, match="No embeddings found"):
        await service.learn_normalizer("never_created")
    with pytest.raises(ValueError, match="Unknown RAG collection 'never_created'"):
        service.build_batch_query_cte(
            embeddings=[[0.1] * EMBEDDING_DIM], top_k=5, collection_name="never_created"
        )


@pytest.mark.asyncio
async def test_learn_normalizer_on_an_empty_collection(service, collection):
    await service.create_collection(collection, embedding_size=EMBEDDING_DIM)
    with pytest.raises(Exception, match="No embeddings found"):
        await service.learn_normalizer(collection)
    await service.drop_collection(collection)


# Half-precision vectors.


@pytest.fixture
def halfvec_service(agents_db_config, migrated_agents_db):
    svc = PostgresRagService.from_uri(
        agents_db_config["uri"],
        EMBEDDING_MODEL,
        schema=agents_db_config["schema"],
        vector_type="halfvec",
    )
    svc.embedding_client = fake_embedding_client()
    return svc


async def _column_type(svc, collection) -> str:
    async with svc.session_maker() as session:
        return (
            await session.execute(
                text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
                ).bindparams(
                    table=svc._qualified(svc.table_name_for_collection(collection))
                )
            )
        ).scalar()


@pytest.mark.asyncio
async def test_a_halfvec_collection_round_trips(
    agents_db_config, halfvec_service, collection
):
    texts = ["apple pie", "banana bread", "cherry tart"]
    await halfvec_service.index_batch(
        texts=texts,
        metadata_list=[{"n": i} for i in range(3)],
        source_ids=["a", "b", "c"],
        collection_name=collection,
    )
    assert (
        await _column_type(halfvec_service, collection) == f"halfvec({EMBEDDING_DIM})"
    )
    assert halfvec_service._collections[collection].vector_type == "halfvec"

    hits = await halfvec_service.query("apple pie", top_k=1, collection_name=collection)
    assert hits[0].content == "apple pie"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-3)

    # A service configured for 32-bit vectors reads the type from the table,
    # so the casts in its queries match the column.
    reader = PostgresRagService.from_uri(
        agents_db_config["uri"], EMBEDDING_MODEL, schema=agents_db_config["schema"]
    )
    reader.embedding_client = fake_embedding_client()
    filtered = await reader.query(
        "banana", top_k=5, collection_name=collection, source_ids=["b"]
    )
    assert [hit.content for hit in filtered] == ["banana bread"]
    assert reader._collections[collection].vector_type == "halfvec"

    normalizer = await reader.learn_normalizer(collection)
    assert len(normalizer.center_vector) == EMBEDDING_DIM
    entries = [entry async for entry in reader.iter_entries(collection)]
    assert {len(entry["embedding"]) for entry in entries} == {EMBEDDING_DIM}
    by_id = await reader.get_embeddings_by_ids(collection, [entries[0]["id"]])
    assert by_id[entries[0]["id"]] == pytest.approx(entries[0]["embedding"])
    matrix = await reader.compute_similarity_matrix(
        texts=["apple pie"], source_ids=["a", "b"], collection_name=collection
    )
    assert matrix[0][0] == pytest.approx(1.0, abs=1e-3)
    joined = await reader.batch_query_with_join(
        texts=["cherry tart"], top_k=1, collection_name=collection
    )
    assert joined[0][0]["content"] == "cherry tart"

    await reader.drop_collection(collection)


@pytest.mark.asyncio
async def test_halfvec_indexes_more_dimensions_than_vector(
    service, halfvec_service, collection
):
    """HNSW stops at 2,000 dimensions for ``vector`` and 4,000 for ``halfvec``."""
    await halfvec_service.create_collection(f"{collection}_half", embedding_size=3072)
    assert await _column_type(halfvec_service, f"{collection}_half") == "halfvec(3072)"

    with pytest.raises(Exception, match="2000 dimensions"):
        await service.create_collection(collection, embedding_size=3072)
    assert collection not in {c["name"] for c in await service.list_collections()}

    await service.create_collection(
        f"{collection}_explicit", embedding_size=3072, vector_type="halfvec"
    )
    await halfvec_service.drop_collection(f"{collection}_half")
    await service.drop_collection(f"{collection}_explicit")


@pytest.mark.asyncio
async def test_halfvec_needs_pgvector_0_7(halfvec_service, collection):
    halfvec_service._pgvector_version = (0, 6, 2)

    with pytest.raises(ValueError, match="need pgvector 0.7 or later.*has 0.6.2"):
        await halfvec_service.create_collection(collection, embedding_size=4)
    assert collection not in {
        c["name"] for c in await halfvec_service.list_collections()
    }


# Iterative index scans.


def _vector_client(vectors: dict[str, list[float]]):
    async def compute_embeddings(texts, *args, **kwargs):
        return [vectors[t] for t in texts], None

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=compute_embeddings)
    return client


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


NEAR_ROWS = 2000
FAR_ROWS = 1000


@pytest.fixture
async def filtered_collection(service, collection):
    """Rows of a "near" source around the query, and a "far" source opposite.

    A third of the rows are "far", which is too many for the planner to fetch
    them through the ``source_id`` index and sort them: it walks the HNSW
    index in distance order and filters as it goes. A plain HNSW scan stops
    after ``ef_search`` (40) candidates, all of them "near", so a query
    restricted to "far" comes back empty; an iterative scan keeps going.
    With only a handful of "far" rows the planner uses the ``source_id``
    index instead and the result is exact either way, which is why this
    fixture is not built that way.
    """
    vectors = {"query": _unit([1.0] + [0.0] * (EMBEDDING_DIM - 1))}
    texts, source_ids = [], []
    for i in range(NEAR_ROWS):
        noise = [((i * 7919 + d * 104729) % 1000) / 1000.0 for d in range(8)]
        vectors[f"near {i}"] = _unit([5.0] + noise + [0.0] * (EMBEDDING_DIM - 9))
        texts.append(f"near {i}")
        source_ids.append("near")
    for i in range(FAR_ROWS):
        noise = [((i * 31 + d * 17) % 100) / 100.0 for d in range(8)]
        vectors[f"far {i}"] = _unit([-5.0] + [0.0] * (EMBEDDING_DIM - 9) + noise)
        texts.append(f"far {i}")
        source_ids.append("far")
    service.embedding_client = _vector_client(vectors)
    for start in range(0, len(texts), 500):
        await service.index_batch(
            texts=texts[start : start + 500],
            metadata_list=[{}] * len(texts[start : start + 500]),
            source_ids=source_ids[start : start + 500],
            collection_name=collection,
        )
    table = service._qualified(service.table_name_for_collection(collection))
    async with service.session_maker() as session:
        await session.execute(text(f"ANALYZE {table}"))
        await session.commit()
    yield service
    await service.drop_collection(collection)


@pytest.mark.asyncio
async def test_a_filter_on_distant_rows_still_returns_top_k(
    filtered_collection, collection
):
    hits = await filtered_collection.query(
        "query", top_k=5, collection_name=collection, source_ids=["far"]
    )

    assert len(hits) == 5
    assert {hit.source_id for hit in hits} == {"far"}


@pytest.mark.asyncio
async def test_without_an_iterative_scan_the_filter_comes_back_short(
    filtered_collection, collection, monkeypatch
):
    """The control for the test above: the planner does use the HNSW index."""

    async def no_iterative_scan(session):
        return None

    monkeypatch.setattr(filtered_collection, "_enable_filtered_scan", no_iterative_scan)

    hits = await filtered_collection.query(
        "query", top_k=5, collection_name=collection, source_ids=["far"]
    )

    assert len(hits) < 5


@pytest.mark.asyncio
async def test_before_pgvector_0_8_no_iterative_scan_is_requested(
    filtered_collection, collection
):
    filtered_collection._pgvector_version = (0, 7, 4)
    executed = []
    original = filtered_collection.session_maker

    @asynccontextmanager
    async def recording_session():
        async with original() as session:
            execute = session.execute

            async def record(statement, *args, **kwargs):
                executed.append(str(statement))
                return await execute(statement, *args, **kwargs)

            session.execute = record
            yield session

    filtered_collection.session_maker = recording_session
    try:
        await filtered_collection.query(
            "query", top_k=5, collection_name=collection, source_ids=["far"]
        )
    finally:
        filtered_collection.session_maker = original

    assert not any("iterative_scan" in statement for statement in executed)
