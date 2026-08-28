import pytest
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch, MagicMock

from sqlalchemy import Table, Column, String, Boolean, MetaData, Float, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from kavalai.rag import PostgresRagService, RagServiceResult
from kavalai.normalizer import Normalizer
from kavalai.db import ModelCallStat, db_manager


@pytest.fixture
def embedding_model():
    return "openai/text-embedding-3-small"


@pytest.mark.asyncio
async def test_rag_service_indexing(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    texts = ["hello", "world"]
    source_metadata = [{"id": 1}, {"id": 2}]

    items = await service.index_batch(
        texts=texts, metadata_list=source_metadata, collection_name="test_coll"
    )
    assert len(items) == 2
    assert items[0]["collection_name"] == "test_coll"
    assert items[0]["content"] == "hello"
    assert len(items[0]["embedding"]) == 1536
    assert items[0]["rag_metadata"] == {"id": 1}
    assert items[1]["content"] == "world"
    assert len(items[1]["embedding"]) == 1536

    result = await service.query("hello", top_k=1, collection_name="test_coll")
    assert len(result) == 1
    assert isinstance(result[0], RagServiceResult)
    assert result[0].content == "hello"
    assert result[0].similarity > 0


@pytest.mark.asyncio
async def test_rag_service_deletion(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    texts = ["item 1", "item 2", "item 3"]
    source_ids = ["sid1", "sid2", "sid3"]
    metadata = [{}, {}, {}]
    collection = "delete_test"

    # Index items
    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Verify they exist
    results = await service.query("item", top_k=10, collection_name=collection)
    assert len(results) == 3

    # Delete sid1 and sid3
    await service.delete_by_source_id(collection, ["sid1", "sid3"])

    # Verify only sid2 remains
    results = await service.query("item", top_k=10, collection_name=collection)
    assert len(results) == 1
    assert results[0].source_id == "sid2"
    assert results[0].content == "item 2"


@pytest.mark.asyncio
async def test_rag_service_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # We index multiple items with the same source_id
    texts = ["apple", "apple pie", "banana"]
    source_ids = ["fruit_1", "fruit_1", "fruit_2"]
    metadata = [{}, {}, {}]
    collection = "keep_best_test"

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query for "apple".
    # Without keep_best (default False), we should get both fruit_1 items
    results = await service.query("apple", top_k=10, collection_name=collection)
    assert len(results) == 3

    # Now query with keep_best=True
    results_best = await service.query(
        "apple", top_k=10, collection_name=collection, keep_best=True
    )

    # Should only have one result per source_id
    assert len(results_best) == 2  # fruit_1 and fruit_2
    source_ids_found = [r.source_id for r in results_best]
    assert len(source_ids_found) == len(set(source_ids_found))

    # "apple" should be better than "apple pie" for the query "apple"
    fruit_1_best = [r for r in results_best if r.source_id == "fruit_1"]
    assert len(fruit_1_best) == 1
    assert fruit_1_best[0].content == "apple"


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
    with patch("kavalai.rag.postgres.make_embedding_client") as mock_llm_client_cls:
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
            texts=["test"], normalizer=normalizer
        )

        # 2. Test query
        mock_llm_client.compute_embeddings.reset_mock()
        await service.query("test query")

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["test query"], normalizer=normalizer
        )

        # 3. Test compute_similarity_matrix
        mock_llm_client.compute_embeddings.reset_mock()
        await service.compute_similarity_matrix(
            texts=["t1"], source_ids=["s1"], collection_name="default"
        )

        mock_llm_client.compute_embeddings.assert_called_with(
            texts=["t1"], normalizer=normalizer
        )


@pytest.mark.asyncio
async def test_rag_service_without_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    model = "openai/text-embedding-3-small"

    with patch("kavalai.rag.postgres.make_embedding_client") as mock_llm_client_cls:
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
            texts=["test query"], normalizer=None
        )


@pytest.mark.asyncio
async def test_rag_service_keep_best_duplicates(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test that keep_best handles duplicate content/distances correctly."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    collection = "duplicate_test"
    # Index exactly the same content for the same source_id multiple times
    texts = ["Tesla Model 3 is an electric car", "Tesla Model 3 is an electric car"]
    source_ids = ["tesla_3", "tesla_3"]
    metadata = [{"brand": "Tesla"}, {"brand": "Tesla"}]

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with keep_best=True
    results = await service.query(
        "Tesla electric car", top_k=10, collection_name=collection, keep_best=True
    )

    # Should only have one result despite multiple identical best matches
    assert len(results) == 1
    assert results[0].source_id == "tesla_3"


@pytest.mark.asyncio
async def test_rag_service_top_k_with_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Index items for 5 different source IDs, each with 2 items
    texts = []
    source_ids = []
    for i in range(1, 6):
        sid = f"source_{i}"
        texts.extend([f"content {i} a", f"content {i} b"])
        source_ids.extend([sid, sid])

    metadata = [{} for _ in texts]
    collection = "top_k_keep_best_test"

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with top_k=3 and keep_best=True
    # We expect exactly 3 results, each from a different source_id
    top_k = 3
    results = await service.query(
        "content", top_k=top_k, collection_name=collection, keep_best=True
    )

    assert len(results) == top_k
    unique_source_ids = set(r.source_id for r in results)
    assert len(unique_source_ids) == top_k

    # Now query with top_k=1 and keep_best=True
    # This might fail if the implementation is buggy (e.g. limit applied before join)
    results_1 = await service.query(
        "content", top_k=1, collection_name=collection, keep_best=True
    )
    assert len(results_1) == 1


@pytest.mark.asyncio
async def test_rag_service_query_source_ids(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    texts = ["apple", "banana", "cherry"]
    source_ids = ["sid_apple", "sid_banana", "sid_cherry"]
    metadata = [{}, {}, {}]
    collection = "source_ids_test"

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with filtering by source_ids
    results = await service.query(
        "fruit",
        top_k=10,
        collection_name=collection,
        source_ids=["sid_apple", "sid_cherry"],
    )

    assert len(results) == 2
    found_ids = {r.source_id for r in results}
    assert found_ids == {"sid_apple", "sid_cherry"}
    assert "sid_banana" not in found_ids


@pytest.mark.asyncio
async def test_rag_service_query_source_ids_with_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Index multiple items for some source_ids
    texts = ["apple 1", "apple 2", "banana 1", "banana 2", "cherry"]
    source_ids = ["sid_apple", "sid_apple", "sid_banana", "sid_banana", "sid_cherry"]
    metadata = [{}, {}, {}, {}, {}]
    collection = "source_ids_keep_best_test"

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        collection_name=collection,
        source_ids=source_ids,
    )

    # Query with filtering by source_ids and keep_best=True
    results = await service.query(
        "fruit",
        top_k=10,
        collection_name=collection,
        source_ids=["sid_apple", "sid_banana"],
        keep_best=True,
    )

    # We expect exactly 2 results: one for sid_apple and one for sid_banana
    assert len(results) == 2
    found_ids = {r.source_id for r in results}
    assert found_ids == {"sid_apple", "sid_banana"}
    assert "sid_cherry" not in found_ids


@pytest.mark.asyncio
async def test_compute_similarity_matrix(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Index some documents
    texts = ["apple", "apple pie", "banana", "banana bread", "cherry"]
    source_ids = ["fruit_1", "fruit_1", "fruit_2", "fruit_2", "fruit_3"]
    await service.index_batch(
        texts=texts,
        metadata_list=[{}] * len(texts),
        collection_name="matrix_test",
        source_ids=source_ids,
    )

    # Compute similarity matrix
    queries = ["apple", "banana"]
    target_source_ids = ["fruit_1", "fruit_2", "fruit_3", "fruit_nonexistent"]

    # Test "min" method (shortest distance = max similarity)
    matrix_min = await service.compute_similarity_matrix(
        texts=queries,
        source_ids=target_source_ids,
        method="min",
        collection_name="matrix_test",
    )

    assert len(matrix_min) == 2  # 2 queries
    assert len(matrix_min[0]) == 4  # 4 target source ids
    assert len(matrix_min[1]) == 4

    # matrix_min[0][0] is similarity between "apple" and "fruit_1" (contains "apple", "apple pie")
    # "apple" vs "apple" should have high similarity
    assert matrix_min[0][0] > 0.9
    # matrix_min[0][1] is similarity between "apple" and "fruit_2" (contains "banana", "banana bread")
    # should be lower
    assert matrix_min[0][0] > matrix_min[0][1]
    # nonexistent source should have 0 similarity
    assert matrix_min[0][3] == 0.0

    # Test "avg" method
    matrix_avg = await service.compute_similarity_matrix(
        texts=queries,
        source_ids=target_source_ids,
        method="avg",
        collection_name="matrix_test",
    )
    assert len(matrix_avg) == 2
    assert len(matrix_avg[0]) == 4

    # For "fruit_1", "apple" query vs ["apple", "apple pie"]
    # min distance will be "apple" vs "apple" (distance near 0)
    # avg distance will be average of ("apple" vs "apple") and ("apple" vs "apple pie")
    # So avg similarity should be less than or equal to min similarity
    assert matrix_avg[0][0] <= matrix_min[0][0]


@pytest.mark.asyncio
async def test_rag_service_query_batch(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Index some test documents
    texts = ["apple", "banana", "cherry", "date", "elderberry"]
    source_ids = ["s1", "s2", "s3", "s4", "s5"]
    collection = "batch_query_test"

    await service.index_batch(
        texts=texts,
        metadata_list=[{}] * len(texts),
        collection_name=collection,
        source_ids=source_ids,
    )

    # 1. Basic batch query with multiple texts
    queries = ["apple", "banana"]
    results = await service.query_batch(
        texts=queries, top_k=1, collection_name=collection
    )

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 1
    assert results[0][0].content == "apple"
    assert results[1][0].content == "banana"

    # 2. Batch query with collection filtering
    # Create another collection
    await service.index_batch(
        texts=["pineapple"],
        metadata_list=[{}],
        collection_name="other_coll",
        source_ids=["s6"],
    )

    # Query in original collection, should not find "pineapple"
    results_filtered = await service.query_batch(
        texts=["apple", "pineapple"], top_k=1, collection_name=collection
    )
    assert len(results_filtered) == 2
    assert results_filtered[0][0].content == "apple"
    # For "pineapple", it should find the closest in "batch_query_test" (likely "apple")
    assert results_filtered[1][0].content != "pineapple"

    # Query in other collection
    results_other = await service.query_batch(
        texts=["pineapple"], top_k=1, collection_name="other_coll"
    )
    assert results_other[0][0].content == "pineapple"

    # 3. Batch query with source_ids filtering
    results_sid = await service.query_batch(
        texts=["apple", "banana"],
        top_k=5,
        collection_name=collection,
        source_ids=["s1", "s3"],
    )
    assert len(results_sid) == 2
    # Result for "apple" should be "apple" (s1)
    assert results_sid[0][0].content == "apple"
    assert results_sid[0][0].source_id == "s1"
    # Result for "banana" should be "cherry" (s3) because "banana" (s2) is filtered out
    # Actually "apple" might be closer to "banana" than "cherry" depending on the model.
    # Let's check which one is closer or just use a more distinct query.
    assert results_sid[1][0].source_id in ["s1", "s3"]
    assert results_sid[1][0].source_id != "s2"

    # 4. Empty input
    assert await service.query_batch(texts=[]) == []

    # 5. Verify top_k
    results_top_k = await service.query_batch(
        texts=["fruit"], top_k=3, collection_name=collection
    )
    assert len(results_top_k) == 1
    assert len(results_top_k[0]) == 3


@pytest.mark.asyncio
async def test_rag_service_batch_query_with_join(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test batch_query_with_join with a mock products table."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    async with service.session_maker() as session:
        # Create a mock products table
        metadata = MetaData()
        _ = Table(
            "products",
            metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
            Column("name", String),
            Column("category", String),
            Column("price", Float),
            Column("in_stock", Boolean),
        )

        # Create the table
        await session.execute(text("DROP TABLE IF EXISTS products"))
        await session.commit()

        create_table_sql = """
            CREATE TABLE products (
                id UUID PRIMARY KEY,
                name VARCHAR,
                category VARCHAR,
                price FLOAT,
                in_stock BOOLEAN
            )
        """
        await session.execute(text(create_table_sql))
        await session.commit()

        # Insert test products
        product_data = [
            (
                uuid.uuid4(),
                "Wireless Headphones",
                "electronics",
                99.99,
                True,
            ),
            (
                uuid.uuid4(),
                "Laptop Stand",
                "electronics",
                49.99,
                True,
            ),
            (
                uuid.uuid4(),
                "Office Chair",
                "furniture",
                299.99,
                True,
            ),
            (
                uuid.uuid4(),
                "Desk Lamp",
                "electronics",
                29.99,
                False,
            ),  # Out of stock
            (
                uuid.uuid4(),
                "USB Cable",
                "electronics",
                9.99,
                True,
            ),
        ]

        for product_id, name, category, price, in_stock in product_data:
            await session.execute(
                text(
                    """
                INSERT INTO products (id, name, category, price, in_stock)
                VALUES (:id, :name, :category, :price, :in_stock)
            """
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

    # Index products in RAG
    texts = [p[1] for p in product_data]  # Product names
    source_ids = [str(p[0]) for p in product_data]  # Product IDs
    collection = "products"

    await service.index_batch(
        texts=texts,
        metadata_list=[{}] * len(texts),
        collection_name=collection,
        source_ids=source_ids,
    )

    # Test 1: Query without filters
    results_no_filter = await service.batch_query_with_join(
        texts=["headphones"],
        top_k=5,
        collection_name=collection,
        join_table="products p",
        join_condition="p.id::text = r.source_id",
        join_columns=["p.name", "p.category", "p.price", "p.in_stock"],
    )

    assert len(results_no_filter) == 1
    assert len(results_no_filter[0]) > 0
    # Should find "Wireless Headphones" as top result
    assert "Wireless Headphones" in results_no_filter[0][0]["name"]

    # Test 2: Query with category filter
    results_electronics = await service.batch_query_with_join(
        texts=["headphones", "chair"],
        top_k=5,
        collection_name=collection,
        join_table="products p",
        join_condition="p.id::text = r.source_id",
        join_columns=["p.name", "p.category", "p.price"],
        additional_where="p.category = 'electronics'",
    )

    assert len(results_electronics) == 2
    # First query should find electronics
    assert all(r["category"] == "electronics" for r in results_electronics[0])
    # Second query "chair" should find electronics (not the furniture chair)
    assert all(r["category"] == "electronics" for r in results_electronics[1])
    # Should NOT find "Office Chair" since it's furniture
    chair_names = [r["name"] for r in results_electronics[1]]
    assert "Office Chair" not in chair_names

    # Test 3: Query with in_stock filter
    results_in_stock = await service.batch_query_with_join(
        texts=["lamp"],
        top_k=5,
        collection_name=collection,
        join_table="products p",
        join_condition="p.id::text = r.source_id",
        join_columns=["p.name", "p.in_stock"],
        additional_where="p.in_stock = true",
    )

    assert len(results_in_stock) == 1
    # Should NOT find "Desk Lamp" (out of stock)
    lamp_names = [r["name"] for r in results_in_stock[0]]
    assert "Desk Lamp" not in lamp_names

    # Test 4: Combined filters
    results_combined = await service.batch_query_with_join(
        texts=["electronics"],
        top_k=5,
        collection_name=collection,
        join_table="products p",
        join_condition="p.id::text = r.source_id",
        join_columns=["p.name", "p.category", "p.price", "p.in_stock"],
        additional_where="p.category = 'electronics' AND p.in_stock = true AND p.price < 100",
    )

    assert len(results_combined) == 1
    for result in results_combined[0]:
        assert result["category"] == "electronics"
        assert result["in_stock"] is True
        assert result["price"] < 100

    # Test 5: Empty results
    results_empty = await service.batch_query_with_join(
        texts=["nonexistent product xyz"],
        top_k=5,
        collection_name=collection,
        join_table="products p",
        join_condition="p.id::text = r.source_id",
        join_columns=["p.name"],
        additional_where="p.category = 'nonexistent'",
    )

    assert len(results_empty) == 1
    assert len(results_empty[0]) == 0  # No results

    # Cleanup
    async with service.session_maker() as session:
        await session.execute(text("DROP TABLE IF EXISTS products"))
        await session.commit()


@pytest.mark.asyncio
async def test_build_batch_query_cte(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test the build_batch_query_cte method directly."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Create some embeddings (mock)
    embeddings = [
        [0.1] * 1536,  # Query 1
        [0.2] * 1536,  # Query 2
    ]

    # The CTE builder requires a provisioned collection (table-per-collection).
    # Dropped afterwards: the schema is shared with other tests, and one of
    # them indexes 2-dimensional vectors into a collection of the same name.
    await service.create_collection("test", embedding_size=1536)
    try:
        await _assert_batch_query_cte(service, embeddings)
    finally:
        await service.drop_collection("test")


async def _assert_batch_query_cte(service, embeddings):
    # Test 1: Basic CTE without filters
    cte_sql, params = service.build_batch_query_cte(
        embeddings=embeddings, top_k=5, collection_name="test"
    )

    assert "rag_results AS" in cte_sql
    assert "unnest" in cte_sql.lower()
    assert "CROSS JOIN LATERAL" in cte_sql
    assert "WITH ORDINALITY" in cte_sql
    assert "rag_c_test" in cte_sql  # collection table, aliased as rag_index
    assert params["top_k"] == 5
    assert "vector_0" in params
    assert "vector_1" in params

    # Test 2: CTE with source filter
    cte_sql_filtered, params_filtered = service.build_batch_query_cte(
        embeddings=embeddings,
        top_k=10,
        collection_name="test",
        source_filter_sql="EXISTS (SELECT 1 FROM products p WHERE p.id::text = rag_index.source_id)",
    )

    assert "EXISTS" in cte_sql_filtered
    assert "products p" in cte_sql_filtered
    assert params_filtered["top_k"] == 10

    # Test 3: CTE with keep_best
    cte_sql_keep_best, params_keep_best = service.build_batch_query_cte(
        embeddings=embeddings,
        top_k=5,
        collection_name="test",
        keep_best=True,
    )

    assert "DISTINCT ON (v.query_idx, results.source_id)" in cte_sql_keep_best
    assert (
        "ORDER BY v.query_idx ASC, results.source_id, results.distance ASC"
        in cte_sql_keep_best
    )
    assert params_keep_best["top_k"] == 5


@pytest.mark.asyncio
async def test_batch_query_with_join_empty_input(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test batch_query_with_join with empty texts."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    results = await service.batch_query_with_join(
        texts=[],
        top_k=5,
        collection_name="products",
    )

    assert results == []


@pytest.mark.asyncio
async def test_batch_query_with_join_keep_best(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test batch_query_with_join with keep_best parameter."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    collection = "test_keep_best_join"

    # Index some items for the same source
    texts = ["part 1 of doc", "part 2 of doc", "different doc"]
    metadata = [{"key": "1"}, {"key": "2"}, {"key": "3"}]
    source_ids = ["source_1", "source_1", "source_2"]

    await service.index_batch(
        texts=texts,
        metadata_list=metadata,
        source_ids=source_ids,
        collection_name=collection,
    )

    # Query with keep_best=True
    # Using patch to avoid real LLM calls if possible, but the fixture embedding_model is used.
    # Actually, batch_query_with_join calls llm_client.compute_embeddings.
    # In test_rag_service_batch_query_with_join, it seems it uses real or mocked embeddings.

    with patch.object(service.embedding_client, "compute_embeddings") as mock_compute:
        stats = ModelCallStat(
            model=embedding_model,
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
            call_type="embedding",
        )
        mock_compute.return_value = ([[0.1] * 1536], stats)

        # Test case for the bug fix: LIMIT should be per query, but inside LATERAL
        # If we have 2 sources, each with multiple matches, we want to see both if top_k=2
        # source_1: "part 1 of doc", "part 2 of doc"
        # source_2: "different doc"
        results = await service.batch_query_with_join(
            texts=["doc"], top_k=2, collection_name=collection, keep_best=True
        )

    # We asked for top_k=2, so we should get 1 list of results for 1 query text
    assert len(results) == 1
    # For that query, we should have 2 unique sources because both source_1 and source_2 match "doc"
    # Even though source_1 has two matches, keep_best=True only keeps the best one,
    # and then the LIMIT :top_k (applied inside LATERAL in the fixed version)
    # allows us to get the next source.
    assert len(results[0]) == 2
    seen_sources = [res["source_id"] for res in results[0]]
    assert "source_1" in seen_sources
    assert "source_2" in seen_sources

    # Test the previous buggy behavior: if LIMIT was outside, top_k=1 would only return 1 source
    # (which is correct for top_k=1, but the problem was when top_k was larger than the number of matches for one source)
    with patch.object(service.embedding_client, "compute_embeddings") as mock_compute:
        mock_compute.return_value = ([[0.1] * 1536], stats)
        results_top1 = await service.batch_query_with_join(
            texts=["doc"], top_k=1, collection_name=collection, keep_best=True
        )
    assert len(results_top1[0]) == 1


@pytest.mark.asyncio
async def test_rag_service_from_session_maker(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test creating PostgresRagService from session maker."""
    session_maker = db_manager.get_sessionmaker(
        uri=agents_db_config["uri"], schema=agents_db_config["schema"]
    )
    service = PostgresRagService.from_session_maker(
        session_maker, embedding_model, schema=agents_db_config["schema"]
    )
    assert isinstance(service, PostgresRagService)
    assert service.model == embedding_model
    assert service.session_maker == session_maker
    assert service.schema == agents_db_config["schema"]


@pytest.mark.asyncio
async def test_rag_service_index_batch_edge_cases(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test index_batch edge cases for coverage."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Empty texts
    assert (
        await service.index_batch(texts=[], metadata_list=[], collection_name="test")
        == []
    )

    # Mismatched texts and metadata
    with pytest.raises(
        ValueError,
        match="The number of texts and metadata dictionaries must be the same.",
    ):
        await service.index_batch(texts=["a"], metadata_list=[], collection_name="test")

    # Mismatched texts and source_ids
    with pytest.raises(
        ValueError, match="The number of texts and source_ids must be the same."
    ):
        await service.index_batch(
            texts=["a"],
            metadata_list=[{}],
            source_ids=["1", "2"],
            collection_name="test",
        )

    # Single index (explicitly)
    item = await service.index(
        text="test single", source_metadata={"key": "val"}, collection_name="single"
    )
    assert item["content"] == "test single"
    assert item["rag_metadata"] == {"key": "val"}
    assert item["collection_name"] == "single"


@pytest.mark.asyncio
async def test_rag_service_compute_similarity_matrix_empty(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test compute_similarity_matrix with empty inputs."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Empty texts
    res1 = await service.compute_similarity_matrix(texts=[], source_ids=["1"])
    assert res1 == []

    # Empty source_ids
    res2 = await service.compute_similarity_matrix(texts=["a"], source_ids=[])
    assert res2 == [[]]


@pytest.mark.asyncio
async def test_rag_service_batch_query_with_join_no_table(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test batch_query_with_join without join_table to cover the 'else' branch (line 479)."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Index something
    await service.index(text="test join", collection_name="join_test", source_id="s1")

    # Query without join_table
    results = await service.batch_query_with_join(
        texts=["test join"],
        top_k=5,
        collection_name="join_test",
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0]["content"] == "test join"
    assert "id" in results[0][0]
    assert "source_id" in results[0][0]
    assert "similarity" in results[0][0]


@pytest.mark.asyncio
async def test_rag_service_learn_normalizer(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test learn_normalizer method."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    # Add some data to learn from
    await service.index_batch(
        texts=["data 1", "data 2", "data 3"],
        metadata_list=[{}, {}, {}],
        collection_name="learn_test",
    )

    # Mock Normalizer.learn_from_rag to avoid actual heavy computation if any,
    # though it should be fast on small data.
    # Actually, let's just run it to be sure it works.
    normalizer = await service.learn_normalizer(collection_name="learn_test")
    assert isinstance(normalizer, Normalizer)


@pytest.mark.asyncio
async def test_rag_service_delete_by_id(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test deleting a single indexed item by its id."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    collection = "delete_by_id_test"
    items = await service.index_batch(
        texts=["first item", "second item"],
        metadata_list=[{}, {}],
        collection_name=collection,
        source_ids=["sid1", "sid2"],
    )

    await service.delete(items[0]["id"])

    results = await service.query("item", top_k=10, collection_name=collection)
    assert len(results) == 1
    assert results[0].id == items[1]["id"]


@pytest.mark.asyncio
async def test_rag_service_delete_by_source_id_single(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Test delete_by_source_id with a single source id string."""
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )

    collection = "delete_single_sid_test"
    await service.index_batch(
        texts=["doc a part 1", "doc a part 2", "doc b"],
        metadata_list=[{}, {}, {}],
        collection_name=collection,
        source_ids=["sid_a", "sid_a", "sid_b"],
    )

    await service.delete_by_source_id(collection, "sid_a")

    results = await service.query("doc", top_k=10, collection_name=collection)
    assert len(results) == 1
    assert results[0].source_id == "sid_b"


@pytest.mark.asyncio
async def test_collection_provisioning_and_drop(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Indexing provisions a typed table + registry row; drop removes both."""
    from sqlalchemy import create_engine, inspect

    from kavalai.migrate_db import ensure_sync_scheme
    from kavalai.rag.postgres import PostgresRagService as Svc

    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    collection = "provision_test"
    await service.index_batch(
        texts=["a", "b"], metadata_list=[{}, {}], collection_name=collection
    )

    table_name = Svc.table_name_for_collection(collection)
    engine = create_engine(ensure_sync_scheme(agents_db_config["uri"]))
    insp = inspect(engine)
    tables = insp.get_table_names(schema=agents_db_config["schema"])
    assert table_name in tables
    assert "rag_collections" in tables
    index_names = {
        i["name"]
        for i in insp.get_indexes(table_name, schema=agents_db_config["schema"])
    }
    assert f"ix_{table_name}_embedding" in index_names
    assert f"ix_{table_name}_metadata" in index_names

    collections = await service.list_collections()
    entry = next(c for c in collections if c["name"] == collection)
    assert entry["model"] == embedding_model
    assert entry["embedding_size"] == 1536
    assert entry["count"] == 2

    stats = await service.get_stats()
    assert collection in stats["collections"]
    assert stats["total_entries"] >= 2

    await service.drop_collection(collection)
    insp = inspect(engine)
    assert table_name not in insp.get_table_names(schema=agents_db_config["schema"])
    assert await service.count_entries(collection) == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_collection_dim_mismatch(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    await service.create_collection("dim_test", embedding_size=3)
    with pytest.raises(ValueError, match="3-dimensional"):
        await service.index_batch(
            texts=["a"], metadata_list=[{}], collection_name="dim_test"
        )
    await service.drop_collection("dim_test")


@pytest.mark.asyncio
async def test_collection_schema_version_upgrade(
    agents_db_config, migrated_agents_db, embedding_model
):
    """Registry schema_version drives in-code upgrades of collection tables."""
    import kavalai.rag.postgres as pg_mod

    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    collection = "upgrade_test"
    await service.create_collection(collection, embedding_size=3)

    schema = agents_db_config["schema"]
    session_maker = db_manager.get_sessionmaker(
        uri=agents_db_config["uri"], schema=schema
    )
    async with session_maker() as session:
        await session.execute(
            text(
                f'UPDATE "{schema}".rag_collections SET schema_version = 0 '
                f"WHERE name = :name"
            ).bindparams(name=collection)
        )
        await session.commit()

    upgraded = []

    async def fake_upgrade(session, svc, info):
        upgraded.append(info.name)

    fresh = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=schema
    )
    original = dict(pg_mod._COLLECTION_UPGRADES)
    pg_mod._COLLECTION_UPGRADES[0] = fake_upgrade
    try:
        assert await fresh.count_entries(collection) == 0  # triggers load+upgrade
    finally:
        pg_mod._COLLECTION_UPGRADES.clear()
        pg_mod._COLLECTION_UPGRADES.update(original)

    assert upgraded == [collection]
    collections = await fresh.list_collections()
    entry = next(c for c in collections if c["name"] == collection)
    assert entry["schema_version"] == pg_mod.RAG_COLLECTION_SCHEMA_VERSION

    # A version newer than the library supports is refused.
    async with session_maker() as session:
        await session.execute(
            text(
                f'UPDATE "{schema}".rag_collections SET schema_version = 99 '
                f"WHERE name = :name"
            ).bindparams(name=collection)
        )
        await session.commit()
    newer = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=schema
    )
    with pytest.raises(ValueError, match="newer than this library"):
        await newer.count_entries(collection)

    await service.drop_collection(collection)


@pytest.mark.asyncio
async def test_iter_entries_and_embeddings_by_ids(
    agents_db_config, migrated_agents_db, embedding_model
):
    service = PostgresRagService.from_uri(
        agents_db_config["uri"], embedding_model, schema=agents_db_config["schema"]
    )
    collection = "export_test"
    items = await service.index_batch(
        texts=["one", "two", "three"],
        metadata_list=[{"n": 1}, {"n": 2}, {"n": 3}],
        source_ids=["s1", "s2", "s3"],
        collection_name=collection,
    )

    exported = [e async for e in service.iter_entries(collection, batch_size=2)]
    assert len(exported) == 3
    assert {e["source_id"] for e in exported} == {"s1", "s2", "s3"}
    assert all(len(e["embedding"]) == 1536 for e in exported)
    assert all(isinstance(e["rag_metadata"], dict) for e in exported)

    ids = [items[0]["id"], items[2]["id"]]
    by_id = await service.get_embeddings_by_ids(collection, ids)
    assert set(by_id.keys()) == set(ids)
    assert all(len(v) == 1536 for v in by_id.values())

    await service.drop_collection(collection)


# --- vector text helpers ---


def test_vector_literal_and_parse_vector_round_trip():
    from kavalai.rag.postgres import _parse_vector, _vector_literal

    literal = _vector_literal((0.5, -1.0, 2.0))
    assert literal == "[0.5, -1.0, 2.0]"
    # pgvector returns the same bracketed text, without spaces.
    assert _parse_vector("[0.5,-1.0,2.0]") == [0.5, -1.0, 2.0]
    assert _parse_vector(literal) == [0.5, -1.0, 2.0]
    # A driver that already decodes the vector hands back a sequence.
    assert _parse_vector((0.5, -1.0)) == [0.5, -1.0]
