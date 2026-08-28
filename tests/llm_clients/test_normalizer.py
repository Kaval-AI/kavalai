import os
import pytest
import math
import numpy as np
from kavalai.normalizer import Normalizer
from kavalai.rag import PostgresRagService


def test_normalize_l1():
    embeddings = [[1.0, -1.0], [0.0, 0.0], [3.0, 4.0]]
    normalizer = Normalizer()
    # Manual call to normalize_l1 now expects numpy array and returns numpy array
    normalized = normalizer.normalize_l1(np.array(embeddings)).tolist()

    # [1.0, -1.0] -> sum(abs) = 2.0 -> [0.5, -0.5]
    assert normalized[0] == [0.5, -0.5]
    assert sum(abs(x) for x in normalized[0]) == 1.0

    # [0.0, 0.0] -> sum(abs) = 0 -> [0.0, 0.0]
    assert normalized[1] == [0.0, 0.0]

    # [3.0, 4.0] -> sum(abs) = 7.0 -> [3/7, 4/7]
    assert pytest.approx(normalized[2]) == [3 / 7, 4 / 7]
    assert pytest.approx(sum(abs(x) for x in normalized[2])) == 1.0


def test_normalize_l2():
    embeddings = [[1.0, 1.0], [0.0, 0.0], [3.0, 4.0]]
    normalizer = Normalizer()
    normalized = normalizer.normalize_l2(np.array(embeddings)).tolist()

    # [1.0, 1.0] -> norm is sqrt(2), so [1/sqrt(2), 1/sqrt(2)]
    assert pytest.approx(sum(x * x for x in normalized[0])) == 1.0

    # [0.0, 0.0] -> norm is 0, should remain [0.0, 0.0]
    assert normalized[1] == [0.0, 0.0]

    # [3.0, 4.0] -> norm is 5, so [0.6, 0.8]
    assert normalized[2] == [0.6, 0.8]
    assert pytest.approx(sum(x * x for x in normalized[2])) == 1.0


def test_centering():
    center_vector = [1.0, 2.0]
    embeddings = [[2.0, 3.0], [1.0, 2.0], [0.0, 0.0]]
    normalizer = Normalizer(center_vector=center_vector)
    centered = normalizer.center(np.array(embeddings)).tolist()

    assert centered[0] == [1.0, 1.0]
    assert centered[1] == [0.0, 0.0]
    assert centered[2] == [-1.0, -2.0]


def test_transform():
    center_vector = [1.0, 1.0]
    embeddings = [[2.0, 2.0]]
    normalizer = Normalizer(center_vector=center_vector, l2=True, center=True)

    # Center: [2, 2] - [1, 1] = [1, 1]
    # L2: [1, 1] -> [1/sqrt(2), 1/sqrt(2)]
    result = normalizer.transform(embeddings)
    assert pytest.approx(result[0]) == [1 / math.sqrt(2), 1 / math.sqrt(2)]


def test_transform_single_vector():
    center_vector = [1.0, 1.0]
    embedding = [2.0, 2.0]
    normalizer = Normalizer(center_vector=center_vector, l2=True, center=True)

    result = normalizer.transform(embedding)
    assert pytest.approx(result) == [1 / math.sqrt(2), 1 / math.sqrt(2)]


def test_yaml_string_load_save():
    center_vector = [1.0, 2.0, 3.0]
    original = Normalizer(center_vector=center_vector, l1=True, l2=False, center=True)
    yaml_str = original.to_yaml()

    assert "l1: true" in yaml_str.lower()
    assert "center: true" in yaml_str.lower()

    loaded = Normalizer.from_yaml(yaml_str)
    assert loaded.l1 is True
    assert loaded.l2 is False
    assert loaded.center_enabled is True
    assert np.allclose(loaded.center_vector, center_vector)


def test_yaml_save_load(tmp_path):
    yaml_path = os.path.join(tmp_path, "normalizer.yaml")
    center_vector = [1.0, 2.0, 3.0]
    original = Normalizer(center_vector=center_vector, l1=True, l2=False, center=True)
    original.save_to_yaml(yaml_path)

    loaded = Normalizer.load_from_yaml(yaml_path)
    assert loaded.l1 is True
    assert loaded.l2 is False
    assert loaded.center_enabled is True
    assert np.allclose(loaded.center_vector, center_vector)

    # Verify it works
    test_emb = [[2.0, 3.0, 4.0]]
    # Center: [1, 1, 1]
    # L1: [1/3, 1/3, 1/3]
    result = loaded.transform(test_emb)
    assert pytest.approx(result[0]) == [1 / 3, 1 / 3, 1 / 3]


@pytest.mark.asyncio
async def test_rag_service_learn_normalizer(agents_db):
    """learn_normalizer computes the centering vector from stored embeddings.

    Storage is backend-owned: seed via the service with a mocked embedding
    client, then learn from the collection.
    """
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock

    from kavalai.db import ModelCallStat

    model = "openai/test-model"

    @asynccontextmanager
    async def session_factory():
        yield agents_db

    service = PostgresRagService(
        session_maker=session_factory, model=model, schema="test_agents"
    )

    async def fake_compute_embeddings(texts, normalizer=None):
        stats = ModelCallStat(call_type="embedding", model=model)
        return [[1.0, 1.0], [3.0, 3.0]][: len(texts)], stats

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=fake_compute_embeddings)
    service.embedding_client = client

    await service.index_batch(
        texts=["a", "b"], metadata_list=[{}, {}], collection_name="test"
    )

    normalizer = await service.learn_normalizer(collection_name="test")
    assert np.allclose(normalizer.center_vector, [2.0, 2.0])

    with pytest.raises(Exception, match="No embeddings found"):
        await service.learn_normalizer(collection_name="does-not-exist")

    await service.drop_collection("test")


@pytest.fixture
def clean_default_normalizer():
    import kavalai.normalizer

    # Save original
    original = kavalai.normalizer._default_normalizer
    kavalai.normalizer._default_normalizer = None
    yield
    # Restore original
    kavalai.normalizer._default_normalizer = original


def test_get_default_normalizer(tmp_path, clean_default_normalizer):
    from kavalai.normalizer import get_default_normalizer, set_default_normalizer

    # 1. Nothing installed: an L2 normalizer, built once.
    normalizer = get_default_normalizer()
    assert normalizer.l2 is True
    assert normalizer.center_enabled is False
    assert get_default_normalizer() is normalizer

    # 2. Installed explicitly — the library reads no environment variable.
    custom = Normalizer(l1=True, l2=False, center=True, center_vector=[0.1, 0.2])
    set_default_normalizer(custom)
    assert get_default_normalizer() is custom

    # 3. ``None`` restores the default.
    set_default_normalizer(None)
    restored = get_default_normalizer()
    assert restored is not custom
    assert restored.l2 is True


def test_center_vector_none():
    normalizer = Normalizer(center_vector=None, center=True)
    embeddings = np.array([[1.0, 2.0]])
    assert np.array_equal(normalizer.center(embeddings), embeddings)


def test_center_vector_size_mismatch():
    normalizer = Normalizer(center_vector=[1.0, 2.0], center=True)
    embeddings = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Embedding size 3 does not match center vector size 2"
    ):
        normalizer.center(embeddings)
