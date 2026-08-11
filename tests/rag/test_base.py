import uuid
from typing import Optional, Union

import pytest

from kavalai.normalizer import Normalizer
from kavalai.rag import BaseRagService, RagServiceResult


def make_result(source_id: str, similarity: float) -> RagServiceResult:
    return RagServiceResult(
        id=uuid.uuid4(),
        model="test/embedding",
        collection_name="default",
        source_id=source_id,
        content=f"content of {source_id}",
        embedding_size=3,
        rag_metadata={},
        similarity=similarity,
    )


class InMemoryRagService(BaseRagService):
    """Minimal concrete backend returning canned query results."""

    def __init__(
        self, canned_results: Optional[dict[str, list[RagServiceResult]]] = None
    ):
        self.canned_results = canned_results or {}
        self.query_batch_calls: list[dict] = []

    async def index(
        self,
        text: str,
        source_metadata: Optional[dict] = None,
        collection_name: str = "default",
        source_id: str = "default",
    ):
        return None

    async def index_batch(
        self,
        texts: list[str],
        metadata_list: list[dict],
        source_ids: Optional[list[str]] = None,
        collection_name: str = "default",
    ):
        return []

    async def query(
        self,
        text: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        keep_best: bool = False,
    ) -> list[RagServiceResult]:
        return self.canned_results.get(text, [])[:top_k]

    async def query_batch(
        self,
        texts: list[str],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
    ) -> list[list[RagServiceResult]]:
        self.query_batch_calls.append(
            {
                "texts": texts,
                "top_k": top_k,
                "collection_name": collection_name,
                "source_ids": source_ids,
            }
        )
        return [self.canned_results.get(text, [])[:top_k] for text in texts]

    async def delete(self, item_id: uuid.UUID) -> None:
        return None

    async def delete_by_source_id(
        self,
        collection_name: str,
        source_id: Union[str, list[str]],
    ) -> None:
        return None


def test_base_rag_service_is_abstract():
    with pytest.raises(TypeError):
        BaseRagService()


@pytest.mark.asyncio
async def test_default_compute_similarity_matrix_min():
    service = InMemoryRagService(
        canned_results={
            "q1": [
                make_result("s1", 0.9),
                make_result("s1", 0.5),
                make_result("s2", 0.4),
            ],
            "q2": [make_result("s2", 0.8)],
        }
    )

    matrix = await service.compute_similarity_matrix(
        texts=["q1", "q2"], source_ids=["s1", "s2"], method="min"
    )

    # "min" distance == max similarity per source; missing pairs score 0.0
    assert matrix == [[0.9, 0.4], [0.0, 0.8]]


@pytest.mark.asyncio
async def test_default_compute_similarity_matrix_avg():
    service = InMemoryRagService(
        canned_results={
            "q1": [
                make_result("s1", 0.9),
                make_result("s1", 0.5),
                make_result("s2", 0.4),
            ],
        }
    )

    matrix = await service.compute_similarity_matrix(
        texts=["q1"], source_ids=["s1", "s2"], method="avg"
    )

    assert matrix == [[pytest.approx(0.7), 0.4]]


@pytest.mark.asyncio
async def test_default_compute_similarity_matrix_ignores_unknown_sources():
    # Backend returns a source that was not requested; it must not appear.
    service = InMemoryRagService(
        canned_results={
            "q1": [make_result("s1", 0.9), make_result("unrequested", 0.99)],
        }
    )

    matrix = await service.compute_similarity_matrix(
        texts=["q1"], source_ids=["s1"], method="min"
    )

    assert matrix == [[0.9]]


@pytest.mark.asyncio
async def test_default_compute_similarity_matrix_empty_inputs():
    service = InMemoryRagService()

    assert await service.compute_similarity_matrix(texts=[], source_ids=["s1"]) == []
    assert await service.compute_similarity_matrix(texts=["q1"], source_ids=[]) == [[]]
    # No query_batch round trip for empty inputs
    assert service.query_batch_calls == []


@pytest.mark.asyncio
async def test_default_compute_similarity_matrix_top_k():
    service = InMemoryRagService(canned_results={"q1": [make_result("s1", 0.9)]})

    await service.compute_similarity_matrix(texts=["q1"], source_ids=["s1", "s2"])

    assert len(service.query_batch_calls) == 1
    call = service.query_batch_calls[0]
    assert call["texts"] == ["q1"]
    assert call["source_ids"] == ["s1", "s2"]
    assert call["top_k"] == 2 * service.similarity_matrix_candidates_per_source


@pytest.mark.asyncio
async def test_default_learn_normalizer():
    service = InMemoryRagService()

    normalizer = await service.learn_normalizer()
    assert isinstance(normalizer, Normalizer)

    # collection_name is accepted (and ignored by the default implementation)
    normalizer_coll = await service.learn_normalizer(collection_name="anything")
    assert isinstance(normalizer_coll, Normalizer)


async def test_count_entries_is_optional_for_backends():
    service = InMemoryRagService()
    with pytest.raises(NotImplementedError, match="count_entries"):
        await service.count_entries("docs")


async def test_iter_entries_is_optional_for_backends():
    service = InMemoryRagService()
    with pytest.raises(NotImplementedError, match="iter_entries"):
        service.iter_entries("docs")
