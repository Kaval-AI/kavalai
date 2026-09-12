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

import base64
import json
import pickle
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sklearn.decomposition import IncrementalPCA

from kavalai.backoffice import db
from kavalai.backoffice.server import app

COLLECTION = "test_col"
COLLECTION_MODEL = "fastembed/BAAI/bge-small-en-v1.5"


class MockSessionMaker:
    """Hands out one already-open session as an async context manager."""

    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        pass


@pytest_asyncio.fixture
async def project(backoffice_db, agents_db, monkeypatch):
    """A project whose collection has a trained PCA model and a cached sample."""
    from kavalai.backoffice import server

    project = db.Project(
        id=uuid.uuid4(),
        name="Test Project",
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
    )
    backoffice_db.add(project)

    ipca = IncrementalPCA(n_components=2)
    ipca.partial_fit(np.random.rand(10, 3))
    samples = [
        {"label": "sample1", "x": 0.1, "y": 0.2},
        {"label": "sample2", "x": 0.3, "y": 0.4},
    ]
    backoffice_db.add_all(
        [
            db.ProjectCache(
                project_id=project.id,
                name=f"pca_model_{COLLECTION}",
                value=base64.b64encode(pickle.dumps(ipca)).decode("utf-8"),
            ),
            db.ProjectCache(
                project_id=project.id,
                name=f"pca_sample_train_data_{COLLECTION}",
                value=json.dumps(samples),
            ),
        ]
    )
    await backoffice_db.commit()

    monkeypatch.setattr(server, "assert_logged_in", lambda r: None)
    monkeypatch.setattr(
        server, "get_project_and_assert_access", AsyncMock(return_value=project)
    )
    monkeypatch.setattr(
        server, "get_backoffice_session", lambda: MockSessionMaker(backoffice_db)
    )
    monkeypatch.setattr(
        server, "get_project_session", lambda p: MockSessionMaker(agents_db)
    )
    return project


@pytest.fixture
def rag_service_class():
    """The project's RAG service class, its instance's storage and embedding
    client mocked. The instance starts out without a model, as one built for
    a request that names none does."""
    with patch("kavalai.backoffice.server.PostgresRagService") as service_class:
        instance = service_class.return_value
        instance.model = None
        instance.query = AsyncMock(return_value=[])
        instance.list_collections = AsyncMock(return_value=[])
        instance.embedding_client.compute_embeddings = AsyncMock(
            return_value=([np.random.rand(3).tolist()], None)
        )
        instance.get_embeddings_by_ids = AsyncMock(return_value={})
        yield service_class


def _hit(model: str = COLLECTION_MODEL):
    hit = MagicMock()
    hit.id = uuid.uuid4()
    hit.content = "result content"
    hit.similarity = 0.9
    hit.model = model
    return hit


async def _query(project, **query_data) -> dict:
    query_data = {"text": "test query", "collection_name": COLLECTION, **query_data}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(f"/projects/{project.id}/rag/query", json=query_data)
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_projects_rag_query_with_pca(project, rag_service_class):
    instance = rag_service_class.return_value
    hit = _hit()
    instance.query.return_value = [hit]
    # The endpoint fetches result embeddings through the service
    # (storage is backend-owned).
    instance.get_embeddings_by_ids.return_value = {hit.id: [0.1, 0.2, 0.3]}

    data = await _query(project, model="test_model", top_k=5)

    assert data["pca_data"] is not None
    assert "query" in data["pca_data"]
    assert len(data["pca_data"]["samples"]) == 2
    assert data["pca_data"]["results"][0]["label"] == "result content"
    # The PCA was fitted on the stored embeddings, so the query is projected
    # from the collection's model, not the one the request named.
    assert instance.model == COLLECTION_MODEL
    instance.embedding_client.compute_embeddings.assert_awaited_once_with(
        texts=["test query"], normalize=False, normalizer=None
    )


@pytest.mark.asyncio
async def test_a_query_without_a_model_is_projected_with_the_collections(
    project, rag_service_class
):
    instance = rag_service_class.return_value
    instance.query.return_value = [_hit()]

    data = await _query(project)

    assert rag_service_class.call_args.args[1] is None
    assert data["pca_data"]["query"]["label"] == "test query"
    assert instance.model == COLLECTION_MODEL
    # The hits carry the model, so the registry is not consulted.
    instance.list_collections.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_query_without_hits_takes_the_model_from_the_registry(
    project, rag_service_class
):
    instance = rag_service_class.return_value
    instance.list_collections.return_value = [
        {"name": "other", "model": "openai/text-embedding-3-small"},
        {"name": COLLECTION, "model": COLLECTION_MODEL},
    ]

    data = await _query(project)

    assert data["results"] == []
    assert data["pca_data"]["results"] == []
    assert instance.model == COLLECTION_MODEL


@pytest.mark.asyncio
async def test_an_unregistered_collection_keeps_the_requested_model(
    project, rag_service_class
):
    instance = rag_service_class.return_value
    instance.model = "test_model"

    data = await _query(project, model="test_model")

    assert data["pca_data"] is not None
    assert instance.model == "test_model"
