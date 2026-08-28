import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient
from kavalai.backoffice.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user_session():
    with (
        patch("kavalai.backoffice.server.is_logged_in", return_value=True),
        patch("kavalai.backoffice.server.assert_logged_in", return_value=None),
        patch(
            "kavalai.backoffice.server.get_project_and_assert_access"
        ) as mock_get_project,
    ):
        project = MagicMock()
        project.db_user = "user"
        project.db_password = "pass"
        project.db_host = "localhost"
        project.db_port = 5432
        project.db_name = "dbname"
        mock_get_project.return_value = project
        yield mock_get_project


@pytest.mark.asyncio
async def test_projects_rag_query_with_normalizer(client, mock_user_session):
    project_id = uuid4()

    # Mock RagService and db_manager
    with (
        patch(
            "kavalai.backoffice.server.db_manager.get_sessionmaker"
        ) as mock_get_sessionmaker,
        patch("kavalai.backoffice.server.PostgresRagService") as MockRagService,
        patch("kavalai.normalizer.Normalizer.from_yaml") as mock_from_yaml,
    ):
        mock_session_maker = MagicMock()
        mock_get_sessionmaker.return_value = mock_session_maker

        mock_session = AsyncMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session

        mock_rag_service = MockRagService.return_value
        mock_rag_service.query = AsyncMock(
            return_value=[{"similarity": 0.9, "content": "test"}]
        )

        mock_normalizer = MagicMock()
        mock_from_yaml.return_value = mock_normalizer

        query_data = {
            "model": "test-model",
            "text": "test query",
            "normalizer_yaml": "l1: false\nl2: true",
        }

        with patch(
            "kavalai.backoffice.server.get_backoffice_session"
        ) as mock_get_bo_session:
            mock_bo_session = AsyncMock()
            mock_get_bo_session.return_value.__aenter__.return_value = mock_bo_session
            mock_bo_session.execute.return_value.scalar_one_or_none.return_value = None

            response = client.post(f"/projects/{project_id}/rag/query", json=query_data)

            assert response.status_code == 200
            assert response.json() == {
                "results": [{"similarity": 0.9, "content": "test"}],
                "pca_data": None,
            }

        mock_from_yaml.assert_called_once_with("l1: false\nl2: true")
        MockRagService.assert_called_once()
        args, kwargs = MockRagService.call_args
        assert kwargs["normalizer"] == mock_normalizer
        assert args[1] == "test-model"


@pytest.mark.asyncio
async def test_projects_rag_query_invalid_normalizer(client, mock_user_session):
    project_id = uuid4()

    with (
        patch("kavalai.backoffice.server.db_manager.get_sessionmaker"),
        patch(
            "kavalai.normalizer.Normalizer.from_yaml",
            side_effect=Exception("Invalid YAML"),
        ),
    ):
        query_data = {
            "model": "test-model",
            "text": "test query",
            "normalizer_yaml": "invalid: yaml",
        }

        response = client.post(f"/projects/{project_id}/rag/query", json=query_data)

        assert response.status_code == 400
        assert "Invalid normalizer YAML" in response.json()["detail"]
