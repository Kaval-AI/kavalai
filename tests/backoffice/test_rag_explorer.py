import pytest
from unittest.mock import patch, MagicMock
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
