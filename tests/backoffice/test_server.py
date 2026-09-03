import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from kavalai.backoffice import db
from kavalai.backoffice.server import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def test_a_missing_required_setting_stops_the_backoffice(monkeypatch):
    """No development fallback: a cookie key nobody set is a cookie key
    everybody knows."""
    from kavalai.backoffice.server import required_setting

    monkeypatch.delenv("KAVALAI_BO_SESSION_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="KAVALAI_BO_SESSION_SECRET_KEY is not set"):
        required_setting("KAVALAI_BO_SESSION_SECRET_KEY")

    monkeypatch.setenv("KAVALAI_BO_SESSION_SECRET_KEY", "")
    with pytest.raises(RuntimeError, match="is not set"):
        required_setting("KAVALAI_BO_SESSION_SECRET_KEY")

    monkeypatch.setenv("KAVALAI_BO_SESSION_SECRET_KEY", "s3cret")
    assert required_setting("KAVALAI_BO_SESSION_SECRET_KEY") == "s3cret"


@pytest.mark.asyncio
async def test_user_details_unauthorized(client):
    response = await client.get("/user/get_details")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_projects_create_unauthorized(client):
    response = await client.post("/projects/create", json={"name": "New Project"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_projects_all(client, backoffice_db):
    # To test this, we need to bypass assert_logged_in or mock the session
    with (
        patch("kavalai.backoffice.server.is_logged_in", return_value=True),
        patch("kavalai.backoffice.server.Request.session", new_callable=MagicMock),
    ):
        user_id = str(uuid.uuid4())
        # mock_session.get.return_value = {"id": user_id} # This doesn't work easily with FastAPI Request

        # Alternative: mock assert_logged_in and get user_info from session differently
        with (
            patch("kavalai.backoffice.server.assert_logged_in"),
            patch("starlette.requests.Request.session", {"user_info": {"id": user_id}}),
        ):
            with patch("kavalai.backoffice.db.get_user_projects", return_value=[]):
                response = await client.get("/projects/all")
                assert response.status_code == 200
                assert response.json() == []


@pytest.mark.asyncio
async def test_projects_test_connection_reaches_the_database(
    client, backoffice_db, postgres_container
):
    """The endpoint opens a session on the described database and runs a
    query, so a real reachable database answers success and a wrong password
    surfaces as a 400 rather than as a crash."""
    data = {
        "name": "Probe",
        "db_user": postgres_container.username,
        "db_password": postgres_container.password,
        "db_host": postgres_container.get_container_host_ip(),
        "db_port": int(postgres_container.get_exposed_port(5432)),
        "db_name": postgres_container.dbname,
    }
    with patch("kavalai.backoffice.server.assert_logged_in"):
        response = await client.post("/projects/test-connection/new", json=data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        response = await client.post(
            "/projects/test-connection/new", json={**data, "db_password": "wrong"}
        )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Failed to connect")


@pytest.mark.asyncio
async def test_access_denied_propagates_403_not_503(client, backoffice_db):
    """A 403 raised inside get_backoffice_session must not be masked as a 503.

    Regression: the session context manager used to wrap the whole body in a
    broad ``except Exception`` and re-raise every error as a 503 "database not
    connected", hiding intentional access-control errors.
    """
    from fastapi import HTTPException

    project_id = uuid.uuid4()
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch(
            "kavalai.backoffice.server.assert_is_member",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=403, detail="Must be a member of the project."
                )
            ),
        ),
    ):
        response = await client.get(f"/agents/all/{project_id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "Must be a member of the project."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path, target, extra_kwargs",
    [
        ("/agents/stats", "get_daily_stats", {"days": 7}),
        ("/agents/summary-stats", "get_summary_stats", {}),
    ],
)
async def test_agents_stats_forward_the_query_parameters(
    client, backoffice_db, path, target, extra_kwargs
):
    """What the route owns is the plumbing: the optional ``agent_id`` query
    parameter (and the ``days`` default) must reach the stats function as
    keyword arguments on the project session."""
    project_id = uuid.uuid4()
    project = db.Project(
        id=project_id,
        name="P1",
        db_user="u",
        db_password="p",
        db_host="h",
        db_port=5432,
        db_name="d",
    )
    backoffice_db.add(project)
    await backoffice_db.commit()

    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch(
            "kavalai.backoffice.server.get_project_and_assert_access",
            return_value=project,
        ),
        patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm,
        patch(
            f"kavalai.backoffice.server.agent_stats.{target}", return_value={}
        ) as mock_stats,
    ):
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        response = await client.get(f"{path}/{project_id}")
        assert response.status_code == 200
        mock_stats.assert_called_once_with(mock_session, agent_id=None, **extra_kwargs)

        agent_id = uuid.uuid4()
        response = await client.get(f"{path}/{project_id}?agent_id={agent_id}")
        assert response.status_code == 200
        mock_stats.assert_called_with(mock_session, agent_id=agent_id, **extra_kwargs)


@pytest.mark.asyncio
async def test_projects_get_llm_call_stats(client, backoffice_db):
    project_id = uuid.uuid4()
    project = db.Project(
        id=project_id,
        name="P1",
        db_user="u",
        db_password="p",
        db_host="h",
        db_port=5432,
        db_name="d",
    )
    backoffice_db.add(project)
    await backoffice_db.commit()

    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch(
            "kavalai.backoffice.server.get_project_and_assert_access",
            return_value=project,
        ),
        patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm,
    ):
        mock_session = AsyncMock()
        # mock_sm returns an async_sessionmaker object
        # which when called returns a context manager (mock_session)
        mock_sessionmaker = MagicMock(return_value=mock_session)
        mock_sm.return_value = mock_sessionmaker
        mock_session.__aenter__.return_value = mock_session

        with patch(
            "kavalai.agent_service.AgentService.get_model_call_stats",
            return_value=[],
        ) as mock_get_stats:
            response = await client.get(f"/projects/{project_id}/llm-call-stats")

            assert response.status_code == 200
            assert response.json() == []
            mock_get_stats.assert_called_once()


@pytest.mark.asyncio
async def test_users_all(client, backoffice_db):
    user_id = str(uuid.uuid4())
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": user_id, "is_admin": True}},
        ),
    ):
        # Add some users to backoffice_db
        u1 = db.User(email="u1@test.com", name="User 1")
        u2 = db.User(email="u2@test.com", name="User 2")
        backoffice_db.add_all([u1, u2])
        await backoffice_db.commit()

        response = await client.get("/users/all")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        emails = [u["email"] for u in data]
        assert "u1@test.com" in emails
        assert "u2@test.com" in emails


@pytest.mark.asyncio
async def test_users_create(client, backoffice_db):
    user_id = str(uuid.uuid4())
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": user_id, "is_admin": True}},
        ),
    ):
        new_user_data = {"email": "new@test.com", "name": "New User", "is_admin": False}
        response = await client.post("/users/create", json=new_user_data)
        assert response.status_code == 200
        assert response.json()["email"] == "new@test.com"


@pytest.mark.asyncio
async def test_users_update(client, backoffice_db):
    target_user_id = uuid.uuid4()
    u = db.User(id=target_user_id, email="old@test.com", name="Old Name")
    backoffice_db.add(u)
    await backoffice_db.commit()

    user_id = str(uuid.uuid4())
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": user_id, "is_admin": True}},
        ),
    ):
        update_data = {"name": "Updated Name"}
        response = await client.put(f"/users/update/{target_user_id}", json=update_data)
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_users_delete(client, backoffice_db):
    target_user_id = uuid.uuid4()
    u = db.User(id=target_user_id, email="del@test.com", name="Del Me")
    backoffice_db.add(u)
    await backoffice_db.commit()

    user_id = str(uuid.uuid4())
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": user_id, "is_admin": True}},
        ),
    ):
        response = await client.delete(f"/users/delete/{target_user_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


@pytest.mark.asyncio
async def test_users_delete_self_fails(client, backoffice_db):
    user_id = str(uuid.uuid4())
    u = db.User(id=uuid.UUID(user_id), email="admin@test.com", name="Admin")
    backoffice_db.add(u)
    await backoffice_db.commit()

    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": user_id, "is_admin": True}},
        ),
    ):
        response = await client.delete(f"/users/delete/{user_id}")
        # Initially this might be 200, but we want it to be 400
        assert response.status_code == 400
        assert response.json()["detail"] == "You cannot delete yourself."


@pytest.mark.asyncio
async def test_render_workflow_svg_unauthorized(client):
    response = await client.post(
        "/workflows/render-svg", json={"workflow": {"nodes": []}}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_render_workflow_svg(client):
    workflow = {
        "start": "s",
        "nodes": [
            {"name": "s", "type": "start", "next": "reply"},
            {"name": "reply", "type": "llm", "next": "e"},
            {"name": "e", "type": "end"},
        ],
    }
    with patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(uuid.uuid4())}},
    ):
        response = await client.post(
            "/workflows/render-svg", json={"workflow": workflow}
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.text.startswith("<svg")
    assert ">reply<" in response.text and ">start<" in response.text


@pytest.mark.asyncio
async def test_render_workflow_svg_bad_input(client):
    with patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(uuid.uuid4())}},
    ):
        response = await client.post(
            "/workflows/render-svg", json={"workflow": "not-a-workflow"}
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_user_details_repairs_stale_active_project(client, backoffice_db):
    """A session pointing at an unreachable project is repaired on page load
    instead of turning every project-scoped request into a 403."""
    user_id = uuid.uuid4()
    revoked = db.Project(id=uuid.uuid4(), name="B Revoked")
    reachable = db.Project(id=uuid.uuid4(), name="A Reachable")
    user = db.User(
        id=user_id,
        email="stale@example.com",
        name="Stale",
        active_project_id=revoked.id,
    )
    membership = db.ProjectMembership(
        user_id=user_id, project_id=reachable.id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([revoked, reachable, user, membership])
    await backoffice_db.commit()

    session = {
        "user_info": {
            "id": str(user_id),
            "email": "stale@example.com",
            "is_admin": False,
            "active_project_id": str(revoked.id),
        }
    }
    with patch("starlette.requests.Request.session", session):
        response = await client.get("/user/get_details")

    assert response.status_code == 200
    assert response.json()["active_project_id"] == str(reachable.id)
    await backoffice_db.refresh(user)
    assert user.active_project_id == reachable.id


@pytest.mark.asyncio
async def test_user_details_keeps_valid_active_project(client, backoffice_db):
    """A reachable selection is returned unchanged."""
    user_id = uuid.uuid4()
    project = db.Project(id=uuid.uuid4(), name="Mine")
    user = db.User(
        id=user_id,
        email="valid@example.com",
        name="Valid",
        active_project_id=project.id,
    )
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project.id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([project, user, membership])
    await backoffice_db.commit()

    session = {
        "user_info": {
            "id": str(user_id),
            "email": "valid@example.com",
            "is_admin": False,
            "active_project_id": str(project.id),
        }
    }
    with patch("starlette.requests.Request.session", session):
        response = await client.get("/user/get_details")

    assert response.status_code == 200
    assert response.json()["active_project_id"] == str(project.id)


def test_rag_service_for_project_follows_the_database_type(tmp_path):
    """One helper picks the backend, so every explorer endpoint agrees."""
    from kavalai.backoffice.server import rag_service_for_project
    from kavalai.rag import PostgresRagService, SqliteRagService

    sqlite_project = db.Project(
        name="local", db_type="sqlite", db_name=str(tmp_path / "agents.db")
    )
    service = rag_service_for_project(sqlite_project, model="fake/model")
    assert isinstance(service, SqliteRagService)
    assert service.model == "fake/model"
    service.close()

    postgres_project = db.Project(
        name="pg",
        db_host="h",
        db_port=5432,
        db_user="u",
        db_password="p",
        db_name="d",
        db_schema="agents",
    )
    factory = MagicMock()
    service = rag_service_for_project(postgres_project, session_factory=factory)
    assert isinstance(service, PostgresRagService)
    assert service.session_maker is factory
    assert service.schema == "agents"
    assert service.model is None


@pytest.mark.asyncio
async def test_a_sqlite_project_is_browsed_from_its_file(
    client, backoffice_db, tmp_path
):
    """End to end on a SQLite project: the agent tables an agent server
    created with ``init_sqlite`` and a RAG index in the same file are read by
    the stats, connection-test and RAG endpoints."""
    from kavalai.db import db_manager
    from kavalai.rag import SqliteRagService

    path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=path)
    index = SqliteRagService(path, model="fake/model")
    index.embedding_client = MagicMock(
        compute_embeddings=AsyncMock(
            return_value=(
                [[1.0, 0.0], [0.0, 1.0]],
                MagicMock(total_tokens=2, batch_size=2),
            )
        )
    )
    await index.index_batch(
        texts=["a", "b"], metadata_list=[{}, {}], collection_name="facts"
    )
    index.close()

    project = db.Project(id=uuid.uuid4(), name="local", db_type="sqlite", db_name=path)
    backoffice_db.add(project)
    await backoffice_db.commit()

    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch(
            "kavalai.backoffice.server.get_project_and_assert_access",
            return_value=project,
        ),
    ):
        response = await client.get(f"/projects/{project.id}/rag/stats")
        assert response.status_code == 200
        assert response.json() == {
            "total_entries": 2,
            "total_collections": 1,
            "collections": ["facts"],
        }

        response = await client.get(f"/projects/{project.id}/llm-call-stats")
        assert response.status_code == 200

        response = await client.post(f"/projects/test-connection/{project.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
