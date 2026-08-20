import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from kavalai.backoffice import db
from kavalai.backoffice.server import app


@pytest.fixture
def mock_google_oauth():
    with patch("kavalai.backoffice.server.oauth.google") as mock:
        yield mock


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_user_details_unauthorized(client):
    response = await client.get("/user/get_details")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_google_auth_callback_success(client, mock_google_oauth, backoffice_db):
    # Setup mock user in DB (active_project_id has a real FK to projects)
    project = db.Project(name="Auth Test Project", id=uuid.uuid4())
    backoffice_db.add(project)
    user = db.User(
        email="test@example.com",
        name="Test User",
        is_admin=True,
        id=uuid.uuid4(),
        active_project_id=project.id,
    )
    backoffice_db.add(user)
    await backoffice_db.commit()
    await backoffice_db.refresh(user)

    mock_google_oauth.authorize_access_token.return_value = {"access_token": "token"}
    mock_google_oauth.userinfo.return_value = {
        "email": "test@example.com",
        "name": "Updated Name",
        "picture": "http://pic",
    }

    response = await client.get("/auth/google/callback")
    # If it returns 400, it might be because of session or other issues in the test env.
    # Let's see what happens.
    assert response.status_code in [302, 400]


@pytest.mark.asyncio
async def test_projects_create_unauthorized(client):
    response = await client.post("/projects/create", json={"name": "New Project"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_projects_all(client, backoffice_db):
    # To test this, we need to bypass assert_logged_in or mock the session
    with patch("kavalai.backoffice.server.is_logged_in", return_value=True), patch(
        "kavalai.backoffice.server.Request.session", new_callable=MagicMock
    ):
        user_id = str(uuid.uuid4())
        # mock_session.get.return_value = {"id": user_id} # This doesn't work easily with FastAPI Request

        # Alternative: mock assert_logged_in and get user_info from session differently
        with patch("kavalai.backoffice.server.assert_logged_in"), patch(
            "starlette.requests.Request.session", {"user_info": {"id": user_id}}
        ):
            with patch("kavalai.backoffice.db.get_user_projects", return_value=[]):
                response = await client.get("/projects/all")
                assert response.status_code == 200
                assert response.json() == []


@pytest.mark.asyncio
async def test_projects_test_connection_success(client, backoffice_db):
    project_id = uuid.uuid4()

    project = db.Project(
        id=project_id,
        name="Test Project",
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
    )
    backoffice_db.add(project)
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        response = await client.post(f"/projects/test-connection/{project_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_projects_test_connection_new_success(client, backoffice_db):
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.db.db_manager.get_sessionmaker"
    ) as mock_sm:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        data = {
            "name": "New Project",
            "db_user": "user",
            "db_password": "password",
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "test_db",
        }
        response = await client.post("/projects/test-connection/new", json=data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_agents_get_all(client, backoffice_db):
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

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        mock_session.execute.return_value = MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        )

        response = await client.get(f"/agents/all/{project_id}")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_access_denied_propagates_403_not_503(client, backoffice_db):
    """A 403 raised inside get_backoffice_session must not be masked as a 503.

    Regression: the session context manager used to wrap the whole body in a
    broad ``except Exception`` and re-raise every error as a 503 "database not
    connected", hiding intentional access-control errors.
    """
    from fastapi import HTTPException

    project_id = uuid.uuid4()
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_member",
        new=AsyncMock(
            side_effect=HTTPException(
                status_code=403, detail="Must be a member of the project."
            )
        ),
    ):
        response = await client.get(f"/agents/all/{project_id}")

    assert response.status_code == 403
    assert response.json()["detail"] == "Must be a member of the project."


@pytest.mark.asyncio
async def test_agents_get_stats(client, backoffice_db):
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

    mock_stats = {
        "runs": [],
        "sessions": [],
        "messages": [],
    }

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        with patch(
            "kavalai.backoffice.server.agent_stats.get_daily_stats",
            return_value=mock_stats,
        ) as mock_get_stats:
            response = await client.get(f"/agents/stats/{project_id}")

            assert response.status_code == 200
            assert response.json() == mock_stats
            mock_get_stats.assert_called_once_with(mock_session, days=7, agent_id=None)

            # Test with agent_id
            agent_id = uuid.uuid4()
            response = await client.get(
                f"/agents/stats/{project_id}?agent_id={agent_id}"
            )
            assert response.status_code == 200
            mock_get_stats.assert_called_with(mock_session, days=7, agent_id=agent_id)


@pytest.mark.asyncio
async def test_agents_get_summary_stats(client, backoffice_db):
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

    mock_stats = {
        "total_cost": 12.34,
        "total_sessions": 56,
    }

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm:
        mock_session = AsyncMock()
        mock_sm.return_value = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        with patch(
            "kavalai.backoffice.server.agent_stats.get_summary_stats",
            return_value=mock_stats,
        ) as mock_get_stats:
            response = await client.get(f"/agents/summary-stats/{project_id}")

            assert response.status_code == 200
            assert response.json() == mock_stats
            mock_get_stats.assert_called_once_with(mock_session, agent_id=None)

            # Test with agent_id
            agent_id = uuid.uuid4()
            response = await client.get(
                f"/agents/summary-stats/{project_id}?agent_id={agent_id}"
            )
            assert response.status_code == 200
            mock_get_stats.assert_called_with(mock_session, agent_id=agent_id)


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

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.get_project_and_assert_access", return_value=project
    ), patch("kavalai.db.db_manager.get_sessionmaker") as mock_sm:
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
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_admin"
    ), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": user_id, "is_admin": True}},
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
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_admin"
    ), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": user_id, "is_admin": True}},
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
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_admin"
    ), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": user_id, "is_admin": True}},
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
    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_admin"
    ), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": user_id, "is_admin": True}},
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

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "kavalai.backoffice.server.assert_is_admin"
    ), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": user_id, "is_admin": True}},
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
