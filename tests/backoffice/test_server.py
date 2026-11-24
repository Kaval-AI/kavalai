import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from kavalai.agent_service import ModelCallSummary, ModelCallTotals
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


def test_google_sign_in_is_optional_but_never_half_configured(monkeypatch):
    from kavalai.backoffice.server import google_oauth_settings

    monkeypatch.delenv("KAVALAI_BO_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("KAVALAI_BO_GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("KAVALAI_BO_FRONTEND_URL", raising=False)
    assert google_oauth_settings() is None

    monkeypatch.setenv("KAVALAI_BO_GOOGLE_CLIENT_ID", "id")
    with pytest.raises(RuntimeError, match="needs both"):
        google_oauth_settings()

    monkeypatch.setenv("KAVALAI_BO_GOOGLE_CLIENT_SECRET", "secret")
    with pytest.raises(RuntimeError, match="KAVALAI_BO_FRONTEND_URL is not set"):
        google_oauth_settings()

    monkeypatch.setenv("KAVALAI_BO_FRONTEND_URL", "http://localhost:4200")
    assert google_oauth_settings() == {
        "client_id": "id",
        "client_secret": "secret",
        "frontend_url": "http://localhost:4200",
    }


@pytest.mark.asyncio
async def test_auth_methods_follow_the_configuration(client):
    response = await client.get("/auth/methods")
    assert response.json() == {"password": True, "google": True}

    with patch("kavalai.backoffice.server.GOOGLE_OAUTH", None):
        response = await client.get("/auth/methods")
        assert response.json() == {"password": True, "google": False}


@pytest.mark.asyncio
async def test_google_routes_are_absent_without_a_client(client):
    with patch("kavalai.backoffice.server.GOOGLE_OAUTH", None):
        assert (await client.get("/login")).status_code == 404
        assert (await client.get("/auth/google/callback")).status_code == 404


@pytest.mark.asyncio
async def test_password_login_opens_a_session(client, backoffice_db):
    """A real sign-in, with no patched session: the cookie the login sets is
    what the next request is authenticated by."""
    from kavalai.backoffice.users import hash_password

    project = db.Project(id=uuid.uuid4(), name="Mine")
    user = db.User(
        id=uuid.uuid4(),
        email="pw@example.com",
        name="Pw",
        is_admin=True,
        password_hash=hash_password("correct horse"),
    )
    membership = db.ProjectMembership(
        user_id=user.id, project_id=project.id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([project, user, membership])
    await backoffice_db.commit()

    response = await client.post(
        "/auth/login", json={"email": "pw@example.com", "password": "correct horse"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": str(user.id),
        "email": "pw@example.com",
        "name": "Pw",
        "picture": None,
        "is_admin": True,
        "active_project_id": str(project.id),
    }

    details = await client.get("/user/get_details")
    assert details.status_code == 200
    assert details.json()["email"] == "pw@example.com"

    assert (await client.get("/logout")).json() == {"logged_in": False}
    assert (await client.get("/user/get_details")).status_code == 401
    # Logging out twice is not an error.
    assert (await client.get("/logout")).status_code == 200


@pytest.mark.asyncio
async def test_password_login_refuses_every_failure_alike(client, backoffice_db):
    from kavalai.backoffice.users import hash_password

    backoffice_db.add_all(
        [
            db.User(
                email="pw@example.com",
                name="Pw",
                password_hash=hash_password("correct horse"),
            ),
            db.User(email="google@example.com", name="Google only"),
        ]
    )
    await backoffice_db.commit()

    for body in [
        {"email": "pw@example.com", "password": "wrong horse"},
        {"email": "google@example.com", "password": "correct horse"},
        {"email": "google@example.com", "password": ""},
        {"email": "nobody@example.com", "password": "correct horse"},
    ]:
        response = await client.post("/auth/login", json=body)
        assert response.status_code == 401, body
        assert response.json() == {"detail": "Invalid email or password."}
        assert (await client.get("/user/get_details")).status_code == 401

    for body in [{}, {"email": "pw@example.com"}, {"email": 1, "password": 2}]:
        response = await client.post("/auth/login", json=body)
        assert response.status_code == 400, body


@pytest.mark.asyncio
async def test_user_endpoints_never_return_the_password_hash(client, backoffice_db):
    from kavalai.backoffice.users import hash_password, verify_password

    existing = db.User(
        id=uuid.uuid4(),
        email="has@example.com",
        name="Has",
        password_hash=hash_password("correct horse"),
    )
    backoffice_db.add(existing)
    await backoffice_db.commit()

    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("kavalai.backoffice.server.assert_is_admin"),
        patch(
            "starlette.requests.Request.session",
            {"user_info": {"id": str(uuid.uuid4()), "is_admin": True}},
        ),
    ):
        listed = (await client.get("/users/all")).json()
        assert listed[0]["has_password"] is True
        assert "password_hash" not in listed[0]
        assert "scrypt$" not in str(listed)

        created = await client.post(
            "/users/create",
            json={
                "email": "new@example.com",
                "name": "New",
                "password": "battery staple",
                "password_hash": "client-chosen",
                "has_password": False,
            },
        )
        assert created.status_code == 200
        assert created.json()["has_password"] is True
        assert "password_hash" not in created.json()
        assert "password" not in created.json()

        too_short = await client.post(
            "/users/create",
            json={"email": "short@example.com", "name": "S", "password": "short"},
        )
        assert too_short.status_code == 400
        assert "at least 8" in too_short.json()["detail"]

        # An edit that leaves the password blank keeps the stored one.
        renamed = await client.put(
            f"/users/update/{existing.id}", json={"name": "Renamed", "password": ""}
        )
        assert renamed.json()["name"] == "Renamed"
        assert renamed.json()["has_password"] is True

        changed = await client.put(
            f"/users/update/{existing.id}", json={"password": "battery staple"}
        )
        assert changed.status_code == 200

    backoffice_db.expire_all()
    rows = {
        user.email: user
        for user in (await backoffice_db.execute(db.select(db.User))).scalars()
    }
    assert verify_password("battery staple", rows["new@example.com"].password_hash)
    assert verify_password("battery staple", rows["has@example.com"].password_hash)
    assert "short@example.com" not in rows


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
    admin = {"user_info": {"id": str(uuid.uuid4()), "is_admin": True}}
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("starlette.requests.Request.session", admin),
    ):
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

        agent_id, session_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        with patch(
            "kavalai.agent_service.AgentService.get_model_call_stats",
            return_value=[],
        ) as mock_get_stats:
            response = await client.get(
                f"/projects/{project_id}/llm-call-stats",
                params={
                    "call_type": "embedding",
                    "limit": 5,
                    "offset": 10,
                    "agent_id": str(agent_id),
                    "session_id": str(session_id),
                    "run_id": str(run_id),
                    "search": "claims",
                    "start_date": "2026-11-23T00:00:00Z",
                    "end_date": "2026-11-24T23:59:59Z",
                },
            )

            assert response.status_code == 200
            assert response.json() == []
            mock_get_stats.assert_called_once_with(
                call_type="embedding",
                limit=5,
                offset=10,
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                search="claims",
                start=datetime(2026, 11, 23, tzinfo=timezone.utc),
                end=datetime(2026, 11, 24, 23, 59, 59, tzinfo=timezone.utc),
            )

        with patch(
            "kavalai.agent_service.AgentService.summarise_model_calls",
            return_value=ModelCallSummary(
                totals=ModelCallTotals(calls=2), by_model=[], by_day=[]
            ),
        ) as mock_summary:
            response = await client.get(
                f"/projects/{project_id}/llm-call-stats/summary",
                params={"agent_id": str(agent_id), "search": "claims"},
            )
            assert response.status_code == 200
            assert response.json()["totals"]["calls"] == 2
            mock_summary.assert_called_once_with(
                call_type=None,
                agent_id=agent_id,
                session_id=None,
                run_id=None,
                search="claims",
                start=None,
                end=None,
            )

        with patch(
            "kavalai.agent_service.AgentService.summarise_model_calls",
            side_effect=OSError("connection refused"),
        ):
            response = await client.get(
                f"/projects/{project_id}/llm-call-stats/summary"
            )
            assert response.status_code == 503
            assert "not connected" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_rag_query_without_text_is_refused(client):
    """Only the text is required; the collection supplies the model."""
    project = db.Project(
        id=uuid.uuid4(),
        name="P1",
        db_user="u",
        db_password="p",
        db_host="h",
        db_port=5432,
        db_name="d",
    )
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch(
            "kavalai.backoffice.server.get_project_and_assert_access",
            return_value=project,
        ),
    ):
        response = await client.post(
            f"/projects/{project.id}/rag/query",
            json={"model": "fake/model", "collection_name": "facts"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "text is required"


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


@pytest.mark.asyncio
async def test_rag_service_for_project_follows_the_database_type(tmp_path):
    """One helper picks the backend, so every explorer endpoint agrees, and
    neither backend may provision: the explorer only reads."""
    from kavalai.backoffice.server import rag_service_for_project
    from kavalai.rag import PostgresRagService, SqliteRagService

    path = str(tmp_path / "agents.db")
    index = SqliteRagService(path, model="fake/model")
    await index.ensure_registry()
    index.close()
    sqlite_project = db.Project(name="local", db_type="sqlite", db_name=path)
    service = rag_service_for_project(sqlite_project, model="fake/model")
    assert isinstance(service, SqliteRagService)
    assert service.model == "fake/model"
    assert service.read_only is True
    assert service.provision is False
    service.close()

    sqlite_project.read_only = False
    service = rag_service_for_project(sqlite_project, model="fake/model")
    assert service.read_only is False
    assert service.provision is False
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
    assert service.provision is False

    postgres_project.rag_schema = "rag"
    service = rag_service_for_project(postgres_project, session_factory=factory)
    assert service.schema == "rag"


def test_a_read_only_sqlite_project_does_not_get_its_index_file_created(tmp_path):
    """A missing file is an error, not a new empty index on disk."""
    from kavalai.backoffice.server import rag_service_for_project

    project = db.Project(
        name="local", db_type="sqlite", db_name=str(tmp_path / "missing.db")
    )

    with pytest.raises(FileNotFoundError):
        rag_service_for_project(project, model="fake/model")
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.asyncio
async def test_a_sqlite_project_is_browsed_from_its_file(
    client, backoffice_db, tmp_path
):
    """End to end on a SQLite project: the agent tables an agent server
    created with ``init_sqlite`` and a RAG index in the same file are read by
    the stats, model-call, connection-test and RAG endpoints."""
    from kavalai.agent_service import AgentService
    from kavalai.db import ModelCallStat, db_manager
    from kavalai.rag import SqliteRagService
    from kavalai.testing import FakeEmbeddingClient, fake_providers

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

    # Written the way an agent server writes: through the file's shared
    # engine. The project's own sessionmaker is read-only.
    session_id, run_id = uuid.uuid4(), uuid.uuid4()
    await AgentService(
        db_manager.get_sqlite_sessionmaker(db_path=path)
    ).add_model_call_stats(
        ModelCallStat(call_type="embedding", model="fake/model", total_tokens=2),
        session_id=session_id,
        run_id=run_id,
    )

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

        response = await client.get(
            f"/projects/{project.id}/llm-call-stats", params={"run_id": str(run_id)}
        )
        assert response.status_code == 200
        [call] = response.json()
        assert (call["session_id"], call["run_id"]) == (str(session_id), str(run_id))

        response = await client.get(
            f"/projects/{project.id}/llm-call-stats",
            params={"run_id": str(uuid.uuid4())},
        )
        assert response.json() == []

        response = await client.get(f"/projects/{project.id}/rag/collections")
        assert response.status_code == 200
        assert [(c["name"], c["model"], c["count"]) for c in response.json()] == [
            ("facts", "fake/model", 2)
        ]

        # No model in the request: the collection is embedded with its own.
        with fake_providers(embedding=FakeEmbeddingClient(dimension=2)):
            response = await client.post(
                f"/projects/{project.id}/rag/query",
                json={"text": "a", "collection_name": "facts"},
            )
        assert response.status_code == 200
        assert response.json()["pca_data"] is None
        assert {r["model"] for r in response.json()["results"]} == {"fake/model"}

        response = await client.post(f"/projects/test-connection/{project.id}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"


async def _member_of_new_project(backoffice_db, role, **project_fields):
    """A user, a project with a stored password, and the user's membership."""
    user = db.User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@test", name="m")
    project = db.Project(id=uuid.uuid4(), name="P", **project_fields)
    backoffice_db.add_all([user, project])
    await backoffice_db.commit()
    backoffice_db.add(
        db.ProjectMembership(user_id=user.id, project_id=project.id, role=role)
    )
    await backoffice_db.commit()
    return user, project


def _signed_in(user, is_admin=False):
    session = {"user_info": {"id": str(user.id), "email": user.email}}
    session["user_info"]["is_admin"] = is_admin
    return (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("starlette.requests.Request.session", session),
    )


@pytest.mark.asyncio
async def test_project_listings_never_carry_the_database_password(
    client, backoffice_db
):
    """A viewer receives the project, so the password would let them connect
    to the agent database around their role and the read-only setting."""
    viewer, project = await _member_of_new_project(
        backoffice_db,
        db.ProjectRole.viewer,
        db_host="h",
        db_user="u",
        db_password="top-secret",
        db_name="d",
    )
    login, session = _signed_in(viewer)
    with login, session:
        listed = await client.get("/projects/all")
        single = await client.get(f"/projects/get/{project.id}")

    for response in (listed, single):
        assert response.status_code == 200
        assert "top-secret" not in response.text
        assert "db_password" not in response.text
    assert listed.json()[0]["has_password"] is True
    assert listed.json()[0]["role"] == "viewer"
    assert single.json()["has_password"] is True


@pytest.mark.asyncio
async def test_only_an_administrator_can_reveal_the_password(client, backoffice_db):
    owner, project = await _member_of_new_project(
        backoffice_db, db.ProjectRole.owner, db_password="top-secret"
    )
    login, session = _signed_in(owner)
    with login, session:
        refused = await client.post(f"/projects/{project.id}/reveal-password")
    login, session = _signed_in(owner, is_admin=True)
    with login, session:
        revealed = await client.post(f"/projects/{project.id}/reveal-password")

    assert refused.status_code == 403
    assert "top-secret" not in refused.text
    assert revealed.status_code == 200
    assert revealed.json() == {"db_password": "top-secret"}


@pytest.mark.asyncio
async def test_saving_without_the_password_keeps_it(client, backoffice_db):
    owner, project = await _member_of_new_project(
        backoffice_db,
        db.ProjectRole.owner,
        db_host="h",
        db_user="u",
        db_password="top-secret",
        db_name="d",
    )
    login, session = _signed_in(owner)
    with login, session:
        listed = (await client.get("/projects/all")).json()[0]
        saved = await client.put(
            f"/projects/update/{project.id}", json={**listed, "description": "d2"}
        )
        moved = await client.put(
            f"/projects/update/{project.id}", json={**listed, "db_host": "evil"}
        )

    assert saved.status_code == 200
    assert "top-secret" not in saved.text
    assert saved.json()["description"] == "d2"
    assert moved.status_code == 400
    assert "password again" in moved.json()["detail"]
    await backoffice_db.refresh(project)
    assert project.db_password == "top-secret"
    assert project.db_host == "h"


@pytest.mark.asyncio
async def test_a_connection_test_checks_unsaved_changes_with_the_stored_password(
    client, backoffice_db, postgres_container
):
    """The owner tests a form before saving it; the stored password is used,
    but never for another server, and a viewer may test only what is saved."""
    reachable = dict(
        db_host=postgres_container.get_container_host_ip(),
        db_port=int(postgres_container.get_exposed_port(5432)),
        db_user=postgres_container.username,
        db_password=postgres_container.password,
        db_name=postgres_container.dbname,
        db_schema="public",
    )
    owner, project = await _member_of_new_project(
        backoffice_db, db.ProjectRole.owner, **reachable
    )
    form = {key: value for key, value in reachable.items() if key != "db_password"}

    login, session = _signed_in(owner)
    with login, session:
        unsaved = await client.post(
            f"/projects/test-connection/{project.id}",
            json={**form, "db_schema": "elsewhere"},
        )
        moved = await client.post(
            f"/projects/test-connection/{project.id}",
            json={**form, "db_host": "203.0.113.1"},
        )
    assert unsaved.status_code == 200
    assert moved.status_code == 400
    assert "password again" in moved.json()["detail"]

    viewer = db.User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@test", name="v")
    backoffice_db.add(viewer)
    await backoffice_db.commit()
    backoffice_db.add(
        db.ProjectMembership(
            user_id=viewer.id, project_id=project.id, role=db.ProjectRole.viewer
        )
    )
    await backoffice_db.commit()
    login, session = _signed_in(viewer)
    with login, session:
        saved = await client.post(f"/projects/test-connection/{project.id}")
        edited = await client.post(f"/projects/test-connection/{project.id}", json=form)
        new = await client.post("/projects/test-connection/new", json=reachable)
    assert saved.status_code == 200
    assert edited.status_code == 403
    assert new.status_code == 403


@pytest.mark.asyncio
async def test_creating_a_project_answers_without_the_password(client, backoffice_db):
    admin = db.User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@test", name="a")
    backoffice_db.add(admin)
    await backoffice_db.commit()
    body = {"name": "New", "db_host": "h", "db_user": "u", "db_password": "pw-123"}

    login, session = _signed_in(admin)
    with login, session:
        refused = await client.post("/projects/create", json=body)
    login, session = _signed_in(admin, is_admin=True)
    with login, session:
        created = await client.post("/projects/create", json=body)

    assert refused.status_code == 403
    assert created.status_code == 200
    assert "pw-123" not in created.text
    assert created.json()["has_password"] is True
    assert created.json()["role"] == "owner"


async def _password_user(client, backoffice_db, **fields):
    """A user with a password, signed in through ``/auth/login``."""
    from kavalai.backoffice.users import hash_password

    user = db.User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@test",
        name="Ada",
        password_hash=hash_password("correct horse"),
        **fields,
    )
    backoffice_db.add(user)
    await backoffice_db.commit()
    response = await client.post(
        "/auth/login", json={"email": user.email, "password": "correct horse"}
    )
    assert response.status_code == 200
    return user


@pytest.mark.asyncio
async def test_the_profile_endpoints_need_a_signed_in_user(client, backoffice_db):
    assert (await client.get("/user/me")).status_code == 401
    assert (await client.patch("/user/me", json={"name": "X"})).status_code == 401
    response = await client.post(
        "/user/me/password", json={"new_password": "battery staple"}
    )
    assert response.status_code == 401

    # A session whose user has since been deleted is refused too.
    gone = db.User(id=uuid.uuid4(), email="gone@test", name="Gone")
    _, session = _signed_in(gone)
    with session:
        assert (await client.get("/user/me")).status_code == 401


@pytest.mark.asyncio
async def test_a_user_reads_and_edits_their_own_profile(client, backoffice_db):
    user = await _password_user(client, backoffice_db, is_admin=False)

    me = await client.get("/user/me")
    assert me.status_code == 200
    assert me.json()["id"] == str(user.id)
    assert me.json()["has_password"] is True
    assert "password_hash" not in me.json()

    saved = await client.patch(
        "/user/me", json={"name": "  Ada Lovelace ", "picture": "https://x/a.png"}
    )
    assert saved.status_code == 200
    assert saved.json()["name"] == "Ada Lovelace"
    assert saved.json()["picture"] == "https://x/a.png"
    assert "password_hash" not in saved.json()
    # The session the shell reads carries the new name.
    details = (await client.get("/user/get_details")).json()
    assert details["name"] == "Ada Lovelace"
    assert details["picture"] == "https://x/a.png"

    cleared = await client.patch("/user/me", json={"picture": "  "})
    assert cleared.json()["picture"] is None
    assert cleared.json()["name"] == "Ada Lovelace"
    assert (await client.patch("/user/me", json={})).json()["name"] == "Ada Lovelace"
    assert (await client.patch("/user/me", json={"picture": None})).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, detail",
    [
        ({"is_admin": True}, "not is_admin"),
        ({"name": "X", "email": "x@test", "id": "1"}, "not email, id"),
        ({"name": ""}, "The name is required."),
        ({"name": 3}, "The name is required."),
        ({"picture": 5}, "The picture must be a URL."),
    ],
)
async def test_a_profile_edit_changes_only_name_and_picture(
    client, backoffice_db, body, detail
):
    user = await _password_user(client, backoffice_db, is_admin=False)
    user_id, email = user.id, user.email

    response = await client.patch("/user/me", json=body)

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    backoffice_db.expire_all()
    row = await backoffice_db.get(db.User, user_id)
    assert (row.name, row.email, row.is_admin) == ("Ada", email, False)


@pytest.mark.asyncio
async def test_changing_a_password_asks_for_the_current_one(client, backoffice_db):
    from kavalai.backoffice.users import verify_password

    user_id = (await _password_user(client, backoffice_db)).id

    for current in ["wrong horse", None, 12]:
        body = {"new_password": "battery staple"}
        if current is not None:
            body["current_password"] = current
        refused = await client.post("/user/me/password", json=body)
        assert refused.status_code == 400
        assert refused.json() == {"detail": "The current password is not correct."}

    short = await client.post(
        "/user/me/password",
        json={"current_password": "correct horse", "new_password": "short"},
    )
    assert short.status_code == 400
    assert "at least 8" in short.json()["detail"]
    missing = await client.post(
        "/user/me/password", json={"current_password": "correct horse"}
    )
    assert missing.json() == {"detail": "The new password is required."}

    changed = await client.post(
        "/user/me/password",
        json={"current_password": "correct horse", "new_password": "battery staple"},
    )
    assert changed.status_code == 200
    assert changed.json()["has_password"] is True
    assert "password_hash" not in changed.json()

    backoffice_db.expire_all()
    row = await backoffice_db.get(db.User, user_id)
    assert verify_password("battery staple", row.password_hash)
    assert not verify_password("correct horse", row.password_hash)


@pytest.mark.asyncio
async def test_a_google_only_user_sets_a_first_password(client, backoffice_db):
    from kavalai.backoffice.users import verify_password

    user = db.User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@test", name="G")
    other = db.User(id=uuid.uuid4(), email=f"{uuid.uuid4()}@test", name="O")
    user_id, other_id = user.id, other.id
    backoffice_db.add_all([user, other])
    await backoffice_db.commit()

    login, session = _signed_in(user)
    with login, session:
        assert (await client.get("/user/me")).json()["has_password"] is False
        response = await client.post(
            "/user/me/password", json={"new_password": "battery staple"}
        )

    assert response.status_code == 200
    assert response.json()["has_password"] is True
    backoffice_db.expire_all()
    assert verify_password(
        "battery staple", (await backoffice_db.get(db.User, user_id)).password_hash
    )
    # Only the signed-in user's own record changed.
    assert (await backoffice_db.get(db.User, other_id)).password_hash is None


@pytest.mark.asyncio
async def test_a_taken_email_address_is_a_conflict_not_an_outage(client, backoffice_db):
    """The database refuses a second user with the same address. That is the
    caller's mistake, so it answers 409 and says so, rather than the 503
    "database is not connected" every other database error becomes."""
    first = db.User(email=f"{uuid.uuid4()}@test", name="First")
    second = db.User(email=f"{uuid.uuid4()}@test", name="Second")
    backoffice_db.add_all([first, second])
    await backoffice_db.commit()

    admin = {"user_info": {"id": str(uuid.uuid4()), "is_admin": True}}
    with (
        patch("kavalai.backoffice.server.assert_logged_in"),
        patch("starlette.requests.Request.session", admin),
    ):
        created = await client.post(
            "/users/create", json={"email": first.email, "name": "Copy"}
        )
        updated = await client.put(
            f"/users/update/{second.id}", json={"email": first.email}
        )

    for response in (created, updated):
        assert response.status_code == 409
        assert (
            response.json()["detail"]
            == "A user with this email address already exists."
        )
