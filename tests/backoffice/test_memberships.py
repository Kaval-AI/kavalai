import uuid
from unittest.mock import patch
import pytest
from httpx import AsyncClient, ASGITransport
from kavalai.backoffice import db
from kavalai.backoffice.server import app


@pytest.fixture
def mock_session_admin():
    user_id = str(uuid.uuid4())
    return {"user_info": {"id": user_id, "is_admin": True}}


@pytest.fixture
def mock_session_user():
    user_id = str(uuid.uuid4())
    return {"user_info": {"id": user_id, "is_admin": False}}


from pytest_asyncio import fixture as pytest_asyncio_fixture


@pytest_asyncio_fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_projects_get_members(client, backoffice_db):
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    # Setup project and user in DB
    user = db.User(id=user_id, email="test@example.com", name="Test User")
    project = db.Project(id=project_id, name="Test Project")
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project_id, role=db.ProjectRole.owner
    )

    backoffice_db.add_all([user, project, membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session", {"user_info": {"id": str(user_id)}}
    ):
        response = await client.get(f"/projects/{project_id}/members")
        assert response.status_code == 200
        members = response.json()
        assert len(members) == 1
        assert members[0]["email"] == "test@example.com"
        assert members[0]["role"] == "owner"


@pytest.mark.asyncio
async def test_projects_add_member(client, backoffice_db):
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Test Project")
    user = db.User(id=user_id, email="user@example.com", name="User")
    backoffice_db.add_all([project, user])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.post(
            f"/projects/{project_id}/members/add",
            json={"user_id": str(user_id), "role": "viewer"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "added"}

        # Verify in DB
        stmt = db.select(db.ProjectMembership).where(
            db.ProjectMembership.project_id == project_id,
            db.ProjectMembership.user_id == user_id,
        )
        result = await backoffice_db.execute(stmt)
        membership = result.scalars().first()
        assert membership is not None
        assert membership.role == db.ProjectRole.viewer


@pytest.mark.asyncio
async def test_projects_update_member(client, backoffice_db):
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Test Project")
    user = db.User(id=user_id, email="user@example.com", name="User")
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project_id, role=db.ProjectRole.viewer
    )
    backoffice_db.add_all([project, user, membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.put(
            f"/projects/{project_id}/members/update",
            json={"user_id": str(user_id), "role": "owner"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "updated"}

        # Verify in DB
        await backoffice_db.refresh(membership)
        assert membership.role == db.ProjectRole.owner


@pytest.mark.asyncio
async def test_projects_remove_member(client, backoffice_db):
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Test Project")
    user = db.User(id=user_id, email="user@example.com", name="User")
    # Add another owner so we can remove one
    other_owner_id = uuid.uuid4()
    other_owner = db.User(id=other_owner_id, email="other@example.com", name="Other")
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project_id, role=db.ProjectRole.owner
    )
    other_membership = db.ProjectMembership(
        user_id=other_owner_id, project_id=project_id, role=db.ProjectRole.owner
    )

    backoffice_db.add_all([project, user, other_owner, membership, other_membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.delete(
            f"/projects/{project_id}/members/remove/{user_id}"
        )
        assert response.status_code == 200
        assert response.json() == {"status": "removed"}

        # Verify in DB
        stmt = db.select(db.ProjectMembership).where(
            db.ProjectMembership.project_id == project_id,
            db.ProjectMembership.user_id == user_id,
        )
        result = await backoffice_db.execute(stmt)
        assert result.scalars().first() is None


@pytest.mark.asyncio
async def test_projects_remove_last_owner_fails(client, backoffice_db):
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Test Project")
    user = db.User(id=user_id, email="user@example.com", name="User")
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project_id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([project, user, membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.delete(
            f"/projects/{project_id}/members/remove/{user_id}"
        )
        assert response.status_code == 400
        assert "Cannot remove the last owner" in response.json()["detail"]


@pytest.mark.asyncio
async def test_projects_demote_last_owner_fails(client, backoffice_db):
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Test Project")
    user = db.User(id=user_id, email="user@example.com", name="User")
    membership = db.ProjectMembership(
        user_id=user_id, project_id=project_id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([project, user, membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.put(
            f"/projects/{project_id}/members/update",
            json={"user_id": str(user_id), "role": "viewer"},
        )
        assert response.status_code == 400
        assert "Cannot demote the last owner" in response.json()["detail"]


@pytest.mark.asyncio
async def test_remove_member_clears_their_active_project(client, backoffice_db):
    """A removed member must not keep the project selected, or every
    project-scoped request they make answers 403 with no way back."""
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    project_id = uuid.uuid4()
    other_owner_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Shared Project")
    user = db.User(
        id=user_id,
        email="removed@example.com",
        name="Removed",
        active_project_id=project_id,
    )
    other_owner = db.User(
        id=other_owner_id,
        email="owner@example.com",
        name="Owner",
        active_project_id=project_id,
    )
    memberships = [
        db.ProjectMembership(
            user_id=uid, project_id=project_id, role=db.ProjectRole.owner
        )
        for uid in (user_id, other_owner_id)
    ]

    backoffice_db.add_all([project, user, other_owner, *memberships])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(admin_id), "is_admin": True}},
    ):
        response = await client.delete(
            f"/projects/{project_id}/members/remove/{user_id}"
        )
        assert response.status_code == 200

    await backoffice_db.refresh(user)
    assert user.active_project_id is None
    # Members who kept their access keep their selection.
    await backoffice_db.refresh(other_owner)
    assert other_owner.active_project_id == project_id


@pytest.mark.asyncio
async def test_delete_project_clears_active_project(client, backoffice_db):
    """Deleting a project must not fail on users.active_project_id."""
    owner_id = uuid.uuid4()
    project_id = uuid.uuid4()

    project = db.Project(id=project_id, name="Doomed Project")
    owner = db.User(
        id=owner_id,
        email="doomed@example.com",
        name="Owner",
        active_project_id=project_id,
    )
    membership = db.ProjectMembership(
        user_id=owner_id, project_id=project_id, role=db.ProjectRole.owner
    )
    backoffice_db.add_all([project, owner, membership])
    await backoffice_db.commit()

    with patch("kavalai.backoffice.server.assert_logged_in"), patch(
        "starlette.requests.Request.session",
        {"user_info": {"id": str(owner_id), "is_admin": True}},
    ):
        response = await client.delete(f"/projects/delete/{project_id}")
        assert response.status_code == 200
        assert response.json() == {"status": "deleted"}

    await backoffice_db.refresh(owner)
    assert owner.active_project_id is None
