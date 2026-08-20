from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.backoffice.db import (
    User,
    Project,
    ProjectMembership,
    ProjectRole,
    ProjectCache,
    is_member,
    is_owner,
    get_user_projects,
    resolve_active_project_id,
)
from kavalai.crud import insert, update, delete, get_one, get_all


@pytest.mark.asyncio
async def test_crud_utilities(backoffice_db: AsyncSession):
    """Test the generic CRUD utility functions in db.py."""
    # Test Insert
    user_data = {"email": "utility@test.com", "name": "Utility Test"}
    user = await insert(backoffice_db, User, user_data)
    assert user.id is not None
    assert user.email == "utility@test.com"

    # Test Get One
    fetched_user = await get_one(backoffice_db, User, user.id)
    assert fetched_user.name == "Utility Test"

    # Test Update
    updated_user = await update(backoffice_db, User, user.id, {"name": "New Name"})
    assert updated_user.name == "New Name"

    # Test Get All
    all_users = await get_all(backoffice_db, User)
    assert len(all_users) >= 1

    # Test Delete
    success = await delete(backoffice_db, User, user.id)
    assert success is True
    deleted_user = await get_one(backoffice_db, User, user.id)
    assert deleted_user is None


@pytest.mark.asyncio
async def test_access_control_logic(backoffice_db: AsyncSession):
    """Test is_member and is_owner helper functions."""
    # Setup
    user = await insert(
        backoffice_db, User, {"email": "acl@test.com", "name": "ACL User"}
    )
    project = await insert(backoffice_db, Project, {"name": "ACL Project"})

    # Add membership as owner
    await insert(
        backoffice_db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.owner},
    )

    # Test helpers
    assert await is_member(backoffice_db, user.id, project.id) is True
    assert await is_owner(backoffice_db, user.id, project.id) is True

    # Test non-existent member
    assert await is_member(backoffice_db, uuid4(), project.id) is False


@pytest.mark.asyncio
async def test_get_user_projects_with_role(backoffice_db: AsyncSession):
    # Setup
    user = await insert(
        backoffice_db, User, {"email": "role@test.com", "name": "Role User"}
    )
    project = await insert(backoffice_db, Project, {"name": "Role Project"})
    await insert(
        backoffice_db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.viewer},
    )

    # Execute
    results = await get_user_projects(backoffice_db, user.id)

    # Assert
    assert len(results) == 1
    assert results[0]["name"] == "Role Project"
    assert results[0]["role"] == "viewer"


@pytest.mark.asyncio
async def test_cascade_delete(backoffice_db: AsyncSession):
    """Ensure deleting a project removes its memberships but not the users."""
    user = await insert(
        backoffice_db, User, {"email": "cascade@test.com", "name": "Cascade"}
    )
    project = await insert(backoffice_db, Project, {"name": "Delete Me"})

    await insert(
        backoffice_db,
        ProjectMembership,
        {"user_id": user.id, "project_id": project.id, "role": ProjectRole.owner},
    )

    # Delete project
    await delete(backoffice_db, Project, project.id)

    # Check that membership is gone (via cascade)
    membership = await get_one(backoffice_db, ProjectMembership, (user.id, project.id))
    assert membership is None

    # Check that user still exists
    still_here = await get_one(backoffice_db, User, user.id)
    assert still_here is not None


@pytest.mark.asyncio
async def test_project_cache_relationship(backoffice_db: AsyncSession):
    """Test the project_cache relationship and cascade delete."""
    project = await insert(backoffice_db, Project, {"name": "Cache Project"})

    # Add cache entries
    cache_entry1 = await insert(
        backoffice_db,
        ProjectCache,
        {"project_id": project.id, "name": "test_key", "value": "test_value"},
    )
    cache_entry2 = await insert(
        backoffice_db,
        ProjectCache,
        {"project_id": project.id, "name": "another_key", "value": None},
    )

    assert cache_entry1.id is not None
    assert cache_entry2.id is not None

    # Fetch project with cache
    fetched_project = await get_one(backoffice_db, Project, project.id)
    # Need to load relationship or it might be lazy
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    stmt = (
        select(Project)
        .where(Project.id == project.id)
        .options(selectinload(Project.cache))
    )
    result = await backoffice_db.execute(stmt)
    fetched_project = result.scalar_one()

    assert len(fetched_project.cache) == 2
    names = [c.name for c in fetched_project.cache]
    assert "test_key" in names
    assert "another_key" in names

    # Test cascade delete
    await delete(backoffice_db, Project, project.id)
    deleted_cache = await get_one(backoffice_db, ProjectCache, cache_entry1.id)
    assert deleted_cache is None


async def _member_of(session: AsyncSession, user: User, project: Project) -> None:
    await insert(
        session,
        ProjectMembership,
        {
            "user_id": user.id,
            "project_id": project.id,
            "role": ProjectRole.owner,
        },
    )


@pytest.mark.asyncio
async def test_resolve_active_project_keeps_valid_selection(
    backoffice_db: AsyncSession,
):
    """A project the user is still a member of is left untouched."""
    user = await insert(backoffice_db, User, {"email": "keep@test.com", "name": "Keep"})
    project = await insert(backoffice_db, Project, {"name": "Kept"})
    await _member_of(backoffice_db, user, project)
    await update(backoffice_db, User, user.id, {"active_project_id": project.id})

    assert await resolve_active_project_id(backoffice_db, user.id) == project.id


@pytest.mark.asyncio
async def test_resolve_active_project_falls_back_when_membership_lost(
    backoffice_db: AsyncSession,
):
    """Losing access to the selected project falls back to another one."""
    user = await insert(backoffice_db, User, {"email": "lost@test.com", "name": "Lost"})
    revoked = await insert(backoffice_db, Project, {"name": "B revoked"})
    kept = await insert(backoffice_db, Project, {"name": "A kept"})
    await _member_of(backoffice_db, user, kept)
    await update(backoffice_db, User, user.id, {"active_project_id": revoked.id})

    resolved = await resolve_active_project_id(backoffice_db, user.id)

    assert resolved == kept.id
    # The repair is persisted, so the next request does not have to redo it.
    refreshed = await get_one(backoffice_db, User, user.id)
    assert refreshed.active_project_id == kept.id


@pytest.mark.asyncio
async def test_resolve_active_project_without_memberships(backoffice_db: AsyncSession):
    """A user with no projects at all resolves to no active project."""
    user = await insert(backoffice_db, User, {"email": "none@test.com", "name": "None"})
    orphan = await insert(backoffice_db, Project, {"name": "Not mine"})
    await update(backoffice_db, User, user.id, {"active_project_id": orphan.id})

    assert await resolve_active_project_id(backoffice_db, user.id) is None

    refreshed = await get_one(backoffice_db, User, user.id)
    assert refreshed.active_project_id is None


@pytest.mark.asyncio
async def test_resolve_active_project_unknown_user(backoffice_db: AsyncSession):
    """An id with no user row resolves to no active project rather than raising."""
    assert await resolve_active_project_id(backoffice_db, uuid4()) is None


@pytest.mark.asyncio
async def test_deleting_project_clears_active_selection(backoffice_db: AsyncSession):
    """The active_project_id foreign key must not block deleting a project."""
    user = await insert(backoffice_db, User, {"email": "del@test.com", "name": "Del"})
    project = await insert(backoffice_db, Project, {"name": "Doomed"})
    await _member_of(backoffice_db, user, project)
    await update(backoffice_db, User, user.id, {"active_project_id": project.id})

    assert await delete(backoffice_db, Project, project.id) is True

    refreshed = await get_one(backoffice_db, User, user.id)
    await backoffice_db.refresh(refreshed)
    assert refreshed.active_project_id is None
