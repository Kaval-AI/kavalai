"""How a project's connection fields become an agent database.

``db_type`` decides the reading: ``postgresql`` builds a URI from host, port,
user, password and database and applies ``db_schema``; ``sqlite`` takes
``db_name`` as a file path and has no schema. ``rag_schema`` is where the RAG
collections are when they are not beside the runtime tables, and
``read_only`` decides whether the connections the backoffice opens may write.
"""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from kavalai.backoffice import db
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from kavalai.backoffice.project_service import (
    PROJECT_FIELDS,
    ProjectService,
    describe_project_database,
    project_fields,
    project_db_uri,
    project_rag_schema,
    project_read_only,
    project_schema,
    project_sessionmaker,
)
from kavalai.db import db_manager


def _postgres_project() -> db.Project:
    return db.Project(
        id=uuid4(),
        name="pg",
        db_type="postgresql",
        db_host="h",
        db_port=5433,
        db_user="u",
        db_password="secret",
        db_name="d",
        db_schema="agents",
    )


def _sqlite_project(path: str, **fields) -> db.Project:
    return db.Project(
        id=uuid4(), name="local", db_type="sqlite", db_name=path, **fields
    )


def test_postgres_project_fields_build_a_uri():
    project = _postgres_project()

    assert project_db_uri(project) == "postgresql+asyncpg://u:secret@h:5433/d"
    assert project_schema(project) == "agents"
    assert "secret" not in describe_project_database(project)
    assert "host=h" in describe_project_database(project)
    assert "rag_schema=agents, read-only" in describe_project_database(project)


def test_rag_schema_defaults_to_the_agent_schema():
    """kavalapp keeps its collections in ``rag_<mode>`` beside
    ``agents_<mode>``; a project that says nothing keeps the one schema."""
    project = _postgres_project()
    assert project_rag_schema(project) == "agents"

    project.rag_schema = "rag"
    assert project_rag_schema(project) == "rag"
    assert "rag_schema=rag" in describe_project_database(project)


def test_a_sqlite_project_has_no_rag_schema():
    project = _sqlite_project("/tmp/agents.db")
    project.rag_schema = "ignored"

    assert project_rag_schema(project) is None


@pytest.mark.parametrize(
    "value, expected", [(None, True), (True, True), (False, False)]
)
def test_read_only_unless_the_project_says_otherwise(value, expected):
    """``None`` is what ``Project(**data)`` from the connection-test form
    carries before any column default applies, and it must read as
    read-only: the safe reading of "unsaid"."""
    project = _postgres_project()
    project.read_only = value

    assert project_read_only(project) is expected
    access = "read-only" if expected else "read-write"
    assert describe_project_database(project).endswith(access)


def test_a_project_without_a_type_is_postgres():
    """Rows from before ``db_type`` existed, and ``Project(**data)`` from the
    connection-test endpoint, carry no type and keep their old meaning."""
    project = _postgres_project()
    project.db_type = None

    assert project_db_uri(project).startswith("postgresql+asyncpg://")
    assert project_schema(project) == "agents"


def test_sqlite_project_name_is_the_file(tmp_path):
    project = _sqlite_project(str(tmp_path / "agents.db"))

    assert project_db_uri(project) == f"sqlite:///{tmp_path / 'agents.db'}"
    assert project_schema(project) is None
    assert (
        describe_project_database(project)
        == f"sqlite file={tmp_path / 'agents.db'}, read-only"
    )


@pytest.mark.asyncio
async def test_sqlite_project_sessionmaker_reads_the_file(tmp_path):
    """The project's sessionmaker is the read-only SQLite engine for that
    file, so what the agent server wrote there is what the backoffice reads,
    and nothing else; a project switched to read-write shares the engine the
    agent server itself uses."""
    path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=path)

    assert (
        project_sessionmaker(_sqlite_project(path)).kw["bind"]
        is db_manager.get_sqlite_sessionmaker(db_path=path, read_only=True).kw["bind"]
    )
    assert (
        project_sessionmaker(_sqlite_project(path, read_only=False)).kw["bind"]
        is db_manager.get_sqlite_sessionmaker(db_path=path).kw["bind"]
    )


@pytest.mark.asyncio
async def test_a_read_only_project_cannot_write_its_database(tmp_path):
    """The refusal comes from the database, not from the backoffice's code."""
    path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=path)

    async with project_sessionmaker(_sqlite_project(path))() as session:
        assert (
            await session.execute(text("SELECT count(*) FROM agents"))
        ).scalar() == 0
        with pytest.raises(OperationalError, match="readonly database"):
            await session.execute(
                text("INSERT INTO agents (id, name) VALUES ('a1', 'bot')")
            )


@pytest.mark.asyncio
async def test_connection_test_on_a_sqlite_project(tmp_path):
    path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=path)
    service = ProjectService(lambda: None)

    result = await service.test_connection(_sqlite_project(path))

    assert result == {"status": "success", "message": "Connection successful"}


@pytest.mark.asyncio
async def test_connection_test_reports_an_unreadable_sqlite_file(tmp_path):
    """A directory where a file was expected is a 400 with the cause, not a
    crash: the form shows the message next to the button."""
    service = ProjectService(lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await service.test_connection(_sqlite_project(str(tmp_path)))

    assert excinfo.value.status_code == 400
    assert "Failed to connect" in excinfo.value.detail


async def _user(session) -> "UUID":
    """A user row to own the project: memberships carry a foreign key."""
    user = db.User(email=f"{uuid4()}@test", name="owner")
    session.add(user)
    await session.commit()
    return user.id


def test_project_fields_are_the_editable_columns_and_nothing_else():
    """Every editable column, so a field added to the model without being
    listed here cannot be saved from the form and is noticed."""
    editable = {
        column.name
        for column in db.Project.__table__.columns
        if column.name not in {"id", "created_at", "updated_at"}
    }
    assert PROJECT_FIELDS == editable
    assert project_fields(
        {"name": "p", "id": "x", "created_at": "2026-01-01T00:00:00", "role": "owner"}
    ) == {"name": "p"}


@pytest.mark.asyncio
async def test_updating_with_the_listed_project_leaves_the_timestamps_alone(
    backoffice_db,
):
    """The projects page sends back what it listed: id, timestamps as ISO
    strings and the caller's role. Only the editable fields are written."""
    service = ProjectService(db.AsyncBackofficeSession)
    owner = await service.create_project(
        {"name": "p", "db_schema": "agents"}, await _user(backoffice_db)
    )
    created_at = owner.created_at

    listed = {
        "id": str(owner.id),
        "name": "p",
        "db_schema": "agents",
        "rag_schema": "rag",
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "role": "owner",
    }
    updated = await service.update_project(owner.id, listed)

    assert updated.rag_schema == "rag"
    assert updated.id == owner.id
    assert updated.created_at == created_at


@pytest.mark.asyncio
async def test_creating_ignores_what_the_server_owns(backoffice_db):
    service = ProjectService(db.AsyncBackofficeSession)

    project = await service.create_project(
        {"name": "p", "id": "not-a-uuid", "created_at": "yesterday", "role": "owner"},
        await _user(backoffice_db),
    )

    assert project.name == "p"
    assert project.read_only is True
    assert project.created_at.year >= 2026
