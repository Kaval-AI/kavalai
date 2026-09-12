"""How a project's connection fields become an agent database.

``db_type`` decides the reading: ``postgresql`` builds a URI from host, port,
user, password and database and applies ``db_schema``; ``sqlite`` takes
``db_name`` as a file path and has no schema.
"""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from kavalai.backoffice import db
from kavalai.backoffice.project_service import (
    ProjectService,
    describe_project_database,
    project_db_uri,
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


def _sqlite_project(path: str) -> db.Project:
    return db.Project(id=uuid4(), name="local", db_type="sqlite", db_name=path)


def test_postgres_project_fields_build_a_uri():
    project = _postgres_project()

    assert project_db_uri(project) == "postgresql+asyncpg://u:secret@h:5433/d"
    assert project_schema(project) == "agents"
    assert "secret" not in describe_project_database(project)
    assert "host=h" in describe_project_database(project)


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
    assert describe_project_database(project) == f"sqlite file={tmp_path / 'agents.db'}"


@pytest.mark.asyncio
async def test_sqlite_project_sessionmaker_reads_the_file(tmp_path):
    """The project's sessionmaker is the shared SQLite engine for that file,
    so what the agent server wrote there is what the backoffice reads."""
    path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=path)
    project = _sqlite_project(path)

    assert (
        project_sessionmaker(project).kw["bind"]
        is db_manager.get_sqlite_sessionmaker(db_path=path).kw["bind"]
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
