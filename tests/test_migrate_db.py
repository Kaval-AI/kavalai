import asyncio
import os
import sqlite3
import sys
import time
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

import kavalai.migrate_db as migrate_db_module
from kavalai.db import ensure_async_scheme
from kavalai.migrate_db import (
    _connect,
    _connect_async,
    _driver_url,
    main,
    migrate,
    migrate_async,
)


@pytest.fixture(scope="module")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15-trixie") as postgres:
        yield postgres


@pytest.fixture
def db_uri(postgres_container):
    return (
        f"postgresql://{postgres_container.username}:{postgres_container.password}"
        f"@{postgres_container.get_container_host_ip()}"
        f":{postgres_container.get_exposed_port(5432)}/{postgres_container.dbname}"
    )


async def _on_connection(uri, fn):
    """Run ``fn(sync_connection)`` over the async driver the runner uses."""
    engine = create_async_engine(ensure_async_scheme(uri), poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            result = await connection.run_sync(fn)
            await connection.commit()
            return result
    finally:
        await engine.dispose()


def on_connection(uri, fn):
    return asyncio.run(_on_connection(uri, fn), loop_factory=asyncio.new_event_loop)


def table_names(uri, schema=None):
    return on_connection(uri, lambda c: set(inspect(c).get_table_names(schema=schema)))


def sqlite_tables(path):
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {name for (name,) in rows}


def test_driver_url_uses_the_async_driver_unless_one_is_named():
    url = _driver_url("postgresql://u:secret@h:5/db")
    assert url.drivername == "postgresql+asyncpg"
    # The password must survive (str(URL) would mask it).
    assert url.password == "secret"
    assert _driver_url("sqlite:///x.db").drivername == "sqlite+aiosqlite"
    assert _driver_url("postgresql+psycopg2://u@h/db").drivername == (
        "postgresql+psycopg2"
    )
    assert _driver_url("sqlite+pysqlite:///x.db").drivername == "sqlite+pysqlite"


def test_migrate_agents(db_uri):
    schema = "agents_schema"
    migrate("agents", uri=db_uri, schema=schema)

    tables = table_names(db_uri, schema)
    assert {
        "agents",
        "sessions",
        "runs",
        "tasks",
        "chat_messages",
        "model_call_stats",
        "alembic_version",
    } <= tables

    # rag_index was extracted in revision 0002: RAG storage is backend-owned
    # (see kavalai/rag/postgres.py), no rag tables in the migration set.
    assert "rag_index" not in tables


def test_migrate_backoffice(db_uri):
    schema = "backoffice_schema"
    migrate("backoffice", uri=db_uri, schema=schema)

    assert {
        "users",
        "projects",
        "project_memberships",
        "project_cache",
        "alembic_version",
    } <= table_names(db_uri, schema)


def test_migrate_idempotency(db_uri):
    schema = "idempotency_schema"
    migrate("agents", uri=db_uri, schema=schema)
    # Second run must be a no-op, not an error.
    migrate("agents", uri=db_uri, schema=schema)

    assert "agents" in table_names(db_uri, schema)


def test_migrate_unknown_set(db_uri):
    with pytest.raises(ValueError, match="Unknown migration set"):
        migrate("nope", uri=db_uri, schema="x")


def test_migrate_needs_exactly_one_target(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        migrate("agents")
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    with engine.connect() as connection:
        with pytest.raises(ValueError, match="exactly one"):
            migrate("agents", uri="sqlite:///y.db", connection=connection)
    engine.dispose()


def test_migrate_sqlite_file(tmp_path):
    db_path = tmp_path / "agents.db"
    uri = f"sqlite:///{db_path}"
    migrate("agents", uri=uri)
    migrate("agents", uri=uri)  # idempotent

    tables = sqlite_tables(db_path)
    assert {"agents", "alembic_version"} <= tables
    # rag_index was extracted in revision 0002 (RAG storage is backend-owned);
    # Postgres-only DDL in 0001 must also have been skipped on SQLite.
    assert "rag_index" not in tables


def test_a_named_sync_driver_is_used_as_named(tmp_path):
    """``sqlite+pysqlite://`` stands in for ``postgresql+psycopg2://``."""
    db_path = tmp_path / "agents.db"
    migrate("agents", uri=f"sqlite+pysqlite:///{db_path}")
    assert "agents" in sqlite_tables(db_path)


async def test_migrate_async_with_a_named_sync_driver(tmp_path):
    db_path = tmp_path / "agents.db"
    await migrate_async("agents", uri=f"sqlite+pysqlite:///{db_path}")
    assert "agents" in sqlite_tables(db_path)


async def test_migrate_async_runs_inside_an_event_loop(tmp_path):
    db_path = tmp_path / "agents.db"
    await migrate_async("agents", uri=f"sqlite:///{db_path}")
    await migrate_async("agents", uri=f"sqlite:///{db_path}")  # idempotent
    assert "agents" in sqlite_tables(db_path)


async def test_migrate_blocks_inside_a_running_event_loop(tmp_path):
    """A synchronous caller inside async code — a fixture, a startup hook —
    still migrates: the upgrade gets a loop of its own on a worker thread."""
    db_path = tmp_path / "agents.db"
    migrate("agents", uri=f"sqlite:///{db_path}")
    assert "agents" in sqlite_tables(db_path)


def test_sqlite_is_not_waited_for(tmp_path):
    """A SQLite file has no server to start; a bad path fails at once."""
    uri = f"sqlite:///{tmp_path / 'missing' / 'agents.db'}"
    started = time.monotonic()
    with pytest.raises(OperationalError):
        migrate("agents", uri=uri, max_wait=60.0)
    assert time.monotonic() - started < 5


def test_a_missing_async_driver_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    with pytest.raises(ImportError, match=r'pip install "kavalai\[runtime\]"'):
        migrate("agents", uri="postgresql://u:p@127.0.0.1:1/db")


def test_migrate_on_a_connection_commits_its_own_transaction(tmp_path):
    db_path = tmp_path / "agents.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as connection:
        migrate("agents", connection=connection)
        assert not connection.in_transaction()
    engine.dispose()
    assert "agents" in sqlite_tables(db_path)


def test_migrate_on_a_connection_joins_the_callers_transaction(tmp_path):
    db_path = tmp_path / "agents.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        migrate("agents", connection=connection)
        # Still the caller's to commit.
        assert connection.in_transaction()
    engine.dispose()
    assert "agents" in sqlite_tables(db_path)


def test_migrate_refuses_an_async_connection():
    engine = create_async_engine("sqlite+aiosqlite://")
    with pytest.raises(TypeError, match="migrate_async"):
        migrate("agents", connection=engine.connect())


async def test_migrate_async_refuses_a_sync_connection(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    with engine.connect() as connection:
        with pytest.raises(TypeError, match="migrate\\(\\)"):
            await migrate_async("agents", connection=connection)
    engine.dispose()


async def test_migrate_async_on_a_connection_is_rolled_back_with_it(db_uri):
    """The set runs in the caller's transaction: its rollback undoes it all."""
    schema = "rolled_back_schema"
    engine = create_async_engine(ensure_async_scheme(db_uri), poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            await migrate_async("agents", connection=connection, schema=schema)
            assert await connection.run_sync(
                lambda c: "agents" in inspect(c).get_table_names(schema=schema)
            )
            await connection.rollback()

        async with engine.connect() as connection:
            await migrate_async("agents", connection=connection, schema="kept")
    finally:
        await engine.dispose()

    schemas = await _on_connection(db_uri, lambda c: inspect(c).get_schema_names())
    assert schema not in schemas
    kept = await _on_connection(
        db_uri, lambda c: set(inspect(c).get_table_names(schema="kept"))
    )
    assert "agents" in kept


def test_migrate_main_with_env_vars(db_uri):
    schema = "main_env_schema"
    env = {
        "KAVALAI_DB_URI": db_uri,
        "KAVALAI_DB_SCHEMA": schema,
    }
    with patch.dict(os.environ, env), patch("sys.argv", ["migrate_db.py", "agents"]):
        main()

    assert "agents" in table_names(db_uri, schema)


def test_migrate_main_backoffice_env_vars(db_uri):
    schema = "main_bo_env_schema"
    env = {
        "KAVALAI_BO_DB_URI": db_uri,
        "KAVALAI_BO_DB_SCHEMA": schema,
    }
    with (
        patch.dict(os.environ, env),
        patch("sys.argv", ["migrate_db.py", "backoffice"]),
    ):
        main()

    assert "users" in table_names(db_uri, schema)


def test_migrate_skip_create_schema(db_uri):
    schema = "skip_create_schema_schema"
    on_connection(db_uri, lambda c: c.execute(text(f'CREATE SCHEMA "{schema}"')))

    migrate("agents", uri=db_uri, schema=schema, skip_create_schema=True)

    assert "agents" in table_names(db_uri, schema)


def test_migrate_skip_create_schema_fails_if_schema_missing(db_uri):
    with pytest.raises(DBAPIError, match="missing_schema"):
        migrate(
            "agents",
            uri=db_uri,
            schema="missing_schema",
            skip_create_schema=True,
        )


# The guard that replaces "keep the SQL files in sync by hand": applying the
# revisions to an empty database and diffing it against the ORM metadata must
# produce no changes.


def _parity_diffs(db_uri, schema, target_metadata, include_object=None):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    def _include(obj, name, type_, reflected, compare_to):
        if type_ == "table" and name == "alembic_version":
            return False
        if include_object is not None:
            return include_object(obj, name, type_, reflected, compare_to)
        return True

    def diff(connection):
        # Make the migrated schema the default so the schema-less metadata
        # compares against it (SQLite has no schemas).
        if schema is not None:
            connection.execute(text(f'SET search_path TO "{schema}"'))
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "include_object": _include},
        )
        return compare_metadata(context, target_metadata)

    return on_connection(db_uri, diff)


def test_agents_migrations_match_models(db_uri):
    from kavalai.db import Base
    from kavalai.migrations.common import agents_include_object

    schema = "parity_agents"
    migrate("agents", uri=db_uri, schema=schema)
    diffs = _parity_diffs(
        db_uri, schema, Base.metadata, include_object=agents_include_object
    )
    assert diffs == [], f"models and migrations diverged: {diffs}"


def test_backoffice_migrations_match_models(db_uri):
    from kavalai.backoffice.db import Base

    schema = "parity_backoffice"
    migrate("backoffice", uri=db_uri, schema=schema)
    diffs = _parity_diffs(db_uri, schema, Base.metadata)
    assert diffs == [], f"models and migrations diverged: {diffs}"


_INDEXES_OF_0005 = {
    "ix_model_call_stats_agent_id_created_at",
    "ix_model_call_stats_session_id",
    "ix_model_call_stats_run_id",
    "ix_sessions_agent_id_external_id",
    "ix_sessions_agent_id_updated_at",
    "ix_sessions_updated_at",
    "ix_chat_messages_session_id_created_at",
    "ix_chat_messages_agent_id_created_at",
}
_REPLACED_BY_0005 = {
    "ix_model_call_stats_agent_id",
    "ix_sessions_agent_id",
    "ix_chat_messages_session_id",
    "ix_chat_messages_agent_id",
}


def _alembic(uri, schema, action, revision):
    """Move the agents set to ``revision``; the runner only goes to head."""
    from alembic import command
    from alembic.config import Config

    def run(connection):
        if schema:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        config = Config()
        config.set_main_option(
            "script_location", migrate_db_module.MIGRATION_SETS["agents"]
        )
        config.attributes["connection"] = connection
        config.attributes["schema"] = schema
        getattr(command, action)(config, revision)

    on_connection(uri, run)


def _translated(connection, schema):
    if schema is None:
        return connection
    return connection.execution_options(schema_translate_map={None: schema})


def _seed_sessions(connection, schema, times):
    """Two sessions written before 0005: one with three runs, one with none."""
    from uuid import uuid4

    from kavalai.db import Agent, Run, Session

    connection = _translated(connection, schema)
    agent_id, active, idle = uuid4(), uuid4(), uuid4()
    connection.execute(Agent.__table__.insert().values(id=agent_id, name="upgrade"))
    for session_id in (active, idle):
        connection.execute(
            Session.__table__.insert().values(
                id=session_id,
                agent_id=agent_id,
                created_at=times["created"],
                updated_at=times["created"],
            )
        )
    # Not in time order: the backfill must take the latest, not the last row.
    for key in ("first", "last", "middle"):
        connection.execute(
            Run.__table__.insert().values(
                id=uuid4(), session_id=active, created_at=times[key]
            )
        )
    return active, idle


def _read_back(connection, schema):
    from sqlalchemy import select

    from kavalai.db import Session

    inspector = inspect(connection)
    columns = {
        column["name"]
        for column in inspector.get_columns("model_call_stats", schema=schema)
    }
    indexes = {
        index["name"]
        for table in ("sessions", "chat_messages", "model_call_stats")
        for index in inspector.get_indexes(table, schema=schema)
    }
    rows = _translated(connection, schema).execute(
        select(Session.id, Session.updated_at)
    )
    return columns, indexes, dict(rows.all())


async def _create_index_concurrently(uri, statement):
    engine = create_async_engine(
        ensure_async_scheme(uri), poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


def _utc(moment):
    from datetime import timezone

    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_revision_0005_backfills_last_activity_and_adds_the_indexes(
    backend, request, tmp_path
):
    from datetime import datetime, timezone

    from kavalai.db import Base
    from kavalai.migrations.common import agents_include_object

    if backend == "sqlite":
        uri, schema = f"sqlite:///{tmp_path / 'agents.db'}", None
    else:
        uri, schema = request.getfixturevalue("db_uri"), "upgrade_0005"
    times = {
        key: datetime(2026, 1, day, 12, tzinfo=timezone.utc)
        for key, day in (("created", 1), ("first", 2), ("middle", 3), ("last", 5))
    }

    _alembic(uri, schema, "upgrade", "0004")
    active, idle = on_connection(uri, lambda c: _seed_sessions(c, schema, times))
    if backend == "postgres":
        # What docs/deploy prescribes for a large database: build the index
        # without blocking writes first; the revision then leaves it alone.
        asyncio.run(
            _create_index_concurrently(
                uri,
                "CREATE INDEX CONCURRENTLY ix_sessions_updated_at "
                f'ON "{schema}".sessions (updated_at)',
            ),
            loop_factory=asyncio.new_event_loop,
        )

    _alembic(uri, schema, "upgrade", "head")

    columns, indexes, updated = on_connection(uri, lambda c: _read_back(c, schema))
    assert {"session_id", "run_id"} <= columns
    assert _INDEXES_OF_0005 <= indexes
    assert not _REPLACED_BY_0005 & indexes
    assert _utc(updated[active]) == times["last"]
    # A session without runs keeps what it had.
    assert _utc(updated[idle]) == times["created"]
    diffs = _parity_diffs(
        uri, schema, Base.metadata, include_object=agents_include_object
    )
    assert diffs == [], f"models and migrations diverged: {diffs}"

    _alembic(uri, schema, "downgrade", "0004")
    columns, indexes, _ = on_connection(uri, lambda c: _read_back(c, schema))
    assert not {"session_id", "run_id"} & columns
    assert _REPLACED_BY_0005 <= indexes
    assert not _INDEXES_OF_0005 & indexes


def test_backoffice_migrations_apply_to_sqlite(tmp_path):
    """The backoffice set runs on SQLite too, and ends at the ORM models.

    Revision 0002 alters a foreign key, which SQLite can only do by rebuilding
    the table, so this is the guard that keeps every revision in batch mode.
    """
    from kavalai.backoffice.db import Base

    db_path = tmp_path / "backoffice.db"
    uri = f"sqlite:///{db_path}"
    migrate("backoffice", uri=uri)
    migrate("backoffice", uri=uri)  # idempotent

    diffs = _parity_diffs(uri, None, Base.metadata)
    assert diffs == [], f"models and migrations diverged: {diffs}"

    with sqlite3.connect(db_path) as connection:
        (ddl,) = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'users'"
        ).fetchone()
    assert "ON DELETE SET NULL" in ddl


class _FlakyEngine:
    """An engine whose first ``failures`` connections fail with ``error``."""

    def __init__(self, failures, error=None):
        self.failures = failures
        self.error = error or OperationalError(
            "connect", {}, Exception("connection refused")
        )
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return "connection"


class _AsyncFlakyEngine(_FlakyEngine):
    async def connect(self):
        return super().connect()


class _StartingUp(Exception):
    """asyncpg's ``CannotConnectNowError`` as the runner sees it."""

    sqlstate = "57P03"


def test_connect_backs_off_until_the_database_is_up():
    slept = []
    engine = _FlakyEngine(failures=2)

    assert _connect(engine, max_wait=30.0, sleep=slept.append) == "connection"
    assert engine.attempts == 3
    # Backoff doubles between attempts.
    assert slept == [1.0, 2.0]


def test_connect_gives_up_after_max_wait():
    engine = _FlakyEngine(failures=99)
    with pytest.raises(OperationalError):
        _connect(engine, max_wait=0.0, sleep=lambda _: None)
    assert engine.attempts == 1


async def test_connect_async_retries_refused_and_starting_servers():
    slept = []

    async def sleep(pause):
        slept.append(pause)

    for error in (ConnectionRefusedError(111, "refused"), _StartingUp()):
        engine = _AsyncFlakyEngine(failures=1, error=error)
        assert await _connect_async(engine, 30.0, sleep=sleep) == "connection"
        assert engine.attempts == 2
    assert slept == [1.0, 1.0]


async def test_connect_async_does_not_retry_a_permanent_error():
    engine = _AsyncFlakyEngine(failures=99, error=ValueError("wrong password"))
    with pytest.raises(ValueError):
        await _connect_async(engine, 30.0, sleep=pytest.fail)
    assert engine.attempts == 1


def test_backoff_is_capped():
    pauses = migrate_db_module._pauses(max_wait=3600.0)
    assert [next(pauses) for _ in range(6)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


def test_sqlite_wait_is_zero():
    assert migrate_db_module._wait_for(make_url("sqlite:///x"), 60.0) == 0.0
    assert migrate_db_module._wait_for(make_url("postgresql://h/d"), 60.0) == 60.0
