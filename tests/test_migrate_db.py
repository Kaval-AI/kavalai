import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, text
from testcontainers.postgres import PostgresContainer

from sqlalchemy.exc import OperationalError

import kavalai.migrate_db as migrate_db_module
from kavalai.migrate_db import (
    _connect_with_retry,
    ensure_sync_scheme,
    migrate,
    main,
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


def _inspector(db_uri):
    engine = create_engine(ensure_sync_scheme(db_uri))
    return engine, inspect(engine)


def test_ensure_sync_scheme():
    assert ensure_sync_scheme("postgresql+asyncpg://u:p@h:5/db").startswith(
        "postgresql+psycopg2://"
    )
    assert ensure_sync_scheme("sqlite+aiosqlite:///x.db") == "sqlite:///x.db"
    # Password must survive the round-trip (str(URL) would mask it).
    assert "p" in ensure_sync_scheme("postgresql://u:p@h:5/db")


def test_migrate_agents(db_uri):
    schema = "agents_schema"
    migrate("agents", uri=db_uri, schema=schema)

    engine, insp = _inspector(db_uri)
    tables = set(insp.get_table_names(schema=schema))
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
    engine.dispose()


def test_migrate_backoffice(db_uri):
    schema = "backoffice_schema"
    migrate("backoffice", uri=db_uri, schema=schema)

    engine, insp = _inspector(db_uri)
    tables = set(insp.get_table_names(schema=schema))
    assert {
        "users",
        "projects",
        "project_memberships",
        "project_cache",
        "alembic_version",
    } <= tables
    engine.dispose()


def test_migrate_idempotency(db_uri):
    schema = "idempotency_schema"
    migrate("agents", uri=db_uri, schema=schema)
    # Second run must be a no-op, not an error.
    migrate("agents", uri=db_uri, schema=schema)

    engine, insp = _inspector(db_uri)
    assert "agents" in insp.get_table_names(schema=schema)
    engine.dispose()


def test_migrate_unknown_set(db_uri):
    with pytest.raises(ValueError, match="Unknown migration set"):
        migrate("nope", uri=db_uri, schema="x")


def test_migrate_sqlite_file(tmp_path):
    db_path = tmp_path / "agents.db"
    uri = f"sqlite:///{db_path}"
    migrate("agents", uri=uri)
    migrate("agents", uri=uri)  # idempotent

    engine = create_engine(uri)
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert {"agents", "alembic_version"} <= tables
    # rag_index was extracted in revision 0002 (RAG storage is backend-owned);
    # Postgres-only DDL in 0001 must also have been skipped on SQLite.
    assert "rag_index" not in tables
    engine.dispose()


def test_migrate_main_with_env_vars(db_uri):
    schema = "main_env_schema"
    env = {
        "KAVALAI_DB_URI": db_uri,
        "KAVALAI_DB_SCHEMA": schema,
    }
    with patch.dict(os.environ, env), patch("sys.argv", ["migrate_db.py", "agents"]):
        main()

    engine, insp = _inspector(db_uri)
    assert "agents" in insp.get_table_names(schema=schema)
    engine.dispose()


def test_migrate_main_backoffice_env_vars(db_uri):
    schema = "main_bo_env_schema"
    env = {
        "KAVALAI_BO_DB_URI": db_uri,
        "KAVALAI_BO_DB_SCHEMA": schema,
    }
    with patch.dict(os.environ, env), patch(
        "sys.argv", ["migrate_db.py", "backoffice"]
    ):
        main()

    engine, insp = _inspector(db_uri)
    assert "users" in insp.get_table_names(schema=schema)
    engine.dispose()


def test_migrate_skip_create_schema(db_uri):
    schema = "skip_create_schema_schema"
    engine = create_engine(ensure_sync_scheme(db_uri))
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    migrate("agents", uri=db_uri, schema=schema, skip_create_schema=True)

    insp = inspect(engine)
    assert "agents" in insp.get_table_names(schema=schema)
    engine.dispose()


def test_migrate_skip_create_schema_fails_if_schema_missing(db_uri):
    from sqlalchemy.exc import ProgrammingError

    with pytest.raises(ProgrammingError):
        migrate(
            "agents",
            uri=db_uri,
            schema="missing_schema",
            skip_create_schema=True,
        )


# --- model <-> migration parity ---------------------------------------------
#
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

    engine = create_engine(ensure_sync_scheme(db_uri))
    try:
        with engine.connect() as connection:
            # Make the migrated schema the default so the schema-less
            # metadata compares against it.
            connection.execute(text(f'SET search_path TO "{schema}"'))
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    "include_object": _include,
                },
            )
            return compare_metadata(context, target_metadata)
    finally:
        engine.dispose()


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


class _FlakyEngine:
    """An engine whose first ``failures`` connections fail as if the DB is down."""

    def __init__(self, failures):
        self.failures = failures
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))
        return _FakeConnection()


class _FakeConnection:
    def execute(self, statement):
        return None


def test_connect_with_retry_backs_off_until_the_database_is_up(monkeypatch):
    slept = []
    monkeypatch.setattr(migrate_db_module.time, "sleep", slept.append)

    engine = _FlakyEngine(failures=2)
    connection = _connect_with_retry(engine, max_wait=30.0)

    assert isinstance(connection, _FakeConnection)
    assert engine.attempts == 3
    # Backoff doubles between attempts.
    assert slept == [1.0, 2.0]


def test_connect_with_retry_gives_up_after_max_wait(monkeypatch):
    monkeypatch.setattr(migrate_db_module.time, "sleep", lambda _: None)

    engine = _FlakyEngine(failures=99)
    with pytest.raises(OperationalError):
        _connect_with_retry(engine, max_wait=0.0)

    assert engine.attempts == 1
