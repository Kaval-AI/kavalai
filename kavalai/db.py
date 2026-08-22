"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    MetaData,
    TEXT,
    ForeignKey,
    DateTime,
    Integer,
    Numeric,
    TypeDecorator,
    Uuid,
    create_engine,
    event,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from kavalai import idb


def uuid_column():
    """UUID column type.

    Uses PostgreSQL's native ``uuid`` on Postgres and SQLAlchemy's generic
    :class:`~sqlalchemy.Uuid` on SQLite, so the same ORM models work against
    both the production Postgres backend and the SQLite/IndexedDB backend used
    under Pyodide.
    """
    return PG_UUID(as_uuid=True).with_variant(Uuid(as_uuid=True), "sqlite")


def json_column():
    """JSON column type: ``JSONB`` on Postgres, generic ``JSON`` on SQLite."""
    return JSONB().with_variant(JSON(), "sqlite")


def parse_db_uri(uri: str) -> dict:
    """Parses a database URI into tokens."""
    url = make_url(uri)
    return {
        "user": url.username,
        "password": url.password,
        "host": url.host,
        "port": url.port,
        "db_name": url.database,
    }


class VectorType(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        # Outside PostgreSQL (e.g. the SQLite/IndexedDB backend used under
        # Pyodide) the ``pgvector`` type is unavailable, so fall back to the
        # TEXT representation produced by ``process_bind_param``.
        if dialect.name != "postgresql":
            return dialect.type_descriptor(TEXT())

        from sqlalchemy.types import UserDefinedType

        class Vector(UserDefinedType):
            def __init__(self, dim=None):
                self.dim = dim

            def get_col_spec(self, **kw):
                if self.dim:
                    return f"public.vector({self.dim})"
                return "public.vector"

        return dialect.type_descriptor(Vector(self.dim))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # value is already a list or string depending on driver
        if isinstance(value, str):
            return [float(x) for x in value.strip("[]").split(",")]
        return value


def ensure_async_scheme(uri: str) -> str:
    """Ensures the URI uses the postgresql+asyncpg driver."""
    if uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        if scheme.startswith("postgresql"):
            return f"postgresql+asyncpg://{rest}"
    return uri


def build_db_uri(
    user, password, host, port, db_name, scheme="postgresql+asyncpg"
) -> str:
    """Builds a database URI from components."""
    return f"{scheme}://{user}:{password}@{host}:{port}/{db_name}"


# Version stamp for SQLite databases bootstrapped via ``create_all`` (written
# to ``PRAGMA user_version``). Bump on any ORM schema change: SQLite stores
# created with a different version are dropped and recreated on init.
# 2: rag_index left the shared metadata (RAG backends self-provision).
SQLITE_SCHEMA_VERSION = 4


def _drop_all_sqlite_tables(connection):
    """Drop every table in a SQLite database (sync connection).

    Used when ``PRAGMA user_version`` doesn't match ``SQLITE_SCHEMA_VERSION``:
    browser/IndexedDB stores have no migration path, a stale file is wiped.
    """
    rows = connection.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    try:
        for (name,) in rows:
            if not name.startswith("sqlite_"):
                connection.exec_driver_sql(f'DROP TABLE IF EXISTS "{name}"')
    finally:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")


class EngineOptionsConflictError(ValueError):
    """Raised when a cached engine is requested with conflicting options.

    Engines are cached per ``(url, schema)`` and their options (``pool_size``,
    ``max_overflow``, ``echo``) are fixed by whichever call creates them. A
    later call that explicitly asks for different options would otherwise
    silently receive an engine configured differently than requested — raising
    surfaces the disagreeing call sites so they can be aligned.
    """


# Engine options applied when a caller leaves them unspecified (``None``).
_ENGINE_OPTION_DEFAULTS = {"echo": False, "pool_size": 1, "max_overflow": 0}


class DatabaseManager:
    """Manages dynamic engine creation and session factories."""

    # Default location of the SQLite file when running under Pyodide. It lives
    # inside the IDBFS mount point so it is persisted to the browser's
    # IndexedDB (see :mod:`kavalai.idb`).
    SQLITE_PYODIDE_DB_PATH = f"{idb.MOUNT_DIR}/kavalai.db"

    def __init__(self):
        self._engines = {}
        # Effective creation options per Postgres engine cache key, used to
        # detect conflicting option requests on cache hits.
        self._engine_options = {}

    def get_sessionmaker(
        self,
        *,
        user=None,
        password=None,
        host=None,
        port=None,
        db_name=None,
        uri=None,
        schema=None,
        echo=None,
        pool_size=None,
        max_overflow=None,
    ):
        """Return an ``async_sessionmaker`` for the given database and schema.

        ``schema`` selects the schema the ORM tables live in. The models are
        defined schema-less; the schema is applied per-engine via SQLAlchemy's
        ``schema_translate_map``, so the same models can target any schema at
        runtime. ``schema=None`` leaves table names unqualified (Postgres then
        resolves them via the connection's ``search_path``, i.e. ``public``).

        ``echo``, ``pool_size`` and ``max_overflow`` are engine-level options,
        fixed by the call that first creates the engine for a given
        ``(url, schema)`` (unspecified options fall back to
        ``echo=False, pool_size=1, max_overflow=0``). A later call may repeat
        the effective options or leave them unspecified; explicitly requesting
        different ones raises :class:`EngineOptionsConflictError` instead of
        silently returning an engine configured otherwise.
        """
        if uri:
            url = ensure_async_scheme(uri)
        else:
            url = build_db_uri(user, password, host, port, db_name)

        # Cache per (url, schema): translate maps are engine-level options.
        key = (url, schema)
        requested = {"echo": echo, "pool_size": pool_size, "max_overflow": max_overflow}
        if key not in self._engines:
            effective = {
                name: default if requested[name] is None else requested[name]
                for name, default in _ENGINE_OPTION_DEFAULTS.items()
            }
            engine = create_async_engine(
                url,
                echo=effective["echo"],
                pool_size=effective["pool_size"],
                max_overflow=effective["max_overflow"],
            )
            if schema:
                engine = engine.execution_options(schema_translate_map={None: schema})
            self._engines[key] = engine
            self._engine_options[key] = effective
        else:
            created_with = self._engine_options[key]
            conflicts = {
                name: {"requested": value, "engine": created_with[name]}
                for name, value in requested.items()
                if value is not None and value != created_with[name]
            }
            if conflicts:
                safe_url = make_url(url).render_as_string(hide_password=True)
                raise EngineOptionsConflictError(
                    f"Engine for {safe_url} (schema={schema!r}) was created with "
                    f"{created_with} but this call requested conflicting options: "
                    f"{conflicts}. Engine options are fixed by the first caller — "
                    f"align the call sites (or omit the options to reuse the engine)."
                )
        engine = self._engines[key]
        return async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

    # -- SQLite / IndexedDB backend -------------------------------------------
    #
    # A pure-Python, pyodide-compatible backend. The ORM models are schema-less
    # and live directly in the SQLite file (no ATTACH aliasing). Under Pyodide
    # the file sits on IDBFS so it is persisted to the browser's IndexedDB.
    #
    # Two flavours are offered: an async engine (for normal CPython, where
    # ``greenlet`` is available) and a *sync* engine. ``greenlet`` has no pyodide
    # build, and SQLAlchemy's async engine depends on it, so in the browser the
    # sync engine is the one that works -- see :meth:`get_sqlite_sync_engine`.
    #
    # ``init_sqlite``/``init_sqlite_sync`` bootstrap the schema with
    # ``create_all`` (Alembic is never imported in the browser) and stamp
    # ``PRAGMA user_version`` with ``SQLITE_SCHEMA_VERSION``. Browser stores are
    # per-user and ephemeral: on a version mismatch the database is dropped and
    # recreated rather than migrated.

    def _resolve_sqlite_path(self, db_path):
        """Resolve the SQLite database file path."""
        if db_path is None:
            db_path = self.SQLITE_PYODIDE_DB_PATH if idb.is_pyodide() else ":memory:"
        return db_path

    @staticmethod
    def _fk_pragma_listener(dbapi_connection, _record):
        """Connect listener that enables SQLite foreign keys."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    def get_sqlite_engine(self, *, db_path=None, echo=False):
        """Return a cached **async** SQLite engine backed by ``db_path``.

        Requires ``greenlet`` (SQLAlchemy's async engine dependency), so this is
        for CPython. Under pyodide use :meth:`get_sqlite_sync_engine` instead.

        ``db_path`` defaults to the IndexedDB-backed file under Pyodide and to an
        in-memory database otherwise. SQLite foreign keys are enabled. A
        :class:`~sqlalchemy.pool.StaticPool` keeps a single shared connection so
        an in-memory database survives across sessions.
        """
        db_path = self._resolve_sqlite_path(db_path)
        key = f"sqlite-async::{db_path}"
        if key not in self._engines:
            engine = create_async_engine(
                f"sqlite+aiosqlite:///{db_path}", echo=echo, poolclass=StaticPool
            )
            event.listen(engine.sync_engine, "connect", self._fk_pragma_listener)
            self._engines[key] = engine
        return self._engines[key]

    def get_sqlite_sessionmaker(self, *, db_path=None, echo=False):
        """Return an ``async_sessionmaker`` bound to the async SQLite engine."""
        engine = self.get_sqlite_engine(db_path=db_path, echo=echo)
        return async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_sqlite(self, *, db_path=None, echo=False):
        """Prepare the async SQLite/IndexedDB backend (mount, create tables, persist).

        Stamps ``PRAGMA user_version``; a database written by a different
        schema version is dropped and recreated (browser stores are ephemeral,
        there is no in-place upgrade path).
        """
        await idb.mount()
        engine = self.get_sqlite_engine(db_path=db_path, echo=echo)
        async with engine.begin() as conn:
            version = (await conn.exec_driver_sql("PRAGMA user_version")).scalar()
            if version not in (0, SQLITE_SCHEMA_VERSION):
                await conn.run_sync(_drop_all_sqlite_tables)
            await conn.run_sync(Base.metadata.create_all)
            await conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")
        await idb.flush()

    def get_sqlite_sync_engine(self, *, db_path=None, echo=False):
        """Return a cached **sync** SQLite engine backed by ``db_path``.

        This is the pyodide-friendly variant: SQLAlchemy's synchronous engine
        does not need ``greenlet`` (which has no pyodide build). It behaves like
        :meth:`get_sqlite_engine` otherwise -- foreign keys enabled and a shared
        connection.
        """
        db_path = self._resolve_sqlite_path(db_path)
        key = f"sqlite-sync::{db_path}"
        if key not in self._engines:
            engine = create_engine(
                f"sqlite:///{db_path}", echo=echo, poolclass=StaticPool
            )
            event.listen(engine, "connect", self._fk_pragma_listener)
            self._engines[key] = engine
        return self._engines[key]

    def get_sqlite_sync_sessionmaker(self, *, db_path=None, echo=False):
        """Return a sync ``sessionmaker`` bound to the sync SQLite engine."""
        engine = self.get_sqlite_sync_engine(db_path=db_path, echo=echo)
        return sessionmaker(bind=engine, expire_on_commit=False)

    @staticmethod
    def _ensure_sqlite_schema_sync(engine):
        """Create the ORM tables on a sync SQLite engine and stamp the version.

        Drops a stale-version database first (browser stores are ephemeral,
        there is no in-place upgrade path). Safe to call repeatedly.
        """
        with engine.begin() as conn:
            version = conn.exec_driver_sql("PRAGMA user_version").scalar()
            if version not in (0, SQLITE_SCHEMA_VERSION):
                _drop_all_sqlite_tables(conn)
            Base.metadata.create_all(conn)
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")

    async def init_sqlite_sync(self, *, db_path=None, echo=False):
        """Prepare the sync SQLite/IndexedDB backend (mount, create tables, persist).

        Mounts IndexedDB (under Pyodide), creates all ORM tables and persists the
        resulting database. The table creation itself is synchronous; only the
        IndexedDB mount/flush are awaited. Safe to call repeatedly. Stamps
        ``PRAGMA user_version`` and drops a stale-version database (see
        :meth:`init_sqlite`).
        """
        await idb.mount()
        engine = self.get_sqlite_sync_engine(db_path=db_path, echo=echo)
        self._ensure_sqlite_schema_sync(engine)
        await idb.flush()

    def get_sqlite_compat_sessionmaker(self, *, db_path=":memory:", echo=False):
        """Return an async-interface sessionmaker over the **sync** SQLite engine.

        For platforms without ``greenlet`` (Pyodide/WASM), where neither
        SQLAlchemy's async engine nor ``aiosqlite`` can run: sessions produced
        here expose the awaitable surface :class:`~kavalai.agent_service.AgentService`
        uses (``execute``/``commit``/``refresh``/``flush``/``rollback``) but run
        each call synchronously, briefly blocking the event loop — fine for
        local SQLite queries.

        Bootstraps the schema on first use. ``db_path`` defaults to a private
        in-memory database; for a persistent Pyodide/IndexedDB store call
        :meth:`init_sqlite_sync` first and pass its (default) file path.
        """
        engine = self.get_sqlite_sync_engine(db_path=db_path, echo=echo)
        self._ensure_sqlite_schema_sync(engine)
        sync_maker = sessionmaker(bind=engine, expire_on_commit=False)
        return lambda: AsyncSessionShim(sync_maker())

    async def flush(self):
        """Persist pending SQLite changes to IndexedDB (no-op outside Pyodide)."""
        await idb.flush()


class AsyncSessionShim:
    """Async facade over a synchronous ORM :class:`~sqlalchemy.orm.Session`.

    Produced by :meth:`DatabaseManager.get_sqlite_compat_sessionmaker` for
    greenlet-less platforms (Pyodide/WASM). The awaitable methods execute
    synchronously on the calling thread.

    The surface has to match what :class:`~kavalai.agent_service.AgentService`
    and :mod:`kavalai.crud` actually call — a method missing here does not fail
    at import, it fails in the browser at the moment a user hits that code path
    (``crud.get_one`` did exactly that). ``tests/test_agent_service.py`` runs
    the service against this shim as well as the real async session so the two
    cannot drift apart again.
    """

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._session.rollback()
        self._session.close()

    def add(self, obj):
        self._session.add(obj)

    async def execute(self, statement):
        return self._session.execute(statement)

    async def flush(self):
        self._session.flush()

    async def commit(self):
        self._session.commit()

    async def rollback(self):
        self._session.rollback()

    async def refresh(self, obj):
        self._session.refresh(obj)

    async def get(self, entity, ident, **kwargs):
        return self._session.get(entity, ident, **kwargs)

    async def delete(self, obj):
        self._session.delete(obj)

    async def merge(self, obj, **kwargs):
        return self._session.merge(obj, **kwargs)

    async def scalar(self, statement, *args, **kwargs):
        return self._session.scalar(statement, *args, **kwargs)

    async def scalars(self, statement, *args, **kwargs):
        return self._session.scalars(statement, *args, **kwargs)

    async def close(self):
        self._session.close()


# Global instance to manage cached engines
db_manager = DatabaseManager()


class Base(DeclarativeBase):
    # Schema-less by design: the target schema is applied per-engine via
    # ``schema_translate_map`` (see ``DatabaseManager.get_sessionmaker``), so
    # the library never reads configuration from the environment.
    metadata = MetaData()


class Agent(Base):
    """A configured agent.

    One row holds an agent's definition: its name, optional description, the
    JSON input/output schemas it exposes, and the workflow document that drives
    its execution. Sessions, tasks and chat messages all reference the agent
    they belong to.
    """

    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)
    input_schema: Mapped[dict | None] = mapped_column(json_column())
    output_schema: Mapped[dict | None] = mapped_column(json_column())
    workflow: Mapped[dict | None] = mapped_column(
        json_column()
    )  # Updated to match SQL 'workflow'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list["Task"]] = relationship(back_populates="agent")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelCallStat(Base):
    """Token usage and timing for a single LLM or embedding call.

    Each row records one model call (``call_type`` distinguishes e.g. chat
    completion from embedding): the model used, the request/response payloads,
    token counts, batch size and duration.

    Cost is deliberately **not** recorded. Providers do not return it — only
    tokens — so any figure here would come from a price table the library would
    have to keep current, and cached input tokens are priced differently enough
    that a total derived from ``prompt_tokens`` alone would be wrong by a
    multiple. ``cached_prompt_tokens`` and ``reasoning_tokens`` are stored
    instead, which is what makes cost computable downstream (e.g. with
    ``genai-prices``) from a row like this one.
    """

    __tablename__ = "model_call_stats"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    call_type: Mapped[str] = mapped_column(TEXT, nullable=False, index=True)
    model: Mapped[str] = mapped_column(TEXT, nullable=False, index=True)
    agent_id: Mapped[UUID | None] = mapped_column(uuid_column(), index=True)
    request_data: Mapped[dict | None] = mapped_column(json_column())
    response_data: Mapped[dict | None] = mapped_column(json_column())
    response_code: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric(10, 6))
    # Subsets of prompt_tokens / completion_tokens respectively, when the
    # provider reports them: cached input is billed at a fraction of fresh
    # input, and reasoning tokens are output tokens the user never sees.
    cached_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Session(Base):
    """A user conversation/session with an agent.

    A session groups together everything exchanged with one agent over a
    conversation: its runs, tasks and chat messages. ``external_id`` lets a
    caller correlate the session with an identifier in their own system.
    """

    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str | None] = mapped_column(TEXT)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="sessions")
    runs: Mapped[list["Run"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class Run(Base):
    """One workflow run within a session.

    A run captures a single invocation of the agent's workflow: the input it
    was called with, the output it produced and the resolved run context. Each
    run belongs to a session and owns the tasks executed during it.
    """

    __tablename__ = "runs"  # Renamed from 'interactions'

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_data: Mapped[dict | None] = mapped_column(json_column())
    output_data: Mapped[dict | None] = mapped_column(json_column())
    context: Mapped[dict | None] = mapped_column(json_column())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    session: Mapped["Session"] = relationship(back_populates="runs")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="run")


class Task(Base):
    """One workflow-node execution within a run.

    A task records the execution of a single node in the workflow: its name and
    node type, the inputs it received, the output it produced, the prompt used
    (if any), any errors raised and how long it took. Tasks belong to a run (and
    its session) and provide the step-by-step trace of a run.
    """

    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), index=True
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    inputs: Mapped[dict | None] = mapped_column(json_column())
    output: Mapped[dict | None] = mapped_column(json_column())
    name: Mapped[str | None] = mapped_column(TEXT)
    node_type: Mapped[str | None] = mapped_column(TEXT)
    prompt: Mapped[str | None] = mapped_column(TEXT)
    errors: Mapped[list[str] | None] = mapped_column(json_column())
    duration_seconds: Mapped[float | None] = mapped_column(Numeric)
    # Trajectory columns. ``seq`` is the per-run execution order and is what
    # actually carries the structure: ordering by it reconstructs the executed
    # path exactly, including the interleaving under a ``parallel`` node.
    # ``parent_task_name`` names the node that produced this row (set on the
    # tool-call rows an agent node emits) and is the readable join key that
    # survives an export to a warehouse. ``tool_uri`` is set by both function
    # nodes and agent tool calls, so one predicate finds every call to a tool
    # regardless of whether a human wired it into the YAML or an agent chose it.
    seq: Mapped[int | None] = mapped_column(Integer)
    parent_task_name: Mapped[str | None] = mapped_column(TEXT, index=True)
    tool_uri: Mapped[str | None] = mapped_column(TEXT, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="tasks")
    session: Mapped["Session"] = relationship(back_populates="tasks")
    run: Mapped["Run"] = relationship(back_populates="tasks")


class ChatMessage(Base):
    """One message in a session's chat history.

    Each row is a single message (``role`` such as ``user`` or ``assistant``
    with its text ``content``) belonging to a session. It optionally references
    the run that produced it, forming the ordered conversation transcript.
    """

    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(uuid_column(), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), index=True
    )
    role: Mapped[str] = mapped_column(TEXT, nullable=False)
    content: Mapped[str] = mapped_column(TEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    agent: Mapped["Agent"] = relationship(back_populates="chat_messages")
    session: Mapped["Session"] = relationship(back_populates="chat_messages")
    run: Mapped["Run"] = relationship(back_populates="chat_messages")
