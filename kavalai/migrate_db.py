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

Kaval.AI database migration runner (Alembic).

Upgrades one migration set (``agents`` or ``backoffice``) to head, with the
target schema applied through ``schema_translate_map`` (see
:mod:`kavalai.migrations.common`). A set can be run in two ways:

* **On a URI.** The runner opens the connection itself, waits for the database
  to accept it, creates the schema and commits. It uses the URI's async driver
  — ``postgresql://`` becomes asyncpg, ``sqlite://`` aiosqlite — the drivers
  the runtime already uses, so no synchronous Postgres driver is needed.
  Alembic itself is synchronous: it runs on the connection's sync facade
  through ``AsyncConnection.run_sync``, Alembic's documented recipe. A URI
  that names a synchronous driver (``postgresql+psycopg2://``) is run on that
  driver instead, so its own query parameters (``sslmode``) keep their
  meaning.
* **On a caller's connection.** The set runs inside the transaction the
  connection already has, and the caller commits it; with no transaction open,
  the runner begins one and commits it. This is how an application runs
  Kaval.AI's sets as a step of its own migrations.

:func:`migrate` is the blocking entry point; :func:`migrate_async` is the same
for code that already runs an event loop. Both take explicit parameters; only
``main()`` reads environment variables.
"""

import argparse
import asyncio
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from alembic import command
from alembic.config import Config
from loguru import logger
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from kavalai.db import ensure_async_scheme
from kavalai.paths import MIGRATIONS_PATH

MIGRATION_SETS = {
    "agents": os.path.join(MIGRATIONS_PATH, "agents"),
    "backoffice": os.path.join(MIGRATIONS_PATH, "backoffice"),
}


def migrate(
    set_name: str,
    uri: str | None = None,
    *,
    connection: Connection | None = None,
    schema: str | None = None,
    skip_create_schema: bool = False,
    max_wait: float = 60.0,
) -> None:
    """Upgrade one migration set to head, blocking until it is done.

    Pass exactly one of ``uri`` and ``connection``. With a URI whose driver is
    asynchronous (the default for ``postgresql://`` and ``sqlite://``) the
    upgrade runs on an event loop of its own. Called from synchronous code
    inside a running event loop — a startup hook, a test fixture — it runs
    that loop on a worker thread and blocks the caller as before; code that
    can await should await :func:`migrate_async` instead.

    Args:
        set_name: ``"agents"`` or ``"backoffice"``.
        uri: Database URI. ``postgresql://`` runs over asyncpg and
            ``sqlite://`` over aiosqlite; a URI naming a synchronous driver
            runs on that driver.
        connection: An open synchronous SQLAlchemy ``Connection``. The set
            runs in its current transaction, which the caller commits; if none
            is open, one is begun and committed here. The connection is not
            closed.
        schema: Target schema for the set's tables and the ``alembic_version``
            table. ``None`` uses the database default (Postgres: ``public``;
            SQLite: the main database).
        skip_create_schema: Don't issue ``CREATE SCHEMA IF NOT EXISTS``.
        max_wait: Seconds to keep retrying while a database server refuses
            connections. Not used with ``connection`` or with SQLite, which
            has no server to wait for.

    Raises:
        ValueError: An unknown set, or not exactly one of ``uri`` and
            ``connection``.
        TypeError: ``connection`` is not a synchronous ``Connection``.
    """
    _check_target(set_name, uri, connection)
    create_schema = not skip_create_schema
    if connection is not None:
        if not isinstance(connection, Connection):
            raise TypeError(
                "migrate() takes a synchronous Connection; "
                "await migrate_async() with an AsyncConnection instead."
            )
        _upgrade(connection, set_name, schema, create_schema)
        return

    url = _driver_url(uri)
    if not url.get_dialect().is_async:
        _migrate_sync_uri(url, set_name, schema, create_schema, max_wait)
        return

    def run_upgrade() -> None:
        asyncio.run(
            _migrate_async_uri(url, set_name, schema, create_schema, max_wait),
            loop_factory=asyncio.new_event_loop,
        )

    if _event_loop_running():
        # asyncio.run refuses to nest; a thread of its own gets a loop of its
        # own, and the caller blocks exactly as a synchronous call would.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run_upgrade).result()
        return
    run_upgrade()


async def migrate_async(
    set_name: str,
    uri: str | None = None,
    *,
    connection: AsyncConnection | None = None,
    schema: str | None = None,
    skip_create_schema: bool = False,
    max_wait: float = 60.0,
) -> None:
    """Upgrade one migration set to head from inside an event loop.

    The arguments are those of :func:`migrate`, except that ``connection`` is
    an ``AsyncConnection``; the set runs on its sync facade through
    ``run_sync``, in its current transaction. A URI naming a synchronous
    driver is migrated in a worker thread, so the loop is not blocked.
    """
    _check_target(set_name, uri, connection)
    create_schema = not skip_create_schema
    if connection is not None:
        if not isinstance(connection, AsyncConnection):
            raise TypeError(
                "migrate_async() takes an AsyncConnection; "
                "call migrate() with a synchronous Connection instead."
            )
        await connection.run_sync(_upgrade, set_name, schema, create_schema)
        return

    url = _driver_url(uri)
    if url.get_dialect().is_async:
        await _migrate_async_uri(url, set_name, schema, create_schema, max_wait)
    else:
        await asyncio.to_thread(
            _migrate_sync_uri, url, set_name, schema, create_schema, max_wait
        )


def _check_target(set_name: str, uri: str | None, connection) -> None:
    if set_name not in MIGRATION_SETS:
        raise ValueError(
            f"Unknown migration set {set_name!r}; expected one of "
            f"{sorted(MIGRATION_SETS)}"
        )
    if (uri is None) == (connection is None):
        raise ValueError("Pass exactly one of uri and connection.")


def _driver_url(uri: str) -> URL:
    """The URL to connect with: the async driver unless ``uri`` names one."""
    url = make_url(uri)
    if "+" in url.drivername:
        return url
    return make_url(ensure_async_scheme(uri))


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _upgrade(
    connection: Connection, set_name: str, schema: str | None, create_schema: bool
) -> None:
    """Run ``set_name`` to head on ``connection``.

    A transaction the connection already has is the caller's to commit;
    Alembic sees it as external and leaves it open. Otherwise the upgrade gets
    a transaction of its own, committed here or rolled back on error.
    """
    logger.info(f"Running {set_name!r} migrations (schema={schema or 'default'}).")
    if connection.in_transaction():
        _run_alembic(connection, set_name, schema, create_schema)
    else:
        with connection.begin():
            _run_alembic(connection, set_name, schema, create_schema)
    logger.info(f"{set_name!r} migrations completed successfully.")


def _run_alembic(
    connection: Connection, set_name: str, schema: str | None, create_schema: bool
) -> None:
    if schema and create_schema and connection.dialect.name == "postgresql":
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    config = Config()
    config.set_main_option("script_location", MIGRATION_SETS[set_name])
    config.attributes["connection"] = connection
    config.attributes["schema"] = schema
    command.upgrade(config, "head")


async def _migrate_async_uri(
    url: URL,
    set_name: str,
    schema: str | None,
    create_schema: bool,
    max_wait: float,
) -> None:
    try:
        engine = create_async_engine(url, poolclass=NullPool)
    except ImportError as exc:
        raise ImportError(
            f"Migrating {url.get_backend_name()} over {url.get_driver_name()} "
            f"needs the '{exc.name}' package. "
            'Install it with: pip install "kavalai[runtime]"'
        ) from exc
    try:
        connection = await _connect_async(engine, _wait_for(url, max_wait))
        try:
            await connection.run_sync(_upgrade, set_name, schema, create_schema)
        finally:
            await connection.close()
    finally:
        await engine.dispose()


def _migrate_sync_uri(
    url: URL,
    set_name: str,
    schema: str | None,
    create_schema: bool,
    max_wait: float,
) -> None:
    engine = create_engine(url, poolclass=NullPool)
    try:
        with _connect(engine, _wait_for(url, max_wait)) as connection:
            _upgrade(connection, set_name, schema, create_schema)
    finally:
        engine.dispose()


def _wait_for(url: URL, max_wait: float) -> float:
    return 0.0 if url.get_backend_name() == "sqlite" else max_wait


def _pauses(max_wait: float) -> Iterator[float]:
    """Pauses between connection attempts, until ``max_wait`` seconds pass.

    They double from one second to at most ten.
    """
    deadline = time.monotonic() + max_wait
    pause = 1.0
    while time.monotonic() < deadline:
        yield pause
        pause = min(pause * 2, 10.0)


def _retryable(exc: Exception) -> bool:
    """Whether a failed connection attempt may succeed later.

    SQLAlchemy wraps a synchronous driver's failures in ``OperationalError``,
    but asyncpg's arrive as they are: a refused or timed-out socket as
    ``OSError``, a server that is still starting as SQLSTATE ``57P03``.
    """
    return (
        isinstance(exc, (OperationalError, OSError))
        or getattr(exc, "sqlstate", None) == "57P03"
    )


def _next_pause(exc: Exception, pauses: Iterator[float]) -> float | None:
    pause = next(pauses, None) if _retryable(exc) else None
    if pause is None:
        logger.error(f"Could not connect to the database: {exc}")
    else:
        logger.info(f"Database connection failed, retrying in {pause}s: {exc}")
    return pause


def _connect(engine: Engine, max_wait: float, sleep=time.sleep) -> Connection:
    """Connect, retrying with backoff for up to ``max_wait`` seconds."""
    pauses = _pauses(max_wait)
    while True:
        try:
            return engine.connect()
        except Exception as exc:
            pause = _next_pause(exc, pauses)
            if pause is None:
                raise
        sleep(pause)


async def _connect_async(
    engine: AsyncEngine, max_wait: float, sleep=asyncio.sleep
) -> AsyncConnection:
    """Connect, retrying with backoff for up to ``max_wait`` seconds."""
    pauses = _pauses(max_wait)
    while True:
        try:
            return await engine.connect()
        except Exception as exc:
            pause = _next_pause(exc, pauses)
            if pause is None:
                raise
        await sleep(pause)


def main():
    parser = argparse.ArgumentParser(description="Kaval.AI Database Migration Tool")
    parser.add_argument(
        "type",
        choices=["agents", "backoffice"],
        help="Type of migrations to run ('agents' = the agents set).",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Database URI (defaults to KAVALAI_DB_URI / KAVALAI_BO_DB_URI).",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Target schema (defaults to KAVALAI_DB_SCHEMA / KAVALAI_BO_DB_SCHEMA).",
    )
    parser.add_argument(
        "--skip-create-schema",
        action="store_true",
        help="Don't create schema, just apply migrations.",
    )
    args = parser.parse_args()

    if args.type == "backoffice":
        uri = args.uri or os.environ["KAVALAI_BO_DB_URI"]
        schema = args.schema or os.environ.get("KAVALAI_BO_DB_SCHEMA")
        set_name = "backoffice"
    else:
        uri = args.uri or os.environ["KAVALAI_DB_URI"]
        schema = args.schema or os.environ.get("KAVALAI_DB_SCHEMA")
        set_name = "agents"

    migrate(
        set_name,
        uri=uri,
        schema=schema,
        skip_create_schema=args.skip_create_schema,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
