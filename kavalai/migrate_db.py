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

Thin wrapper around ``alembic.command.upgrade``: it resolves the migration set
(``agents`` or ``backoffice``), waits for the database to accept connections,
creates the target schema if needed, and runs the set's revisions with the
schema applied via ``schema_translate_map`` (see
:mod:`kavalai.migrations.common`).

``migrate`` is a pure function taking explicit parameters; only ``main()``
reads environment variables — this module is an application entry point, the
rest of the library never touches the environment.
"""

import argparse
import os
import time

from alembic import command
from alembic.config import Config
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from kavalai.paths import MIGRATIONS_PATH

#: Migration set name -> script directory.
MIGRATION_SETS = {
    "agents": os.path.join(MIGRATIONS_PATH, "agents"),
    "backoffice": os.path.join(MIGRATIONS_PATH, "backoffice"),
}


def ensure_sync_scheme(uri: str) -> str:
    """Return ``uri`` with a synchronous driver (Alembic runs sync)."""
    url = make_url(uri)
    backend = url.get_backend_name()
    if backend == "postgresql":
        url = url.set(drivername="postgresql+psycopg2")
    elif backend == "sqlite":
        url = url.set(drivername="sqlite")
    # str(URL) masks the password; render it fully for engine creation.
    return url.render_as_string(hide_password=False)


def _connect_with_retry(engine, max_wait: float):
    """Connect to the database, retrying with backoff for up to ``max_wait`` s."""
    start_time = time.time()
    retry_interval = 1.0
    while True:
        try:
            connection = engine.connect()
            connection.execute(text("SELECT 1"))
            return connection
        except OperationalError:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.error(f"Failed to connect to database after {max_wait} seconds.")
                raise
            logger.info(
                f"Database connection failed, retrying in {retry_interval}s... "
                f"({int(elapsed)}s elapsed)"
            )
            time.sleep(retry_interval)
            retry_interval = min(retry_interval * 2, 10.0)


def migrate(
    set_name: str,
    uri: str,
    schema: str | None = None,
    skip_create_schema: bool = False,
    max_wait: float = 60.0,
):
    """Upgrade one migration set to head.

    Args:
        set_name: ``"agents"`` or ``"backoffice"``.
        uri: Database URI (async driver schemes are converted to sync).
        schema: Target schema for the set's tables and the ``alembic_version``
            table. ``None`` uses the database default (Postgres: ``public``;
            SQLite: the main database).
        skip_create_schema: Don't issue ``CREATE SCHEMA IF NOT EXISTS``.
        max_wait: How long to wait for the database to accept connections.
    """
    if set_name not in MIGRATION_SETS:
        raise ValueError(
            f"Unknown migration set {set_name!r}; expected one of "
            f"{sorted(MIGRATION_SETS)}"
        )

    sync_uri = ensure_sync_scheme(uri)
    logger.info(f"Running {set_name!r} migrations (schema={schema or 'default'}).")

    engine = create_engine(sync_uri)
    try:
        connection = _connect_with_retry(engine, max_wait=max_wait)
        with connection:
            is_postgres = connection.dialect.name == "postgresql"
            if schema and is_postgres and not skip_create_schema:
                connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
                connection.commit()

            config = Config()
            config.set_main_option("script_location", MIGRATION_SETS[set_name])
            config.attributes["connection"] = connection
            config.attributes["schema"] = schema

            command.upgrade(config, "head")
            connection.commit()
        logger.info(f"{set_name!r} migrations completed successfully.")
    finally:
        engine.dispose()


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
