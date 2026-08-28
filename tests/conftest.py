import os
import logging
from loguru import logger

import pytest
import pytest_asyncio
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer
from kavalai.migrate_db import migrate

from kavalai.db import Base, build_db_uri, db_manager


@pytest.fixture(autouse=True)
def caplog_loguru(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15-trixie") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def agents_db_config(postgres_container):
    config = dict(
        uri=build_db_uri(
            user=postgres_container.username,
            password=postgres_container.password,
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            db_name=postgres_container.dbname,
        ),
        schema="test_agents",
    )
    os.environ["KAVALAI_DB_URI"] = config["uri"]
    return config


@pytest.fixture(scope="session")
def migrated_agents_db(agents_db_config):
    migrate(
        "agents",
        uri=agents_db_config["uri"],
        schema=agents_db_config["schema"],
    )


@pytest.fixture(scope="session")
def agents_session_maker(agents_db_config, migrated_agents_db):
    return db_manager.get_sessionmaker(
        uri=agents_db_config["uri"],
        schema=agents_db_config["schema"],
        pool_size=1,
        max_overflow=0,
    )


@pytest_asyncio.fixture(scope="function")
async def agents_db(agents_session_maker):
    async with agents_session_maker() as session:
        tables = ", ".join(
            f"test_agents.{table.name}"
            for table in reversed(Base.metadata.sorted_tables)
        )

        if tables:
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE;"))
            await session.commit()

        yield session

        # Explicitly roll back any uncommitted changes to keep tests isolated
        await session.rollback()
