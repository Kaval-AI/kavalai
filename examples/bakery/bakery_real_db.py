"""The bakery email assistant again, on Postgres instead of in memory.

Bring the database up and run it from the repository root::

    docker compose up -d postgres_db
    dotenv run python -m kavalai.migrate_db agents
    dotenv run python -m examples.bakery.bakery_real_db

Then send it an email::

    curl -s http://localhost:25101/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"email": {
              "sender": "Mari Tamm <mari@example.test>",
              "subject": "Kringle order",
              "body": "Hello, I would like 4 kringles for 2026-09-12. Mari Tamm"
            }}}'

Same :file:`assistant.yaml`, same tools, same replies — and this one still has
the order tomorrow. Two databases are involved and they are not the same kind
of thing: the **agent database** is Kaval.AI's own (sessions, runs, tasks,
model-call statistics) and its tables come from ``kavalai.migrate_db``; the
**order book** belongs to the bakery, so this example owns its DDL and creates
its table at startup. They share one engine and one connection pool.
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.server import mask_db_uri

from examples.bakery.bakery import (
    PostgresOrderBook,
    build_engine,
    create_app,
    use_order_book,
)

# Host and port to serve the agent
HOST = "0.0.0.0"
PORT = 25101

DB_URI = os.environ["KAVALAI_DB_URI"]
DB_SCHEMA = os.environ.get("KAVALAI_DB_SCHEMA", "public")
DB_POOL_SIZE = int(os.environ.get("KAVALAI_DB_POOL_SIZE", "0"))
DB_MAX_OVERFLOW = int(os.environ.get("KAVALAI_DB_MAX_OVERFLOW", "0"))

logger.info(f"Agent database: {mask_db_uri(DB_URI)} (schema {DB_SCHEMA})")
session_maker = db_manager.get_sessionmaker(
    uri=DB_URI,
    schema=DB_SCHEMA,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
)
agent_service = AgentService(session_maker)

# The order book the workflow's `store_order` tool writes to. Bound once, here,
# because assistant.yaml names the tool and not its dependencies.
order_book = use_order_book(PostgresOrderBook(session_maker, schema=DB_SCHEMA))

engine = build_engine(agent_service)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the bakery's own table before the first request arrives.

    The agent database's tables are Alembic's job (``python -m
    kavalai.migrate_db agents``) and are deliberately not created here: a server
    that migrates on startup is a server that migrates from several replicas at
    once.
    """
    await order_book.create_table()
    logger.info(f"Order book ready at {order_book.qualified_table}")
    yield


def main() -> None:
    logger.info(f"Serving the bakery email assistant on http://{HOST}:{PORT}")
    uvicorn.run(create_app(engine, session_maker, lifespan), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
