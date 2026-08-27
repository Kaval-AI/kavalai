"""The bakery email assistant on an in-memory database, served over REST.

Run it from the repository root::

    dotenv run python -m examples.bakery.bakery_in_memory

Then send it an email::

    curl -s http://localhost:25100/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"email": {
              "sender": "Mari Tamm <mari@example.test>",
              "subject": "Kringle order",
              "body": "Hello, I would like 4 kringles for 2026-09-12. Mari Tamm"
            }}}'

Nothing here describes what the assistant *does* — that is
:file:`assistant.yaml`, and this module only decides where its runs and its
orders are written down. Both go to memory: the SQLite agent database is
``:memory:`` and the order book is a Python list, so the process is the whole
world and restarting it is the reset button. :file:`bakery_real_db.py` is the
same agent with both of those pointed at Postgres.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from kavalai import db_manager
from kavalai.agent_service import AgentService

from examples.bakery.bakery import (
    InMemoryOrderBook,
    build_engine,
    create_app,
    use_order_book,
)

# Host and port to serve the agent
HOST = "0.0.0.0"
PORT = 25100

# In-memory SQLite for the agent database: sessions, runs, tasks and model-call
# stats live only as long as the process.
AGENT_DB_PATH = ":memory:"

session_maker = db_manager.get_sqlite_sessionmaker(db_path=AGENT_DB_PATH)
agent_service = AgentService(session_maker)

# The order book the workflow's `store_order` tool writes to. Bound once, here,
# because assistant.yaml names the tool and not its dependencies.
order_book = use_order_book(InMemoryOrderBook())

engine = build_engine(agent_service)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the agent database tables before the first request arrives."""
    await db_manager.init_sqlite(db_path=AGENT_DB_PATH)
    yield


def main() -> None:
    logger.info(f"Serving the bakery email assistant on http://{HOST}:{PORT}")
    uvicorn.run(create_app(engine, session_maker, lifespan), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
