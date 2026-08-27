"""The bakery email assistant again, recording its sessions into Postgres.

Bring the database up and run it from the repository root::

    docker compose up -d postgres_db
    dotenv run python -m kavalai.migrate_db agents
    dotenv run python -m examples.bakery.bakery_real_db

Then send it an email::

    curl -s http://localhost:25001/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"email": {
              "sender": "Mari Tamm <mari@example.test>",
              "subject": "Kringle order",
              "body": "Hello, I would like 4 kringles for 2026-09-12. Mari Tamm"
            }}}'

Same `assistant.yaml`, same tools, same replies. The only difference from
`bakery_in_memory.py` is the agent database — sessions, runs, tasks and
model-call statistics survive a restart here, and can be read afterwards in the
backoffice. The order book stays in memory either way: it belongs to the
example, not to Kaval.AI.
"""

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from loguru import logger

from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.server import create_agent_router, mask_db_uri
from kavalai.workflow import WorkflowEngine

HOST = "0.0.0.0"
PORT = 25001

DB_URI = os.environ["KAVALAI_DB_URI"]
DB_SCHEMA = os.environ.get("KAVALAI_DB_SCHEMA", "public")
DB_POOL_SIZE = int(os.environ.get("KAVALAI_DB_POOL_SIZE", "0"))
DB_MAX_OVERFLOW = int(os.environ.get("KAVALAI_DB_MAX_OVERFLOW", "0"))

WORKFLOW_PATH = Path(__file__).with_name("assistant.yaml")

logger.info(f"Agent database: {mask_db_uri(DB_URI)} (schema {DB_SCHEMA})")
session_maker = db_manager.get_sessionmaker(
    uri=DB_URI,
    schema=DB_SCHEMA,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
)
agent_service = AgentService(session_maker)


def build_engine(agent_service: AgentService) -> WorkflowEngine:
    """Load assistant.yaml and give it somewhere to record its runs."""
    return WorkflowEngine.from_yaml_path(
        str(WORKFLOW_PATH), agent_service=agent_service
    )


def create_app(engine: WorkflowEngine) -> FastAPI:
    """Serve the workflow over POST /run_agent and POST /stream_agent.

    No lifespan hook: the tables are Alembic's job (`python -m
    kavalai.migrate_db agents`). A server that migrates on startup is a server
    that migrates from several replicas at once.
    """
    app = FastAPI(
        title=engine.graph.name,
        description=engine.graph.description,
        version=engine.graph.version,
    )
    app.state.engine = engine
    app.include_router(create_agent_router(engine, session_provider=session_maker))
    return app


def main() -> None:
    logger.info(f"Serving the bakery email assistant on http://{HOST}:{PORT}")
    uvicorn.run(create_app(build_engine(agent_service)), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
