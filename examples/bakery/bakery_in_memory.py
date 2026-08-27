"""The bakery email assistant, served over REST with nothing to set up.

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

What the assistant does is `assistant.yaml`; this module only decides where its
*sessions* are recorded. Here that is an in-memory SQLite database, so the
process is the whole world and restarting it is the reset button.
`bakery_real_db.py` is the same agent recording into Postgres instead.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from loguru import logger

from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.server import create_agent_router
from kavalai.workflow import WorkflowEngine

HOST = "0.0.0.0"
PORT = 25100

# In-memory SQLite: sessions, runs, tasks and model-call stats live only as
# long as the process.
AGENT_DB_PATH = ":memory:"

WORKFLOW_PATH = Path(__file__).with_name("assistant.yaml")

session_maker = db_manager.get_sqlite_sessionmaker(db_path=AGENT_DB_PATH)
agent_service = AgentService(session_maker)


def build_engine(agent_service: AgentService) -> WorkflowEngine:
    """Load assistant.yaml and give it somewhere to record its runs."""
    return WorkflowEngine.from_yaml_path(
        str(WORKFLOW_PATH), agent_service=agent_service
    )


def create_app(engine: WorkflowEngine) -> FastAPI:
    """Serve the workflow over POST /run_agent and POST /stream_agent."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Create the agent database tables before the first request arrives.
        await db_manager.init_sqlite(db_path=AGENT_DB_PATH)
        yield

    app = FastAPI(
        title=engine.graph.name,
        description=engine.graph.description,
        version=engine.graph.version,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.include_router(create_agent_router(engine, session_provider=session_maker))
    return app


def main() -> None:
    logger.info(f"Serving the bakery email assistant on http://{HOST}:{PORT}")
    uvicorn.run(create_app(build_engine(agent_service)), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
