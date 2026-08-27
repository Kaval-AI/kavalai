"""The business research agent, served over REST with nothing to set up.

Run it from the repository root::

    dotenv run python -m examples.business_info_agent.business_info_in_memory

Then ask it about a company::

    curl -s http://localhost:25200/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"business_query": "Kaval.AI (kaval.ai)"}}'

and grade it with the case file next door::

    dotenv run kavalai-eval examples/business_info_agent/eval_cases.yaml \
        --port 25200 --tag baseline

What the agent does is `business_info.py`; this module only decides where its
*sessions* are recorded. Here that is an in-memory SQLite database, so the
process is the whole world and restarting it is the reset button — which is
what makes it the server to evaluate against.

The pages it reads are the live web, through Crawl4AI's browser: a run takes
tens of seconds, and two runs of the same case can differ because the web did.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from examples.business_info_agent.business_info import build_engine
from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.server import create_agent_router
from kavalai.workflow import WorkflowEngine

HOST = "0.0.0.0"
PORT = 25200

# In-memory SQLite: sessions, runs, tasks and model-call stats live only as
# long as the process.
AGENT_DB_PATH = ":memory:"

session_maker = db_manager.get_sqlite_sessionmaker(db_path=AGENT_DB_PATH)
agent_service = AgentService(session_maker)


def create_app(engine: WorkflowEngine) -> FastAPI:
    """Serve the workflow over POST /run_agent and POST /stream_agent."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Create the agent database tables, and open the engine's tool
        # servers, before the first request arrives.
        await db_manager.init_sqlite(db_path=AGENT_DB_PATH)
        await engine.connect()
        try:
            yield
        finally:
            await engine.aclose()

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
    logger.info(f"Serving the business research agent on http://{HOST}:{PORT}")
    uvicorn.run(
        create_app(build_engine(agent_service=agent_service)), host=HOST, port=PORT
    )


if __name__ == "__main__":
    main()
