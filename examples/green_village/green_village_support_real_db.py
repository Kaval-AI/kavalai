"""The Green Village chatbot again, but on Postgres instead of in-memory SQLite.

Bring up the Postgres database and run it from the repository root::

    docker compose up -d postgres_db
    dotenv run python -m kavalai.migrate_db agents
    dotenv run python -m examples.green_village.green_village_support_real_db

Then talk to it from the terminal chat client::

    python -m examples.chat_client.chat_client --base-url http://localhost:25001

or straight over HTTP::

    curl -s http://localhost:25001/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"user_message": "How deep is Lake Miller?"}}'

The reply carries a ``session_id``; send it back on the next request and the
two turns are one conversation — the same as the in-memory example, except this
one still remembers it tomorrow.

Recording a run takes two pieces. The ``AgentService`` records the session and
the run; a ``PostgresTaskLogger`` records what happened inside it — one row per
node with its inputs, prompt, output and duration, plus the model calls the
engine's token accumulator forwards to it. The second one is what the
backoffice task debugger reads, so a conversation served here can be opened in
the frontend and inspected node by node. The embedding calls the RAG service
makes are written by ``PostgresRagService`` itself, into the same
``model_call_stats`` table.
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.rag import PostgresRagService
from kavalai.server import create_agent_router, mask_db_uri
from kavalai.workflow import WorkflowBuilder
from kavalai.workflow.tasklog import PostgresTaskLogger

# See https://qdrant.github.io/fastembed/examples/Supported_Models/ for the
# full list. This one is small, local and needs no API key.
EMBEDDING_MODEL = "fastembed/BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-5.6-luna"

# Host and port to serve the agent
HOST = "0.0.0.0"
PORT = 25001


DB_URI = os.environ["KAVALAI_DB_URI"]
DB_SCHEMA = os.environ.get("KAVALAI_DB_SCHEMA", "public")
DB_POOL_SIZE = int(os.environ.get("KAVALAI_DB_POOL_SIZE", "0"))
DB_MAX_OVERFLOW = int(os.environ.get("KAVALAI_DB_MAX_OVERFLOW", "0"))

# A collection is a table of its own here, rather than a filter over one index,
# so give it a name of its own instead of retrieving from "default".
COLLECTION = "green_village"

# List of green village facts.
FACTS = [
    "President of Green Village is Thomas Cook (born 12.04.1994).",
    "Green Village has 104 residents.",
    "Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.",
    "The tallest building in Green Village is the Old Grain Tower at 23 metres.",
    "Green Village's official flower is the marsh marigold.",
    "The village bakery, run by Greta Lindqvist, sells exactly 340 loaves every week.",
    "Green Village has one school with 14 pupils and 2 teachers.",
    (
        "The annual Turnip Festival takes place every year on the third "
        "Saturday of October."
    ),
    (
        "Green Village's fire brigade consists of 7 volunteers and one "
        "dalmatian named Pepper."
    ),
    "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
    "Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).",
    "The village has 3 streets: Main Road, Willow Lane, and Cobbler's Path.",
    "The local church bell weighs 412 kilograms and was cast in 1901.",
    "Green Village produces 8 tons of honey per year from its 26 beehives.",
    "The village library owns 1,847 books and is open on Tuesdays and Fridays.",
    "The speed limit everywhere in Green Village is 30 km/h.",
    "Green Village's only pub, The Rusty Anchor, has been operating since 1923.",
]


class Message(BaseModel):
    """Represents user message to the agent."""

    user_message: str


class Reply(BaseModel):
    """Represents agent reply to the user."""

    agent_response: str


logger.info(f"Agent database: {mask_db_uri(DB_URI)} (schema {DB_SCHEMA})")
session_maker = db_manager.get_sessionmaker(
    uri=DB_URI,
    schema=DB_SCHEMA,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
)
agent_service = AgentService(session_maker)

# Per-node debugging data: one `tasks` row per node, and the `model_call_stats`
# rows the engine's token accumulator forwards here. Both land in the same
# database as the sessions, which is what the backoffice needs to show a run's
# full result.
task_logger = PostgresTaskLogger(agent_service)

# The same sessionmaker serves the index: one database, one pool. The RAG
# tables are not part of the Alembic set — this backend owns its own DDL and
# provisions a collection on first index, taking the vector dimension from the
# embeddings it just computed.
logger.info(f"Initializing RAG with model {EMBEDDING_MODEL}")
rag = PostgresRagService.from_session_maker(
    session_maker, EMBEDDING_MODEL, schema=DB_SCHEMA
)

# Build the workflow engine.
# 1. Query the RAG index for the facts related to the user message.
# 2. Write the reply from those facts.
engine = (
    WorkflowBuilder(
        "Green Village support",
        llm_model=LLM_MODEL,
        rag_collection=COLLECTION,
    )
    .data_model("input", Message)
    .data_model("output", Reply)
    .start("get_related_facts")
    .rag_query(
        "get_related_facts",
        query="{{ context.input.user_message }}",
        output="facts",
        top_k=5,
        # "content" keeps just the hit texts, which is all the prompt
        # below wants — no ids, scores or timestamps.
        store="content",
        next="reply",
    )
    .llm(
        "reply",
        prompt=(
            "You are the AI assistant of the Green Village tourist "
            "information centre. Help users with their inquiries.\n"
            "NB! Green Village is a fictional village, so rely only on the facts given in the context.\n"
            "Steer any offtopic requests back to green village.\n"
            "Related facts:\n{{ context.facts }}"
        ),
        inputs={"input": "input", "facts": "facts"},
        output="output",
        next="end",
    )
    .end()
    .build_engine(
        rag_services=rag,
        agent_service=agent_service,
        task_logger=task_logger,
    )
)


async def index_facts() -> None:
    """Rebuild the index from FACTS on every start."""
    indexed = await rag.count_entries(COLLECTION)
    if indexed:
        logger.info(f"Clearing {indexed} facts from collection '{COLLECTION}'")
        await rag.drop_collection(COLLECTION)

    logger.info(f"Indexing {len(FACTS)} facts")
    await rag.index_batch(
        texts=FACTS,
        metadata_list=[{}] * len(FACTS),
        source_ids=[f"fact-{i:02d}" for i in range(len(FACTS))],
        collection_name=COLLECTION,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Lifespan of the app.

        During startup, rebuild the fact index, so the server only starts
        serving once there is something to retrieve. On shutdown, drain the
        task logger: its writes are fire-and-forget, so the last conversation
        before a restart is only complete in the backoffice if they are
        awaited.
        """
        await index_facts()
        yield
        await task_logger.close()

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
    logger.info(f"Serving Green Village support on http://{HOST}:{PORT}")
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
