"""The Green Village tourist information chatbot, served over REST.

Run it::

    python -m examples.green_village.green_village_support_in_memory

Then talk to it from the terminal chat client::

    python -m examples.chat_client.chat_client --base-url http://localhost:25000

or straight over HTTP::

    curl -s http://localhost:25000/run_agent \
        -H 'Content-Type: application/json' \
        -d '{"data": {"user_message": "How deep is Lake Miller?"}}'

"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from kavalai import db_manager
from kavalai.agent_service import AgentService
from kavalai.rag import SqliteRagService
from kavalai.server import create_agent_router
from kavalai.workflow import WorkflowBuilder

# See https://qdrant.github.io/fastembed/examples/Supported_Models/ for the
# full list. This one is small, local and needs no API key.
EMBEDDING_MODEL = "fastembed/BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-5.6-luna"

# Host and port to serve the agent
HOST = "0.0.0.0"
PORT = 25000

# In-memory SQLite for both the agent database and the RAG index.
AGENT_DB_PATH = ":memory:"
INDEX_PATH = ":memory:"

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
    "Green Village has 3 streets: Main Road, Willow Lane, and Cobbler's Path.",
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


session_maker = db_manager.get_sqlite_sessionmaker(db_path=AGENT_DB_PATH)
agent_service = AgentService(session_maker)

logger.info(f"Initializing RAG with model {EMBEDDING_MODEL}")
rag = SqliteRagService(INDEX_PATH, EMBEDDING_MODEL)

# Build the workflow engine.
# 1. Query the RAG index for the facts related to the user message.
# 2. Write the reply from those facts.
engine = (
    WorkflowBuilder("Green Village support", llm_model=LLM_MODEL)
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
    .build_engine(rag_services=rag, agent_service=agent_service)
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Lifespan of the app.

        During startup, create the agent database tables and perform indexing
        in the RAG, so the server only starts serving once there is something
        to retrieve and somewhere to record the conversation.
        """
        await db_manager.init_sqlite(db_path=AGENT_DB_PATH)
        logger.info(f"Indexing {len(FACTS)} facts")
        await rag.index_batch(
            texts=FACTS,
            metadata_list=[{}] * len(FACTS),
            source_ids=[str(i) for i in range(len(FACTS))],
            collection_name="default",
        )
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
    logger.info(f"Serving Green Village support on http://{HOST}:{PORT}")
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
