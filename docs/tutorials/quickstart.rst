Quickstart
==========

Four steps: call a model, get structured data back, wrap it in a workflow
that remembers the conversation, then keep that memory in Postgres. Every
output below is real.

Readers wanting the full tour should start at :doc:`llm_clients`. Readers new
to language models altogether should read :doc:`../guides/concepts` first.

Install
-------

.. code-block:: bash

   pip install "kavalai[common]"
   export OPENAI_API_KEY="sk-..."

Kaval.AI needs Python 3.12+. Any of OpenAI, Gemini, Anthropic or a local Ollama
will do — see :doc:`installation`.

1. Call a model
---------------

:func:`~kavalai.make_client` builds a client from a ``provider/model`` id and
finds the matching API key in your environment.

.. code-block:: python

   from kavalai import make_client

   client = make_client("openai/gpt-5.4-mini")

   answer = await client.prompt(
       "What is the capital of Estonia? Answer in one sentence."
   )
   print(answer)

.. code-block:: text

   The capital of Estonia is Tallinn.

The clients are async. In a script, wrap the calls in ``asyncio.run(main())``;
in a notebook, ``await`` directly as above.

Switching provider is a change of one string — ``gemini/gemini-3.1-flash-lite``,
``anthropic/claude-sonnet-5``, ``ollama/llama3``. Nothing else moves.

2. Get structured data, not prose
---------------------------------

Pass a Pydantic model and you get a validated object instead of text to parse:

.. code-block:: python

   from pydantic import BaseModel

   class City(BaseModel):
       name: str
       country: str
       population: int
       fun_fact: str

   city = await client.prompt("Describe Tallinn.", response_model=City)

   print(city.country)
   print(city.population)
   print(city.fun_fact)

.. code-block:: text

   Estonia
   450000
   Tallinn's UNESCO-listed Old Town is one of the best-preserved
   medieval city centres in Europe.

``city.population`` is an ``int``. Declare the shape once and every field arrives
typed — with any provider.

3. Build a workflow
-------------------

A workflow is a small typed graph: input in, output out, one node per step. Even
a one-node graph buys you validation, persistence and a recorded conversation.

.. code-block:: python

   import asyncio

   from pydantic import BaseModel

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder


   class Message(BaseModel):
       user_message: str


   class Reply(BaseModel):
       agent_response: str
       choices: list[str]


   async def main():
       # In-memory tables here; Postgres in step 4.
       await db_manager.init_sqlite()

       workflow = (
           WorkflowBuilder("Village greeter", llm_model="openai/gpt-5.4-mini")
           .data_model("input", Message)
           .data_model("output", Reply)
           .start("reply")
           .llm(
               "reply",
               prompt=(
                   "You are the greeter of Green Village "
                   "(104 residents, one pub). "
                   "Reply warmly in one sentence, and suggest up to 3 short "
                   "quick-reply choices the visitor might tap next."
               ),
               inputs={"message": "input"},
               output="output",
               next="end",
           )
           .end()
           .build_engine(
               agent_service=AgentService(db_manager.get_sqlite_sessionmaker())
           )
       )

       state = await workflow.run(
           {"user_message": "Hi, I'm visiting from Tallinn!"}
       )
       print(state.output_data["agent_response"])
       print("choices:", state.output_data["choices"])
       print("path   :", " → ".join(state.trace))
       print("tokens :", state.token_usage["total_tokens"])


   asyncio.run(main())

.. code-block:: text

   Welcome to Green Village—so lovely to have you here from Tallinn!
   choices: ['Thanks!', 'What's nearby?', 'Tell me about the pub']
   path   : start → reply → end
   tokens : 165

Read that back a piece at a time:

``.data_model("input", Message)`` and ``.data_model("output", Reply)``
   Register the workflow's input and output types. Declaring them is what lets
   the engine validate every value and ask the model for exactly the right
   shape — ``choices`` comes back as a real list of strings.

``.start("reply")`` and ``next="end"``
   The edges. A graph always begins at ``start`` and finishes at an ``end``
   node.

``inputs={"message": "input"}``
   Takes the workflow's ``input`` from the run context and hands it to the
   prompt under the local name ``message``.

``.build_engine(agent_service=…)``
   Validates the graph and returns a ready :class:`~kavalai.WorkflowEngine`.
   Passing an :class:`~kavalai.agent_service.AgentService` gives the bot
   **memory**: LLM nodes replay the session's history by default, so the next
   turn sees this one. Here that history lives in in-memory SQLite; step 4
   points the same service at Postgres and nothing else changes.

Every run returns a :class:`~kavalai.WorkflowState` — the status, the ``trace``
of visited nodes, the full context, the output and the token usage. It is
JSON-serialisable and persisted, so you can reload and inspect it later.

4. Store the runs in Postgres
-----------------------------

In-memory SQLite is fine for a first look, but it is gone when the process
exits. Anything you intend to keep — chat history, runs, token usage — belongs
in Postgres. Two things change: the tables have to exist, and the service has
to point at them.

**Start a database.** The repository's ``docker-compose.yml`` ships a
development one, on the ``pgvector`` image so RAG works against it too:

.. code-block:: bash

   docker compose up -d postgres_db

It listens on ``localhost:6543``, with ``kavalai_dev`` as user, password and
database name.

**Create the tables.** The schema is managed with Alembic; the agent runtime
tables are the ``agents`` set:

.. code-block:: bash

   python -m kavalai.migrate_db agents \
       --uri postgresql://kavalai_dev:kavalai_dev@localhost:6543/kavalai_dev \
       --schema agents

.. code-block:: text

   INFO | Running 'agents' migrations (schema=agents).
   INFO | 'agents' migrations completed successfully.

The runner waits for the database to accept connections, creates the schema if
it is missing, and upgrades to head. It is idempotent — run it again after
every upgrade, before starting the service that uses it. Without ``--uri`` and
``--schema`` it reads ``KAVALAI_DB_URI`` and ``KAVALAI_DB_SCHEMA``, which is
how the Docker images are configured.

Right after the migration the schema holds the runtime data model, and nothing
else:

.. code-block:: bash

   docker compose exec postgres_db \
       psql -U kavalai_dev -d kavalai_dev -c "\dt agents.*"

.. code-block:: text

                   List of relations
    Schema |       Name       | Type  |    Owner
   --------+------------------+-------+-------------
    agents | agents           | table | kavalai_dev
    agents | alembic_version  | table | kavalai_dev
    agents | chat_messages    | table | kavalai_dev
    agents | model_call_stats | table | kavalai_dev
    agents | runs             | table | kavalai_dev
    agents | sessions         | table | kavalai_dev
    agents | tasks            | table | kavalai_dev
   (7 rows)

**Point the workflow at it.** Drop the ``await db_manager.init_sqlite()`` line
from step 3 and build the session maker from the same URI instead:

.. code-block:: python

   session_maker = db_manager.get_sessionmaker(
       uri="postgresql://kavalai_dev:kavalai_dev@localhost:6543/kavalai_dev",
       schema="agents",
   )
   agent_service = AgentService(session_maker)

Then pass that to the builder in place of the SQLite one, as
``.build_engine(agent_service=agent_service)``.

Nothing else moves — the graph, the models and the ``run()`` call are
unchanged. The agent, the session, every run and every model call are now rows
you can query, and the next turn of the conversation picks up the history from
there.

The backoffice keeps its own tables in a second, independent set
(``python -m kavalai.migrate_db backoffice``, reading ``KAVALAI_BO_DB_URI``
and ``KAVALAI_BO_DB_SCHEMA``); the two may share one Postgres instance or not.
See :doc:`../deploy/index`.

Where to next
-------------

Pick whichever matches what you are building:

* :doc:`llm_clients` — streaming, conversations, timeouts, embeddings.
* :doc:`agents` — give the model tools and let it act.
* :doc:`workflow` — branching, tool nodes, agent nodes, deterministic tests.
* :doc:`rag` — answer from your own documents.
* :doc:`serving` — put it behind an HTTP endpoint.
* :doc:`run_in_browser` — run the whole stack client-side, no API key.
* :doc:`observability_storage` — what is stored, and how to read it back.
