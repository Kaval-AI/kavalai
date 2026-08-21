Observability & storage
========================

Kaval.AI persists what your agents do, so you can give them **memory**, **reload**
and inspect past runs, and **browse** everything in the backoffice UI. Two small
pieces handle it:

* :class:`~kavalai.agent_service.AgentService` — agents, sessions, runs and chat
  history, over any database the ORM supports.
* ``TaskLogger`` — per-node debug data and per-call model statistics.

This tutorial focuses on storage: how a conversation is recorded, how a workflow
carries data between steps, what each table holds, and how to swap the backend.
For the schema itself — every column and its purpose — see
:doc:`../guides/data_model`.
For the *why* behind it, read the :doc:`../guides/observability` guide.

Chat history at the client level
--------------------------------

LLM clients are **stateless** — each call sees exactly the messages you give it.
A single ``prompt`` sends one message; to hold a conversation you build a
``ChatHistory`` (an ordered list of ``ChatMessage``) and append to it yourself:

.. code-block:: python

   from kavalai import OpenAIClient, ChatHistory, ChatMessage

   client = OpenAIClient("gpt-5.4-mini")

   history = ChatHistory(messages=[
       ChatMessage(role="system", content="You are a terse assistant."),
       ChatMessage(role="user", content="My name is Ada."),
       ChatMessage(role="assistant", content="Noted, Ada."),
       ChatMessage(role="user", content="What is my name?"),
   ])
   print(await client.chat_completions(chat_history=history))  # -> "Ada."

Managing that list by hand gets tedious quickly. Workflows do it for you: the
engine writes each turn to storage and replays the conversation on the next one,
which is what gives a chatbot memory across turns (the ``use_history`` flag, on
by default for LLM nodes). The rest of this page is about where those turns — and
everything else — are kept.

Context: how a workflow carries data
------------------------------------

Within a single run a workflow passes data between nodes through its **context**
— a dictionary the engine fills as it goes. The input lands under ``input``; each
node writes its result under its ``output`` name; later nodes read it back by
**context path**, either as a node input or inside a prompt:

.. code-block:: python

   from pydantic import BaseModel
   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder

   class Email(BaseModel):
       text: str

   class Analysis(BaseModel):
       category: str

   class Reply(BaseModel):
       agent_response: str

   engine = (
       WorkflowBuilder("Triage", llm_model="openai/gpt-5.4-mini")
       .data_model("input", Email)
       .data_model("analysis", Analysis)
       .data_model("output", Reply)
       .start("classify")
       # writes its result into the context under `analysis`
       .llm("classify", prompt="Classify the email as billing, support or other.",
            inputs={"email": "input"}, output="analysis", next="reply")
       # reads it back by context path — here inside the prompt
       .llm("reply",
            prompt="Write a reply for a "
                   "{{ context.analysis.category }} email.",
            inputs={"email": "input"}, output="output", next="end")
       .end()
       .build_engine(
           agent_service=AgentService(db_manager.get_sqlite_sessionmaker())
       )
   )

When the run finishes, the whole context is returned as
:class:`~kavalai.WorkflowState`'s ``data`` and persisted to the run's ``context``
column, so you can reload exactly what each step saw.

Sessions and runs
-----------------

A **session** is one conversation between a user and an agent. Every message the
user sends triggers a **run** — a single invocation of the workflow that produces
one reply. Reuse the same ``external_id`` — any identifier from your own system,
a user, ticket or thread id — across turns and all those runs (and their chat
messages) belong to one thread, which is how memory works:

.. code-block:: python

   state1 = await engine.run({"text": "Hi, I'm Ada."}, external_id="user-42")
   state2 = await engine.run({"text": "What's my name?"}, external_id="user-42")

(``session_id`` does the same with the session's own primary id, e.g. the
``state.session_id`` of an earlier run.) Pass neither and the engine starts a
fresh session each time — a one-off interaction.

One service, any database
-------------------------

There is a single persistence service, :class:`~kavalai.agent_service.AgentService`;
what varies is the database its sessionmaker points at. You can drive it
directly — here over in-memory SQLite:

.. code-block:: python

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager

   await db_manager.init_sqlite()
   service = AgentService(db_manager.get_sqlite_sessionmaker())

   # A run belongs to a session. Reuse the external id and the conversation
   # accumulates under one session.
   agent, session, run1 = await service.initialize_workflow_run(
       agent_name="Greeter", external_id="user-42")
   await service.add_chat_message(agent_id=agent.id, session_id=session.id,
                                  run_id=run1.id, role="user",
                                  content="My name is Ada.")
   await service.add_chat_message(agent_id=agent.id, session_id=session.id,
                                  run_id=run1.id, role="assistant",
                                  content="Hi Ada!")

   agent2, session2, run2 = await service.initialize_workflow_run(
       agent_name="Greeter", external_id="user-42")
   await service.add_chat_message(agent_id=agent2.id, session_id=session2.id,
                                  run_id=run2.id, role="user",
                                  content="What's my name?")

   # Same agent + session across both runs; two different runs.
   print("same agent  :", agent.id == agent2.id)
   print("same session:", session.id == session2.id)
   print("two runs    :", run1.id != run2.id)

   for m in await service.get_chat_history(session.id):
       print(f"{m.role:>9}: {m.content}")

.. code-block:: text

   same agent  : True
   same session: True
   two runs    : True
        user: My name is Ada.
   assistant: Hi Ada!
        user: What's my name?

For local development and tests use SQLite — in-memory or file-backed:

.. code-block:: python

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager

   await db_manager.init_sqlite()               # create the tables
   service = AgentService(db_manager.get_sqlite_sessionmaker())

``init_sqlite()`` creates the tables; ``get_sqlite_sessionmaker()`` hands the
service an ordinary async session. Under Pyodide, where neither greenlet nor
``aiosqlite`` exists, use ``get_sqlite_compat_sessionmaker()`` instead — it
exposes the same awaitable surface over a synchronous engine. See
:doc:`run_in_browser`.

In production, point the very same service at Postgres:

.. code-block:: python

   from kavalai import db_manager
   from kavalai.agent_service import AgentService

   session_maker = db_manager.get_sessionmaker(
       uri="postgresql://user:pass@localhost:5432/kavalai"
   )
   service = AgentService(session_maker)

Hand it to the engine the same way — only the connection changes:

.. code-block:: python

   from kavalai.workflow import WorkflowEngine

   engine = WorkflowEngine.from_yaml(workflow_yaml, agent_service=service)

The tables
----------

The same ORM models define the schema everywhere — SQLite and Postgres hold
identical tables — so a run looks identical wherever it lives:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Table
     - What it holds
   * - ``agents``
     - One row per workflow/agent: its name, description, input/output schemas and
       the workflow definition.
   * - ``sessions``
     - One row per conversation, linked to an agent. An optional ``external_id``
       ties it to your own user, ticket or thread id.
   * - ``runs``
     - One row per workflow invocation: the ``input_data``, the ``output_data``
       and the resolved run ``context`` (what each step saw).
   * - ``chat_messages``
     - The conversation turns (``role`` + ``content``) — the basis of chat history.
   * - ``tasks``
     - Per-node debug data: each node's inputs and output, for drilling into what a
       step actually did.
   * - ``model_call_stats``
     - One row per LLM or embedding call: model, token counts, duration and the
       provider's status code. Failed attempts are recorded too, so a rate-limit
       storm shows up here rather than only in the logs. Where the provider
       reports them, ``cached_prompt_tokens`` and ``reasoning_tokens`` break out
       the parts of the totals that are billed differently. There is no ``cost``
       column: see :doc:`../guides/observability`.

The first four are written by ``AgentService``; ``tasks`` and
``model_call_stats`` come from ``TaskLogger``. The relationship is a simple
hierarchy:
**agent → sessions → runs → (chat_messages, tasks, model_call_stats)**.

Every column of every table, the reasoning behind the schema, the retrieval
tables the RAG services provision, and how the schema is created and migrated
are documented in :doc:`../guides/data_model`.

Custom persistence
------------------

To send runs or chat history elsewhere — to Redis, MongoDB or a managed
store — subclass ``AgentService`` and override the methods to be redirected
(``add_chat_message`` / ``get_chat_history`` for the conversation,
``initialize_workflow_run`` / ``update_run`` for runs), then pass your instance
to the engine. The engine only ever talks to the service's public methods, so
no engine changes are required.

Browsing it in the backoffice
-----------------------------

The **backoffice UI** is a separate service with its own database (it does not
share your agents' tables). It browses agent databases through **projects**: you
register a database — host, port and schema — as a project, and the backoffice
connects to it to show its sessions, runs, tasks and model calls. You can
register several (local, staging, production) behind one UI.

So point your ``AgentService`` at a database, add that database as a project,
and every session, run, task and model call becomes browsable — drill from a
*conversation* down to an individual *run*, *node* or *model call*, with token
and token metrics along the way. See :doc:`../ui/index`.

Where to next
-------------

* :doc:`../guides/data_model` — every table and column, in detail.
* :doc:`../guides/observability` — the concepts behind observability.
* :doc:`../ui/index` — browse sessions and interactions in the backoffice.
* :doc:`workflow` — the workflows whose runs all of this records.
* :doc:`architecture` — where persistence sits in the design as a whole.
