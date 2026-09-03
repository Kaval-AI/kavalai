===========================
The database and data model
===========================

Kaval.AI persists everything a run does, and it persists it in a database you
own. This page documents that store: which tables exist, what each one is for,
how they relate, and how the schema is created and evolved.

Two stores are described here, and they are independent of one another:

* the **agent runtime store**, written by
  :class:`~kavalai.agent_service.AgentService` and the ``TaskLogger`` while a
  workflow executes;
* the **retrieval store**, provisioned by the RAG services when documents are
  indexed.

The backoffice keeps a third database of its own — users, projects and
memberships — which is a deployment concern rather than part of the runtime
data model and is therefore excluded from this discussion. See :doc:`../ui/index`
for that interface.

Design principles
=================

Four decisions explain the shape of the schema, and it is easier to read the
tables once they are stated.

**The runtime owns its data.** There is no hosted collector and no vendor
account. ``AgentService`` writes into an ordinary SQLAlchemy session maker, so
the runtime store is whichever database that session maker points at. The same
ORM models produce the same tables under SQLite and under PostgreSQL, which is
what allows a notebook, a test suite and a production deployment to be examined
with identical queries.

**The models are schema-less.** No table declares a schema qualifier. The
target schema is applied per engine through SQLAlchemy's
``schema_translate_map`` (``DatabaseManager.get_sessionmaker(..., schema=...)``),
so a single set of models serves many tenants in one PostgreSQL instance, and
library code never reads configuration from the environment. Only the
entry-point ``main()`` functions do that.

**Structured payloads are stored as JSON.** ``input_data``, ``output_data``,
``context``, ``inputs``, ``output``, ``request_data`` and ``response_data`` all
hold the JSON serialisation of validated Pydantic models. The validation has
already happened at the node boundary (see :doc:`../tutorials/architecture`);
the database records the result rather than re-imposing a relational shape on
values whose schema is defined by the workflow rather than by the library.

**Usage is recorded, cost is not.** ``model_call_stats`` stores token counts
and never a monetary amount. The reasoning is set out in
:doc:`observability`; in short, providers report tokens rather than money, and
cached input tokens are billed differently enough that a figure derived from
``prompt_tokens`` alone would be wrong rather than merely stale.

The agent runtime store
=======================

Six tables record execution. They form a strict hierarchy, with
``model_call_stats`` attached loosely to the side::

    agents
      └── sessions                 (one conversation)
            ├── runs               (one workflow invocation)
            │     ├── tasks        (one node execution)
            │     └── chat_messages
            └── chat_messages

    model_call_stats               (one provider call; references agent_id)

Deletion cascades down this hierarchy: removing an agent removes its sessions,
and removing a session removes its runs, tasks and messages. ``tasks.agent_id``
and ``chat_messages.run_id`` are nullable and use ``ON DELETE SET NULL``, so a
partial deletion leaves the surviving rows readable rather than removing them.

``agents``
----------

One row per configured agent, defined by
:class:`kavalai.db.Agent`.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Purpose
   * - ``id``
     - Primary key; referenced by every other table in the hierarchy.
   * - ``name``, ``description``
     - Human-readable identification, shown in the backoffice listings.
   * - ``input_schema``, ``output_schema``
     - The JSON schemas the agent exposes at its boundary. They allow a caller
       to discover the contract without loading the workflow.
   * - ``workflow``
     - The workflow document itself, stored as JSON. This is what makes an
       agent row self-contained: the graph that produced a run can be recovered
       from the same database as the run.
   * - ``created_at``, ``updated_at``
     - Timestamps, maintained by the ORM.

``initialize_workflow_run`` gets or creates the agent row at the start of a
run, so an agent row appears the first time a workflow executes; registration
is not a separate step.

``sessions``
------------

One row per conversation with an agent (:class:`kavalai.db.Session`). A session
is the unit across which memory persists: chat history is read back per
session, and ``history:`` inputs resolve against it.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Purpose
   * - ``agent_id``
     - The agent this conversation belongs to.
   * - ``external_id``
     - An optional identifier from the calling system — a user id, ticket
       number or thread id. Supplying the same ``external_id`` again reuses the
       existing session, which is how a stateless HTTP caller continues a
       conversation without holding a Kaval.AI identifier.

``runs``
--------

One row per workflow invocation (:class:`kavalai.db.Run`). A session may
contain many runs; each run is a single traversal of the graph from ``start``
to an ``end`` node.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Purpose
   * - ``session_id``
     - The conversation this invocation belongs to.
   * - ``input_data``
     - The validated input the workflow was called with.
   * - ``output_data``
     - The validated output it produced. It remains ``NULL`` for a run that
       failed before reaching an ``end`` node, which is how incomplete runs are
       identified.
   * - ``context``
     - The resolved :class:`~kavalai.RunContext` — every value each node saw.
       This is the field that makes a run reproducible: the inputs to any node
       can be reconstructed from it without re-executing the graph.

The row is written when the run begins (``initialize_workflow_run``) and
completed when it ends (``update_run``). There is no intermediate checkpoint,
which is why durable resume is not offered: a crashed process loses the run.

``tasks``
---------

One row per node execution within a run (:class:`kavalai.db.Task`). Where
``runs`` records what the workflow was asked and what it answered, ``tasks``
records how it got there.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Purpose
   * - ``run_id``, ``session_id``, ``agent_id``
     - The execution this node belonged to. The redundant ``session_id`` and
       ``agent_id`` make per-agent and per-session aggregation a single indexed
       query rather than a join through ``runs``.
   * - ``name``, ``node_type``
     - Which node executed, and of which kind (``start``, ``end``, ``llm``,
       ``agent``, ``function``, ``rag_query``, ``if``, ``switch``,
       ``parallel``). The tool calls an ``agent`` node makes are rows of their
       own, with ``node_type`` ``tool_call``.
   * - ``inputs``, ``output``
     - The values the node received and produced, after validation.
   * - ``prompt``
     - The prompt as it was actually rendered, with every interpolation
       resolved. A prompt that behaved unexpectedly can therefore be read
       exactly as the model received it.
   * - ``errors``
     - Any errors raised by the node, as a list of strings.
   * - ``duration_seconds``
     - Wall-clock duration, which is what identifies the slow step in a graph.
   * - ``seq``
     - Position of the row in the run's execution order.
   * - ``parent_task_name``
     - The node that produced the row, set on the ``tool_call`` rows an
       ``agent`` node emits.
   * - ``tool_uri``
     - The tool the row executed, set by ``function`` nodes and by agent tool
       calls alike.

Ordering ``tasks`` by ``seq`` within a run reproduces the execution trace,
including the interleaving of concurrent ``parallel`` branches, which
``created_at`` cannot order reliably. See :doc:`observability` for how the
three columns are read.

``chat_messages``
-----------------

The conversation transcript (:class:`kavalai.db.ChatMessage`): one row per
message, with ``role`` (``user``, ``assistant``, ``system``) and ``content``.
Rows belong to a session and optionally reference the run that produced them,
so a transcript can be read as a conversation or attributed run by run.

This table is the memory a chatbot has across turns. Nodes that declare
``use_history`` read it, and the engine appends to it as the run proceeds.

``model_call_stats``
--------------------

One row per call to a provider (:class:`kavalai.db.ModelCallStat`), covering
both completions and embeddings. Unlike the tables above it is not written by
the engine but by the LLM clients themselves, through the
``ModelStatsReceiver`` interface, which is why it records calls made outside a
workflow as well.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Purpose
   * - ``call_type``, ``model``
     - What kind of call it was, and against which ``provider/model``.
   * - ``agent_id``
     - The agent on whose behalf the call was made, where one applies. It is a
       plain indexed column rather than a foreign key, so statistics survive
       independently of the agent row.
   * - ``request_data``, ``response_data``
     - The payloads exchanged with the provider.
   * - ``response_code``
     - The provider's HTTP status. Attempts that returned no completion are
       recorded with their status and error text in place of token counts, so a
       rate-limit episode or an outage leaves evidence rather than an
       implausibly healthy table.
   * - ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
     - Usage as reported by the provider.
   * - ``cached_prompt_tokens``
     - The subset of ``prompt_tokens`` served from the provider's cache, which
       is billed at a fraction of fresh input.
   * - ``reasoning_tokens``
     - The subset of ``completion_tokens`` spent on reasoning the caller never
       sees.
   * - ``batch_size``
     - The number of items in an embedding batch.
   * - ``duration_seconds``
     - Latency of the call.

The two token-detail columns are what make cost computable downstream — with
``genai-prices`` or an equivalent price table — from a row that the library
itself does not have to keep current.

Reading the store back
======================

Everything above is queryable through ``AgentService`` without writing SQL:

.. code-block:: python

   from uuid import UUID

   history = await service.get_chat_history(UUID(state.session_id))
   stats = await service.get_model_call_stats(call_type="llm", limit=20)

For traversals the service does not expose — the runs of a session, the tasks
of a run — the ORM models are ordinary SQLAlchemy classes and
``kavalai.crud`` provides generic ``get_all`` / ``get_one`` helpers over them.
The backoffice interface is built on the same queries. See
:doc:`../tutorials/observability_storage` for a worked example, and
:doc:`../ui/index` for the interface.

The retrieval store
===================

Retrieval-augmented generation uses its own storage, provisioned by the RAG
service rather than by the migrations. This separation is deliberate: an index
has a different lifecycle from a run log, is frequently rebuilt, and may live
in an entirely different database.

One storage model, two databases
--------------------------------

Both RAG services share the storage model defined in
:mod:`kavalai.rag.collections`, and maintain it themselves:

* ``rag_collections`` — the registry. One row per collection, holding its
  ``name``, the ``table_name`` its vectors live in, the embedding ``model``,
  the ``embedding_size`` and a ``schema_version``. It exists because the
  dimension of a vector column is fixed at creation, so a collection's model
  must be recorded alongside it; indexing the same collection with a different
  dimension is rejected rather than silently corrupting the index.
* **one table per collection**, named deterministically from the collection
  name (a readable slug plus a short hash to keep distinct names distinct).
  Each row holds ``id``, ``source_id``, ``content``, an ``embedding`` of the
  registered dimension, ``metadata`` and two timestamps.

One table per collection rather than one shared table means each collection has
its own typed vector column and its own index, so a scan never crosses
collections and a collection is dropped by dropping a table. The registry is
also what the backoffice RAG explorer reads, so it shows the same view of an
index whichever database holds it. Only the column types differ:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Column
     - :class:`~kavalai.rag.PostgresRagService`
     - :class:`~kavalai.rag.SqliteRagService`
   * - ``id``
     - ``UUID``
     - ``TEXT`` (the UUID's canonical form)
   * - ``embedding``
     - ``vector(N)`` from ``pgvector``, with an HNSW index under cosine
       distance
     - ``BLOB`` of 32-bit floats, scanned by the ``sqlite-vector`` extension
       under cosine distance
   * - ``metadata``
     - ``JSONB``, with a GIN index
     - ``TEXT`` holding JSON
   * - timestamps
     - ``TIMESTAMPTZ``
     - ``TEXT`` in ISO 8601

The PostgreSQL service creates the ``vector`` extension on first use. The
SQLite service keeps registry and collections in one ordinary file, which
needs no server and can be copied wherever it is needed; the file written by
``kavalai`` 1.0, with a single ``rag_index`` table, is refused with a message
asking for the index to be rebuilt.

``source_id`` and ``rag_metadata`` behave identically in both backends, so a
retrieval written against one runs unchanged against the other. See
:doc:`../tutorials/rag`.

How the schema is created
=========================

Three mechanisms exist, for three deployments.

**Alembic, for PostgreSQL.** The migrations live in
``kavalai/migrations/agents/`` and are run with:

.. code-block:: bash

   python -m kavalai.migrate_db agents

The ORM models are the single source of truth; revisions are autogenerated
against an empty scratch database and a parity test fails if models and
revisions diverge. Because the models carry no schema qualifier, the target
schema is supplied by the runner and applied through ``schema_translate_map``.
Two consequences follow, and both have caused defects worth recording: raw SQL
bypasses the translation and must qualify its schema explicitly, and reflection
does too, so ``op.batch_alter_table`` must be given the translated schema or it
looks for the table in ``public``.

**``create_all``, for local SQLite.** ``DatabaseManager.init_sqlite()`` creates
the tables directly. This is the path used by tests and notebooks, where a
migration history serves no purpose.

**A version stamp, for the browser.** Pyodide cannot run Alembic, so the
in-browser store is created with ``create_all`` and stamped with
``PRAGMA user_version``. When the stamp does not match
``SQLITE_SCHEMA_VERSION`` the database is discarded and recreated. Any schema
change must therefore bump that constant, or browsers will keep using a store
that no longer matches the models.

Where to next
=============

* :doc:`../tutorials/architecture` — how the components that write these tables
  fit together.
* :doc:`observability` — what to do with the recorded data, and why cost is
  absent.
* :doc:`../tutorials/observability_storage` — a runnable tour of the same
  store.
* :doc:`../ui/index` — the interface that reads it.
