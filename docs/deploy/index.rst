Deployment
==========

A Kaval.AI deployment has up to three moving parts, and you may not need all
three:

* the **agent database** — the Postgres schema (or SQLite file) your runs,
  sessions, chat history and statistics are written to;
* the **agent server** — your workflow behind an HTTP endpoint (optional; a
  workflow can equally run inside your own application);
* the **backoffice** — the management and monitoring UI, with its own separate
  database.

The database is the only part that is genuinely required, and only if you want
persistence. PostgreSQL is the production choice. Both databases also accept a
``sqlite:///path`` URI, which suits a single machine — development, a demo, a
laptop — where the agent server and the backoffice share a filesystem; SQLite
on a network mount is not safe for a writer.

Local stack with Docker Compose
-------------------------------

The repository's ``docker-compose.yml`` brings up everything needed for
development:

.. code-block:: bash

   docker compose up postgres_db backoffice-migrations backoffice

That starts PostgreSQL with ``pgvector`` (on host port ``6543``), migrates the
backoffice schema, and serves the UI at ``http://localhost:8000``.

Add the runtime tables for your agents. There is no Compose service for this
step — the agent database belongs to the deployment that runs the workflow —
so run the migration set against the development instance directly:

.. code-block:: bash

   KAVALAI_DB_URI=postgresql://kavalai_dev:kavalai_dev@localhost:6543/kavalai_dev \
   KAVALAI_DB_SCHEMA=agents python -m kavalai.migrate_db agents

Optional services, for when you need them:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Service
     - What it is
   * - ``ollama``
     - Local model server on ``11434``, for ``ollama/…`` models.
   * - ``crawl4ai``
     - The crawler behind ``crawl_url`` / ``web_search`` on ``11235``.
   * - ``torproxy``
     - Tor + Privoxy, for ``http_request(use_proxy=True)``.

Migrations
----------

Schema changes are managed with Alembic, in two independent sets: one for the
agent runtime tables, one for the backoffice.

.. code-block:: bash

   # Agent runtime tables — reads KAVALAI_DB_URI and KAVALAI_DB_SCHEMA
   python -m kavalai.migrate_db agents

   # Backoffice tables — reads KAVALAI_BO_DB_URI and KAVALAI_BO_DB_SCHEMA
   python -m kavalai.migrate_db backoffice

Both are idempotent: run them on every deploy, before the service that uses
them. They are also what the ``agent-migrations`` and ``backoffice-migrations``
commands of the Docker images run. The runner retries for up to 60 seconds
while a database server refuses connections, so it can start alongside one.

The migrations use the runtime's own drivers: a ``postgresql://`` URI runs over
asyncpg and a ``sqlite://`` URI over aiosqlite, so ``kavalai[runtime]`` is all
a migration step needs and no synchronous Postgres driver is installed.
Alembic is synchronous; it runs on the connection's synchronous facade through
``AsyncConnection.run_sync``, which is Alembic's own recipe for async drivers.
A URI that names a synchronous driver, such as ``postgresql+psycopg2://``, is
run on that driver instead, provided it is installed, so its query parameters
(``sslmode``) keep their meaning.

The two schemas are independent and may live in the same Postgres instance
(``agents`` and ``backoffice`` by convention) or in different ones entirely. The
backoffice reaches an agent database through a **project** — see
:doc:`../ui/index`.

With a SQLite URI the schema variable is ignored (SQLite has no schemas) and
the same commands create the tables in the named file:

.. code-block:: bash

   KAVALAI_DB_URI=sqlite:///local_data/agents.db \
       python -m kavalai.migrate_db agents
   KAVALAI_BO_DB_URI=sqlite:///local_data/backoffice.db \
       python -m kavalai.migrate_db backoffice

Revision 0005 on a large database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Revision ``0005`` of the agents set adds ``session_id`` and ``run_id`` to
``model_call_stats``, creates eight indexes, drops the four single-column
indexes they replace, and sets each session's ``updated_at`` to the time of its
last run. That last step rewrites every session that has runs, once;
``sessions`` holds one row per conversation, far fewer than ``chat_messages``.

A plain ``CREATE INDEX`` blocks writes to its table while the index is built,
and every running conversation writes to ``chat_messages``. On a large
PostgreSQL database the six indexes on existing columns can therefore be built
beforehand, without blocking writes, under the names the revision uses — with
your schema in place of ``agents``:

.. code-block:: sql

   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_sessions_agent_id_external_id
       ON agents.sessions (agent_id, external_id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_sessions_agent_id_updated_at
       ON agents.sessions (agent_id, updated_at);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_sessions_updated_at
       ON agents.sessions (updated_at);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_chat_messages_session_id_created_at
       ON agents.chat_messages (session_id, created_at);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_chat_messages_agent_id_created_at
       ON agents.chat_messages (agent_id, created_at);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS
       ix_model_call_stats_agent_id_created_at
       ON agents.model_call_stats (agent_id, created_at);

``CONCURRENTLY`` cannot run inside a transaction block, so the statements are
issued one at a time, from ``psql`` for example. The revision creates every
index with ``IF NOT EXISTS`` and leaves these as they are. A concurrent build
that fails leaves an invalid index under its name, which the revision would
then keep; drop it and build it again before upgrading.

The indexes on the two new columns, ``ix_model_call_stats_session_id`` and
``ix_model_call_stats_run_id``, cannot be built in advance, because the columns
do not exist before the revision. The revision builds them itself and reads
``model_call_stats`` once for each. The task logger's writes to that table wait
meanwhile; runs do not, because the logger writes behind them. On SQLite the
revision runs as it is.

In earlier versions the OpenAI embedding client recorded the provider's whole
response in ``response_data``, the vectors included; it now keeps the model
name and the usage. The revision does not clear the rows already written.
Doing so is an ``UPDATE`` of what may be the largest table in the database,
and inside the migration's transaction it would lock that table for as long as
it takes. The statement is therefore left to the operator, to run after the
upgrade at a time of their choosing:

.. code-block:: sql

   update model_call_stats set response_data = null
   where call_type = 'embedding' and response_data is not null;

From an application's own migrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An application with migrations of its own can apply Kaval.AI's sets in the
same step, on a connection it has already opened. ``migrate`` from
``kavalai.migrate_db`` takes a synchronous SQLAlchemy ``Connection`` in place
of a URI:

.. code-block:: python

   from sqlalchemy import create_engine, inspect

   from kavalai.migrate_db import migrate

   engine = create_engine("sqlite:///app.db")
   with engine.begin() as connection:
       migrate("agents", connection=connection)
       print(", ".join(sorted(inspect(connection).get_table_names())))

.. code-block:: text

   agents, alembic_version, chat_messages, model_call_stats, runs, sessions, tasks

The set runs inside the transaction the connection already has. ``begin()``
commits it at the end of the block together with the application's own
changes, and a rollback undoes both. A connection with no transaction open is
given one, which ``migrate`` commits. ``schema=`` and the creation of the schema
behave as they do with a URI; the connection is not closed.

``migrate`` blocks until the upgrade is done, and with a URI it runs the async
driver on an event loop of its own. It therefore refuses to run where an event
loop is already running, and asks for ``migrate_async`` instead. That function
takes the same arguments, with an ``AsyncConnection`` as the connection:

.. code-block:: python

   import asyncio

   from sqlalchemy import inspect
   from sqlalchemy.ext.asyncio import create_async_engine

   from kavalai.migrate_db import migrate_async


   async def main():
       engine = create_async_engine("postgresql+asyncpg://user:pass@db/kavalai")
       async with engine.begin() as connection:
           await migrate_async("agents", connection=connection, schema="agents")
           tables = await connection.run_sync(
               lambda sync: inspect(sync).get_table_names(schema="agents")
           )
       await engine.dispose()
       print(", ".join(sorted(tables)))


   asyncio.run(main())

.. code-block:: text

   agents, alembic_version, chat_messages, model_call_stats, runs, sessions, tasks

Running the agent server
------------------------

The supported entry point reads its configuration from the environment:

.. code-block:: bash

   export KAVALAI_AGENT_WORKFLOW_PATH=/app/workflows/support_agent.yaml
   export KAVALAI_DB_URI=postgresql://user:pass@db:5432/kavalai
   export KAVALAI_DB_SCHEMA=agents
   export KAVALAI_AGENT_PORT=10000
   export OPENAI_API_KEY=sk-...

   python -m kavalai.server

Every variable is listed in :doc:`../reference/config`, and the endpoints in
:doc:`../tutorials/serving`.

Mounting it in your own app
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``python -m kavalai.server`` is a convenience, not the only way in. If you
already have a FastAPI application, mount the router wherever you like and keep
your own middleware, auth and lifespan:

.. code-block:: python

   from kavalai.server import create_agent_router
   from kavalai.workflow import WorkflowEngine

   engine = WorkflowEngine.from_yaml_path("support_agent.yaml")
   app.include_router(create_agent_router(engine), prefix="/agents/support")

If you do, connect the engine's tool servers at startup and release them at
shutdown — ``create_agent_app`` does this for you, a bare router does not:

.. code-block:: python

   @asynccontextmanager
   async def lifespan(app):
       await engine.connect()      # starts MCP servers, discovers their tools
       yield
       await engine.aclose()

One engine serves every request. That is the intended shape: it parses the
workflow once, keeps one set of tool-server connections, and each run does its
own token accounting.

.. warning::

   Authentication is **off** unless ``KAVALAI_AGENT_BASIC_AUTH_USER`` and
   ``KAVALAI_AGENT_BASIC_AUTH_PASSWORD`` are both set, and the server logs a
   warning at startup saying so. With it off, every endpoint is public —
   including ``GET /workflow``, which returns the workflow definition, prompts
   included. MCP server environment values are redacted from that response, but
   nothing else is.

A minimal image
^^^^^^^^^^^^^^^

If you are packaging your own workflow, an image is small:

.. code-block:: dockerfile

   FROM python:3.12-slim

   RUN pip install --no-cache-dir "kavalai[runtime]"

   COPY workflows/ /app/workflows/
   ENV KAVALAI_AGENT_WORKFLOW_PATH=/app/workflows/support_agent.yaml \
       KAVALAI_AGENT_HOST=0.0.0.0 \
       KAVALAI_AGENT_PORT=10000

   EXPOSE 10000
   CMD ["python", "-m", "kavalai.server"]

``runtime`` serves a workflow against hosted models and runs the migrations. A
workflow that embeds with a local ``fastembed/…`` model needs
``kavalai[runtime,fastembed]``, and one that uses ``crawl_url`` or
``web_search`` needs ``webtools`` as well. The repository's
``dockerfiles/agent.Dockerfile`` installs ``runtime`` and takes the extras as
the ``EXTRAS`` build argument (``--build-arg EXTRAS=runtime,fastembed``).

Point your orchestrator's liveness probe at ``GET /liveness`` and its readiness
probe at ``GET /health`` — the latter also checks the database, so a pod with a
broken connection is taken out of rotation instead of failing requests.

Running the backoffice
----------------------

The backoffice needs its own database and Google OAuth credentials:

.. code-block:: bash

   export KAVALAI_BO_DB_URI=postgresql://user:pass@db:5432/kavalai
   export KAVALAI_BO_DB_SCHEMA=backoffice
   export KAVALAI_BO_GOOGLE_CLIENT_ID=...
   export KAVALAI_BO_GOOGLE_CLIENT_SECRET=...
   export KAVALAI_BO_SESSION_SECRET_KEY=...
   export KAVALAI_BO_FRONTEND_URL=https://backoffice.example.com

   python -m kavalai.backoffice.server

Access is per project, with ``owner`` and ``viewer`` roles checked server-side
on every request. The last owner of a project cannot be removed or demoted.

Production checklist
--------------------

**Set ``KAVALAI_BO_SESSION_SECRET_KEY``.** The backoffice refuses to start
without it; there is no development fallback, so a cookie is never signed with
a key that is also in the documentation.

**Keep credentials out of the workflow YAML.** Use the ``url_env``,
``command_env``, ``username_env`` and ``password_env`` fields so the file can be
committed safely — see :doc:`../reference/yaml`.

**Protect the agent server.** Basic auth is enabled only when both
``KAVALAI_AGENT_BASIC_AUTH_USER`` and ``KAVALAI_AGENT_BASIC_AUTH_PASSWORD`` are
set; otherwise the endpoints are open. For anything else, mount the router in
your own app with your own dependency.

**Size the connection pool.** ``KAVALAI_DB_POOL_SIZE`` and
``KAVALAI_DB_MAX_OVERFLOW`` both default to ``0``. Raise them for a service
handling concurrent runs.

**Cache the embedding model.** If you use ``fastembed``, set
``FASTEMBED_CACHE_DIR`` to a mounted volume so each container start does not
re-download it.

**Watch the token counts.** Every run records its usage in
``model_call_stats``; the backoffice charts it. See :doc:`../guides/observability`.

**Decide how long conversations are kept.** Nothing is deleted on its own. A
scheduled job that calls ``AgentService.purge_sessions(cutoff)`` deletes the
sessions idle since before ``cutoff``, with their runs, tasks and chat
messages, and clears the payloads of their model calls while keeping the token
counts. See :ref:`observability-retention`.

A runtime role without DDL
^^^^^^^^^^^^^^^^^^^^^^^^^^

The agent tables are created by the migrations, which run before the service
and may run under a role of their own. The RAG tables are not: a RAG service
creates the registry, a collection's table and, on PostgreSQL, the ``vector``
extension on its first write, so by default its role needs the privilege to
create tables in the RAG schema. Where the runtime role must hold data
privileges only, the service is built with ``provision=False`` and the DDL is
left to a job that runs as the schema's owner — the migration step, or an
administrative task run when a collection is added:

.. code-block:: python

   from kavalai.rag import PostgresRagService

   MODEL = "fastembed/BAAI/bge-small-en-v1.5"

   owner = PostgresRagService.from_uri(OWNER_URI, MODEL, schema="rag")
   await owner.ensure_registry()
   await owner.create_collection("handbook", 384)

``create_collection`` takes the embedding dimension, and ``model=`` and
``vector_type=`` where the collection is to differ from the service's own.
``ensure_registry`` also runs ``CREATE EXTENSION IF NOT EXISTS vector``, which
does nothing once the extension exists; where the owner role may not create
extensions, an administrator creates it beforehand.

The runtime role then needs to reach the schema and to read, insert and delete
rows in the tables the owner creates, including those it creates later:

.. code-block:: sql

   GRANT USAGE ON SCHEMA rag TO kavalai_runtime;
   GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA rag
       TO kavalai_runtime;
   ALTER DEFAULT PRIVILEGES FOR ROLE kavalai_owner IN SCHEMA rag
       GRANT SELECT, INSERT, DELETE ON TABLES TO kavalai_runtime;

The service built for that role indexes into and queries the collections that
exist, and refuses to create one:

.. code-block:: python

   rag = PostgresRagService.from_uri(
       RUNTIME_URI, MODEL, schema="rag", provision=False
   )
   await rag.index(
       "The library is open on Tuesdays and Fridays.",
       collection_name="handbook",
       source_id="library",
   )
   hits = await rag.query(
       "When is the library open?", collection_name="handbook"
   )
   print(hits[0].model, round(hits[0].similarity, 3))

   try:
       await rag.index("The pub opens at noon.", collection_name="notices")
   except RuntimeError as error:
       print(error)

.. code-block:: text

   fastembed/BAAI/bge-small-en-v1.5 0.781
   RAG collection 'notices' does not exist, and this PostgresRagService was
   created with provision=False, so it issues no DDL. Create it with
   create_collection() from a role that may.

A collection whose table layout a newer ``kavalai`` changes is upgraded when a
provisioning service first opens it; a service with ``provision=False`` raises
instead. After an upgrade, the owner job therefore opens each collection once
— ``create_collection`` with an existing name and its dimension does — before
the runtime reaches it.

Where to next
-------------

* :doc:`../tutorials/serving` — the endpoints, streaming and mounting.
* :doc:`../reference/config` — every environment variable.
* :doc:`../ui/index` — the backoffice, project by project.
