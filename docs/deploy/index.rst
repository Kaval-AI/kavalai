Deployment
==========

A Kaval.AI deployment has up to three moving parts, and you may not need all
three:

* the **agent database** — the Postgres schema your runs, sessions, chat history
  and statistics are written to;
* the **agent server** — your workflow behind an HTTP endpoint (optional; a
  workflow can equally run inside your own application);
* the **backoffice** — the management and monitoring UI, with its own separate
  database.

The database is the only part that is genuinely required, and only if you want
persistence.

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
commands of the Docker images run.

The two schemas are independent and may live in the same Postgres instance
(``agents`` and ``backoffice`` by convention) or in different ones entirely. The
backoffice reaches an agent database through a **project** — see
:doc:`../ui/index`.

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

   RUN pip install --no-cache-dir "kavalai[common]"

   COPY workflows/ /app/workflows/
   ENV KAVALAI_AGENT_WORKFLOW_PATH=/app/workflows/support_agent.yaml \
       KAVALAI_AGENT_HOST=0.0.0.0 \
       KAVALAI_AGENT_PORT=10000

   EXPOSE 10000
   CMD ["python", "-m", "kavalai.server"]

Point your orchestrator's liveness probe at ``GET /liveness`` and its readiness
probe at ``GET /health`` — the latter also checks the database, so a pod with a
broken connection is taken out of rotation instead of failing requests.

Running the backoffice
----------------------

The backoffice needs its own database and Google OAuth credentials:

.. code-block:: bash

   export KAVALAI_BO_DB_URI=postgresql://user:pass@db:5432/kavalai
   export KAVALAI_BO_DB_SCHEMA=backoffice
   export GOOGLE_OAUTH_CLIENT_ID=...
   export GOOGLE_OAUTH_CLIENT_SECRET=...
   export SESSION_SECRET_KEY=...        # set this; the fallback is a dev value
   export FRONTEND_URL=https://backoffice.example.com

   python -m kavalai.backoffice.server

Access is per project, with ``owner`` and ``viewer`` roles checked server-side
on every request. The last owner of a project cannot be removed or demoted.

Production checklist
--------------------

**Set ``SESSION_SECRET_KEY``.** Without it the backoffice signs session cookies
with a well-known development value.

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

Where to next
-------------

* :doc:`../tutorials/serving` — the endpoints, streaming and mounting.
* :doc:`../reference/config` — every environment variable.
* :doc:`../ui/index` — the backoffice, project by project.
