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

Add the runtime tables for your agents:

.. code-block:: bash

   docker compose up agent-migrations

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
   python -m kavalai.migrate_db app

   # Backoffice tables — reads KAVALAI_BO_DB_URI and KAVALAI_BO_DB_SCHEMA
   python -m kavalai.migrate_db backoffice

Both are idempotent: run them on every deploy, before the service that uses
them. They are also what the ``agent-migrations`` and ``backoffice-migrations``
container commands run.

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

.. warning::

   The ``agent-server`` command in ``dockerfiles/agent.entrypoint.sh`` is
   currently **out of date** — it invokes a module path that no longer exists
   and reads different variable names. Use ``python -m kavalai.server`` as your
   container command until that is fixed; see :doc:`../todo`.

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
