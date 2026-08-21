Serving a workflow over HTTP
=============================

A workflow is useful to your application only once something can call it. The
agent server turns any :class:`~kavalai.WorkflowEngine` into a small FastAPI
service whose request and response schemas *are* the workflow's own ``input``
and ``output`` data types — so the API is generated from the graph rather than
maintained by hand next to it.

This tutorial starts a server, calls it, streams from it, adds authentication
and then mounts it inside an existing application.

Starting a server
-----------------

The quickest way is the built-in entry point, which reads its configuration from
the environment:

.. code-block:: bash

   export KAVALAI_AGENT_WORKFLOW_PATH=examples/v2_workflow_support_agent.yaml
   export KAVALAI_DB_URI=postgresql://user:pass@localhost:5432/kavalai
   export KAVALAI_DB_SCHEMA=agents
   export OPENAI_API_KEY=sk-...

   python -m kavalai.server

.. code-block:: text

   INFO | Loading workflow from examples/v2_workflow_support_agent.yaml.
   INFO | Starting agent <Support agent>.
   INFO: Uvicorn running on http://0.0.0.0:10000 (Press CTRL+C to quit)

The database must already be migrated — see :doc:`../deploy/index`. Host and
port come from ``KAVALAI_AGENT_HOST`` and ``KAVALAI_AGENT_PORT`` (default
``0.0.0.0:10000``); the full list is in :doc:`../reference/config`.

The endpoints
-------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Endpoint
     - What it does
   * - ``POST /run_agent``
     - Runs the workflow and returns the final output in one response.
   * - ``POST /stream_agent``
     - Runs it and streams progress as Server-Sent Events.
   * - ``GET /workflow``
     - Returns the workflow graph (the backoffice renders it from this).
   * - ``GET /liveness``
     - Liveness: is the process up?
   * - ``GET /health``
     - Readiness: is the process up *and* the database reachable?

.. code-block:: console

   $ curl -s localhost:10000/health
   {"status":"ok","database":"connected"}

One request, one answer
-----------------------

The workflow input goes in a ``data`` object. Two optional fields sit beside it:
``session_id`` continues an existing conversation, and ``external_id`` keys a
session by an identifier from your own system — a user, ticket or thread id.
Both address a row in the ``sessions`` table; see
:doc:`../guides/data_model`.

.. code-block:: bash

   curl -s -X POST localhost:10000/run_agent \
       -H 'Content-Type: application/json' \
       -d '{"data":
              {"user_message": "The church bell has been stuck since Tuesday."},
            "external_id": "villager-42"}'

.. code-block:: json

   {
     "session_id": "496e9b1a-7a69-486c-81be-cf30ec3581d6",
     "data": {
       "agent_response": "I'm sorry to hear that. If you want, I can help
                          you turn this into a clear report or notice…"
     }
   }

The response mirrors the request: your workflow's ``output`` type under ``data``,
plus the ``session_id`` the run belongs to. Send that id back — or reuse the same
``external_id`` — and the next call continues the same conversation, with the
chat history replayed into every ``llm`` node that has ``use_history`` on.

Because the schemas come from the graph, the generated OpenAPI docs at
``/docs`` describe your actual data types, and a malformed request is rejected
with a normal FastAPI validation error before the model is ever called.

Streaming a run
---------------

``POST /stream_agent`` takes the same body and streams
:class:`~kavalai.workflow.models.WorkflowStreamEvent` frames as they happen:

.. code-block:: bash

   curl -N -X POST localhost:10000/stream_agent \
       -H 'Content-Type: application/json' \
       -d '{"data":
              {"user_message": "Can I get a refund for the village hall booking?"}}'

.. code-block:: text

   event: workflow_started
   data: {"type":"workflow_started","name":"Support agent",
          "session_id":"79c1636c-…","run_id":"9d8ff6b4-…"}

   event: node_started
   data: {"type":"node_started","name":"classify"}

   event: node_completed
   data: {"type":"node_completed","name":"classify"}

   event: node_started
   data: {"type":"node_started","name":"handle_refund"}

   …

   event: workflow_completed
   data: {"type":"workflow_completed","name":"Support agent",
          "output_data":{…},"token_usage":{…}}

Lifecycle events always arrive. Token-by-token content only arrives from nodes
that opted in with ``stream_output`` — see :doc:`../reference/yaml`.

Three properties of SSE are worth planning for:

**A failed run still returns 200.** A response cannot change its status code
after the headers are sent, so a failure ends the stream with a
``workflow_failed`` event instead of an error status. Clients must treat that
event, not the status code, as the failure signal.

**Disconnecting aborts the run.** Closing the stream cancels the engine
generator, and the abort is recorded on the run row.

**Browsers cannot use ``EventSource``.** It supports neither a request body nor
auth headers, and this is a ``POST``. Use ``fetch()`` with a streaming reader.

A ``: ping`` comment frame is sent during silent stretches — a long tool call,
say — so proxies do not drop the connection.

Calling it from Python
----------------------

:class:`~kavalai.client.AgentClient` wraps both endpoints:

.. code-block:: python

   from kavalai.client import AgentClient

   client = AgentClient("http://localhost:10000")

   # The client reads the server's OpenAPI spec and rebuilds the workflow's
   # input and output models, so you do not redeclare them on the caller side.
   await client.discover_schemas()
   print(list(client.input_schema.model_fields))    # ['user_message']

   reply = await client.run_agent(
       client.input_schema(user_message="Is the pub open on Sundays?"),
       external_id="villager-7",
   )
   print(reply.agent_response)

   # Streamed call — each chunk is one event payload as a JSON string.
   async for chunk in client.stream_agent(
       client.input_schema(user_message="Tell me about the grain tower.")
   ):
       print(chunk)

Two conveniences are easily overlooked. ``run_agent`` and ``stream_agent`` take an
**instance of the agent's input model**, not a dict — build it from
``client.input_schema``, or pass your own matching model. And the client stores
the ``session_id`` from each response and sends it with the next call, so a
single ``AgentClient`` is one continuous conversation. Use a fresh client (or an
``external_id`` per user) when you need separate ones.

Pass ``username=`` and ``password=`` to the constructor for a server behind
basic auth.

Authentication
--------------

HTTP basic auth is enabled by setting both variables:

.. code-block:: bash

   export KAVALAI_AGENT_BASIC_AUTH_USER=village
   export KAVALAI_AGENT_BASIC_AUTH_PASSWORD=…

With neither set, the endpoints are open — which is fine behind an internal
gateway and not fine on the public internet.

For anything else — OAuth, an API key header, per-tenant rules — build the
router yourself and pass your own dependency.

Mounting it in your own app
---------------------------

``create_agent_router`` returns a plain ``APIRouter``, so a workflow can live
inside an existing FastAPI service, several workflows can be mounted side by
side under different prefixes, and you control auth and middleware:

.. code-block:: python

   from fastapi import FastAPI
   from kavalai import WorkflowEngine
   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.server import create_agent_router

   app = FastAPI()

   session_maker = db_manager.get_sessionmaker(
       uri="postgresql://…/kavalai", schema="agents"
   )
   service = AgentService(session_maker)

   support = WorkflowEngine.from_yaml_path(
       "support_agent.yaml", agent_service=service
   )
   triage = WorkflowEngine.from_yaml_path("triage.yaml", agent_service=service)

   app.include_router(
       create_agent_router(support, session_maker), prefix="/agents/support"
   )
   app.include_router(
       # Disable this router's auth and rely on the app's own.
       create_agent_router(triage, session_maker, auth_dependency=lambda: None),
       prefix="/agents/triage",
   )

``create_agent_app`` does the same for a standalone application, and
``create_app_from_env_conf`` is what ``python -m kavalai.server`` calls.

Where to next
-------------

* :doc:`../deploy/index` — Docker images, migrations and production settings.
* :doc:`../api/server` — the endpoint reference and event contract.
* :doc:`observability_storage` — what each call records, and where.
* :doc:`../ui/index` — watch the runs arrive in the backoffice.
