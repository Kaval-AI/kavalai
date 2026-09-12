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

   export KAVALAI_AGENT_WORKFLOW_PATH=examples/support_agent/support_agent.yaml
   export KAVALAI_DB_URI=postgresql://user:pass@localhost:5432/kavalai
   export KAVALAI_DB_SCHEMA=agents
   export OPENAI_API_KEY=sk-...

   python -m kavalai.server

.. code-block:: text

   INFO | Loading workflow from examples/support_agent/support_agent.yaml.
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
     "session_id": "738e1f1a-1b7e-4c92-b0ba-8ffb8b997372",
     "data": {
       "agent_response": "It sounds like the church bell has been out of
                          order since Tuesday. If you need to report it,
                          you could say: “The church bell has been stuck
                          since Tuesday.” If you want, I can also help you
                          turn that into a clearer maintenance report…"
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
          "session_id":"0c4b8f61-…","run_id":"b2f72a1c-…"}

   event: node_started
   data: {"type":"node_started","name":"begin"}

   event: node_completed
   data: {"type":"node_completed","name":"begin"}

   event: node_started
   data: {"type":"node_started","name":"classify"}

   event: node_completed
   data: {"type":"node_completed","name":"classify"}

   event: node_started
   data: {"type":"node_started","name":"route"}

   event: node_completed
   data: {"type":"node_completed","name":"route"}

   event: node_started
   data: {"type":"node_started","name":"handle_refund"}

   event: node_completed
   data: {"type":"node_completed","name":"handle_refund"}

   event: node_started
   data: {"type":"node_started","name":"finish"}

   event: node_completed
   data: {"type":"node_completed","name":"finish"}

   event: workflow_completed
   data: {"type":"workflow_completed","name":"Support agent",
          "session_id":"0c4b8f61-…","output_data":{…},"token_usage":{…}}

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

Public streams
^^^^^^^^^^^^^^

The stream above is the operator's view. It names every node, reports token
usage, and a failed run's ``workflow_failed`` carries the exception text,
which can quote a provider's error message. A server that people other than
the operator reach — a chat widget on a website — serves the stream through
:func:`~kavalai.server.public_events` instead. It drops ``node_started`` and
``node_completed``, removes ``token_usage`` and the reason attached to a
``restart``, and replaces a failure's text with a fixed message quoting the run
id, so a report can be matched to the recorded run while the detail stays in
the server log and on the run row. Which nodes stream content remains the
workflow's ``stream_output``.

.. code-block:: python

   from fastapi.testclient import TestClient

   from kavalai import WorkflowEngine
   from kavalai.agent_service import AgentService
   from kavalai.db import DatabaseManager
   from kavalai.server import create_agent_app, public_events
   from kavalai.testing import ScriptedLlmClient

   WORKFLOW = """
   name: Village help
   llm_model: openai/gpt-5.6-luna
   data_types:
     input: {type: object, properties: {user_message: {type: string}}}
     output: {type: object, properties: {agent_response: {type: string}}}
   nodes:
     - {name: begin, type: start, next: reply}
     - name: reply
       type: llm
       prompt: Answer the villager in one sentence.
       inputs: {input: {type: context, value: input}}
       output: output
       next: finish
       stream_output: true
     - {name: finish, type: end, output: output}
   """

   model = ScriptedLlmClient(
       [
           {"agent_response": "The hall is free on Saturday."},
           RuntimeError("Error code: 401 - invalid x-api-key sk-…"),
       ],
       chunk_size=64,
   )
   engine = WorkflowEngine.from_yaml(
       WORKFLOW,
       client_factory=model,
       agent_service=AgentService(
           DatabaseManager().get_sqlite_compat_sessionmaker(db_path="agents.db")
       ),
   )
   app = create_agent_app(
       engine, auth_dependency=lambda: None, event_filter=public_events
   )
   client = TestClient(app)

   for question in ["Is the hall free?", "And on Sunday?"]:
       response = client.post(
           "/stream_agent", json={"data": {"user_message": question}}
       )
       print(response.text)

.. code-block:: text

   event: workflow_started
   data: {"type":"workflow_started","name":"Village help",
          "session_id":"24526583-…","run_id":"c35d3163-…"}

   event: partial
   data: {"type":"partial","name":"reply",
          "value":"{\"agent_response\": \"The hall is free on Saturday.\"}"}

   event: complete
   data: {"type":"complete","name":"reply",
          "value":"{\"agent_response\": \"The hall is free on Saturday.\"}"}

   event: workflow_completed
   data: {"type":"workflow_completed","name":"Village help",
          "session_id":"24526583-…","run_id":"c35d3163-…",
          "output_data":{"agent_response":"The hall is free on Saturday."}}

   event: workflow_started
   data: {"type":"workflow_started","name":"Village help",
          "session_id":"04e02631-…","run_id":"99be2cbf-…"}

   event: workflow_failed
   data: {"type":"workflow_failed","name":"Village help",
          "value":"The agent could not complete this request. Reference:
                   99be2cbf-8833-4ca2-bfc4-a185d7b7cba8.",
          "session_id":"04e02631-…","run_id":"99be2cbf-…"}

The second question's model call failed with a provider error that quoted a
key; the client received the reference and nothing else. Under
``python -m kavalai.server`` the same filter is switched on with
``KAVALAI_AGENT_PUBLIC_EVENTS=true``, and ``KAVALAI_AGENT_RUN_TIMEOUT_SECONDS``
bounds how long any run may take (see :doc:`../reference/config`).

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

Two conveniences are worth noting. ``run_agent`` and ``stream_agent`` take an
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

A route of your own
^^^^^^^^^^^^^^^^^^^

A host whose chat route must first check something the router cannot know —
a site token, a budget, a rate limit — writes the route itself and keeps the
protocol. :class:`~kavalai.server.AgentRequest` is the request body the router
accepts, typed by the workflow's input, and
:func:`~kavalai.server.sse_response` returns the same frames, keepalive pings
and headers as ``/stream_agent``, with an optional event filter and headers of
the host's own:

.. code-block:: python

   import json

   from fastapi import FastAPI, Header, HTTPException
   from fastapi.testclient import TestClient
   from pydantic import BaseModel

   from kavalai.server import AgentRequest, public_events, sse_response


   class Question(BaseModel):
       user_message: str


   app = FastAPI()


   @app.post("/chat")
   async def chat(body: AgentRequest[Question], x_site_token: str = Header("")):
       if x_site_token != "village":
           raise HTTPException(status_code=403, detail="Unknown site")
       events = engine.run_stream(
           body.data.model_dump(), external_id=body.external_id
       )
       return sse_response(
           events,
           event_filter=public_events,
           headers={"X-Conversation": body.external_id or ""},
       )


   client = TestClient(app)
   body = {"external_id": "visitor-17", "data": {"user_message": "Market day?"}}

   refused = client.post("/chat", json=body)
   print(refused.status_code, refused.json())

   answered = client.post(
       "/chat", json=body, headers={"X-Site-Token": "village"}
   )
   print(answered.status_code, answered.headers["x-conversation"])
   last = [l for l in answered.text.splitlines() if l.startswith("data:")]
   print(json.loads(last[-1][len("data: ") :])["output_data"])

.. code-block:: text

   403 {'detail': 'Unknown site'}
   200 visitor-17
   {'agent_response': 'The market is on Friday.'}

Here ``engine`` answers from a scripted model, as in the previous section. The
admission check runs before the engine is touched, so a refused request costs
nothing. There is deliberately no router option that picks an engine per
request: the router's request and response schemas are generated from one
engine's graph, and :class:`~kavalai.client.AgentClient` and ``kavalai-eval``
discover them from the OpenAPI schema, so an engine chosen per request would
leave the schema untyped. Several engines are served by mounting several
routers, as above.

Where to next
-------------

* :doc:`../deploy/index` — Docker images, migrations and production settings.
* :doc:`../api/server` — the endpoint reference and event contract.
* :doc:`observability_storage` — what each call records, and where.
* :doc:`../ui/index` — watch the runs arrive in the backoffice.
