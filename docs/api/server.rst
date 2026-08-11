Agent Server API
================

:mod:`kavalai.server` serves a workflow over HTTP. A router built from a
:class:`~kavalai.WorkflowEngine` exposes the workflow's own input and output
types as the request and response schemas, so the endpoints are typed by the
graph rather than by hand:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Endpoint
     - What it does
   * - ``POST /run_agent``
     - Runs the workflow and returns the final output in one response.
   * - ``POST /stream_agent``
     - Runs the workflow and streams progress as Server-Sent Events; each frame
       is a :class:`~kavalai.workflow.models.WorkflowStreamEvent`.
   * - ``GET /workflow``
     - Returns the workflow graph (used by the backoffice to render it).
   * - ``GET /liveness`` / ``GET /health``
     - Liveness and readiness (readiness also checks the database).

Both run endpoints accept an optional ``session_id`` to continue a conversation,
or an ``external_id`` to let the caller key a session by its own identifier.
HTTP basic auth is optional and configured from the environment
(``KAVALAI_AGENT_BASIC_AUTH_USER`` / ``KAVALAI_AGENT_BASIC_AUTH_PASSWORD``).

Streaming a run over SSE
------------------------

``POST /stream_agent`` drives :meth:`~kavalai.WorkflowEngine.run_stream` and
renders each event as an SSE frame — ``event:`` carries the event type and
``data:`` its JSON payload. A ``: ping`` comment frame is emitted during silent
stretches (a long tool call, say) so proxies do not drop the connection.

Two consequences of SSE are worth planning for. A response cannot change its
status code once the headers are sent, so a failed run ends the stream after a
``workflow_failed`` event instead of returning an error status — clients must
treat that event as the failure signal. And disconnecting **aborts the run**:
closing the stream cancels the engine generator, which records the abort on the
run row.

Because the endpoint is a ``POST`` with a JSON body, browsers cannot consume it
with ``EventSource`` (which supports neither a request body nor auth headers) —
use ``fetch()`` with a streaming reader. From Python, use
:meth:`~kavalai.client.AgentClient.stream_agent`, which yields each ``data:``
payload as it arrives:

.. code-block:: python

   from kavalai.client import AgentClient

   client = AgentClient("http://localhost:8000")
   async for chunk in client.stream_agent(Message(message="Hi there")):
       print(chunk)

See :class:`~kavalai.workflow.models.WorkflowStreamEvent` for the event contract
and :doc:`/guides/workflows` for which nodes emit content events.

Server
------

.. automodule:: kavalai.server
