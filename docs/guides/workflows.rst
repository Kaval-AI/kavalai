==========
Workflows
==========

A **workflow** is the backbone of Kaval.AI: a directed graph — a small state
machine — that turns an input into an output by walking from a ``start`` node to
an ``end`` node. Workflows make agentic pipelines *predictable*: every step is
named, every edge is explicit, and every value that crosses a node boundary is
typed and validated.

You author a workflow in two equivalent ways: declaratively in **YAML**, or
programmatically with the fluent :class:`~kavalai.WorkflowBuilder`. Both compile
to the same :class:`~kavalai.WorkflowGraph` and run on the same
:class:`~kavalai.WorkflowEngine`.

For a hands-on walkthrough see :doc:`../tutorials/workflow`; for every key,
field by field, see :doc:`../reference/yaml`.

The graph model
---------------

A workflow consists of three elements:

* a **name**,
* a set of **data_types**, and
* a set of **nodes**.

The caller hands an input to the ``start`` node and reads the result off the
``end`` node. By convention the input is the data type named ``input``, and the
returned value is the ``output`` variable named on the ``end`` node.

The arrows are the transitions: the output of one node flows into the next.
Branch nodes choose between several arrows, each labelled with the condition
that selects it — a ``switch`` here routes a classified request to one of three
handlers:

.. image:: /_static/workflows/support-agent.svg
   :alt: A support-agent workflow: begin → classify → a switch routing to
         handle_technical, handle_refund or handle_general, all ending at finish.
   :align: center

Data types and validation
--------------------------

``data_types`` are JSON-schema fragments that Kaval.AI compiles into Pydantic
models. Because every node input and output is described by one of these models,
the engine validates data at each boundary — a node can never silently receive
or emit a malformed value. This typed-I/O guarantee is one of the pillars of
:doc:`safety` in Kaval.AI.

Node types
----------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Node
     - What it does
   * - ``start``
     - Entry point; receives the workflow input.
   * - ``end``
     - Exit point; names the ``output`` variable to return.
   * - ``llm``
     - One structured LLM completion.
   * - ``agent``
     - A multi-step, tool-using agent loop bounded by ``max_steps`` (see :doc:`agents`).
   * - ``function``
     - A single tool call through the FunctionKernel, addressed by URI (see :doc:`tools`).
   * - ``if``
     - Branches on a boolean ``condition``.
   * - ``switch``
     - Evaluates ``expr``, stringifies it, matches against ``cases``, else ``default``.

The corresponding node classes — :class:`~kavalai.StartNode`,
:class:`~kavalai.EndNode`, :class:`~kavalai.LLMNode`,
:class:`~kavalai.AgentNode`, :class:`~kavalai.FunctionNode`,
:class:`~kavalai.IfNode`, :class:`~kavalai.SwitchNode` — are importable from the
top-level :mod:`kavalai` package.

Context and interpolation
--------------------------

As a workflow runs, values accumulate in a shared **context**. Inside a prompt
you interpolate from it with ``{{ context.<path> }}``. A node input is written
as ``{type: context, value: <path>}``; in the :class:`~kavalai.WorkflowBuilder`
a bare string input is treated as a context path. Branching nodes (``if`` /
``switch``) read the context through the safe expression language described in
:doc:`safety`.

YAML vs. WorkflowBuilder
------------------------

The :class:`~kavalai.WorkflowBuilder` mirrors the YAML structure with chainable
methods — ``data_type``, ``start``, ``end``, ``llm``, ``agent``, ``function``,
``if_``, ``switch`` — each returning ``self``. Finish with ``build()`` for a
:class:`~kavalai.WorkflowGraph` or ``build_engine()`` for a ready
:class:`~kavalai.WorkflowEngine`.

To load and run from YAML:

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml(yaml, agent_service=..., task_logger=...)
   state = await engine.run({...})

Other constructors include ``WorkflowEngine.from_yaml_path`` and
``WorkflowEngine.from_dict``.

Streaming a run
---------------

Every run is a stream. :meth:`~kavalai.WorkflowEngine.run_stream` yields
:class:`~kavalai.workflow.models.WorkflowStreamEvent` objects as the graph
executes, and :meth:`~kavalai.WorkflowEngine.run` is that stream drained
to completion — there is one execution path, so a streamed run and a blocking
run behave identically:

.. code-block:: python

   async for event in engine.run_stream({"user_message": "I want a refund"}):
       print(event.type, event.name, event.value)

Lifecycle events frame the run; nodes with streaming enabled contribute content
events in between:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event type
     - When it arrives
   * - ``workflow_started``
     - Once, at the start; carries ``session_id`` and ``run_id``.
   * - ``node_started`` / ``node_completed``
     - Around each visited node; ``name`` is the node name.
   * - ``partial`` / ``complete``
     - Streamed content from a node that opted in (see below).
   * - ``restart``
     - A retried LLM call is starting its stream over — discard what you have
       accumulated under that ``name``, it will be re-sent.
   * - ``workflow_completed``
     - Once, on success; carries ``output_data`` and ``token_usage``.
   * - ``workflow_failed``
     - Once, on failure; the error message is in ``value``. Yielded *before* the
       :class:`~kavalai.WorkflowException` is raised to the caller.

Streaming is opt-in per node, because most nodes are not worth streaming. An
``llm`` node takes ``stream_output`` to stream its completion, and an ``agent``
node adds ``stream_instructions`` (each step's "thinking out loud" line) and
``stream_partials`` (a debug firehose of raw step output). The node's own output
streams under the node name; auxiliary streams are prefixed with it —
``<node>_thought``, ``<node>_instructions``, ``<node>_step<N>``.

``stream_delta`` chooses what a ``partial`` carries. Left off, each partial holds
the full accumulated, safe-parsed value so far: render-ready with no client-side
assembly, but the whole buffer goes over the wire on every chunk. Set it and each
partial carries only the new text, which the client reassembles — prefer that for
long outputs. Both YAML and the :class:`~kavalai.WorkflowBuilder` accept these
flags:

.. code-block:: python

   .llm("reply", prompt=prompt, output="output", next="end",
        stream_output=True, stream_delta=True)

To serve a stream over HTTP, use the agent server's ``POST /stream_agent``
endpoint, which renders these events as Server-Sent Events — see
:doc:`/api/server`.

The WorkflowState
-----------------

Every run produces a :class:`~kavalai.WorkflowState`, which is both the result
and the audit trail:

* ``status`` — terminal state of the run.
* ``trace`` — ordered list of visited node names.
* ``data`` — the full context.
* ``input_data`` / ``output_data`` — the values in and out.
* ``run_id`` / ``session_id`` / ``invocation_id`` — identifiers; the 8-char
  ``invocation_id`` prefixes every log line of the run.
* ``token_usage`` — ``model_calls``, ``prompt_tokens``, ``completion_tokens``,
  ``total_tokens``.

The state is serialisable via ``state.to_json()`` and
``WorkflowState.from_json()``, and the run is recorded through the
:class:`~kavalai.agent_service.AgentService` — so it can be reloaded and
inspected later, including runs that ended early because a streaming client
disconnected. See :doc:`observability` for how this powers the backoffice UI,
and :doc:`data_model` for the ``runs`` and ``tasks`` rows it becomes.
