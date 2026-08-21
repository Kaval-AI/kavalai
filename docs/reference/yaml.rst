Workflow YAML reference
=======================

A workflow file is a YAML document describing one
:class:`~kavalai.WorkflowGraph`. This page lists every key, exhaustively. For a
guided introduction read :doc:`../tutorials/workflow`; for the ideas behind the
model read :doc:`../guides/workflows`.

Load one with:

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml_path(
       "support_agent.yaml", agent_service=service
   )
   state = await engine.run({"user_message": "I want a refund"})

``from_yaml`` (a string) and ``from_dict`` (an already-parsed dict) take the same
keyword arguments.

Top level
---------

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Key
     - Required
     - Description
   * - ``name``
     - yes
     - Workflow name. Runs are recorded under this name as the agent.
   * - ``description``
     - no
     - Human-readable description. Shown in the backoffice.
   * - ``version``
     - no
     - Schema version. Defaults to ``"2.0"``.
   * - ``llm_model``
     - no
     - Default model for every ``llm`` and ``agent`` node, as
       ``provider/model``. A node may override it. If omitted here and on the
       node, the ``KAVALAI_DEFAULT_LLM_MODEL`` environment variable is used.
   * - ``llm_kwargs``
     - no
     - Default sampling/reliability options merged into
       :class:`~kavalai.LlmClientParameters` (``temperature``, ``top_p``,
       ``timeout_seconds``, …). Nodes may override individual keys.
   * - ``data_types``
     - yes
     - JSON-schema fragments compiled into Pydantic models. See below.
   * - ``nodes``
     - yes
     - The graph. Exactly one ``start`` node and at least one ``end`` node.
   * - ``rest_servers``
     - no
     - REST tool servers to register before the run.
   * - ``mcp_servers``
     - no
     - MCP tool servers to register before the run.
   * - ``python_functions``
     - no
     - Python tools to import and register by path.
   * - ``templates``
     - no
     - Named, reusable prompt fragments.

Validation happens when the graph is loaded, not when it runs. A workflow is
rejected if node names collide, if there is not exactly one ``start`` node, if
there is no ``end`` node, if two ``end`` nodes return different data types, if a
transition names a node that does not exist, or if a node writes to an ``output``
that is not declared in ``data_types``.

data_types
----------

Each entry is a JSON-schema object compiled into a Pydantic model, so every
value crossing a node boundary is validated.

Two names are special:

* ``input`` — the workflow's own input type; what the caller passes to ``run()``.
* the type named by the ``end`` node's ``output`` (``output`` by convention) —
  what the caller gets back.

.. code-block:: yaml

   data_types:
     input:
       type: object
       properties:
         user_message: {type: string}
     classification:
       type: object
       properties:
         intent: {type: string}
         confidence: {type: number}
       required: [intent]
     output:
       type: object
       properties:
         agent_response: {type: string}

.. tip::

   Field descriptions are worth writing. They are part of the schema sent to the
   model, so ``confidence: {type: number, description: "0.0-1.0"}`` measurably
   improves what comes back.

Nodes
-----

Every node has a ``name`` (unique) and a ``type``. The remaining keys depend on
the type.

start
^^^^^

Entry point; receives the workflow input. Exactly one per graph.

.. code-block:: yaml

   - {name: begin, type: start, next: classify}

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``next``
     - Name of the first node to run. Required.

end
^^^

Exit point. A graph may have several, but they must all name the same
``output`` type.

.. code-block:: yaml

   - {name: finish, type: end, output: output}

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``output``
     - Context variable returned to the caller. Defaults to ``output``.

llm
^^^

One structured LLM completion. The prompt is rendered, the model is called, and
the validated result is stored under ``output``.

.. code-block:: yaml

   - name: classify
     type: llm
     prompt: |
       Classify the villager's message as exactly one of: repair, permit, other.
       Respond with that single lowercase word in the `intent` field.
     inputs:
       message: {type: context, value: input}
     output: classification
     next: route
     use_history: false
     llm_model: openai/gpt-5.4-mini
     stream_output: true
     stream_delta: true

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``prompt``
     - Instruction text. Supports interpolation (see below). Required.
   * - ``inputs``
     - Mapping of local name → argument, each resolved before the call. See
       `Node inputs`_.
   * - ``output``
     - Data type/context variable the result is written to. Required.
   * - ``next``
     - Node to run afterwards. Required.
   * - ``use_history``
     - Replay this session's chat history into the call. Default ``true`` —
       which is what gives a chatbot memory across turns.
   * - ``llm_model``
     - Overrides the workflow default for this node.
   * - ``llm_kwargs``
     - Per-node sampling/reliability overrides.
   * - ``stream_output``
     - Emit this node's completion as ``partial`` events. Default ``false``.
   * - ``stream_delta``
     - Send only new text per ``partial`` instead of the accumulated value.
       Prefer it for long outputs. Default ``false``.

agent
^^^^^

A multi-step, tool-using :class:`~kavalai.Agent` loop inside the graph. Use it
when the model should decide which tools to call; use ``function`` when you
already know.

.. code-block:: yaml

   - name: research
     type: agent
     prompt: "Research the company and summarise what it sells."
     inputs:
       company: {type: context, value: input.company}
     output: summary
     allowed_tools: ["python://web.crawl", "rest://crm.*"]
     max_steps: 6
     next: write_up

Takes every ``llm`` key above except ``use_history``, plus:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``max_steps``
     - Maximum reasoning/tool-calling iterations. Default ``10``. This is the
       bound that stops a runaway loop.
   * - ``allowed_tools``
     - Tool URIs this node may use. Tools outside the list are neither
       described to the model nor callable. ``proto://server.*`` allows one
       whole server and ``"*"`` allows every registered tool; omitting the key
       means the same as ``"*"``, and ``[]`` means no tools at all. The values
       mean exactly what they do in the Python
       :class:`~kavalai.Agent` API.

       .. code-block:: yaml

          allowed_tools: ["*"]                         # every tool
          allowed_tools: ["mcp://github.*"]            # one server
          allowed_tools: ["python://lookup_resident"]  # one tool
          allowed_tools: []                            # none
   * - ``stream_instructions``
     - Stream each step's "thinking out loud" line as ``<node>_instructions``.
   * - ``stream_partials``
     - Stream raw per-step output as ``<node>_step<N>``. A debug firehose.

function
^^^^^^^^

Exactly one tool call through the :class:`~kavalai.FunctionKernel`, addressed by
URI.

.. code-block:: yaml

   - name: measure
     type: function
     tool: python://measure_pond
     inputs:
       name: {type: context, value: input.pond}
     output: reading
     next: phrase

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``tool``
     - Tool URI: ``python://name``, ``rest://server.tool`` or
       ``mcp://server.tool``. Required.
   * - ``inputs``
     - Arguments for the call, resolved like any other node inputs.
   * - ``output``
     - Where the (validated) return value is stored. Required.
   * - ``next``
     - Node to run afterwards. Required.
   * - ``method``
     - HTTP method for ``rest://`` tools. Default ``get``.

if
^^

Branches on a boolean condition, evaluated by the safe expression language.

.. code-block:: yaml

   - name: check_confidence
     type: if
     condition: "classification.confidence >= 0.8"
     then: reply
     else: escalate

``then`` and ``else`` are both required. Note ``else`` is a YAML key, not a
Python keyword — write it plainly.

switch
^^^^^^

Evaluates ``expr``, converts the result to a string, and matches it against
``cases``.

.. code-block:: yaml

   - name: route
     type: switch
     expr: classification.intent
     cases:
       repair: repair_reply
       permit: permit_reply
     default: general_reply

If no case matches and there is no ``default``, the run fails with a
:class:`~kavalai.WorkflowException`.

parallel
^^^^^^^^

Runs several independent branches concurrently and rejoins them at a single
node.

.. code-block:: yaml

   - name: gather
     type: parallel
     branches: [fetch_weather, fetch_news, fetch_stocks]
     next: summarise
     max_concurrency: 4

Each name in ``branches`` is the **entry node of a branch** — an ordinary
subgraph, walked exactly as the main graph is, up to but not including the join.
All branches start together, and the run resumes at ``next`` once every branch
has arrived there.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``branches``
     - Entry node of each branch. At least one, and no duplicates.
   * - ``next``
     - The join node. Every branch ends by transitioning to it.
   * - ``max_concurrency``
     - Optional cap on how many branches run at once. Omit it to run them all;
       set it when the branches share a rate-limited provider.

Each branch receives its own copy of the run context, so a node in one branch
cannot observe a sibling's output while both are running; outputs are merged
into the parent context at the join. Because branches must therefore be
independent, the graph validator rejects a workflow at load time if branch
subgraphs overlap, if two branches write the same ``output`` variable, or if a
branch contains an ``end`` node or re-enters the ``parallel`` node itself.

Loops, ``if`` / ``switch`` routing and nested ``parallel`` nodes inside a branch
are all permitted.

Three properties of a fan-out are worth knowing when reading a run back. Branch
events interleave onto the single event stream as they are produced, each
tagged with its own node name, so a streaming client can separate them. The
recorded trace, by contrast, collects each branch's nodes and appends them in
declaration order, so ``state.trace`` is stable across runs even though the
execution is not. And the first branch to raise cancels its siblings and
propagates, so a failed run does not leave a long branch running behind it.

Note that tool calls made *within* a single ``agent`` node already execute
concurrently; ``parallel`` concerns concurrency between nodes.

Node inputs
-----------

``inputs`` maps a local name to a value resolved before the node runs:

.. code-block:: yaml

   inputs:
     message:  {type: context, value: input.user_message}
     tone:     {type: literal, value: "formal"}
     previous: {type: history, value: last_order_id}

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - ``type``
     - Meaning
   * - ``context``
     - A dotted path into the current run's context: ``input``,
       ``input.user_message``, ``classification.intent``.
   * - ``literal``
     - The value exactly as written.
   * - ``history``
     - A value recorded in an **earlier run** of the same session. Requires an
       ``AgentService`` and a session.

In the :class:`~kavalai.WorkflowBuilder` a bare string is shorthand for a
context path — ``inputs={"message": "input"}``.

Interpolation in prompts
------------------------

A prompt may interpolate three prefixes:

.. code-block:: yaml

   prompt: |
     {{ templates.house_style }}

     The villager wrote: {{ context.input.user_message }}
     Their last order was {{ history.last_order_id }}.

* ``{{ context.PATH }}`` — the current run context.
* ``{{ templates.NAME }}`` — a fragment from the top-level ``templates`` list.
* ``{{ history.PATH }}`` — a value from an earlier run in the session.

Dicts and lists are inserted as JSON. An unresolvable reference raises rather
than rendering empty, so a typo fails loudly instead of silently weakening the
prompt.

.. note::

   This is a small, fixed substitution — not Jinja2. Only those three prefixes
   are recognised, and there are no loops, filters or conditionals. (The
   :class:`~kavalai.Agent`'s own system-prompt template *is* Jinja2; that is a
   different template.)

Tool servers
------------

Tools declared at the top level are registered on the engine's kernel before the
run, so ``function`` and ``agent`` nodes can address them.

.. code-block:: yaml

   python_functions:
     - {name: measure_pond, path: green_village.tools.measure_pond}

   rest_servers:
     - name: crm
       url: https://crm.example.com/api
       username_env: CRM_USER
       password_env: CRM_PASSWORD

   mcp_servers:
     - name: village
       command: python
       args: ["-m", "village_mcp"]

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Description
   * - ``python_functions[].name`` / ``.path``
     - Registered name, and the import path of a ``@pythontool`` function
       (``package.module.function``).
   * - ``rest_servers[].url`` / ``.url_env``
     - Base URL, given directly or read from an environment variable.
   * - ``rest_servers[].username_env`` / ``.password_env``
     - Environment variables holding basic-auth credentials.
   * - ``mcp_servers[].command`` / ``.command_env`` / ``.args`` / ``.env``
     - Command to launch a stdio MCP server, its arguments and extra
       environment.
   * - ``mcp_servers[].url`` / ``.url_env``
     - For an HTTP/SSE MCP server instead of a subprocess.

Use the ``*_env`` variants for anything secret: they keep credentials out of the
workflow file, which is usually committed to source control.

.. warning::

   ``mcp_servers[].env`` is the exception — it takes literal values, not
   variable names, so an API key written there sits in the workflow file. Those
   values are redacted from ``GET /workflow`` on the agent server, but the file
   itself is not protected. Prefer ``command_env`` and a wrapper script when a
   stdio server needs a credential.

MCP servers are started and asked for their tools when the engine connects, so
an ``agent`` node sees them on the first run. Call ``await engine.connect()``
at startup if you want a misconfigured server to fail there rather than
mid-run — see :doc:`../guides/tools`.

REST tools themselves are declared in code with
:meth:`~kavalai.FunctionKernel.register_rest_tool` (they need input and output
schemas); the YAML declares the *server*.

Execution limits
----------------

A run stops with a :class:`~kavalai.WorkflowException` after
``max_node_visits`` node visits — 1000 by default, set on the engine, not in
YAML. It is a backstop against a cycle that never terminates: loops in a graph
are allowed, so this is what makes them safe.

.. code-block:: python

   engine = WorkflowEngine.from_yaml_path("workflow.yaml", max_node_visits=50)

A complete example
------------------

.. literalinclude:: ../../examples/v2_workflow_support_agent.yaml
   :language: yaml
   :caption: examples/v2_workflow_support_agent.yaml
