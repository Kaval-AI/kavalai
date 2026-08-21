Open questions
==============

Things found while writing this documentation that need a decision or a code
change rather than a documentation fix. Each entry says what was observed, how
it was verified, and what the docs currently say about it.

.. note::

   This page is a working list for the maintainers, not user-facing guidance.

Bugs
----

MCP tools are invisible to an agent until something calls one
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``FunctionKernel.register_mcp_server`` only records the server. The process is
started and ``list_tools`` is called inside ``_get_mcp_session``, which runs on
the **first tool call** to that server. So ``get_tool_descriptions()`` returns
``[]``, and an :class:`~kavalai.Agent` (or an ``agent`` node) handed a
freshly-registered MCP server is told it has no tools and answers without them —
silently, with a plausible answer.

Verified: registering a stdio MCP server and calling ``get_tool_descriptions()``
returns ``[]``; after one direct ``call_tool("mcp://…")`` the same call lists the
tools. ``WorkflowEngine`` only calls ``register_mcp_server`` too
(``engine.py:130``), so YAML-declared MCP servers have the same problem.

*Suggested fix*: open sessions eagerly (a public ``connect_mcp_servers()``
awaited at engine start), or have ``get_tool_descriptions()`` open sessions for
registered-but-unconnected servers.

*Docs meanwhile*: :doc:`tutorials/agents` shows a warm-up call and explains why;
:doc:`reference/yaml` carries a warning.

The ``agent-server`` container command is out of date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dockerfiles/agent.entrypoint.sh`` runs
``python -m kavalai.agents.server "$WORKFLOW_YAML_PATH" --port … --host …``.

* ``kavalai.agents`` no longer exists (``ModuleNotFoundError``); the server now
  lives at ``kavalai.server``.
* The current entry point takes no positional arguments and reads
  ``KAVALAI_AGENT_WORKFLOW_PATH``, ``KAVALAI_AGENT_HOST`` and
  ``KAVALAI_AGENT_PORT`` — not ``WORKFLOW_YAML_PATH`` / ``AGENT_PORT`` /
  ``AGENT_HOST``.
* ``docker-compose.yml`` defines ``agent-migrations`` but no ``agent-server``
  service, so this path is not exercised locally.

``CLAUDE.md`` also documents ``WORKFLOW_YAML_PATH`` for the agent image.

*Docs meanwhile*: :doc:`deploy/index` documents ``python -m kavalai.server`` and
flags the container command as broken.

Cost is recorded nowhere, but exposed everywhere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ModelCallStat`` has **no** ``cost`` field, yet
``create_model_call_stat()`` accepts a ``cost=`` argument and passes it to the
constructor, where Pydantic silently discards it. The ``model_call_stats`` table
has ``cost`` (``db.py:522``) and the frontend has ``cost`` and ``currency``
(``frontend/src/app/models/llm-call-stat.ts``) — both always empty.

*Decide*: add a ``cost`` field plus per-model pricing, or drop the column and
the UI field. Until then, "cost" should not be promised in the docs.

*Docs meanwhile*: all cost claims replaced with token usage, and a note that
usage is recorded but cost is not computed.

``response_code`` is never set for LLM calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only the embedding path builds stats through ``create_model_call_stat`` (which
defaults ``response_code=200``). The provider LLM clients construct
``ModelCallStat`` directly and never set it, so it is always ``None`` for LLM
calls — including in the backoffice's Model Calls table.

Concurrent runs on one engine corrupt ``token_usage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``WorkflowEngine`` holds a single ``TokenAccumulator`` (``engine.py:112``) and
**resets it at the start of every run** (``engine.py:460``). Runs that overlap on
the same engine therefore share one counter: each reports whatever total the
accumulator happened to hold when it finished.

Reproduced with four concurrent ``engine.run()`` calls on one engine, each a
single-node workflow making exactly one model call. Outputs were all correct;
the reported usage was:

.. code-block:: text

   {'model_calls': 2, ... 'total_tokens': 189}
   {'model_calls': 3, ... 'total_tokens': 287}
   {'model_calls': 1, ... 'total_tokens': 94}
   {'model_calls': 4, ... 'total_tokens': 391}

This matters in production: ``kavalai.server`` builds **one** engine and serves
it to every concurrent request, so per-run token figures are unreliable under
any real load — including the ones written to the run row and shown in the
backoffice. Individual ``model_call_stats`` rows come from the client and remain
correct, so the aggregate can be recomputed from them.

*Suggested fix*: make the accumulator per-run — create it in ``run_stream`` and
thread it through the client factory — rather than per-engine.

*Docs meanwhile*: the :doc:`cookbook/index` batch recipe builds one engine per
concurrent run and says why.

Rough edges
-----------

A tool annotated ``-> dict`` silently bypasses validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``def f(...) -> dict``, the generated output model is ``{result: dict}``,
which the returned dict cannot satisfy. ``Validator.cast_result`` catches the
``ValidationError``, logs a warning, and returns the **raw dict** — so the call
succeeds unvalidated and the caller gets a different type than for every other
tool (no ``.result``).

*Decide*: build the output model from the dict's contents, raise instead of
warning, or leave as is and keep documenting it.

*Docs meanwhile*: :doc:`tutorials/agents` has a table of return shapes and
steers readers to Pydantic returns.

``allowed_tools`` means opposite things in YAML and Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``AgentNode.allowed_tools`` defaults to ``[]``, and ``engine.py:297`` passes
``node.allowed_tools or None`` — so an empty list in YAML means *all tools*. In
the Python API, ``Agent(allowed_tools=[])`` means *no tools*.

A YAML author therefore cannot express "this node gets no tools", and the same
value read from two places behaves differently. Consider making ``None`` the
node default and treating ``[]`` consistently.

``SqliteTaskLogger`` has no public way to read rows back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspecting locally-logged tasks means calling ``task_logger._connect()`` and
writing SQL. :doc:`tutorials/workflow` does exactly that, which is a poor thing
to teach. A small read API (``get_tasks(run_id)``, ``get_model_calls(run_id)``)
would make the observability tutorial honest.

``AsyncSessionShim`` does not implement ``get()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``crud.get_one()`` calls ``db.get(model, id)``, which the browser/Pyodide compat
session shim does not provide (``AttributeError``). Any code path shared between
server and browser that loads a row by primary key breaks under Pyodide.

Configuration drift
-------------------

The project ``.env`` sets several variables the code never reads:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Variable
     - Observation
   * - ``ANTHROPIC_KEY``
     - :class:`~kavalai.AnthropicClient` reads ``ANTHROPIC_API_KEY``. As
       written, Anthropic models get no key.
   * - ``KAVALAI_LLM_TIMEOUT``
     - Not read anywhere. Timeouts come from
       :class:`~kavalai.LlmClientParameters`.
   * - ``SERPER_API_KEY``
     - Not read anywhere. The bundled search tools use
       ``LANGSEARCH_API_KEY`` or ``GOOGLE_CUSTOM_SEARCH_API_KEY``.
   * - ``VERTEX_API_KEY``
     - Not read anywhere.
   * - ``BACKOFFICE_HOST`` / ``BACKOFFICE_PORT``
     - Not read; ``kavalai.backoffice.server`` hard-codes port 8000.

``README.md`` (line 35) points readers at extras ``gemini``, ``rag`` and
``server``. ``pyproject.toml`` defines only ``common``, ``common_web``, ``test``
and ``docs``, so those install commands fail. The README was left untouched as
instructed — this needs a maintainer edit.

Unverified in these docs
------------------------

Every example in the tutorials was executed against a live provider, except:

* **Anthropic** — the configured account returns
  ``"Your credit balance is too low to access the Anthropic API"``, so no
  ``anthropic/…`` example has real output. The provider tables list it; no page
  shows a captured Claude response.
* **Ollama** — nothing was listening on ``OLLAMA_HOST`` (``localhost:11434``)
  on the machine these docs were built on, so ``ollama/…`` examples are
  likewise untested.
* **``crawl_url`` / ``web_search``** — these drive a headless browser and were
  documented from their signatures and ``examples/business_info_agent.py``
  rather than executed in a notebook.
* **The backoffice UI screenshots** in :doc:`ui/index` predate this pass and
  were not re-captured.

Documentation follow-ups
------------------------

* The in-browser playground and chat widget only work when a ``kavalai`` wheel
  has been staged into ``_static/pyodide`` (``uv build --wheel`` before
  ``sphinx-build``, as the CI does). A local build without it silently falls
  back to plain-Python mode.
* ``docs/ui/index.rst`` describes the RAG explorer's PCA projection and the
  Workflows timeline from the screenshots; neither was exercised against a live
  backoffice during this pass.

Capability gaps found while comparing frameworks
-------------------------------------------------

Written up honestly in :doc:`tutorials/comparison`. None is a bug; each is a
decision worth taking deliberately, since they are the questions evaluators ask
first.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Gap
     - Notes
   * - No parallel node execution
     - The engine walks one node at a time (no ``gather``/``TaskGroup`` in
       ``engine.py``). A fan-out over N items costs N times one item. LangGraph,
       LlamaIndex Workflows and n8n all fan out natively. Tool calls inside an
       ``agent`` node *are* concurrent (``agent.py:350``).
   * - No durable resume
     - The run row is written at ``initialize_workflow_run`` and again at
       ``_finish`` / ``_record_failure``; there is no mid-run checkpoint, so a
       crash loses the run. This is LangGraph's headline feature and the reason
       Pydantic AI integrates Temporal/DBOS/Prefect.
   * - No human-in-the-loop
     - No way to pause a run for approval and resume it mid-graph. CrewAI has
       ``human_input``, LangGraph has interrupts, n8n has wait nodes. The
       documented workaround makes the pause a boundary between runs.
   * - No multi-agent patterns
     - No handoff, delegation, crew or group-chat primitive — one agent loop per
       node, with the graph doing the routing.
   * - No evaluation tooling
     - No dataset runner, scoring or regression harness for prompt changes. The
       recorded runs are good raw material for one.
   * - No OpenTelemetry export
     - Observability is Kaval.AI's own tables plus loguru. Pydantic AI, the
       OpenAI Agents SDK and the Microsoft Agent Framework all emit OTel, which
       drops into existing tracing stacks.
   * - No long-term memory
     - Memory is the session's chat history plus ``history:`` inputs. There is
       no semantic or summarising memory across sessions.
