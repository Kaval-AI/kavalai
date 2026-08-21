Open questions
==============

Things found while writing this documentation that need a decision or a code
change rather than a documentation fix. Each entry says what was observed, how
it was verified, and what the docs currently say about it.

.. note::

   This page is a working list for the maintainers, not user-facing guidance.

Resolved
--------

Everything previously listed under *Bugs*, *Rough edges* and *Configuration
drift* has landed, and the pages that worked around those behaviours have been
rewritten to describe the current ones:

* MCP servers are connected and their tools discovered before a run, so an
  agent is no longer told it has none (``FunctionKernel.connect_mcp_servers``,
  ``WorkflowEngine.connect``).
* The token accumulator is per run rather than per engine, so concurrent runs
  on one engine report their own usage. The kernel is no longer closed at the
  end of each run either — tool servers belong to the engine and are released
  by ``WorkflowEngine.aclose``.
* Model call statistics now carry the provider's status code, and failed
  attempts are recorded rather than vanishing.
* ``cost`` and ``currency`` are gone from ``model_call_stats``;
  ``cached_prompt_tokens`` and ``reasoning_tokens`` replace them.
* Ollama structured output sends the response model's JSON Schema instead of
  the legacy ``format="json"``, and a system-only chat history is normalised to
  a user turn in every client.
* ``allowed_tools`` means the same thing in YAML and Python, with ``"*"`` for
  "everything"; a tool result that cannot satisfy its output model now raises
  instead of being passed through unvalidated.
* ``SqliteTaskLogger`` has ``get_tasks`` / ``get_model_calls``;
  ``AsyncSessionShim`` implements what ``AgentService`` actually calls, with
  the service's test suite running against both session types.
* ``GET /workflow`` redacts MCP server environment values, and the agent server
  warns at startup when authentication is disabled. Exposure otherwise follows
  the same rule as every other endpoint, which is useful in development.
* ``.env`` drift is gone, ``.env.example`` documents every variable the code
  reads, and ``tests/test_config_drift.py`` keeps the two in step.

Unverified in these docs
------------------------

Every example in the tutorials was executed against a live provider, except:

* **Anthropic** — the account had no credit when these docs were written, so no
  ``anthropic/…`` example has real output. It is funded now and the key is under
  the right name; the examples still need to be run and captured.
* **Ollama** — nothing was listening on ``OLLAMA_HOST`` when these docs were
  built. A container is running now with ``llama3.2:1b`` and
  ``nomic-embed-text-v2-moe``, and the two client bugs that would have been
  captured into the docs are fixed, so ``ollama/…`` examples can be added and
  executed.
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
     - **Being addressed** — a ``parallel`` node with branch fan-out and a join
       is in the tree. :doc:`tutorials/comparison` and
       :doc:`reference/yaml` still describe the old behaviour and need updating
       once it settles. Tool calls inside an ``agent`` node were already
       concurrent (``agent.py:350``).
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
