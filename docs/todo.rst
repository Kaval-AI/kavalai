Open questions
==============

Observations made while writing this documentation that require a decision or a
code change rather than a documentation fix. Each entry records what was
observed, how it was verified, and what the documentation currently says about
it.

.. note::

   This page is a working list for the maintainers rather than user-facing
   guidance. The design decisions that produce the gaps below are set out in
   :doc:`tutorials/architecture`.

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
* **Parallel node execution has landed.** A ``parallel`` node fans out across
  named branches and rejoins at a declared join node, with branch isolation and
  independence enforced by ``WorkflowGraph.validate_graph`` at load time, a
  per-run node-visit budget, optional ``max_concurrency``, interleaved branch
  events on the single stream and a deterministic recorded trace. It is
  documented in :doc:`reference/yaml` and reflected in
  :doc:`tutorials/comparison`. Fan-out over a runtime-sized list remains open;
  see the capability table below.

Unverified in these docs
------------------------

Every example in the tutorials was executed against a live provider, with the
following exceptions:

* **Anthropic** — the account had no credit when these pages were written, so
  no ``anthropic/…`` example carries real output. It is funded now, the key is
  under the right name, and the client bug that rejected requests setting both
  sampling parameters is fixed; the examples still need to be run and captured.
* **Ollama** — nothing was listening on ``OLLAMA_HOST`` when these pages were
  built. A container is now running with ``llama3.2:1b`` and
  ``nomic-embed-text-v2-moe``, and the two client defects that would otherwise
  have been captured into the documentation are fixed, so ``ollama/…`` examples
  can be added and executed.
* **``crawl_url`` / ``web_search``** — these drive a headless browser and were
  documented from their signatures and ``examples/business_info_agent.py``
  rather than executed in a notebook.
* **The backoffice screenshots** in :doc:`ui/index` predate this pass and were
  not re-captured.

Documentation follow-ups
------------------------

* The in-browser playground and chat widget only work when a ``kavalai`` wheel
  has been staged into ``_static/pyodide`` (``uv build --wheel`` before
  ``sphinx-build``, as CI does). A local build without it falls back silently
  to plain-Python mode.
* ``docs/ui/index.rst`` describes the RAG explorer's PCA projection and the
  Workflows timeline from the screenshots; neither was exercised against a live
  backoffice during this pass.
* The ``parallel`` node is documented in :doc:`reference/yaml`, but no tutorial
  demonstrates a fan-out end to end with real output. The workflow notebook is
  the natural home for it, and doing so requires re-executing that notebook
  against live providers.
* :doc:`guides/data_model` documents the retrieval tables from the DDL the RAG
  services issue. The PostgreSQL collection registry was read from the source
  rather than from a live database during this pass.
* Code examples were rewrapped to fit the rendered column, and notebook output
  blocks now wrap rather than scroll. Two lines still exceed the column: the
  greeter prompt in the workflow notebook and the critique prompt in
  :doc:`cookbook/index`. Both sit inside YAML literal (``|``) blocks, where
  adding a line break changes the prompt the model received, so rewrapping them
  requires re-executing the example rather than editing the text.

Capability gaps found while comparing frameworks
-------------------------------------------------

Recorded candidly in :doc:`tutorials/comparison`, and summarised in
:doc:`tutorials/architecture` under *Known limitations*. None is a defect; each
is a decision worth taking deliberately, since these are the questions
evaluators ask first.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Gap
     - Notes
   * - No fan-out over a list
     - The ``parallel`` node fans out over a fixed set of authored branches; it
       cannot fan out over a runtime-sized list, so "summarise each of these
       forty documents" still requires a loop. A per-item node would reuse the
       branch contexts, event queue and shared budget that ``parallel`` already
       provides; the new parts are the per-item context variable and collecting
       the results into an array data type.
   * - No durable resume
     - The run row is written at ``initialize_workflow_run`` and again at
       ``_finish`` / ``_record_failure``; there is no mid-run checkpoint, so a
       crash loses the run. This is LangGraph's headline feature and the reason
       Pydantic AI integrates Temporal, DBOS and Prefect. The task log is
       already close to a checkpoint, and because ``parallel`` guarantees that
       branch writes are disjoint, branch completion order cannot affect the
       resulting context — which is what would make replay across a fan-out
       deterministic.
   * - No human-in-the-loop
     - There is no way to pause a run for approval and resume it mid-graph.
       CrewAI has ``human_input``, LangGraph has interrupts, and n8n has wait
       nodes. The documented workaround makes the pause a boundary between
       runs rather than a point inside one.
   * - No multi-agent patterns
     - There is no handoff, delegation, crew or group-chat primitive — one
       agent loop per node, with the graph doing the routing.
   * - No OpenTelemetry export
     - Observability is Kaval.AI's own tables plus ``loguru``. Pydantic AI, the
       OpenAI Agents SDK and the Microsoft Agent Framework all emit OTel spans,
       which drop into an existing tracing stack unchanged.
   * - No long-term memory
     - Memory is the session's chat history plus ``history:`` inputs. There is
       no semantic or summarising memory that persists across sessions.
   * - Provider credentials are read inside the clients
     - :class:`~kavalai.OpenAIClient` and the other provider and embedding
       clients fall back to ``os.getenv`` in ``__init__``, which is at odds
       with "library code reads no environment variables". Keeping the
       fallback is deliberate for now --- it is what makes "set the key and it
       works" true --- and an explicit ``api_key=`` (or a registration default)
       already wins. The real fix is for entry points to pass credentials in,
       which touches every construction site.
   * - Only two RAG backends
     - PostgreSQL/pgvector and SQLite. The interface, the capability tiers and
       ``tests/rag/test_conformance.py`` were all shaped with hosted vector
       databases in mind, so a Pinecone or Qdrant backend is a contained piece
       of work --- but nothing verifies that until one exists. Two constraints
       found while designing for them and worth re-checking against current
       vendor docs before starting: ``similarity`` must be normalised to
       higher-is-better cosine, and the default
       ``compute_similarity_matrix`` asks for ``len(source_ids) * 100``
       candidates, which is at or beyond what a hosted store will return.
