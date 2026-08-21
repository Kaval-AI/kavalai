How Kaval.AI compares
=====================

There is no best agent framework, only trade-offs. This page states Kaval.AI's
honestly, alongside the tools people most often weigh it against, so you can
tell quickly whether it fits your problem — or whether something else does.

Everything here was checked against each project's own documentation and source
in August 2026. Frameworks move fast; verify anything load-bearing before you
commit to it.

The short version
-----------------

**Kaval.AI is a typed, declarative workflow engine with agents inside it** —
not an agent library that grew a workflow API. A workflow is a YAML graph, every
value crossing a node boundary is a validated Pydantic model, every run is
persisted, and a backoffice UI reads those runs back. It also runs in a browser,
which nothing else here does.

That focus costs breadth. If you need parallel fan-out, durable resume after a
crash, human-in-the-loop approvals, or a catalogue of hundreds of integrations,
the frameworks below are ahead today. See `Where Kaval.AI is behind`_.

The landscape
-------------

The tools fall into three groups, and comparing across groups is mostly
unhelpful.

**Code-first agent libraries** — you write Python; the framework provides
agents, tools and orchestration primitives.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Framework
     - The idea
   * - `LangGraph <https://github.com/langchain-ai/langgraph>`__ (MIT)
     - Agents as a stateful graph. Headline features are durable execution,
       human-in-the-loop, memory and streaming. The closest thing here to
       Kaval.AI's model — but the graph is defined in Python, not data.
   * - `LangChain <https://www.langchain.com/>`__ (MIT)
     - The integration layer: model wrappers, retrievers, tool adapters. Its
       reach is the real product — swapping providers is one line, and almost
       every vendor ships a LangChain adapter.
   * - `CrewAI <https://github.com/crewAIInc/crewAI>`__ (MIT)
     - Role-based agents ("researcher", "writer") grouped into crews, plus
       Flows for deterministic control. Agents and tasks are declared in JSONC
       (or YAML in classic projects) with a Python class binding them together.
   * - `LlamaIndex Workflows <https://developers.llamaindex.ai/>`__ (MIT)
     - Event-driven steps: each step consumes an event and emits another.
       Branches are ordinary ``if`` statements, loops are events routed
       backwards, and concurrency falls out of emitting several events.
   * - `Pydantic AI <https://pydantic.dev/docs/ai/>`__ (MIT)
     - Typed agents. Closest to Kaval.AI in philosophy: give the agent an output
       type and every run comes back validated. Durability is delegated to
       Temporal, DBOS or Prefect; observability is OpenTelemetry/Logfire.
   * - `OpenAI Agents SDK <https://openai.github.io/openai-agents-python/>`__ (MIT)
     - A deliberately small surface: agents, handoffs, guardrails, sessions,
       tracing. The fastest path to a working agent.
   * - `Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`__ (MIT)
     - The merger of AutoGen and Semantic Kernel. Graph workflows, group chat
       and handoff patterns, Python and .NET, Azure-shaped.
   * - `Haystack <https://github.com/deepset-ai/haystack>`__ (Apache-2.0)
     - Pipelines of components, RAG-first. Pipelines serialise to YAML, so
       Haystack is the other framework here with a real declarative format.

**Visual platforms** — you draw the workflow in a browser and the platform hosts
it.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Platform
     - The idea
   * - `n8n <https://github.com/n8n-io/n8n>`__ (fair-code, Sustainable Use License)
     - General workflow automation with 400+ integrations and AI nodes on top.
       If the hard part of your problem is *connecting to twelve SaaS tools*,
       this is a different and probably better tool than any library here.
   * - `Dify <https://github.com/langgenius/dify>`__ (Apache-2.0 with conditions)
     - An LLM app platform: visual builder, RAG pipeline, prompt IDE and an
       admin UI, self-hostable. The closest peer to Kaval.AI's
       *library plus backoffice* shape, approached from the no-code side.

**Kaval.AI** sits between the two: workflows are data (YAML) as in the visual
tools, but they live in your repository, run in your process, and are reviewed
like code.

Feature comparison
------------------

"Yes" means the capability is built in and documented, not that it is
achievable with enough glue code.

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 12 12 12 12

   * - Capability
     - Kaval.AI
     - LangGraph
     - CrewAI
     - LlamaIndex
     - Pydantic AI
     - n8n
   * - Declarative definition file
     - YAML
     - no (Python)
     - JSONC/YAML + Python
     - no (Python)
     - no (Python)
     - visual JSON
   * - Typed & validated at every step boundary
     - yes
     - partial [#f1]_
     - task outputs
     - typed events
     - yes
     - no
   * - Explicit graph with cycles
     - yes
     - yes
     - Flows
     - yes
     - via Pydantic Graph
     - yes
   * - Parallel step execution
     - **no**
     - yes
     - async tasks
     - yes
     - yes
     - yes
   * - Durable resume after a crash
     - **no**
     - yes
     - partial
     - via context store
     - via Temporal/DBOS
     - yes
   * - Human-in-the-loop pause/approve
     - **no**
     - yes
     - yes
     - yes
     - yes
     - yes
   * - Multi-agent handoff/delegation
     - **no** [#f2]_
     - yes
     - yes
     - yes
     - manual
     - yes
   * - Streaming
     - yes
     - yes
     - yes
     - yes
     - yes
     - n/a
   * - Tools: Python / REST / MCP
     - all three [#f3]_
     - Python + MCP
     - Python + MCP
     - Python + MCP
     - Python + MCP
     - nodes + MCP
   * - RAG built in
     - yes
     - via LangChain
     - yes
     - yes (its focus)
     - no
     - yes
   * - Persistence + monitoring UI included
     - yes
     - LangSmith (SaaS)
     - AMP (SaaS)
     - integrations
     - Logfire (SaaS)
     - yes
   * - Runs client-side in a browser
     - **yes, unique**
     - no
     - no
     - no
     - no
     - no
   * - HTTP serving included
     - yes
     - LangGraph Server
     - no
     - no
     - AG-UI adapter
     - yes
   * - Evaluation tooling
     - **no**
     - yes
     - yes
     - yes
     - yes
     - partial
   * - License
     - Apache-2.0
     - MIT
     - MIT
     - MIT
     - MIT
     - fair-code

.. [#f1] LangGraph state is typically a ``TypedDict``; validation is not
   enforced at each edge the way a Pydantic model is. Pydantic state is
   supported if you opt in.
.. [#f2] An ``agent`` node runs a full tool-using loop, and a graph can route
   between several of them — but there is no agent-to-agent handoff primitive,
   and no agent can delegate to another agent by itself.
.. [#f3] Every framework here can call a REST API — by writing a Python function
   that does, or through an OpenAPI toolkit. The distinction is that Kaval.AI
   registers the endpoint itself as a tool with its own schemas, so there is no
   wrapper function to write or maintain.

Where Kaval.AI is ahead
-----------------------

**The workflow is data, and the data is typed.** CrewAI and Haystack also have
declarative formats, but in Kaval.AI the *whole graph* — nodes, edges, branch
conditions, data types — is one YAML file, and ``data_types`` are JSON-schema
fragments compiled into Pydantic models. A malformed value fails at the
boundary that produced it. In most code-first frameworks, state is a dict and
the failure surfaces several steps later. See :doc:`../reference/yaml`.

**One kernel for every kind of tool.** Python functions, REST endpoints and MCP
tools are all addressed as URIs (``python://``, ``rest://``, ``mcp://``), all
validated through generated Pydantic models, all restrictable per node with
``allowed_tools``. Most frameworks treat REST as "write a Python wrapper".

**Observability without a SaaS account.** Runs, sessions, chat history, per-node
tasks and per-call token counts land in *your* Postgres, and the backoffice UI
reads them from there. The comparable experience elsewhere — LangSmith, Logfire,
CrewAI AMP — is a hosted product. Dify and n8n also self-host their UI, but you
adopt their whole platform to get it.

**Determinism is cheap.** Inject a ``client_factory`` and the graph runs with no
network at all, so branching logic is testable in CI for free. See
:doc:`../guides/safety`.

**It runs in a browser.** Engine, model and embeddings execute client-side over
WebGPU and Pyodide — no server, no API key, no data leaving the device. Nothing
else in this comparison does this. See :doc:`run_in_browser`.

Where Kaval.AI is behind
------------------------

Stated plainly, because choosing a framework on marketing is expensive.

**No parallel step execution.** The engine walks one node at a time. A fan-out
over ten documents runs ten times as long as one — LangGraph, LlamaIndex and n8n
all fan out natively. Tool calls *within* one agent node do run concurrently, and
you can run several workflows concurrently from your own code (see the
:doc:`../cookbook/index`), but there is no parallel node in the graph.

**No durable resume.** A run's row is written when it starts and when it
finishes; per-node data goes to the task logger as it happens. There is no
mid-run checkpoint to resume from, so a process crash loses the run. LangGraph's
whole pitch is the opposite, and Pydantic AI borrows Temporal for it. For short
request-shaped runs this rarely matters; for a twenty-minute research job it
does.

**No human-in-the-loop primitive.** You cannot pause a run for approval and
resume it. The workaround is to end the run, keep the state in the session, and
start a new run after the human answers — workable, but it is a pattern you
build, not a feature you call. CrewAI (``human_input``), LangGraph (interrupts)
and n8n (wait nodes) all support this directly.

**No multi-agent patterns.** There are no crews, handoffs, group chats or
delegation. You get one agent loop per node and a graph to route between them.
If your mental model is "a team of specialists negotiating", CrewAI or the
Microsoft Agent Framework fit that shape and Kaval.AI does not.

**No evaluation tooling.** No dataset runner, no scoring, no regression harness
for prompt changes. You get raw material — every run, task and model call is
recorded — but the analysis is yours to write.

**A much smaller ecosystem.** Four LLM providers, four embedding providers and
six bundled tools, against LangChain's thousand-plus integrations and n8n's four
hundred connectors. Kaval.AI's answer is that a REST endpoint or MCP server is a
first-class tool without an adapter — real, but not the same as an integration
someone else already tested.

**Fewer eyes on it.** These are mature projects with large communities, years of
production use and a Stack Overflow trail. Kaval.AI is young. Some of what you
would find by searching, you will find by reading its source.

Choosing
--------

Reach for **Kaval.AI** when the workflow should be a reviewable artefact rather
than code, when typed boundaries and recorded runs matter more than breadth, when
you want observability you host yourself, or when it has to run in a browser.

Reach for **LangGraph** when runs are long, must survive restarts, or need a
human in the middle. **CrewAI** when the work splits into roles and you want a
prototype this afternoon. **LlamaIndex** when retrieval *is* the product.
**Pydantic AI** when you want typed agents without a graph engine at all.
**OpenAI Agents SDK** when the smallest possible surface wins. **Haystack** for
classic RAG pipelines. **n8n** when the hard part is connecting SaaS tools, and
**Dify** when non-engineers should build the thing.

These also compose. Kaval.AI exposes a workflow over HTTP
(:doc:`serving`), and n8n, Dify or another service can call that endpoint — the
typed graph stays in your repository while the integration sprawl lives where
integration sprawl belongs.

.. note::

   Gaps listed here are tracked in :doc:`../todo`. If one of them is blocking
   you, that page is the place to look before assuming it will never exist.
