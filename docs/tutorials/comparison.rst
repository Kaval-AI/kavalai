How Kaval.AI compares
=====================

There is no best agent framework, only trade-offs. This page states
Kaval.AI's explicitly, alongside the tools it is most often weighed against, so
that a reader can determine quickly whether it suits their problem, or whether
something else does.

Every claim here was checked against each project's own documentation and
source in August 2026. Frameworks move quickly; anything load-bearing should be
verified before it is relied upon.

The short version
-----------------

**Kaval.AI is a typed, declarative workflow engine with agents inside it** —
not an agent library that grew a workflow API. A workflow is a YAML graph, every
value crossing a node boundary is a validated Pydantic model, every run is
persisted, and a backoffice UI reads those runs back. It also runs in a browser,
which nothing else here does.

That focus costs breadth. Where durable resume after a crash,
human-in-the-loop approval, multi-agent delegation or a catalogue of hundreds of
integrations is required, the frameworks below are ahead today. See
`Where Kaval.AI is behind`_.

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
     - yes [#f4]_
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
   * - Model providers
     - 5 built in, plus a registry [#fprov]_
     - ~30 packages
     - many
     - many
     - 14 native + 14 compatible
     - many
   * - RAG built in
     - yes [#frag]_
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
     - yes
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
.. [#fprov] OpenAI, Gemini, Anthropic, Ollama and in-browser WebLLM ship with
   Kaval.AI; :func:`~kavalai.register_llm_provider` adds any other under a name
   of your own, usable from YAML like a built-in.

.. [#frag] Two backends (pgvector, SQLite) behind one interface, plus a
   ``rag_query`` node, so retrieval is expressible in the workflow document
   rather than only from Python.

.. [#f3] Every framework here can call a REST API, either by writing a Python
   function that does so or through an OpenAPI toolkit. The distinction is that
   Kaval.AI registers the endpoint itself as a tool with its own schemas, so
   there is no wrapper function to write or maintain.
.. [#f4] A ``parallel`` node fans out across named branches and rejoins them at
   a single node; tool calls within one ``agent`` node were already concurrent.
   What Kaval.AI does not yet have is fan-out over a *runtime-sized list* — one
   subgraph executed once per item. See :doc:`../reference/yaml`.

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

**Observability without a hosted account.** Runs, sessions, chat history,
per-node tasks and per-call token counts land in *your* PostgreSQL instance,
and the backoffice interface reads them from there. Attempts that returned no
completion are recorded with the provider's status code rather than vanishing,
and where a provider reports them, cached input and reasoning tokens are broken
out from the totals, which is what makes cost computable downstream. The
comparable experience elsewhere — LangSmith, Logfire, CrewAI AMP — is a hosted
product. Dify and n8n also self-host their interface, but adopting it means
adopting the whole platform. See :doc:`../guides/data_model`.

**Concurrency is declared, not inferred.** A ``parallel`` node names its
branches, so a reader can tell from the file — and from the rendered diagram —
which steps run together. The alternative, inferring a dataflow graph from
``inputs`` and reordering independent steps automatically, was rejected because
it would silently change the order of side effects the author wrote down. See
:doc:`architecture`.

**Determinism is inexpensive.** Injecting a ``client_factory`` runs the graph
with no network at all, so branching logic can be exercised in continuous
integration at no cost. See :doc:`../guides/safety`.

**Evaluation grades the deployment, and needs no service.** A suite is one
YAML file of cases run against an agent server that is already up: the
evaluators discover its input and output types from its OpenAPI specification
and judge what a caller would see, so the artefact under test is the one you
are about to promote rather than a graph reassembled in a test process. Which
agent is graded is named on the command line and never in the file, which is
what makes two model versions comparable. LangSmith and agentevals need their
SDK inside your process and their service outside it; pydantic-evals and
promptfoo grade inputs and outputs as this does, without the sessions the
graded runs leave behind in your own database. Cases are files in your
repository, so a behaviour change is a diff in code review rather than a number
in a dashboard. See :doc:`../guides/evaluation`.

**It runs in a browser.** Engine, model and embeddings execute client-side over
WebGPU and Pyodide — no server, no API key, no data leaving the device. Nothing
else in this comparison does this. See :doc:`run_in_browser`.

Where Kaval.AI is behind
------------------------

Stated plainly, because choosing a framework on the strength of its marketing
is expensive.

**No fan-out over a list.** A ``parallel`` node runs named branches
concurrently, which covers work that is known when the workflow is written. It
does not cover work discovered at run time: there is no node that executes one
subgraph per item of a list, so summarising forty documents still requires
either forty declared branches or a sequential loop. LangGraph's ``Send``,
LlamaIndex's event fan-out and n8n's item-based execution all address this
directly.

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

**No OpenTelemetry export.** Observability is Kaval.AI's own tables plus
``loguru``. Pydantic AI, the OpenAI Agents SDK and the Microsoft Agent Framework
all emit OTel spans, which drop into an existing tracing stack without further
work.

**No long-term memory.** Memory is the session's chat history and ``history:``
inputs. There is no semantic or summarising memory that persists across
sessions.

**A considerably smaller ecosystem.** Five LLM providers, five embedding
providers and three bundled tools, against LangChain's thousand-plus integrations
and n8n's four hundred connectors. Kaval.AI's answer is that a REST endpoint or
an MCP server is a first-class tool requiring no adapter — which is true, but
not equivalent to an integration someone else has already tested.

**Fewer eyes on it.** These are mature projects with large communities, years of
production use and an extensive record of questions already answered. Kaval.AI
is young; some of what would otherwise be found by searching must instead be
found by reading its source.

Choosing
--------

Choose **Kaval.AI** when the workflow should be a reviewable artefact rather
than code, when typed boundaries and recorded runs matter more than breadth,
when observability must be self-hosted, or when the application has to run in a
browser.

Choose **LangGraph** when runs are long, must survive restarts, or require a
human in the middle. **CrewAI** when the work divides into roles and a prototype
is wanted the same day. **LlamaIndex** when retrieval *is* the product.
**Pydantic AI** when typed agents are wanted without a graph engine at all.
**OpenAI Agents SDK** when the smallest possible surface wins. **Haystack** for
classic RAG pipelines. **n8n** when the difficult part is connecting SaaS tools,
and **Dify** when non-engineers are to build the application themselves.

These also compose. Kaval.AI exposes a workflow over HTTP (see :doc:`serving`),
and n8n, Dify or another service can call that endpoint: the typed graph remains
in the repository while the integration sprawl lives where integration sprawl
belongs.

.. note::

   The reasoning behind the design that produces the gaps listed here is
   set out in :doc:`architecture`. A
   gap that is blocking should be checked against those pages before it is
   assumed to be permanent.
