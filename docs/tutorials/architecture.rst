============
Architecture
============

This page describes how Kaval.AI is put together and why. It is intended for
two audiences: engineers deciding whether the design fits their problem, and
contributors — human or automated — who need to know where a change belongs
before making it. Where a design decision has a rationale that is not evident
from the code, that rationale is recorded here.

For the vocabulary of language-model applications, see :doc:`../guides/concepts`.
For the persisted schema, see :doc:`../guides/data_model`. For an assessment
against comparable frameworks, see :doc:`comparison`.

The thesis
==========

Most agent frameworks are libraries of behaviour: you call an agent, it decides
what to do, and the framework's job is to make that decision easy to express.
Kaval.AI takes a different position. **An agentic application is a program, and
a program should have a declared structure, typed boundaries and an execution
record.** The framework's job is to hold the model to that structure.

Three commitments follow from this, and nearly every design decision below is
an instance of one of them.

*The structure is data.* A workflow is a graph, and that graph is a document —
YAML in a repository, reviewed in a pull request, rendered as a diagram,
served over HTTP and stored beside the runs it produced. It is not a
side effect of the order in which Python statements happen to execute.

*Every boundary is typed.* Each value crossing a node boundary is a validated
Pydantic model. A malformed value fails at the node that produced it, not three
steps later in a dictionary lookup that returns ``None``.

*Execution is a matter of record.* Every session, run, node execution and model
call is written to a database the operator owns. Observability is not a
subscription; it is a table.

The components
==============

.. image:: /_static/architecture.svg
   :alt: Components of Kaval.AI and the paths between them
   :width: 100%

The figure reads downwards. Definition becomes a validated graph; the engine
walks that graph; nodes reach outwards to models, tools and indexes; and
everything that happened is recorded in a database the backoffice interface
later reads. The sections below take each band in turn.

Definition: two front doors, one graph
--------------------------------------

A workflow may be written as a YAML document or assembled with
:class:`~kavalai.WorkflowBuilder`. These are not two systems. Both produce a
:class:`~kavalai.WorkflowGraph`, a Pydantic model, and everything downstream
sees only that graph. The consequence worth stating explicitly is that the two
front doors cannot drift apart: a feature reachable from one is reachable from
the other, because both terminate in the same model.

The graph is validated when it is loaded rather than when it runs. Loading
fails if node names collide, if there is not exactly one ``start`` node, if
there is no ``end`` node, if two ``end`` nodes return different data types, if
a transition names a node that does not exist, or if a node writes an
``output`` not declared in ``data_types``. Structural errors therefore surface
at deployment rather than in production, on the branch that is taken once a
month.

``data_types`` entries are JSON-schema fragments, compiled by ``SchemaParser``
into Pydantic models. Two names are reserved: ``input`` and ``output``, the
types of the workflow itself. This is what allows a workflow to be served over
HTTP with a generated, accurate request and response schema without the author
writing one.

.. note::

   *Design constraint.* Anything that changes the shape of a workflow belongs
   in :mod:`kavalai.workflow.models` first. The builder, the YAML loader, the
   engine, the SVG renderer and the backoffice all derive from those models;
   adding a capability to the engine alone produces a feature that YAML cannot
   express and the diagram cannot show.

Execution: one engine, many runs
--------------------------------

:class:`~kavalai.WorkflowEngine` walks the graph. Its single execution path is
``run_stream()``, an asynchronous generator of ``WorkflowStreamEvent``; the
convenient ``run()`` drains it. Having one path rather than two is a
deliberate constraint, and it is why streaming behaviour cannot diverge from
non-streaming behaviour: there is no second implementation in which to
introduce the divergence.

The engine is designed to be shared. One engine may serve many concurrent runs,
which requires a clear division of state:

* **Per-run state lives on** :class:`~kavalai.RunContext` — the resolved data
  each node has seen and the token accumulator into which every model call
  reports. Anything scoped to a single invocation belongs here, and
  ``_branch_context`` must forward it when a run splits. The node-visit budget
  that terminates runaway cycles is counted per run for the same reason: were
  it counted per walk, each branch of a ``parallel`` node would receive the
  full allowance.
* **Engine-level state is opened once** — the
  :class:`~kavalai.FunctionKernel` and its MCP sessions are established by
  ``await engine.connect()`` and released by ``await engine.aclose()``. They
  are never opened or closed per run.

Both halves of that rule have been violated in this codebase before, and both
produced the same class of defect: a token accumulator shared between
concurrent runs reported one run's usage against another, and a kernel closed
in a per-run ``finally`` block tore down MCP sessions that other runs were
still using. The division above is what prevents recurrence.

Node kinds
----------

The walker dispatches on node type, and the set is deliberately small:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Node
     - Role
   * - ``start`` / ``end``
     - The graph's boundary. Exactly one ``start``; any number of ``end``
       nodes, all returning the same type.
   * - ``llm``
     - One structured completion. The prompt is rendered, the model is called,
       and the validated result is stored under ``output``.
   * - ``agent``
     - A full tool-using loop inside one node, for the case where the model
       should decide which tools to call.
   * - ``function``
     - Exactly one tool call, addressed by URI, for the case where the author
       already knows.
   * - ``if`` / ``switch``
     - Routing, evaluated by a restricted expression language rather than by
       ``eval``.
   * - ``parallel``
     - Fan-out across independent branches, rejoining at a named join node.
   * - ``rag_query``
     - One retrieval against a RAG service, read-only. Indexing is not
       reachable from a document.

The distinction between ``agent`` and ``function`` deserves emphasis, because
it is where Kaval.AI differs most visibly from agent-first frameworks.
Delegating a decision to a model is a choice with a cost — latency, tokens and
non-determinism — and the design makes that choice explicit at the node rather
than implicit in a prompt. A graph in which only two of eleven nodes are
``agent`` nodes is a graph whose behaviour is mostly determined by its author.

Concurrency is likewise explicit. A ``parallel`` node names its branches, and
the run continues at ``next`` once every branch has arrived. The alternative —
inferring a dataflow graph from ``inputs`` and reordering independent steps
automatically — was rejected. A Kaval.AI graph is a state machine with declared
edges, so reordering ``a → b → c`` because ``b`` does not read ``a``'s output
would silently change the order of side effects the author wrote down, and
nothing in the file would reveal that the workflow now runs concurrently.
Concurrency should be legible in the document and visible in the diagram.

Branch isolation follows from the same reasoning: each branch receives its own
``RunContext`` seeded with a shallow copy of the parent's data, so no branch
observes a sibling's output while both are running, and outputs are merged only
at the join. The graph validator enforces at load time that branches are
disjoint, that no two branches write the same output variable, and that a
branch contains no ``end`` node.

Reaching outwards: models, tools and indexes
--------------------------------------------

Three subsystems connect a graph to the world, and each presents a single
interface over several implementations.

**LLM clients.** OpenAI, Gemini, Anthropic, Ollama and an in-browser WebLLM
client sit behind one asynchronous interface, selected by a ``provider/model``
string. That set is open rather than fixed: the built-ins are entries in a
registry, and :func:`~kavalai.register_llm_provider` adds more under names of
your own, resolvable from YAML like any other. Structured output, streaming, retries and usage statistics are the
client's responsibility, not the caller's — which is what makes substituting a
provider a change of one string. Provider SDKs are imported lazily, so
``import kavalai`` succeeds in a Pyodide environment where none of them exists.

**The function kernel.** Python functions, REST endpoints and MCP tools are all
registered on one :class:`~kavalai.FunctionKernel` and addressed by a uniform
URI: ``protocol://[name|module].function_name(args: type) -> return_type``.
Every tool has generated Pydantic argument and result models, so a tool result
that cannot satisfy its declared type raises rather than propagating unvalidated.
``allowed_tools`` restricts what a node may see and call, and means the same
thing in YAML and in Python.

Treating REST as a first-class protocol rather than something to be wrapped is
a deliberate trade. Most frameworks expect a Python function around an HTTP
call; Kaval.AI registers the endpoint itself, with its own schemas, so there is
no wrapper to maintain. The cost is that Kaval.AI has no catalogue of
pre-tested integrations, which :doc:`comparison` states plainly.

**RAG services.** ``BaseRagService`` has two implementations: PostgreSQL with
pgvector, and a single-file SQLite index. They are interchangeable because the
interface, not the storage, is the contract — which is what allows an index to
be built on a server and shipped to a browser.

The interface is generic; the implementations deliberately are not. Postgres
carries a dozen methods the interface never mentions, and that is the intended
shape — a backend should expose what its store does well. What the interface
declares comes in three tiers: six required methods, two optional ones
(``count_entries`` and ``iter_entries``, guarded by ``supports()``), and two
with working defaults that a backend may override. ``tests/rag/test_conformance.py``
runs the declared contract against every backend, so "implements the interface"
is checked rather than assumed.

Recording: the run is the artefact
----------------------------------

Persistence is split in two, and the split is by writer rather than by table.
:class:`~kavalai.agent_service.AgentService` records what the workflow was
asked and what it answered — agents, sessions, runs and chat messages. The
``TaskLogger`` records how it got there — per-node tasks and per-call model
statistics, written behind the run so that logging never becomes the critical
path. The LLM clients themselves emit ``ModelCallStat`` records, which is why
calls made outside a workflow are recorded too.

Both write to a database the operator supplies. There is no hosted collector,
and the backoffice interface is a reader of those tables rather than a
privileged component: it adds nothing the tables do not already contain. The
schema, and the reasoning behind each table, is documented in
:doc:`../guides/data_model`.

Portability: the same engine in three places
============================================

The same graph executes in a server process, in a test suite and in a browser
tab. This is not an incidental property; it constrains the base package.

``pyproject.toml`` keeps the base install free of greenlet and of native
extensions beyond the prebuilt Pyodide packages, with everything else in the
``common`` extra. Under Pyodide, where ``greenlet`` and ``aiosqlite`` do not
exist, ``AgentService`` runs over a synchronous SQLite engine through
``AsyncSessionShim``, which presents the awaitable surface the service expects.
The browser store is created with ``create_all`` and stamped with
``PRAGMA user_version``, because Alembic cannot run there.

The practical consequence for contributors is that a dependency added to the
base package must be justified against browser execution, and a schema change
must bump ``SQLITE_SCHEMA_VERSION`` or stale browser databases will silently
diverge from the models.

Being Pythonic
==============

The library is meant to be usable without first learning it. Several
conventions serve that goal, and they are worth stating because they are
constraints on future work rather than accidents of the current implementation.

**One import surface.** Everything a user ordinarily needs is reachable as
``from kavalai import X``. The ORM row classes live in :mod:`kavalai.db` so
that ``Agent`` at the top level unambiguously means the agent, not a table row.

**Async throughout, with no hidden event loop.** Every I/O-bound entry point is
a coroutine. The library never starts a loop on the caller's behalf, so it
composes with FastAPI, with a notebook and with ``asyncio.run`` alike.

**Types the user already knows.** Inputs, outputs, tool arguments and tool
results are Pydantic models. There is no bespoke schema language to learn, and
an editor's completion works on a workflow's output because it is an ordinary
model.

**Configuration is passed, not discovered.** Library code never reads
environment variables; only entry-point ``main()`` functions do. A component's
behaviour is therefore determined by its arguments, which is what makes the
library testable and multi-tenant deployment possible.

The backend registries are a bounded exception, and worth stating rather than
glossing. A workflow names its model as a string, and under
``python -m kavalai.server`` the user does not construct the engine, so a
constructor argument cannot reach that far; the registries are how a name
becomes a backend there. They are bounded: registration is an explicit call in
code the user wrote — no scanning, no entry points — arguments still win
(``client_factory`` outranks the LLM registry, ``rag_services=`` outranks the
RAG one), only ``replace=True`` can change an existing name and it logs, and
``registered_llm_providers()`` reports what a process actually supports. The
short form: *the set of backends is discovered; the behaviour of any given run
is still passed.*

**Deterministic by construction.** Injecting a ``client_factory`` replaces the
model with a stub, so routing logic can be exercised in continuous integration
with no network and no API key. See :doc:`../guides/safety`.

**Failures are loud.** An unresolvable prompt reference raises rather than
rendering as an empty string; a tool result that does not match its declared
model raises rather than passing through; duplicate tool or server names raise
at registration. Silence at the point of failure is what produces the
inexplicable output three steps later.

Working on Kaval.AI with a coding assistant
===========================================

The repository is set up so that an automated contributor can work in it
without first being told the conventions:

* ``AGENTS.md`` at the repository root is the entry point: layout, commands,
  invariants and the places changes usually belong.
* ``CLAUDE.md`` holds the same material in the form Claude Code reads
  directly.
* This page is the reference for larger decisions — where a component belongs,
  and which of the commitments above a proposed change would violate.

The invariants most often at issue are collected here for convenience:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Invariant
     - Why it holds
   * - Workflow shape changes start in ``workflow/models.py``
     - The builder, loader, engine, renderer and backoffice all derive from
       those models.
   * - ``run_stream()`` is the only execution path
     - Streaming and non-streaming behaviour cannot diverge if there is one
       implementation.
   * - Per-run state on ``RunContext``; kernel state on the engine
     - One engine serves many concurrent runs.
   * - Library code reads no environment variables
     - Only entry-point ``main()`` functions do; everything else is passed in.
   * - A workflow document names a registration, never a Python path
     - ``GET /workflow`` serves it and the backoffice edits it, so a dotted
       path or a connection string in one would cross a privilege boundary.
   * - ORM models are the single source of truth for the schema
     - Migrations are autogenerated from them, and a parity test fails if they
       diverge.
   * - The base package stays Pyodide-compatible
     - The same engine has to run in a browser tab.
   * - Every boundary validates
     - A malformed value must fail where it was produced.

Known limitations
=================

The architecture has costs, and they are recorded rather than elided. A run has
no mid-execution checkpoint, so a crashed process loses the run; there is no
primitive for pausing a run pending human approval; there is no agent-to-agent
handoff, delegation or group-chat pattern; and observability is Kaval.AI's
own tables rather than OpenTelemetry. Each is discussed in :doc:`comparison`.

Where to next
=============

* :doc:`../guides/workflows` — the workflow model in depth.
* :doc:`../guides/data_model` — the tables the runtime writes.
* :doc:`../reference/yaml` — every key of the workflow document.
* :doc:`comparison` — the same design assessed against other frameworks.
