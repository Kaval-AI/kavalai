=============
Observability
=============

A pipeline you cannot see into is a pipeline you cannot trust. Kaval.AI makes
every run **observable** by default: each run carries its own state and trace,
every node and model call is logged, and the whole history is persisted so you
can reload and inspect it later — in code or in the backoffice UI.

This guide covers the *why*; for a hands-on tour of storage — chat history,
context, sessions, the tables and writing your own backend — see the
:doc:`../tutorials/observability_storage` tutorial, and for the schema itself,
table by table and column by column, see :doc:`data_model`.

What a run records
------------------

Every run produces a :class:`~kavalai.WorkflowState` (see :doc:`workflows`). For
observability the key fields are:

* ``trace`` — the ordered list of visited node names, i.e. the exact path the
  run took through the graph.
* ``token_usage`` — a roll-up of ``model_calls``, ``prompt_tokens``,
  ``completion_tokens``, and ``total_tokens``.
* ``run_id`` / ``session_id`` / ``invocation_id`` — identifiers that tie logs,
  storage, and chat history together. The 8-char ``invocation_id`` prefixes
  every log line of the run, so the log for one run can be isolated with a
  single search.

Persistence and logging
-----------------------

Persistence and logging are split into two pieces handed to the engine when
you build it:

* :class:`~kavalai.agent_service.AgentService` — agents, sessions, runs, and
  chat history.
* **TaskLogger** — per-node logs and model call stats.

.. code-block:: python

   from kavalai import WorkflowEngine

   engine = WorkflowEngine.from_yaml(yaml, agent_service=..., task_logger=...)

Local vs. production databases
------------------------------

The ``AgentService`` runs against any database the ORM models support: for
local development and tests point it at in-memory SQLite
(``AgentService(db_manager.get_sqlite_sessionmaker())``), in production at
Postgres — the same ``agents`` / ``sessions`` / ``runs`` / ``chat_messages``
tables either way. ``SqliteTaskLogger`` is the local counterpart of the
production ``PostgresTaskLogger``. The same code runs against either — only
the connection changes.

AgentService
------------

The engine records each run through the service: ``initialize_workflow_run``
starts it, ``update_run`` lands the output and resolved context, and
``add_chat_message`` / ``get_chat_history`` carry the conversation. To pull a
finished run's conversation back:

.. code-block:: python

   history = await engine.agent_service.get_chat_history(UUID(state.session_id))

Per-model-call statistics come from the LLM clients themselves: every call
produces a ``ModelCallStat`` with token usage and timing, delivered through the
``ModelStatsReceiver`` callback interface (``ModelStatsLogger`` merely logs
them). During a run the receiver is the run's ``TokenAccumulator``, which adds
each call to ``state.token_usage`` and passes it to the task logger with the
agent, session and run that made it. See :doc:`../tutorials/llm_clients`.

Attempts that never produced a response are recorded too, with the provider's
status code and the error text in place of token counts — so a rate-limit storm
or an outage leaves a trace instead of a suspiciously healthy-looking table.

Why there is no cost column
---------------------------

The runtime records **usage**, not spend, and that is a deliberate choice
rather than a gap.

Providers do not return a price. OpenAI, Anthropic and Gemini return token
counts; only aggregators like OpenRouter put a cost on the response. So any
framework that reports money carries a price table, and prices change often
enough that every project which takes this seriously keeps that table outside
the library — LiteLLM ships a JSON file it re-fetches at runtime, Langfuse and
LangSmith price at ingestion, Pydantic AI defers to the separate
``genai-prices`` database.

There is a second reason, and it is the stronger one: **cached input tokens are
billed at a fraction of fresh ones**, so a cost derived from an
undifferentiated ``prompt_tokens`` is not merely stale, it is wrong by a
multiple. Kaval.AI therefore records ``cached_prompt_tokens`` and
``reasoning_tokens`` alongside the totals, wherever the provider reports them,
which is exactly what a price table needs to be applied correctly:

.. code-block:: python

   for call in await service.get_model_call_stats(call_type="llm", limit=5):
       fresh = (call.prompt_tokens or 0) - (call.cached_prompt_tokens or 0)
       print(call.model, fresh, call.cached_prompt_tokens,
             call.completion_tokens)

Pricing those numbers is a few lines against a table you control — and it keeps
the price of a model somewhere you can correct in an afternoon, rather than
inside a library release.

The trajectory: what the run actually did
-----------------------------------------

Every node visit writes exactly one ``tasks`` row, and **ordering by ``seq``
reconstructs the executed path** — including the interleaving of concurrent
``parallel`` branches, which ``created_at`` cannot (it is approximate, and ties
are unordered). Three columns carry that structure:

``seq``
   Position in the run's execution order.

``parent_task_name``
   The node that produced this row, set on the tool-call rows an agent node
   emits. A name rather than an id: nothing has to be allocated up front, and
   the readable column is the one that survives an export to a warehouse.

``tool_uri``
   The tool this row executed, set by both function nodes *and* agent tool
   calls — so one predicate finds every call to a tool regardless of whether a
   human wired it into the YAML or an agent chose it at step three.

That last point is what closed a real gap. An agent node used to produce one
row holding its final answer, and everything the agent actually *did* was built
during the run and then discarded — which made an agent failure
un-debuggable after the fact. Now:

.. code-block:: text

   seq  name              node_type   parent_task_name  tool_uri
   0    begin             start
   1    search            function                      python://web_search
   2    research          agent
   3    crawl_url         tool_call   research           python://crawl_url
   4    route             switch
   5    summarize         llm
   6    finish            end

Branch nodes record the decision they made, and the *value* they made it on:

.. code-block:: text

   name    = "route"
   inputs  = {"expr": "parsed.intent", "value": "refund"}
   output  = {"taken": "handle_refund", "matched": true}

The value is the diagnostic. Nine times in ten a mis-route is not a routing bug:
it is the upstream classifier emitting ``"Refund"`` or ``"refund "`` or
``"refunds"``. ``matched: false`` on a ``switch`` is the same signal
pre-computed — the model returned a label outside the enum and the run silently
took ``default``, which the engine also warns about in the log.

The backoffice renders all of this: tool calls indented under the node that made
them, and a branch as *expr = value → target*.

.. note::

   ``tasks`` is the biggest table you will own, and tool payloads are exactly
   where personal data lives. It grows without bound by design — the assumption
   is that you export it to a warehouse and do not retain it indefinitely, so
   schedule a retention job on day one (see :ref:`observability-retention`)
   rather than discovering the need later.

   ``max_payload_bytes`` on the task logger (256 KiB by default) replaces an
   oversized payload — of a node or of a model call — with a marker carrying
   its real size and a preview. It is there so one four-megabyte crawl result
   does not break the writer, the row and the backoffice task list — an
   operational limit, not a compliance control. The compliance controls are
   described in :ref:`observability-recording-less`.


Reading a trajectory without a database
---------------------------------------

:class:`~kavalai.workflow.tasklog.MemoryTaskLogger` records the same rows into a
list. Pass one per run to see exactly what a call did, from a notebook or a
test, with no database at all:

.. code-block:: python

   from kavalai.workflow.tasklog import MemoryTaskLogger

   tasklog = MemoryTaskLogger()
   state = await engine.run({"user_message": "hi"}, task_logger=tasklog)

   for row in tasklog.records:
       print(row.seq, row.name, row.node_type, row.tool_uri or "")

It overrides the engine's logger for that run only, so one engine can serve many
concurrent runs that each want their own trace — which is how a test can assert
on what a run *did* with no database anywhere near it.


.. _observability-external-id:

Marking non-production traffic
------------------------------

``Session.external_id`` is a caller-supplied key: pass your own user, ticket or
thread id and the engine reuses that conversation. Evaluation runs use a
structured prefix, which is a convention worth respecting:

.. code-block:: text

   eval:{tag}:{case}
   eval:pr-412:missing_quantity

``LIKE 'eval:%'`` then separates test traffic from real traffic in one
predicate; ``LIKE 'eval:pr-412:%'`` narrows it to one experiment, which is what
``--tag`` is for. The backoffice's **External ID** filter turns a failing case
in a run's output into the conversation that produced it — see
:doc:`evaluation`.

.. warning::

   Do not use the ``eval:`` prefix for production session ids, or you will not
   be able to tell them apart.


TaskLogger and fire-and-forget
------------------------------

``TaskLogger`` exposes ``log_node``, ``log_model_call``, ``flush``, and
``close``. Logging is **fire-and-forget** — writes happen in the background so
they never block a run. When you need the writes to land (e.g. at the end of a
test or a batch), await them explicitly:

.. code-block:: python

   await tasklog.flush()

.. _observability-recording-less:

Recording less
--------------

Everything above is recorded by default, and two parts of it are copies of the
conversation: a node's ``inputs``, ``output`` and ``prompt``, and a model
call's ``request_data``, which is the whole prompt — chat history and retrieved
passages included. A deployer who must not keep a second copy of the
transcript, under the GDPR for example, can leave those copies out. Every task
logger takes the same options:

``record_nodes``
   When ``False``, node executions are not recorded at all. Model calls still
   are.

``record_payloads``
   When ``False``, a node's ``inputs``, ``output`` and ``prompt`` and a model
   call's ``request_data`` and ``response_data`` are not stored. Names,
   timings, errors and token counts are.

``max_payload_bytes``
   The size cap described above, which applies to model-call payloads as it
   does to node payloads.

The engine adds one more: ``record_context=False`` keeps only the input and
the output in ``runs.context``, instead of every value each node produced.
Each option is stated where the logger or the engine is built, so what a
deployment leaves out can be read from the code that builds it; nothing is
left out silently.

In the example below a scripted client stands in for the model, so it runs
without a provider key; its token counts are estimates (see
:mod:`kavalai.testing`).

.. code-block:: python

   from uuid import UUID

   from pydantic import BaseModel

   from kavalai.agent_service import AgentService
   from kavalai.db import Run, db_manager
   from kavalai.testing import ScriptedLlmClient
   from kavalai.workflow import WorkflowBuilder
   from kavalai.workflow.tasklog import SqliteTaskLogger


   class Message(BaseModel):
       user_message: str


   class Topic(BaseModel):
       topic: str


   class Reply(BaseModel):
       agent_response: str


   await db_manager.init_sqlite()
   service = AgentService(db_manager.get_sqlite_sessionmaker())
   tasklog = SqliteTaskLogger("tasklog.db", record_payloads=False)
   scripted = ScriptedLlmClient([
       {"topic": "opening hours"}, {"agent_response": "From noon."},
       {"topic": "opening hours"}, {"agent_response": "Closed on Mondays."},
   ])

   engine = (
       WorkflowBuilder("Village guide", llm_model="openai/gpt-5.6-luna")
       .data_model("input", Message)
       .data_model("topic", Topic)
       .data_model("output", Reply)
       .start("classify")
       .llm("classify", prompt="Name the topic of the question.",
            inputs={"message": "input"}, output="topic", next="reply")
       .llm("reply", prompt="You are a concise guide to Green Village.",
            inputs={"message": "input"}, output="output", next="end")
       .end()
       .build_engine(agent_service=service, task_logger=tasklog,
                     record_context=False, client_factory=scripted)
   )
   state = await engine.run({"user_message": "Is the pub open on Sundays?"})

   for task in await tasklog.get_tasks(state.run_id):
       print(task["seq"], task["name"], task["inputs"], task["output"])
   for call in await tasklog.get_model_calls(state.run_id):
       print(call["model"], call["request_data"], call["total_tokens"])

   async with service.session_maker() as db:
       run = await db.get(Run, UUID(state.run_id))
   print(sorted(run.context))

.. code-block:: text

   0 start None None
   1 classify None None
   2 reply None None
   3 end None None
   openai/gpt-5.6-luna None 39
   openai/gpt-5.6-luna None 43
   ['input', 'output']

The path, the timings and the token counts remain; the prompts and the answers
do not, and ``runs.context`` no longer holds the topic the first node produced.
In production the same options go to
``PostgresTaskLogger(service, record_payloads=False)``.

.. _observability-custom-loggers:

Custom loggers and composition
------------------------------

A task logger is a subclass of ``TaskLogger`` with two hooks, ``write_node``
and ``write_model_call``. The base class calls them in the background, after
the recording options and the payload cap have been applied, and passes
``write_model_call`` the Pydantic ``ModelCallStat`` together with the
``agent_id``, ``session_id`` and ``run_id`` of the call. A logger with a
purpose of its own does not subclass the database logger. It sits beside it,
through ``TeeTaskLogger``, which hands every record to each of its loggers;
each keeps its own options. A meter that counts each conversation's tokens for
billing, beside the logger of the example above, for a second turn:

.. code-block:: python

   from collections import Counter

   from kavalai.workflow.tasklog import TaskLogger, TeeTaskLogger


   class Meter(TaskLogger):
       """Counts the tokens each conversation used, for billing."""

       def __init__(self):
           super().__init__(record_nodes=False, record_payloads=False)
           self.tokens = Counter()

       async def write_node(self, **record):
           """Not called: the meter records no nodes."""

       async def write_model_call(self, stats, *, agent_id, session_id,
                                  run_id):
           self.tokens[session_id] += stats.total_tokens or 0


   meter = Meter()
   both = TeeTaskLogger(tasklog, meter)
   later = await engine.run({"user_message": "And on Mondays?"},
                            session_id=state.session_id, task_logger=both)
   await both.flush()
   print(meter.tokens[later.session_id], later.token_usage["total_tokens"])

.. code-block:: text

   91 91

The meter's count equals the second run's ``token_usage``: ``task_logger=`` on
``run`` replaces the engine's logger for that run only, so the meter saw the
second turn and not the first.

.. _observability-cost-per-run:

Cost per run and per conversation
---------------------------------

Every model call row carries the ``agent_id``, ``session_id`` and ``run_id``
of the run that made it. The engine gives the run's ``TokenAccumulator`` those
ids as soon as the run is recorded, and a ``rag_query`` node hands the same
accumulator to its RAG service, so the embedding of a query is attributed in
the same way. What a run used is then one query:

.. code-block:: sql

   SELECT model,
          count(*)                               AS calls,
          sum(prompt_tokens)                     AS prompt_tokens,
          coalesce(sum(cached_prompt_tokens), 0) AS cached_tokens,
          sum(completion_tokens)                 AS completion_tokens
   FROM model_call_stats
   WHERE run_id = :run_id
   GROUP BY model;

For the first run above, and — with ``session_id = :session_id`` in the
``WHERE`` clause — for the whole conversation after the second turn:

.. code-block:: text

   model                calls  prompt_tokens  cached_tokens  completion_tokens
   openai/gpt-5.6-luna      2             67              0                 15

   model                calls  prompt_tokens  cached_tokens  completion_tokens
   openai/gpt-5.6-luna      4            141              0                 32

The query reads the same columns in the file of a ``SqliteTaskLogger``, as
here, and in the agent database that ``PostgresTaskLogger`` writes to, where
``service.get_model_call_stats(run_id=…)`` and ``session_id=…`` return the rows
themselves. None of the three ids is a foreign key, so the rows outlive the
conversation they belong to. Pricing the four columns remains outside the
library, for the reasons given above.

.. _observability-retention:

Retention
---------

Nothing in the runtime store is deleted on its own, and ``tasks`` and
``chat_messages`` grow with every turn.
:meth:`~kavalai.agent_service.AgentService.purge_sessions` deletes the
sessions whose last activity — ``sessions.updated_at``, the time of the last
run — is older than a cutoff:

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   cutoff = datetime.now(timezone.utc) - timedelta(days=90)
   async for batch in service.purge_sessions(cutoff, batch_size=500):
       print(f"deleted {len(batch)} sessions")

Against a database holding 1,200 sessions idle for 100 days, it prints:

.. code-block:: text

   deleted 500 sessions
   deleted 500 sessions
   deleted 200 sessions

Each batch is one transaction, oldest sessions first, and its session ids are
yielded once it is committed, so a caller can delete rows of its own keyed by
session in step. A session's runs, tasks and chat messages go with it through
the foreign keys. Its model calls stay: the token counts are a cost record,
while the payloads are the conversation again, so a purge sets
``request_data`` and ``response_data`` to ``NULL`` and keeps the rest.

.. code-block:: python

   (call,) = await service.get_model_call_stats(limit=1)
   print(call.total_tokens, call.request_data)

.. code-block:: text

   27 None

``agent_ids`` restricts a purge to some agents — the agents of one tenant, for
example. ``None`` means every agent, and an empty list purges nothing, so a
list computed from an empty selection cannot purge the whole database.
``delete_history_for_session`` treats a single session's model calls in the
same way, while keeping the session and its runs.

The backoffice UI
-----------------

All of this surfaces in the backoffice as **Conversations -> Runs -> Tasks**,
plus **Metrics** and **Model Calls** pages — letting you drill from a
conversation down to an individual node or model call. See :doc:`../ui/index`.
