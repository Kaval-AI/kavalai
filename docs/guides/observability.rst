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
them). See :doc:`../tutorials/llm_clients`.

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
   schedule a ``DELETE FROM tasks WHERE created_at < …`` job on day one rather
   than discovering the need later.

   ``max_payload_bytes`` on the task logger (256 KiB by default) replaces an
   oversized payload with a marker carrying its real size and a preview. It is
   there so one four-megabyte crawl result does not break the writer, the row
   and the backoffice task list — an operational limit, not a compliance
   control.


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
concurrent runs that each want their own trace. This is what the evaluation
runner uses, and it is why trajectory assertions need no database — see
:doc:`evaluation`.


.. _observability-external-id:

Marking non-production traffic
------------------------------

``Session.external_id`` is a caller-supplied key: pass your own user, ticket or
thread id and the engine reuses that conversation. Evaluation runs use a
structured prefix, which is a convention worth respecting:

.. code-block:: text

   eval:{suite}:{tag}:{case}:{repeat}
   eval:bakery-acceptance:pr-412:vague_quantity:0

``LIKE 'eval:%'`` then separates test traffic from real traffic in one
predicate, and the backoffice's **External ID** filter turns a failing case in a
result file into the conversation that produced it.

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

The backoffice UI
-----------------

All of this surfaces in the backoffice as **Conversations -> Runs -> Tasks**,
plus **Metrics** and **Model Calls** pages — letting you drill from a
conversation down to an individual node or model call. See :doc:`../ui/index`.
