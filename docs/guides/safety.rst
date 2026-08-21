======
Safety
======

Kaval.AI's mission is to make agentic AI pipelines **predictable, observable,
and safe**. Safety is not a single feature but a set of guarantees baked into
the runtime: data is typed at every boundary, control-flow expressions are
evaluated without ``eval``, agents are explicitly bounded, and any pipeline can
be run fully offline for deterministic testing.

For a hands-on walkthrough, see :doc:`../tutorials/workflow`.

Typed I/O everywhere
--------------------

Every value that crosses a boundary is validated. Workflow ``data_types`` are
JSON-schema fragments compiled to Pydantic models, so each node input and output
is checked (see :doc:`workflows`). Likewise, every tool in the
:class:`~kavalai.FunctionKernel` has a Pydantic input and output model, and the
kernel validates and coerces arguments before a call and the return value after
(see :doc:`tools`). A malformed value is caught at the boundary, not deep inside
a downstream step.

A safe expression language
--------------------------

Branching nodes (``if`` / ``switch``) and context lookups are powered by a small
expression language in :mod:`kavalai.workflow.expressions`:
:func:`~kavalai.evaluate_expression`, ``evaluate_bool``, ``evaluate_value``, and
``ExpressionError``.

Expressions are evaluated **safely via an AST whitelist — never Python eval**.
Only comparisons, ``and`` / ``or`` / ``not``, ``in``, arithmetic, and
dotted/indexed access into the context are permitted. Unknown names resolve to
``None``, so guard checks degrade gracefully rather than crashing.

Bounded agents, explicit tools
------------------------------

Agentic loops are constrained on two fronts. An :class:`~kavalai.Agent` (and the
``agent`` workflow node) is bounded by an explicit ``max_steps`` so it cannot
run away (see :doc:`agents`). And it can only act through tools addressed
explicitly by URI — there is no implicit capability, only the tools you have
registered with the kernel.

Deterministic testing
---------------------

Because LLMs are non-deterministic, Kaval.AI lets you replace them entirely for
testing. Inject a ``client_factory`` into the engine to run a workflow fully
offline with canned model output:

.. code-block:: python

   from kavalai import BaseLlmClient, WorkflowEngine


   class StubClient(BaseLlmClient):
       """Returns canned structured output — no network, no API key."""

       def __init__(self, *args, **kwargs):
           super().__init__()

       async def _run_chat_completions(
           self, chat_history, response_model, streamer
       ):
           value_streamer = streamer.get_value_streamer(
               "response", response_model=response_model
           )
           canned = response_model(
               **{
                   name: ("repair" if name == "intent" else "Stubbed reply.")
                   for name in response_model.model_fields
               }
           )
           await value_streamer.stream_partial(canned.model_dump_json())
           await value_streamer.stream_complete()


   engine = WorkflowEngine.from_yaml(
       yaml,
       client_factory=lambda model, parameters, stats_receiver: StubClient(),
   )

Note the method: every run goes through :meth:`~kavalai.WorkflowEngine.run_stream`,
so the engine drives clients via ``_run_chat_completions``, pushing values into a
streamer. A stub that overrides ``chat_completions`` instead is never called.

This makes workflow logic — branching, data flow, node wiring — testable and
repeatable without a single live model call, which complements the run-level
auditing described in :doc:`observability`.
