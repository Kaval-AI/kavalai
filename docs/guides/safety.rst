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

Outbound requests
-----------------

A tool that fetches a URL the model composes is a path for server-side request
forgery: a prompt can steer the model towards ``http://169.254.169.254/``,
where a cloud instance serves its credentials, or towards a database port on
``localhost``. :mod:`kavalai.net` refuses such targets, and the bundled
``http_request`` and ``crawl_url`` tools apply it by default (see
:doc:`../reference/tools`).

:func:`~kavalai.net.is_public_address` accepts only globally routable unicast
addresses; an IPv4-mapped or NAT64 IPv6 address is judged as the IPv4 address
it reaches. :func:`~kavalai.net.ensure_public_url` checks a URL: the scheme,
credentials or a backslash in the authority, internal names such as
``localhost`` and ``*.internal``, and every address the host resolves to. IP
literals are read in every encoding the C library accepts, and non-ASCII names
are IDNA-encoded before the name checks run.

.. code-block:: python

   from kavalai.net import UnsafeUrlError, ensure_public_url, is_public_address

   print(is_public_address("93.184.216.34"))
   print(is_public_address("::ffff:169.254.169.254"))

   for url in (
       "http://0x7f.1:8080/admin",
       "http://printer.local/",
       "https://api.open-meteo.com/v1/forecast",
   ):
       try:
           print(await ensure_public_url(url))
       except UnsafeUrlError as error:
           print("refused:", error)

.. code-block:: text

   True
   False
   refused: 127.0.0.1 (0x7f.1) is a non-public address
   refused: printer.local is an internal host name
   https://api.open-meteo.com/v1/forecast

A check on the URL alone is defeated by DNS rebinding: the name resolves to a
public address when it is checked and to an internal one when the client
resolves it again to connect. :class:`~kavalai.net.PublicOnlyTransport`
removes the second lookup. Its httpcore network backend resolves the host,
checks every address and opens the socket to the address it checked, while TLS
verifies the certificate against the host name and sends it as SNI. Each
redirect is a new request through the same transport and is checked in the
same way. The transport serves any httpx client in your own tools:

.. code-block:: python

   import httpx
   from kavalai.net import PublicOnlyTransport, UnsafeUrlError

   async with httpx.AsyncClient(transport=PublicOnlyTransport()) as client:
       response = await client.get("https://example.com/")
       print(response.status_code)
       try:
           await client.get("http://localhost:8000/")
       except UnsafeUrlError as error:
           print("refused:", error)

.. code-block:: text

   200
   refused: localhost is an internal host name

The guard has two limits, both stated with the tools. With ``use_proxy=True``
the proxy resolves the name, so ``http_request`` can apply only the URL check.
``crawl_url`` drives a browser in another process, so only the requested and
the final URL are checked; full protection there requires network egress
control around the browser. An agent meant to reach an intranet is given a
tool built with ``allow_private_networks=True`` — an argument of the tool's
factory, not of the tool, so the switch rests with whoever registers the tool
and never with the model.

Deterministic testing
---------------------

A model's answer varies between calls, so a test of workflow logic —
branching, data flow, node wiring — replaces the model and keeps everything
else. :mod:`kavalai.testing` supplies the replacement.
:class:`~kavalai.testing.ScriptedLlmClient` answers each call with the next
reply from a list, and the engine accepts it as its ``client_factory``. Here it
drives the council desk workflow of :doc:`../cookbook/index`:

.. code-block:: python

   import asyncio

   from kavalai import WorkflowEngine
   from kavalai.testing import ScriptedLlmClient

   model = ScriptedLlmClient(
       [
           {"intent": "permit"},
           {"agent_response": "Apply at the parish office."},
       ]
   )
   engine = WorkflowEngine.from_yaml_path(
       "council_desk.yaml", client_factory=model
   )
   state = asyncio.run(engine.run({"user_message": "May I build a shed?"}))

   print(state.trace)
   print(state.output_data)
   print(model.calls[1].model, model.calls[1].response_model.__name__)

.. code-block:: text

   ['start', 'classify', 'route', 'permit_reply', 'end']
   {'agent_response': 'Apply at the parish office.'}
   openai/gpt-5.6-luna output

Only the provider call is replaced. Each reply passes through the same
streamer, retry and restart handling as a provider's answer, and the engine
validates it into the node's data type, so a scripted reply that does not fit
the type fails the run exactly as a model's reply would. The client records
every request in ``model.calls`` — the last line shows that the second call was
made as ``openai/gpt-5.6-luna`` and asked for the ``output`` data type — and
reports a :class:`~kavalai.ModelCallStat` for each call, with token counts
estimated from the text, so ``state.token_usage`` and a task logger see the run
as they would in production.

Workflow logic thereby becomes testable and repeatable without a single live
model call, which complements the run-level auditing described in
:doc:`observability`. :func:`~kavalai.testing.fake_providers` extends the
substitution to a workflow that names its model rather than receiving a
factory, and :class:`~kavalai.testing.FakeEmbeddingClient` extends it to
retrieval; the cookbook recipe "Testing a workflow without calling a model"
covers both.
