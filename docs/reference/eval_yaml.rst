Evaluation YAML & CLI
=====================

Every key in a case file, every command-line flag of ``kavalai-eval``, and the
Python API underneath both. For what these are *for*, read
:doc:`../guides/evaluation`.

.. contents:: On this page
   :local:
   :depth: 2


The case file
-------------

One file, one suite. It is a :class:`~kavalai.eval.EvalSuite`, and it is
validated in full before a single case runs.

.. code-block:: yaml

   name: green-village
   judge_model: openai/gpt-5.4-mini      # optional

   cases:
     - name: president
       input:
         user_message: Who is the president of Green Village?
       expected:
         agent_response: {contains: [Thomas Cook]}

     - name: no_budget
       type: judge
       input:
         user_message: What is the village's annual budget?
       expected: >-
         The answer says the information is not available instead of
         inventing a figure.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``name``
     - Names the suite in the run's header line. Required.
   * - ``judge_model``
     - ``provider/model`` grading the judged cases. Defaults to
       :data:`~kavalai.eval.DEFAULT_JUDGE_MODEL`
       (``openai/gpt-5.4-mini``); ``--judge-model`` overrides it for one run.
   * - ``cases``
     - The cases, run in the order they are written.

There is deliberately **no** ``base_url`` **key**. Which agent a suite grades is
a property of the run, not of the cases — see
:ref:`the guide <eval-sessions>`.

Unknown keys are refused, in the suite and in every case. A silently ignored key
is a case that never ran.


A case
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``name``
     - How the case is reported, and what its session is recorded under.
       Required.
   * - ``type``
     - ``simple`` (default) compares the answer with expected values;
       ``judge`` asks a model whether the answer is acceptable.
   * - ``input``
     - Field values for the agent's input type. Validated against that type
       before the call, so a mistyped field is an error rather than a puzzling
       answer.
   * - ``expected``
     - A mapping of output field to expected value or matcher for a simple
       case; a plain-language criterion for a judged one.

Two combinations are refused by :class:`~kavalai.eval.EvalCase` itself, because
both would otherwise pass on any answer whatsoever:

.. code-block:: text

   Case 'x' is judged, so `expected` must be a plain-language criterion.
   Case 'y' is simple, so `expected` must map output fields to expected
   values. Use `type: judge` to grade a plain-language criterion.


Matchers
--------

A simple case's ``expected`` maps an output field to either a literal value or a
mapping of matcher names. A bare value is shorthand for ``equals``:

.. code-block:: yaml

   expected:
     status: needs_details                       # equals
     order_id: {equals: ''}
     missing: {equals: ['items[0].quantity']}
     agent_response:
       contains: ["1.2"]
       not_contains: ["approximately"]

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Matcher
     - Argument
     - Passes when
   * - ``equals``
     - Any value
     - The field equals it exactly.
   * - ``contains``
     - A value, or a list of them
     - Text: every argument appears as a substring, **case-insensitively**.
       List, tuple or set: every argument is a member; mapping: every argument
       is a key. Any other type never contains anything.
   * - ``not_contains``
     - A value, or a list of them
     - No argument is contained, by the same rule.
   * - ``regex``
     - A pattern
     - :func:`re.search` finds it in the field rendered as text.
   * - ``one_of``
     - A list of values
     - The field is one of them.

Matchers on one field are all checked, and each failure is reported separately.
Fields the expectation does not mention are ignored, so a case states what it
cares about and nothing more. Naming a field the agent's output does not have is
a failure, not a skip.

A mapping is read as matchers only when **every** key is a matcher name, so an
agent that genuinely answers with a dictionary can still be compared with
``equals``:

.. code-block:: yaml

   expected:
     totals: {cases: 26, passed: 26}      # a literal dict, not matchers

An empty or absent ``expected`` asserts only that the agent answered.


``kavalai-eval``
----------------

.. code-block:: console

   $ kavalai-eval <cases.yaml> --port <port> [options]

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Flag
     - Meaning
   * - ``suite``
     - Path to the YAML file of cases. Positional, required.
   * - ``--port``
     - Agent server port. **Required**: which agent is being evaluated is
       never left to a default.
   * - ``--host``
     - Agent server host. Default ``localhost``.
   * - ``--tag``
     - Names this run inside each case's ``external_id`` — a model version, a
       prompt variant, a build. Without it, the sessions of two runs cannot be
       told apart afterwards.
   * - ``--auth USER:PASSWORD``
     - HTTP basic auth, when the server has
       ``KAVALAI_AGENT_BASIC_AUTH_USER`` / ``_PASSWORD`` configured.
   * - ``--judge-model``
     - ``provider/model`` grading the judged cases, overriding the suite's
       ``judge_model``.
   * - ``--timeout``
     - Seconds to wait for one agent run. Default ``120``.

The run prints a header, one line per case as it finishes, and a count:

.. code-block:: text

   bakery-email-assistant: 26 cases tagged baseline against http://localhost:25100

   PASS  order_single_item
   FAIL  missing_quantity  — order_id: expected '', got 'ord-0007'
   ...

   25/26 passed

.. list-table::
   :header-rows: 1
   :widths: 10 22 68

   * - Exit
     - Constant
     - Meaning
   * - ``0``
     - ``EXIT_PASSED``
     - Every case passed.
   * - ``1``
     - ``EXIT_FAILED``
     - At least one case failed.
   * - ``2``
     - ``EXIT_ERROR``
     - The run never reached a verdict: the file would not load, or the run
       itself broke.

The constants live in ``kavalai.eval.eval_runner``, so a test asserting on an
exit code names the meaning rather than the number.

A failing *agent call* is not an ``EXIT_ERROR``. It fails its own case with the
error as the reason and the run continues, so one unreachable case cannot end a
suite.


From Python
-----------

The console script is a thin wrapper over four public pieces.

``load_suite`` and ``run_suite``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kavalai.eval import load_suite, run_suite

   suite = load_suite("examples/bakery/eval_cases.yaml")
   results = await run_suite(
       suite,
       base_url="http://localhost:25100",
       tag="ci",
       on_result=lambda r: print(r.name, "ok" if r else r.reason),
   )

:func:`~kavalai.eval.run_suite` takes ``base_url``, ``username``, ``password``,
``timeout``, ``judge_model``, ``tag``, ``transport`` and ``on_result``, and
returns one :class:`~kavalai.eval.EvalResult` per case in file order.
``on_result`` is called with each verdict as it arrives — it is how the CLI
prints progress, and how a test can stream one.

The evaluators
~~~~~~~~~~~~~~

.. code-block:: python

   from kavalai.eval import JudgeEvaluator, SimpleEvaluator

   simple = SimpleEvaluator("http://localhost:25000", tag="ci")
   judge = JudgeEvaluator(
       "http://localhost:25000",
       tag="ci",
       model="openai/gpt-5.4-mini",
   )

Both take ``base_url`` (no default), ``username``, ``password``, ``timeout``,
``tag`` and ``transport``; :class:`~kavalai.eval.JudgeEvaluator` adds ``model``,
``llm_client`` and ``prompt``. ``transport`` is an ``httpx`` transport, which is
what lets a test serve the requests with no network at all.

``evaluate(inputs, expected, name=...)`` runs one case and returns its verdict.
It raises only for a judged case with no criterion; everything else — a refused
connection, a rejected input, a judge that fell over — comes back as a failed
result with the reason attached.

Overriding the judge:

``model``
   A ``provider/model`` name resolved through
   :func:`~kavalai.make_client` on first use (:doc:`providers`). Nothing is
   built until a case is actually judged, so a run of literal cases needs no
   API key.

``llm_client``
   A ready-made :class:`~kavalai.BaseLlmClient`, used instead of ``model``.

``prompt``
   The grading prompt, which must accept ``{inputs}``, ``{output}`` and
   ``{criterion}``. The default instructs the judge to grade the stated
   criterion and nothing else, and to answer with a
   :class:`~kavalai.eval.JudgeVerdict` — ``passed`` and a one-sentence
   ``reason`` when it is false.

``EvalResult``
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Field
     - Meaning
   * - ``name``
     - The case name.
   * - ``passed``
     - Whether the answer satisfied the expectation. ``bool(result)`` is this,
       so ``if not result`` reads correctly.
   * - ``reason``
     - Why it failed; empty when it passed. Every failing matcher, joined with
       ``;``, or the judge's own sentence.
   * - ``inputs``
     - What was sent to the agent.
   * - ``output``
     - What the agent answered, or ``None`` when the run never got that far.

``check_output``
~~~~~~~~~~~~~~~~

The matcher engine is exported on its own, for asserting on a payload you
already have — a recorded answer, a fixture, an object built in a test — without
a server:

.. code-block:: python

   from kavalai.eval import check_output

   failures = check_output(
       {"status": "needs_details", "order_id": ""},
       {"status": "needs_details", "order_id": {"equals": ""}},
   )
   assert not failures, failures

It returns a list of failure messages, empty when everything matched.


Sessions
--------

Each case runs in a fresh session, recorded by the agent server under

.. code-block:: text

   eval:{tag}:{case}      # with --tag
   eval:{case}            # without

built by ``AgentEvaluator.external_id``. Sessions are written only when the
server has an :class:`~kavalai.agent_service.AgentService`; the evaluators
behave identically when it does not. ``eval:`` is the reserved prefix the
backoffice Conversations page filters on — see :doc:`../guides/observability`.


Environment
-----------

:mod:`kavalai.eval` reads **no environment variables** of its own. The base URL,
the auth pair, the judging model and the timeout are all arguments, which is
what lets a suite run from a notebook or a test without a hidden dependency on
the shell.

What does read the environment is the provider client a judged case builds —
``OPENAI_API_KEY`` and its equivalents (:doc:`config`). That is why a run with
judged cases is written ``dotenv run kavalai-eval …`` while a run of purely
literal cases needs nothing at all.
