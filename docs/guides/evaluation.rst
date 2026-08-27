Evaluation & acceptance testing
===============================

An agent that works when you try it and fails on the third customer is not
working. Evaluation is how you find out which one you have — before you deploy,
rather than afterwards.

:mod:`kavalai.eval` is deliberately small. It is two evaluators and a file of
cases, pointed at an agent server that is **already running**:

.. code-block:: console

   $ uv run --env-file .env kavalai-eval examples/bakery/eval_cases.yaml \
       --port 25100 --tag baseline

It prints one line per case, a count at the end, and exits ``0``, ``1`` or
``2``. That exit code is the whole product: it is what turns a file of cases
into a gate.

.. contents:: On this page
   :local:
   :depth: 1


What is under test
------------------

The evaluators speak HTTP and nothing else. They discover the agent's input and
output types from the server's OpenAPI specification (through
:class:`~kavalai.client.AgentClient`), send one input, and judge what comes
back. They know nothing about the workflow engine, the YAML graph, the tools or
the database.

That is a deliberate boundary rather than a missing feature:

- **The artefact under test is the deployment.** A suite that imports your
  workflow grades a graph; a suite that calls a server grades the thing you are
  about to promote, with its real tools, its real network and its real secrets.
- **The same cases run anywhere.** A laptop, a staging deployment and two model
  versions in turn are three values of ``--port``, not three files.
- **The suite cannot drift from the agent.** Inputs are validated against the
  agent's *own* input type before they are sent, so a mistyped field is a clear
  error rather than a puzzling answer.

The cost of the boundary is equally plain: an evaluator cannot see which branch
the run took, which tool it called, or what retrieval returned. What it can see
is what a caller sees. Where a run's internals matter, read them afterwards in
the backoffice — every graded case is an ordinary session (see
:ref:`eval-sessions`).


Two evaluators
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * -
     - :class:`~kavalai.eval.SimpleEvaluator`
     - :class:`~kavalai.eval.JudgeEvaluator`
   * - Expectation
     - Output field → value or matcher
     - A plain-language criterion
   * - Calls a model
     - No
     - Yes, one per case
   * - Verdict
     - The same every time
     - A judgement, and it can move
   * - Cost
     - Nothing
     - Tokens
   * - Use for
     - Facts, ids, numbers, classifications, anything a rule decides
     - Explanations, refusals, comparisons — substance fixed, wording free

**Reach for the deterministic one first, and one line sooner than feels
natural.** A judge costs money, can flake, and is a dependency that moves under
you: the day a model alias re-points, every historical verdict becomes
incomparable, and you cannot tell a regression in your agent from a change in
its grader.

Most of the time the way to earn a literal comparison is to change the *agent*
rather than the evaluator. The bakery example is the argument: its workflow
lets the model extract fields and lets ordinary Python decide whether the
resulting order is complete, so ``status``, ``order_id`` and ``missing`` are
values to compare rather than prose to interpret — and nineteen of its
twenty-six cases need no judge at all. See :doc:`../cookbook/bakery_eval`.


Checking an answer literally
----------------------------

:class:`~kavalai.eval.SimpleEvaluator` compares the fields an expectation
mentions and ignores the rest, so a case states what it cares about and nothing
more:

.. code-block:: python

   from kavalai.eval import SimpleEvaluator

   evaluator = SimpleEvaluator("http://localhost:25000")
   result = await evaluator.evaluate(
       {"user_message": "Who is the president of Green Village?"},
       {"agent_response": {"contains": "Thomas Cook"}},
   )
   assert result.passed, result.reason

Five matchers, and a bare value as shorthand for ``equals``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Matcher
     - Meaning
   * - ``equals``
     - Exact equality. A bare value — ``status: needs_details`` — means this.
   * - ``contains``
     - Substring for text, membership for a list. A list of arguments requires
       every one of them.
   * - ``not_contains``
     - The negation, argument for argument.
   * - ``regex``
     - :func:`re.search` over the field rendered as text.
   * - ``one_of``
     - The value is one of the listed alternatives.

``contains`` compares text **case-insensitively**. A model that writes "Marsh
Marigold" where the fact says "marsh marigold" has not made a mistake worth
failing a build over, and a suite that fails on capitalisation is a suite people
learn to ignore.

Several matchers may be combined on one field, and each is checked
independently:

.. code-block:: yaml

   expected:
     agent_response:
       contains: ["1.2"]
       not_contains: ["4 metres", "approximately"]

An empty expectation is legitimate: it asserts that the agent answered at all,
which is a smoke test worth having in its own right.


Letting a model decide
----------------------

Some correct answers cannot be written down in advance. Whether a chatbot
*declined* to invent a figure, whether a reply asked for the right missing
detail, whether an explanation actually compared the two things asked about —
these are matters of substance with free wording, and a literal matcher either
passes everything or fails correct answers.

:class:`~kavalai.eval.JudgeEvaluator` takes the criterion in plain language and
reports the judge's own reason when a case fails:

.. code-block:: python

   from kavalai.eval import JudgeEvaluator

   evaluator = JudgeEvaluator("http://localhost:25000")
   result = await evaluator.evaluate(
       {"user_message": "What is the village's annual budget?"},
       "The answer says the information is not available instead of "
       "inventing a number.",
   )

The grading prompt instructs the judge to grade *the stated criterion and
nothing else* — not style, not length, not politeness unless the criterion asks
about them — because a judge given latitude invents requirements, and a case
that fails for an unstated reason is indistinguishable from a broken agent.

The judging model is built on first use, so a run of purely literal cases needs
no API key at all. It defaults to :data:`~kavalai.eval.DEFAULT_JUDGE_MODEL` and
is overridden per suite (``judge_model:``) or per run (``--judge-model``).

.. tip::

   Write a criterion a person could check by reading the answer once —
   "the reply says the order has been received" rather than "the reply is
   helpful". A vague criterion does not produce a lenient grader; it produces
   an inconsistent one.

.. warning::

   Judged cases are a sample, not a measurement. Before trusting a criterion to
   gate a deployment, run it against a handful of answers you have labelled by
   hand and check that the judge agrees with you. A judge that disagrees with
   you a fifth of the time is generating noise that will be read as signal.


A file of cases
---------------

A suite is one YAML file. It names itself, optionally names the judging model,
and lists cases in the order they should run:

.. code-block:: yaml

   name: green-village

   cases:
     - name: direct_fact-00
       input:
         user_message: Who is the president of Green Village?
       expected:
         agent_response: {contains: [Thomas Cook]}

     - name: unanswerable_00
       type: judge
       input:
         user_message: What is Green Village's annual budget?
       expected: >-
         The answer says the Green Village facts do not cover this,
         instead of inventing a figure.

A case is ``simple`` unless it says ``type: judge``. ``input`` is field values
for the agent's input type; ``expected`` is a mapping for a simple case and a
criterion for a judged one.

**A case file never names the server it grades.** There is no ``base_url`` key,
``--port`` is required, and :class:`~kavalai.eval.AgentEvaluator` has no default
base URL. The agent under evaluation is a property of the *run*, not of the
cases — which is exactly what makes two model versions comparable: run the same
file twice, one ``--tag`` each.

See :doc:`../reference/eval_yaml` for every key.


What the file will not accept
-----------------------------

:func:`~kavalai.eval.load_suite` validates before anything runs, and refuses
three things outright. Each refusal exists because the alternative is a case
that reports a pass without having tested anything:

- **A judged case with no criterion.** Judging against nothing passes on any
  answer at all. It is refused in :class:`~kavalai.eval.EvalCase` and again in
  the evaluator.
- **A simple case whose** ``expected`` **is a string.** The literal matcher
  would look for output fields in a sentence and find none, so the case would
  quietly assert nothing. The error names ``type: judge`` as the fix.
- **A key nobody recognises.** Both models forbid extra fields. A silently
  ignored key is a case that never ran, and the moment it matters is the moment
  nobody is looking.

A case file is therefore worth a unit test of its own, and one that costs
neither a server nor a key:

.. code-block:: python

   from kavalai.eval import load_suite

   def test_the_shipped_cases_are_a_valid_suite():
       suite = load_suite("examples/green_village/eval_cases.yaml")
       assert {case.type for case in suite.cases} == {"simple", "judge"}

``examples/green_village/test_eval_cases.py`` goes one step further and
constructs the chatbot's input model from every case, so a field renamed in the
agent fails the test rather than sixty-four cases at run time.


Running a suite
---------------

Bring the agent up:

.. code-block:: console

   $ dotenv run python -m examples.green_village.green_village_support_in_memory
   Serving Green Village support on http://0.0.0.0:25000

and grade it from another shell:

.. code-block:: console

   $ dotenv run kavalai-eval examples/green_village/eval_cases.yaml \
       --port 25000 --tag baseline

.. code-block:: text

   green-village: 64 cases tagged baseline against http://localhost:25000

   PASS  direct_fact-00
   PASS  direct_fact-01
   FAIL  direct_fact-04  — agent_response: 'The official flower is the
         cowslip.' is missing ['marsh marigold']
   ...

   63/64 passed

Cases run **one at a time**, in file order. An evaluation that is easy to read
while it runs is worth more than one that finishes a few seconds sooner, and a
suite whose cases share a database or an order book gives the same answer twice
only if they do not overlap.

Three exit codes, and the third is the one that matters:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - ``0``
     - Every case passed.
   * - ``1``
     - At least one case failed. The agent is wrong.
   * - ``2``
     - The run never reached a verdict — the file would not load, or the run
       itself broke.

Continuous integration has to be able to tell "the suite is broken" from "the
agent is wrong". Collapsing the two teaches people that a red build means
somebody forgot to start the server.

**One failing case cannot end a run.** An agent call that raises — a connection
refused, an input the agent will not accept, a timeout — fails *its* case with
that error as the reason and the run continues. A judge that fails does the
same, and says it was the judge. Twenty-five useful verdicts and one honest
error beat a stack trace on case two.


.. _eval-sessions:

Naming a run, and reading it afterwards
---------------------------------------

Every case runs in a **fresh session**, so no case can leak conversation history
into the next one. When the agent server has an
:class:`~kavalai.agent_service.AgentService` — as both example servers do — that
session is recorded under a structured external id:

.. code-block:: text

   eval:{tag}:{case}
   eval:baseline:direct_fact-04

Without ``--tag`` it is ``eval:{case}``. The evaluators behave identically
against a server that records nothing; persistence is the server's property,
not the suite's.

Three prefixes, three questions:

- ``eval:baseline:direct_fact-04`` — the exact conversation that failed, with
  its runs and its task rows, in the backoffice **Conversations** page's
  *External ID* filter.
- ``eval:baseline:`` — everything in that one run.
- ``eval:`` — all test traffic, separated from all real traffic in one
  predicate.

``--tag`` is what names an experiment: a model version, a prompt variant, a
build number. Two runs that share a tag cannot be told apart afterwards, which
is the whole reason the flag exists. See :doc:`observability`.

.. warning::

   ``eval:`` is a reserved prefix by convention. Do not use it for production
   session ids, or the one predicate that separates test traffic from real
   traffic stops working.


From a test
-----------

Both evaluators are meant to be called straight from ``pytest``. There is no
harness to configure and no fixture to inherit — an evaluator is an object with
one asynchronous method:

.. code-block:: python

   import pytest
   from kavalai.eval import SimpleEvaluator

   @pytest.mark.asyncio
   async def test_the_president_is_named():
       evaluator = SimpleEvaluator("http://localhost:25000", tag="ci")
       result = await evaluator.evaluate(
           {"user_message": "Who is the president of Green Village?"},
           {"agent_response": {"contains": "Thomas Cook"}},
           name="president",
       )
       assert result.passed, result.reason

A whole file of cases runs the same way through
:func:`~kavalai.eval.run_suite`, which is what the console script itself calls:

.. code-block:: python

   from kavalai.eval import load_suite, run_suite

   results = await run_suite(
       load_suite("examples/bakery/eval_cases.yaml"),
       base_url="http://localhost:25100",
       tag="ci",
   )
   assert all(r.passed for r in results), [r.reason for r in results if not r]

:class:`~kavalai.eval.EvalResult` is truthy exactly when its case passed, so
``if not result`` reads the way it should.

.. note::

   :mod:`kavalai.eval` reads **no environment variables**. Only
   ``eval_runner.py:main()`` does. Everything else — the base URL, the auth
   pair, the judging model, the timeout — is an argument. That is what lets a
   suite run from a notebook or a test without a hidden dependency on the
   shell, and it is the same rule the rest of the library follows
   (:doc:`../reference/config`).


Grading against a world you do not control
------------------------------------------

An agent that reads the live web cannot be graded like one that reads a fixed
corpus. ``examples/business_info_agent/eval_cases.yaml`` is the worked example:
ten cases against an agent that searches DuckDuckGo, crawls whatever pages it
decides to crawl, and summarises what it read. Nothing about that is stable
from one week to the next.

The rule those cases follow is to **assert only what a business keeps stating
about itself** — its name, its own domain, the kind of business it is — and
never a page's wording, which result came first, or a figure a site may revise:

.. code-block:: yaml

   - name: kavalai_website
     input:
       business_query: Kaval.AI (kaval.ai)
     expected:
       website: {regex: '(?i)kaval\.ai'}

A case that fails when a homepage is redesigned grades the web rather than the
agent, and a suite that goes red for reasons nobody can act on is a suite people
stop reading.

The case the example exists for is the one about a business that does not
exist:

.. code-block:: yaml

   - name: unknown_business_is_not_invented
     type: judge
     input:
       business_query: Vorrembling Tsakumets Kringlefabrik OU
     expected: >-
       The answer does not invent an identity for a business it could not
       find. `address`, `website`, `phone` and `owners` are null rather than
       filled in with plausible-looking values, and the description and summary
       say the business could not be found or that nothing is known about it,
       instead of describing a company as though it exists.

For a research agent, the interesting behaviour is the empty field rather than
the fluent paragraph — and an empty field is something a judged criterion can
insist on while a matcher over free text cannot.


What this cannot tell you
-------------------------

Worth knowing on the first day rather than in the third month.

**The suite quietly becomes the specification.** Once a file of cases gates a
deployment, people tune until it is green, and the agent ends up excellent at
those sixty-four cases and no better at anything else. Watch the *first-contact
failure rate* — how many newly written cases fail the first time they are run.
That number rising while the pass rate rises is the signature of overfitting.

**Written cases have a distribution, and it is not your users'.** Cases written
by hand, or by a model, are better punctuated, more polite and more on-topic
than real traffic. A suite built entirely from them will be green while
production struggles with top-posted replies, mobile signatures and three
questions in one paragraph. Write the ugly tail deliberately.

**Side effects are real.** The agent does not know it is being evaluated, and
nothing here stops it writing to whatever it writes to. Grade a side-effecting
agent against a deployment whose world is disposable — the bakery example keeps
its order book in an ordinary Python list for exactly this reason — and never
point a suite at production.

**Prompt injection cases are a regression guard, not a security assessment.** A
fixed set of injection strings tests yesterday's attacks. The durable protection
is a deterministic rule the model cannot reach: the bakery caps the quantity one
email may order unattended, and no amount of persuasive text moves that. See
:doc:`safety`.

**A green suite means the agent does what you specified.** Whether what you
specified is right is a different question, and the answer to it comes from
production.


Where to go next
----------------

- :doc:`../reference/eval_yaml` — every key in a case file, every command-line
  flag, and the Python API.
- :doc:`../cookbook/green_village_eval` — grading a RAG chatbot: sixty-four
  cases over a fictional village, literal where the answer is a fact and judged
  where it is not.
- :doc:`../cookbook/bakery_eval` — grading a workflow with side effects, where
  the assertion is about what did *not* end up in the order book.
- :doc:`observability` — the sessions, runs and task rows a graded case leaves
  behind.
