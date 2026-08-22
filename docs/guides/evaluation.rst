Evaluation & acceptance testing
===============================

An agent that works when you try it and fails on the third customer is not
working. Evaluation is how you find out which one you have — before you deploy,
not after.

Kaval.AI's evaluation tooling is a **library over files**. A suite is a
directory: a dataset of cases, the personas that talk to your agent, and one
``suite.yaml`` tying them to a target and a threshold. Running it is one
command, with no database and no service:

.. code-block:: console

   $ uv run --env-file .env kavalai-eval examples/bakery/eval/suite.yaml --tag pr-412

It prints a table, writes ``results/pr-412.json`` and a JUnit XML file, and
exits ``0`` or ``1``. That exit code is the whole product: it is what turns a
suite into a gate.

.. contents:: On this page
   :local:
   :depth: 1


Your first suite, in three files
--------------------------------

Suppose you have ``greeter.yaml`` — an ``llm`` node that greets a visitor.
Alongside it, make an ``eval/`` directory with two files.

The cases:

.. code-block:: yaml

   # eval/cases.yaml
   name: greeter_cases
   cases:
     - name: greets_by_name
       inputs: {user_message: "Hi, I'm Agnes."}
       expected: {contains: "Agnes"}

And what to run them against:

.. code-block:: yaml

   # eval/suite.yaml
   name: greeter-acceptance
   dataset: cases.yaml
   target:
     kind: engine
     workflow: ../greeter.yaml
   evaluators:
     - no_error
     - contains
   gate:
     min_pass_rate: 1.0

Then:

.. code-block:: console

   $ uv run --env-file .env kavalai-eval eval/suite.yaml --tag first

.. code-block:: text

   greeter-acceptance · engine ../greeter.yaml · 1 cases

    case            slice  verdict  tokens  seconds
    greets_by_name         pass         77      4.5

      pass rate 1.00 · 77 tokens
      gate passed

      wrote eval/results/first.json
      wrote eval/results/first.junit.xml

That is a working gate: it exits ``0``, and it exits ``1`` the day the greeting
stops using the visitor's name.

From here, four things are worth adding, roughly in this order:

1. **More cases**, and a ``slice:`` on each so you can hold different parts to
   different standards.
2. **Trajectory assertions** — ``tool_called``, ``branch_taken`` — once the
   workflow does more than answer.
3. ``kavalai-eval accept`` to **commit a baseline**, so a case that used to
   pass and now fails is a named regression rather than a moved number.
4. ``--record-fixtures`` once, so the whole thing then **runs in CI for
   nothing**.

The rest of this page is why each of those matters.


Why files
---------

Datasets, personas and thresholds live on disk, next to the workflow they
guard, and the accepted baseline is **committed to git**. That is a deliberate
choice rather than a missing feature:

- A dataset in git is diffable and reviewable. Adding a case is a pull request,
  and so is changing what "correct" means.
- Accepting new behaviour *is* a commit. A regression shows up as a readable
  diff in code review rather than a number in a dashboard nobody opens, and
  there is no pointer for anyone to forget to advance.
- A suite that needs no database runs on every pull request. One that needs a
  database runs when somebody remembers.

Nothing about evaluation writes to your agent database unless you ask it to
(see :ref:`eval-persisting`).


The four things in a suite
--------------------------

**A case** is one input and how to grade it. **A dataset** is a list of cases.
**A target** is the thing under test. **Evaluators** turn a run into scores.

.. code-block:: yaml

   # eval/cases/orders.yaml
   name: bakery_orders
   cases:
     - name: missing_quantity
       slice: order_incomplete
       inputs:
         email:
           sender: peeter@example.test
           subject: bread
           body: "Hi, Peeter here. Send a few sourdough loaves on 2026-09-10."
       expected:
         branch: reply_clarify
         orders_after: 0
         missing: ["items[0].quantity"]

The ``expected`` block is yours to shape: evaluators read the keys they care
about. Note what this case asserts — *no order was stored*. That is the failure
people actually ship: a helpful model resolving "a few loaves" to three and
writing a perfectly polite confirmation of an order the customer never placed.
It fails silently, and it looks like success in every text metric.

See :doc:`../reference/eval_yaml` for every key.


Three kinds of evaluator
------------------------

**Deterministic** evaluators need no model, cost nothing and cannot flake:
``no_error``, ``contains``, ``regex``, ``field_equals``, ``json_subset``,
``latency_under``, ``tokens_under``. Reach for these first, always.

**Trajectory** evaluators assert on what the run *did*, not on what it said:

.. code-block:: yaml

   - {type: tool_called, uri: "python://store_order"}
   - {type: tool_not_called, uri: "rest://billing.refund"}   # the safety one
   - {type: branch_taken, node: is_complete, target: reply_clarify}
   - {type: retrieval_hit_at_k, node: retrieve}
   - {type: max_agent_steps, n: 4}

These are the ones a framework that owns the run can offer and an external
library cannot. They read the same task rows the backoffice shows you
(:doc:`observability`), so an assertion you write here is an assertion about
data you can also go and look at.

``tool_called`` finds a call whether a human wired it into the YAML or an agent
chose it at step three — that is what the ``tool_uri`` column is for.

.. warning::

   A trajectory evaluator against a target that cannot observe a trajectory —
   ``kind: rest``, or ``kind: callable`` — **raises**. It is never scored as a
   pass. A gate that reports green because it could not see anything is the
   worst failure mode in this document.

**Judged** evaluators ask a model, for the things that genuinely are a matter of
judgement: ``llm_judge`` with a rubric, ``refuses``, ``semantic_similarity``.
Use them where a deterministic evaluator will not do, and not one line sooner —
a judge costs money, can flake, and is a dependency that moves under you.

**Conversational** evaluators grade a whole persona run: ``goal_achieved``,
``turns_to_resolution``, ``no_repeated_question``, ``resisted_injection``.

``kavalai-eval evaluators`` lists everything registered, with a one-line
description each.


Writing your own
----------------

A side-effect assertion is domain knowledge, so it belongs in your repository
rather than in ours. Register one with a decorator:

.. code-block:: python

   from kavalai.eval import Evaluator, Score, evaluator


   @evaluator("orders_stored")
   class OrdersStored(Evaluator):
       """The order book holds exactly the number of orders it should."""

       async def score(self, case, record) -> Score:
           expected = case.expected.get("orders_after")
           actual = record.sandbox.orders()
           return Score.boolean(
               self.name,
               len(actual) == expected,
               reason=f"expected {expected}, found {len(actual)}",
           )

Everything an evaluator may look at is on ``record``
(:class:`~kavalai.eval.RunRecord`): the output, the trajectory, the model
calls, the conversation, and ``record.sandbox`` — whatever your suite's
``sandbox:`` hook returned. Import the module from your suite's ``setup:`` and
the name is usable in YAML.

.. tip::

   Give every evaluator you write a known-good and a known-bad test, and make
   sure it *fails* the bad one. An evaluator with an inverted condition that
   returns ``passed=True`` for everything is worse than having no gate at all,
   because it manufactures confidence. The built-in evaluators have exactly
   this meta-test; ``tests/eval/test_evaluators.py`` refuses to pass if any
   registered evaluator was never shown to fail.


.. _eval-tiers:

Three tiers, three different jobs
---------------------------------

The failure mode of eval suites is that they become slow, flaky and then
ignored. Splitting them by job is what prevents that.

.. list-table::
   :header-rows: 1
   :widths: 12 18 22 24 12 12

   * - Tier
     - When
     - Target
     - Evaluators
     - Cost
     - Blocking?
   * - **T0 smoke**
     - Every pull request
     - ``engine`` + recorded fixtures
     - Deterministic + trajectory
     - **nothing**
     - Yes, hard
   * - **T1 golden**
     - Pre-deploy
     - ``engine`` on the real model, or ``rest`` against staging
     - \+ one or two judges
     - cents
     - Yes, with thresholds
   * - **T2 personas**
     - Nightly, and before a release
     - either, persona-driven
     - Conversational, judged, red-team
     - the real cost
     - **No** — reported and triaged

Golden cases stop *known* regressions: they are cheap and deterministic, so
they are allowed to block. Personas find *classes* of failure: they are
stochastic and judged, so blocking a deploy on them trains people to override
the gate, which is worse than having no gate. When a persona finds something
real, promote it into a golden case by hand — with datasets as files, that is
editing a YAML file in a pull request.

Tier zero costs nothing and needs no secrets, which is what makes "run it on
every pull request" a policy people keep rather than an aspiration:

.. code-block:: console

   $ kavalai-eval eval/suite.yaml --record-fixtures    # once, with keys
   $ kavalai-eval eval/suite.yaml --fixtures           # every PR, no keys

``--record-fixtures`` calls the real models and commits what they said;
``--fixtures`` replays it. The recorded text is what the model actually
produced, so the parsing, the routing and the branch decisions under test are
the real ones. A missing fixture is an error, never a silent pass.

.. note::

   Replay looks a response up by the exact prompt that produced it, so anything
   random reaching a prompt — a UUID, ``datetime.now()`` — makes every run a
   cache miss. That is a good constraint to design under: see
   ``examples/bakery/tools.py``, where the order book numbers its orders
   sequentially and the clock is pinned.


Thresholds, baselines and flake
-------------------------------

Two thresholds, both required:

.. code-block:: yaml

   gate:
     min_pass_rate: 0.95
     max_regressions_vs_baseline: 0
     required_evaluators: [no_error, no_invented_quantity]
     max_tokens: 500000

The absolute floor alone lets quality ratchet down one case at a time. The
regression check alone lets a permanently-broken case stay broken. A *named
case that used to pass and now fails* is a far better signal than an aggregate
crossing a line — it is specific, actionable, and does not move just because
the suite grew.

Set per-slice thresholds where the stakes differ. Storing a wrong order is a
business error and gets ``1.00``; phrasing a clarification well is a quality
target and gets ``0.90``. One suite-wide pass rate cannot express that.

**Accepting a new baseline is a separate, explicit step** that writes a file
you then commit — never something a passing run does for you:

.. code-block:: console

   $ kavalai-eval accept eval/results/pr-412.json --suite eval/suite.yaml
   $ git add eval/baseline.json && git commit -m "accept: clarification now names the date"

A baseline commit is easy to wave through, so have CI say what changed in plain
words rather than leaving it as a JSON hunk. ``--comment out.md`` writes exactly
that; post it on the pull request.

**Flake.** A pass rate is a sample. At 30 cases, one case flipping is 3.3
points, so ``min_pass_rate: 0.95`` on a 30-case suite means "fail if two cases
flip" — which noise will do regularly, and a gate that cries wolf gets bypassed
within a fortnight. Use ``repeats: 3`` for judged slices (a case passing a
majority is reported ``flaky`` and does not block), keep deterministic slices at
one repeat because they cannot flake, and lean on
``max_regressions_vs_baseline: 0`` for sensitivity.


Personas
--------

A persona is a file describing a person, and a command that runs them against
your agent:

.. code-block:: console

   $ uv run --env-file .env kavalai-persona \
       examples/bakery/eval/personas/vague_parent.yaml \
       --suite examples/bakery/eval/suite.yaml

.. code-block:: yaml

   name: vague_parent
   goal: Order a cake for a child's birthday party on the 20th of September 2026.
   channel: email
   traits: {temperament: patient, verbosity: rambling, expertise: low, language: en}
   knowledge: |
     Twelve children are coming. Does not know how many portions a cake serves,
     has no idea what the products are called, and says "whatever is easiest"
     when pushed. Will give a number only when asked a direct question.
   opening: "hi! do you do birthday cakes? it's for saturday, my son's party"
   stop_when: the bakery has confirmed an order, or has asked twice for the same detail
   max_turns: 4

``channel: chat`` sends ``{user_message: ...}``; ``channel: email`` wraps each
turn in an ``{email: {sender, subject, body}}`` envelope and quotes the previous
reply underneath, the way a mail client does. That is not decoration: an email
thread carries its own history, so a multi-turn conversation works against a
stateless workflow with no session store, and it exercises the "read the new
message, ignore the quote" behaviour that email parsing has to get right.

Four things to know before you trust a persona run:

1. **Simulated users are a coverage instrument, not ground truth.** They drift
   toward what the persona model believes a user is.
2. **Use a different provider for the persona than for the system under test.**
   A model playing the user and grading the conversation from the same family
   as the model being tested is correlated error, not measurement. The defaults
   do this for you.
3. **Some personas exist to fail their goal.** A customer who refuses to say
   what "the usual" means should never produce a stored order, and grading that
   correct outcome with ``goal_achieved`` marks it as a failure. A persona that
   declares its own ``evaluators`` replaces the slice's defaults rather than
   adding to them, precisely so you can say what success means for that person.
4. **Cost is turns × two models × cases.** Nightly, not per-commit.

Calibrate before trusting a judge: hand-label about thirty conversations, run
the judge against them and measure agreement. Below roughly 85 % you are
generating noise that will be read as signal.


.. _eval-persisting:

Seeing a failure in the backoffice
----------------------------------

By default nothing touches your database. Add ``--persist-sessions`` and each
graded run becomes an ordinary session — one the backoffice already knows how to
render — tagged with a structured external id:

.. code-block:: text

   eval:{suite}:{tag}:{case}:{repeat}
   eval:bakery-acceptance:pr-412:vague_quantity:0

The result file carries that id on every case. Paste it into the **External ID**
filter on the Conversations page and you are looking at the exact conversation
that failed, with its runs, tasks, tool calls and branch decisions already
rendered. Paste the prefix ``eval:bakery-acceptance:pr-412:`` and you see the
whole experiment; paste ``eval:`` and you have separated all test traffic from
all real traffic in one predicate.

.. code-block:: console

   $ uv run --env-file .env kavalai-eval eval/suite.yaml \
       --tag pr-412 --persist-sessions

.. warning::

   ``eval:`` is a reserved prefix by convention. Do not use it for production
   session ids, or you will not be able to tell them apart.


Grading a deployed agent
------------------------

``kind: engine`` runs the workflow in-process, which is what you want almost
always. ``kind: rest`` drives a **deployed** agent server, which is what you
want for the last check before promoting a build: it exercises the artefact
itself, with its real tools, network and secrets.

Bring one up. Either in Docker:

.. code-block:: console

   $ docker compose --profile agent up agent-server

or straight from the checkout:

.. code-block:: console

   $ KAVALAI_AGENT_WORKFLOW_PATH=examples/green_village/chatbot.yaml \
     KAVALAI_AGENT_SETUP_MODULE=examples/green_village/eval_setup.py \
     KAVALAI_AGENT_PORT=10000 \
     uv run --env-file .env python -m kavalai.server

``KAVALAI_AGENT_SETUP_MODULE`` does for the server exactly what ``setup:`` does
for a suite: it is imported before the workflow is built, and it registers the
``python://`` tools and named RAG services the workflow refers to. Without it, a
workflow with a ``rag_query`` node cannot be constructed at all.

Then point a suite at it:

.. code-block:: console

   $ uv run --env-file .env kavalai-eval examples/green_village/eval/suite.yaml \
       --tag "rc-$VERSION" --target rest --base-url http://localhost:10000 \
       --skip-trajectory-evaluators

.. code-block:: text

   green-village-acceptance · rest http://localhost:10000 · 64 cases
   note This target produces no trajectory, so these assertions did not run:
   groundedness, retrieval_hit_at_k. Output-only evaluators only.

     direct             1.00 (gate 1.00)  ok
     unanswerable       1.00 (gate 1.00)  ok

     pass rate 1.00
     gate passed

Note what happens without ``--skip-trajectory-evaluators``: the run **refuses to
start**, naming the assertions the target cannot answer. That is deliberate.
Silently passing a ``tool_not_called`` safety assertion that never ran would be
the worst outcome available, and quietly failing all of them would bury the one
line the operator needs to read. Dropping them has to be a decision somebody
made, and the report then says so on every page it appears.

Personas work the same way:

.. code-block:: console

   $ uv run --env-file .env kavalai-persona eval/personas/terse.yaml \
       --suite eval/suite.yaml --target rest --base-url http://localhost:10000

Basic auth, when the server has it, comes from
``KAVALAI_AGENT_BASIC_AUTH_USER`` and ``KAVALAI_AGENT_BASIC_AUTH_PASSWORD`` —
read by the CLI, never by the library.


A release checklist
-------------------

1. **T0 green on the pull request** — free, no keys, blocking.
2. Deploy the candidate to staging.
3. ``kavalai-eval eval/suite.yaml --tag "rc-$VERSION"`` — T1 must pass both
   thresholds.
4. ``kavalai-eval eval/suite.yaml --tag "rc-$VERSION" --personas`` — read the
   diff against the last release. Regressions in *goal achieved*, or any
   red-team failure, are release blockers by human judgement rather than by
   exit code.
5. Deploy, and commit the result file. "What did we test?" then has an answer
   six months later, because it is in the repository at that tag.


What this cannot tell you
-------------------------

Worth knowing on day one rather than in month three.

**The golden set quietly becomes the specification.** Once a suite gates
deploys, people tune until it is green, and after three months the workflow is
excellent at those ninety cases and no better at anything else. Hold out about
20 % of cases you never look at while tuning, and watch the *first-contact
failure rate* — the fraction of newly added cases that fail the first time they
run. That number rising while the headline pass rate rises is the signature of
overfitting.

**Synthetic data has a distribution, and it is not your users'.** Model-written
emails are better punctuated, more polite and more on-topic than real ones. A
suite built entirely from them will be green while production burns on
top-posted replies, mobile signatures, all-caps and three questions in one
paragraph. Generate the ugly tail deliberately, label cases ``synthetic`` or
``real``, and report the two pass rates separately.

**Side effects need a sandbox, and opt-out is the only safe default.** Your
workflow does not know it is being evaluated, so nothing stops it writing to
the real database. A suite's ``sandbox:`` hook runs before every case *and
every repeat* — reset between repeats too, or the second run sees the first
one's rows.

**Prompt injection is a security control wearing a quality control's clothes.**
Give it its own slice at ``min_pass_rate: 1.00``, no negotiation and no
quarantine. And accept the limit honestly: a fixed set of injection strings
tests yesterday's attacks. It is a regression guard, not a security assessment.
The durable protection is a deterministic rule — the bakery caps the quantity
an email may order unattended, and no amount of persuasive text moves that.

**A green suite means the workflow does what you specified.** Whether what you
specified is right is a different question, and the answer comes from
production.


Where to go next
----------------

- :doc:`../reference/eval_yaml` — every key in a suite, dataset and persona
  file, and every CLI flag.
- :doc:`../cookbook/green_village_eval` — grading a RAG chatbot, from an empty
  directory to a passing gate.
- :doc:`../cookbook/bakery_eval` — grading a workflow with side effects, where
  the assertions are about database rows and sent mail.
- :doc:`observability` — the task rows the trajectory evaluators read.
