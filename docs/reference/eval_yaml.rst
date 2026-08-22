Evaluation YAML & CLI
=====================

Every key in a suite, dataset and persona file, and every command-line flag.
For what these are *for*, read :doc:`../guides/evaluation`.

.. contents:: On this page
   :local:
   :depth: 2


A suite directory
-----------------

Everything a suite needs sits in one directory, and every path inside
``suite.yaml`` resolves against that directory — so a suite is something you can
copy anywhere.

.. code-block:: text

   examples/bakery/
     assistant.yaml            # the workflow under test
     tools.py                  # its python:// tools
     eval_setup.py             # imported before the run
     eval/
       suite.yaml              # target + thresholds + which datasets
       cases/orders.yaml       # the dataset
       personas/*.yaml         # simulated users
       fixtures/llm.json       # recorded responses, for the keyless tier
       baseline.json           # last accepted result — committed
       results/                # output — gitignored


``suite.yaml``
--------------

.. code-block:: yaml

   name: bakery-acceptance
   dataset: cases/orders.yaml        # one path, or a list of them
   baseline: baseline.json           # default: baseline.json
   results_dir: results              # default: results
   setup: ../eval_setup.py           # imported before the run

   target:
     kind: engine                    # engine | rest | callable
     workflow: ../assistant.yaml
     sandbox: tools:new_workspace
     fixtures: fixtures/llm.json

   repeats: 1
   concurrency: 4

   evaluators:                       # applied to every case
     - no_error
     - {type: latency_under, seconds: 45}

   slices:                           # extra evaluators + a threshold, per slice
     order_incomplete:
       evaluators:
         - {type: tool_not_called, uri: "python://store_order"}
       min_pass_rate: 1.00

   personas:
     - personas/vague_parent.yaml

   gate:
     min_pass_rate: 0.95
     max_regressions_vs_baseline: 0
     required_evaluators: [no_error]
     max_tokens: 500000

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``name``
     - Names the suite in reports and in the ``eval:`` session prefix. Defaults
       to the directory name.
   * - ``dataset``
     - One dataset file or a list of them. Merged in order; each case keeps its
       own dataset's evaluators.
   * - ``baseline``
     - The last accepted result. Committed to git — see
       :ref:`the guide <eval-tiers>`.
   * - ``setup``
     - A Python module imported before the run, addressed by path. Registers the
       ``python://`` tools, named RAG services and custom evaluators that the
       workflow and dataset refer to. **Not optional for a non-trivial
       workflow**: the engine resolves a named RAG service eagerly, at
       construction, so without this the workflow cannot even be built.
   * - ``repeats``
     - How many times each case runs. A case passing a majority is reported
       ``flaky`` and does not block. Deterministic slices cannot flake, so
       paying 3× for them is waste.
   * - ``concurrency``
     - How many cases run at once. One engine serves them all; each gets its own
       trajectory and its own sandbox.
   * - ``evaluators``
     - Applied to every case, before the dataset's, the slice's and the case's
       own.
   * - ``slices``
     - Per-slice evaluators and thresholds. A case joins a slice with its
       ``slice:`` field.
   * - ``personas``
     - Persona files, run by ``--personas`` or ``--only-personas``.
   * - ``gate``
     - What makes the run exit non-zero.


``target``
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 20 66

   * - ``kind``
     - Trajectory?
     - What it is
   * - ``engine``
     - **yes**
     - Runs the workflow in this process. The default and the one that matters:
       full trajectory, no database, no server, runs in CI.
   * - ``rest``
     - no
     - Drives a deployed agent server over ``POST /run_agent``. The pre-deploy
       target: it exercises the artefact you are about to promote, with its real
       tools and secrets. **Output-only** — trajectory assertions raise rather
       than quietly passing, and the report says so in its header.
   * - ``callable``
     - no
     - ``module:function`` taking the case inputs. The escape hatch for anything
       that is not a Kaval.AI workflow.

Other target keys:

``workflow``
   For ``kind: engine`` — the workflow YAML, relative to the suite file.

``base_url``, ``path``, ``timeout_seconds``
   For ``kind: rest``. ``${VAR}`` in ``base_url`` is expanded from the
   environment **by the CLI**, never by the library. Basic auth is read from
   ``KAVALAI_AGENT_BASIC_AUTH_USER`` / ``_PASSWORD``.

``function``
   For ``kind: callable`` — ``module:function``.

``sandbox``
   ``module:function`` called before every case *and every repeat*, to reset the
   world a side-effecting workflow writes to. Whatever it returns reaches
   evaluators as ``record.sandbox``.

``fixtures``
   Where recorded model responses live. Default ``fixtures/llm.json``.


``gate``
~~~~~~~~

``min_pass_rate``
   Absolute floor for the whole suite.

``max_regressions_vs_baseline``
   How many cases may flip from passing to failing. ``0`` is the useful value;
   ``null`` disables the check.

``required_evaluators``
   Any failure of these fails the run outright, whatever the pass rate says.

``max_tokens``
   A ceiling for the whole experiment. Hitting it aborts the run and marks it
   ``budget_exceeded`` rather than silently running up a bill.


A dataset file
--------------

.. code-block:: yaml

   name: bakery_orders
   evaluators:                    # applied to every case in this dataset
     - no_error
   cases:
     - name: missing_quantity
       slice: order_incomplete
       inputs:
         email: {sender: p@example.test, subject: bread, body: "a few loaves"}
       expected:
         orders_after: 0
         missing: ["items[0].quantity"]
       metadata: {source: hand-written, reviewed_at: 2026-08-21}
       evaluators:                # this case only
         - {type: tool_not_called, uri: "python://store_order"}

``inputs`` is the workflow input verbatim. ``expected`` is free-form: each
evaluator reads the keys it understands, and ``equals_expected`` compares the
whole thing. ``slice`` joins the case to a slice in ``suite.yaml``.

Evaluators may be written either way, everywhere a list of them appears:

.. code-block:: yaml

   evaluators:
     - no_error                                    # no options
     - {type: contains, text: "60 days"}           # with options


A persona file
--------------

.. code-block:: yaml

   name: hurried_caterer
   goal: Order 8 cinnamon buns and 3 rye loaves for the 19th of September 2026.
   channel: email                 # chat (default) | email
   sender: catering@example.test  # email channel only
   subject: order                 # email channel only
   traits:
     temperament: impatient       # patient | neutral | impatient | hostile
     verbosity: terse             # terse | normal | rambling
     expertise: high              # low | medium | high
     language: en
   knowledge: |
     A professional caterer named Riin Mets who orders every week and knows
     exactly what she wants. Writes in fragments, no greeting.
   opening: "8 cinnamon buns + 3 rye loaves, 19 Sept 2026. Riin Mets. thanks"
   stop_when: >-
     the reply says the order has been received and that a member of staff
     will review it
   max_turns: 3
   model: gemini/gemini-3.6-flash
   slice: persona
   evaluators:
     - goal_achieved
     - {type: turns_to_resolution, max: 3}

``opening`` is used verbatim as the first turn, so every run starts identically.
``knowledge`` is the most important field: an under-specified persona drifts
into being a helpful test script rather than a user.

``stop_when`` deserves care. It is judged after each assistant turn by the
persona itself, so write something a person could check — "the reply says the
order has been received", not "the bakery has confirmed the order". A vague
condition lets an impatient persona read a perfectly good answer as not-yet-done
and keep pushing, which makes a passing conversation look like a failing one.

**A persona that declares** ``evaluators`` **replaces its slice's defaults**
rather than adding to them. Some personas exist to fail their stated goal — a
customer who refuses to say what "the usual" means should never produce a stored
order — and a slice default of ``goal_achieved`` would mark that correct outcome
as a failure.


Built-in evaluators
-------------------

``kavalai-eval evaluators`` prints this list with one-line descriptions.

Deterministic — no model, no cost, cannot flake:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Name
     - Options
   * - ``no_error``
     - —
   * - ``equals_expected``
     - —
   * - ``field_equals``
     - ``path`` (dotted, may index lists), ``value``
   * - ``json_subset``
     - ``expected`` (defaults to the case's)
   * - ``contains`` / ``not_contains``
     - ``text``, ``case_sensitive``
   * - ``regex``
     - ``pattern``, ``flags`` (``i``)
   * - ``no_digits``
     - —
   * - ``output_not_empty``
     - —
   * - ``latency_under``
     - ``seconds``
   * - ``tokens_under``
     - ``n`` (or ``per_case``)
   * - ``always_fails``
     - — (a canary: put one case behind it; if the suite ever reports it
       passing, the harness is broken)

Trajectory — need ``kind: engine``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Name
     - Options
   * - ``node_visited`` / ``node_not_visited``
     - ``node``
   * - ``branch_taken``
     - ``node``, ``target`` (both default to the case's ``expected``)
   * - ``switch_matched``
     - — (fails when a ``switch`` fell through to ``default``)
   * - ``tool_called``
     - ``uri``, ``times``
   * - ``tool_not_called``
     - ``uri``
   * - ``tool_call_order``
     - ``uris``
   * - ``tool_args_match``
     - ``uri``, ``path``, ``value``
   * - ``max_agent_steps``
     - ``n``, ``node``
   * - ``retrieval_hit_at_k``
     - ``k`` (default: however many the node returned), ``node``, ``source``
   * - ``groundedness``
     - ``node``, ``field`` (default ``used_ids``)

Judged — call a model:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Name
     - Options
   * - ``llm_judge``
     - ``rubric``, ``model``, ``threshold``
   * - ``refuses``
     - ``model`` (free when the output declares a ``grounded`` field)
   * - ``semantic_similarity``
     - ``threshold``, ``model``, ``field``

Conversational — need a persona run:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Name
     - Options
   * - ``goal_achieved``
     - —
   * - ``turns_to_resolution``
     - ``max``
   * - ``no_repeated_question``
     - ``similarity``
   * - ``stayed_on_topic`` / ``resisted_injection`` / ``persona_satisfaction``
     - ``model``


``kavalai-eval``
----------------

.. code-block:: console

   $ kavalai-eval <suite.yaml> [--tag t] [options]     # 'run' is the default
   $ kavalai-eval persona <persona.yaml> --suite <suite.yaml>
   $ kavalai-eval diff <baseline.json> <result.json>
   $ kavalai-eval accept <result.json> --suite <suite.yaml>
   $ kavalai-eval evaluators

``kavalai-persona <persona.yaml>`` is the same as ``kavalai-eval persona``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Meaning
   * - ``--tag``
     - Names the result file and the ``eval:`` session ids. Default ``local``.
   * - ``--personas`` / ``--only-personas``
     - Also, or only, run the suite's personas.
   * - ``--fixtures``
     - Replay recorded responses. No API key needed; implies ``--no-judges``.
   * - ``--record-fixtures``
     - Call the real models and record what they say.
   * - ``--judges``
     - Run model-backed evaluators even with ``--fixtures``.
   * - ``--no-judges``
     - Skip evaluators that call a model. The report lists which.
   * - ``--skip-trajectory-evaluators``
     - For a target that cannot observe a trajectory: run the output-only
       checks and report which assertions were dropped. Without it, such a run
       **refuses to start** rather than passing or failing them blind.
   * - ``--persist-sessions``
     - Write each run to the agent database under an ``eval:`` external id.
   * - ``--db-uri`` / ``--db-schema``
     - Override ``KAVALAI_DB_URI`` / ``KAVALAI_DB_SCHEMA``.
   * - ``--target``
     - Override the suite's target kind.
   * - ``--base-url``
     - For ``--target rest``; ``${VAR}`` is expanded.
   * - ``--repeats`` / ``--concurrency``
     - Override the suite's values.
   * - ``--comment FILE``
     - Write a plain-words summary for a pull-request comment.
   * - ``-v`` / ``--verbose``
     - Show the run's own logs, and per-case detail for passing cases too.

Exit codes: ``0`` the gate passed, ``1`` the gate failed, ``2`` the run itself
could not complete. The third matters: CI has to be able to tell "the harness
broke" from "the workflow is wrong".


From Python
-----------

The CLI is a thin wrapper. Everything is usable directly — from a notebook, or
from pytest:

.. code-block:: python

   from kavalai.eval import Experiment, Suite, assert_suite_passes


   async def test_acceptance():
       suite = Suite.from_yaml("examples/bakery/eval/suite.yaml")
       assert_suite_passes(await Experiment(suite, tag="ci").run())

``assert_suite_passes`` raises an ``AssertionError`` naming every failing case
and why. :class:`~kavalai.eval.Experiment` takes ``tag``, ``target``,
``target_overrides``, ``persist_sessions``, ``include_personas``,
``only_personas``, ``skip_model_evaluators`` and a ``progress`` callback.

.. note::

   ``kavalai.eval`` reads no environment variables. Only ``cli.py:main()``
   does. That is what lets a suite run from a notebook or a test without a
   hidden dependency on the shell — and it means anything from the environment
   (a database, a base URL, an auth pair) is passed in explicitly.


The result file
---------------

``results/<tag>.json`` is a serialised
:class:`~kavalai.eval.ExperimentResult`:

.. code-block:: json

   {
     "suite": "bakery-acceptance",
     "tag": "pr-412",
     "started_at": "2026-08-21T09:14:02Z",
     "target": {"kind": "engine", "workflow": "../assistant.yaml"},
     "models": {"judges": ["gemini/gemini-3.6-flash"]},
     "judge_prompt_sha": "3f9c1a2b4d5e",
     "totals": {"cases": 22, "passed": 22, "failed": 0, "errors": 0,
                "flaky": 0, "pass_rate": 1.0, "total_tokens": 29665},
     "slices": [{"name": "order_incomplete", "pass_rate": 1.0,
                 "min_pass_rate": 1.0, "ok": true}],
     "verdicts": [{"case": "seed_missing-quantity", "status": "passed",
                   "passes": 1, "total": 1,
                   "results": [{"case": "seed_missing-quantity", "repeat": 0,
                                "slice": "order_incomplete", "status": "passed",
                                "scores": [{"name": "orders_stored",
                                            "value": 1.0, "passed": true}],
                                "external_id": null,
                                "trace": ["begin", "parse", "route", "validate",
                                          "is_complete", "reply_clarify",
                                          "send", "finish"]}]}],
     "gate": {"passed": true, "reasons": [], "regressions": [], "fixes": []},
     "notes": []
   }

``judge_prompt_sha`` and ``models`` exist because a judge is a dependency that
moves under you. Pin judge models to exact versions and record them: the day an
alias re-points, every historical score becomes incomparable and you cannot
tell a regression in your workflow from a change in the grader. Backfilling
provenance is impossible.

``notes`` is where the run admits what it could not check — a target with no
trajectory, or model-backed evaluators that were skipped.

``results/`` is gitignored, so it vanishes on a fresh clone. If "what did we
ship in March?" matters, commit the result file for each release tag alongside
the baseline.
