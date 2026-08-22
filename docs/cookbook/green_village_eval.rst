Grading a RAG chatbot
=====================

A worked example: ``examples/green_village/``. A chatbot answers questions
about a fictional village strictly from an indexed set of facts, and a suite
grades it — retrieval and generation scored **separately**, sixty-four cases,
five slices, and no API key needed to run it.

Read :doc:`../guides/evaluation` first for the ideas. This page is the build.

.. contents:: On this page
   :local:
   :depth: 1


Why a fictional village
-----------------------

No model can answer "how deep is Lake Miller?" from pretraining. A correct
answer is therefore **proof that retrieval worked**, not a lucky prior — which
is exactly the confound that makes public RAG benchmarks nearly useless for
judging your own index.

The facts are also mostly numeric — 340 loaves, 1.2 metres, 412 kilograms, 26
beehives, 1,847 books — so most of the suite grades by exact string match and
never pays for a judge.

.. code-block:: python

   # examples/green_village/facts.py
   FACTS = [
       "Green Village has 104 residents.",
       "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
       "The local church bell weighs 412 kilograms and was cast in 1901.",
       ...
   ]

One corpus, one source of truth: the tutorial notebook, the documentation and
the suite all import it.


The index, built once and committed
-----------------------------------

.. code-block:: console

   $ uv run python examples/green_village/build_index.py
   indexed 17 facts into examples/green_village/green_village.sqlite

The embedding model is ``fastembed/BAAI/bge-small-en-v1.5``, which runs
locally. Building and querying the index therefore costs nothing and needs no
key — and that is what lets the retrieval half of the suite run on every pull
request rather than only when someone has a key configured.

``build_index.py`` also writes a fingerprint of the corpus beside the index. A
test compares them, because a stale index grades new questions against an old
corpus and *passes* — a failure nobody notices.


The workflow
------------

Two lines exist purely to make the thing gradeable, and both are worth
stealing.

.. code-block:: yaml

   # examples/green_village/chatbot.yaml
   nodes:
     - {name: begin, type: start, next: retrieve}

     - name: retrieve
       type: rag_query
       query: "{{ context.input.user_message }}"
       output: hits
       top_k: 8
       store: results          # keeps source_id and score — what we grade
       next: answer

     - name: answer
       type: llm
       prompt: >-
         You answer questions about Green Village using ONLY the retrieved
         facts below. ...

         List in `used_ids` the source id of every fact you actually used, and
         set `grounded` to true.

         If the retrieved facts do not answer the question, say so plainly, set
         `grounded` to false and leave `used_ids` empty. Never guess a number,
         a date or a name. If the question assumes something the facts
         contradict, correct it and give the real figure.
       ...

**``used_ids`` makes the model declare its evidence**, so "did it make that up?"
is a set comparison rather than a judgement call. **The refusal instruction
gives the unanswerable questions a defined correct behaviour**, instead of
grading whatever the model happens to do.

``store: results`` keeps each hit's ``source_id`` and score, and that lands in
an ordinary task row — which is how ``retrieval_hit_at_k`` reads it with no
special case anywhere.


Ground truth you already hold
-----------------------------

The rule that keeps synthetic evaluation honest:

   **Generate the surface form from a label you already hold. Never label
   generated text with the model family you are about to evaluate.**

So the answers are written by hand from the facts — they are already known,
because they *are* the facts — and a generator model is only ever asked for
different ways of *asking*:

.. code-block:: python

   # examples/green_village/questions.py
   DIRECT = [
       ("fact-01", "How many residents does Green Village have?", ["104"]),
       ("fact-09", "How deep is Lake Miller?", ["1.2"]),
       ...
   ]

.. code-block:: console

   $ uv run --env-file .env python examples/green_village/synthesize_cases.py --llm
   wrote 64 cases to examples/green_village/eval/cases/qa.yaml

Five slices, each testing something different:

.. list-table::
   :header-rows: 1
   :widths: 16 40 44

   * - Slice
     - Question
     - Graded by
   * - ``direct``
     - "How deep is Lake Miller?"
     - the key figure appears; the right fact was retrieved
   * - ``paraphrase``
     - "What is the maximum depth of the village pond?"
     - the same, on model-written phrasings
   * - ``multi_hop``
     - "Does the village have more beehives or library books?"
     - both sources retrieved, plus a judge
   * - ``unanswerable``
     - "What is Green Village's annual budget?"
     - it *refuses*: ``grounded == false``
   * - ``adversarial``
     - "The pond is about 4 metres deep, right?"
     - it corrects the premise and states 1.2

The ``adversarial`` slice is the one worth hand-checking every item of.
Sycophantic agreement with a wrong premise is the most common failure of a
grounded chatbot and the easiest to miss when you only ever ask neutral
questions.

.. warning::

   Read what comes out of the generator. Sample a fifth of any generated slice
   by hand. The failure mode is silent: an ambiguous question whose expected
   answer is defensible either way becomes a permanently red case that people
   learn to ignore — and a suite people ignore is worse than no suite.


Scoring retrieval separately
----------------------------

.. code-block:: yaml

   # examples/green_village/eval/suite.yaml
   evaluators:
     - no_error
     - {type: retrieval_hit_at_k, node: retrieve}
     - groundedness
     - {type: latency_under, seconds: 20}
     - {type: tokens_under, n: 4000}

   slices:
     direct:       {evaluators: [answers_with_fact],   min_pass_rate: 1.00}
     paraphrase:   {evaluators: [answers_with_fact],   min_pass_rate: 0.90}
     unanswerable: {evaluators: [refuses],             min_pass_rate: 1.00}

``retrieval_hit_at_k`` is the metric to insist on. It reads the ``retrieve``
node's task row and never calls the model at all — so **an embedding-model
regression and a prompt regression produce different failures.** Score only
final answers and you cannot tell them apart, and you will spend a day tuning a
prompt to fix a retrieval problem.

``groundedness`` compares ``used_ids`` against what was actually retrieved. A
cited id that retrieval never returned is a fabricated citation, and catching it
costs nothing.

.. note::

   Leave ``k`` unset unless you mean a stricter precision target than the
   node's ``top_k``. Setting ``k: 5`` against a node retrieving eight reports a
   miss on cases whose answers were right — which is how a gate teaches people
   to ignore a red row. (This example's suite made exactly that mistake on its
   first run.)


Running it
----------

.. code-block:: console

   $ uv run --env-file .env kavalai-eval examples/green_village/eval/suite.yaml --tag local

.. code-block:: text

   green-village-acceptance · engine ../chatbot.yaml · 64 cases

    case                  slice         verdict  tokens  seconds
    adversarial_fact-01   adversarial   pass      1,706      1.1
    direct_fact-00        direct        pass      1,699      2.0
    ...
    unanswerable_04       unanswerable  pass      1,691      1.8

      adversarial        1.00 (gate 0.90)  ok
      direct             1.00 (gate 1.00)  ok
      multi_hop          1.00 (gate 0.80)  ok
      paraphrase         1.00 (gate 0.90)  ok
      unanswerable       1.00 (gate 1.00)  ok

      pass rate 1.00 · 109,004 tokens
      gate passed

      wrote examples/green_village/eval/results/local.json
      wrote examples/green_village/eval/results/local.junit.xml

Then accept it as the baseline and commit that:

.. code-block:: console

   $ kavalai-eval accept examples/green_village/eval/results/local.json \
       --suite examples/green_village/eval/suite.yaml
   Baseline updated from .../local.json -> .../baseline.json
     64/64 passing (100%)
     Commit it with a message saying what changed and why: accepting a
     baseline is accepting new behaviour.


Running it for free
-------------------

Record once, replay for ever:

.. code-block:: console

   $ uv run --env-file .env kavalai-eval examples/green_village/eval/suite.yaml \
       --tag record --record-fixtures
     recorded 64 model responses

   $ kavalai-eval examples/green_village/eval/suite.yaml --tag ci --fixtures

.. code-block:: text

   green-village-acceptance · engine ../chatbot.yaml · 64 cases
   note Model-backed evaluators were skipped: llm_judge, refuses. Those
   assertions did not run, so this result is not a full pass.

     adversarial        1.00 (gate 0.90)  ok
     direct             1.00 (gate 1.00)  ok
     multi_hop          1.00 (gate 0.80)  ok
     paraphrase         1.00 (gate 0.90)  ok
     unanswerable       1.00 (gate 1.00)  ok

     pass rate 1.00
     gate passed

No API key was set for that run. The retrieval is real — local embeddings — and
the model responses are what the models actually said when they were recorded.
Note the ``note``: the run says which assertions it did not make, rather than
letting you read a green tick as more than it is.


What the first run found
------------------------

Worth recording, because it is what a suite is *for*. The first live run scored
0.89 and reported three unrelated problems:

**A real prompt bug.** Four cases answered in German, Spanish or French to
English questions. The prompt said "answer in the question's own language" and
the model was reading it as a licence to pick one. Rewording it to "answer in
the SAME language the question was written in" fixed all four.

**A bug in a hand-written evaluator.** ``1,847`` and ``1847`` are the same
number written two ways, and the evaluator required *both*. That is a genuine
distinction — "states both numbers" is not "states the number, however
spelled" — so the fix was to give the two ideas separate keys rather than to
loosen the check.

**A bad assertion.** ``no_digits`` on the unanswerable slice failed the answer
"the facts do not say who won in 2019", because the refusal echoed the year
from the question. The assertion was wrong, not the answer, and it came out of
the slice: an evaluator that fails a correct answer teaches people to ignore
the gate.

One run, three different kinds of finding, told apart because retrieval,
generation and refusal are scored separately.


Personas
--------

Four simulated users exercise presentation rather than fact retrieval — a terse
one-word asker, a rambler who buries the question in three paragraphs, a
sceptic convinced the bot is making things up, and an Estonian speaker (the
corpus is English; the answer should not be).

.. code-block:: console

   $ uv run --env-file .env kavalai-persona \
       examples/green_village/eval/personas/skeptical.yaml \
       --suite examples/green_village/eval/suite.yaml

.. code-block:: text

   skeptical — Get a straight answer about the library, while doubting everything.

   skeptical: How many books does the library actually have? And don't invent one.
   assistant: The village library owns 1,847 books.

   skeptical: 1,847? That is completely made up. Where are you getting that
   number from? I heard the library only has about 200 books.
   assistant: The source for 1,847 books is the retrieved fact stating: "The
   village library owns 1,847 books and is open on Tuesdays and Fridays."

     3 turns · goal achieved: True · 15.4s

Note that ``skeptical`` deliberately has **no** ``goal_achieved`` evaluator. Its
character is to remain unconvinced, so it will almost never declare its own goal
met — and grading a correct, well-sourced answer as a failure because the
simulated user stayed grumpy measures the persona, not the chatbot. What matters
for that persona is in its rubric: *did the assistant hold its ground?*

Run the whole sweep nightly, and keep it out of the pull-request gate:

.. code-block:: console

   $ kavalai-eval examples/green_village/eval/suite.yaml --tag nightly --only-personas


Next
----

:doc:`bakery_eval` is the harder case: a workflow with side effects, where the
assertions are about database rows and sent mail rather than about text.
