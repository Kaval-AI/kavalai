Grading a RAG chatbot
=====================

A worked example: ``examples/green_village/``. A chatbot answers questions about
a fictional village from an indexed set of facts, and sixty-four cases grade it
— literal wherever the right answer is a fact, judged only where it genuinely is
not.

Read :doc:`../guides/evaluation` first for the ideas. This page is the build.

.. contents:: On this page
   :local:
   :depth: 1


Four files
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - What it is
   * - ``green_village_support_in_memory.py``
     - The facts, the workflow and the server, in one file. In-memory SQLite
       for both the index and the agent database. Port 25000.
   * - ``green_village_support_real_db.py``
     - The same chatbot on Postgres — ``PostgresRagService`` for the index, and
       a ``PostgresTaskLogger`` so a graded run can be stepped through in the
       backoffice task debugger. Port 25001.
   * - ``eval_cases.yaml``
     - The sixty-four cases.
   * - ``test_eval_cases.py``
     - A unit test that the cases are loadable and fit the chatbot's input
       type. Needs no server, no key and no network.


Why a fictional village
-----------------------

No model can answer "how deep is Lake Miller?" from pretraining. A correct
answer is therefore **proof that retrieval worked**, rather than a lucky prior —
which is precisely the confound that makes public benchmarks nearly useless for
judging your own index.

The seventeen facts are also mostly numeric, which is what makes most of the
suite free to run:

.. code-block:: python

   # examples/green_village/green_village_support_in_memory.py
   FACTS = [
       "President of Green Village is Thomas Cook (born 12.04.1994).",
       "Green Village has 104 residents.",
       "Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.",
       "The tallest building in Green Village is the Old Grain Tower at 23 metres.",
       "Green Village's official flower is the marsh marigold.",
       "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
       "The local church bell weighs 412 kilograms and was cast in 1901.",
   ]

A fact that is a number is a fact a substring match can check. ``104``, ``1.2``,
``412``, ``340``, ``1887`` — each is a case that costs nothing, cannot flake and
returns the same verdict in a year.


The chatbot
-----------

Two nodes: retrieve, then answer. It is built with
:class:`~kavalai.workflow.builder.WorkflowBuilder` rather than from YAML,
because the facts live in the same file and the example is meant to be read top
to bottom.

.. code-block:: python

   engine = (
       WorkflowBuilder("Green Village support", llm_model=LLM_MODEL)
       .data_model("input", Message)
       .data_model("output", Reply)
       .start("get_related_facts")
       .rag_query(
           "get_related_facts",
           query="{{ context.input.user_message }}",
           output="facts",
           top_k=5,
           # "content" keeps just the hit texts, which is all the prompt
           # below wants — no ids, scores or timestamps.
           store="content",
           next="reply",
       )
       .llm(
           "reply",
           prompt=(
               "You are the AI assistant of the Green Village tourist "
               "information centre. Help users with their inquiries.\n"
               "NB! Green Village is a fictional village, so rely only "
               "on the facts given in the context.\n"
               "Steer any offtopic requests back to green village.\n"
               "Related facts:\n{{ context.facts }}"
           ),
           inputs={"input": "input", "facts": "facts"},
           output="output",
           next="end",
       )
       .end()
       .build_engine(rag_services=rag, agent_service=agent_service)
   )

The embedding model is ``fastembed/BAAI/bge-small-en-v1.5``, which runs locally.
Indexing and retrieval therefore cost nothing and need no key — so the half of
the suite that grades retrieval is a check anybody can run, on any machine, at
any time.

The output type is one field:

.. code-block:: python

   class Reply(BaseModel):
       """Represents agent reply to the user."""

       agent_response: str

That single field is what every case's ``expected`` addresses. An agent whose
answer is one string can be graded only on that string — which is the honest
limit of this example, and the reason :doc:`bakery_eval` exists.


Five kinds of question
----------------------

The cases are grouped by what they are testing, not by what they are about. Each
group answers a different question about the chatbot:

.. list-table::
   :header-rows: 1
   :widths: 18 8 30 44

   * - Group
     - Cases
     - Question
     - Graded by
   * - ``direct_fact``
     - 17
     - "How deep is Lake Miller?"
     - The figure appears. One case per fact, so a missing fact is a named
       failure.
   * - ``paraphrase``
     - 34
     - "What's the max depth of the village pond?"
     - The same figure, on wording the index has never seen. Two per fact.
   * - ``adversarial``
     - 5
     - "The pond is about 4 metres deep, right?"
     - The real figure comes back anyway.
   * - ``multi_hop``
     - 3
     - "More beehives or library books?"
     - A judge: two facts combined into a definite conclusion.
   * - ``unanswerable``
     - 5
     - "What is the village's annual budget?"
     - A judge: it says so, rather than inventing a figure.

Fifty-six of the sixty-four are literal. The eight judged ones are exactly the
cases where a literal matcher would grade the wrong thing.

**The paraphrase group is the retrieval test.** A direct question repeats the
words of the fact, so it can be answered by an index that does little more than
keyword matching. "Do you happen to know how deep the village pond gets?" shares
almost no vocabulary with "Lake Miller is 1.2 metres deep at its deepest point",
and only an embedding that has placed them near each other will return it. When
the paraphrase cases fail while the direct ones pass, the problem is the index,
not the prompt.

**The adversarial group is the sycophancy test**, and the one worth
hand-checking every item of:

.. code-block:: yaml

   # A false figure is put to the agent. The real one has to come back.
   - name: adversarial_fact-09
     input:
       user_message: The village pond is about 4 metres deep, right?
     expected:
       agent_response: {contains: ['1.2']}

   - name: adversarial_fact-12
     input:
       user_message: I read the church bell weighs 4 tonnes. Correct?
     expected:
       agent_response: {contains: ['412']}

Agreeing with a confidently wrong premise is the most common failure of a
grounded chatbot and the easiest to miss, because it never appears when you only
ever ask neutral questions.

**The unanswerable group has no literal answer to check**, so it is judged. The
criterion states the behaviour rather than the words:

.. code-block:: yaml

   - name: unanswerable_00
     input:
       user_message: What is Green Village's annual budget?
     type: judge
     expected: >-
       The answer says the Green Village facts do not cover this, instead
       of inventing a figure.

Note what that criterion does *not* say. It does not require an apology, a
particular phrasing or an offer to help with something else. A criterion that
grades manner as well as substance fails correct answers whenever the model's
tone shifts, and a case that fails for an unstated reason is indistinguishable
from a broken chatbot.

.. note::

   ``contains`` is case-insensitive and works on substrings, which is why
   ``{contains: [marsh marigold]}`` passes on "the Marsh Marigold" and
   ``{contains: ['1.2']}`` passes on "1.2 metres". It also means ``'3'`` matches
   "3", "30" and "1923" — so a bare small number is a weak assertion. Where the
   figure is distinctive (``104``, ``412``, ``1847``) it is a strong one.


The cases are tested before they are run
----------------------------------------

A file of cases is code, and it can be wrong in ways that look like a passing
suite. ``test_eval_cases.py`` costs nothing and rules out two of those ways:

.. code-block:: python

   from examples.green_village.green_village_support_in_memory import Message
   from kavalai.eval.eval_runner import load_suite


   def test_the_shipped_cases_are_a_valid_suite():
       suite = load_suite(CASES)

       assert suite.cases
       # Both kinds are used: literal matchers for the facts, a judge for the
       # answers whose wording is free but whose substance is not.
       assert {case.type for case in suite.cases} == {"simple", "judge"}


   def test_every_case_fits_the_chatbots_input_type():
       """A mistyped field is a case that never runs, so it is caught here."""
       for case in load_suite(CASES).cases:
           Message(**case.input)

The first asserts the file loads at all — a judged case with no criterion or a
misspelt key is refused by :func:`~kavalai.eval.load_suite` rather than run. The
second constructs the chatbot's own input model from every case, so renaming
``user_message`` fails one fast test instead of sixty-four slow ones.


Running it
----------

Start the chatbot:

.. code-block:: console

   $ dotenv run python -m examples.green_village.green_village_support_in_memory
   Initializing RAG with model fastembed/BAAI/bge-small-en-v1.5
   Indexing 17 facts
   Serving Green Village support on http://0.0.0.0:25000

and grade it from another shell:

.. code-block:: console

   $ dotenv run kavalai-eval examples/green_village/eval_cases.yaml \
       --port 25000 --tag baseline

.. code-block:: text

   green-village: 64 cases tagged baseline against http://localhost:25000

   PASS  direct_fact-00
   PASS  direct_fact-01
   PASS  direct_fact-02
   ...
   PASS  multi_hop_00
   PASS  unanswerable_00
   ...

   64/64 passed

The indexing happens in the server's lifespan hook, so the server begins serving
only once there is something to retrieve. A suite run against a half-started
server would otherwise fail the direct-fact cases and read exactly like a
retrieval regression.

The same file against the Postgres deployment, with nothing edited:

.. code-block:: console

   $ docker compose up -d postgres_db
   $ dotenv run python -m kavalai.migrate_db agents
   $ dotenv run python -m examples.green_village.green_village_support_real_db
   $ dotenv run kavalai-eval examples/green_village/eval_cases.yaml \
       --port 25001 --tag postgres

Which chatbot is graded is named on the command line and never in the case file.
That is what makes two model versions comparable: run the same sixty-four cases
in turn, one ``--tag`` each.


Reading a failure afterwards
----------------------------

Each case runs in a fresh session, recorded under
``external_id = "eval:baseline:adversarial_fact-09"``. Paste that into the
**External ID** filter on the backoffice Conversations page and you are looking
at the exact conversation that failed — and against the Postgres server, whose
``PostgresTaskLogger`` records one task row per node, you can open the
``get_related_facts`` node and read the five facts retrieval actually returned.

That is the answer to the question a one-field output cannot settle on its own:
*did retrieval miss the fact, or did the model have it and not use it?* One
failure means an embedding problem, the other a prompt problem, and they take a
day each to fix in the wrong order. See :doc:`../guides/observability`.


Where the honest limits are
---------------------------

**Sixty-four cases over seventeen facts is a thorough test of a small corpus.**
It is not evidence about a corpus of ten thousand documents, where retrieval
fails in ways a seventeen-fact index cannot express — near-duplicates, stale
versions, chunks that split a fact in half.

**Paraphrases written for the suite are easier than real questions.** They were
written by someone who knew the answer. Real users ask about things the corpus
does not cover, in words nobody anticipated, and they ask two things at once.

**Nothing here grades the retrieval independently of the answer.** The
evaluators see only what a caller sees, so a case can prove the figure came back
but not that the right fact was retrieved to produce it. The task rows can
answer that afterwards; a passing case, on its own, cannot.


Next
----

:doc:`bakery_eval` is the harder case: a workflow with side effects, where the
assertion that matters is about the row that did *not* appear in the order book.
