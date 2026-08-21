Cookbook
========

Short, self-contained recipes for the tasks practitioners most often build. Each is
complete enough to paste into a file and run once you have set a provider key,
and each was executed while writing this page.

Several are the standard use cases other agent frameworks demonstrate —
structured extraction, routing, evaluator–optimizer loops, batch classification
— written the Kaval.AI way, so you can judge the fit. See
:doc:`../tutorials/comparison` for where that fit ends.

For the reasoning behind them, follow the links into the tutorials.

A chatbot that remembers
------------------------

The trick is not the graph — it is reusing the session. Pass the same
``external_id`` (your own user, ticket or thread id) on every turn and the
engine replays that conversation's history into each ``llm`` node.

.. code-block:: python

   import asyncio

   from pydantic import BaseModel

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder


   class Message(BaseModel):
       user_message: str


   class Reply(BaseModel):
       agent_response: str


   async def main():
       await db_manager.init_sqlite()

       bot = (
           WorkflowBuilder("Village guide", llm_model="openai/gpt-5.4-mini")
           .data_model("input", Message)
           .data_model("output", Reply)
           .start("reply")
           .llm("reply", prompt="You are a warm, concise guide to Green Village.",
                inputs={"message": "input"}, output="output", next="end")
           .end()
           .build_engine(
               agent_service=AgentService(
                   db_manager.get_sqlite_sessionmaker()
               )
           )
       )

       turns = [
           "I'm Agnes, visiting on Friday.",
           "What did I say my name was?",
       ]
       for turn in turns:
           state = await bot.run({"user_message": turn}, external_id="villager-42")
           print(f"> {turn}\n  {state.output_data['agent_response']}")


   asyncio.run(main())

Drop the ``external_id`` and each call starts a fresh session — which is what you
want for one-off, stateless invocations. See
:doc:`../tutorials/observability_storage`.

Answering from your own documents
---------------------------------

Retrieval in a ``function`` node, generation in an ``llm`` node. The retrieval
tool is an ordinary Python function, so it can query anything — here a portable
SQLite index.

.. code-block:: python

   import asyncio

   from pydantic import BaseModel

   from kavalai import pythontool
   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.rag import SqliteRagService
   from kavalai.workflow import WorkflowBuilder

   FACTS = [
       "Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).",
       "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
       "Green Village's only pub, The Rusty Anchor, has been operating since 1923.",
       "The village library owns 1,847 books and is open on Tuesdays and Fridays.",
   ]

   rag = SqliteRagService(":memory:", model="fastembed/BAAI/bge-small-en-v1.5")


   class Passages(BaseModel):
       context: str


   @pythontool
   async def search_village(question: str) -> Passages:
       """Find the village facts most relevant to a question."""
       hits = await rag.query(question, top_k=3)
       return Passages(context="\n".join(f"- {hit.content}" for hit in hits))


   class Question(BaseModel):
       user_message: str


   class Answer(BaseModel):
       agent_response: str


   async def main():
       await rag.index_batch(
           texts=FACTS,
           metadata_list=[{}] * len(FACTS),
           source_ids=[f"fact-{i}" for i in range(len(FACTS))],
       )
       await db_manager.init_sqlite()

       engine = (
           WorkflowBuilder("Village FAQ", llm_model="openai/gpt-5.4-mini")
           .data_model("input", Question)
           .data_model("passages", Passages)
           .data_model("output", Answer)
           .start("retrieve")
           .function(
               "retrieve",
               tool="python://search_village",
               inputs={"question": "input.user_message"},
               output="passages",
               next="answer",
           )
           .llm(
               "answer",
               prompt=(
                   "Answer the villager's question using only these facts:\n"
                   "{{ context.passages.context }}\n"
                   "If they are not enough, say so."
               ),
               inputs={"question": "input"},
               output="output",
               next="end",
           )
           .end()
           .build_engine(
               agent_service=AgentService(
                   db_manager.get_sqlite_sessionmaker()
               )
           )
       )
       engine.kernel.register_python_tool("search_village", search_village)

       state = await engine.run({"user_message": "When is the library open?"})
       print(state.output_data["agent_response"])


   asyncio.run(main())

.. code-block:: text

   The library is open on Tuesdays and Fridays.

Note ``{{ context.passages.context }}`` in the prompt: the retrieved passages are
interpolated straight from the run context. See :doc:`../tutorials/rag`.

Routing a request to the right handler
--------------------------------------

Classify with a cheap call, then branch. Each branch can use a different model,
prompt, or even an agent with tools.

.. code-block:: yaml

   name: Council desk
   llm_model: openai/gpt-5.4-mini
   data_types:
     input:
       type: object
       properties:
         user_message: {type: string}
     classification:
       type: object
       properties:
         intent: {type: string}
     output:
       type: object
       properties:
         agent_response: {type: string}
   nodes:
     - {name: start, type: start, next: classify}
     - name: classify
       type: llm
       prompt: |
         Classify the villager's message as exactly one of: repair, permit, other.
         Respond with that single lowercase word in the `intent` field.
       inputs: {input: {type: context, value: input}}
       output: classification
       next: route
     - name: route
       type: switch
       expr: classification.intent
       cases:
         repair: repair_reply
         permit: permit_reply
       default: general_reply
     - name: repair_reply
       type: llm
       prompt: "Acknowledge the repair request and name the next step."
       inputs: {input: {type: context, value: input}}
       output: output
       next: end
     - name: permit_reply
       type: llm
       prompt: "Explain briefly how to apply for this permit."
       inputs: {input: {type: context, value: input}}
       output: output
       next: end
     - name: general_reply
       type: llm
       prompt: "Answer the villager's question helpfully and briefly."
       inputs: {input: {type: context, value: input}}
       output: output
       next: end
     - {name: end, type: end, output: output}

Give the classifier a small, closed set of labels and tell it to answer with one
word — that is what makes ``switch`` reliable. See :doc:`../tutorials/workflow`.

Giving an agent only the tools it should have
----------------------------------------------

One kernel often hosts more tools than any single agent should touch.
``allowed_tools`` is enforced, not advisory: excluded tools are neither described
to the model nor callable.

.. code-block:: python

   from kavalai import Agent, FunctionKernel, make_client
   from kavalai.tools.webtools.crawl4ai import crawl_url, web_search

   kernel = FunctionKernel()
   kernel.register_python_tool("web.search", web_search)
   kernel.register_python_tool("web.crawl", crawl_url)
   # Registered on the kernel, but withheld from the agent below.
   kernel.register_python_tool("db.delete_customer", delete_customer)

   researcher = Agent(
       llm_client=make_client("openai/gpt-5.4-mini"),
       kernel=kernel,
       allowed_tools=["python://web.search", "python://web.crawl"],
   )

   print(await researcher.prompt(
       "Find out what Kaval.AI does and summarise it in three sentences.",
       max_steps=6,
   ))

In YAML, set ``allowed_tools`` on the ``agent`` node. See
:doc:`../reference/yaml`.

Testing a workflow without calling a model
-------------------------------------------

Inject a ``client_factory`` and the engine builds *your* client. Graph logic
becomes deterministic and free.

.. code-block:: python

   import pytest

   from kavalai import BaseLlmClient, WorkflowEngine


   class StubClient(BaseLlmClient):
       """Returns canned structured output — no network, no API key."""

       def __init__(self, *args, **kwargs):
           super().__init__()

       async def _run_chat_completions(self, chat_history, response_model, streamer):
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


   async def test_repair_requests_route_to_the_repair_handler():
       engine = WorkflowEngine.from_yaml_path(
           "council_desk.yaml", client_factory=lambda *a, **k: StubClient()
       )
       state = await engine.run({"user_message": "anything"})

       assert state.trace == ["start", "classify", "route", "repair_reply", "end"]

The engine drives clients through ``_run_chat_completions``, so that is the
method a stub implements — not ``chat_completions``. See
:doc:`../guides/safety`.

Streaming a run to a UI
-----------------------

Turn on ``stream_output`` for the node whose text the user should watch, then
forward the events. ``stream_delta`` sends only new text per chunk, which is
what is wanted for long answers.

.. code-block:: python

   engine = (
       WorkflowBuilder("Village guide", llm_model="openai/gpt-5.4-mini")
       .data_model("input", Message)
       .data_model("output", Reply)
       .start("reply")
       .llm("reply", prompt="You are a warm guide to Green Village.",
            inputs={"message": "input"}, output="output", next="end",
            stream_output=True, stream_delta=True)
       .end()
       .build_engine(agent_service=service)
   )

   question = {"user_message": "Tell me about the tower."}
   async for event in engine.run_stream(question):
       if event.type == "partial" and event.name == "reply":
           print(event.value, end="", flush=True)
       elif event.type == "workflow_failed":
           print("\nrun failed:", event.value)

Over HTTP, ``POST /stream_agent`` serves these same events as Server-Sent
Events — see :doc:`../tutorials/serving`.

Extracting structured records from unstructured text
----------------------------------------------------

The most reliably useful operation a language model performs: turning prose
into a typed record.
Nested models and lists work, so one call can return a whole document.

.. code-block:: python

   import asyncio

   from pydantic import BaseModel, Field

   from kavalai import make_client

   NOTES = """
   Village council, 14 March. Present: Thomas Cook, Greta Lindqvist, Agnes Whitlow.
   Greta will order 200kg of flour before the Turnip Festival. Thomas agreed to
   get three quotes for repairing the church bell by 1 April. We rejected the
   proposal to widen Cobbler's Path. Agnes will ask the library to open on
   Saturdays during the festival week.
   """


   class ActionItem(BaseModel):
       owner: str
       task: str
       due: str | None = Field(
           default=None, description="Deadline if one was stated."
       )


   class Minutes(BaseModel):
       date: str
       attendees: list[str]
       actions: list[ActionItem]
       decisions: list[str]


   async def main():
       client = make_client("openai/gpt-5.4-mini")
       minutes = await client.prompt(
           f"Extract the minutes from these notes:\n{NOTES}", response_model=Minutes
       )

       print("date     :", minutes.date)
       print("attendees:", ", ".join(minutes.attendees))
       for action in minutes.actions:
           due = f" (due {action.due})" if action.due else ""
           print(f"  - {action.owner}: {action.task}{due}")


   asyncio.run(main())

.. code-block:: text

   date     : 14 March
   attendees: Thomas Cook, Greta Lindqvist, Agnes Whitlow
     - Greta Lindqvist: Order 200kg of flour before the Turnip Festival
     - Thomas Cook: Get three quotes for repairing the church bell (due 1 April)
     - Agnes Whitlow: Ask the library to open on Saturdays during the festival week

Note ``due`` is ``str | None`` with a description: optional fields let the model
say "not stated" without inventing a date, and the description is part of the
schema it sees. See :doc:`../tutorials/llm_clients`.

A self-correcting draft (evaluator–optimizer)
----------------------------------------------

A graph may contain cycles, which is what makes the classic
*write → critique → revise → critique* loop expressible directly. The ``if``
node decides whether to go round again, and a counter keeps the loop finite.

.. code-block:: yaml

   name: Notice writer
   description: Drafts a village notice, critiques it, and revises until it passes.
   llm_model: openai/gpt-5.4-mini
   data_types:
     input:
       type: object
       properties:
         topic: {type: string}
     draft:
       type: object
       properties:
         text: {type: string}
     review:
       type: object
       properties:
         approved: {type: boolean}
         feedback: {type: string}
     attempts:
       type: object
       properties:
         count: {type: integer}
     output:
       type: object
       properties:
         agent_response: {type: string}
   nodes:
     - {name: start, type: start, next: write}
     - name: write
       type: llm
       prompt: |
         Write a short notice for the Green Village noticeboard about:
         {{ context.input.topic }}
         Keep it under 40 words.
       inputs: {input: {type: context, value: input}}
       output: draft
       next: critique
     - name: critique
       type: llm
       prompt: |
         You are a strict village clerk. Review this notice:
         {{ context.draft.text }}
         Approve only if it states what, when and where. Set approved and give feedback.
       inputs: {draft: {type: context, value: draft}}
       output: review
       next: count
     - name: count
       type: function
       tool: python://bump
       inputs: {current: {type: context, value: attempts.count}}
       output: attempts
       next: decide
     - name: decide
       type: if
       condition: "review.approved == True or attempts.count >= 3"
       then: finish
       else: revise
     - name: revise
       type: llm
       prompt: |
         Rewrite the notice addressing this feedback.
         Notice: {{ context.draft.text }}
         Feedback: {{ context.review.feedback }}
       inputs: {review: {type: context, value: review}}
       output: draft
       next: critique
     - name: finish
       type: llm
       prompt: >
         Return the final notice text unchanged in agent_response:
         {{ context.draft.text }}
       inputs: {draft: {type: context, value: draft}}
       output: output
       next: end
     - {name: end, type: end, output: output}

The counter is an ordinary tool. Note the ``Optional`` — on the first pass
``attempts.count`` does not exist yet and resolves to ``None``:

.. code-block:: python

   from typing import Optional

   from pydantic import BaseModel

   from kavalai import pythontool


   class Attempts(BaseModel):
       count: int


   @pythontool
   def bump(current: Optional[int] = None) -> Attempts:
       """Increment the revision counter."""
       return Attempts(count=(current or 0) + 1)

The trace records the loop, so you can see exactly how many rounds it took:

.. code-block:: text

   trace   : start → write → critique → count → decide → revise
             → critique → count → decide → finish → end
   attempts: {'count': 2}
   approved: True

Always bound the loop. ``max_node_visits`` (1000 by default) will stop a runaway
graph, but a stuck critique burns real tokens until it does — the explicit
counter is what makes the cost predictable.

Classifying a backlog concurrently
-----------------------------------

The engine executes one node at a time, but nothing stops you running many
workflows at once. This is how you fan out today.

.. code-block:: python

   import asyncio

   from pydantic import BaseModel

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder


   class Note(BaseModel):
       user_message: str


   class Tagged(BaseModel):
       agent_response: str
       topic: str
       urgent: bool


   NOTES = [
       "The church bell has been stuck since Tuesday.",
       "May I put a beehive in my front garden?",
       "The pub sign fell down and nearly hit someone.",
       "When does the library open?",
   ]


   def build(service):
       return (
           WorkflowBuilder("Noticeboard triage", llm_model="openai/gpt-5.4-mini")
           .data_model("input", Note)
           .data_model("output", Tagged)
           .start("tag")
           .llm(
               "tag",
               prompt=(
                   "Tag this note from the Green Village noticeboard. "
                   "topic is one of: repair, permit, other. "
                   "urgent is true only if someone could get hurt. "
                   "agent_response is a one-line acknowledgement."
               ),
               inputs={"note": "input"},
               output="output",
               next="end",
               use_history=False,
           )
           .end()
           .build_engine(agent_service=service)
       )


   async def main():
       await db_manager.init_sqlite()
       service = AgentService(db_manager.get_sqlite_sessionmaker())
       engine = build(service)           # one engine, shared by every run

       async def classify(note: str):
           return await engine.run({"user_message": note})

       states = await asyncio.gather(*map(classify, NOTES))
       for note, state in zip(NOTES, states):
           out = state.output_data
           print(f"{out['topic']:<7} urgent={out['urgent']!s:<5} {note}")


   asyncio.run(main())

.. code-block:: text

   repair  urgent=False The church bell has been stuck since Tuesday.
   permit  urgent=False May I put a beehive in my front garden?
   repair  urgent=True  The pub sign fell down and nearly hit someone.
   other   urgent=False When does the library open?

**``use_history=False``** is what makes it work. Classification is not a
conversation: left on, each run would replay the session's earlier turns into
the prompt — more tokens, and one note's wording nudging the next one's label.

One engine serves all four runs. Each run keeps its own token accounting, so
the figures stay per-run no matter how much they overlap:

.. code-block:: python

   for state in states:
       print(state.token_usage)

.. code-block:: text

   {'model_calls': 1, 'prompt_tokens': 119,
    'completion_tokens': 29, 'total_tokens': 148}
   {'model_calls': 1, 'prompt_tokens': 122,
    'completion_tokens': 37, 'total_tokens': 159}
   {'model_calls': 1, 'prompt_tokens': 120,
    'completion_tokens': 34, 'total_tokens': 154}
   {'model_calls': 1, 'prompt_tokens': 116,
    'completion_tokens': 30, 'total_tokens': 146}

Sharing the engine is in fact the better shape: it parses the workflow once and
keeps one set of tool-server connections, which matters when the workflow
declares MCP servers, since those are subprocesses.

Pausing for a human
-------------------

Kaval.AI has no interrupt-and-resume primitive: a run goes from ``start`` to
``end``. The pattern that works is to make the pause a *boundary between runs*,
with the session carrying the state.

.. code-block:: python

   # Run 1 — draft a reply and stop. Nothing is sent.
   draft = await engine.run(
       {"user_message": "The church bell has been stuck since Tuesday."},
       external_id="ticket-91",
   )
   show_to_reviewer(draft.output_data["agent_response"])

   # The reviewer decides here. That may take minutes or days; the
   # process is free to exit in the meantime.

   # Run 2 — same session, so the draft and its context are already in history.
   final = await engine.run(
       {"user_message": "Approved, but mention the bell ringer's name."},
       external_id="ticket-91",
   )

Because both runs share a session, the second one sees the first through chat
history, and a node can read a specific earlier value with a ``history:`` input
(see :doc:`../reference/yaml`). What you do not get is a suspended run resuming
mid-graph — the second run starts at ``start`` again. For approval steps in the
*middle* of a long graph, LangGraph or n8n do this natively; see
:doc:`../tutorials/comparison`.

Watching what a run used
------------------------

Token usage is aggregated per run, and every individual call is recorded.

.. code-block:: python

   state = await engine.run({"user_message": "Is the pub open on Sundays?"})

   print(state.token_usage)
   # {'model_calls': 2, 'prompt_tokens': 181,
   #  'completion_tokens': 121, 'total_tokens': 302}

   for call in await service.get_model_call_stats(call_type="llm", limit=5):
       print(call.model, call.total_tokens, f"{call.duration_seconds:.2f}s")

Multiply by your provider's published prices to turn tokens into money — and
subtract ``cached_prompt_tokens`` from ``prompt_tokens`` first, because cached
input is billed at a fraction of the rest. :doc:`../guides/observability`
explains why the runtime records usage rather than cost.
