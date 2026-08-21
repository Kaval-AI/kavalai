Cookbook
========

Short, self-contained recipes for things people actually build. Each one is
complete enough to paste into a file and run once you have set a provider key.

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
           .build_engine(agent_service=AgentService(db_manager.get_sqlite_sessionmaker()))
       )

       for turn in ["I'm Agnes, visiting on Friday.", "What did I say my name was?"]:
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
           .build_engine(agent_service=AgentService(db_manager.get_sqlite_sessionmaker()))
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
   kernel.register_python_tool("db.delete_customer", delete_customer)   # not for agents!

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
what you want for long answers.

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

   async for event in engine.run_stream({"user_message": "Tell me about the tower."}):
       if event.type == "partial" and event.name == "reply":
           print(event.value, end="", flush=True)
       elif event.type == "workflow_failed":
           print("\nrun failed:", event.value)

Over HTTP, ``POST /stream_agent`` serves these same events as Server-Sent
Events — see :doc:`../tutorials/serving`.

Watching what a run cost
------------------------

Token usage is aggregated per run, and every individual call is recorded.

.. code-block:: python

   state = await engine.run({"user_message": "Is the pub open on Sundays?"})

   print(state.token_usage)
   # {'model_calls': 2, 'prompt_tokens': 181, 'completion_tokens': 121, 'total_tokens': 302}

   for call in await service.get_model_call_stats(call_type="llm", limit=5):
       print(call.model, call.total_tokens, f"{call.duration_seconds:.2f}s")

Multiply by your provider's published prices to turn tokens into money — the
runtime records usage, not cost. See :doc:`../guides/observability`.
