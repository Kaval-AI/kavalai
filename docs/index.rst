.. image:: _static/iconlogo.svg
   :width: 300
   :align: left
   :alt: Kaval.AI

|

**Kaval.AI is an opinionated Python library for building production-grade
agentic workflows, chatbots and tools.**

Every input, output and intermediate step is a validated `Pydantic
<https://docs.pydantic.dev/latest/>`__ model, every run is recorded and
inspectable, and every provider sits behind one small async interface. The
result is agentic software you can debug, test and put on call.

.. code-block:: bash

   pip install "kavalai[common]"

Features
--------

* **Every provider behind one interface** — OpenAI, Google Gemini, Anthropic and
  Ollama, chosen by a ``provider/model`` string.
  → :doc:`tutorials/llm_clients`
* **Structured inputs, outputs, tool calls and responses**, all with Pydantic
  semantics and validated at every boundary.
  → :doc:`tutorials/quickstart`
* **A full workflow engine** — a typed YAML graph with conditional routing, tool
  calls, agent loops and cycles.
  → :doc:`tutorials/workflow`, :doc:`reference/yaml`
* **Agents with real tools** — Python functions, REST endpoints and MCP servers
  through one validated kernel, restrictable per node.
  → :doc:`tutorials/agents`, :doc:`reference/tools`
* **Retrieval-augmented generation** on PostgreSQL/pgvector or a portable SQLite
  file, with local or hosted embeddings.
  → :doc:`tutorials/rag`
* **Streaming everywhere** — model tokens, per-node run events, and
  Server-Sent Events over HTTP.
  → :doc:`tutorials/streamer`
* **Observability you host yourself** — every session, run, node and model call
  recorded in your own database, browsable in the backoffice UI.
  → :doc:`guides/observability`, :doc:`ui/index`
* **Serving built in** — any workflow behind a typed FastAPI endpoint, with
  optional basic auth.
  → :doc:`tutorials/serving`
* **Runs in the browser** via Pyodide and WebLLM — no server, no API key, no
  data leaving the device.
  → :doc:`tutorials/run_in_browser`
* **Deterministic tests** — swap the model for a stub and run the graph offline.
  → :doc:`guides/safety`

Weighing it against LangGraph, CrewAI, n8n or Dify? :doc:`tutorials/comparison`
lays out the trade-offs, including where Kaval.AI is behind.

Call a model
------------

One interface over OpenAI, Gemini, Anthropic and Ollama. The provider lives in
the model id, so switching is a one-string change.

.. code-block:: python

   from kavalai import make_client

   client = make_client("openai/gpt-5.4-mini")

   print(await client.prompt("What is the capital of Estonia? Answer in one sentence."))

.. code-block:: text

   The capital of Estonia is Tallinn.

Get data back, not prose
------------------------

Pass a Pydantic model and the answer comes back validated, with any provider —
no parsing, no coaxing.

.. code-block:: python

   from pydantic import BaseModel

   class City(BaseModel):
       name: str
       country: str
       population: int
       fun_fact: str

   city = await client.prompt("Describe Tallinn.", response_model=City)

   print(city.population, "—", city.fun_fact)

.. code-block:: text

   450000 — Tallinn's UNESCO-listed Old Town is one of the best-preserved medieval city centers in Europe.

Answer from your own documents
------------------------------

Index text once, retrieve what matters, and let the model answer from it. Here
the corpus is the collected knowledge of Green Village, a place no model has
heard of.

.. code-block:: python

   from kavalai.rag import SqliteRagService

   FACTS = [
       "Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).",
       "Green Village has 104 residents.",
       "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
       # …and a dozen more
   ]

   rag = SqliteRagService(":memory:", model="fastembed/BAAI/bge-small-en-v1.5")
   await rag.index_batch(
       texts=FACTS,
       metadata_list=[{}] * len(FACTS),
       source_ids=[f"fact-{i}" for i in range(len(FACTS))],
   )

   for hit in await rag.query("Who has lived in the village the longest?", top_k=2):
       print(f"{hit.similarity:.3f}  {hit.content}")

.. code-block:: text

   0.676  Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).
   0.628  Green Village has 104 residents.

Retrieval is semantic, not lexical: nothing in the question says "oldest" or
"Agnes", yet the right fact comes first. Hand those passages to a model and you
have grounded answers; see :doc:`tutorials/rag`.

Compose it into a workflow
--------------------------

A workflow is a typed graph: input in, output out, one node per step. The engine
validates every boundary, records every run, and gives the bot memory across
turns.

.. code-block:: python

   workflow = (
       WorkflowBuilder("Village greeter", llm_model="openai/gpt-5.4-mini")
       .data_model("input", Message)
       .data_model("output", Reply)
       .start("reply")
       .llm(
           "reply",
           prompt="You are the greeter of Green Village. Reply warmly in one "
                  "sentence, and suggest up to 3 quick-reply choices.",
           inputs={"message": "input"},
           output="output",
           next="end",
       )
       .end()
       .build_engine(agent_service=AgentService(db_manager.get_sqlite_sessionmaker()))
   )

   state = await workflow.run({"user_message": "Hi, I'm visiting from Tallinn!"})

.. code-block:: text

   Welcome to Green Village—so lovely to have you here from Tallinn!
   choices: ['Thanks!', 'What's nearby?', 'Tell me about the pub']
   path   : start → reply → end
   tokens : 165

The same graph is equally expressible in YAML, and the same engine serves it
over HTTP. See :doc:`tutorials/quickstart` for this example end to end.

Where to start
--------------

* Never built with LLMs before? :doc:`guides/concepts` explains the vocabulary.
* Want something running in five minutes? :doc:`tutorials/quickstart`.
* Want the full tour? Start at :doc:`tutorials/llm_clients` and work down.
* Looking something up? :doc:`reference/index` has the YAML keys, the bundled
  tools and every environment variable.
* Comparing frameworks? :doc:`tutorials/comparison` is honest about both sides.
* Want worked recipes? :doc:`cookbook/index` has them, from RAG to
  self-correcting loops.

.. toctree::
   :hidden:
   :caption: Get started

   tutorials/installation
   tutorials/quickstart

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Learn

   tutorials/index
   guides/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Operate

   ui/index
   deploy/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   reference/index
   cookbook/index
   api/index

.. toctree::
   :hidden:
   :caption: Project

   todo
   genindex
