.. image:: _static/iconlogo.svg
   :width: 300
   :align: left
   :alt: Kaval.AI

|

**Kaval.AI is an opinionated Python library for building production-grade
agentic workflows, chatbots and tools.**

Every input, output and intermediate step is a validated `Pydantic
<https://docs.pydantic.dev/latest/>`__ model, every run is recorded and
open to inspection, and every provider is reached through a single small
asynchronous interface. The result is agentic software that can be debugged,
tested and operated under production conditions.

.. code-block:: bash

   pip install "kavalai[common]"

Features
--------

* **Every provider behind one interface** — OpenAI, Google Gemini, Anthropic
  and Ollama, selected by a ``provider/model`` string
  (see :doc:`tutorials/llm_clients`).
* **Structured inputs, outputs, tool calls and responses**, expressed with
  Pydantic semantics and validated at every boundary
  (see :doc:`tutorials/quickstart`).
* **A complete workflow engine** — a typed YAML graph with conditional
  routing, parallel fan-out, tool calls, agent loops and cycles
  (see :doc:`tutorials/workflow`, :doc:`reference/yaml`).
* **Agents with real tools** — Python functions, REST endpoints and MCP
  servers through a single validated kernel, restrictable per node
  (see :doc:`tutorials/agents`, :doc:`reference/tools`).
* **Retrieval-augmented generation** over PostgreSQL/pgvector or a portable
  SQLite file, with local or hosted embeddings (see :doc:`tutorials/rag`).
* **Streaming throughout** — model tokens, per-node run events and
  Server-Sent Events over HTTP (see :doc:`tutorials/streamer`).
* **Self-hosted observability** — every session, run, node and model call
  recorded in a database you own and browsable in the backoffice interface
  (see :doc:`guides/data_model`, :doc:`guides/observability`, :doc:`ui/index`).
* **Serving included** — any workflow placed behind a typed FastAPI endpoint,
  with optional basic authentication (see :doc:`tutorials/serving`).
* **Client-side execution** through Pyodide and WebLLM, requiring no server,
  no API key and no transfer of data off the device
  (see :doc:`tutorials/run_in_browser`).
* **Deterministic tests** — substitute a stub for the model and execute the
  graph offline (see :doc:`guides/safety`).
* **Evaluation and acceptance gates** — datasets, simulated users and
  thresholds as files in your repository, asserting on what a run *did* (which
  tool, which branch, which rows in the database) rather than only on what it
  said (see :doc:`guides/evaluation`).

For a comparison against LangGraph, CrewAI, n8n and Dify, including the
respects in which Kaval.AI remains behind, see :doc:`tutorials/comparison`.
The design that produced these properties is set out in
:doc:`tutorials/architecture`.

Call a model
------------

A single interface covers OpenAI, Gemini, Anthropic and Ollama. The provider
is part of the model identifier, so substituting one for another is a
change of one string.

.. code-block:: python

   from kavalai import make_client

   client = make_client("openai/gpt-5.4-mini")

   answer = await client.prompt(
       "What is the capital of Estonia? Answer in one sentence."
   )
   print(answer)

.. code-block:: text

   The capital of Estonia is Tallinn.

Receive data rather than prose
------------------------------

Supplying a Pydantic model causes the answer to be returned as a validated
object, with any provider, so that no parsing step is required.

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

   450000 — Tallinn's UNESCO-listed Old Town is one of the
   best-preserved medieval city centres in Europe.

Answer from your own documents
------------------------------

Text is indexed once, the relevant passages are retrieved, and the model
answers from them. The corpus below is the collected knowledge of Green
Village, a settlement that appears in no model's training data.

.. code-block:: python

   from kavalai.rag import SqliteRagService

   FACTS = [
       "Green Village's oldest resident is Agnes Whitlow "
       "(born 02.06.1929).",
       "Green Village has 104 residents.",
       "The village pond, Lake Miller, is 1.2 metres deep at its "
       "deepest point.",
   ]

   rag = SqliteRagService(
       ":memory:", model="fastembed/BAAI/bge-small-en-v1.5"
   )
   await rag.index_batch(
       texts=FACTS,
       metadata_list=[{}] * len(FACTS),
       source_ids=[f"fact-{i}" for i in range(len(FACTS))],
   )

   question = "Who has lived in the village the longest?"
   for hit in await rag.query(question, top_k=2):
       print(f"{hit.similarity:.3f}  {hit.content}")

.. code-block:: text

   0.676  Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).
   0.628  Green Village has 104 residents.

Retrieval is semantic rather than lexical: the question contains neither
"oldest" nor "Agnes", yet the correct fact is ranked first. Passing those
passages to a model yields grounded answers; see :doc:`tutorials/rag`.

Compose the parts into a workflow
---------------------------------

A workflow is a typed graph: an input enters, an output leaves, and each step
is one node. The engine validates every boundary, records every run, and gives
the assistant memory across turns. The persisted structure is described in
:doc:`guides/data_model`.

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
       .build_engine(
           agent_service=AgentService(
               db_manager.get_sqlite_sessionmaker()
           )
       )
   )

   state = await workflow.run(
       {"user_message": "Hi, I'm visiting from Tallinn!"}
   )

.. code-block:: text

   Welcome to Green Village—so lovely to have you here from Tallinn!
   choices: ['Thanks!', 'What's nearby?', 'Tell me about the pub']
   path   : start → reply → end
   tokens : 165

The same graph is equally expressible in YAML, and the same engine serves it
over HTTP. The example is developed in full in :doc:`tutorials/quickstart`.

Where to start
--------------

* Readers new to language models will find the vocabulary defined in
  :doc:`guides/concepts`.
* For a running system within a few minutes, begin at
  :doc:`tutorials/quickstart`.
* For the complete treatment, begin at :doc:`tutorials/llm_clients` and
  proceed in order.
* For the design and its rationale, see :doc:`tutorials/architecture`.
* For individual definitions, :doc:`reference/index` documents the YAML keys,
  the bundled tools and every environment variable.
* For an assessment against other frameworks, see :doc:`tutorials/comparison`.
* For worked recipes, from retrieval-augmented generation to self-correcting
  loops, see :doc:`cookbook/index`.
* Before deploying a change, :doc:`guides/evaluation` sets out how to gate it
  on a suite you can run in continuous integration for nothing.

.. toctree::
   :hidden:
   :caption: Get started

   tutorials/installation
   tutorials/quickstart
   tutorials/architecture

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
