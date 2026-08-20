Installation
============

This guide takes you from an empty environment to a running chatbot in a few
minutes: install Kaval.AI, configure an LLM provider, and run your first
**workflow** — a small typed state machine that takes an input and returns a
structured result.

Install Kaval.AI
----------------

Kaval.AI targets **Python 3.12+**. Install it into a virtual environment. The
example below calls OpenAI, so install the ``common`` extra.

With ``uv`` (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv venv
   uv pip install "kavalai[common]"

With ``pip``
^^^^^^^^^^^^

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install "kavalai[common]"

The bare ``kavalai`` package is deliberately small and provider-agnostic: it is
restricted to libraries that also work under Pyodide, so the core can be
imported and run in the browser. Two extras cover the rest:

* ``kavalai[common]`` — everything else: the OpenAI, Gemini, Anthropic and
  Ollama clients, embeddings and RAG, the PostgreSQL drivers, MCP, the REST/SSE
  servers, and the bundled web tools.
* ``kavalai[common_web]`` — the browser counterpart, for running the core under
  Pyodide / WebLLM (see :doc:`run_in_browser`).

Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Kaval-AI/kaval.ai.git
   cd kaval.ai
   uv pip install -e ".[common]"

Configure a provider
---------------------

LLM nodes call a provider, so it needs an API key. Export it in your shell (or a
``.env`` file)…

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."

…or set it from Python, as the example below does.

Your first workflow: a chatbot
------------------------------

A workflow is built from typed **data models** and **nodes**. The example below
is a one-node chatbot: it validates the incoming ``Message``, asks the model to
reply, and returns a structured ``Reply`` — both the prose answer and a list of
suggested quick-reply ``choices``. An in-memory ``AgentService`` gives it
memory, so each turn can use the conversation so far.

.. code-block:: python

   import asyncio
   import os

   from pydantic import BaseModel
   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder

   # Paste your OpenAI key here, or export OPENAI_API_KEY in your shell / .env.
   os.environ["OPENAI_API_KEY"] = ""


   class Message(BaseModel):
       message: str


   class Reply(BaseModel):
       agent_response: str
       choices: list[str]


   # A short example or two helps the model fill the choices in the right shape;
   # the structured output schema enforces the format.
   prompt = """You are a friendly, concise chatbot. Reply to the user in
   agent_response, and suggest up to 3 short quick-reply choices the user might
   tap next. Use earlier messages in the conversation for context.

   Example:
   user: hi
   agent_response: Hey there! What can I help you with today?
   choices: ["What can you do?", "Tell me a joke", "Give me an idea"]
   """

   workflow = (
       WorkflowBuilder("Chatbot", llm_model="openai/gpt-5.4-mini")
       .data_model("input", Message)
       .data_model("output", Reply)
       .start("reply")
       .llm(
           "reply",
           prompt=prompt,
           inputs={"message": "input"},
           output="output",
           next="end",
       )
       .end()
       .build_engine(
           agent_service=AgentService(db_manager.get_sqlite_compat_sessionmaker())
       )
   )


   async def main():
       state = await workflow.run({"message": "Hi, what can you do?"})
       print(state.output_data["agent_response"])
       print("choices:", state.output_data["choices"])


   asyncio.run(main())

Every run returns a :class:`~kavalai.WorkflowState`: the status, the ordered
``trace`` of visited nodes, the full context ``data``, the final ``output_data``,
and an aggregate ``token_usage``. It is JSON-serialisable, and the run's output
and context are persisted through the agent service, so runs can be reloaded
and inspected later.

Try it with no install or API key
---------------------------------

Want to experiment without installing anything or signing up for a provider?
Kaval.AI can run a small open model **entirely in your browser** over WebGPU.
See :doc:`run_in_browser`.

Where to next
-------------

* :doc:`workflow` — branching, function/agent nodes, observability and
  deterministic testing.
* :doc:`agents` — the ``FunctionKernel`` and the multi-step ``Agent`` loop.
* :doc:`llm_clients` — calling models directly, structured output and streaming.
* :doc:`../guides/index` — the concepts behind the engine.
* :doc:`../ui/index` — manage and monitor agents in the backoffice UI.
