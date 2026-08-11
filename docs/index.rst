.. image:: _static/iconlogo.svg
   :width: 300
   :align: left
   :alt: Kaval.AI

**Kaval.AI is an opinionated, elegant, production-grade library for building
LLM-powered workflows, chatbots and agents, and connecting them to databases and
tooling — with a focus on observability and debuggability.**

Kaval.AI uses `Pydantic <https://pydantic.dev/docs/validation/latest/get-started/>`__
for every workflow input and output, and for all intermediate steps including
tool calls.


.. code-block:: python

   from pydantic import BaseModel

   class Message(BaseModel):
       message: str

   class Reply(BaseModel):
       agent_response: str
       choices: list[str]


Using structured data types makes developing agentic AI workflows pleasant and
predictable. The simplest way to create a workflow is the
:class:`~kavalai.WorkflowBuilder`:

.. code-block:: python

   from kavalai.agent_service import AgentService
   from kavalai.db import db_manager
   from kavalai.workflow import WorkflowBuilder

   # A short example or two helps a tiny model answer well and fill the choices
   # in the right shape; the structured output schema enforces the format.
   prompt = """You are a friendly, concise chatbot. Reply to the user in
   agent_response, and suggest up to 3 short quick-reply choices the user might
   tap next. Use earlier messages in the conversation for context.

   Example:
   user: hi
   agent_response: Hey there! What can I help you with today?
   choices: ["What can you do?", "Tell me a joke", "Give me an idea"]
   """

   workflow = (
       WorkflowBuilder("Chatbot", llm_model="browser/Llama-3.2-1B-Instruct-q4f32_1-MLC")
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

Let's go over the code step-by-step.

.. code-block:: python

   WorkflowBuilder("Chatbot", llm_model="browser/Llama-3.2-1B-Instruct-q4f32_1-MLC")

This creates the builder for a workflow named ``Chatbot``. The ``llm_model``
argument is the default model every LLM node uses unless it overrides it.
Kaval.AI supports several providers — OpenAI, Gemini, Anthropic, Ollama and an
in-browser WebGPU provider; see :doc:`/tutorials/llm_clients`.

Here we use the ``browser`` provider so a small model runs **right in this page**
with no API key, backed by the in-browser ``BrowserLLMClient`` (see
:doc:`/api/llm_clients`). ``Llama-3.2-1B-Instruct-q4f32_1-MLC`` is a tiny model
that downloads on first use; the live playground below also lets you pick a
different one from its dropdown.

.. code-block:: python

   .data_model("input", Message)
   .data_model("output", Reply)

``data_model`` registers a Pydantic model as a workflow data type. ``input`` and
``output`` are special: they are the workflow's overall input and output types.
Declaring them explicitly is what lets the runner validate every value and ask
the model for exactly the right shape.

.. code-block:: python

   .start("reply")

This names the first node to run — here, the ``reply`` node defined next.

.. code-block:: python

   .llm(
       "reply",
       prompt=prompt,
       inputs={"message": "input"},
       output="output",
       next="end",
   )

This adds an LLM node named ``reply``. ``inputs`` maps a **local name** to a
value from the run context: ``{"message": "input"}`` takes the workflow's
``input`` and hands it to the prompt under the name ``message``, so the model
sees the message text next to your instructions. ``output="output"`` stores the
model's answer — validated against ``Reply`` — under ``output``, and
``next="end"`` says which node runs after this one.

.. code-block:: python

   .build_engine(
       agent_service=AgentService(db_manager.get_sqlite_compat_sessionmaker())
   )

Finally, ``build_engine`` validates the graph and returns a ready-to-run
:class:`~kavalai.WorkflowEngine`. Passing an
:class:`~kavalai.agent_service.AgentService` gives the bot **memory**: LLM
nodes read the conversation history by default (``use_history=True``), so each
turn sees what was said before. The chat keeps one session, stored here in an
in-browser SQLite database that lives for the page's lifetime; in production
the very same service points at Postgres instead.

Here is that chatbot, running entirely in your browser. The first run takes a
moment to download the model (or load it from cache), and because the model is
tiny its answers will vary — but it remembers the conversation, and its reply is
a structured ``Reply`` whose ``choices`` become the quick-reply buttons.

.. raw:: html
   :file: _includes/chatbot-demo.html


Under the hood that chatbot is a tiny workflow — one LLM node between ``start``
and ``end``:

.. image:: /_static/workflows/chatbot.svg
   :alt: The chatbot workflow: start → reply (an LLM node) → end.
   :align: center

This is just a small glimpse of what Kaval.AI can do. Check out the tutorials and
examples — they cover a wide range of topics for building production-grade
agentic workflows.

Get started
-----------

Kaval.AI is more than a workflow engine. It ships native LLM clients for OpenAI,
Gemini, Anthropic and Ollama behind one streaming interface (plus the in-browser
WebGPU client you just used); a :class:`~kavalai.FunctionKernel` that exposes Python,
REST and MCP tools to your agents through a single validated interface; a
:class:`~kavalai.RagService` for indexing and querying embeddings, so agents can
answer from your own documents (retrieval-augmented generation); and a
**backoffice UI** for configuring agents and monitoring conversations, runs,
tasks, token usage and cost. The guides, tutorials and API reference below cover
each of these in depth.

.. toctree::
   :maxdepth: 1

   Installation <tutorials/installation>

.. toctree::
   :maxdepth: 2
   :caption: Learn

   tutorials/index
   guides/index

.. toctree::
   :maxdepth: 2
   :caption: Operate

   ui/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
