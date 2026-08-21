==========================
Core concepts
==========================

This page explains the ideas the rest of the documentation assumes. If you have
built with LLMs before, skim it. If you have not, it is the shortest path to
reading everything else comfortably.

What a language model actually does
-----------------------------------

A **large language model** (LLM) takes text and predicts what text comes next.
That is the whole primitive. Everything else — chat, agents, tool use — is built
on top of it by arranging what goes in.

Three consequences shape every design decision in Kaval.AI:

**A model is stateless.** It has no memory between calls. A chatbot that
"remembers" is one where your code sends the earlier turns again with every
request. Kaval.AI does that for you (see `Sessions and runs`_).

**A model is non-deterministic.** The same prompt can produce different words
each time. This is why workflows are explicit graphs rather than a model
deciding control flow, and why you can swap in a stub client for tests.

**A model does not know what it does not know.** Asked about your database it
will answer anyway, plausibly. Retrieval (see `RAG`_) is the fix.

Tokens
------

Models read and write **tokens** — chunks of text roughly ¾ of a word. "Green
Village" is about three tokens. Tokens matter for three reasons: providers bill
per token, every model has a maximum **context window** measured in tokens, and
generation time scales with the tokens produced.

Every Kaval.AI call reports its token usage as a
:class:`~kavalai.ModelCallStat`, and every workflow run aggregates them into
``token_usage``. See :doc:`observability`.

Prompts, and the roles in a conversation
----------------------------------------

A **prompt** is the text you send. A conversation is a list of messages, each
with a **role**:

* ``system`` — instructions that set behaviour ("You are a terse village guide").
* ``user`` — what the person said.
* ``assistant`` — what the model previously replied.

Kaval.AI models this as :class:`~kavalai.ChatHistory` of
:class:`~kavalai.ChatMessage`. ``prompt()`` is the one-message shortcut.

Structured output
-----------------

Left alone, a model returns prose. Prose is a bad interface for a program: to
get a field out of it you write a parser against wording that changes between
calls.

**Structured output** flips this. You declare a Pydantic model, pass it as
``response_model``, and the model is constrained to produce data in that shape,
validated before you see it:

.. code-block:: python

   class Classification(BaseModel):
       intent: str
       confidence: float

   result = await client.prompt("Classify this message…", response_model=Classification)
   result.confidence   # a float, not "about 80%"

This is the backbone of the whole library. Workflow ``data_types`` are the same
idea in YAML: every value crossing a node boundary is a validated model, so a
malformed value is caught at the boundary instead of three steps later. See
:doc:`../tutorials/llm_clients`.

Tools and tool calling
----------------------

A model can only produce text — it cannot query your database or send an email.
**Tool calling** bridges that: you describe the functions available, and the
model responds with *which* function to call and with what arguments. Your code
runs it and hands back the result.

The model never executes anything. It only ever asks. That is worth internalising
when reasoning about safety: an agent's capabilities are exactly the tools you
registered, no more.

Kaval.AI routes all of them through one :class:`~kavalai.FunctionKernel`, whether
they are Python functions, REST endpoints or MCP tools. See :doc:`tools`.

MCP
---

The **Model Context Protocol** is an open standard for exposing tools to LLM
applications. Instead of writing an integration per application, a service ships
one MCP server and any MCP-aware client can use it. Kaval.AI is such a client:
point it at a server and its tools join the kernel alongside your own.

Agents
------

An **agent** is a model in a loop with tools: it decides what to call, sees the
results, and decides again, until it can answer. That loop is what lets one
request span several steps — look up a resident, compute their age, then reply.

The loop is bounded by ``max_steps``, because a model that keeps deciding to
call one more tool would otherwise never stop. See :doc:`agents`.

Workflows
---------

An agent decides its own path. A **workflow** is the opposite: a graph you
define, where each node does one thing and the edges are explicit. The engine
walks from ``start`` to ``end``, validating every value along the way.

Most real systems want both. A workflow gives you predictability, observability
and typed boundaries; an ``agent`` node inside it gives you flexibility exactly
where flexibility is wanted. See :doc:`workflows`.

Sessions and runs
-----------------

Two words used precisely throughout these docs:

* A **run** is one execution of a workflow — one input in, one output out.
* A **session** is the conversation those runs belong to.

Reuse a session (by ``session_id``, or by your own ``external_id``) and the runs
in it share chat history — which is what makes memory work. See
:doc:`../tutorials/observability_storage`.

Embeddings
----------

An **embedding** turns text into a vector of numbers positioned so that similar
meanings land near each other. "Estonia's capital city is Tallinn" and "Tallinn
is the capital of Estonia" share almost no words but sit close together.

Closeness is measured by **cosine similarity**, a number from -1 to 1 where
higher means more similar. Normalised vectors make it a plain dot product.

RAG
---

**Retrieval-augmented generation** is the standard answer to "the model does not
know my data":

1. Embed your documents once and store the vectors.
2. At question time, embed the question and retrieve the nearest passages.
3. Put those passages in the prompt and ask the model to answer from them.

No retraining, and the answer is grounded in text you control — you can even cite
which passages were used. The usual failure mode is retrieval, not generation:
if the right passage was not retrieved, no prompt will save the answer. See
:doc:`../tutorials/rag`.

Where to next
-------------

* :doc:`../tutorials/quickstart` — the shortest path to a working example.
* :doc:`workflows`, :doc:`agents`, :doc:`tools` — each concept in depth.
* :doc:`safety` — the guarantees these choices buy you.
