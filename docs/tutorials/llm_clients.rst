LLM clients
===========

Kaval.AI ships native, async **LLM clients** with one small interface over every
provider. The headline methods are:

* ``prompt(message)`` — a single call that returns the model's answer.
* ``stream_prompt(message)`` — the same call, streamed as it is generated.
* ``chat_completions(chat_history=...)`` — a full multi-message conversation.

Each one takes an optional ``response_model`` (a Pydantic model) to get
**validated, structured output** instead of plain text. The clients are
standalone — you can use them on their own, or let a workflow build them for you.

The fastest way to construct one is :func:`~kavalai.make_client`, which picks the
right client from a ``"provider/model"`` string (``openai/…``, ``gemini/…``,
``anthropic/…``, ``ollama/…`` or ``browser/…``).

Try it in your browser: text vs. structured output
--------------------------------------------------

The ``browser/`` provider runs a small model **right on this page** over WebGPU —
no API key, no server — so you can try the two output modes without installing
anything. The snippets below have a **Run in browser ▶** button; the model id
comes from the panel's dropdown (exposed to your code as ``KAVAL_BROWSER_MODEL``).

**Text in, text out.** With no ``response_model``, ``prompt`` returns a plain
string:

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_client

   client = make_client(f"browser/{KAVAL_BROWSER_MODEL}")

   answer = await client.prompt("In one sentence, what is Tallinn?")
   print(type(answer).__name__, "->", answer)

**Structured in, structured out.** Pass a Pydantic ``response_model`` and the
model is constrained to that schema; you get back a *validated instance*, not a
string to parse:

.. code-block:: python
   :class: run-in-browser

   from pydantic import BaseModel
   from kavalai import make_client

   class City(BaseModel):
       name: str
       country: str
       fun_fact: str

   client = make_client(f"browser/{KAVAL_BROWSER_MODEL}")

   city = await client.prompt("Describe Tallinn.", response_model=City)
   print(type(city).__name__, "->", city)
   print("country  :", city.country)
   print("fun fact :", city.fun_fact)

That is the whole point of structured output: instead of coaxing facts out of
free-form prose, you declare the shape you want once and read typed fields
(``city.country``, ``city.fun_fact``) straight off the result. The same
``response_model`` works with every provider below.

Streaming responses
-------------------

For long answers you often do not want to wait for the whole response.
``stream_prompt`` returns a :class:`~kavalai.Streamer` you can iterate as the
model produces output:

.. code-block:: python

   client = make_client("openai/gpt-5.4-mini")

   streamer = await client.stream_prompt("Write a short story about a curious robot.")
   async for chunk in streamer:
       # chunk.type is "partial" while generating and "complete" at the end.
       print(chunk.value, end="", flush=True)

Streaming lowers *perceived* latency (text appears immediately), lets you show
progress on long generations, and makes it easy to cancel early. When you stream
**structured** output, every partial chunk is still valid JSON, so a UI can
render a partially-filled object safely.

This is just the gist — backpressure, timeouts and structured streaming are
covered in the dedicated :doc:`streamer` tutorial.

Provider clients: OpenAI, Gemini, Anthropic and Ollama
------------------------------------------------------

Outside the browser, Kaval.AI ships native clients for OpenAI, Google Gemini,
Anthropic (Claude) and Ollama. You can build them directly and pass the API key
two ways — straight to the constructor, or via an environment variable:

.. code-block:: python

   from kavalai import OpenAIClient, GeminiClient, AnthropicClient, OllamaClient

   # 1) Pass the key to the client...
   openai = OpenAIClient("gpt-5.4-mini", api_key="sk-...")

   # 2) ...or omit it and the client reads it from the environment.
   gemini = GeminiClient("gemini-3.1-flash-lite")     # reads GEMINI_API_KEY
   claude = AnthropicClient("claude-sonnet-5")        # reads ANTHROPIC_API_KEY
   ollama = OllamaClient("llama3")                    # local; reads OLLAMA_HOST

The environment variables and extra options per provider:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Provider
     - API key / host env var
     - Notes
   * - :class:`~kavalai.OpenAIClient`
     - ``OPENAI_API_KEY``
     - ``base_url`` for Azure / OpenAI-compatible endpoints.
   * - :class:`~kavalai.GeminiClient`
     - ``GEMINI_API_KEY``
     - Google Gemini models.
   * - :class:`~kavalai.AnthropicClient`
     - ``ANTHROPIC_API_KEY``
     - Claude models over the Messages API; ``base_url`` supported. Structured
       output uses closed JSON schemas, and every request needs an output
       ceiling — the client sends a generous ``max_tokens`` default. Leave
       ``temperature`` / ``top_p`` unset: recent Claude models reject them.
   * - :class:`~kavalai.OllamaClient`
     - ``OLLAMA_HOST`` (default ``http://localhost:11434``)
     - Runs locally; no API key.

:func:`~kavalai.make_client` is the shortcut — it builds the matching client from
a ``"provider/model"`` id and reads the same environment variables:

.. code-block:: python

   from kavalai import make_client

   client = make_client("openai/gpt-5.4-mini")
   reply = await client.prompt("Say hello in Estonian.")

**Inside a workflow** you rarely construct a client yourself — you name the
model and the engine builds it. Set it per node or as the workflow default with
``llm_model="openai/gpt-5.4-mini"``, or set the ``KAVALAI_DEFAULT_LLM_MODEL``
environment variable so you can omit ``llm_model`` entirely:

.. code-block:: bash

   export KAVALAI_DEFAULT_LLM_MODEL="openai/gpt-5.4-mini"

Using clients without a workflow
--------------------------------

Everything above used the clients on their own — no workflow, no engine. The
same clients power workflow ``llm`` nodes under the hood, but you can drop one
into any async code to call a model, get structured output or stream. Reach for
:doc:`workflows <workflow>` when you want to orchestrate several steps, branch on
results or call tools; reach for a client directly when you just need a model.

Model statistics and observability
-----------------------------------

Every call reports a :class:`~kavalai.ModelCallStat` — the model, prompt /
completion / total token counts, the HTTP status and the wall-clock duration. By
default Kaval.AI logs these through ``loguru`` with the built-in
:class:`~kavalai.ModelStatsLogger`. Pass your own :class:`~kavalai.ModelStatsReceiver`
to send them anywhere — a metrics backend, a database, or just stdout:

.. code-block:: python

   from kavalai import OpenAIClient, ModelStatsReceiver, ModelCallStat

   class PrintStats(ModelStatsReceiver):
       def receive_model_stats(self, stats: ModelCallStat):
           print(f"{stats.model}: {stats.total_tokens} tokens "
                 f"in {stats.duration_seconds:.2f}s")

   client = OpenAIClient("gpt-5.4-mini", model_stats_receiver=PrintStats())
   await client.prompt("What is 2 + 2?")

Inside workflows these stats are aggregated per run (``WorkflowState.token_usage``)
and surfaced in the backoffice UI as per-call token and cost metrics — see
:doc:`../guides/observability`.

Timeouts and retries
--------------------

Reliability and sampling are controlled with :class:`~kavalai.LlmClientParameters`:

.. code-block:: python

   from kavalai import OpenAIClient, LlmClientParameters

   client = OpenAIClient(
       "gpt-5.4-mini",
       llm_client_parameters=LlmClientParameters(
           temperature=0.2,
           timeout_seconds=60,   # cap each attempt (default: 30s)
       ),
   )

``timeout_seconds`` bounds each request. On **transient** failures — rate limits,
timeouts, dropped connections and 5xx errors — the client retries automatically
with exponential backoff (up to 5 attempts, with jitter). It does **not** retry
on errors you should fix yourself: authentication failures, ``404`` responses
and other bad requests are raised immediately.

Sampling parameters (``temperature``, ``top_p``, …) default to ``None`` and are
only sent when you set them, so each provider's own defaults apply otherwise.
That matters for Claude: recent models reject sampling parameters outright, so
leaving them unset is what keeps the same ``LlmClientParameters`` usable across
providers.

When you **stream**, a second timeout guards the gap *between* chunks:
``stream_timeout_seconds``. It defaults to twice ``timeout_seconds`` so a stream
survives one timed-out attempt plus its retry backoff — a single
``timeout_seconds`` would kill the retry it is supposed to allow. Each retry
also pushes a ``restart`` chunk (``chunk.type == "restart"``), which tells
consumers to discard what they have accumulated for that stream: the new attempt
re-sends its content from the beginning.

.. code-block:: python

   client = OpenAIClient(
       "gpt-5.4-mini",
       llm_client_parameters=LlmClientParameters(
           timeout_seconds=60,          # per attempt
           stream_timeout_seconds=180,  # inter-chunk inactivity (default: 2x)
       ),
   )

Where to next
-------------

* :doc:`streamer` — stream text and structured output in depth.
* :doc:`workflow` — wire clients into typed, branching workflows.
* :doc:`agents` — the ``FunctionKernel`` and the multi-step ``Agent`` loop.
* :doc:`../guides/observability` — stats, tracing and the backoffice UI.
