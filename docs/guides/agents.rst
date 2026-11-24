======
Agents
======

An :class:`~kavalai.Agent` is a lightweight, multi-step reasoning loop: give it
a task and a set of tools, and it works toward an answer one step at a time. The
loop is deliberately small and explicit — bounded by ``max_steps`` and driven by
tools addressed by URI — so that agentic behaviour stays predictable and
auditable rather than open-ended.

For a hands-on walkthrough, see :doc:`../tutorials/agents`.

Construction
------------

.. code-block:: python

   from kavalai import Agent, OpenAIClient, FunctionKernel

   agent = Agent(
       llm_client=OpenAIClient("gpt-5.6-luna"),
       kernel=FunctionKernel(),   # optional
       run_context=...,           # optional
       prompt_template=...,       # optional Jinja2 Template, system message
       step_template=...,         # optional Jinja2 Template, step message
       allowed_tools=None,        # optional; None = every registered tool
       on_step=None,              # optional observer, called after each step
       debug=False,               # print each step's reasoning and tool calls
   )

Any provider client works, since they share the :class:`~kavalai.BaseLlmClient`
interface — :class:`~kavalai.OpenAIClient`, :class:`~kavalai.GeminiClient`,
:class:`~kavalai.AnthropicClient`, :class:`~kavalai.ScalewayClient` or
:class:`~kavalai.OllamaClient`, or
``make_client("gemini/gemini-3.1-flash-lite")``. See
:doc:`../tutorials/llm_clients`.

``on_step`` is called once per completed step with that step's record — its
tool calls, their results and durations, and the step's output. The workflow
engine uses it to turn an agent node's tool calls into task rows.

Running a prompt
----------------

.. code-block:: python

   result = await agent.prompt(
       "Summarise the latest filings",
       response_model=MySchema,   # optional Pydantic model
       max_steps=10,
       history=earlier_turns,     # optional list of ChatMessage
       inputs={"city": "Turku"},  # optional context variables
   )

When you pass a ``response_model`` the agent returns an instance of it; without
one it returns a plain string.

``history`` is the earlier turns of a conversation, as
:class:`~kavalai.ChatMessage` objects. They are sent with every step, so the
agent answers in the context of what was said before. The current user message
belongs in the prompt, not in the history, or the model reads it twice.

``inputs`` is the mapping the system prompt lists as **context variables**, and
what a tool call's ``input`` reference resolves against (see
:ref:`agents-tool-calls`). Without it the agent lists
the run context's ``data``. The workflow engine passes an agent node's
resolved ``inputs``, so the model sees the values the node declares — a
retrieved passage list, a classification — and not every output the run has
produced, which would otherwise be rendered in full on every step.

The four-step cycle
-------------------

Each step of the loop performs the same four operations:

#. **Render** the conversation for this step: a system message from
   ``prompt_template`` (the task, the context variables and the tool
   descriptions), the chat ``history`` if there is one, and a user message from
   ``step_template`` (the steps executed so far, with their tool results, and
   the instruction to produce the next step). Keeping the step trace in the
   last message leaves the system prompt constant across the steps of a run,
   which is what a provider's prompt cache can reuse.
#. **Reason** — the LLM returns a ``StepOutput``: a list of ``tool_calls`` plus
   an optional final output (see :ref:`agents-tool-calls`).
#. **Act** — the requested tool calls execute *in parallel* through the
   :class:`~kavalai.FunctionKernel`, and their results feed into the next step.
#. **Decide** — the loop stops when the model returns output with no further
   tool calls, or when ``max_steps`` is reached.

This explicit bound is a safety feature: an agent can never loop forever, and it
can only act through tools you have registered (see :doc:`tools` and
:doc:`safety`).

.. _agents-tool-calls:

How a tool call is written
--------------------------

A step is one structured output, and the model is asked for it with a schema
built from the tools the agent may use: one call model per tool, with ``name``
fixed to that tool's URI and ``args`` held to its input schema. A provider that
enforces its structured output therefore cannot produce a call that misspells
an argument or gives it a value of the wrong type. The first step of an agent
with one tool, ``get_weather(city, unit)``, asked for the weather in Tartu:

.. code-block:: json

   {
     "instructions": "Get the current weather in Tartu in Celsius.",
     "tool_calls": [
       {
         "name": "python://get_weather",
         "args": {"city": "Tartu", "unit": "celsius"},
         "refs": [],
         "call_id": "weather_tartu"
       }
     ],
     "output": null
   }

``args``
    The argument values the model writes out. A ``null`` means the argument
    is not supplied, so the tool's default applies.
``refs``
    Arguments taken from elsewhere, so that the model does not retype them.
    Each entry names the argument (``arg``), a ``source`` and a ``key``:
    ``{"arg": "document", "source": "step", "key": "doc"}`` passes the whole
    output of the earlier tool call whose ``call_id`` is ``doc``, and
    ``"source": "input"`` passes the context variable named by ``key``. A
    value written in ``args`` takes precedence over a reference to the same
    argument.
``call_id``
    The name under which later steps refer to the call's result.

A tool whose input has a free-form value — a ``dict`` parameter, and most MCP
and REST tools, whose nested objects are not described — cannot be typed,
because a schema that enforces nothing is refused by strict structured output.
Such a tool takes ``args_json`` in place of ``args``: the same object, written
as a JSON string. Declaring a Pydantic model for the parameter makes the tool
typed.

Nothing about a call's arguments ends a run. Arguments that are not valid
JSON, a reference to a ``call_id`` or a context variable that does not exist,
and arguments the tool refuses each become that call's result, as an error
message the model reads on its next step. A tool is not run with part of its
arguments.

Not every provider holds a model to the schema it is given. The agent
therefore reads a reply more leniently than it asks for it, and accepts the
form used before arguments were typed — ``literal_args``, ``planner_context_args`` and
``input_args`` as JSON strings — where a model, or a scripted test reply,
still writes it.

A custom ``prompt_template`` has to describe this form. It receives
``json_argument_tools``, the URIs of the tools that take ``args_json``.

Narrow it further with ``allowed_tools``, a list of tool URIs
(``python://web.crawl``, or ``rest://api.*`` for a whole server). Excluded tools
are neither described to the model nor callable, so the restriction is enforced
rather than suggested. ``None`` allows every registered tool; an empty list
allows none.

Structured output
-----------------

Because the final answer can be validated against a ``response_model``, an agent
fits cleanly into a typed pipeline — its output is another typed value,
the same as any other node boundary.

The ``agent`` workflow node
---------------------------

The ``agent`` node in a workflow graph runs this exact same loop inside the
graph, with its own ``max_steps``. So you can drop an agent into a larger,
deterministic :doc:`workflow <workflows>` and still get the per-node trace and
token accounting described in :doc:`observability`.

The node fills ``history`` from the session's chat history when
``use_history`` is on (the default), windowed by ``history_limit`` and
``history_max_chars`` as on an ``llm`` node, and leaves out the current user
message because the node's ``prompt`` carries it. The agent's intermediate steps
are not written to the chat history: the session records the user message and
the workflow's answer, and the tool calls go to the task log.
