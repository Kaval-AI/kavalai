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
:class:`~kavalai.AnthropicClient` or :class:`~kavalai.OllamaClient`, or
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
   )

When you pass a ``response_model`` the agent returns an instance of it; without
one it returns a plain string.

``history`` is the earlier turns of a conversation, as
:class:`~kavalai.ChatMessage` objects. They are sent with every step, so the
agent answers in the context of what was said before. The current user message
belongs in the prompt, not in the history, or the model reads it twice.

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
   an optional final output.
#. **Act** — the requested tool calls execute *in parallel* through the
   :class:`~kavalai.FunctionKernel`, and their results feed into the next step.
#. **Decide** — the loop stops when the model returns output with no further
   tool calls, or when ``max_steps`` is reached.

This explicit bound is a safety feature: an agent can never loop forever, and it
can only act through tools you have registered (see :doc:`tools` and
:doc:`safety`).

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
