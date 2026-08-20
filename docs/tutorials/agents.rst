Agents & tools
==============

An LLM on its own can only produce text. To *do* things — query a database, open
a support ticket, send an email, scrape a website, call an internal API — an
**agent** needs **tools**. Agentic workflows are mostly the art of giving a model
the right tools and letting it decide when to use them.

In Kaval.AI a tool is any Python function, REST endpoint or MCP tool, all reached
through one uniform, type-checked interface. Two pieces work together:

* The :class:`~kavalai.FunctionKernel` hosts tools and calls them, validating
  every input and output against a Pydantic model.
* The :class:`~kavalai.Agent` is a small reasoning loop: it picks tools, calls
  them through the kernel, reads the results, and repeats until it has a typed
  answer.

This page builds tools from plain Python functions (try them right here in your
browser), then hands them to an agent, and finally shows the REST and MCP tool
types.

Your first tools
----------------

A tool is just a function decorated with ``@kavalai.pythontool``. The decorator
marks it as a Kaval.AI tool but does **not** change its behaviour, so you can
still call the function directly. Register it on a :class:`~kavalai.FunctionKernel`
and it becomes callable through a ``python://<name>`` URI. This runs entirely in
your browser — there is no model involved yet, just tools:

.. code-block:: python
   :class: run-in-browser

   from datetime import datetime, timezone
   from kavalai import FunctionKernel, pythontool

   @pythontool
   def get_time() -> str:
       """Return the current UTC time as an ISO-8601 string."""
       return datetime.now(timezone.utc).isoformat(timespec="seconds")

   @pythontool
   def add(a: int, b: int) -> int:
       """Add two integers."""
       return a + b

   # A @pythontool function is unchanged — you can still call it directly.
   print("direct:", add(2, 40))

   # Register the tools on a kernel; each is now callable via a python:// URI.
   kernel = FunctionKernel()
   kernel.register_python_tool("get_time", get_time)
   kernel.register_python_tool("add", add)

   # Call them through the kernel. Plain-return tools wrap their value in
   # `.result`; a tool that returns a Pydantic model exposes its fields directly.
   now = await kernel.call_tool("python://get_time")
   total = await kernel.call_tool("python://add", {"a": 2, "b": 40})
   print("time :", now.result)
   print("2 + 40 =", total.result)

The tool **name** (``"get_time"``) is what the kernel exposes; dotted names like
``fs.read`` are just a convention for grouping related tools.

Typed inputs and outputs
------------------------

The kernel builds an **input** and an **output** Pydantic model from each
function's signature, so an agent always works with validated, well-typed data:

* every parameter becomes an input field — its type hint is the field type, and a
  default value makes the field optional;
* the return annotation becomes the output type (wrapped in a ``result`` field
  unless the function already returns a Pydantic model).

Because the kernel validates against those models, arguments are **coerced** to
the declared types — the string ``"50"`` arrives at the function as the int
``50``. Try it:

.. code-block:: python
   :class: run-in-browser

   from kavalai import FunctionKernel, pythontool

   @pythontool
   def add(a: int, b: int) -> int:
       """Add two integers."""
       return a + b

   kernel = FunctionKernel()
   kernel.register_python_tool("add", add)

   print("input fields :", list(kernel.get_input_model("python://add").model_fields))
   print("output fields:", list(kernel.get_output_model("python://add").model_fields))

   # "50" is coerced to an int before the function runs.
   total = await kernel.call_tool("python://add", {"a": "50", "b": 1})
   print("'50' + 1 =", total.result)

Agents: tools in a loop
-----------------------

An :class:`~kavalai.Agent` wraps an LLM client and a kernel into a reasoning loop.
On each step the model returns the tools it wants to call plus an optional final
answer; the agent runs those calls through the kernel, feeds the results back,
and repeats until the model is done or ``max_steps`` is reached. Pass a
``response_model`` and the final answer is validated into your Pydantic type.

Agents need a capable model, so this example uses a provider client (see
:doc:`llm_clients` for setup) — it is not run in the browser. The tool here is a
mock ``create_ticket``; swap in a real call to your ticketing system:

.. code-block:: python

   from uuid import uuid4
   from pydantic import BaseModel
   from kavalai import Agent, FunctionKernel, pythontool, OpenAIClient

   @pythontool
   def create_ticket(subject: str, body: str, priority: str = "normal") -> dict:
       """Open a support ticket and return its id and tracking URL."""
       ticket_id = "T-" + uuid4().hex[:6].upper()
       return {"ticket_id": ticket_id,
               "url": f"https://support.example.com/{ticket_id}"}

   kernel = FunctionKernel()
   kernel.register_python_tool("create_ticket", create_ticket)

   class Handled(BaseModel):
       ticket_id: str
       url: str
       reply: str   # a short message to send back to the customer

   # Any client from the LLM clients tutorial works here.
   agent = Agent(llm_client=OpenAIClient("gpt-5.4-mini"), kernel=kernel)

   result = await agent.prompt(
       "A customer writes: 'My router order #123 won't turn on.' "
       "Open a support ticket for it, then write a short, friendly reply that "
       "includes the ticket link.",
       response_model=Handled,
       max_steps=5,
   )
   print(result)

A typical run takes two steps: the model calls ``create_ticket`` with a subject
and body, then — given the returned id and url — fills in ``Handled`` with a
reply for the customer. With no ``response_model`` the agent returns a plain
string instead.

REST and MCP tools
------------------

Python functions are one of three tool types behind the same kernel interface.

**REST tools** wrap an HTTP endpoint. Register a server with its base URL, then
each endpoint as a tool with its method and input/output JSON schemas. This uses
the free `Open-Meteo <https://open-meteo.com/>`_ API — no key required:

.. code-block:: python

   from kavalai import Agent, FunctionKernel, OpenAIClient, RestServer

   kernel = FunctionKernel()
   kernel.register_rest_server(RestServer(name="weather", url="https://api.open-meteo.com/v1"))
   kernel.register_rest_tool(
       server_name="weather",
       tool_name="forecast",
       method="GET",
       input_schema={
           "type": "object",
           "properties": {
               "latitude": {"type": "number"},
               "longitude": {"type": "number"},
               "current": {"type": "string", "description": "e.g. temperature_2m"},
           },
           "required": ["latitude", "longitude"],
       },
       output_schema={"type": "object", "properties": {"current": {"type": "object"}}},
       description="Current weather for a GPS location from Open-Meteo.",
   )

   agent = Agent(llm_client=OpenAIClient("gpt-5.4-mini"), kernel=kernel)
   print(await agent.prompt("What's the current temperature in Tallinn, Estonia?", max_steps=3))

**MCP tools** come from a `Model Context Protocol <https://modelcontextprotocol.io/>`_
server. Register it with a command; the kernel starts the process, discovers its
tools over stdio and routes calls to them (needs the ``mcp`` extra):

.. code-block:: python

   import sys, textwrap
   from kavalai import Agent, FunctionKernel, OpenAIClient, McpServer

   # A tiny MCP server exposing two math tools.
   server_path = "/tmp/demo_mcp.py"
   with open(server_path, "w") as f:
       f.write(textwrap.dedent('''
           from mcp.server.fastmcp import FastMCP
           mcp = FastMCP("demo-math")

           @mcp.tool()
           def add(a: float, b: float) -> float:
               """Add two numbers."""
               return a + b

           @mcp.tool()
           def multiply(a: float, b: float) -> float:
               """Multiply two numbers."""
               return a * b

           if __name__ == "__main__":
               mcp.run()
       '''))

   kernel = FunctionKernel()
   kernel.register_mcp_server(McpServer(name="math", command=sys.executable, args=[server_path]))

   agent = Agent(llm_client=OpenAIClient("gpt-5.4-mini"), kernel=kernel)
   print(await agent.prompt("What is (42 + 8) multiplied by 7?", max_steps=3))
   await kernel.close()

Kaval.AI also ships ready-made tools — for example a Crawl4AI web scraper that
renders a page in a headless browser and returns clean Markdown, plus a
Crawl4AI-backed ``web_search`` tool that needs no search API key
(``pip install "kavalai[common]"``). ``examples/business_info_agent.py`` combines
both in a workflow — a search node finds candidate pages, an agent node
restricted to the crawl tool reads the promising ones, and an LLM node writes
the summary. Run with ``KAVALAI_DB_URI`` pointing at a migrated agent database,
the engine records every company it researches — a session and run per company,
a task per node, the chat history and the model-call stats — which is a quick
way to fill the backoffice with real data.

Where to next
-------------

* :doc:`llm_clients` — the clients that power an agent's reasoning.
* :doc:`workflow` — orchestrate agents and tools as typed, branching workflows.
* :doc:`../guides/observability` — trace every tool call and model call.
