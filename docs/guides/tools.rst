=====
Tools
=====

Agents and workflows are useful only when they can *act*. In Kaval.AI every
action goes through the :class:`~kavalai.FunctionKernel` — a single interface
that hosts three kinds of tool behind one calling convention. Routing actions
through one typed kernel keeps a pipeline observable (every call is the same
shape) and safe (every argument and return value is validated).

For a hands-on walkthrough see :doc:`../tutorials/agents`; for the tools
Kaval.AI already ships, see :doc:`../reference/tools`.

Three tool kinds, one interface
-------------------------------

Each tool is addressed by a URI, and the scheme tells the kernel where the work
happens:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - URI
     - Tool kind
   * - ``python://<name>``
     - A local Python function.
   * - ``rest://<server>.<tool>``
     - A REST endpoint on a registered server.
   * - ``mcp://<server>.<tool>``
     - A tool exposed by an MCP server.

Regardless of kind, you call a tool the same way and get back a validated output
model instance:

.. code-block:: python

   result = await kernel.call_tool("python://fs.ls", arguments={...})
   # for wrapped returns, read result.result

Typed in, typed out
-------------------

Every tool has a Pydantic **input model** and **output model**. Before a call
the kernel validates and coerces the arguments; after a call it coerces the
return value (so a tool that yields ``"50"`` becomes ``50``). This is the same
typed-I/O guarantee that node boundaries enjoy — see :doc:`safety`.

There are only two shapes to remember:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - The tool returns
     - You get back
   * - A Pydantic model
     - That model, validated.
   * - Anything else — ``str``, ``int``, ``dict``, a list, an MCP payload
     - A generated single-field model. The value is ``result.result``, whole
       and unmodified.

A return value that cannot satisfy the model raises
``FunctionKernelException``. It used to be logged as a warning and the raw
value handed back instead, which meant a caller could silently receive
something other than what the tool's own signature promised.

Python tools
------------

A Python tool is an ordinary function decorated with
:func:`~kavalai.pythontool`. The decorator sets an internal flag; the function
stays directly callable. Register it to expose it:

.. code-block:: python

   from kavalai import FunctionKernel, pythontool

   @pythontool
   def ls(path: str) -> list[str]:
       ...

   kernel = FunctionKernel()
   kernel.register_python_tool("fs.ls", ls)   # -> python://fs.ls

The input model is generated from the signature: each parameter becomes a field,
its type hint becomes the field type, a default makes it optional, and a missing
hint becomes ``Any``. For the output model, a returned Pydantic ``BaseModel`` is
used as-is; anything else is wrapped in a model with a single ``result`` field.

REST tools
----------

Register a server, then its tools:

.. code-block:: python

   from kavalai import FunctionKernel, RestServer

   kernel.register_rest_server(RestServer(name=..., url=...))
   kernel.register_rest_tool(
       server_name, tool_name, method,
       input_schema, output_schema, description,
   )

MCP tools
---------

For MCP, register a server; the kernel starts the process, speaks MCP over
stdio, discovers the tools it offers and routes calls to them:

.. code-block:: python

   from kavalai import FunctionKernel, McpServer

   kernel.register_mcp_server(McpServer(name=..., command=..., args=[...]))
   await kernel.connect_mcp_servers()   # start the process(es), list their tools
   ...
   await kernel.close()                 # shut them down

Registration only records the configuration. ``connect_mcp_servers()`` is what
starts each process and asks it what it offers, and it is worth calling
explicitly: a misconfigured server then fails at startup rather than in the
middle of a run, before any tokens are spent.

You are not required to. ``get_tool_descriptions()`` connects anything still
unconnected before it answers, and a tool call connects on demand, so the
tools are never invisible to a model. Connecting up front just moves the
subprocess spawn off the request path.

The connections belong to the kernel, which usually means they outlive any one
run. ``WorkflowEngine`` wraps both ends of this as ``await engine.connect()``
and ``await engine.aclose()``, or use it as an async context manager.

Both :class:`~kavalai.RestServer` and :class:`~kavalai.McpServer` are importable
from the top-level :mod:`kavalai` package.

Introspection
-------------

Beyond ``call_tool``, the kernel exposes ``get_input_model``,
``get_output_model``, ``get_tool_descriptions``, and ``get_tool_definition`` —
which is exactly how an :doc:`agent <agents>` learns what tools it may use, and
how a ``function`` workflow node resolves the URI it was given.
