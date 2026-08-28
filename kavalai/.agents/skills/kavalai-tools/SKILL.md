---
name: kavalai-tools
description: Give a Kaval.AI agent or workflow tools — `@pythontool` functions, REST servers, MCP servers — and scope what a node may call with `allowed_tools`. Use when registering a tool, writing a tool URI, wiring MCP, using the bundled web/HTTP tools, or when a tool call fails validation or a tool is invisible to the model.
---

# Kaval.AI tools

Every action goes through one `FunctionKernel`. Three kinds of tool, one
calling convention, and every argument and return value validated.

| URI | Tool kind |
|---|---|
| `python://<name>` | A local Python function |
| `rest://<server>.<tool>` | A REST endpoint on a registered server |
| `mcp://<server>.<tool>` | A tool exposed by an MCP server |

```python
result = await kernel.call_tool("python://fs.ls", arguments={"path": "/tmp"})
```

## Typed in, typed out

There are only two shapes to remember:

| The tool returns | You get back |
|---|---|
| A Pydantic model | That model, validated |
| Anything else — `str`, `int`, `dict`, a list, an MCP payload | A generated single-field model; the value is `result.result`, whole and unmodified |

Arguments are validated and coerced before the call, the return value after (so
a tool yielding `"50"` becomes `50`). A return value that cannot satisfy the
model raises `FunctionKernelException` — it is never quietly handed back raw.
When reading a non-model return, remember `.result`.

## Python tools

```python
from pydantic import BaseModel
from kavalai import FunctionKernel, pythontool


class PondReading(BaseModel):
    depth_m: float
    water: str


@pythontool
def measure_pond(name: str) -> PondReading:
    """Measure a Green Village pond by name."""
    ...


kernel = FunctionKernel()
kernel.register_python_tool("measure_pond", measure_pond)   # python://measure_pond
```

- **The decorator is mandatory.** `register_python_tool` refuses a plain
  function with `Function 'f' must be decorated with @kavalai.pythontool`. The
  decorator only sets a flag; the function stays directly callable and
  directly testable. (A function named under a workflow's `python_functions`
  is decorated by the engine when it is not already, so the error is only
  raised on the Python API.)
- The input model is generated from the signature: each parameter becomes a
  field, its hint becomes the type, a default makes it optional, a missing hint
  becomes `Any`. **Type every parameter** — an untyped one becomes `Any` and
  the model gets no guidance.
- **The docstring and the type hints _are_ the interface the model sees.**
  Write them for a reader who knows nothing about your codebase. This is the
  highest-leverage thing you control about tool-calling accuracy: a vague
  docstring is a wrong tool call.
- Return a Pydantic model when the shape matters. It documents the tool and
  saves the caller from `.result`.

In a workflow, declare the same function under `python_functions` with its
import path:

```yaml
python_functions:
  - {name: measure_pond, path: green_village.tools.measure_pond}
```

## REST tools

**The YAML declares the _server_; the tools are registered in code**, because
they need input and output schemas. Declaring `rest_servers` alone gives you no
callable tools — this is the commonest REST mistake.

```python
from kavalai import FunctionKernel, RestServer

kernel.register_rest_server(RestServer(name="crm", url="https://crm.example.com/api"))
kernel.register_rest_tool(
    "crm", "lookup_customer", "get",
    input_schema, output_schema, "Look up a customer by id.",
)   # -> rest://crm.lookup_customer
```

Credentials belong in `username_env` / `password_env` on the server, never
inline.

## MCP tools

```python
from kavalai import FunctionKernel, McpServer

kernel.register_mcp_server(McpServer(name="village", command="python",
                                     args=["-m", "village_mcp"]))
await kernel.connect_mcp_servers()   # start processes, discover tools
...
await kernel.close()                 # shut them down
```

- Registration only records configuration. `connect_mcp_servers()` starts each
  process and asks what it offers. Call it explicitly at startup so a
  misconfigured server fails there rather than mid-run, before tokens are
  spent. You are not *required* to: `get_tool_descriptions()` connects anything
  unconnected before answering, and a call connects on demand.
- **stdio and HTTP are mutually exclusive, and one is required.** Give either
  `command`/`command_env` (+ `args`, `env`) or `url`/`url_env` — and only one
  of each pair.
- **Connections are kernel lifetime, never per run.** `WorkflowEngine` wraps
  both ends as `await engine.connect()` / `await engine.aclose()`, or use it as
  an async context manager. Opening a kernel or engine per request is a bug.
- Server names are unique; a duplicate raises `FunctionKernelException`
  (`MCP server 'x' is already registered.`), a subclass of
  `WorkflowException`. The same holds for REST servers and Python tool names.

## Secrets in a workflow file

Use `url_env`, `command_env`, `username_env`, `password_env` — they name an
environment variable, so the workflow file stays committable.

**`mcp_servers[].env` is the exception**: it takes *literal* values, so an API
key written there sits in the file. Those values are redacted from the agent
server's `GET /workflow`, but the file itself is not protected. When a stdio
server needs a credential, use `command_env` and a wrapper script instead.

## Scoping what an agent may call

`allowed_tools` applies to both the tools described to the model and the tools
it may execute. Identical in YAML and in `Agent(allowed_tools=…)`:

```yaml
allowed_tools: ["*"]                         # every tool
allowed_tools: ["mcp://github.*"]            # one whole server
allowed_tools: ["python://lookup_resident"]  # one tool
allowed_tools: []                            # none
# key omitted                                # same as ["*"]
```

Scope every agent node deliberately. An agent has no implicit capability — only
what the kernel holds and this list permits — and that is the property worth
keeping.

## The setup module

The agent server imports a module *before* the workflow loads, which is where
`python://` tools and named RAG services get registered (an eval run needs no
setup module — it talks to the server over HTTP):

```bash
export KAVALAI_AGENT_SETUP_MODULE=myapp/agent_setup.py   # dotted name or .py path
```

```python
# myapp/agent_setup.py
from kavalai import register_rag_service
from myapp.tools import measure_pond   # @pythontool functions, imported for registration

register_rag_service("village", ...)
```

**Not optional for a non-trivial workflow**: a graph with a `rag_query` node
naming a registered service cannot even be *built* without it. This is the
usual cause of "it works in my script but not on the server".

## Bundled tools

Ordinary `@pythontool` functions; register them like any other. One of them
reads the environment (`http_request`, for its proxy) — the exception to the
rule that library code does not.

| Tool | Import path | Needs |
|---|---|---|
| `crawl_url` | `kavalai.tools.webtools.crawl4ai` | a headless browser |
| `web_search` | `kavalai.tools.webtools.crawl4ai` | a headless browser |
| `http_request` | `kavalai.tools.webtools.http_client` | — |

```yaml
python_functions:
  - {name: web.search, path: kavalai.tools.webtools.crawl4ai.web_search}
  - {name: web.crawl,  path: kavalai.tools.webtools.crawl4ai.crawl_url}
```

- `crawl_url(url, include_html=False, bypass_cache=False, timeout=60.0)` returns
  clean Markdown — what you want to feed a model, not raw HTML. Failures come
  back as `success=False` with an `error_message` rather than raising, so an
  agent can read the error and try something else.
- `web_search(query, count=10, timeout=60.0)` needs **no API key** (it scrapes
  the DuckDuckGo HTML endpoint through the same browser).
- Both drive a real browser: seconds, not milliseconds. Budget timeouts
  accordingly, and prefer a keyed API for production search volume.
  `docker-compose.yml` includes a `crawl4ai` service to run the crawler as a
  container.
- `http_request(...)` is the escape hatch for an endpoint that does not deserve
  a full `rest://` registration; `use_proxy=True` routes through the Tor proxy
  at `KAVALAI_TOR_PROXY_HOST` / `KAVALAI_TOR_PROXY_PORT`.

**Security**: handing an agent a general HTTP tool lets it call any URL it can
compose, internal addresses included. Prefer specific `rest://` tools, or pin
the node with `allowed_tools`, whenever the model chooses the target.

## Introspection

`get_input_model`, `get_output_model`, `get_tool_descriptions(allowed_tools)`
and `get_tool_definition` are how an agent learns what it may use and how a
`function` node resolves its URI. `get_tool_descriptions(None)` is every tool,
`[]` is none, `"*"` is all, `proto://server.*` is one server.
