# Node reference

Every node has a unique `name` and a `type`. The remaining keys depend on the
type. This file is the exhaustive list; the skill body covers the rules.

## Top level

| Key | Required | Description |
|---|---|---|
| `name` | yes | Workflow name. Runs are recorded under this name as the agent. |
| `description` | no | Shown in the backoffice. |
| `version` | no | Schema version. Defaults to `"2.0"`. |
| `llm_model` | no | Default `provider/model` for every `llm` and `agent` node. Falls back to the engine's `default_llm_model` (`KAVALAI_DEFAULT_LLM_MODEL` under `python -m kavalai.server`). |
| `llm_kwargs` | no | Defaults merged into `LlmClientParameters` (`temperature`, `top_p`, `timeout_seconds`, …). Nodes may override individual keys. |
| `rag_service` | no | Default RAG service **name** for `rag_query` nodes. Falls back to `"default"`. |
| `rag_collection` | no | Default collection for `rag_query` nodes. |
| `data_types` | yes | JSON-schema fragments compiled into Pydantic models. |
| `nodes` | yes | The graph. Exactly one `start`, at least one `end`. |
| `rest_servers` | no | REST tool servers registered before the run. |
| `mcp_servers` | no | MCP tool servers registered before the run. |
| `python_functions` | no | Python tools imported and registered by path. |
| `templates` | no | Named, reusable prompt fragments. |

## start

| Key | Description |
|---|---|
| `next` | First node to run. Required. |

```yaml
- {name: begin, type: start, next: classify}
```

## end

| Key | Description |
|---|---|
| `output` | Context variable returned to the caller. Defaults to `output`. |

Several `end` nodes are allowed; they must all name the same output type.

## llm

One structured completion. The prompt is rendered, the model called, the
validated result stored under `output`.

| Key | Description |
|---|---|
| `prompt` | Instruction text, with interpolation. Required. |
| `inputs` | Local name → argument, resolved before the call. |
| `output` | Data type / context variable the result is written to. Required. |
| `next` | Node to run afterwards. Required. |
| `use_history` | Replay the session's chat history into the call. **Default `true`** — this is what gives a chatbot memory across turns. |
| `llm_model` | Overrides the workflow default for this node. |
| `llm_kwargs` | Per-node sampling/reliability overrides. |
| `stream_output` | Emit this node's completion as `partial` events. Default `false`. |
| `stream_delta` | Send only new text per `partial`. Prefer for long outputs. Default `false`. |

## agent

A multi-step, tool-using `Agent` loop inside the graph. Use it when the model
should decide which tools to call; use `function` when you already know.

Takes every `llm` key **except `use_history`**, plus:

| Key | Description |
|---|---|
| `max_steps` | Maximum reasoning/tool-calling iterations. Default `10`. The bound that stops a runaway loop. |
| `allowed_tools` | Tool URIs this node may use. Tools outside the list are neither described to the model nor callable. |
| `stream_instructions` | Stream each step's "thinking out loud" line as `<node>_instructions`. |
| `stream_partials` | Stream raw per-step output as `<node>_step<N>`. A debug firehose. |

`allowed_tools` semantics — identical in YAML and in `Agent(allowed_tools=…)`:

```yaml
allowed_tools: ["*"]                         # every tool
allowed_tools: ["mcp://github.*"]            # one whole server
allowed_tools: ["python://lookup_resident"]  # one tool
allowed_tools: []                            # none
# key omitted                                # same as ["*"]
```

## function

Exactly one tool call through the `FunctionKernel`, addressed by URI.

| Key | Description |
|---|---|
| `tool` | `python://name`, `rest://server.tool` or `mcp://server.tool`. Required. |
| `inputs` | Arguments for the call, resolved like any node inputs. |
| `output` | Where the validated return value is stored. Required. |
| `next` | Node to run afterwards. Required. |
| `method` | HTTP method for `rest://` tools. Default `get`. |

## rag_query

One retrieval. **Read-only** — it reaches `BaseRagService.query` and nothing
else, so no workflow document can write to an index. `query` is a template,
rendered exactly like an `llm` prompt.

| Key | Description |
|---|---|
| `query` | Query text, rendered as a template. Required. |
| `output` | Where hits are stored. Required. **Need not appear in `data_types`.** |
| `next` | Node to run afterwards. Required. |
| `service` | Registered service name. Defaults to the graph's `rag_service`, then `"default"`. Never a connection string. |
| `collection` | Defaults to the graph's `rag_collection`. |
| `top_k` | Maximum hits. Default `5`. |
| `source_ids` | Restrict to these source identifiers. |
| `keep_best` | Keep only the best hit per `source_id`, for documents indexed as many chunks. Default `false`. |
| `store` | `results` (default) keeps the full hit list with scores and metadata, so `if`/`switch` can read them; `content` stores just the hit texts joined by blank lines, which is what a following prompt usually wants. |

## if

| Key | Description |
|---|---|
| `condition` | Boolean expression over the run context. Required. |
| `then` | Node when true. Required. |
| `else` | Node when false. Required — a plain YAML key. |

## switch

| Key | Description |
|---|---|
| `expr` | Expression; its result is converted to a string. Required. |
| `cases` | Mapping of string value → node name. |
| `default` | Node when nothing matches. Without it, no match fails the run. |

## parallel

| Key | Description |
|---|---|
| `branches` | Entry node of each branch. At least one, no duplicates. |
| `next` | The join node. Every branch ends by transitioning to it. |
| `max_concurrency` | Cap on branches running at once. Omit to run them all; set it when branches share a rate-limited provider. |

## Node inputs

| `type` | Meaning |
|---|---|
| `context` | Dotted path into the run context: `input`, `input.user_message`, `classification.intent`. |
| `literal` | The value exactly as written. |
| `history` | A value recorded in an **earlier run** of the same session. Requires an `AgentService` and a session. |

## Tool servers

```yaml
python_functions:
  - {name: measure_pond, path: green_village.tools.measure_pond}

rest_servers:
  - name: crm
    url: https://crm.example.com/api
    username_env: CRM_USER
    password_env: CRM_PASSWORD

mcp_servers:
  - name: village
    command: python
    args: ["-m", "village_mcp"]
```

| Key | Description |
|---|---|
| `python_functions[].name` / `.path` | Registered name, and the import path of a `@pythontool` function (`package.module.function`). |
| `rest_servers[].url` / `.url_env` | Base URL, directly or from an environment variable. |
| `rest_servers[].username_env` / `.password_env` | Environment variables holding basic-auth credentials. |
| `mcp_servers[].command` / `.command_env` / `.args` / `.env` | Command to launch a stdio MCP server, its arguments and extra environment. |
| `mcp_servers[].url` / `.url_env` | For an HTTP/SSE MCP server instead of a subprocess. |

Use the `*_env` variants for anything secret. **`mcp_servers[].env` is the
exception** — it takes literal values, so a key written there sits in the
committed file. Prefer `command_env` and a wrapper script. See `kavalai-tools`.

REST *tools* are declared in code with `register_rest_tool` (they need input and
output schemas); the YAML declares the *server* only.
