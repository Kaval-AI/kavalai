---
name: kavalai-workflows
description: Author, validate and debug Kaval.AI workflow graphs — YAML files with `data_types` and `nodes`, or `WorkflowBuilder` in Python. Use when writing a workflow, adding an llm/agent/function/rag_query/if/switch/parallel node, wiring prompts and node inputs, or when a workflow fails to load or a run takes the wrong branch.
---

# Kaval.AI workflows

A workflow is a small typed graph: input in, output out, one node per step.
Values crossing a node boundary are Pydantic-validated, branching is evaluated
without `eval`, and **almost all validation happens when the graph loads, not
when it runs** — so the fastest way to check your work is to load it.

```python
from kavalai import WorkflowEngine

engine = WorkflowEngine.from_yaml_path("workflow.yaml")   # raises with the reason
state = await engine.run({"user_message": "I want a refund"})
print(state.output_data, state.trace, state.token_usage)
```

Run that after every edit. The exceptions name the node and the problem; see
`references/errors.md` for the message-to-fix table.

`from_yaml` (a string) and `from_dict` (a parsed dict) take the same keyword
arguments.

## Skeleton

```yaml
name: Support agent
description: Routes a support request and produces a tailored response.
llm_model: openai/gpt-5.4-mini

data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  classification:
    type: object
    properties:
      intent: {type: string, description: "refund, technical or other"}
  output:
    type: object
    properties:
      agent_response: {type: string}

nodes:
  - {name: begin, type: start, next: classify}

  - name: classify
    type: llm
    prompt: >
      Classify the user's intent as one of: refund, technical, other.
      Return it in the `intent` field.
    inputs:
      message: {type: context, value: input.user_message}
    output: classification
    next: route

  - name: route
    type: switch
    expr: classification.intent
    cases:
      refund: handle_refund
      technical: handle_technical
    default: handle_other

  # … one llm node per branch, each with `output: output` and `next: finish`

  - {name: finish, type: end, output: output}
```

Full key tables for every node type: `references/nodes.md`.

## data_types

JSON-schema fragments compiled into Pydantic models. Two names are special:
`input` is what the caller passes to `run()`, and the type named by the `end`
node's `output` is what the caller gets back.

**Write a `description` on every non-obvious field.** It is part of the schema
sent to the model, so `confidence: {type: number, description: "0.0-1.0"}`
measurably improves what comes back. This is a rule, not a nicety.

## The invariants (all checked at load)

- Exactly **one** `start` node. `start` is a derived property of the graph, not
  a top-level key you write.
- At least one `end` node, and **all `end` nodes must name the same output
  type**.
- Node names are unique; every `next` / `then` / `else` / case target must
  exist.
- **Every node's `output` must be declared in `data_types`** — except
  `rag_query`, whose output shape is Kaval.AI's, not yours.
- `llm_model` is `provider/model`; a provider containing `.` is rejected (a
  workflow names a registration, never a Python path).
- `rag_service` is a registered **name**; a value containing `://` or `@` is
  rejected (never a connection string).
- A `rag_query` node naming a service that was neither passed to the engine nor
  registered fails when the engine is **constructed**, not on the first request
  that reaches the node.

## Prompt interpolation is not Jinja2

Only three prefixes are recognised, and there are no loops, filters or
conditionals:

```yaml
prompt: |
  {{ templates.house_style }}

  The customer wrote: {{ context.input.user_message }}
  Their last order was {{ history.last_order_id }}.
```

- `{{ context.PATH }}` — the current run context
- `{{ templates.NAME }}` — a fragment from the top-level `templates` list
- `{{ history.PATH }}` — a value from an **earlier run in the same session**
  (needs an `AgentService` and a session)

Dicts and lists are inserted as JSON. **An unresolvable reference raises**
rather than rendering empty, so a typo fails loudly instead of silently
weakening the prompt. Do not write `{% if %}`, filters, or `{{ x | default }}`
— they are not supported.

(The `Agent`'s own system-prompt template *is* Jinja2. That is a different
template, and the distinction does not generalise to node prompts.)

## Node inputs

```yaml
inputs:
  message:  {type: context, value: input.user_message}
  tone:     {type: literal, value: "formal"}
  previous: {type: history, value: last_order_id}
```

`context` is a dotted path into the run context; `literal` is the value as
written; `history` is a value recorded in an earlier run of the session. In
`WorkflowBuilder`, a bare string is shorthand for a context path:
`inputs={"message": "input"}`.

## Branching

`if` evaluates a boolean condition; **`then` and `else` are both required, and
`else` is a plain YAML key** (not a Python keyword — write it as-is).

```yaml
- name: check_confidence
  type: if
  condition: "classification.confidence >= 0.8"
  then: reply
  else: escalate
```

`switch` stringifies `expr` and matches it against `cases`; with no match and
no `default` the run fails with a `WorkflowException` (`Workflow halted at node
'x' with no next node and without reaching an end node.`).

Expressions are evaluated through an **AST whitelist, never Python `eval`**:
comparisons, `and` / `or` / `not`, `in`, arithmetic, and dotted/indexed access
into the context. Unknown names resolve to `None`, so guards degrade
gracefully. Do not reach for function calls, comprehensions or method access —
they are not in the language.

## parallel

```yaml
- name: gather
  type: parallel
  branches: [fetch_weather, fetch_news, fetch_stocks]
  next: summarise
  max_concurrency: 4
```

Each name in `branches` is the **entry node of a branch** — an ordinary
subgraph walked up to but not including the join. All branches start together
and the run resumes at `next` once every branch has arrived there.

Rejected at load time: overlapping branch subgraphs, two branches writing the
same `output` variable, an `end` node inside a branch, a branch re-entering
the parallel node, or a branch reaching the `start` node. Each branch gets its own copy of the run context (a node
cannot observe a sibling's output mid-flight); outputs merge at the join.

Three properties worth designing around: branch events interleave on the single
stream, each tagged with its node name; `state.trace` collects branches in
declaration order, so it is stable across runs even though execution is not;
and the first branch to raise cancels its siblings.

Tool calls *within* one `agent` node already run concurrently — `parallel` is
about concurrency between nodes.

## Loops and the backstop

Cycles in a graph are allowed. What makes them safe is `max_node_visits`, an
**engine constructor argument** (default 1000), not a YAML key:

```python
engine = WorkflowEngine.from_yaml_path("workflow.yaml", max_node_visits=50)
```

## Engine lifecycle — one engine, many runs

**Do not build an engine per request.** One engine serves many concurrent runs.
Per-run state (the token accumulator) lives on `RunContext`; engine-level state
(the `FunctionKernel` and its MCP sessions) is opened once and released once:

```python
engine = WorkflowEngine.from_yaml_path("workflow.yaml", agent_service=service)
await engine.connect()      # start MCP servers, discover tools — at startup
...                          # many concurrent engine.run(...) calls
await engine.aclose()       # at shutdown
```

`WorkflowEngine` also works as an async context manager. In FastAPI, do this in
the lifespan hook.

## Streaming

`run_stream()` yielding `WorkflowStreamEvent`s is the single execution path;
`run()` just drains it. Lifecycle events always arrive. Token-by-token content
is **opt-in per node**: `stream_output`, plus `stream_delta` to send only new
text (prefer it for long outputs), plus agent-only `stream_instructions` and
`stream_partials` (a debug firehose). Settable from YAML and from
`WorkflowBuilder`.

## The Python builder

Same graph, same validation, useful when the shape is computed:

```python
from kavalai.workflow import WorkflowBuilder

workflow = (
    WorkflowBuilder("Village greeter", llm_model="openai/gpt-5.4-mini")
    .data_model("input", Message)        # an existing Pydantic model
    .data_model("output", Reply)
    .start("reply")
    .llm("reply", prompt="…", inputs={"message": "input"},
         output="output", next="end")
    .end()
)
engine = workflow.build_engine(agent_service=service)
```

`data_type` takes a JSON-schema fragment, `data_model` takes a Pydantic class.
`build()` returns the `WorkflowGraph`; `build_engine(**kwargs)` returns a ready
engine. Prefer YAML for a graph a human will read and the backoffice will
render; prefer the builder when the graph is generated.

## Checklist before you hand a workflow over

1. `WorkflowEngine.from_yaml_path(...)` loads without raising.
2. Every `data_types` field that is not self-evident has a `description`.
3. Every `agent` node has an `allowed_tools` list and a deliberate `max_steps`.
4. Prompts use only the three interpolation prefixes.
5. Nothing secret is in the file — use `url_env` / `command_env` /
   `username_env` / `password_env` (see `kavalai-tools`).
6. A run produced the expected `state.trace`, not just the expected output.
