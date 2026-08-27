# Load-time errors and what they mean

Kaval.AI validates when the graph is built. The messages are precise — read the
message, make one edit, reload. Do not guess-and-retry.

| Message (abridged) | Cause | Fix |
|---|---|---|
| `found {n}` start nodes | Zero or several `type: start` | Exactly one `start` node. `start` is derived, never a top-level key. |
| `All 'end' nodes must return the same output data type, found …` | Two `end` nodes with different `output` | Give every `end` the same output type, or converge earlier. |
| `Duplicate node names: …` | Two nodes share a `name` | Rename one. |
| `Node 'x' transitions to unknown node 'y'` | `next`/`then`/`else`/a case target does not exist | Fix the target or add the node. |
| `Node 'x' output 'y' is not declared in data_types` | Node writes to an undeclared type | Declare `y` in `data_types`. (`rag_query` is exempt — its shape is ours.) |
| `Model must be in 'provider/model' form, got 'x'` | Bare model name | `openai/gpt-5.4-mini`, not `gpt-5.4-mini`. |
| `… names provider 'x'. A workflow names a …` | Provider contains `.` — a Python path was given | Register the provider and name the registration. |
| `Node 'x' service 'y' looks like a connection string` | `service:` contains `://` | Name a registered RAG service, never a URI. |
| `Node 'x' needs RAG service 'y', which is neither passed to the engine … nor …` | Service not passed nor registered | Pass `rag_services=` or call `register_rag_service`, usually from the setup module. |
| `No LLM model configured (set node.llm_model, graph.llm_model …)` | No model anywhere | Set `llm_model` on the node, on the graph, or `KAVALAI_DEFAULT_LLM_MODEL`. |
| `End node 'x' was reached inside a parallel branch; only the main path may end a run.` | `end` inside a branch | Rejoin at the parallel node's `next` and end after the join. |
| `… into the parallel node itself. A branch must rejoin at …` | Branch re-enters the parallel node | Point the branch's last node at the join node. |
| `… join node ('next'), so the branch is empty.` | Branch entry is the join itself | Give the branch at least one node. |
| `Exceeded max node visits (n)` | A cycle that does not terminate | Fix the exit condition; raise `max_node_visits` on the engine only if the loop is genuinely long. |
| `Invalid tool URI format: 'x'. Expected protocol://[name\|module].function_name` | Malformed `tool:` | Use `python://name`, `rest://server.tool` or `mcp://server.tool`. |
| `Function 'f' must be decorated with @kavalai.pythontool` | Registered a plain function | Add the decorator. |
| `MCP server 'x': Either stdio (command/command_env) or HTTP (url/url_env) must be specified.` | Neither given | Give exactly one transport. |
| `MCP server 'x': Cannot specify both stdio … and HTTP … configurations.` | Both given | Remove one. |
| `MCP server 'x' is already registered.` / `… not registered.` | Name collision, or a URI naming an unregistered server | Names are unique and must exist. |
| `Install it with: pip install kavalai[common]` | Optional SDK missing | Install the extra, not the bare SDK. |

## Runtime errors

- An unresolvable `{{ context.… }}` / `{{ history.… }}` reference **raises**;
  it never renders empty. A failing prompt usually means a typo in a path or a
  node that has not run yet on this path.
- A `switch` with no matching case and no `default` raises `WorkflowException`.
- A tool return value that cannot satisfy the tool's output model raises
  `FunctionKernelException` — it is never silently handed back raw.
- Expression names that do not exist resolve to `None` rather than raising, so
  a guard degrades gracefully. If a branch is silently never taken, suspect a
  misspelled context path.
