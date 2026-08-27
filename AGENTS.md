# AGENTS.md

Instructions for coding agents working in this repository. Humans are welcome
to read it too; it is the short form of what a new contributor needs.

Claude Code reads `CLAUDE.md`, which carries the same rules in the form that
tool expects. Keep the two in step: a convention worth adding to one belongs in
the other.

**Before making a structural change, read
[`docs/tutorials/architecture.rst`](docs/tutorials/architecture.rst).** It
states the design commitments, the invariants that follow from them, and where
each kind of change belongs. Most "where should this live?" questions are
answered there.

## What this project is

Kaval.AI is a Python library for building agentic workflows. A workflow is a
typed graph — written as YAML or built in Python — executed by an engine that
validates every boundary and records every run in a database the operator owns.

Two components:

- **`kavalai`** — the SDK and runtime. Modules live directly in the top-level
  package (`agent.py`, `db.py`, `server.py`, …).
- **`kavalai.backoffice`** + `frontend/` — a management interface that reads
  the runtime's tables.

## Layout

| Path | Contents |
|------|----------|
| `kavalai/` | Runtime: `agent.py`, `agent_service.py`, `db.py`, `server.py`, `run_context.py`, `functionkernel.py`, `schema_parser.py` |
| `kavalai/workflow/` | Engine v2: `models.py` (the graph), `engine.py`, `builder.py`, `expressions.py`, `render.py`, `tasklog/` |
| `kavalai/llm_clients/` | OpenAI, Gemini, Anthropic, Ollama and in-browser clients behind one streaming interface; `registry.py` makes that set extensible |
| `kavalai/eval/` | Evaluation & acceptance testing: `models.py` (Case/Dataset/Suite/results), `targets.py` (Engine/Rest/Callable + `RunRecord`), `trajectory.py`, `evaluators/` (deterministic, trajectory, judged, conversation), `runner.py`, `report.py`, `persona.py`, `fixtures.py`, `cli.py`. Console scripts `kavalai-eval` / `kavalai-persona` |
| `kavalai/rag/` | `BaseRagService` (three capability tiers), `PostgresRagService` (pgvector), `SqliteRagService` (portable file index) |
| `kavalai/tools/` | Bundled tools: browser, RSS, web search, HTTP |
| `kavalai/migrations/` | Alembic sets: `agents` and `backoffice` |
| `backoffice/`, `frontend/` | Management API and Angular UI |
| `tests/` | Pytest suite; mock MCP servers in `tests/helpers/` |
| `docs/`, `notebooks/` | Sphinx documentation; the five tutorial notebooks are the source of truth |
| `examples/` | Runnable examples |

## Invariants

Violating one of these produces a defect that is hard to see in review, so
check a change against the list before proposing it.

1. **Workflow shape changes start in `kavalai/workflow/models.py`.** The
   builder, the YAML loader, the engine, the SVG renderer and the backoffice
   all derive from those models. A capability added to the engine alone is one
   that YAML cannot express and the diagram cannot draw.
2. **`run_stream()` is the only execution path.** `run()` drains it. Do not add
   a second implementation for the non-streaming case.
3. **Per-run state belongs on `RunContext`; engine-level state belongs on the
   engine.** The token accumulator is per run and must be forwarded by
   `_branch_context`. The `FunctionKernel` and its MCP sessions are opened by
   `await engine.connect()` and released by `await engine.aclose()` — never per
   run.
4. **Library code reads no environment variables.** Only entry-point `main()`
   functions do. Everything else takes its configuration as an argument. The
   backend registries are the one bounded exception, and arguments still win:
   `client_factory` outranks the LLM registry, `rag_services=` outranks the RAG
   one.
   **A workflow document names a registration, never a Python path or a
   connection string.** `GET /workflow` serves it and the backoffice edits it,
   so a dotted path there would be arbitrary code execution in the agent
   server. Both are rejected when the graph loads.
5. **The ORM models are the single source of truth for the schema.** Change
   `kavalai/db.py`, then autogenerate the revision; parity tests in
   `tests/test_migrate_db.py` fail if the two diverge. Bump
   `SQLITE_SCHEMA_VERSION` on any schema change, or stale browser databases
   will not be rebuilt.
6. **The base package stays Pyodide-compatible.** No greenlet, no native
   extensions beyond the prebuilt Pyodide packages. Everything else goes in the
   `common` extra.
7. **Every boundary validates, and failures are loud.** An unresolvable prompt
   reference raises; a tool result that does not match its declared model
   raises; duplicate tool or server names raise at registration.
8. **Models are schema-less.** The target schema is applied per engine through
   `schema_translate_map`. Raw SQL and reflection bypass it and must qualify
   the schema explicitly.

## Working copy and git

- **Never use git worktrees.** Do not run `git worktree add`, and do not launch
  agents or workflows with worktree isolation. Worktree branches are invisible
  in the normal PyCharm window, which makes the diff impossible to review.
- Make edits directly in the checked-out branch of the repository.
- **Leave changes uncommitted** so they can be reviewed in PyCharm's Local
  Changes view. Do not commit, branch or push without asking.
- This holds for background jobs too; `.claude/settings.json` sets
  `"worktree": {"bgIsolation": "none"}` for that reason. Do not remove it.

## Running code and tests

Use the project `.venv` — it is the same interpreter PyCharm uses.

```bash
# Tests, with .env loaded (mirrors the IDE run configuration)
.venv/bin/dotenv run -- .venv/bin/python -m pytest

# Equivalently
uv run --env-file .env pytest

# One file
.venv/bin/python -m pytest tests/test_functionkernel.py
```

`conftest.py` does not auto-load `.env`, so integration tests gated on
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`
or the `KAVALAI_*` database settings will silently skip without it. Avoid
`set -a && source .env && set +a`: changing shell option state defeats static
command analysis and forces a permission prompt.

`.env.example` must list every variable the code reads and nothing else —
`tests/test_config_drift.py` checks both directions, so adding a `getenv`
without documenting it fails the suite.

Frontend:

```bash
cd frontend && npm test -- --watch=false
```

## Conventions

- `loguru` for logging, f-strings for formatting.
- Target **100% coverage** for new and modified code. Keep the tests for one
  source file in one test file (`agent.py` → `test_agent.py`).
- Run the tests at the end of every task, before reporting it done.
- Python tools are decorated with `@kavalai.pythontool` and registered through
  `register_python_tool`.
- Angular: modern control flow (`@if`, `@for`), never `*ngIf` / `*ngFor`.
  Prefer `common.css`, Tailwind and DaisyUI.
- Refactor blocks with distinct responsibilities into named functions.
- Do not edit `README.md` unless asked; it deep-links to documentation anchors,
  so renaming a heading in the tutorials breaks it.

## Documentation

- `sphinx-build -b html docs docs/_build/html` must finish with **zero
  warnings**.
- Five tutorials are notebooks under `notebooks/`, symlinked into
  `docs/tutorials/`. Edit the notebook, never the symlink, and **re-execute
  it** — `nb_execution_mode` is `"off"`, so the rendered page shows whatever
  outputs the file already holds. Examples must be run against live providers,
  not written by hand.
- Every code example on an `.rst` page should be executed before it is
  committed, with the real output pasted underneath.
- Keep code blocks within roughly 80 characters so the rendered page needs no
  horizontal scrollbar.
- Anything that needs a code change rather than a documentation fix goes in
  `docs/todo.rst`.

## Evaluation & acceptance testing

`kavalai/eval/` is a **library over files**. A suite is a directory holding a
dataset, personas, a `suite.yaml` and a **committed `baseline.json`**. There are
no eval tables and nothing is written to a database unless asked — deliberately,
so a suite runs on every pull request rather than when somebody remembers.

```bash
uv run --env-file .env kavalai-eval examples/bakery/eval/suite.yaml --tag pr-412
uv run --env-file .env kavalai-persona <persona.yaml> --suite <suite.yaml>
```

Exit `0` gate passed, `1` gate failed, `2` the run itself broke — CI needs the
third to tell "the harness broke" from "the workflow is wrong".

Rules that are load-bearing:

- **`kavalai.eval` reads no environment variables**; only `cli.py:main()` does.
  A database, a base URL or an auth pair is passed in explicitly.
- A suite's `setup:` imports a module before the run, registering the
  `python://` tools, named RAG services and custom evaluators the workflow and
  dataset name. **Not optional** for a non-trivial workflow: the engine resolves
  a named RAG service eagerly, at construction. `KAVALAI_AGENT_SETUP_MODULE`
  does the same for `kavalai/server.py`.
- Every evaluator ships with a known-good *and* a known-bad test.
  `tests/eval/test_evaluators.py` fails if any registered evaluator was never
  shown to fail — an evaluator that cannot fail manufactures confidence.
- **Never let an unrunnable assertion read as a pass.** A trajectory assertion
  against a target that cannot observe one raises, and a run that would drop
  them refuses to start unless `--skip-trajectory-evaluators` says otherwise.
  Skipped evaluators are always named in `result.notes`.
- `--fixtures` replays recorded model responses (no API key);
  `--record-fixtures` records them. Replay keys on the exact prompt, so
  **anything reaching a prompt must be deterministic** — see the bakery's
  sequential order ids and pinned clock.
- `--persist-sessions` tags each run
  `external_id = "eval:{suite}:{tag}:{case}:{repeat}"`. `eval:` is reserved; the
  backoffice sessions page filters on it.

## Where things usually belong

| Change | Start here |
|--------|-----------|
| New node type or node option | `kavalai/workflow/models.py`, then `engine.py`, `builder.py`, `render.py`, `docs/reference/yaml.rst` |
| New LLM provider | `kavalai/llm_clients/`, behind `BaseLlmClient`; add to the built-in table in `llm_clients/registry.py` as a **dotted string**, never an imported class |
| Third-party provider (not shipped) | Nothing in this repo: `register_llm_provider` / `register_embedding_provider` name it at runtime. Worked examples in `notebooks/llm_clients.ipynb` and `docs/cookbook/index.rst` |
| A provider you do not want in-tree | Nothing here — `register_llm_provider()` in your own package; see `docs/cookbook/index.rst` |
| New RAG backend | `kavalai/rag/`, behind `BaseRagService`, then add it to `tests/rag/test_conformance.py` |
| New bundled tool | `kavalai/tools/`, decorated with `@kavalai.pythontool` |
| New persisted field | `kavalai/db.py`, then an autogenerated Alembic revision, then `SQLITE_SCHEMA_VERSION` |
| New REST endpoint | `kavalai/server.py` (runtime) or `kavalai/backoffice/server.py` (management) |
| New documentation page | `docs/`, added to the toctree in the matching `index.rst` |
| New evaluator | `kavalai/eval/evaluators/`, registered with `@evaluator("name")`, plus a known-good and a known-bad case in `tests/eval/test_evaluators.py`. A *domain* evaluator belongs in the customer's own setup module, not here |
| New target kind | `kavalai/eval/targets.py`, behind the `Target` protocol; set `observes_trajectory` honestly |

## Further reading

- [`docs/tutorials/architecture.rst`](docs/tutorials/architecture.rst) — the
  design and its rationale.
- [`docs/guides/data_model.rst`](docs/guides/data_model.rst) — the tables and
  what each is for.
- [`docs/reference/yaml.rst`](docs/reference/yaml.rst) — every workflow key.
- [`docs/reference/providers.rst`](docs/reference/providers.rst) — the LLM,
  embedding and RAG providers that ship with Kaval.AI, and how to find out
  which models each one offers.
- [`docs/todo.rst`](docs/todo.rst) — open questions and known gaps.
- <https://docs.kaval.ai/llms.txt> — a machine-readable index of the published
  documentation.
