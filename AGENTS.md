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
| `kavalai/eval/` | Evaluation against a **running** agent server: `base.py` (`AgentEvaluator`, `EvalResult`), `simple_evaluator.py` (literal matchers), `judge_evaluator.py` (a model grades a plain-language criterion), `eval_runner.py` (YAML cases + the `kavalai-eval` console script) |
| `kavalai/rag/` | `BaseRagService` (three capability tiers), `PostgresRagService` (pgvector), `SqliteRagService` (portable file index) |
| `kavalai/tools/` | Bundled tools: browser, web search, HTTP |
| `kavalai/migrations/` | Alembic sets: `agents` and `backoffice` |
| `backoffice/`, `frontend/` | Management API and Angular UI |
| `tests/` | Pytest suite for the library; mock MCP servers in `tests/helpers/`. An example's tests live beside it under `examples/`, and `testpaths` covers both |
| `docs/`, `notebooks/` | Sphinx documentation; the five tutorial notebooks are the source of truth |
| `examples/` | Runnable examples; `green_village/` (RAG chatbot, port 25000), `bakery/` (a YAML workflow with side effects, ports 25100/25001) and `business_info_agent/` (web research: search, crawl, summarise, port 25200) each ship an `eval_cases.yaml`; `support_agent/support_agent.yaml` is the branching-DAG example the docs load |

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
   functions do, through `kavalai/settings.py` where two of them read the
   same names; `tests/test_config_drift.py` pins the list of modules allowed
   to touch `os.environ`. Everything else takes its configuration as an
   argument — the engine's `default_llm_model` / `default_llm_parameters`,
   `set_default_normalizer()`. The backend registries are the one bounded
   exception, and arguments still win:
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

# Only the tests that talk to real providers (deselected by default)
.venv/bin/dotenv run -- .venv/bin/python -m pytest -m integration
```

Tests marked `integration` hit a real provider or the internet; `addopts` in
`pyproject.toml` deselects them, and CI runs them in a separate step. There is
one smoke test per provider, not one per model: the runtime keeps no model
catalogue and never drops a parameter — an unsupported one fails the call with
the provider's own error.

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
- Comments: no `#:` attribute comments and no dash-rule section dividers.
  Explain a field in the class docstring and a constant in the module or
  function docstring. A comment answers *why*, not *what*; if the name already
  says it, or the only content is implementation history, omit it. Say a thing
  once, where the reader looks first.
- Do not edit `README.md` unless asked; it deep-links to documentation anchors,
  so renaming a heading in the tutorials breaks it.

## README style

`README.md` is the shop window, not the documentation, and is edited only
when asked. When it is, keep to the pattern it already follows:

- Each example is one short lead sentence saying what the snippet does
  ("Index the data", "Query the dataset"), then the code, then the real
  output under `Response:`. No commentary after the output — the code and the
  result make the point.
- No parameter explanations, internals or design rationale (`store="content"`,
  "not Jinja2", the missing cost column). They belong in the docs; the README
  says what a thing does, not how.
- One link per section, at the end, in the form "See [X](…) for more" — not a
  list of sub-topic anchors.
- The intro is one sentence and a bulleted feature list, each bullet a bold
  phrase, an em dash, and one clause. Install, backoffice and licence are a
  line each.
- Every output is genuine: run the snippet against the named model and paste
  what it printed, wrapped to 80 columns so it reads on a phone. Examples
  build on each other (the RAG index feeds the workflow), so a later section
  may reuse a name defined earlier without redefining it.
- It deep-links to documentation anchors, so renaming a tutorial heading
  breaks it.

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
- **Writing style.** The documentation is written in an academic, clean
  register, and new pages and edits must match it:
  - Declarative sentences in the present tense; plain, precise vocabulary.
    Prefer "the engine records every run" to "the engine will go ahead and
    record every run for you".
  - No marketing or enthusiasm — no "powerful", "seamless", "simply",
    "easily", exclamation marks or rhetorical questions. State what a thing
    does and why it is designed so; let the reader judge.
  - Impersonal or third-person phrasing by default. "You" is acceptable for
    instructions the reader performs; "we" and "let's" are not.
  - Every claim about behaviour is one the code exhibits — check the source
    before describing a control, a default or an error message, and name it
    exactly as it appears.
  - Explain the *reason* behind a design choice where one exists (the missing
    cost column, the refused unregistered user), in one or two sentences.
  - British spelling (behaviour, normalise, artefact), a spaced em dash
    (` — `) for asides, and `**bold**` only for the first mention of a UI
    element or a defined term.
  - Screenshots: describe what the image *shows the reader*, not that an image
    exists; give each one a full `:alt:` sentence. Crop captures to the
    application viewport — never ship a desktop or IDE capture.
- Keep code blocks within roughly 80 characters so the rendered page needs no
  horizontal scrollbar.

## Evaluation & acceptance testing

`kavalai/eval/` evaluates an agent server that is **already running**. It holds
two evaluators and one runner, and knows nothing about the engine, the database
or the workflow file — it speaks HTTP to a server and discovers that server's
input and output types from its OpenAPI spec (`kavalai/client.py`).

- `simple_evaluator.py` — send one input, compare the answer with expected
  values. No model is involved, so the verdict is the same every time. Its
  matcher vocabulary is five names: `equals`, `contains`, `not_contains`,
  `regex`, `one_of`.
- `judge_evaluator.py` — send one input, then ask a judging model whether the
  answer meets a plain-language criterion. A failing case carries the judge's
  reason. The judge is built on first use, so a run of literal cases needs no
  API key.
- `eval_runner.py` — a YAML file of cases (`type: simple` or `type: judge`),
  run in order against one server. `kavalai-eval <cases.yaml> --host … --port …
  --tag … --auth user:password`.

```bash
uv run --env-file .env kavalai-eval \
    examples/green_village/eval_cases.yaml --port 25000 --tag gpt-5.4-mini
```

Exit `0` every case passed, `1` a case failed, `2` the run never reached a
verdict — CI needs the third to tell "the suite is broken" from "the agent is
wrong".

Rules that are load-bearing:

- **`kavalai.eval` reads no environment variables**; only
  `eval_runner.py:main()` does. A base URL, an auth pair and the judge model
  are passed in.
- Both evaluators are usable straight from a unit test: construct one, `await
  evaluator.evaluate(inputs, expected)`, assert on `result.passed` and
  `result.reason`.
- **A judged case must state a criterion.** `EvalCase` rejects one that does
  not, because judging against nothing passes on any answer at all.
- A failing agent call fails its case with a reason instead of raising, so one
  broken case cannot end a run — but a malformed *suite* refuses to start.
- **A case file never names the server it grades.** No `base_url` key,
  `--port` required, no default base URL on the evaluators: the agent under
  evaluation is a property of the run, not of the cases.
- Each case runs in its own session, recorded under
  `external_id = "eval:{tag}:{case}"` — `eval:{case}` when no `--tag` is
  given. The tag names the run (model version, prompt variant, build) and is
  what makes two runs comparable afterwards; `eval:` is reserved, and the
  backoffice sessions page filters on it. Nothing is written when the agent
  server has no `AgentService`.

The three worked examples:

- `examples/green_village/eval_cases.yaml` — 64 cases against a RAG chatbot;
  literal ones for the facts, judged ones for the comparisons and refusals.
- `examples/business_info_agent/eval_cases.yaml` — 10 cases against an agent
  that reads the live web, so they assert only what a company keeps stating
  about itself, plus the two cases that grade a refusal: a business that does
  not exist, and a near-match that does.
- `examples/bakery/eval_cases.yaml` — 26 cases against a workflow that *writes*
  orders. It can be mostly literal because the model only writes `subject` and
  `body`: `compose_reply` stamps `status`, `order_id` and `missing` onto the
  answer from what deterministic Python decided, so the outcome of a run is a
  value to compare rather than prose to interpret.

## Agent skills

`kavalai/.agents/skills/` holds the six skills shipped to clients so their
coding agents write Kaval.AI code correctly: `kavalai` (entry + router),
`kavalai-workflows`, `kavalai-tools`, `kavalai-serving`, `kavalai-rag`,
`kavalai-eval`. Long tables live in a skill's `references/`, read only when
needed.

- They ship inside the wheel (`[tool.setuptools.package-data]`), and
  `kavalai-skills install` (`kavalai/skills.py`) copies them into a client's
  `.claude/skills/`. An already-installed skill is kept unless `--force`, so a
  client's edits survive an upgrade.
- A skill carries only what a model gets *wrong* without it: the unguessable
  schemas, the places the design deliberately differs from the obvious
  (interpolation that is not Jinja2, `rag_service` a name not a URI, a failed
  SSE run returning 200, no cost column), and the load-time invariants.
- `tests/test_skills.py` is the drift guard: node types, node input types, eval
  matchers, `EvalCase`/`EvalSuite` keys and every environment variable a skill
  names are checked against the code. Rename any of them and update the skill
  in the same commit — a stale skill misleads a client's agent silently.
- User-facing docs: `docs/reference/skills.rst`.

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
| New skill, or a rename a skill documents | `kavalai/.agents/skills/`, then `tests/test_skills.py` and `docs/reference/skills.rst` |
| New eval matcher | `kavalai/eval/simple_evaluator.py`, added to `MATCHERS` and to `check_field`, with a passing *and* a failing test in `tests/eval/test_simple_evaluator.py` — a matcher never shown to fail manufactures confidence |
| New eval case type | `kavalai/eval/eval_runner.py` (`EvalCase.type`) and a matching evaluator class; the validator has to reject an expectation the new evaluator cannot read |

## Further reading

- [`docs/tutorials/architecture.rst`](docs/tutorials/architecture.rst) — the
  design and its rationale.
- [`docs/guides/data_model.rst`](docs/guides/data_model.rst) — the tables and
  what each is for.
- [`docs/reference/yaml.rst`](docs/reference/yaml.rst) — every workflow key.
- [`docs/reference/providers.rst`](docs/reference/providers.rst) — the LLM,
  embedding and RAG providers that ship with Kaval.AI, and how to find out
  which models each one offers.
- <https://docs.kaval.ai/llms.txt> — a machine-readable index of the published
  documentation.
