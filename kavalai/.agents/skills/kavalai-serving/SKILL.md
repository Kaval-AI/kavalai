---
name: kavalai-serving
description: Serve, deploy, persist and monitor a Kaval.AI workflow — the agent server and its endpoints, SSE streaming, `AgentClient`, mounting in an existing FastAPI app, environment variables, Alembic migrations, sessions and run statistics. Use when deploying a workflow, calling an agent over HTTP, consuming the event stream, or when runs are missing from the backoffice.
---

# Serving and deploying a Kaval.AI workflow

The agent server turns any `WorkflowEngine` into a FastAPI service whose request
and response schemas **are** the workflow's own `input` and `output` data types
— the API is generated from the graph, not maintained beside it.

Every environment variable: `references/config.md`.

## Start one

```bash
export KAVALAI_AGENT_WORKFLOW_PATH=support_agent.yaml
export KAVALAI_AGENT_SETUP_MODULE=myapp/agent_setup.py   # registers python:// tools, RAG services
export KAVALAI_DB_URI=postgresql://user:pass@localhost:5432/kavalai
export KAVALAI_DB_SCHEMA=agents
export OPENAI_API_KEY=sk-...

python -m kavalai.server
```

**Migrate the database first** (see below) — the server does not create tables.

| Endpoint | What it does |
|---|---|
| `POST /run_agent` | Runs the workflow, returns the final output in one response |
| `POST /stream_agent` | Runs it and streams progress as Server-Sent Events |
| `GET /workflow` | The workflow graph (the backoffice renders it from this) |
| `GET /liveness` | Is the process up? |
| `GET /health` | Is the process up **and** the database reachable? |

Use `/liveness` for a liveness probe and `/health` for readiness. Pointing both
at `/health` makes a database blip restart healthy pods.

## The request envelope

```bash
curl -s -X POST localhost:10000/run_agent \
    -H 'Content-Type: application/json' \
    -d '{"data": {"user_message": "The bell is stuck."},
         "external_id": "villager-42"}'
```

The workflow input goes in `data`. Beside it: `session_id` continues an
existing conversation, `external_id` keys a session by an identifier from your
own system (a user, ticket or thread id). The response mirrors the request —
your `output` type under `data`, plus the `session_id` the run belongs to. Send
that id back, or reuse the same `external_id`, and the next call continues the
conversation, with history replayed into every `llm` node that has
`use_history` on (the default).

Because the schemas come from the graph, `/docs` describes your actual types and
a malformed request is rejected before the model is ever called.

## Streaming — three properties of the endpoint

```bash
curl -N -X POST localhost:10000/stream_agent -H 'Content-Type: application/json' \
     -d '{"data": {"user_message": "…"}}'
```

1. **A failed run still returns 200.** A response cannot change its status code
   after the headers are sent, so a failure ends the stream with a
   `workflow_failed` event. **Clients must treat the event, not the status
   code, as the failure signal.**
2. **Disconnecting aborts the run.** Closing the stream cancels the engine
   generator, and the abort is recorded on the run row.
3. **Browsers cannot use `EventSource`** — it supports neither a request body
   nor auth headers, and this is a `POST`. Use `fetch()` with a streaming
   reader.

A `: ping` comment frame is sent during silent stretches (a long tool call) so
proxies do not drop the connection.

Lifecycle events (`workflow_started`, `node_started`, `node_completed`,
`workflow_completed` / `workflow_failed`) always arrive. Token-by-token content
only comes from nodes that opted in with `stream_output` — see
`kavalai-workflows`.

## Calling it from Python

```python
from kavalai.client import AgentClient

client = AgentClient("http://localhost:10000")     # username=, password= for basic auth
await client.discover_schemas()                     # rebuilds the models from the OpenAPI spec
print(list(client.input_schema.model_fields))       # ['user_message']

reply = await client.run_agent(
    client.input_schema(user_message="Is the pub open on Sundays?"),
    external_id="villager-7",
)
print(reply.agent_response)

async for chunk in client.stream_agent(client.input_schema(user_message="…")):
    print(chunk)                                    # one event payload per chunk, as JSON
```

Two things are easy to miss: `run_agent` / `stream_agent` take an **instance of
the input model**, not a dict — build it from `client.input_schema`. And the
client stores the `session_id` from each response and sends it with the next
call, so **one `AgentClient` is one continuous conversation**. Use a fresh
client, or an `external_id` per user, when you need separate ones.

## Mounting in your own app

`create_agent_router` returns a plain `APIRouter`, so workflows can live inside
an existing service, several side by side, with your own auth:

```python
from fastapi import FastAPI
from kavalai import WorkflowEngine
from kavalai.agent_service import AgentService
from kavalai.db import db_manager
from kavalai.server import create_agent_router

app = FastAPI()
session_maker = db_manager.get_sessionmaker(uri="postgresql://…/kavalai", schema="agents")
service = AgentService(session_maker)

support = WorkflowEngine.from_yaml_path("support_agent.yaml", agent_service=service)
app.include_router(create_agent_router(support, session_maker), prefix="/agents/support")

triage = WorkflowEngine.from_yaml_path("triage.yaml", agent_service=service)
app.include_router(
    create_agent_router(triage, session_maker, auth_dependency=lambda: None),
    prefix="/agents/triage",
)
```

Build engines **once** at startup and `await engine.connect()` / `aclose()` in
the lifespan hook — one engine serves many concurrent runs. `create_agent_app`
does the same for a standalone app; `create_app_from_env_conf` is what
`python -m kavalai.server` calls.

## Authentication

Basic auth is enabled by setting **both** `KAVALAI_AGENT_BASIC_AUTH_USER` and
`KAVALAI_AGENT_BASIC_AUTH_PASSWORD`. With neither set the endpoints are open —
fine behind an internal gateway, not fine on the public internet. For OAuth, an
API-key header or per-tenant rules, build the router yourself and pass your own
dependency.

## Configuration is read at entry points only

**Library code never reads environment variables** — only `main()`s (the
server, `kavalai.migrate_db`, the backoffice) and the client constructors'
key fallback do. Anything you build passes values explicitly, which is what
keeps it testable. Do not add `os.getenv` inside library-shaped code; take a
parameter.

In development keep them in `.env` and `dotenv.load_dotenv()`. A `.env` holds
credentials: keep it out of source control, and prefer your platform's secret
store in production. The workflow YAML has `url_env` / `command_env` /
`username_env` / `password_env` for exactly this reason.

## Migrations

Alembic, in two independent sets. **Run them on every deploy, before the
service that uses them** — both are idempotent.

```bash
python -m kavalai.migrate_db agents       # runtime tables: KAVALAI_DB_URI, KAVALAI_DB_SCHEMA
python -m kavalai.migrate_db backoffice   # backoffice tables: KAVALAI_BO_DB_URI, KAVALAI_BO_DB_SCHEMA
```

The two schemas are independent and may share a Postgres instance (`agents` and
`backoffice` by convention) or live apart.

If you extend the schema yourself, three rules matter:

- The ORM models are the single source of truth; autogenerate revisions from
  them against an empty scratch database.
- **Models are schema-less.** The schema is applied per engine via
  `schema_translate_map` (`db_manager.get_sessionmaker(..., schema=...)`).
- **Raw SQL and reflection bypass `schema_translate_map`.** Qualify the schema
  explicitly, and pass it to `op.batch_alter_table` — which reflects before
  altering, so without it the ALTER looks in `public`.

## What a run records, and why it might not

Two pieces, handed to the engine:

```python
engine = WorkflowEngine.from_yaml_path(path, agent_service=service, task_logger=logger)
```

- **`AgentService`** — agents, sessions, runs, chat history. In-memory SQLite
  locally (`AgentService(db_manager.get_sqlite_sessionmaker())`), Postgres in
  production; the same tables either way.
- **`TaskLogger`** — per-node task rows and model-call stats. `SqliteTaskLogger`
  locally, `PostgresTaskLogger` in production (drain it in the lifespan hook).
  This is what fills the backoffice task debugger, so a run can be stepped
  through node by node.

**Nothing is recorded without them.** "The backoffice is empty" almost always
means the server was built without an `AgentService`, or the task logger was
never wired, or the project points at a different database or schema.

A run's own `WorkflowState` carries `trace` (the exact path through the graph),
`token_usage`, and `run_id` / `session_id` / `invocation_id`. The 8-character
`invocation_id` prefixes every log line of the run, so one run's logs isolate
with a single search.

Attempts that never produced a response are recorded too, with the provider's
status code and error text in place of token counts — a rate-limit storm leaves
a trace instead of a suspiciously healthy table.

**`model_call_stats` records tokens only and has no cost column, deliberately.**
Providers return tokens, not money, and cached input is priced differently
enough that a derived total would be wrong rather than merely stale. Do not add
one; compute cost outside, from the token columns
(`cached_prompt_tokens` and `reasoning_tokens` are there).

## Docker

```bash
docker compose up postgres_db backoffice-migrations backoffice   # UI on :8000
```

`docker-compose.yml` defines `postgres_db` (pgvector), `backoffice-migrations`,
`backoffice`, and the optional `ollama` (11434), `crawl4ai` (11235) and
`torproxy` services. The agent server is not in it: build
`dockerfiles/agent.Dockerfile` and run the image with the entrypoint mode
`agent-migrations` (needs `KAVALAI_DB_URI`, `KAVALAI_DB_SCHEMA`) and then
`agent-server` (additionally `KAVALAI_AGENT_WORKFLOW_PATH`). The backoffice
image (`dockerfiles/backoffice.Dockerfile`) takes `backoffice-migrations` and
`backoffice-server` the same way, and needs `KAVALAI_BO_DB_URI` and
`KAVALAI_BO_DB_SCHEMA`.

The backoffice reaches an agent database through a **project** — a project row
carries the host, port, database and schema — so one UI can watch local,
staging and production separately.
