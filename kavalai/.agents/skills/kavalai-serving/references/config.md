# Environment variables

Read by entry points only — `python -m kavalai.server`,
`python -m kavalai.migrate_db`, the backoffice, and the client constructors
that fall back to a provider key. Library code reads none of them; anything you
build can pass values explicitly instead.

## Provider credentials

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | `OpenAIClient` and `openai/…` models |
| `GEMINI_API_KEY` | `GeminiClient` and `gemini/…` models |
| `ANTHROPIC_API_KEY` | `AnthropicClient` and `anthropic/…` models. Note the `_API_` — `ANTHROPIC_KEY` is **not** read |
| `OLLAMA_HOST` | Ollama endpoint. Default `http://localhost:11434`. No key needed |

Each client also accepts `api_key=` / `host=` directly, which wins over the
environment.

## Models

| Variable | Description |
|---|---|
| `KAVALAI_DEFAULT_LLM_MODEL` | Model when a workflow and its nodes both omit `llm_model`, as `provider/model` |
| `KAVALAI_DEFAULT_EMBEDDING_MODEL` | Embedding model `python -m kavalai.tools.index_csv` uses when `--model` is omitted, as `provider/model`. `make_embedding_client` and the RAG services take the model as an argument and do not read it |
| `FASTEMBED_THREADS` | Thread count for local `fastembed` embedding |
| `FASTEMBED_CACHE_DIR` | Where `fastembed` caches models. Worth setting in a container so the model is not re-downloaded on every start |
| `KAVALAI_EMBEDDING_NORMALIZER_YAML` | Path to a YAML file describing a custom `Normalizer` |
| `KAVALAI_PROVIDER_MODULES` | Comma-separated modules the agent server imports before loading the workflow, so backends they register can be named from YAML. Every dotted registration is resolved afterwards, so a mistyped path fails at start-up rather than at the first request reaching that node |
| `KAVALAI_RAG_SERVICE` | Registered RAG service for `python -m kavalai.tools.index_csv` to index into. Unset means the Postgres service built from `KAVALAI_DB_URI` |

## Agent database

Read by `python -m kavalai.server` and `python -m kavalai.migrate_db agents`.

| Variable | Description |
|---|---|
| `KAVALAI_DB_URI` | e.g. `postgresql://user:pass@host:5432/kavalai` |
| `KAVALAI_DB_SCHEMA` | Schema holding the runtime tables. Default `public`; `agents` by convention |
| `KAVALAI_DB_POOL_SIZE` | SQLAlchemy pool size. Default `0` |
| `KAVALAI_DB_MAX_OVERFLOW` | Pool overflow. Default `0` |
| `KAVALAI_SQL_ECHO` | Log every SQL statement. Default `false`. Useful once, noisy always |

## Agent server

| Variable | Description |
|---|---|
| `KAVALAI_AGENT_WORKFLOW_PATH` | Path to the workflow YAML to serve. **Required** |
| `KAVALAI_AGENT_SETUP_MODULE` | Module imported before the workflow loads — a dotted name or a `.py` path. Registers the `python://` tools and named RAG services the workflow refers to. A workflow with a `rag_query` node naming a registered service **cannot be built without it** |
| `KAVALAI_AGENT_HOST` | Bind address. Default `0.0.0.0` |
| `KAVALAI_AGENT_PORT` | Port. Default `10000` |
| `KAVALAI_AGENT_BASIC_AUTH_USER` | Basic-auth username. Auth is enabled only when **both** this and the password are set |
| `KAVALAI_AGENT_BASIC_AUTH_PASSWORD` | Basic-auth password |

## Backoffice

| Variable | Description |
|---|---|
| `KAVALAI_BO_DB_URI` | The backoffice's **own** database — separate from any agent database |
| `KAVALAI_BO_DB_SCHEMA` | Schema for the backoffice tables |
| `KAVALAI_BO_HOST` | Interface `python -m kavalai.backoffice.server` binds to. Default `127.0.0.1` |
| `KAVALAI_BO_PORT` | Port. Default `8000` |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth sign-in |
| `SESSION_SECRET_KEY` | Signing key for session cookies. **Set this in production** — the fallback is a well-known development value |
| `FRONTEND_URL` | Where the Angular frontend is served from; used for the post-login redirect and CORS |

## Tools

| Variable | Description |
|---|---|
| `RSS_AUTH_USER` / `RSS_AUTH_PASSWORD` | Basic auth protecting the bundled RSS tool's own HTTP endpoint. Default `admin` / `password` |
| `TOR_PROXY_HOST` / `TOR_PROXY_PORT` | Tor proxy used by `http_request(use_proxy=True)` |

## A worked `.env` for local development

```bash
# Provider
OPENAI_API_KEY=sk-...
KAVALAI_DEFAULT_LLM_MODEL=openai/gpt-5.4-mini

# Agent database (runtime tables)
KAVALAI_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
KAVALAI_DB_SCHEMA=agents

# Agent server
KAVALAI_AGENT_WORKFLOW_PATH=support_agent.yaml
KAVALAI_AGENT_PORT=10000

# Backoffice (its own database)
KAVALAI_BO_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
KAVALAI_BO_DB_SCHEMA=backoffice
SESSION_SECRET_KEY=change-me
```

Keep `.env` out of source control and prefer a secret store in production.
