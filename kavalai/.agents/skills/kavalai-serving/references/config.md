# Environment variables

Read by the processes only — `python -m kavalai.server`,
`python -m kavalai.migrate_db`, `kavalai-eval`, the backoffice — and by the
client constructors that fall back to a provider key. Library code reads none
of them: the engine takes `default_llm_model` / `default_llm_parameters` as
arguments and a normalizer is installed with `set_default_normalizer`.

## Provider credentials

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | `OpenAIClient` and `openai/…` models |
| `GEMINI_API_KEY` | `GeminiClient` and `gemini/…` models |
| `ANTHROPIC_API_KEY` | `AnthropicClient` and `anthropic/…` models. Note the `_API_` — `ANTHROPIC_KEY` is **not** read |
| `OLLAMA_HOST` | Ollama endpoint. Default `http://localhost:11434`. No key needed |
| `OPENAI_BASE_URL` | Read by the OpenAI SDK, not Kaval.AI: an OpenAI-compatible endpoint for `openai/…` models |
| `GOOGLE_API_KEY` | Read by the `google-genai` SDK, which prefers it over `GEMINI_API_KEY`. Kaval.AI reads only `GEMINI_API_KEY` |

Each client also accepts `api_key=` / `host=` directly, which wins over the
environment.

## Models

Read by `python -m kavalai.server` and, for the judge, by `kavalai-eval`.

| Variable | Description |
|---|---|
| `KAVALAI_DEFAULT_LLM_MODEL` | Model when a workflow and its nodes both omit `llm_model`, as `provider/model`. Passed to the engine as `default_llm_model` |
| `KAVALAI_LLM_TEMPERATURE` / `KAVALAI_LLM_TOP_P` / `KAVALAI_LLM_REASONING_EFFORT` / `KAVALAI_LLM_SERVICE_TIER` | Fleet-wide defaults for every model call, passed as `default_llm_parameters`. node `llm_kwargs` > graph `llm_kwargs` > these > provider defaults. Unset leaves the provider default |
| `KAVALAI_LLM_TIMEOUT_SECONDS` / `KAVALAI_LLM_STREAM_TIMEOUT_SECONDS` | Seconds before a call is abandoned (default `30`) and between streamed chunks (default twice the plain one) |
| `KAVALAI_EMBEDDING_NORMALIZER_YAML` | Path to a YAML file describing a custom `Normalizer`, installed at start-up with `set_default_normalizer` |
| `FASTEMBED_THREADS` | Thread count for local `fastembed` embedding |
| `FASTEMBED_CACHE_DIR` | Where `fastembed` caches models. Worth setting in a container so the model is not re-downloaded on every start |
| `KAVALAI_PROVIDER_MODULES` | Comma-separated modules the agent server imports before loading the workflow, so backends they register can be named from YAML. Every dotted registration is resolved afterwards, so a mistyped path fails at start-up rather than at the first request reaching that node |

There is no default *embedding* model variable: the model is always an
argument to `make_embedding_client`, the RAG services and the CSV indexer.

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
| `KAVALAI_BO_GOOGLE_CLIENT_ID` / `KAVALAI_BO_GOOGLE_CLIENT_SECRET` | Google OAuth sign-in. **Required** |
| `KAVALAI_BO_SESSION_SECRET_KEY` | Signing key for session cookies. **Required, no development fallback** — the backoffice refuses to start without it |
| `KAVALAI_BO_FRONTEND_URL` | Where a completed sign-in is redirected to. **Required** |

## Tools

| Variable | Description |
|---|---|
| `KAVALAI_TOR_PROXY_HOST` / `KAVALAI_TOR_PROXY_PORT` | Tor proxy used by `http_request(use_proxy=True)`. Default `localhost` / `8118` |

## A worked `.env` for local development

```bash
# Provider
OPENAI_API_KEY=sk-...
KAVALAI_DEFAULT_LLM_MODEL=openai/gpt-5.6-luna

# Agent database (runtime tables)
KAVALAI_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
KAVALAI_DB_SCHEMA=agents

# Agent server
KAVALAI_AGENT_WORKFLOW_PATH=support_agent.yaml
KAVALAI_AGENT_PORT=10000

# Backoffice (its own database)
KAVALAI_BO_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
KAVALAI_BO_DB_SCHEMA=backoffice
KAVALAI_BO_GOOGLE_CLIENT_ID=...
KAVALAI_BO_GOOGLE_CLIENT_SECRET=...
KAVALAI_BO_SESSION_SECRET_KEY=change-me
KAVALAI_BO_FRONTEND_URL=http://localhost:4200
```

Keep `.env` out of source control and prefer a secret store in production.
