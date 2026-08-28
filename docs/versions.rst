Versions
========

The main changes in each release of the ``kavalai`` package. Releases are
tagged ``vX.Y.Z`` in the repository and published to PyPI.

1.0.3 — 2026-08-28
------------------

Added
^^^^^

* Provider registries (:mod:`kavalai.llm_clients.registry`):
  :func:`~kavalai.register_llm_provider`,
  :func:`~kavalai.register_embedding_provider` and
  :func:`~kavalai.register_rag_service` accept a class, a dotted path or a
  callable, so a third-party client is one registration away. ``BaseLlmClient``,
  ``BaseEmbeddingClient`` and ``ensure_user_turn`` are exported for that purpose,
  and ``KAVALAI_PROVIDER_MODULES`` loads such modules at start-up.
* ``rag_query`` workflow node and ``WorkflowEngine(rag_services=…)``; a node
  resolves its service as *node → graph → "default"*.
* ``parallel`` workflow node with concurrent branch execution.
* Engine lifecycle: ``await engine.connect()`` / ``await engine.aclose()``, or
  the async context manager, open and release the MCP sessions once per engine.
* ``Agent(allowed_tools=…)`` and ``allowed_tools`` on ``agent`` nodes, with
  ``"*"`` and ``proto://server.*`` patterns.
* Fleet-wide model defaults: ``WorkflowEngine(default_llm_model=…,
  default_llm_parameters=…)``, filled by the server from
  ``KAVALAI_DEFAULT_LLM_MODEL`` and the ``KAVALAI_LLM_*`` variables.
* Evaluation package :mod:`kavalai.eval` (``SimpleEvaluator``,
  ``JudgeEvaluator``, YAML case files) and the ``kavalai-eval`` console script.
* Six agent skills shipped in the wheel and installed by ``kavalai-skills
  install``.
* ``gpu`` extra (``fastembed-gpu``) for local embedding on an NVIDIA GPU.
* Key-free ``web_search`` tool over DuckDuckGo, and a ``crawl4ai`` Compose
  service.
* ``KAVALAI_AGENT_SETUP_MODULE`` registers tools and RAG services before the
  agent server loads its workflow.
* Migrations: ``model_call_stats`` records ``cached_prompt_tokens`` and
  ``reasoning_tokens``; ``tasks`` records ``seq``, ``parent_task_name`` and
  ``tool_uri``; ``users.active_project_id`` is cleared when its project is
  deleted.
* Documentation: quickstart, architecture, comparison, serving, guides,
  reference, cookbook and deployment pages; examples ``green_village``,
  ``bakery``, ``business_info_agent``, ``ragindex`` and ``chat_client``.

Changed
^^^^^^^

* **Breaking.** Packaging extras collapsed to ``common``, ``common_web``,
  ``gpu``, ``test`` and ``docs``; the per-provider extras (``openai``,
  ``gemini``, ``anthropic``, ``ollama``, ``rag``, ``postgres``, ``mcp``,
  ``server``, ``tools``, ``all``, …) are gone. Install with
  ``pip install "kavalai[common]"``.
* **Breaking.** Environment variables renamed: ``GOOGLE_OAUTH_CLIENT_ID`` /
  ``_SECRET`` → ``KAVALAI_BO_GOOGLE_CLIENT_ID`` / ``_SECRET``,
  ``FRONTEND_URL`` → ``KAVALAI_BO_FRONTEND_URL``, ``BACKOFFICE_HOST`` /
  ``_PORT`` → ``KAVALAI_BO_HOST`` / ``_PORT``, ``KAVALAI_LLM_TIMEOUT`` →
  ``KAVALAI_LLM_TIMEOUT_SECONDS``, ``TOR_PROXY_HOST`` / ``_PORT`` →
  ``KAVALAI_TOR_PROXY_HOST`` / ``_PORT``. ``KAVALAI_BO_SESSION_SECRET_KEY`` is
  required and has no fallback.
* ``create_model_call_stat(duration_sections=…)`` renamed to
  ``duration_seconds``.
* The Gemini client retries only HTTP 429; 400, 401 and 403 raise immediately.
* Integration tests carry the ``integration`` marker and are deselected by
  default.

Removed
^^^^^^^

* **Breaking.** ``KAVALAI_OPENAI_SERVICE_TIER`` — use
  ``KAVALAI_LLM_SERVICE_TIER`` or ``llm_kwargs``.
* ``kavalai.tools.websearch`` (Serper, LangSearch, Google Custom Search) and
  their API-key variables; the RSS tool.
* ``cost`` and ``currency`` columns on ``model_call_stats`` (see
  :doc:`guides/observability` for the reason).
* ``KAVALAI_DEFAULT_EMBEDDING_MODEL``.
* ``kavalai/tools/index_csv.py`` and ``cli_chat.py`` — now
  ``examples/ragindex`` and ``examples/chat_client``.

Fixed
^^^^^

* ``PostgresTaskLogger`` dropped ``cached_prompt_tokens`` and
  ``reasoning_tokens``.
* A stale ``active_project_id`` after a project was deleted or a member removed
  made every project-scoped backoffice endpoint answer 403.
* The backoffice sessions page issued one query per session.

1.0.2 — 2026-08-11
------------------

Added
^^^^^

* ``AnthropicClient``.
* ``SqliteRagService`` — a sqlite-vector file index that also runs in the
  browser — behind the ``BaseRagService`` interface shared with
  ``PostgresRagService``.
* Streaming: ``WorkflowEngine.run_stream()`` yielding ``WorkflowStreamEvent``,
  ``POST /stream_agent`` on the agent server, per-node ``stream_delta`` /
  ``stream_instructions`` / ``stream_partials`` flags and
  ``stream_timeout_seconds`` on ``LlmClientParameters``.
* Alembic migration sets ``agents`` and ``backoffice`` replace the plain-SQL
  scripts.

Changed
^^^^^^^

* **Breaking.** Package layout flattened: ``kavalai.agents.*`` moved to the top
  level (``kavalai.agent``, ``kavalai.db``, ``kavalai.server``, …).
* **Breaking.** ``LlmClientParameters`` no longer defaults ``temperature`` and
  ``top_p``; the provider's defaults apply.
* All workflow persistence goes through ``AgentService``.

Removed
^^^^^^^

* **Breaking.** ``kavalai.workflow.storage`` (``DataStorage``, ``RunHandle``,
  ``InMemoryDataStorage``, ``SqliteDataStorage``) and the ``RagService`` class,
  replaced by ``AgentService`` and the RAG services above.

1.0.1 — 2026-07-06
------------------

First release on PyPI.

* A minimal, Pyodide-compatible core with optional extras; Python 3.12 or
  later.
* LLM clients for OpenAI, Gemini and Ollama behind one streaming interface, a
  FastEmbed embedding client, and ``BrowserLLMClient`` (WebLLM) for execution
  in the browser.
* The planning agent and function kernel (Python, REST and MCP tools).
* The workflow engine with ``start``, ``end``, ``llm``, ``agent``,
  ``function``, ``if`` and ``switch`` nodes, YAML definitions and SVG rendering.
* PostgreSQL/pgvector RAG service.
* The agent REST server and the backoffice (FastAPI and Angular) with agents,
  conversations, RAG, workflow and task pages.
* The documentation site.
