Versions
========

The main changes in each release of the ``kavalai`` package. Releases are
tagged ``vX.Y.Z`` in the repository and published to PyPI.

1.0.4 — 2026-09-12
------------------

Added
^^^^^

* SQLite for the backoffice. ``KAVALAI_BO_DB_URI`` accepts a ``sqlite:///path``
  URI, the backoffice migration set applies to SQLite, and a project has a
  ``db_type`` — ``postgresql`` (default) or ``sqlite``, in which case its
  ``db_name`` is the agent database file. The project form and the projects
  page show the fields the chosen type needs.
* ``sqlite:///path`` is accepted wherever a database URI is: ``KAVALAI_DB_URI``
  for the agent server and migrations, ``DatabaseManager.get_sessionmaker``,
  ``PostgresRagService.from_uri``'s counterpart
  :func:`~kavalai.rag.rag_service_from_uri`, which picks the RAG service from
  the URI scheme.
* :class:`~kavalai.rag.CollectionRagService` (:mod:`kavalai.rag.collections`):
  the storage model the two RAG services share — a ``rag_collections``
  registry and one table per collection — with the browse methods the
  backoffice needs (``list_collections``, ``get_stats``, ``create_collection``,
  ``drop_collection``, ``get_embeddings_by_ids``) on both backends. ``model``
  is optional on both constructors; without one a service browses existing
  collections.
* ``LLMNode.history_limit`` (default ``50``) and ``history_max_chars``: the
  chat history an ``llm`` node sends is the most recent messages, and at most
  that many characters of them — whole messages are dropped from the oldest
  end. ``AgentService.get_chat_history`` takes ``max_chars``.
* ``templates=`` on :meth:`~kavalai.WorkflowEngine.run` and
  :meth:`~kavalai.WorkflowEngine.run_stream`: values for templates the
  document declares, for one run. Rendering is one pass, so the values need no
  escaping; they are recorded under ``run_templates`` in ``runs.context``.
* ``timeout=`` on ``run`` / ``run_stream`` and ``run_timeout=`` on the engine
  cancel a run after that many seconds, parallel branches included, and record
  it as failed with the new :class:`~kavalai.WorkflowTimeoutError`. The agent
  server reads ``KAVALAI_AGENT_RUN_TIMEOUT_SECONDS``.
* ``WorkflowEngine(record_context=False)`` keeps only a run's input and output
  in ``runs.context``.
* ``rag_query`` nodes take ``min_similarity``, and record their hits — ``id``,
  ``source_id``, ``similarity``, ``metadata`` — on the task row and in the
  ``output_data`` of ``node_completed``, whatever ``store`` says. The query
  embedding is reported to the run's token accumulator.
* ``workflow_completed`` and ``workflow_failed`` events carry ``run_id``.
* :func:`kavalai.server.public_events`, an event filter for streams served to
  people other than the operator (no node events, token usage or restart
  reasons; a failure's text replaced by a fixed message quoting the run id),
  and ``event_filter=`` on ``stream_sse_events``, ``create_agent_router`` and
  ``create_agent_app``. ``KAVALAI_AGENT_PUBLIC_EVENTS`` applies it under
  ``python -m kavalai.server``.
* ``kavalai.server.sse_response`` and ``AgentRequest[InputT]``, so a route of
  a host's own speaks the same SSE protocol as the router.
* ``max_output_tokens`` on :class:`~kavalai.LlmClientParameters`, sent under
  each provider's own name: ``max_output_tokens`` (OpenAI, Gemini),
  ``max_tokens`` (Anthropic, WebLLM), ``options.num_predict`` (Ollama). On
  reasoning models the cap includes the reasoning tokens.
  ``KAVALAI_LLM_MAX_OUTPUT_TOKENS`` sets it for the agent server and the eval
  judge.
* :class:`~kavalai.OutputTruncatedError`, raised when a provider stops at the
  output cap instead of the partial answer being returned. It is not retried,
  carries the partial output and token counts, and the truncated call is still
  recorded. It and :class:`~kavalai.LlmClientException` are exported from
  ``kavalai``.
* ``reasoning_effort`` reaches Gemini, as ``thinking_level``, and Ollama, as
  ``think`` (``"none"`` sends ``false``).
* ``dimensions=`` on the OpenAI and Gemini embedding clients. Bound at
  registration —
  ``register_embedding_provider("openai-512", OpenAIEmbeddingClient,
  dimensions=512)`` — it gives a reduced model a name of its own.
* Four extras dividing ``common`` by weight: ``runtime`` (provider SDKs, MCP,
  FastAPI, asyncpg, sqlite-vector), ``webtools`` (crawl4ai), ``fastembed`` and
  ``backoffice`` (Authlib, itsdangerous, scikit-learn, sse-starlette).
  ``common`` installs all four and keeps its meaning.
* ``migrate(set_name, connection=…)`` runs a migration set on an open
  connection, inside the caller's transaction, and ``migrate_async`` does the
  same from inside an event loop, with an ``AsyncConnection`` or a URI.
* :mod:`kavalai.net`, a guard against server-side request forgery for tools
  that fetch a URL the model chooses: :func:`~kavalai.net.is_public_address`,
  :func:`~kavalai.net.ensure_public_url` and
  :class:`~kavalai.net.PublicOnlyTransport`, an httpx transport that resolves
  a host once, checks every address and connects to the address it checked,
  so DNS rebinding cannot redirect it; each redirect hop is checked again.
  ``make_http_request(allow_private_networks=True)`` and
  ``make_crawl_url(allow_private_networks=True)`` build the bundled tools for
  an agent meant to reach an intranet; the switch is an argument of the
  factory, not of the tool, so the model cannot set it.
* :mod:`kavalai.testing`, test doubles in the base install that replace the
  model and nothing else. :class:`~kavalai.testing.ScriptedLlmClient` answers
  from a list of replies or a function, streams each reply through the base
  client's streamer, retry and restart handling, reports a
  :class:`~kavalai.ModelCallStat` for every call and records the requests it
  received; the instance is itself an engine ``client_factory``.
  :class:`~kavalai.testing.FakeEmbeddingClient` embeds any text
  deterministically by hashing its words, and
  :func:`~kavalai.testing.fake_providers` registers both under a provider name
  for a ``with`` block and restores the registries on exit.
* :mod:`kavalai.text`: ``parse_html`` reduces a page to its title, text blocks
  with their heading paths, links and robots directives in one pass, with the
  boilerplate rules as parameters; ``chunk_blocks`` and ``chunk_markdown`` pack
  blocks into chunks of about ``target_chars`` (1200) and never over
  ``max_chars`` (2000), splitting at lines, sentences and words before
  cutting. Standard library only, so it runs under Pyodide, and linear in its
  input. ``tools/zerocostchatbot`` uses it, and ``build_index.py`` gains
  ``--target-chars``.
* The chat widget ships in the wheel as ``kavalai.widget``
  (``kavalai/widget/kaval-chatbot.js`` and ``.css``, moved from
  ``chatbotwidget/``); ``widget_dir()`` and ``asset_path(name)`` locate the
  installed files for a Python host, and :doc:`reference/widget` documents it.
  New options: ``texts`` for every visible string, aria labels and error
  messages included; a disclosure label, ``AI assistant``, on by default
  (``disclosure: false`` removes it); ``onFeedback(runId, vote, comment)`` with
  thumbs and an optional comment box; ``widget.on(...)`` events (``open``,
  ``close``, ``reply``, ``error``, ``resize``, ``feedback``);
  ``agentConnector`` ``headers`` (an object or a function), ``onResponse``,
  ``setConversationId`` and ``storage`` (``"session"``, ``"local"``,
  ``"none"``); the reply carries the ``runId`` of ``workflow_started``;
  ``--kcb-font-family`` and ``--kcb-font-size``.
* ``delete_many(ids, collection_name=None)`` on every RAG service (one
  statement per collection on the SQL backends; a loop over ``delete`` by
  default), and, in the optional tier, ``delete_by_metadata(collection_name,
  match)`` and ``replace(collection_name, texts, metadata_list,
  source_ids=None, *, match=None, source_id=None)``. A ``match`` is equality
  on top-level keys with scalar values, served by ``metadata @> :match`` on
  PostgreSQL. ``replace`` embeds first, then deletes and inserts in one
  transaction, so a failure leaves the old rows; empty ``texts`` deletes the
  selection.
* ``min_similarity=`` on ``query`` and ``query_batch``.
* ``stats_receiver=`` on ``query``, ``query_batch``, ``index``, ``index_batch``
  and ``replace``, and on the RAG service constructors as the default. Each
  embedding call's ``ModelCallStat`` goes to it; a ``rag_query`` node passes
  the run's receiver.
* ``provision=False`` on the RAG services: no DDL; a missing registry reads as
  empty and indexing into a missing collection raises ``RuntimeError`` naming
  ``create_collection()``. ``ensure_registry()`` is public, and
  ``create_collection(name, dim, *, model=None, vector_type=None)`` issues DDL
  whatever ``provision`` says.
* ``vector_type="halfvec"`` on ``PostgresRagService`` (pgvector 0.7 or
  later): 16-bit embeddings and HNSW indexes for up to 4,000 dimensions. The
  type is read back from the collection's column.
* Filtered PostgreSQL queries enable ``hnsw.iterative_scan = relaxed_order``
  for their transaction on pgvector 0.8 or later, so a filter whose rows lie
  away from the query still returns ``top_k`` rows.
* ``rag_service_from_uri`` forwards further keyword options to the service.
* ``KAVALAI_RAG_MODEL`` registers the ``default`` RAG service when the agent
  server starts, over the index at ``KAVALAI_RAG_URI`` (a Postgres URI or
  ``sqlite:///file``) in the optional ``KAVALAI_RAG_SCHEMA``, with the
  normalizer from ``KAVALAI_EMBEDDING_NORMALIZER_YAML``; a deployment with one
  index needs no setup module. The URI is required with the model: the index
  is not assumed to live in the agent database.
* ``model_call_stats.session_id`` and ``run_id`` (agents migration ``0005``):
  every model call a run makes — the query embedding of a ``rag_query`` node
  included — is attributed to its agent, session and run, so cost per run and
  per conversation is a query. ``TaskLogger.log_model_call``,
  ``TokenAccumulator``, ``StatsBridge`` and
  ``AgentService.add_model_call_stats`` take ``session_id`` and ``run_id``;
  ``get_model_call_stats`` filters by ``agent_id``, ``session_id`` and
  ``run_id``; ``MemoryTaskLogger.model_calls`` holds ``ModelCallRecord``
  objects carrying the ids, with ``model_calls_for_run``; ``SqliteTaskLogger``'s own
  table gains the columns and ``get_model_calls(run_id)`` filters by them.
* ``record_nodes`` and ``record_payloads`` on every task logger: record no
  node rows, or keep names, timings, errors and token counts but not the
  prompts, inputs, outputs and model-call payloads.
* ``write_node`` and ``write_model_call``, the public hooks a task logger
  backend implements, and ``TeeTaskLogger`` to put a logger of one's own —
  a meter, say — beside the database one.
* ``AgentService.list_sessions(agent_ids, *, search, external_id_prefix,
  exclude_external_id_prefix, start, end, limit, offset)`` — the backoffice
  list's query, scoped to several agents at once — and
  ``AgentService.purge_sessions(before, agent_ids=None, batch_size=1000)``,
  which deletes idle sessions in batches, yields each batch's ids, and blanks
  the purged sessions' model-call payloads while keeping their token counts.
* Backoffice: ``GET /projects/{project_id}/llm-call-stats`` accepts
  ``agent_id``, ``session_id`` and ``run_id``. The **Model calls** page links
  each call to its conversation and to its run's tasks, filters by the
  ``run_id`` and ``session_id`` query parameters, and is reached from a
  conversation's **Model calls** button. The task debugger lists the run's
  model calls — tokens and duration per call — beneath its tasks.
  ``GET /projects/{project_id}/rag/collections`` returns each collection's
  name, model, dimension, schema version and entry count.

Changed
^^^^^^^

* An ``llm_kwargs`` key that is not a field of ``LlmClientParameters`` fails
  the workflow load, naming the key and listing the valid ones, with a hint
  towards ``max_output_tokens`` for ``max_tokens``,
  ``max_completion_tokens``, ``num_predict`` and ``max_new_tokens``; a value of
  the wrong type fails too. Such keys were ignored before.
  ``workflow.clients.build_parameters`` refuses them as well.
* A ``restart`` event is emitted only by a node that streams; a node that
  streams nothing has no partial output for a client to discard.
* A run that fails with a :class:`~kavalai.WorkflowException` is recorded as
  failed on its run row, as other failures already were.
* A RAG service registered by name is built once per engine and kept, instead
  of on every ``rag_query`` execution.
* The OpenAI embedding client records the response's ``model`` and ``usage``
  only; the vectors are no longer copied into ``model_call_stats``.
* The OpenAI client sends a structured-output schema as ``text.format``, so
  the SDK no longer validates a truncated answer before the stream reports
  why it stopped. An OpenAI response that is incomplete for another reason,
  and an Anthropic stop at ``model_context_window_exceeded``, raise
  ``LlmClientException``. An error a client raises as a ``RuntimeError``
  subclass reaches the stream consumer as that type.
* Migrations run over the runtime's async drivers — asyncpg for
  ``postgresql://``, aiosqlite for ``sqlite://`` — through Alembic's
  ``run_sync`` recipe, instead of converting the URI to psycopg2. A URI that
  names a synchronous driver is run on that driver. ``migrate()`` called
  inside a running event loop runs on a worker thread. Its ``schema``,
  ``skip_create_schema`` and ``max_wait`` are keyword-only, and SQLite is no
  longer waited for.
* The agent image installs ``kavalai[runtime]`` (``--build-arg EXTRAS=…`` for
  more) and the backoffice image ``kavalai[runtime,backoffice]``.
* Install hints name the narrowest extra, e.g.
  ``pip install "kavalai[runtime]"``; the agent server, the backoffice, its
  embedding projector, FastEmbed and the crawl4ai tools fail with a hint
  instead of a bare ``ImportError``.
* ``http_request`` refuses private, loopback, link-local and cloud metadata
  targets with :class:`~kavalai.net.UnsafeUrlError`, and is a coroutine
  function. The guarded client ignores ``HTTP_PROXY`` / ``HTTPS_PROXY``; with
  ``use_proxy=True`` only the URL pre-check applies, since the proxy resolves
  the name. ``crawl_url`` refuses such a URL with ``success=False`` and a
  ``Refused:`` message, and withholds the content when crawl4ai reports a
  final URL that fails the check.
* The widget no longer retries a turn the server has started: once an SSE
  frame has arrived, a broken stream ends with the ``interrupted`` text
  instead of sending the message again and paying for the run twice. Statuses
  below 500 are not retried either.
* ``source_ids=[]`` matches nothing: ``query`` returns ``[]`` and
  ``query_batch`` one empty list per text, without embedding; ``None`` still
  means no filter. ``index_batch`` with ``source_ids=[]`` and non-empty texts
  raises the length error. A query on a collection that does not exist returns
  ``[]`` without embedding.
* The registry's model is authoritative: an existing collection is queried and
  indexed with the model recorded for it, one embedding client per model. The
  constructor's ``model`` is the model for *new* collections; a mismatch logs
  a warning. A service built with ``model=None`` can query and index existing
  collections and refuses only to create one.
* Read paths (``list_collections``, ``get_stats``, ``count_entries``,
  ``query``, the deletes) no longer create the registry.
* When two processes create one collection at once with different dimensions
  or models, the one whose registry row lost raises ``ValueError`` instead of
  carrying on.
* ``examples/ragindex/index_csv.py --replace`` uses ``replace``, one source
  per transaction. ``--index`` in ``index_csv.py`` and ``query_index.py`` is a
  database URI or a SQLite file path; the ``postgres`` shorthand, which read
  ``KAVALAI_DB_URI``, is gone.
* ``sessions.updated_at`` is moved by every run and so records a session's
  last activity; before, nothing updated it after the session was created.
  Migration ``0005`` backfills it from each session's last run, adds composite
  indexes — ``sessions(agent_id, external_id)``,
  ``sessions(agent_id, updated_at)``, ``sessions(updated_at)``,
  ``chat_messages(session_id, created_at)``,
  ``chat_messages(agent_id, created_at)``,
  ``model_call_stats(agent_id, created_at)`` — and drops the single-column
  indexes they replace. ``SQLITE_SCHEMA_VERSION`` is ``5``.
* A task logger's ``max_payload_bytes`` caps model-call payloads as well as
  node payloads.
* ``AgentService.delete_history_for_session`` also blanks the session's
  model-call payloads.
* ``kavalai.backoffice.sessions.get_sessions_summary`` delegates to
  :func:`kavalai.agent_service.summarise_sessions`, where ``SessionSummary``
  now lives.
* Backoffice: the conversation list is ordered by last activity — the time of
  a session's most recent run — and labels its timestamp accordingly.
  ``POST /projects/{project_id}/rag/query`` no longer requires ``model``: an
  existing collection is embedded with its recorded model, which the PCA
  projection uses as well. The RAG explorer shows the selected collection's
  recorded model instead of asking for one, and drops **All Collections**,
  since a query searches exactly one collection.
* ``SqliteRagService`` uses the shared storage model: a registry and a table
  per collection, each with its own embedding dimension, instead of a single
  ``rag_index`` table with one dimension per file. A file in the old layout is
  refused with a message asking for the index to be rebuilt; the
  ``table_name`` constructor argument is gone. ``collection_name=None`` now
  means ``"default"``, as on Postgres, rather than every collection.
* The backoffice session list no longer depends on Postgres-only SQL
  (``DISTINCT ON``, ``jsonb_typeof``): :func:`~kavalai.db.json_typeof` and
  :func:`~kavalai.db.json_array_length` render per dialect and a window
  function ranks the messages.
* Backoffice migration ``0002`` runs in batch mode, and UUID columns in the
  backoffice set use ``uuid_column()``, so the set applies on SQLite as it
  does on Postgres.

Removed
^^^^^^^

* ``kavalai.llm_clients.kwargs_mapper``, which nothing imported and which
  disagreed with the clients.
* ``psycopg2-binary`` from every extra, and
  ``kavalai.migrate_db.ensure_sync_scheme``.
* The RAG services no longer write embedding statistics into
  ``model_call_stats`` themselves. A standalone indexing job that wants the
  record passes ``stats_receiver=StatsBridge(task_logger, agent_id)``.
* ``PostgresRagService``'s ``agent`` parameter.
* The private task logger hooks ``_log_node_impl`` and
  ``_log_model_call_impl`` (now ``write_node`` / ``write_model_call``), and
  ``kavalai.workflow.tasklog.postgres._to_orm_stat``.
* ``chatbotwidget/`` at the repository root; the widget lives in
  ``kavalai/widget/``. ``tools/zerocostchatbot/htmlmd.py`` and the tool's own
  chunker, replaced by :mod:`kavalai.text`.

Fixed
^^^^^

* ``AgentService.get_chat_history`` returned the oldest ``limit`` messages of
  a session instead of the most recent ones, so a long conversation lost its
  latest turns from the model's view.
* A response cut off at the output limit was returned as if complete: partial
  text, or for structured output JSON repaired into a model with content
  missing. Every client now checks the provider's stop reason.
* Gemini usage without a candidate token count (a cap spent on thoughts alone)
  no longer fails the stats record.
* The wheel carries ``default_prompt_template.j2``; ``Agent()`` from an
  installed 1.0.x wheel raised ``FileNotFoundError``.
* ``sse-starlette``, imported by the backoffice, is declared rather than
  arriving through ``mcp``; the dead ``demo_agents/**`` package-data globs are
  gone.
* Widget styles: the button reset no longer overrides the send button,
  launcher, greeting and choice chips, and the ``hidden`` attribute is
  honoured, so the typing indicator and a closed window no longer show.
  Conversation ids are generated on pages without a secure context.
* A ``normalizer`` given to a RAG service was never applied: the services
  passed it to the embedding client without asking for normalisation. It is
  now applied on the index side and on the query side alike.
* SQLite ``delete_by_metadata`` does not treat JSON ``true`` as the number 1,
  matching PostgreSQL.
* The Postgres insert casts embeddings to the schema-qualified
  ``public.vector`` type, as the query always did.

Upgrading
^^^^^^^^^

* A workflow whose ``llm_kwargs`` carries a key the clients do not know — most
  often ``max_tokens`` — no longer loads. Rename it (``max_output_tokens``) or
  remove it; it had no effect before.
* Code that used psycopg2 because ``kavalai[common]`` happened to install it
  — a sync engine on a ``postgresql://`` URI, for instance — must now depend
  on ``psycopg2-binary`` itself, or move to asyncpg. ``python -m
  kavalai.migrate_db`` needs neither change: it uses asyncpg, and an explicit
  ``postgresql+psycopg2://`` URI still works where psycopg2 is installed.
* A service that installed ``common`` only to serve workflows can install
  ``kavalai[runtime]``; add ``fastembed`` for local embedding models and
  ``webtools`` for ``crawl_url`` / ``web_search``.
* An agent that must reach an intranet through ``http_request`` or
  ``crawl_url`` registers ``make_http_request(allow_private_networks=True)``
  / ``make_crawl_url(allow_private_networks=True)`` instead. A direct
  synchronous call of ``http_request`` now returns a coroutine.
* Pages that load the widget from ``chatbotwidget/`` load it from
  ``kavalai/widget/`` (or from the installed package, through
  ``kavalai.widget.widget_dir()``).
* ``PostgresRagService(session_maker, model=None, *, normalizer=None,
  schema=None, provision=True, stats_receiver=None, vector_type="vector")``:
  the positional ``agent`` parameter is gone and every option after ``model``
  is keyword-only. Pass ``normalizer=`` by name; replace ``agent=`` with
  ``stats_receiver=StatsBridge(task_logger, agent_id)`` where the statistics
  are wanted.
* ``model=`` on a RAG service now names the model for collections it creates.
  Existing collections are embedded with their recorded model, and a service
  without a model can query them, where it used to raise.
* An index built under 1.0.3 by a RAG service that had a ``normalizer`` holds
  unnormalised vectors, while its queries are now normalised. Rebuild such an
  index, or construct the service without the normalizer.
* A caller that computes ``source_ids`` and may produce an empty list now gets
  no hits for it, rather than a search of the whole collection. Pass ``None``
  where "no restriction" is meant.
* Run the agents migrations (``python -m kavalai.migrate_db agents``) for
  revision ``0005``. It rewrites ``sessions`` once. On a large PostgreSQL
  database the indexes can be built beforehand with ``CREATE INDEX
  CONCURRENTLY`` under the names above, and the revision then leaves them.
  Embedding vectors that earlier releases copied into
  ``model_call_stats.response_data`` are not removed by the migration; clear
  them when convenient with ``update model_call_stats set response_data =
  null where call_type = 'embedding' and response_data is not null;``.
* A task logger subclass overriding ``_log_node_impl`` /
  ``_log_model_call_impl`` implements ``write_node`` /
  ``write_model_call(stats, *, agent_id, session_id, run_id)`` instead. Code
  that imported ``_to_orm_stat`` to store statistics passes the Pydantic
  ``ModelCallStat`` to ``AgentService.add_model_call_stats`` or to a
  ``StatsBridge``.
* Browser (Pyodide) SQLite databases are recreated on first use, since
  ``SQLITE_SCHEMA_VERSION`` moved to ``5``.

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
