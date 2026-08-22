Configuration
=============

Kaval.AI is configured with environment variables. Library code never reads them
on its own — only entry points do (``python -m kavalai.server``,
``python -m kavalai.migrate_db``, the backoffice, and the client constructors
that fall back to a provider key). Anything you build yourself can pass values
explicitly instead.

In development, keep them in a ``.env`` file and load it with
`python-dotenv <https://pypi.org/project/python-dotenv/>`_:

.. code-block:: python

   import dotenv

   dotenv.load_dotenv()

Provider credentials
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``OPENAI_API_KEY``
     - Used by :class:`~kavalai.OpenAIClient` and ``openai/…`` models.
   * - ``GEMINI_API_KEY``
     - Used by :class:`~kavalai.GeminiClient` and ``gemini/…`` models.
   * - ``ANTHROPIC_API_KEY``
     - Used by :class:`~kavalai.AnthropicClient` and ``anthropic/…`` models.
       Note the ``_API_`` — ``ANTHROPIC_KEY`` is **not** read.
   * - ``OLLAMA_HOST``
     - Ollama endpoint. Default ``http://localhost:11434``. No key needed.

Each client also accepts ``api_key=`` (or ``host=``) directly, which wins over
the environment.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_DEFAULT_LLM_MODEL``
     - Model used when a workflow and its nodes both omit ``llm_model``, as
       ``provider/model``.
   * - ``KAVALAI_DEFAULT_EMBEDDING_MODEL``
     - Default embedding model for RAG and embedding clients.
   * - ``KAVALAI_OPENAI_SERVICE_TIER``
     - OpenAI service tier (for example ``priority``). Read by the agent server.
   * - ``FASTEMBED_THREADS``
     - Thread count for local ``fastembed`` embedding.
   * - ``FASTEMBED_CACHE_DIR``
     - Where ``fastembed`` caches downloaded models. Worth setting in a
       container so the model is not re-downloaded on every start.
   * - ``KAVALAI_EMBEDDING_NORMALIZER_YAML``
     - Path to a YAML file describing a custom embedding
       :class:`~kavalai.Normalizer`.
   * - ``KAVALAI_PROVIDER_MODULES``
     - Comma-separated modules the agent server imports before loading the
       workflow, so any backends they register with
       :func:`~kavalai.register_llm_provider`,
       :func:`~kavalai.register_embedding_provider` or
       :func:`~kavalai.register_rag_service` can be named from the YAML. Every
       dotted registration is resolved afterwards, so a mistyped path fails at
       start-up rather than at the first request that reaches that node.
   * - ``KAVALAI_RAG_SERVICE``
     - Name of a registered RAG service for
       ``python -m kavalai.tools.index_csv`` to index into. Unset means the
       Postgres service built from ``KAVALAI_DB_URI``.

Agent database
--------------

Read by ``python -m kavalai.server`` and ``python -m kavalai.migrate_db app``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_DB_URI``
     - Connection string for the agent database, e.g.
       ``postgresql://user:pass@host:5432/kavalai``.
   * - ``KAVALAI_DB_SCHEMA``
     - Schema holding the runtime tables. Default ``public``; ``agents`` by
       convention.
   * - ``KAVALAI_DB_POOL_SIZE``
     - SQLAlchemy pool size. Default ``0``.
   * - ``KAVALAI_DB_MAX_OVERFLOW``
     - Pool overflow. Default ``0``.
   * - ``KAVALAI_SQL_ECHO``
     - Log every SQL statement. Default ``false``. Useful once, noisy always.

Agent server
------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_AGENT_WORKFLOW_PATH``
     - Path to the workflow YAML to serve. Required.
   * - ``KAVALAI_AGENT_SETUP_MODULE``
     - Optional module imported before the workflow is loaded — a dotted name
       or a ``.py`` path. Registers the ``python://`` tools and named RAG
       services the workflow refers to; a workflow with a ``rag_query`` node
       naming a registered service cannot be built without it. Same job as a
       suite's ``setup:`` key (:doc:`eval_yaml`).
   * - ``KAVALAI_AGENT_HOST``
     - Bind address. Default ``0.0.0.0``.
   * - ``KAVALAI_AGENT_PORT``
     - Port. Default ``10000``.
   * - ``KAVALAI_AGENT_BASIC_AUTH_USER``
     - Basic-auth username. Auth is enabled only when both this and the
       password are set.
   * - ``KAVALAI_AGENT_BASIC_AUTH_PASSWORD``
     - Basic-auth password.

See :doc:`../tutorials/serving`.

Backoffice
----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_BO_DB_URI``
     - Connection string for the backoffice's **own** database — separate from
       any agent database.
   * - ``KAVALAI_BO_DB_SCHEMA``
     - Schema for the backoffice tables.
   * - ``KAVALAI_BO_HOST``
     - Interface ``python -m kavalai.backoffice.server`` binds to. Default
       ``127.0.0.1``.
   * - ``KAVALAI_BO_PORT``
     - Port for the backoffice server. Default ``8000``.
   * - ``GOOGLE_OAUTH_CLIENT_ID``
     - Google OAuth client id for sign-in.
   * - ``GOOGLE_OAUTH_CLIENT_SECRET``
     - Google OAuth client secret.
   * - ``SESSION_SECRET_KEY``
     - Signing key for session cookies. **Set this in production** — the
       fallback is a well-known development value.
   * - ``FRONTEND_URL``
     - Where to redirect after a successful login.

Tools
-----

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``GOOGLE_CUSTOM_SEARCH_API_KEY`` / ``GOOGLE_CUSTOM_SEARCH_CX``
     - Credentials and search-engine id for ``google_custom_search``.
   * - ``LANGSEARCH_API_KEY``
     - Key for ``langsearch_web_search``.
   * - ``RSS_AUTH_USER`` / ``RSS_AUTH_PASSWORD``
     - Basic-auth credentials for protected RSS feeds.
   * - ``TOR_PROXY_HOST`` / ``TOR_PROXY_PORT``
     - Tor proxy used by ``http_request(use_proxy=True)``.

See :doc:`tools`.

A worked example
----------------

A ``.env`` for local development against Docker Compose:

.. code-block:: bash

   # Provider
   OPENAI_API_KEY=sk-...
   KAVALAI_DEFAULT_LLM_MODEL=openai/gpt-5.4-mini

   # Agent database (runtime tables)
   KAVALAI_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
   KAVALAI_DB_SCHEMA=agents

   # Agent server
   KAVALAI_AGENT_WORKFLOW_PATH=examples/v2_workflow_support_agent.yaml
   KAVALAI_AGENT_PORT=10000

   # Backoffice (its own database)
   KAVALAI_BO_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
   KAVALAI_BO_DB_SCHEMA=backoffice
   SESSION_SECRET_KEY=change-me

.. warning::

   A ``.env`` holds credentials. Keep it out of source control, and prefer your
   platform's secret store in production. The workflow YAML supports
   ``url_env`` / ``command_env`` / ``username_env`` / ``password_env`` for
   exactly this reason — see :doc:`yaml`.
