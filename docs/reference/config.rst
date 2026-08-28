Configuration
=============

Kaval.AI is configured with environment variables. Library code never reads them
on its own — only the processes do (``python -m kavalai.server``,
``python -m kavalai.migrate_db``, ``kavalai-eval``, the backoffice), and the
client constructors fall back to their provider's key variable. Anything you
build yourself can pass the same values explicitly: the engine takes
``default_llm_model`` and ``default_llm_parameters``, and a normalizer is
installed with :func:`~kavalai.set_default_normalizer`.
``.env.example`` in the repository lists every variable, and
``tests/test_config_drift.py`` checks it against the code in both directions.

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
   * - ``OPENAI_BASE_URL``
     - Read by the OpenAI SDK, not by Kaval.AI: points ``openai/…`` models at
       a proxy or an OpenAI-compatible endpoint when no ``base_url`` is passed.
   * - ``GOOGLE_API_KEY``
     - Read by the ``google-genai`` SDK, which prefers it over
       ``GEMINI_API_KEY`` when both are set. Kaval.AI reads only
       ``GEMINI_API_KEY``.

Each client also accepts ``api_key=`` (or ``host=``) directly, which wins over
the environment. :doc:`providers` lists every provider these credentials belong
to, and how to find out which models each one offers.

Models
------

Read by ``python -m kavalai.server`` and, for the judge, by ``kavalai-eval``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_DEFAULT_LLM_MODEL``
     - Model used when a workflow and its nodes both omit ``llm_model``, as
       ``provider/model``. The agent server passes it to the engine as
       ``default_llm_model``; an engine built in Python takes the argument
       directly.
   * - ``KAVALAI_LLM_TEMPERATURE``, ``KAVALAI_LLM_TOP_P``,
       ``KAVALAI_LLM_REASONING_EFFORT``, ``KAVALAI_LLM_SERVICE_TIER``
     - Fleet-wide defaults for every model call, passed to the engine as
       ``default_llm_parameters`` and to the eval judge. A graph's or node's
       ``llm_kwargs`` override them: node > graph > these > provider defaults.
       An unset value leaves the provider's own default in force.
   * - ``KAVALAI_LLM_TIMEOUT_SECONDS``
     - Seconds before a model call is abandoned. Default ``30``.
   * - ``KAVALAI_LLM_STREAM_TIMEOUT_SECONDS``
     - Inactivity timeout between streamed chunks. Defaults to twice the
       plain timeout.
   * - ``KAVALAI_EMBEDDING_NORMALIZER_YAML``
     - Path to a YAML file describing a custom embedding
       :class:`~kavalai.Normalizer`. The agent server and the ``ragindex``
       example install it with :func:`~kavalai.set_default_normalizer` at
       start-up.
   * - ``FASTEMBED_THREADS``
     - Thread count for local ``fastembed`` embedding.
   * - ``FASTEMBED_CACHE_DIR``
     - Where ``fastembed`` caches downloaded models. Worth setting in a
       container so the model is not re-downloaded on every start.
   * - ``KAVALAI_PROVIDER_MODULES``
     - Comma-separated modules the agent server imports before loading the
       workflow, so any backends they register with
       :func:`~kavalai.register_llm_provider`,
       :func:`~kavalai.register_embedding_provider` or
       :func:`~kavalai.register_rag_service` can be named from the YAML. Every
       dotted registration is resolved afterwards, so a mistyped path fails at
       start-up rather than at the first request that reaches that node.

Agent database
--------------

Read by ``python -m kavalai.server`` and ``python -m kavalai.migrate_db agents``.

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
       naming a registered service cannot be built without it.
   * - ``KAVALAI_AGENT_HOST``
     - Bind address. Default ``0.0.0.0``.
   * - ``KAVALAI_AGENT_PORT``
     - Port. Default ``10000``.
   * - ``KAVALAI_AGENT_BASIC_AUTH_USER``
     - Basic-auth username. Auth is disabled only when both this and the
       password are unset; setting either one enables it.
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
   * - ``KAVALAI_BO_GOOGLE_CLIENT_ID``
     - Google OAuth client id for sign-in. Required.
   * - ``KAVALAI_BO_GOOGLE_CLIENT_SECRET``
     - Google OAuth client secret. Required.
   * - ``KAVALAI_BO_SESSION_SECRET_KEY``
     - Signing key for session cookies. Required, with no development
       fallback: a cookie signed with a well-known key is a backoffice that
       looks as if it works until it is exposed.
   * - ``KAVALAI_BO_FRONTEND_URL``
     - Where a completed sign-in is redirected to. Required.

The backoffice refuses to start when any of the four is unset, with a message
naming the missing variable.

Tools
-----

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Description
   * - ``KAVALAI_TOR_PROXY_HOST`` / ``KAVALAI_TOR_PROXY_PORT``
     - Tor proxy used by ``http_request(use_proxy=True)``. Default
       ``localhost`` / ``8118``.

See :doc:`tools`.

A worked example
----------------

A ``.env`` for local development against Docker Compose:

.. code-block:: bash

   # Provider
   OPENAI_API_KEY=sk-...
   KAVALAI_DEFAULT_LLM_MODEL=openai/gpt-5.6-luna

   # Agent database (runtime tables)
   KAVALAI_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
   KAVALAI_DB_SCHEMA=agents

   # Agent server
   KAVALAI_AGENT_WORKFLOW_PATH=examples/support_agent/support_agent.yaml
   KAVALAI_AGENT_PORT=10000

   # Backoffice (its own database)
   KAVALAI_BO_DB_URI=postgresql://kavalai:kavalai@localhost:5432/kavalai
   KAVALAI_BO_DB_SCHEMA=backoffice
   KAVALAI_BO_GOOGLE_CLIENT_ID=...
   KAVALAI_BO_GOOGLE_CLIENT_SECRET=...
   KAVALAI_BO_SESSION_SECRET_KEY=change-me
   KAVALAI_BO_FRONTEND_URL=http://localhost:4200

.. warning::

   A ``.env`` holds credentials. Keep it out of source control, and prefer your
   platform's secret store in production. The workflow YAML supports
   ``url_env`` / ``command_env`` / ``username_env`` / ``password_env`` for
   exactly this reason — see :doc:`yaml`.
