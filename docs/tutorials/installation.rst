Installation
============

Kaval.AI needs **Python 3.12 or newer**. Install it into a virtual environment,
configure one provider, and you are ready for the :doc:`quickstart`.

Install
-------

With ``uv`` (recommended):

.. code-block:: bash

   uv venv
   uv pip install "kavalai[common]"

With ``pip``:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install "kavalai[common]"

Which extra?
------------

The bare ``kavalai`` package is deliberately small and provider-agnostic: it is
restricted to libraries that also work under Pyodide, so the core can run in a
browser. Almost everyone wants ``common`` on top of it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - What it adds
   * - ``kavalai[common]``
     - The normal install: the OpenAI, Gemini, Anthropic and Ollama clients,
       embeddings and RAG, the PostgreSQL drivers, MCP, the REST/SSE servers
       and the bundled web tools.
   * - ``kavalai[common_web]``
     - The browser counterpart, for running the core under Pyodide / WebLLM.
       See :doc:`run_in_browser`.
   * - ``kavalai[test]``
     - Test tooling. Pulls in ``common``; this is what CI installs.
   * - ``kavalai[docs]``
     - Sphinx, the theme and the notebook kernel, for building these docs.

From source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Kaval-AI/kaval.ai.git
   cd kaval.ai
   uv pip install -e ".[common]"

The repository also carries the runnable notebooks behind every tutorial, under
``notebooks/``.

Configure a provider
--------------------

A model call needs a credential, read from the environment. Pick whichever you
have:

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."          # openai/…
   export GEMINI_API_KEY="..."             # gemini/…
   export ANTHROPIC_API_KEY="..."          # anthropic/…
   # Ollama runs locally and needs no key; set OLLAMA_HOST if it is not
   # on http://localhost:11434

In development, keep these in a ``.env`` file and load it explicitly — nothing
in the library reads ``.env`` on your behalf:

.. code-block:: python

   import dotenv

   dotenv.load_dotenv()

Every variable Kaval.AI understands is listed in :doc:`../reference/config`.

Check it works
--------------

.. code-block:: python

   import asyncio

   from kavalai import make_client


   async def main():
       client = make_client("openai/gpt-5.4-mini")
       print(await client.prompt("Say hello in Estonian."))


   asyncio.run(main())

.. code-block:: text

   Tere!

Optional pieces
---------------

**A database.** Nothing above needs one. Persistence — sessions, runs, chat
history, statistics — arrives when you hand an
:class:`~kavalai.agent_service.AgentService` to a workflow. In-memory SQLite
works for development; PostgreSQL is the production target, with ``pgvector``
if you use RAG. See :doc:`observability_storage` and :doc:`../deploy/index`.

**The backoffice UI.** A separate service for configuring and monitoring agents.
``docker compose up postgres_db backoffice-migrations backoffice`` brings it up
locally. See :doc:`../ui/index`.

**No install at all.** Kaval.AI can run a small open model entirely in your
browser over WebGPU — no Python, no API key. See :doc:`run_in_browser`.

Where to next
-------------

* :doc:`quickstart` — a model call, structured output and a workflow, in five
  minutes.
* :doc:`../guides/concepts` — the vocabulary, if LLMs are new to you.
