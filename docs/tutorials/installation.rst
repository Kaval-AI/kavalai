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

Choosing an extra
-----------------

The bare ``kavalai`` package is deliberately small and provider-agnostic: it is
restricted to libraries that also work under Pyodide, so the core can run in a
browser. ``common`` on top of it is the normal install.

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
   * - ``kavalai[gpu]``
     - Local embedding on an NVIDIA GPU. Replaces the CPU ``fastembed``
       rather than adding to it — see below.
   * - ``kavalai[test]``
     - Test tooling. Pulls in ``common``; this is what CI installs.
   * - ``kavalai[docs]``
     - Sphinx, the theme and the notebook kernel, for building these docs.

Embedding on a GPU
^^^^^^^^^^^^^^^^^^

``fastembed`` runs the local embedding models on the CPU. ``fastembed-gpu`` is
the same library under the same import name, built against ``onnxruntime-gpu``,
so the two cannot be installed together — the GPU extra *replaces* the CPU
package instead of adding to it:

.. code-block:: bash

   uv pip uninstall fastembed
   uv pip install "kavalai[common,gpu]"

Nothing else changes. FastEmbed defaults to ``cuda=Device.AUTO`` and selects
the CUDA execution provider when one is available, so a model name like
``fastembed/BAAI/bge-small-en-v1.5`` keeps working and runs on the GPU.
To be explicit instead, register the provider with ``cuda=True`` (and
``device_ids=[0]`` to pick a card):
:class:`~kavalai.llm_clients.embeddings.FastEmbedClient` passes both straight
through to FastEmbed.

Check the card before taking the newest wheel. The current
``onnxruntime-gpu`` on PyPI is a CUDA 13 build, and CUDA 13 dropped compute
capability below 7.5 — on a Pascal or Volta card it fails to load
``libcublasLt.so.13`` and quietly falls back to the CPU. Such a card needs the
CUDA 12 build and a cuDNN 9 still carrying its kernels:

.. code-block:: bash

   uv pip install "onnxruntime-gpu==1.22.0"
   uv pip install nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 \
       nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
       "nvidia-cudnn-cu12==9.1.0.70"

Those wheels install under ``site-packages/nvidia/*/lib``, which the dynamic
loader does not search by default, so put them on the path when running:

.. code-block:: bash

   export LD_LIBRARY_PATH=$(python -c "import glob, os, site; \
       print(':'.join(sorted({os.path.dirname(f) for f in \
       glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib/*.so*')})))")

FastEmbed logs which providers it ended up with; if the list is
``['CPUExecutionProvider']`` the GPU was not picked up, and the embeddings are
still correct but computed on the CPU.

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
       client = make_client("openai/gpt-5.6-luna")
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

**Skills for your coding agent.** ``kavalai-skills install`` copies Kaval.AI's
agent skills into your project, so an assistant writing workflows against this
framework works from how it actually behaves rather than from another
framework's habits. See :doc:`../reference/skills`.

Where to next
-------------

* :doc:`quickstart` — a model call, structured output and a workflow, in five
  minutes.
* :doc:`../guides/concepts` — the vocabulary, if LLMs are new to you.
