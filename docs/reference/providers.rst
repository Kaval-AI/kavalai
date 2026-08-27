Model providers
===============

Every model in Kaval.AI is named with a single string, ``provider/model`` --- in
Python (``make_client("openai/gpt-5.4-mini")``), as a workflow's ``llm_model``,
in a YAML node, and in the ``model`` argument of a RAG service. The part before
the first ``/`` selects the client; everything after it is the provider's own
model name, passed through untouched. ``fastembed/BAAI/bge-small-en-v1.5`` is
therefore provider ``fastembed``, model ``BAAI/bge-small-en-v1.5``.

This page lists the providers that ship in the box and shows how to find out
which models each one currently offers --- because Kaval.AI deliberately keeps
no list of model names of its own.

Chat models
-----------

.. list-table::
   :header-rows: 1
   :widths: 14 26 24 36

   * - Prefix
     - Client
     - Credential
     - Model names published at
   * - ``openai/``
     - :class:`~kavalai.llm_clients.openai_client.OpenAIClient`
     - ``OPENAI_API_KEY``
     - `OpenAI models <https://platform.openai.com/docs/models>`_
   * - ``gemini/``
     - :class:`~kavalai.llm_clients.gemini_client.GeminiClient`
     - ``GEMINI_API_KEY``
     - `Gemini models <https://ai.google.dev/gemini-api/docs/models>`_
   * - ``anthropic/``
     - :class:`~kavalai.llm_clients.anthropic_client.AnthropicClient`
     - ``ANTHROPIC_API_KEY``
     - `Claude models
       <https://docs.claude.com/en/docs/about-claude/models/overview>`_
   * - ``ollama/``
     - :class:`~kavalai.llm_clients.ollama_client.OllamaClient`
     - ``OLLAMA_HOST`` (default ``http://localhost:11434``)
     - `Ollama library <https://ollama.com/library>`_
   * - ``browser/``
     - :class:`~kavalai.llm_clients.browser_client.BrowserLLMClient`
     - none --- runs in the page
     - `WebLLM prebuilt models
       <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts>`_

The client accepts ``api_key=`` (or ``host=``) directly, which wins over the
environment. ``openai/`` also takes ``base_url=``, which is what makes it serve
any OpenAI-compatible endpoint --- see :doc:`../tutorials/llm_clients`.

Embedding models
----------------

Embeddings use the same string through a separate registry: the lookup behind
:func:`~kavalai.llm_clients.embeddings.make_embedding_client` and behind the
``model`` argument of
:class:`~kavalai.rag.postgres.PostgresRagService` and
:class:`~kavalai.rag.sqllite.SqliteRagService`.

.. list-table::
   :header-rows: 1
   :widths: 14 26 24 36

   * - Prefix
     - Client
     - Credential
     - Model names published at
   * - ``openai/``
     - :class:`~kavalai.llm_clients.embeddings.OpenAIEmbeddingClient`
     - ``OPENAI_API_KEY``
     - `OpenAI embeddings
       <https://platform.openai.com/docs/guides/embeddings>`_
   * - ``gemini/``
     - :class:`~kavalai.llm_clients.embeddings.GeminiEmbeddingClient`
     - ``GEMINI_API_KEY``
     - `Gemini embeddings
       <https://ai.google.dev/gemini-api/docs/embeddings>`_
   * - ``ollama/``
     - :class:`~kavalai.llm_clients.embeddings.OllamaEmbeddingClient`
     - ``OLLAMA_HOST``
     - `Ollama embedding models
       <https://ollama.com/search?c=embedding>`_
   * - ``fastembed/``
     - :class:`~kavalai.llm_clients.embeddings.FastEmbedClient`
     - none --- local ONNX, downloaded once
     - `FastEmbed supported models
       <https://qdrant.github.io/fastembed/examples/Supported_Models/>`_
   * - ``browser/``
     - :class:`~kavalai.llm_clients.embeddings.BrowserEmbeddingClient`
     - none --- runs in the page
     - `WebLLM prebuilt models
       <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts>`_

Anthropic publishes no embeddings endpoint, so ``anthropic/`` is chat-only.
Nothing requires the two halves of a RAG pipeline to come from one vendor:
answer with ``anthropic/claude-sonnet-5`` and index with
``fastembed/BAAI/bge-small-en-v1.5`` if that is the combination you want.

RAG services
------------

A RAG service is registered under a plain name --- there is no
``provider/model`` split, because the name is what a ``rag_query`` node's
``service`` field refers to.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Name
     - Class
     - Store
   * - ``postgres``
     - :class:`~kavalai.rag.postgres.PostgresRagService`
     - PostgreSQL with `pgvector <https://github.com/pgvector/pgvector>`_; one
       table and one HNSW index per collection
   * - ``sqlite``
     - :class:`~kavalai.rag.sqllite.SqliteRagService`
     - A single file through `sqlite-vector
       <https://github.com/sqliteai/sqlite-vector>`_, readable in the browser

Both are registered bare, with nothing bound, so ``make_rag_service("sqlite")``
still needs whatever the backend requires. What a workflow actually names is a
*configured* service, registered under the name its ``rag_query`` nodes use ---
usually ``default``, which is what a node resolves to when neither it nor the
graph says otherwise:

.. code-block:: python

   from kavalai import register_rag_service
   from kavalai.rag import SqliteRagService

   register_rag_service(
       "default", SqliteRagService,
       filename="handbook.db",
       model="fastembed/BAAI/bge-small-en-v1.5",
   )

The workflow document then mentions neither a filename nor a connection string
--- see ``examples/green_village/eval_setup.py`` and :doc:`../guides/workflows`.

Which names are registered right now
------------------------------------

The registries answer for themselves, and they answer for your own
registrations too:

.. code-block:: python

   from kavalai import (
       registered_embedding_providers,
       registered_llm_providers,
       registered_rag_services,
   )

   print(registered_llm_providers())
   print(registered_embedding_providers())
   print(registered_rag_services())

.. code-block:: text

   ['anthropic', 'browser', 'gemini', 'ollama', 'openai']
   ['browser', 'fastembed', 'gemini', 'ollama', 'openai']
   ['postgres', 'sqlite']

Note what these are: **provider names, not model catalogues**. Anything added
with :func:`~kavalai.llm_clients.registry.register_llm_provider`,
:func:`~kavalai.llm_clients.registry.register_embedding_provider` or
:func:`~kavalai.llm_clients.registry.register_rag_service` appears in the same
list, which is why the check is worth running from the process that will do
the work --- a
provider module that was never imported is exactly what is missing from the
output.

Why there is no built-in list of models
---------------------------------------

The provider half of the string is validated; the model half is not. Providers
ship models faster than a library can vendor a list of them, and a stale
allow-list refuses models that work.

The two failures therefore look different. An unknown **provider** fails before
any request is made, and the message names every provider that does exist:

.. code-block:: python

   from kavalai import make_embedding_client

   make_embedding_client("cohere/embed-v4")

.. code-block:: text

   kavalai.llm_clients.registry.RegistryError: Unsupported embedding provider
   'cohere/embed-v4': tried 'cohere/embed-v4' and 'cohere'. Registered:
   browser, fastembed, gemini, ollama, openai. Add your own with
   register_embedding_provider(), which must run before the workflow is loaded.

An unknown **model** reaches the provider and fails there, with the provider's
own message --- ``The requested model 'gpt-nonexistent-9' does not exist`` from
OpenAI, a 404 from Gemini, a "model not found" from Ollama. Read that message
literally: the provider does not have that model, and Kaval.AI is not what
rejected it.

Asking a provider what it offers
--------------------------------

Every provider SDK can enumerate its own models, and the SDK in question is the
one ``kavalai[common]`` already installed for that provider. These are the
calls behind the catalogue links in the tables above, and the answer they give
is the provider's own, current one.

OpenAI
~~~~~~

.. code-block:: python

   from openai import OpenAI

   models = [model.id for model in OpenAI().models.list()]
   print(sorted(name for name in models if "embedding" in name))

.. code-block:: text

   ['text-embedding-3-large', 'text-embedding-3-small',
    'text-embedding-ada-002']

Gemini
~~~~~~

``supported_actions`` separates the chat models (``generateContent``) from the
embedding ones (``embedContent``):

.. code-block:: python

   from google import genai

   client = genai.Client()
   for model in client.models.list():
       if "embedContent" in (model.supported_actions or []):
           print(model.name)

.. code-block:: text

   models/gemini-embedding-001
   models/gemini-embedding-2-preview
   models/gemini-embedding-2

Strip the ``models/`` prefix or leave it --- the client accepts both, so
``gemini/gemini-embedding-001`` is the id to use.

Anthropic
~~~~~~~~~

.. code-block:: python

   import anthropic

   client = anthropic.Anthropic()
   print([model.id for model in client.models.list(limit=5).data])

.. code-block:: text

   ['claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'claude-opus-4-8',
    'claude-opus-4-7']

Ollama
~~~~~~

``ollama list`` (or ``ollama.list()``) reports what the host has **pulled**,
which is the set you can actually use; `ollama.com/library
<https://ollama.com/library>`_ is the set you can pull. An embedding model has
to be pulled like any other:

.. code-block:: bash

   ollama pull nomic-embed-text
   ollama list

FastEmbed
~~~~~~~~~

FastEmbed publishes the vector dimension and the download size alongside each
name, which is exactly what you need when sizing a collection:

.. code-block:: python

   from fastembed import TextEmbedding

   for model in TextEmbedding.list_supported_models()[:5]:
       print(f"{model['model']:<32} {model['dim']:>5}  "
             f"{model['size_in_GB']} GB")

.. code-block:: text

   BAAI/bge-base-en                   768  0.42 GB
   BAAI/bge-base-en-v1.5              768  0.21 GB
   BAAI/bge-large-en-v1.5            1024  1.2 GB
   BAAI/bge-small-en                  384  0.13 GB
   BAAI/bge-small-en-v1.5             384  0.067 GB

Thirty models are listed at the time of writing; the model file is downloaded
from the Hugging Face Hub on first use and cached in ``FASTEMBED_CACHE_DIR``.

In the browser
~~~~~~~~~~~~~~

``browser/`` model ids are WebLLM build ids, taken from the prebuilt list in
`web-llm's config.ts
<https://github.com/mlc-ai/web-llm/blob/main/src/config.ts>`_ and passed to the
page's bridge verbatim. Prefer the ``q4f32`` builds on GPUs without FP16
shaders. :doc:`../tutorials/run_in_browser` names the ones the playground
carries.

Choosing an embedding model for a RAG service
---------------------------------------------

A RAG service is given its embedding model by name, and builds the client
lazily:

.. code-block:: python

   from kavalai.rag import SqliteRagService

   rag = SqliteRagService(
       "handbook.db", model="fastembed/BAAI/bge-small-en-v1.5"
   )

Because the client is built on first use, a name the registry cannot resolve
raises on the first ``index`` or ``query`` call rather than at construction ---
so validate the string early if the service is built at start-up. Four
practical points then matter more than a model's benchmark score:

**Index and query with the same model.** Vectors from two different models are
not comparable, and the search returns nonsense rather than failing. Under
PostgreSQL a collection's vector column is sized on the first batch, so a model
of a *different* dimension at least fails loudly::

   ValueError: Collection 'handbook' stores 384-dimensional embeddings; got
   1536.

A model of the same dimension does not, which is the case worth being careful
about.

**Every hit says what produced it.** The model and dimension travel with the
data, so an index of unknown provenance can identify itself:

.. code-block:: python

   hit = (await rag.query(
       "how many residents?", top_k=1, collection_name="handbook"
   ))[0]
   print("model     :", hit.model)
   print("dimensions:", hit.embedding_size)

.. code-block:: text

   model     : fastembed/BAAI/bge-small-en-v1.5
   dimensions: 384

:class:`~kavalai.rag.postgres.PostgresRagService` additionally records the
model per collection, and ``await rag.list_collections()`` returns it
alongside the dimension, the entry count and the schema version --- so an
existing database can be asked which model it was built with before anything
queries it.
Changing embedding model means re-indexing into a new collection, not editing
the old one.

**Dimension is a storage and latency decision.** 384 dimensions
(``bge-small``) against 3072 (``text-embedding-3-large``) is eight times the
vector storage and index size for a modest retrieval gain on short factual
text. Start small and measure with an evaluation suite
(:doc:`../guides/evaluation`) before paying for width.

**Local or hosted is a deployment decision.** ``fastembed`` needs no API key
and no network after the first download, which makes it the reproducible choice
for tests, CI and eval fixtures; ``openai`` and ``gemini`` embeddings need a key
and a round trip per batch; ``browser`` keeps the text on the device. All four
implement the same interface, so the choice is one string.

``KAVALAI_DEFAULT_EMBEDDING_MODEL`` supplies the model for
``python -m kavalai.tools.index_csv`` when ``--model`` is omitted. Library code
reads no environment variables of its own --- see :doc:`config`.

Adding a provider of your own
-----------------------------

The tables above are a starting set, not a closed one. Register a name and it
becomes indistinguishable from a built-in --- usable in ``make_client``, in a
YAML node, and as a workflow default:

.. code-block:: python

   import os

   from kavalai import OpenAIClient, register_llm_provider

   register_llm_provider(
       "deepseek", OpenAIClient,
       base_url="https://api.deepseek.com",
       api_key=os.environ["DEEPSEEK_API_KEY"],
   )

Registration must run **before** the workflow is loaded, since model names and
``rag_query`` services are resolved when the graph is parsed. For the agent
server, ``KAVALAI_PROVIDER_MODULES`` names the modules to import first.

The worked examples --- an OpenAI-compatible endpoint, a provider with its own
wire protocol, and an embedding client that calls nothing at all --- are in
:doc:`../tutorials/llm_clients` and :doc:`../cookbook/index`.
