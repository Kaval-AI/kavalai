Model providers
===============

Every model in Kaval.AI is named with a single string, ``provider/model`` — in
Python (``make_client("openai/gpt-5.6-luna")``), as a workflow's ``llm_model``,
in a YAML node, and in the ``model`` argument of a RAG service. The part before
the first ``/`` selects the client; everything after it is the provider's own
model name, passed through untouched. ``fastembed/BAAI/bge-small-en-v1.5`` is
therefore provider ``fastembed``, model ``BAAI/bge-small-en-v1.5``.

This page lists the providers that ship in the box and shows how to find out
which models each one currently offers — because Kaval.AI deliberately keeps
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
     - none — runs in the page
     - `WebLLM prebuilt models
       <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts>`_

The client accepts ``api_key=`` (or ``host=``) directly, which wins over the
environment. ``openai/`` also takes ``base_url=``, which is what makes it serve
any OpenAI-compatible endpoint — see :doc:`../tutorials/llm_clients`.

Output caps and truncation
--------------------------

:class:`~kavalai.LlmClientParameters` names each setting once, and each client
sends it under its provider's own name. A parameter left at ``None`` is not
sent, so the provider's default applies.

.. list-table::
   :header-rows: 1
   :widths: 14 30 26 30

   * - Prefix
     - ``max_output_tokens`` sent as
     - ``reasoning_effort`` sent as
     - Truncation reported as
   * - ``openai/``
     - ``max_output_tokens``
     - ``reasoning.effort``
     - ``status: incomplete``, reason ``max_output_tokens``
   * - ``gemini/``
     - ``max_output_tokens``
     - ``thinking_config.thinking_level``
     - ``finish_reason: MAX_TOKENS``
   * - ``anthropic/``
     - ``max_tokens``; ``64000`` when unset, since the API requires a value
     - ``output_config.effort``
     - ``stop_reason: max_tokens``
   * - ``ollama/``
     - ``options.num_predict``
     - ``think``; ``"none"`` sends ``false``
     - ``done_reason: length``
   * - ``browser/``
     - ``max_tokens`` in the bridge request
     - not sent
     - ``finish_reason: length``, if the bridge reports it

On reasoning models the cap covers the reasoning tokens as well as the answer,
so a small cap can be spent before any visible text is produced.

A call that reaches the cap raises
:class:`~kavalai.llm_clients.base_client.OutputTruncatedError` instead of
returning what was generated. For structured output that text is JSON cut off
mid-value, which either fails to parse or, repaired leniently, validates into a
model with content missing; neither is an answer. The error is not retried —
the same cap cuts the same answer and the retry is billed again — and the
truncated call is still recorded as a model call with its token counts and the
partial output. A streaming consumer receives the partial chunks that arrived,
then the error, and no ``complete`` chunk:

.. code-block:: python

   from kavalai import LlmClientParameters, make_client
   from kavalai.llm_clients.base_client import OutputTruncatedError

   client = make_client(
       "anthropic/claude-sonnet-5",
       LlmClientParameters(max_output_tokens=40),
   )
   try:
       await client.prompt("Write a 300-word story about a lighthouse keeper.")
   except OutputTruncatedError as error:
       print(error)

.. code-block:: text

   The output of 'anthropic/claude-sonnet-5' was cut off at the output cap of
   40 tokens after 40 output tokens (stop reason 'max_tokens'). The partial
   output is not returned. Raise max_output_tokens (LlmClientParameters, or
   llm_kwargs in a workflow) or ask for a shorter answer.

``KAVALAI_LLM_MAX_OUTPUT_TOKENS`` sets the cap for every call the agent server
makes — see :doc:`config`.

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
     - none — local ONNX, downloaded once
     - `FastEmbed supported models
       <https://qdrant.github.io/fastembed/examples/Supported_Models/>`_
   * - ``browser/``
     - :class:`~kavalai.llm_clients.embeddings.BrowserEmbeddingClient`
     - none — runs in the page
     - `WebLLM prebuilt models
       <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts>`_

Anthropic publishes no embeddings endpoint, so ``anthropic/`` is chat-only.
Nothing requires the two halves of a RAG pipeline to come from one vendor:
answer with ``anthropic/claude-sonnet-5`` and index with
``fastembed/BAAI/bge-small-en-v1.5`` if that is the combination you want.

RAG services
------------

A RAG service is registered under a plain name — there is no
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
       <https://github.com/sqliteai/sqlite-vector>`_, with the same registry
       and one table per collection

Both keep the storage model described in :doc:`../guides/data_model`, so an
index can be browsed in the backoffice whichever database holds it.
:func:`~kavalai.rag.rag_service_from_uri` picks the service from a database
URI — ``postgresql://`` or ``sqlite:///path`` — which is what the
``ragindex`` example and the backoffice do.

Both are registered bare, with nothing bound, so ``make_rag_service("sqlite")``
still needs whatever the backend requires. What a workflow actually names is a
*configured* service, registered under the name its ``rag_query`` nodes use —
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
— see :doc:`../guides/workflows`.

:class:`~kavalai.rag.postgres.PostgresRagService` takes the session maker and
the model positionally; ``normalizer``, ``schema``, ``provision``,
``stats_receiver`` and ``vector_type`` are keyword-only, and
:class:`~kavalai.rag.sqllite.SqliteRagService` takes the last three as
keywords too. Neither service writes statistics of its own: each embedding
call is reported to the ``stats_receiver`` passed to the call, or else to the
one given to the constructor, and to nothing when neither is set.

**The collection's model is the one used.** Each collection records in
``rag_collections`` the embedding model it was created with, and a service
embeds every query and every new batch for that collection with the recorded
model. Its own ``model`` names the model for the collections it *creates*. Two
models of the same dimension produce vectors that are not comparable, and a
query embedded with the wrong one returns plausible neighbours without any
error; taking the model from the registry rules that mistake out. One
embedding client is kept per model, so one service serves collections of
different models, and a service built with ``model=None`` queries and indexes
existing collections and refuses only to create one:

.. code-block:: python

   from kavalai.rag import SqliteRagService

   rag = SqliteRagService(
       "handbook.db", model="fastembed/BAAI/bge-small-en-v1.5"
   )
   await rag.index(
       "The library is open on Tuesdays and Fridays.",
       collection_name="handbook",
   )

   other = SqliteRagService(
       "handbook.db", model="fastembed/snowflake/snowflake-arctic-embed-xs"
   )
   hit = (await other.query(
       "When is the library open?", collection_name="handbook"
   ))[0]
   print(hit.model)

.. code-block:: text

   fastembed/BAAI/bge-small-en-v1.5

Where the two disagree, as here, the service logs a warning — ``RAG collection
'handbook' was built with fastembed/BAAI/bge-small-en-v1.5; it is embedded
with that model, not with this service's
fastembed/snowflake/snowflake-arctic-embed-xs.`` A registry entry is cached
for the life of the service, so a collection that another process drops or
recreates is not noticed until the service is recreated.

**Provisioning.** With ``provision=True``, the default, a service creates what
it needs on the first write: the registry, a collection's table and indexes
and, on PostgreSQL, the ``vector`` extension. With ``provision=False`` it
issues no DDL. A database without a registry then reads as empty —
``list_collections()`` returns ``[]`` and a query returns no hits — and
indexing into a collection that does not exist raises instead of creating it.
``ensure_registry()`` and ``create_collection()`` issue DDL whatever
``provision`` says; they are the calls an owner-privileged job makes, which
:doc:`../deploy/index` describes. Read paths create nothing under either
setting: a read that performs DDL fails under a role without the privilege
and surprises one that has it.

**Half-precision vectors on PostgreSQL.** ``vector_type="halfvec"`` stores the
embeddings of new collections as 16-bit floats. That halves their memory, at a
small cost in the precision of the scores, and raises pgvector's HNSW limit
from 2,000 dimensions to 4,000, so a 3,072-dimension model such as
``openai/text-embedding-3-large`` can be indexed only this way:

.. code-block:: python

   from kavalai.rag import PostgresRagService

   rag = PostgresRagService.from_uri(
       "postgresql://user:pass@db/kavalai",
       "openai/text-embedding-3-large",
       schema="rag",
   )
   try:
       await rag.create_collection("large", 3072)
   except Exception as error:
       print(error.orig)

   await rag.create_collection("large", 3072, vector_type="halfvec")
   print([c["embedding_size"] for c in await rag.list_collections()])

.. code-block:: text

   <class 'asyncpg.exceptions.ProgramLimitExceededError'>: column cannot
   have more than 2000 dimensions for hnsw index
   [3072]

``halfvec`` needs pgvector 0.7 or later; on an older one creating such a
collection raises ``ValueError``. An existing collection keeps the type it was
created with, which the service reads from the table's column rather than from
its own setting. :class:`~kavalai.rag.sqllite.SqliteRagService` stores 32-bit
floats only and refuses any other ``vector_type`` with ``ValueError``.

**Filtered queries on PostgreSQL.** An HNSW scan visits a bounded set of
nearest candidates (``hnsw.ef_search``, 40 by default) and applies the
``WHERE`` clause to those, so a query restricted by ``source_ids`` whose
matching rows lie away from the query vector can return fewer than ``top_k``
rows, or none. A very selective filter does not suffer from this: PostgreSQL
then reads the matching rows through the ``source_id`` index and sorts them
exactly. The shortfall arises when the filter matches too many rows for that
plan — a site's pages in a collection shared by a few sites, for instance. On
pgvector 0.8 or later the service runs a filtered query with
iterative index scans — ``SET LOCAL hnsw.iterative_scan = relaxed_order``,
which lasts for that query's transaction only — and the index keeps scanning
until enough rows pass the filter or pgvector's ``hnsw.max_scan_tuples`` limit
is reached. ``batch_query_with_join`` does the same when it is given
``additional_where``. On an older pgvector the setting is not issued and the
shortfall remains.

**One table per collection, for now.** Each collection has its own table, typed
vector column and index, which keeps a scan inside one collection and makes
dropping a collection a ``DROP TABLE``. A shared layout — one hash-partitioned
table per dimension — is the alternative once a database holds more than
2,000 collections or more than 5 million vectors. It is deliberately not
implemented: a second layout would double both the conformance suite every
backend runs and the upgrade paths between collection schema versions, and
the deployments Kaval.AI serves are below that threshold. The decision is to be
revisited when one reaches it.

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
the work — a
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
own message — ``The requested model 'gpt-nonexistent-9' does not exist`` from
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

Strip the ``models/`` prefix or leave it — the client accepts both, so
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
raises on the first ``index`` or ``query`` call rather than at construction —
so validate the string early if the service is built at start-up. The model
may also be omitted: a service built with ``model=None`` embeds each existing
collection with the model that collection records, so it can query and index
it, and refuses only to create a collection — which is how the backoffice
opens an index it did not build. Four practical points then matter more than
a model's benchmark score:

**Index and query with the same model.** Vectors from two different models are
not comparable, and a search across them returns plausible neighbours rather
than failing. Both services therefore take an existing collection's model from
the registry instead of from their own ``model`` (see `RAG services`_ above),
and check the dimension recorded beside it on every write::

   ValueError: Collection 'handbook' stores 384-dimensional embeddings; got
   1536.

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
alongside the dimension, the entry count and the schema version — so an
existing database can be asked which model it was built with before anything
queries it.
Changing embedding model means re-indexing into a new collection, not editing
the old one.

**Dimension is a storage and latency decision.** 384 dimensions
(``bge-small``) against 3072 (``text-embedding-3-large``) is eight times the
vector storage and index size for a modest retrieval gain on short factual
text. Start small and measure with an evaluation suite
(:doc:`../guides/evaluation`) before paying for width.

**A shorter vector can come from the same model.** ``text-embedding-3-*`` and
``gemini-embedding-001`` return shortened vectors on request, and
:class:`~kavalai.llm_clients.embeddings.OpenAIEmbeddingClient` and
:class:`~kavalai.llm_clients.embeddings.GeminiEmbeddingClient` take
``dimensions=`` for it. Bind it at registration under a provider name of its
own: the model string is what a collection records, so the reduced model must
not share a name with the full one, whose vectors it cannot be compared with.

.. code-block:: python

   from kavalai import make_embedding_client, register_embedding_provider
   from kavalai.llm_clients.embeddings import OpenAIEmbeddingClient

   register_embedding_provider(
       "openai-512", OpenAIEmbeddingClient, dimensions=512
   )
   client = make_embedding_client("openai-512/text-embedding-3-small")
   vectors, _ = await client.compute_embeddings(["How deep is Lake Miller?"])
   print(len(vectors[0]))

.. code-block:: text

   512

**Local or hosted is a deployment decision.** ``fastembed`` needs no API key
and no network after the first download, which makes it the reproducible choice
for tests, CI and eval fixtures; ``openai`` and ``gemini`` embeddings need a key
and a round trip per batch; ``browser`` keeps the text on the device. All four
implement the same interface, so the choice is one string.

The embedding model is always an argument — to ``make_embedding_client``, to a
RAG service, to the ``ragindex`` example's ``--model`` flag. Library code reads
no environment variables of its own — see :doc:`config`.

Adding a provider of your own
-----------------------------

The tables above are a starting set, not a closed one. Register a name and it
becomes indistinguishable from a built-in — usable in ``make_client``, in a
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

The worked examples — an OpenAI-compatible endpoint, a provider with its own
wire protocol, and an embedding client that calls nothing at all — are in
:doc:`../tutorials/llm_clients` and :doc:`../cookbook/index`.
