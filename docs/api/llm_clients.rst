LLM Clients API
===============

:mod:`kavalai.llm_clients` provides a unified, observable interface over LLM and
embedding providers. Every call returns a :class:`~kavalai.ModelCallStat` with
token usage and timing, and structured output is validated against a Pydantic
``response_model``.

Run in the browser
------------------

The ``browser/`` provider runs a model entirely client-side over WebGPU — no API
key, no server, no CORS. The same :func:`~kavalai.make_client` /
:func:`~kavalai.make_embedding_client` factories you use on the server return a
:class:`~kavalai.BrowserLLMClient` / :class:`~kavalai.BrowserEmbeddingClient`,
so your code is identical apart from the ``provider/model`` string. The two
snippets below have a **Run in browser ▶** button (the model id comes from the
panel's dropdown):

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_client

   client = make_client(f"browser/{KAVAL_BROWSER_MODEL}")
   colours = await client.prompt("Name the three primary colours, comma-separated.")
   print(colours)

Embeddings work the same way. Embedding models are distinct from chat models;
``KAVAL_BROWSER_EMBED_MODEL`` is a small, full-precision Snowflake Arctic model:

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_embedding_client

   client = make_embedding_client(f"browser/{KAVAL_BROWSER_EMBED_MODEL}")
   texts = [
       "Tallinn is the capital of Estonia.",
       "Estonia's capital city is Tallinn.",
       "I had pasta for dinner last night.",
   ]
   vectors, stats = await client.compute_embeddings(texts, normalize=True)
   print(f"{len(vectors)} vectors of dimension {len(vectors[0])}")

   # Vectors are L2-normalised, so cosine similarity is just their dot product.
   def similarity(a, b):
       return sum(x * y for x, y in zip(a, b))

   print(f"sim(0, 1) = {similarity(vectors[0], vectors[1]):.3f}  # same meaning")
   print(f"sim(0, 2) = {similarity(vectors[0], vectors[2]):.3f}  # unrelated")

.. note::

   ``browser/`` models need a WebGPU-capable browser (recent Chrome/Edge, or
   Firefox with ``dom.webgpu.enabled``). The model downloads on first use and is
   cached by the browser. Outside the browser, use an ``openai/``, ``gemini/``,
   ``anthropic/`` or ``ollama/`` model instead.

Base client and models
----------------------

.. automodule:: kavalai.llm_clients.base_client

Provider clients
----------------

.. automodule:: kavalai.llm_clients.openai_client

.. automodule:: kavalai.llm_clients.gemini_client

.. automodule:: kavalai.llm_clients.anthropic_client

.. automodule:: kavalai.llm_clients.ollama_client

.. automodule:: kavalai.llm_clients.browser_client

Embeddings
----------

.. automodule:: kavalai.llm_clients.embeddings

Streaming
---------

.. automodule:: kavalai.llm_clients.streamer
