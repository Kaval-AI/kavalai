LLM Clients API
===============

:mod:`kavalai.llm_clients` provides a unified, observable interface over LLM and
embedding providers. Every call returns a :class:`~kavalai.ModelCallStat` with
token usage and timing, and structured output is validated against a Pydantic
``response_model``.

Providers
---------

:func:`~kavalai.make_client` builds a client from a ``"provider/model"`` id and
reads the matching credential from the environment:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Prefix
     - Class
     - Credential
   * - ``openai/``
     - :class:`~kavalai.OpenAIClient`
     - ``OPENAI_API_KEY``
   * - ``gemini/``
     - :class:`~kavalai.GeminiClient`
     - ``GEMINI_API_KEY``
   * - ``anthropic/``
     - :class:`~kavalai.AnthropicClient`
     - ``ANTHROPIC_API_KEY``
   * - ``ollama/``
     - :class:`~kavalai.OllamaClient`
     - ``OLLAMA_HOST`` (no key)
   * - ``browser/``
     - :class:`~kavalai.BrowserLLMClient`
     - none — runs client-side over WebGPU

:func:`~kavalai.make_embedding_client` mirrors it for embeddings, adding the
``fastembed/`` prefix for local, key-free embedding.

The ``browser/`` provider needs a WebGPU-capable page rather than a Python
process; see :doc:`/tutorials/run_in_browser` for what it can do and how models
are downloaded and cached.

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
