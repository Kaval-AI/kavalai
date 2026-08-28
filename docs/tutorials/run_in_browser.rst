Running in the browser
======================

Kaval.AI can run **entirely in the browser** — the workflow engine, the LLM and
the embeddings all execute on the user's device, with no API key, no server and
no CORS. Python runs through `Pyodide <https://pyodide.org>`_ and models run over
WebGPU through a `WebLLM <https://github.com/mlc-ai/web-llm>`_ bridge. The
**Run in browser ▶** buttons on this page use exactly this setup, as does the
chatbot below.

A chatbot running in this page
------------------------------

Before the details, here is the artefact itself: a complete Kaval.AI workflow — a
one-node chatbot with structured output and memory — running entirely in your
browser. No server, no API key, nothing sent anywhere.

The workflow on top is what the chat below it runs: press **Run ▶** to build it,
then talk to it — edit the code and run again to change the bot. The first load
downloads the model (or takes it from the browser cache), so give it a moment.

.. raw:: html
   :file: ../_includes/chatbot-demo.html

Because the workflow returns a structured ``Reply`` (an ``agent_response`` plus
a list of ``choices``), the chat can render those choices as quick-reply chips —
a concrete reason to ask for typed output rather than prose. The conversation is
kept in an in-browser SQLite database, which is what lets each turn see the last
one.

Why run in the browser
----------------------

The browser is an unexpectedly suitable place to *ship* an agent where one would
rather not run infrastructure:

* **No infrastructure.** There is no server to deploy, scale or pay for, and no
  provider account to manage — the whole stack runs in the page. For a demo, an
  internal tool, a docs example or a small app, that can be the entire backend.
* **Privacy by default.** Because inference is local, the user's text never
  leaves their machine — useful for sensitive data.
* **Offline after the first load.** Once the model is cached, it keeps working
  with no network.

The trade-off is capacity: you are limited to **small** open models and need a
WebGPU-capable browser (recent Chrome/Edge, or Firefox with
``dom.webgpu.enabled``). For heavy reasoning you will still want a hosted
provider (see :doc:`llm_clients`) — but for much agentic interface work, in-browser
is enough.

To embed the playground on your own page (or self-host it), see the
``webwidget/`` folder in the repository; it is the single source of the widget
used here.

Models, and how they download (WebLLM)
--------------------------------------

A ``"browser/<model-id>"`` provider id routes inference to the page's WebLLM
bridge. The first time a model is used it is **downloaded** (hundreds of MB to a
few GB) and then **cached by the browser**, so later runs start instantly and
work offline. The model id maps to a WebLLM build:

* ``Llama-3.2-1B-Instruct-q4f32_1-MLC`` (~1.1 GB) and the 3B variant (~2.9 GB)
* ``Qwen2.5-0.5B-Instruct-q4f32_1-MLC`` (~0.6 GB) and the 1.5B variant
* an embedding model, ``snowflake-arctic-embed-s-q0f32-MLC-b4``

The ``q4f32`` builds run on GPUs **without** FP16 shaders (e.g. older cards);
``q4f16`` builds are smaller/faster but need an FP16-capable GPU. In the
playground you pick the chat model from the toolbar dropdown — it is exposed to
your code as ``KAVAL_BROWSER_MODEL`` (and the embedding model as
``KAVAL_BROWSER_EMBED_MODEL``) — so you never hardcode an id:

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_client

   client = make_client(f"browser/{KAVAL_BROWSER_MODEL}")
   print(f"Loading {KAVAL_BROWSER_MODEL} "
         "(the first run downloads it; afterwards it is cached)…")
   print(await client.prompt("Say hello in one short sentence."))

Embeddings in the browser
-------------------------

Embeddings work the same way through ``make_embedding_client``. Embedding models
are small and distinct from chat models; ``snowflake-arctic-embed-s`` runs even
on GPUs without FP16. ``compute_embeddings`` returns ``(vectors, stats)``;
``normalize=True`` yields unit vectors, so cosine similarity reduces to a dot
product:

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_embedding_client

   embedder = make_embedding_client(f"browser/{KAVAL_BROWSER_EMBED_MODEL}")
   texts = ["Hello darkness, my old friend", "We will rock you"]
   vectors, _ = await embedder.compute_embeddings(texts, normalize=True)
   print(f"{len(vectors)} vectors of dimension {len(vectors[0])}")

A RAG you can query in the browser
----------------------------------

In production, retrieval-augmented generation uses :doc:`rag` backed by Postgres
+ pgvector. The browser has no pgvector — but for a **pre-built, read-only**
corpus it is unnecessary: embed the documents, embed the query, and rank by
cosine similarity in a few lines of Python. Then hand the best matches to the
model. The whole loop — retrieve **and** generate — runs in the page:

.. code-block:: python
   :class: run-in-browser

   from kavalai import make_client, make_embedding_client

   # A small lyric corpus. A real corpus is pre-built offline; see below.
   lyrics = [
       "Is this the real life? Is this just fantasy?",
       "We will, we will rock you",
       "Hello darkness, my old friend, I've come to talk with you again",
   ]

   embedder = make_embedding_client(f"browser/{KAVAL_BROWSER_EMBED_MODEL}")
   doc_vectors, _ = await embedder.compute_embeddings(lyrics, normalize=True)

   # Embed the question with the *same* model and rank by cosine similarity.
   question = "Which song is about silence?"
   (q_vector,), _ = await embedder.compute_embeddings(
       [question], normalize=True
   )

   def cosine(a, b):
       return sum(x * y for x, y in zip(a, b))  # unit vectors -> dot product

   ranked = sorted(
       zip(lyrics, doc_vectors),
       key=lambda d: cosine(q_vector, d[1]),
       reverse=True,
   )
   top = [line for line, _ in ranked[:2]]
   print("Retrieved:", top)

   # Generate a grounded answer from the retrieved lines.
   llm = make_client(f"browser/{KAVAL_BROWSER_MODEL}")
   context = "\n".join(top)
   print(await llm.prompt(f"Using only these lyrics:\n{context}\n\n{question}"))

Pre-building and shipping a RAG
-------------------------------

Embedding a large corpus on every page load is wasteful, so **pre-build the
index offline and ship it** alongside the page. The one rule: build and query
with the **same embedding model** so the vectors are comparable. The browser
uses ``snowflake-arctic-embed-s``; offline, ``fastembed`` runs the same model
(``pip install "kavalai[common]"``).

Run this once, locally, over the song lyrics in ``local_data/``:

.. code-block:: python

   import asyncio
   import csv
   import json
   from itertools import islice

   from kavalai import make_embedding_client

   # Same model as the browser's q0f32 build, so the vectors line up.
   embedder = make_embedding_client("fastembed/snowflake/snowflake-arctic-embed-s")


   async def build():
       rows = list(islice(csv.DictReader(open("local_data/song_lyrics.csv")), 500))
       texts = [r["lyrics"][:2000] for r in rows]
       vectors, _ = await embedder.compute_embeddings(texts, normalize=True)
       index = [
           {"title": r["title"], "artist": r["artist"], "embedding": vec}
           for r, vec in zip(rows, vectors)
       ]
       json.dump(index, open("lyrics_index.json", "w"))
       print(f"Indexed {len(index)} songs -> lyrics_index.json")


   asyncio.run(build())

.. code-block:: text

   Indexed 500 songs -> lyrics_index.json

Then, in the browser, **fetch** the pre-built index and query it — only the small
query embedding is computed on the device:

.. code-block:: python

   import pyodide.http
   from kavalai import make_embedding_client

   # The index you shipped next to your page (same origin).
   response = await pyodide.http.pyfetch("lyrics_index.json")
   index = await response.json()

   embedder = make_embedding_client(f"browser/{KAVAL_BROWSER_EMBED_MODEL}")
   (q_vector,), _ = await embedder.compute_embeddings(
       ["heartbreak and rain"], normalize=True
   )

   def cosine(a, b):
       return sum(x * y for x, y in zip(a, b))

   ranked = sorted(
       index,
       key=lambda song: cosine(q_vector, song["embedding"]),
       reverse=True,
   )
   for song in ranked[:3]:
       print(f"{song['artist']} — {song['title']}")

That is the whole pattern: pre-compute the expensive part (document embeddings)
where you have the compute, ship a static JSON, and let the browser do only the
cheap per-query work. Combined with a ``browser/`` chat model, you have a RAG
chatbot with **no backend at all**.

Where to next
-------------

* :doc:`llm_clients` — the clients (including the ``browser/`` provider) in depth.
* :doc:`rag` — retrieval-augmented generation with Postgres + pgvector.
* :doc:`observability_storage` — sessions, runs and chat history, in the browser and out.
