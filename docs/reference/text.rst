Parsing and chunking text
=========================

:mod:`kavalai.text` turns an HTML page or a markdown document into the chunks
a RAG index stores. It uses the standard library only, so it also runs under
Pyodide, and ``import kavalai`` does not import it.

The module does three things:

- :func:`~kavalai.text.parse_html` reduces a page, in one pass, to its title,
  its visible text as **blocks** that each carry the heading path above them,
  its links and its robots directives;
- :func:`~kavalai.text.chunk_blocks` packs blocks into **chunks** of about a
  target size and never over a cap;
- :func:`~kavalai.text.chunk_markdown` reads markdown into the same blocks and
  packs them with the same function.

Fetching a page, reading PDF and detecting the language are outside its
scope. Every step is linear in the size of its input and no regular
expression is involved, so an adversarial page — tags nested fifty thousand
deep, a megabyte-long attribute, a paragraph without a full stop — costs time
in proportion to its length.

Worked example
--------------

Parse a page. The navigation and the footer are dropped, the table rows and
the paragraphs become blocks, and each block knows the headings above it:

.. code-block:: python

   from kavalai.text import chunk_blocks, parse_html

   HTML = """
   <html><head><title>Green Village Hall</title></head>
   <body>
   <nav><a href="/">Home</a> <a href="/events">Events</a></nav>
   <main>
     <h1>Booking the hall</h1>
     <p>The hall seats 120 people and can be booked by residents.</p>
     <h2>Prices</h2>
     <table>
       <tr><th>Slot</th><th>Price</th></tr>
       <tr><td>Half day</td><td>40 EUR</td></tr>
       <tr><td>Full day</td><td>70 EUR</td></tr>
     </table>
     <h2>Cancellation</h2>
     <p>Cancel at least 7 days ahead for a full refund.</p>
   </main>
   <footer>© Green Village <a href="/imprint">Imprint</a></footer>
   </body></html>
   """

   page = parse_html(HTML, base_url="https://hall.example/booking")
   print(page.title, page.noindex)
   print("\n".join(page.links))
   for block in page.blocks:
       print(f"{block.kind:9} {block.text}")
   print(page.blocks[4].heading)

.. code-block:: text

   Green Village Hall False
   https://hall.example/
   https://hall.example/events
   https://hall.example/imprint
   heading   Booking the hall
   paragraph The hall seats 120 people and can be booked by residents.
   heading   Prices
   row       Slot | Price
   row       Half day | 40 EUR
   row       Full day | 70 EUR
   heading   Cancellation
   paragraph Cancel at least 7 days ahead for a full refund.
   Booking the hall › Prices

Chunk the blocks. Each heading section becomes a chunk, prefixed with the
title and its heading path:

.. code-block:: python

   chunks = chunk_blocks(page.blocks, title=page.title)
   for chunk in chunks:
       print(f"--- chunk {chunk.position}, heading {chunk.heading!r}")
       print(chunk.text)

.. code-block:: text

   --- chunk 0, heading 'Booking the hall'
   Green Village Hall › Booking the hall

   The hall seats 120 people and can be booked by residents.
   --- chunk 1, heading 'Booking the hall › Prices'
   Green Village Hall › Booking the hall › Prices

   Slot | Price

   Half day | 40 EUR

   Full day | 70 EUR
   --- chunk 2, heading 'Booking the hall › Cancellation'
   Green Village Hall › Booking the hall › Cancellation

   Cancel at least 7 days ahead for a full refund.

Index the chunks and query them. The example embeds locally with FastEmbed,
so it needs no API key; the heading and position go into the metadata so an
answer can cite its section:

.. code-block:: python

   from kavalai import SqliteRagService

   rag = SqliteRagService(
       ":memory:", model="fastembed/snowflake/snowflake-arctic-embed-s"
   )
   await rag.index_batch(
       texts=[chunk.text for chunk in chunks],
       metadata_list=[
           {"heading": chunk.heading, "chunk": chunk.position}
           for chunk in chunks
       ],
       source_ids=["https://hall.example/booking"] * len(chunks),
       collection_name="hall",
   )
   hits = await rag.query(
       "How much is a full day?", top_k=1, collection_name="hall"
   )
   print(round(hits[0].similarity, 2), hits[0].rag_metadata["heading"])

.. code-block:: text

   0.74 Booking the hall › Prices

Parsing HTML
------------

:func:`~kavalai.text.parse_html` returns a :class:`~kavalai.text.ParsedPage`:

- ``title`` — the ``<title>``, whitespace collapsed, or the first heading when
  the page has no title;
- ``blocks`` — the visible text in document order, heading blocks included;
- ``links`` — the ``href`` of every ``<a>`` and ``<area>``, in document order
  without repeats, resolved against ``base_url`` or the page's
  ``<base href>``; an ``href`` that cannot be parsed as a URL is dropped;
- ``noindex`` and ``nofollow`` — whether a robots ``<meta>`` tag says so
  (``none`` sets both);
- ``markdown`` — a property rendering the blocks as markdown.

Links are collected from the whole document, navigation and footer included,
because the navigation is how a crawler discovers a site that has no sitemap.
Only the inert content of a ``<template>`` is excluded.

Each :class:`~kavalai.text.Block` has a ``kind``:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Kind
     - Source
     - Text
   * - ``heading``
     - ``h1`` to ``h6``
     - the heading; ``level`` is 1 to 6
   * - ``paragraph``
     - ``p``, ``div`` and other block elements
     - whitespace collapsed; a ``<br>`` is kept as a line break
   * - ``item``
     - ``li``, ``dt``, ``dd``
     - as a paragraph
   * - ``row``
     - ``tr``
     - the cells joined by ``" | "``
   * - ``code``
     - ``pre``
     - the whitespace as written
   * - ``quote``
     - ``blockquote``
     - as a paragraph

A block's ``heading`` is the path of headings above it, joined by
:data:`~kavalai.text.HEADING_SEPARATOR` (``" › "``). A heading resets every
deeper level, so an ``h2`` after an ``h3`` ends the ``h3``'s section.

The boilerplate rules are parameters, and their defaults are module
constants:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Parameter
     - Default
     - Effect
   * - ``content_tags``
     - ``main``, ``article``
     - When the page has text inside one of these, only that text is kept,
       and only the headings inside them form heading paths. An element
       whose ``role`` names one counts as it.
   * - ``drop_tags``
     - ``script``, ``style``, ``noscript``, ``template``, ``svg``,
       ``iframe``, ``canvas``, ``object``, ``nav``, ``dialog``, ``button``,
       ``select``, ``textarea``
     - Their text is dropped wherever they occur.
   * - ``chrome_tags``
     - ``header``, ``footer``, ``aside``
     - Their text is dropped outside a content element and kept inside one.
   * - ``drop_roles``
     - ``navigation``, ``banner``, ``contentinfo``, ``complementary``,
       ``search``, ``dialog``, ``alertdialog``, ``menu``, ``menubar``
     - An element with one of these ``role`` values is dropped.
   * - ``drop_hidden``
     - ``True``
     - Drops elements with ``hidden`` (except ``hidden="until-found"``),
       ``aria-hidden="true"``, or an inline ``display: none`` or
       ``visibility: hidden``.
   * - ``strip_chars``
     - ``"¶"``
     - Characters removed from all text: by default the permalink glyph that
       Sphinx and MkDocs attach to every heading.
   * - ``robots_names``
     - ``robots``
     - The ``<meta name>`` values whose directives are read. Add a crawler's
       own token to honour directives addressed to it.
   * - ``base_url``
     - ``None``
     - The URL the page was fetched from; relative links are resolved
       against it.

``header``, ``footer`` and ``aside`` are chrome only at page level: inside an
article, the ``<header>`` holds the article's own title and byline. ``form``
is not dropped, because some sites wrap the entire page in a single
``<form>``.

The ``markdown`` property renders headings as ``#`` lines, items as ``-``
lines, rows as one line each and code as a fenced block. It serves chunking
and a readable archive rather than fidelity: inline formatting and link
targets are not reproduced. For the page above:

.. code-block:: python

   print(page.markdown)

.. code-block:: text

   # Booking the hall

   The hall seats 120 people and can be booked by residents.

   ## Prices

   Slot | Price
   Half day | 40 EUR
   Full day | 70 EUR

   ## Cancellation

   Cancel at least 7 days ahead for a full refund.

Chunking
--------

:func:`~kavalai.text.chunk_blocks` walks the blocks in order. Blocks under the
same heading path are joined by a blank line until the next one would pass
``target_chars``; a new heading path always starts a new chunk. Heading
blocks contribute through the heading path rather than as body text, and
blocks with no text are skipped.

A block longer than the room left under ``max_chars`` is split at line
breaks, then at sentence ends, then between words, and only a single word
longer than that is cut. A sentence ends at ``.``, ``!``, ``?`` or ``…``
followed by whitespace, after any closing quotes or brackets, and at ``。``,
``！`` or ``？`` without one, so Chinese and Japanese text is split at its
sentences too. The pieces of a split block are again packed towards
``target_chars``.

.. list-table::
   :header-rows: 1
   :widths: 20 16 64

   * - Parameter
     - Default
     - Effect
   * - ``title``
     - ``""``
     - The document title for the prefix.
   * - ``target_chars``
     - ``1200``
     - The size chunks are packed towards. ``None`` packs up to
       ``max_chars``; a target above ``max_chars`` is lowered to it.
   * - ``max_chars``
     - ``2000``
     - The size no chunk exceeds, prefix included. ``None`` never splits a
       block; with ``target_chars=None`` as well, each heading section is
       one chunk.
   * - ``prefix``
     - ``True``
     - Begins each chunk with the title and heading path and a blank line.

The cap defaults to 2000 characters because small embedding models truncate
their input at 512 tokens, so a longer chunk would be embedded only in part.
The prefix exists because a chunk is retrieved on its own: a table row or a
short paragraph means little without the section it came from. A heading path
that already begins with the title does not repeat it, and a prefix longer
than half of ``max_chars`` is shortened with an ellipsis, so the body always
keeps at least half of the room. A size below 1 raises :class:`ValueError`.

:func:`~kavalai.text.chunk_markdown` takes the same keyword arguments. It
reads ATX headings (``#`` to ``######``) into the heading path, ends a
paragraph block at a blank line, and keeps a fenced code block (three or more
backticks or tildes) as one block whose ``#`` lines are not headings.

The full pipeline
-----------------

``tools/zerocostchatbot/`` in the repository builds a site chatbot's index
with this module. ``scrape.py`` crawls a site, parses every page fetched over
plain HTTP with :func:`~kavalai.text.parse_html` and stores the page's
markdown in a pages database while its links feed the crawl frontier;
``build_index.py`` chunks the stored markdown with
:func:`~kavalai.text.chunk_markdown` and indexes the chunks:

.. code-block:: bash

   python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 50
   python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db \
       --target-chars 1200 --max-chars 2000

The objects themselves are documented in :doc:`../api/text`.
