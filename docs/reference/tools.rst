Bundled tools
=============

``kavalai[common]`` ships a handful of ready-made tools. They are ordinary
``@pythontool`` functions, so you register them like any other Python tool:

.. code-block:: python

   from kavalai import FunctionKernel
   from kavalai.tools.webtools.http_client import http_request

   kernel = FunctionKernel()
   kernel.register_python_tool("http.request", http_request)

   response = await kernel.call_tool(
       "python://http.request", {"method": "GET", "url": "https://example.com"}
   )

In a workflow, declare them under ``python_functions`` instead — see
:doc:`yaml`:

.. code-block:: yaml

   python_functions:
     - {name: web.search, path: kavalai.tools.webtools.crawl4ai.web_search}
     - {name: web.crawl,  path: kavalai.tools.webtools.crawl4ai.crawl_url}

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Tool
     - Import path
     - Needs
   * - ``crawl_url``
     - ``kavalai.tools.webtools.crawl4ai``
     - a headless browser
   * - ``web_search``
     - ``kavalai.tools.webtools.crawl4ai``
     - a headless browser
   * - ``http_request``
     - ``kavalai.tools.webtools.http_client``
     - —
   * - ``get_rss_feed``
     - ``kavalai.tools.rss``
     - —

Web crawling
------------

``crawl_url``
^^^^^^^^^^^^^

Renders a page in a headless browser and returns clean Markdown — which is what
you want to feed a model, rather than raw HTML full of navigation and scripts.

.. code-block:: python

   crawl_url(url: str, include_html: bool = False,
             bypass_cache: bool = False, timeout: float = 60.0) -> Crawl4aiResponse

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Argument
     - Description
   * - ``url``
     - The page to fetch.
   * - ``include_html``
     - Also return the cleaned HTML. Default ``False``.
   * - ``bypass_cache``
     - Refetch instead of using the crawler cache. Default ``False``.
   * - ``timeout``
     - Page load timeout in seconds. Default ``60``.

Returns a ``Crawl4aiResponse`` with ``url``, ``success``, ``markdown``,
``html``, ``status_code``, ``metadata`` and ``error_message``. Failures come
back as ``success=False`` with an ``error_message`` rather than raising, so an
agent can read the error and try something else.

``web_search``
^^^^^^^^^^^^^^

A web search that needs **no API key**: it scrapes the DuckDuckGo HTML endpoint
through the same browser.

.. code-block:: python

   web_search(query: str, count: int = 10,
              timeout: float = 60.0) -> WebSearchResponse

Returns ``query``, ``success``, ``error_message`` and ``results`` — a list of
``WebSearchResult`` (``title``, ``url``, ``snippet``).

.. note::

   Both tools drive a real browser, so they are slow (seconds, not
   milliseconds) relative to an HTTP call, and scraped search results depend on
   a page layout that DuckDuckGo may change. For production search volume,
   register a keyed search API as a ``rest://`` server instead.

   ``docker-compose.yml`` includes a ``crawl4ai`` service if you would rather
   run the crawler as a container.

The pair is combined in ``examples/business_info_agent/business_info.py``: a
search node finds candidate pages, an agent node restricted to the crawl tool
reads the promising ones, and an LLM node writes the summary. The case file
beside it grades what that agent reports about a company it cannot answer from
memory.

HTTP
----

``http_request``
^^^^^^^^^^^^^^^^

Any HTTP request, with optional basic auth and an optional Tor proxy. This is
the general-purpose escape hatch when an endpoint does not deserve a full
``rest://`` server registration.

.. code-block:: python

   http_request(method: str, url: str, params=None, headers=None,
                json_body=None, data_body=None, auth_user=None,
                auth_password=None, timeout: float = 30.0,
                use_proxy: bool = False) -> HttpResponse

Returns ``status_code``, ``headers``, ``text`` and ``json_data`` (parsed when
the response is JSON, otherwise ``None``).

``use_proxy=True`` routes the request through the Tor proxy configured by
``TOR_PROXY_HOST`` / ``TOR_PROXY_PORT``; ``docker-compose.yml`` provides a
``torproxy`` service.

.. warning::

   Handing an agent a general HTTP tool lets it call any URL it can compose,
   including internal addresses. Prefer registering specific ``rest://`` tools,
   or restrict the node with ``allowed_tools``, when the model chooses the
   target.

Feeds
-----

``get_rss_feed``
^^^^^^^^^^^^^^^^

Fetches and parses an RSS or Atom feed.

.. code-block:: python

   get_rss_feed(url: str, max_results: int = 5) -> Feed

Returns ``title``, ``url`` and ``items`` (each with ``title``, ``link`` and
``summary``). The same module is also a small FastAPI service
(``python -m kavalai.tools.rss``) exposing the tool at ``GET /get_rss_feed``;
``RSS_AUTH_USER`` / ``RSS_AUTH_PASSWORD`` are the basic-auth credentials that
service requires from its callers, not credentials for the feed.

Writing your own
----------------

These are unremarkable functions — a decorator, type hints and a docstring.
Yours will look the same:

.. code-block:: python

   from pydantic import BaseModel
   from kavalai import pythontool


   class PondReading(BaseModel):
       depth_m: float
       water: str


   @pythontool
   def measure_pond(name: str) -> PondReading:
       """Measure a Green Village pond by name."""
       ...

The docstring and the type hints *are* the interface the model sees, so write
them for a reader who knows nothing about your codebase. See
:doc:`../tutorials/agents` and :doc:`../guides/tools`.
