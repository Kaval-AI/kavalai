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

``crawl_url`` refuses a URL whose host is, or resolves to, a private,
loopback, link-local or cloud metadata address; the check is
:func:`kavalai.net.ensure_public_url`. A refusal is a failure like any other:
``success=False`` and an ``error_message`` beginning ``Refused:``. When
crawl4ai reports that the page ended on a different URL, that URL is checked in
the same way and the content is withheld if it fails.

.. code-block:: python

   from kavalai.tools.webtools.crawl4ai import crawl_url

   response = await crawl_url(
       "http://metadata.google.internal/computeMetadata/v1/"
   )
   print(response.success, response.error_message)

.. code-block:: text

   False Refused: metadata.google.internal is an internal host name

The browser runs in a separate process and resolves names itself, so the SDK
sees only the requested URL and the final one. A redirect through an internal
address in between, or a DNS answer that changes after the check, is not
observed. Full protection for browser-based crawling therefore requires
network egress control — a firewall rule or network policy that gives the
browser, or the ``crawl4ai`` container, no route to internal addresses (see
:doc:`../deploy/index`).

An agent meant to read an intranet is given a tool built with the check
switched off. The switch is an argument of the factory, not of the tool, so the
model cannot set it:

.. code-block:: python

   from kavalai.tools.webtools.crawl4ai import make_crawl_url

   kernel.register_python_tool(
       "intranet.crawl", make_crawl_url(allow_private_networks=True)
   )

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

   async http_request(method: str, url: str, params=None, headers=None,
                      json_body=None, data_body=None, auth_user=None,
                      auth_password=None, timeout: float = 30.0,
                      use_proxy: bool = False) -> HttpResponse

Returns ``status_code``, ``headers``, ``text`` and ``json_data`` (parsed when
the response is JSON, otherwise ``None``). The tool is a coroutine function, so
a direct call is awaited. Redirects are not followed: a 3xx response is
returned as it is.

Private addresses
"""""""""""""""""

By default the tool refuses a target that is, or resolves to, a private,
loopback, link-local or cloud metadata address, and raises
:class:`kavalai.net.UnsafeUrlError`; through the kernel the same message
arrives in a ``FunctionKernelException``. The request goes through
:class:`kavalai.net.PublicOnlyTransport`, which checks every address the host
resolves to and then connects to the address it checked. A second DNS lookup,
which could return a different answer, never takes place, while the ``Host``
header, TLS SNI and certificate verification still use the host name.

.. code-block:: python

   from kavalai.net import UnsafeUrlError
   from kavalai.tools.webtools.http_client import http_request

   try:
       await http_request("GET", "http://169.254.169.254/latest/meta-data/")
   except UnsafeUrlError as error:
       print(error)

.. code-block:: text

   169.254.169.254 is a non-public address

An agent meant to call an intranet API is given a tool built with
``allow_private_networks=True``. The switch is an argument of the factory, not
of the tool, so the model cannot set it:

.. code-block:: python

   from kavalai import FunctionKernel
   from kavalai.tools.webtools.http_client import make_http_request

   kernel = FunctionKernel()
   kernel.register_python_tool(
       "intranet.request", make_http_request(allow_private_networks=True)
   )

A workflow names an importable attribute under ``python_functions``, so the
tool is built once in a module of your own and declared by that path:

.. code-block:: python

   # myapp/tools.py
   from kavalai.tools.webtools.http_client import make_http_request

   intranet_request = make_http_request(allow_private_networks=True)

.. code-block:: yaml

   python_functions:
     - {name: intranet.request, path: myapp.tools.intranet_request}

The guarded client connects directly and does not use the ``HTTP_PROXY`` /
``HTTPS_PROXY`` variables, since a proxy would resolve the name itself.

Proxy
"""""

``use_proxy=True`` routes the request through the Tor proxy configured by
``KAVALAI_TOR_PROXY_HOST`` / ``KAVALAI_TOR_PROXY_PORT``; ``docker-compose.yml``
provides a ``torproxy`` service. The proxy is allowed whatever its own
address. Because the proxy resolves the target name, the connection cannot be
pinned, and only the pre-check :func:`kavalai.net.ensure_public_url` applies:
the name is resolved locally once, the proxy may receive a different answer,
and the local resolver sees the name as well as Tor.

.. warning::

   The guard decides where a request may go, not what it does there. A model
   with a general HTTP tool can still call any public endpoint with any method
   and body. Prefer registering specific ``rest://`` tools, or restrict the
   node with ``allowed_tools``, when the model chooses the target.

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
