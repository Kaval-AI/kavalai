Network Guard API
=================

:mod:`kavalai.net` decides which addresses a request to a model-chosen URL may
reach. The bundled ``http_request`` and ``crawl_url`` tools use it by default,
and :class:`~kavalai.net.PublicOnlyTransport` gives the same protection to any
httpx client in your own tools. It is not re-exported from ``kavalai``, because
it needs httpx and ``import kavalai`` has to work without it.

For what the guard covers and where it stops, read the *Outbound requests*
section of :doc:`/guides/safety`; for the tools, :doc:`/reference/tools`.

.. automodule:: kavalai.net
