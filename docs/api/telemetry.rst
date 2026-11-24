Telemetry API
=============

:class:`kavalai.Telemetry` is the interface through which the engine, the
function kernel and the token accumulator report a run to a tracing system.
Every method of the base class does nothing.
:class:`kavalai.otel.OtelTelemetry` implements it over OpenTelemetry.
:mod:`kavalai.otel` is not re-exported from ``kavalai``, because it needs the
``otel`` extra and ``import kavalai`` has to work without it.

For what is exported and why, read :ref:`observability-opentelemetry`.

.. automodule:: kavalai.telemetry

.. automodule:: kavalai.otel
