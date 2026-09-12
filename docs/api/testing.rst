Testing API
===========

:mod:`kavalai.testing` replaces the model and nothing else, so a test runs the
real :class:`~kavalai.WorkflowEngine`, streamer and validation without a
network connection or an API key. The module is part of the base install and
imports neither a provider SDK nor pytest, so it also runs under Pyodide.

The recipe "Testing a workflow without calling a model" in
:doc:`/cookbook/index` shows the clients in use, and :doc:`/guides/safety`
explains why a scripted model is the right substitute.

.. automodule:: kavalai.testing
