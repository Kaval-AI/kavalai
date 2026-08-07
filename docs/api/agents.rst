Agents API
==========

The agent runtime lives directly in the top-level :mod:`kavalai` package. The
headline class is :class:`~kavalai.agent.Agent` — a multi-step reasoning loop
that calls tools through a :class:`~kavalai.FunctionKernel` until it produces
a final, optionally structured, answer.

Agent
-----

.. automodule:: kavalai.agent

Run context
-----------

.. automodule:: kavalai.run_context

Agent service & persistence
---------------------------

.. automodule:: kavalai.agent_service

.. automodule:: kavalai.backoffice.sessions

Remote agent client
--------------------

.. automodule:: kavalai.client

RAG service
-----------

.. automodule:: kavalai.rag.base

.. automodule:: kavalai.rag.postgres
   :exclude-members: build_batch_query_cte, batch_query_with_join

.. automodule:: kavalai.rag.sqllite
