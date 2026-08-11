Workflow API
============

The workflow engine lives in :mod:`kavalai.workflow`. It turns a YAML graph (or a
:class:`~kavalai.WorkflowBuilder` chain) into an executable state machine, runs
it as a stream of :class:`~kavalai.workflow.models.WorkflowStreamEvent` events,
and records a serialisable state through an
:class:`~kavalai.agent_service.AgentService` and a pluggable task-logger backend.

Engine and builder
-------------------

.. automodule:: kavalai.workflow.engine

.. automodule:: kavalai.workflow.builder

Graph models
------------

.. automodule:: kavalai.workflow.models

Run state
---------

.. automodule:: kavalai.workflow.state

Expressions
-----------

.. automodule:: kavalai.workflow.expressions

Client factory
--------------

.. automodule:: kavalai.workflow.clients

Task logging backends
---------------------

.. automodule:: kavalai.workflow.tasklog.base

.. automodule:: kavalai.workflow.tasklog.sqlite

.. automodule:: kavalai.workflow.tasklog.postgres
