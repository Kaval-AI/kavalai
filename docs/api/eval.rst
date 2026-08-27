Evaluation API
==============

:mod:`kavalai.eval` grades an agent server that is **already running**. It
speaks HTTP and discovers the agent's input and output types from the server's
OpenAPI specification through :class:`~kavalai.client.AgentClient`, so it knows
nothing about the workflow engine, the YAML graph or the database behind it.

Two evaluators share that plumbing and differ only in how they judge an answer:
:class:`~kavalai.eval.SimpleEvaluator` compares it with expected values and
calls no model at all, while :class:`~kavalai.eval.JudgeEvaluator` asks a model
whether it satisfies a plain-language criterion. Either can be called straight
from a test.

For the ideas, read :doc:`/guides/evaluation`; for the case-file keys and the
command-line flags, :doc:`/reference/eval_yaml`.

Shared plumbing
---------------

.. automodule:: kavalai.eval.base

Literal comparison
------------------

.. automodule:: kavalai.eval.simple_evaluator

Model-graded comparison
-----------------------

.. automodule:: kavalai.eval.judge_evaluator

Running a file of cases
-----------------------

.. automodule:: kavalai.eval.eval_runner
