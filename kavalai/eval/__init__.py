"""Evaluation of a running Kaval.AI agent.

Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Two evaluators, both talking to an agent server that is already running and
both discovering its input and output types from its OpenAPI spec:

* :class:`~kavalai.eval.simple_evaluator.SimpleEvaluator` compares the answer
  with expected values — no model, same verdict every time.
* :class:`~kavalai.eval.judge_evaluator.JudgeEvaluator` asks a model whether
  the answer meets a plain-language criterion, for answers whose wording is
  free but whose substance is not.

Either can be used directly from a unit test. A YAML file of cases can be run
by :func:`~kavalai.eval.eval_runner.run_suite`, or from the command line with
``kavalai-eval``.
"""

from kavalai.eval.base import DEFAULT_BASE_URL, AgentEvaluator, EvalResult
from kavalai.eval.eval_runner import (
    EvalCase,
    EvalSuite,
    load_suite,
    run_suite,
)
from kavalai.eval.judge_evaluator import (
    DEFAULT_JUDGE_MODEL,
    JudgeEvaluator,
    JudgeVerdict,
)
from kavalai.eval.simple_evaluator import SimpleEvaluator, check_output

__all__ = [
    "AgentEvaluator",
    "EvalResult",
    "DEFAULT_BASE_URL",
    "SimpleEvaluator",
    "check_output",
    "JudgeEvaluator",
    "JudgeVerdict",
    "DEFAULT_JUDGE_MODEL",
    "EvalCase",
    "EvalSuite",
    "load_suite",
    "run_suite",
]
