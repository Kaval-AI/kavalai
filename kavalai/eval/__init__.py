"""Evaluation and acceptance testing for Kaval.AI workflows.

A suite is a directory of files: a dataset of cases, the personas that talk to
the agent, and one ``suite.yaml`` tying them to a target and a threshold.
Running it is one command and needs no database::

    kavalai-eval examples/bakery/eval/suite.yaml --tag pr-412

The same thing from Python, which is what the pytest integration does::

    from kavalai.eval import Experiment, Suite, assert_suite_passes

    suite = Suite.from_yaml("examples/bakery/eval/suite.yaml")
    assert_suite_passes(await Experiment(suite, tag="ci").run())

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
"""

from kavalai.eval.evaluators import (
    EvaluationError,
    Evaluator,
    build_evaluator,
    evaluator,
    known_evaluators,
)
from kavalai.eval.models import (
    Case,
    CaseResult,
    CaseVerdict,
    Dataset,
    EvaluatorSpec,
    ExperimentResult,
    GateSpec,
    Score,
    SliceSpec,
    Suite,
    TargetSpec,
)
from kavalai.eval.persona import Conversation, Persona, PersonaRunner
from kavalai.eval.report import comment_body, print_diff, print_report, write_junit
from kavalai.eval.runner import (
    BudgetExceeded,
    Experiment,
    assert_suite_passes,
    diff_against,
    external_id_for,
    run_suite,
)
from kavalai.eval.targets import (
    CallableTarget,
    EngineTarget,
    RestTarget,
    RunRecord,
    Target,
    build_target,
)
from kavalai.eval.trajectory import Trajectory

__all__ = [
    "BudgetExceeded",
    "CallableTarget",
    "Case",
    "CaseResult",
    "CaseVerdict",
    "Conversation",
    "Dataset",
    "EngineTarget",
    "EvaluationError",
    "Evaluator",
    "EvaluatorSpec",
    "Experiment",
    "ExperimentResult",
    "GateSpec",
    "Persona",
    "PersonaRunner",
    "RestTarget",
    "RunRecord",
    "Score",
    "SliceSpec",
    "Suite",
    "Target",
    "TargetSpec",
    "Trajectory",
    "assert_suite_passes",
    "build_evaluator",
    "build_target",
    "comment_body",
    "diff_against",
    "evaluator",
    "external_id_for",
    "known_evaluators",
    "print_diff",
    "print_report",
    "run_suite",
    "write_junit",
]
