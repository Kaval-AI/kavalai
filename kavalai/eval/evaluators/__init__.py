"""Built-in evaluators. Importing this module registers every one of them.

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

from kavalai.eval.evaluators.base import (
    EvaluationError,
    Evaluator,
    build_evaluator,
    build_evaluators,
    evaluator,
    known_evaluators,
)

# Imported for their side effect: each module registers its evaluators by name,
# which is how a YAML ``{type: ...}`` block resolves.
from kavalai.eval.evaluators import conversation as _conversation  # noqa: F401
from kavalai.eval.evaluators import deterministic as _deterministic  # noqa: F401
from kavalai.eval.evaluators import judged as _judged  # noqa: F401
from kavalai.eval.evaluators import trajectory as _trajectory  # noqa: F401

__all__ = [
    "EvaluationError",
    "Evaluator",
    "build_evaluator",
    "build_evaluators",
    "evaluator",
    "known_evaluators",
]
