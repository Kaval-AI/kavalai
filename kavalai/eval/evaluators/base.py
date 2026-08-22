"""The evaluator interface and the name registry a YAML file resolves through.

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

from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Union

from loguru import logger

from kavalai.eval.models import Case, EvaluatorSpec, Score
from kavalai.eval.targets import RunRecord


class EvaluationError(Exception):
    """Raised when an evaluator cannot score — as opposed to scoring a fail.

    The distinction is the whole point: a trajectory assertion against a target
    that has no trajectory must be an error the report shows, never a pass.
    """


class Evaluator(ABC):
    """Scores one run of one case.

    Subclasses take their configuration as keyword arguments in ``__init__``
    (which is what a YAML ``{type: ..., **options}`` block becomes) and return
    one :class:`~kavalai.eval.models.Score` or a list of them.
    """

    #: Name this evaluator is referred to by in YAML. Set by the decorator.
    name: str = "evaluator"
    #: Whether it needs the target to have observed a trajectory.
    needs_trajectory: bool = False
    #: Whether scoring calls a provider. The free tier skips these, and says
    #: in the report which assertions it therefore did not run — a gate that
    #: quietly drops half its checks is worse than one that admits it.
    needs_model: bool = False

    @abstractmethod
    async def score(
        self, case: Case, record: RunRecord
    ) -> Union[Score, list[Score]]: ...

    def require_trajectory(self, record: RunRecord) -> None:
        if not record.trajectory.observed:
            raise EvaluationError(
                f"'{self.name}' needs a trajectory, and this target does not "
                "produce one. Use target kind 'engine', or drop the assertion — "
                "scoring it as a pass would report a gate as green because it "
                "could not see anything."
            )


#: name -> evaluator class. Populated by the decorator below.
REGISTRY: dict[str, Type[Evaluator]] = {}


def evaluator(
    name: str, *, replace: bool = False
) -> Callable[[Type[Evaluator]], Type[Evaluator]]:
    """Register an evaluator under the name a YAML file uses.

    The plugin point: a customer adds a domain evaluator — "the stored order
    matches the spec" — without forking the evaluator module.

    >>> @kavalai.evaluator("refund_amount_correct")
    ... class RefundAmountCorrect(Evaluator):
    ...     async def score(self, case, record): ...
    """

    def decorate(cls: Type[Evaluator]) -> Type[Evaluator]:
        if name in REGISTRY and not replace:
            raise ValueError(
                f"Evaluator '{name}' is already registered by "
                f"{REGISTRY[name].__module__}. Pass replace=True to override it."
            )
        if name in REGISTRY:
            logger.warning(f"Replacing already-registered evaluator '{name}'.")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return decorate


def build_evaluator(spec: Union[EvaluatorSpec, dict, str]) -> Evaluator:
    """Turn one declarative spec into an evaluator instance."""
    spec = EvaluatorSpec.coerce(spec)
    cls = REGISTRY.get(spec.type)
    if cls is None:
        known = ", ".join(sorted(REGISTRY)) or "none registered"
        raise ValueError(f"Unknown evaluator '{spec.type}'. Known: {known}.")
    try:
        return cls(**spec.options)
    except TypeError as exc:
        raise ValueError(
            f"Evaluator '{spec.type}' rejected its options {spec.options}: {exc}"
        ) from exc


def build_evaluators(specs: list) -> list[Evaluator]:
    return [build_evaluator(s) for s in specs]


def known_evaluators() -> list[str]:
    return sorted(REGISTRY)


def _value_at(data: object, path: str) -> object:
    """Read ``a.b[0].c`` out of nested dicts and lists.

    Returns ``None`` for anything missing rather than raising: a field
    assertion on a run that produced no output should read "missing", not
    crash the whole experiment.
    """
    current: object = data
    for part in path.replace("]", "").replace("[", ".").split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.lstrip("-").isdigit():
            index = int(part)
            current = current[index] if -len(current) <= index < len(current) else None
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _missing_reason(path: str, expected: object) -> str:
    return f"{path} is missing; expected {expected!r}"


__all__ = [
    "EvaluationError",
    "Evaluator",
    "REGISTRY",
    "build_evaluator",
    "build_evaluators",
    "evaluator",
    "known_evaluators",
]
