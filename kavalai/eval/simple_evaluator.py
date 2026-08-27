"""Check an agent's answer against expected values, without a model.

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

import re
from typing import Any, Optional

from kavalai.eval.base import AgentEvaluator, EvalResult

#: The matcher names a field expectation may use.
MATCHERS = ("equals", "contains", "not_contains", "regex", "one_of")


def _as_list(value: Any) -> list[Any]:
    """One value or many, always returned as a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _contains(actual: Any, needle: Any) -> bool:
    """Substring for text, membership for a collection.

    Text is compared case-insensitively: a model that writes "Marsh
    Marigold" where the fact says "marsh marigold" has not made a mistake
    worth failing a build over.
    """
    if isinstance(actual, str):
        return str(needle).lower() in actual.lower()
    if isinstance(actual, (list, tuple, set, dict)):
        return needle in actual
    return False


def is_matcher_spec(spec: Any) -> bool:
    """Whether ``spec`` is a matcher mapping rather than a literal value.

    A dict is read as matchers only when every key is a known matcher name,
    so an agent that genuinely answers with a dict can still be compared for
    equality.
    """
    return (
        isinstance(spec, dict)
        and len(spec) > 0
        and all(key in MATCHERS for key in spec)
    )


def check_field(name: str, actual: Any, spec: Any) -> list[str]:
    """Check one output field against its expectation.

    Args:
        name: Field name, used in the failure messages.
        actual: What the agent answered in that field.
        spec: Either a literal value to compare for equality, or a mapping of
            matcher name to argument (``equals``, ``contains``,
            ``not_contains``, ``regex``, ``one_of``).

    Returns:
        A list of failure messages; empty when the field is as expected.
    """
    if not is_matcher_spec(spec):
        spec = {"equals": spec}

    failures = []
    for matcher, argument in spec.items():
        if matcher == "equals":
            if actual != argument:
                failures.append(f"{name}: expected {argument!r}, got {actual!r}")
        elif matcher == "contains":
            missing = [x for x in _as_list(argument) if not _contains(actual, x)]
            if missing:
                failures.append(f"{name}: {actual!r} is missing {missing!r}")
        elif matcher == "not_contains":
            present = [x for x in _as_list(argument) if _contains(actual, x)]
            if present:
                failures.append(f"{name}: {actual!r} should not contain {present!r}")
        elif matcher == "regex":
            if not re.search(str(argument), str(actual)):
                failures.append(f"{name}: {actual!r} does not match /{argument}/")
        elif matcher == "one_of":
            if actual not in _as_list(argument):
                failures.append(f"{name}: {actual!r} is not one of {argument!r}")
    return failures


def check_output(
    output: dict[str, Any], expected: Optional[dict[str, Any]]
) -> list[str]:
    """Check every expected field of one agent answer.

    Fields the expectation does not mention are ignored, so a case states
    what it cares about and nothing more.
    """
    failures = []
    for name, spec in (expected or {}).items():
        if name not in output:
            failures.append(f"{name}: the agent's output has no such field")
            continue
        failures.extend(check_field(name, output[name], spec))
    return failures


class SimpleEvaluator(AgentEvaluator):
    """Send one input to a running agent and check the answer literally.

    Use it wherever the right answer is a fact rather than a matter of
    phrasing — an extracted field, an id, a classification, a number that has
    to appear. It calls no model of its own, so it is fast, free and gives
    the same verdict every time.

    Example:
        .. code-block:: python

            evaluator = SimpleEvaluator("http://localhost:25000")
            result = await evaluator.evaluate(
                {"user_message": "Who is the president of Green Village?"},
                {"agent_response": {"contains": "Thomas Cook"}},
            )
            assert result.passed, result.reason
    """

    async def evaluate(
        self,
        inputs: dict[str, Any],
        expected: Optional[dict[str, Any]] = None,
        name: str = "case",
    ) -> EvalResult:
        """Run one case and compare the answer with ``expected``.

        Args:
            inputs: Field values for the agent's input type.
            expected: Output field name to expected value or matcher mapping.
                An empty expectation asserts only that the agent answered at
                all, which is a useful smoke test in its own right.
            name: Case name, carried into the result.

        Returns:
            The :class:`~kavalai.eval.base.EvalResult` for this case. A failed
            agent call is reported as a failure rather than raised, so one
            broken case cannot end a whole run.
        """
        try:
            output = (
                await self.run_agent(inputs, external_id=self.external_id(name))
            ).model_dump()
        except Exception as e:
            return EvalResult(
                name=name,
                passed=False,
                reason=f"the agent run failed: {e}",
                inputs=inputs,
            )

        failures = check_output(output, expected)
        return EvalResult(
            name=name,
            passed=not failures,
            reason="; ".join(failures),
            inputs=inputs,
            output=output,
        )
