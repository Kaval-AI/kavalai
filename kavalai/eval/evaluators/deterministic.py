"""Evaluators that need no model: free, exact, and safe in a pull-request gate.

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

from kavalai.eval.evaluators.base import Evaluator, _value_at, evaluator
from kavalai.eval.models import Case, Score
from kavalai.eval.targets import RunRecord
from kavalai.utils import to_plain


@evaluator("no_error")
class NoError(Evaluator):
    """The run completed. Nearly always the first evaluator in a suite."""

    async def score(self, case: Case, record: RunRecord) -> Score:
        return Score.boolean(self.name, record.ok, reason=record.error)


@evaluator("equals_expected")
class EqualsExpected(Evaluator):
    """The output equals the case's ``expected`` value exactly."""

    async def score(self, case: Case, record: RunRecord) -> Score:
        expected = to_plain(case.expected)
        actual = to_plain(record.output)
        ok = expected == actual
        return Score.boolean(
            self.name,
            ok,
            reason=None if ok else f"expected {expected!r}, got {actual!r}",
        )


@evaluator("field_equals")
class FieldEquals(Evaluator):
    """One field of the output equals ``value``.

    ``path`` is dotted and may index lists: ``order.items[0].quantity``.
    """

    def __init__(self, path: str, value: Any):
        self.path = path
        self.value = value

    async def score(self, case: Case, record: RunRecord) -> Score:
        actual = _value_at(to_plain(record.output), self.path)
        ok = actual == self.value
        return Score.boolean(
            self.name,
            ok,
            reason=None if ok else f"{self.path} = {actual!r}, expected {self.value!r}",
            path=self.path,
        )


@evaluator("json_subset")
class JsonSubset(Evaluator):
    """Every key/value in ``expected`` appears in the output, nested included.

    The forgiving cousin of ``equals_expected``: assert the fields that matter
    and stay silent about the ones that do not, so adding an output field does
    not turn a whole suite red.
    """

    def __init__(self, expected: Optional[dict] = None):
        self.expected = expected

    async def score(self, case: Case, record: RunRecord) -> Score:
        expected = (
            self.expected if self.expected is not None else to_plain(case.expected)
        )
        if expected is None:
            return Score(name=self.name, reason="no expected value to compare against")
        missing = _diff_subset(expected, to_plain(record.output))
        return Score.boolean(
            self.name, not missing, reason=None if not missing else "; ".join(missing)
        )


def _diff_subset(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Describe every way ``actual`` fails to contain ``expected``."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [
                f"{path or 'output'}: expected an object, got {type(actual).__name__}"
            ]
        problems: list[str] = []
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                problems.append(f"{child} is missing")
            else:
                problems += _diff_subset(value, actual[key], child)
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path or 'output'}: expected a list, got {type(actual).__name__}"]
        if len(actual) < len(expected):
            return [
                f"{path or 'output'}: expected {len(expected)} items, got {len(actual)}"
            ]
        problems = []
        for index, value in enumerate(expected):
            problems += _diff_subset(value, actual[index], f"{path}[{index}]")
        return problems
    if expected != actual:
        return [f"{path or 'output'} = {actual!r}, expected {expected!r}"]
    return []


@evaluator("contains")
class Contains(Evaluator):
    """The output text contains ``text`` (case-insensitive by default)."""

    def __init__(self, text: Optional[str] = None, case_sensitive: bool = False):
        self.text = text
        self.case_sensitive = case_sensitive

    def _needle(self, case: Case) -> str:
        # Falling back to the case's own expected value is what lets one
        # ``contains`` line in the suite grade a whole generated slice.
        if self.text is not None:
            return self.text
        expected = case.expected
        if isinstance(expected, dict):
            return str(expected.get("contains", ""))
        return "" if expected is None else str(expected)

    async def score(self, case: Case, record: RunRecord) -> Score:
        needle = self._needle(case)
        haystack = record.output_text()
        if not needle:
            return Score(name=self.name, reason="nothing to look for")
        if self.case_sensitive:
            ok = needle in haystack
        else:
            ok = needle.casefold() in haystack.casefold()
        return Score.boolean(
            self.name, ok, reason=None if ok else f"{needle!r} not in the answer"
        )


@evaluator("not_contains")
class NotContains(Evaluator):
    """The output text does **not** contain ``text``.

    Note the empty-output case: an empty answer contains nothing, so this
    passes. That is correct but rarely what you meant on its own — pair it with
    ``no_error`` so a crashed run cannot look like a well-behaved refusal.
    """

    def __init__(self, text: str, case_sensitive: bool = False):
        self.text = text
        self.case_sensitive = case_sensitive

    async def score(self, case: Case, record: RunRecord) -> Score:
        haystack = record.output_text()
        if self.case_sensitive:
            hit = self.text in haystack
        else:
            hit = self.text.casefold() in haystack.casefold()
        return Score.boolean(
            self.name,
            not hit,
            reason=f"{self.text!r} appears in the answer" if hit else None,
        )


@evaluator("regex")
class Regex(Evaluator):
    """The output text matches ``pattern``."""

    def __init__(self, pattern: str, flags: str = ""):
        self.pattern = pattern
        self._compiled = re.compile(pattern, re.IGNORECASE if "i" in flags else 0)

    async def score(self, case: Case, record: RunRecord) -> Score:
        ok = bool(self._compiled.search(record.output_text()))
        return Score.boolean(
            self.name, ok, reason=None if ok else f"no match for /{self.pattern}/"
        )


@evaluator("no_digits")
class NoDigits(Evaluator):
    """The answer states no numbers.

    The cheap half of "did it refuse rather than guess": a grounded chatbot
    that cannot find a figure must not produce one.
    """

    async def score(self, case: Case, record: RunRecord) -> Score:
        found = re.findall(r"\d+", record.output_text())
        return Score.boolean(
            self.name,
            not found,
            reason=None if not found else f"answer states {', '.join(found[:5])}",
        )


@evaluator("latency_under")
class LatencyUnder(Evaluator):
    """The run finished within ``seconds``.

    Fails the case and puts the measurement in the reason, so the JUnit failure
    explains itself without anyone opening the result file.
    """

    def __init__(self, seconds: float):
        self.seconds = float(seconds)

    async def score(self, case: Case, record: RunRecord) -> Score:
        measured = record.duration_seconds
        ok = measured <= self.seconds
        return Score(
            name=self.name,
            value=measured,
            passed=ok,
            reason=None if ok else f"{measured:.1f}s > {self.seconds:.1f}s",
        )


@evaluator("tokens_under")
class TokensUnder(Evaluator):
    """The run spent no more than ``n`` tokens.

    Honest now that the token accumulator is per run: this is the case's own
    spend, not the process's.
    """

    def __init__(self, n: Optional[int] = None, per_case: Optional[int] = None):
        limit = n if n is not None else per_case
        if limit is None:
            raise TypeError("tokens_under needs 'n' (or 'per_case')")
        self.limit = int(limit)

    async def score(self, case: Case, record: RunRecord) -> Score:
        measured = record.total_tokens
        ok = measured <= self.limit
        return Score(
            name=self.name,
            value=float(measured),
            passed=ok,
            reason=None if ok else f"{measured:,} tokens > {self.limit:,}",
        )


@evaluator("output_not_empty")
class OutputNotEmpty(Evaluator):
    """The run produced an answer with something in it."""

    async def score(self, case: Case, record: RunRecord) -> Score:
        text = record.output_text().strip()
        return Score.boolean(
            self.name, bool(text), reason=None if text else "the answer is empty"
        )


@evaluator("always_fails")
class AlwaysFails(Evaluator):
    """A canary. Put one case per suite behind it, designed to fail.

    The eval harness gates deploys and is itself code: an evaluator with an
    inverted condition that passes everything is worse than no gate, because it
    manufactures confidence. If the suite ever reports this case as passing,
    the harness is broken.
    """

    async def score(self, case: Case, record: RunRecord) -> Score:
        return Score.boolean(self.name, False, reason="canary: this case must fail")
