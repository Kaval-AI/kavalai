"""Evaluators that assert on what the run *did*, not just what it said.

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

from typing import Any, Optional

from kavalai.eval.evaluators.base import Evaluator, _value_at, evaluator
from kavalai.eval.models import Case, Score
from kavalai.eval.targets import RunRecord


class _TrajectoryEvaluator(Evaluator):
    """Refuses to score when the target could not observe a trajectory."""

    needs_trajectory = True


@evaluator("node_visited")
class NodeVisited(_TrajectoryEvaluator):
    """The run passed through ``node``.

    Use sparingly. Node names are graph internals, so these assertions couple a
    suite to the shape of the workflow rather than to its behaviour — prefer
    ``tool_called`` or ``branch_taken``, which survive a refactor.
    """

    def __init__(self, node: str):
        self.node = node

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        ok = record.trajectory.visited(self.node)
        return Score.boolean(
            self.name,
            ok,
            reason=None
            if ok
            else f"'{self.node}' never ran; path was "
            f"{' -> '.join(record.trajectory.names())}",
        )


@evaluator("node_not_visited")
class NodeNotVisited(_TrajectoryEvaluator):
    """The run never entered ``node`` — the human-handoff arm, say."""

    def __init__(self, node: str):
        self.node = node

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        hit = record.trajectory.visited(self.node)
        return Score.boolean(
            self.name, not hit, reason=f"'{self.node}' ran" if hit else None
        )


@evaluator("branch_taken")
class BranchTaken(_TrajectoryEvaluator):
    """Branch node ``node`` routed to ``target``.

    With no arguments it reads the expectation from the case
    (``expected.branch``), which is how one line in a suite grades a whole
    generated dataset of routing cases.
    """

    def __init__(self, node: Optional[str] = None, target: Optional[str] = None):
        self.node = node
        self.target = target

    def _expected(self, case: Case) -> tuple[Optional[str], Optional[str]]:
        expected = case.expected if isinstance(case.expected, dict) else {}
        return (
            self.node or expected.get("branch_node"),
            self.target or expected.get("branch"),
        )

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        node, target = self._expected(case)
        branches = record.trajectory.branches()
        if target is None:
            return Score(name=self.name, reason="no expected branch for this case")

        if node is None:
            # No node named: assert the arm was entered by any branch. This is
            # the form that survives someone renaming the router.
            taken = [(b.output or {}).get("taken") for b in branches]
            ok = target in taken
            return Score.boolean(
                self.name,
                ok,
                reason=None if ok else f"no branch routed to '{target}'; took {taken}",
            )

        decision = record.trajectory.branch(node)
        if decision is None:
            return Score.boolean(
                self.name, False, reason=f"branch node '{node}' never ran"
            )
        taken = (decision.output or {}).get("taken")
        value = (decision.inputs or {}).get("value")
        ok = taken == target
        return Score.boolean(
            self.name,
            ok,
            reason=None
            if ok
            else f"'{node}' routed to '{taken}' on value {value!r}, expected '{target}'",
            value=value,
        )


@evaluator("switch_matched")
class SwitchMatched(_TrajectoryEvaluator):
    """Every ``switch`` matched an explicit case rather than falling to default.

    An unmatched label almost always means an upstream classifier returned
    something outside the enum — ``"Refund"``, ``"refund "``, ``"refunds"``.
    """

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        fell_through = [
            b
            for b in record.trajectory.branches()
            if b.node_type == "switch" and not (b.output or {}).get("matched", True)
        ]
        if not fell_through:
            return Score.boolean(self.name, True)
        detail = ", ".join(
            f"{b.name}={((b.inputs or {}).get('value'))!r}" for b in fell_through
        )
        return Score.boolean(
            self.name, False, reason=f"fell through to default: {detail}"
        )


@evaluator("tool_called")
class ToolCalled(_TrajectoryEvaluator):
    """Tool ``uri`` ran — whether a function node called it or an agent chose it."""

    def __init__(self, uri: str, times: Optional[int] = None):
        self.uri = uri
        self.times = times

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        calls = record.trajectory.calls_to(self.uri)
        if self.times is not None:
            ok = len(calls) == self.times
            reason = None if ok else f"called {len(calls)}x, expected {self.times}x"
        else:
            ok = bool(calls)
            reason = (
                None
                if ok
                else f"never called; tools used: {record.trajectory.tool_uris() or 'none'}"
            )
        return Score.boolean(self.name, ok, reason=reason, calls=len(calls))


@evaluator("tool_not_called")
class ToolNotCalled(_TrajectoryEvaluator):
    """Tool ``uri`` never ran.

    The safety assertion — "never call ``rest://billing.refund`` on this
    case" — and the one that catches a helpful model doing damage.
    """

    def __init__(self, uri: str):
        self.uri = uri

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        calls = record.trajectory.calls_to(self.uri)
        return Score.boolean(
            self.name,
            not calls,
            reason=f"called {len(calls)}x with {[c.inputs for c in calls][:2]}"
            if calls
            else None,
        )


@evaluator("tool_call_order")
class ToolCallOrder(_TrajectoryEvaluator):
    """``uris`` were called in this relative order, other calls allowed between."""

    def __init__(self, uris: list[str]):
        self.uris = list(uris)

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        actual = record.trajectory.tool_uris()
        remaining = list(self.uris)
        for uri in actual:
            if remaining and uri == remaining[0]:
                remaining.pop(0)
        ok = not remaining
        return Score.boolean(
            self.name,
            ok,
            reason=None if ok else f"never reached {remaining[0]}; order was {actual}",
        )


@evaluator("tool_args_match")
class ToolArgsMatch(_TrajectoryEvaluator):
    """Some call to ``uri`` passed ``path`` equal to ``value``.

    ``path`` is read from the call's arguments — ``order.quantity``, say.
    """

    def __init__(self, uri: str, path: str, value: Any):
        self.uri = uri
        self.path = path
        self.value = value

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        calls = record.trajectory.calls_to(self.uri)
        if not calls:
            return Score.boolean(
                self.name, False, reason=f"{self.uri} was never called"
            )
        seen = []
        for call in calls:
            args = (call.inputs or {}).get("args", call.inputs or {})
            actual = _value_at(args, self.path)
            seen.append(actual)
            if actual == self.value:
                return Score.boolean(self.name, True)
        return Score.boolean(
            self.name, False, reason=f"{self.path} was {seen}, expected {self.value!r}"
        )


@evaluator("max_agent_steps")
class MaxAgentSteps(_TrajectoryEvaluator):
    """The agent reached an answer within ``n`` reasoning steps."""

    def __init__(self, n: int, node: Optional[str] = None):
        self.n = int(n)
        self.node = node

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        steps = record.trajectory.agent_steps(self.node)
        ok = steps <= self.n
        return Score(
            name=self.name,
            value=float(steps),
            passed=ok,
            reason=None if ok else f"took {steps} steps > {self.n}",
        )


@evaluator("retrieval_hit_at_k")
class RetrievalHitAtK(_TrajectoryEvaluator):
    """The case's expected source was among the top ``k`` retrieved hits.

    The metric to insist on for a RAG workflow: it involves no LLM at all, so
    an embedding-model regression and a prompt regression produce *different*
    failures. Score only final answers and you cannot tell them apart, and you
    will spend a day tuning a prompt to fix a retrieval problem.

    Reads the ``rag_query`` node's task row, which holds the hits complete with
    their ``source_id`` and score.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        node: str = "retrieve",
        source: Optional[str] = None,
    ):
        #: How far down the list still counts as a hit. Left unset it means
        #: "anywhere the node actually returned", which is the question most
        #: suites are asking. Set it *below* the node's ``top_k`` only when you
        #: mean a stricter precision target — a mismatch you did not intend
        #: reports a miss on a case whose answer was right, which is the
        #: fastest way to teach people to ignore a red row.
        self.k = int(k) if k is not None else None
        self.node = node
        self.source = source

    def _expected_sources(self, case: Case) -> list[str]:
        if self.source:
            return [self.source]
        expected = case.expected if isinstance(case.expected, dict) else {}
        sources = expected.get("source_ids") or expected.get("source_id")
        if sources is None:
            return []
        return [sources] if isinstance(sources, str) else list(sources)

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        wanted = self._expected_sources(case)
        if not wanted:
            return Score(name=self.name, reason="no expected source for this case")

        node = record.trajectory.node(self.node)
        if node is None:
            return Score.boolean(
                self.name, False, reason=f"retrieval node '{self.node}' never ran"
            )
        all_ids = _retrieved_source_ids(node.output)
        cutoff = self.k if self.k is not None else len(all_ids)
        retrieved = all_ids[:cutoff]
        missing = [s for s in wanted if s not in retrieved]
        ok = not missing
        return Score(
            name=self.name,
            value=1.0 - (len(missing) / len(wanted)),
            passed=ok,
            reason=None
            if ok
            else f"{missing} not in the top {cutoff} retrieved: {retrieved}",
            meta={"retrieved": retrieved},
        )


def _retrieved_source_ids(output: Any) -> list[str]:
    """Pull ``source_id`` out of whatever a ``rag_query`` node stored."""
    if isinstance(output, dict):
        output = output.get("result", output)
    if not isinstance(output, list):
        return []
    ids = []
    for hit in output:
        if isinstance(hit, dict):
            value = hit.get("source_id")
            if value is not None:
                ids.append(str(value))
    return ids


@evaluator("groundedness")
class Groundedness(_TrajectoryEvaluator):
    """Every source the answer cited was actually retrieved.

    Checkable without a judge, which is the point of asking the workflow to
    declare its evidence in ``used_ids``: an id the model produced that
    retrieval never returned is a fabricated citation.
    """

    def __init__(self, node: str = "retrieve", field: str = "used_ids"):
        self.node = node
        self.field = field

    async def score(self, case: Case, record: RunRecord) -> Score:
        self.require_trajectory(record)
        node = record.trajectory.node(self.node)
        retrieved = set(_retrieved_source_ids(node.output)) if node else set()
        output = record.output if isinstance(record.output, dict) else {}
        cited = output.get(self.field) or []
        if not isinstance(cited, list):
            cited = [cited]
        invented = [str(c) for c in cited if str(c) not in retrieved]
        ok = not invented
        return Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            passed=ok,
            reason=None
            if ok
            else f"cited sources that were never retrieved: {invented}",
            meta={"cited": [str(c) for c in cited]},
        )
