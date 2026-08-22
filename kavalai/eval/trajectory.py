"""An ordered, queryable view over what a run actually did.

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

from pydantic import BaseModel, Field

from kavalai.workflow.tasklog.memory import TaskRecord

#: Node types that are not a step of the graph but a call made from inside one.
TOOL_CALL = "tool_call"
#: Node types that record a routing decision rather than work.
BRANCH_TYPES = ("if", "switch")


class Trajectory(BaseModel):
    """What a run did, in order, with the questions evaluators actually ask.

    Built from :class:`~kavalai.workflow.tasklog.TaskRecord` records — the same
    shape as the ``tasks`` table — so an assertion written against a suite that
    runs in memory would read identically against recorded production rows.

    Empty when the target could not observe one (a deployed agent behind HTTP).
    Trajectory evaluators raise on an empty trajectory rather than scoring it as
    a pass: a gate that reports green because it could not see anything is the
    worst failure mode there is.
    """

    records: list[TaskRecord] = Field(default_factory=list)

    @property
    def observed(self) -> bool:
        """Whether the target could see a trajectory at all."""
        return bool(self.records)

    def names(self) -> list[str]:
        """The executed path: node names in execution order, tool calls aside."""
        return [r.name for r in self.records if r.node_type != TOOL_CALL]

    def node(self, name: str) -> Optional[TaskRecord]:
        """The first visit to ``name``, or ``None``."""
        return next((r for r in self.records if r.name == name), None)

    def nodes(self, name: str) -> list[TaskRecord]:
        """Every visit to ``name``, in order."""
        return [r for r in self.records if r.name == name]

    def visited(self, name: str) -> bool:
        return any(r.name == name for r in self.records)

    def tools(self) -> list[TaskRecord]:
        """Every tool that ran: function nodes and agent-chosen calls alike.

        This is what ``tool_uri`` buys — one question, one answer, regardless
        of whether a human wired the call into the YAML or the agent picked it
        at step 3.
        """
        return [r for r in self.records if r.tool_uri]

    def tool_uris(self) -> list[str]:
        return [r.tool_uri for r in self.tools() if r.tool_uri]

    def called(self, uri: str) -> bool:
        return any(r.tool_uri == uri for r in self.tools())

    def calls_to(self, uri: str) -> list[TaskRecord]:
        return [r for r in self.tools() if r.tool_uri == uri]

    def branch(self, node: str) -> Optional[TaskRecord]:
        """The routing decision made at ``node``, if it is a branch node."""
        return next(
            (r for r in self.records if r.name == node and r.node_type in BRANCH_TYPES),
            None,
        )

    def branches(self) -> list[TaskRecord]:
        return [r for r in self.records if r.node_type in BRANCH_TYPES]

    def taken(self, node: str) -> Optional[str]:
        """Which arm ``node`` routed to."""
        record = self.branch(node)
        return (record.output or {}).get("taken") if record else None

    def agent_steps(self, node: Optional[str] = None) -> int:
        """How many reasoning steps an agent node took.

        The step index is a field on each tool-call row rather than a level of
        nesting, so the count is one past the highest index seen. A step that
        called no tool leaves no row, so this is a lower bound — which is the
        right direction for a ``max_agent_steps`` assertion.
        """
        steps = [
            (r.inputs or {}).get("step")
            for r in self.records
            if r.node_type == TOOL_CALL and (node is None or r.parent_task_name == node)
        ]
        indices = [s for s in steps if isinstance(s, int)]
        return max(indices) + 1 if indices else 0

    def children_of(self, node: str) -> list[TaskRecord]:
        """The tool calls one *visit* to ``node`` made.

        ``parent_task_name`` is unique per node, not per visit, and the engine
        permits a node to be visited more than once. ``seq`` recovers the
        grouping: children always follow their parent and precede the next
        parent row, so segmenting the ordered records on parent boundaries is
        exact. Done here, once, so no evaluator has to think about it.
        """
        groups = self.child_groups(node)
        return groups[0] if groups else []

    def child_groups(self, node: str) -> list[list[TaskRecord]]:
        """One list of tool calls per visit to ``node``, in visit order."""
        groups: list[list[TaskRecord]] = []
        collecting = False
        for record in self.records:
            if record.node_type == TOOL_CALL:
                if collecting and record.parent_task_name == node:
                    groups[-1].append(record)
                continue
            # A non-tool row is a new node visit, which closes the previous
            # group and opens one only when it is the node we are after.
            collecting = record.name == node
            if collecting:
                groups.append([])
        return groups

    def as_table(self) -> list[dict[str, Any]]:
        """A flat, printable view — what ``--show-trajectory`` renders."""
        return [
            {
                "seq": r.seq,
                "name": r.name,
                "type": r.node_type,
                "tool": r.tool_uri,
                "parent": r.parent_task_name,
                "seconds": round(r.duration_seconds, 3),
            }
            for r in self.records
        ]
