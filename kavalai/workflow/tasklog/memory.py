"""
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

from pydantic import BaseModel

from kavalai.llm_clients.base_client import ModelCallStat
from kavalai.workflow.tasklog.base import TaskLogger


class TaskRecord(BaseModel):
    """One recorded node execution, field-for-field the ``tasks`` table.

    Deliberately the same shape as the persisted row: an evaluator written
    against a list of these reads identically whether the records came from
    memory or from ``SELECT * FROM tasks``.
    """

    run_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    name: str
    node_type: str
    inputs: Optional[dict] = None
    output: Any = None
    prompt: Optional[str] = None
    duration_seconds: float = 0.0
    errors: Optional[list[str]] = None
    seq: Optional[int] = None
    parent_task_name: Optional[str] = None
    tool_uri: Optional[str] = None


class MemoryTaskLogger(TaskLogger):
    """Task logger keeping every record in a list instead of a database.

    The evaluation runner's task logger, and the quickest way to see what a
    workflow actually did from a notebook: same recording path the Postgres
    backend uses in production, no database, no flush latency, and the records
    are available the moment the run returns.

    Records arrive in completion order because writes are fire-and-forget;
    :attr:`records` is sorted by ``seq``, which is execution order.
    """

    def __init__(self, max_payload_bytes: int = 0):
        # Nothing is written to a database, so the operational size cap that
        # protects the writer does not apply. Off by default keeps evaluators
        # looking at exactly what the tool returned.
        super().__init__(max_payload_bytes=max_payload_bytes)
        self.nodes: list[TaskRecord] = []
        self.model_calls: list[ModelCallStat] = []

    @property
    def records(self) -> list[TaskRecord]:
        """Every recorded node, in execution order."""
        return sorted(self.nodes, key=lambda r: (r.seq is None, r.seq or 0))

    def for_run(self, run_id: Optional[str]) -> list[TaskRecord]:
        """The records of one run, in execution order.

        One logger can be reused across runs (the engine holds it for its
        lifetime); this is how a single run's trajectory is picked out.
        """
        if run_id is None:
            return self.records
        return [r for r in self.records if r.run_id == run_id]

    def clear(self) -> None:
        """Drop everything recorded so far."""
        self.nodes.clear()
        self.model_calls.clear()

    async def _log_node_impl(
        self,
        *,
        run_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        node_name: str,
        node_type: str,
        inputs: Optional[dict],
        output: Any,
        prompt: Optional[str],
        duration: float,
        errors: Optional[list[str]],
        seq: Optional[int] = None,
        parent_task_name: Optional[str] = None,
        tool_uri: Optional[str] = None,
    ) -> None:
        self.nodes.append(
            TaskRecord(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                name=node_name,
                node_type=node_type,
                inputs=inputs,
                output=output,
                prompt=prompt,
                duration_seconds=duration,
                errors=errors,
                seq=seq,
                parent_task_name=parent_task_name,
                tool_uri=tool_uri,
            )
        )

    async def _log_model_call_impl(
        self, stats: ModelCallStat, agent_id: Optional[str]
    ) -> None:
        self.model_calls.append(stats)


class TeeTaskLogger(TaskLogger):
    """Writes every record to several loggers at once.

    The evaluation runner's use: grade against an in-memory trajectory *and*
    keep the same rows in Postgres, so a failing case in the result file can be
    opened in the backoffice by its ``external_id``. Without this, asking for a
    private trajectory would silently switch the database recording off.

    Each logger keeps its own payload cap, so a memory logger can hold a full
    tool result while the database one still truncates the 4 MB crawl.
    """

    def __init__(self, *loggers: TaskLogger):
        # The cap is applied by each wrapped logger; applying it here as well
        # would truncate twice and hide the untruncated value from every one.
        super().__init__(max_payload_bytes=0)
        self.loggers = [logger for logger in loggers if logger is not None]

    def log_node(self, **kwargs: Any) -> None:
        for logger in self.loggers:
            logger.log_node(**kwargs)

    def log_model_call(
        self, stats: ModelCallStat, agent_id: Optional[str] = None
    ) -> None:
        for logger in self.loggers:
            logger.log_model_call(stats, agent_id)

    async def flush(self) -> None:
        for logger in self.loggers:
            await logger.flush()

    async def _log_node_impl(self, **kwargs: Any) -> None:  # pragma: no cover
        """Unreachable: :meth:`log_node` fans out instead of spawning here."""

    async def _log_model_call_impl(self, *args: Any) -> None:  # pragma: no cover
        """Unreachable: :meth:`log_model_call` fans out instead of spawning here."""
