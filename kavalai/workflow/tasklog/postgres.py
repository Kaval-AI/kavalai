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
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.agent_service import AgentService
from kavalai.workflow.tasklog.base import DEFAULT_MAX_PAYLOAD_BYTES, TaskLogger
from kavalai.llm_clients.base_client import ModelCallStat


def _uuid(value: Optional[str]) -> Optional[UUID]:
    return UUID(value) if value else None


class PostgresTaskLogger(TaskLogger):
    """Database-backed :class:`TaskLogger` delegating to :class:`AgentService`.

    Node executions become ``tasks`` rows (with their ``node_type``) and model
    calls become ``model_call_stats`` rows carrying the agent, session and run
    that made them, which is what the backoffice dashboards read. Despite the
    name it writes to whatever database the service's sessionmaker points at,
    SQLite included.

    ``record_nodes``, ``record_payloads`` and ``max_payload_bytes`` are the
    :class:`TaskLogger` options.
    """

    def __init__(
        self,
        agent_service: AgentService,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        record_nodes: bool = True,
        record_payloads: bool = True,
    ):
        super().__init__(
            max_payload_bytes,
            record_nodes=record_nodes,
            record_payloads=record_payloads,
        )
        self.agent_service = agent_service

    @classmethod
    def from_session_maker(
        cls, session_maker: async_sessionmaker[AsyncSession], **options: Any
    ) -> "PostgresTaskLogger":
        return cls(AgentService(session_maker), **options)

    async def write_node(
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
        # A task row requires a run + session; skip if the engine ran without them.
        if not run_id or not session_id:
            return
        if output is not None and not isinstance(output, dict):
            output = {"result": output}
        await self.agent_service.add_task(
            session_id=UUID(session_id),
            run_id=UUID(run_id),
            agent_id=_uuid(agent_id),
            name=node_name,
            node_type=node_type,
            inputs=inputs,
            output=output,
            prompt=prompt,
            errors=errors,
            duration_seconds=duration,
            seq=seq,
            parent_task_name=parent_task_name,
            tool_uri=tool_uri,
        )

    async def write_model_call(
        self,
        stats: ModelCallStat,
        *,
        agent_id: Optional[str],
        session_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        await self.agent_service.add_model_call_stats(
            stats,
            agent_id=_uuid(agent_id),
            session_id=_uuid(session_id),
            run_id=_uuid(run_id),
        )
