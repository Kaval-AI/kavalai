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
from kavalai.db import ModelCallStat as DbModelCallStat
from kavalai.workflow.tasklog.base import TaskLogger
from kavalai.llm_clients.base_client import ModelCallStat


def _uuid(value: Optional[str]) -> Optional[UUID]:
    return UUID(value) if value else None


def _to_orm_stat(stats: ModelCallStat) -> DbModelCallStat:
    """Convert a v2 (Pydantic) ModelCallStat into the persistable ORM row."""
    if isinstance(stats, DbModelCallStat):
        return stats
    return DbModelCallStat(
        call_type=stats.call_type,
        model=stats.model or "",
        request_data=stats.request_data,
        response_data=stats.response_data,
        response_code=stats.response_code,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        total_tokens=stats.total_tokens,
        batch_size=stats.batch_size,
        duration_seconds=stats.duration_seconds,
    )


class PostgresTaskLogger(TaskLogger):
    """Postgres-backed :class:`TaskLogger` delegating to :class:`AgentService`.

    Node executions become ``tasks`` rows (with their ``node_type``) and model
    calls become ``model_call_stats`` rows, keeping the backoffice dashboards
    populated for v2 runs.
    """

    def __init__(self, agent_service: AgentService):
        super().__init__()
        self.agent_service = agent_service

    @classmethod
    def from_session_maker(
        cls, session_maker: async_sessionmaker[AsyncSession]
    ) -> "PostgresTaskLogger":
        return cls(AgentService(session_maker))

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
        # A task row requires a run + session; skip if the engine ran without them.
        if not run_id or not session_id:
            return
        await self.agent_service.add_task(
            session_id=UUID(session_id),
            run_id=UUID(run_id),
            agent_id=_uuid(agent_id),
            name=node_name,
            node_type=node_type,
            inputs=inputs,
            output=output if isinstance(output, dict) else {"result": output},
            prompt=prompt,
            errors=errors,
            duration_seconds=duration,
            seq=seq,
            parent_task_name=parent_task_name,
            tool_uri=tool_uri,
        )

    async def _log_model_call_impl(
        self, stats: ModelCallStat, agent_id: Optional[str]
    ) -> None:
        await self.agent_service.add_model_call_stats(
            _to_orm_stat(stats), agent_id=_uuid(agent_id)
        )
