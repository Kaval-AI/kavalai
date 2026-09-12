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

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import asc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.agent_service import SessionSummary, SessionsResponse, summarise_sessions
from kavalai.db import ChatMessage, Run, Task

__all__ = [
    "ChatMessageSummary",
    "RunSummary",
    "SessionDetails",
    "SessionSummary",
    "SessionsResponse",
    "TaskSummary",
    "get_session_details",
    "get_sessions_summary",
]


class TaskSummary(BaseModel):
    """Summary of a single task (workflow-node execution) for the Tasks view.

    Exposes the task's inputs, output, name, prompt, any errors and duration.

    ``seq`` is the run's execution order — order by it and you have the path
    the run actually took. ``parent_task_name`` is set on the tool calls an
    agent node made, so they render indented under it, and ``tool_uri`` names
    the tool that ran.
    """

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    agent_id: UUID | None
    session_id: UUID
    run_id: UUID
    inputs: Any | None
    output: Any | None
    name: str | None = None
    node_type: str | None = None
    prompt: str | None = None
    errors: list[str] | None = None
    duration_seconds: float | None = None
    seq: int | None = None
    parent_task_name: str | None = None
    tool_uri: str | None = None
    created_at: datetime
    updated_at: datetime


class RunSummary(BaseModel):
    """Summary of a single workflow run for the Runs view.

    Exposes the run's input/output data, resolved context and the number of
    tasks it executed.
    """

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    session_id: UUID
    input_data: Any | None
    output_data: Any | None
    context: Any | None
    tasks_count: int
    created_at: datetime
    updated_at: datetime


class ChatMessageSummary(BaseModel):
    """Summary of a single chat message for the conversation transcript.

    Exposes the message's role, content and the run it is associated with.
    """

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    agent_id: UUID
    session_id: UUID
    run_id: UUID | None
    role: str
    content: str
    created_at: datetime
    updated_at: datetime


class SessionDetails(BaseModel):
    """Full detail of one session: its messages, runs and tasks.

    Powers the per-conversation detail view in the backoffice, bundling the
    session's chat transcript together with all of its runs and tasks.
    """

    session_id: UUID
    messages: list[ChatMessageSummary]
    runs: list[RunSummary]
    tasks: list[TaskSummary]


async def get_sessions_summary(
    session: AsyncSession,
    agent_id: UUID | None = None,
    search: str | None = None,
    external_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> SessionsResponse:
    """List sessions, most recently active first, with their counts.

    The conversation list's query, which is
    :func:`kavalai.agent_service.summarise_sessions` scoped to at most one
    agent. ``search`` matches message content; ``external_id`` matches the
    session's caller-supplied key as a prefix. The two are separate because
    they answer different questions — "what did someone ask about" versus
    "show me this exact conversation".
    """
    return await summarise_sessions(
        session,
        [agent_id] if agent_id else None,
        search=search,
        external_id_prefix=external_id,
        start=start_date,
        end=end_date,
        limit=limit,
        offset=offset,
    )


async def get_session_details(
    session: AsyncSession,
    session_id: UUID,
) -> SessionDetails:
    """The transcript, runs and tasks of one session."""
    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(asc(ChatMessage.created_at))
    )
    msg_result = await session.execute(msg_stmt)
    messages = [
        ChatMessageSummary.model_validate(m) for m in msg_result.scalars().all()
    ]

    tasks_count_sub = (
        select(Task.run_id, func.count(Task.id).label("count"))
        .where(Task.session_id == session_id)
        .group_by(Task.run_id)
        .subquery()
    )

    run_stmt = (
        select(
            Run.id,
            Run.session_id,
            Run.input_data,
            Run.output_data,
            Run.context,
            func.coalesce(tasks_count_sub.c.count, 0).label("tasks_count"),
            Run.created_at,
            Run.updated_at,
        )
        .outerjoin(tasks_count_sub, Run.id == tasks_count_sub.c.run_id)
        .where(Run.session_id == session_id)
        .order_by(asc(Run.created_at))
    )
    run_result = await session.execute(run_stmt)
    runs = [RunSummary.model_validate(r) for r in run_result.all()]

    # Ordered by run and then by ``seq``, which is the run's
    # actual execution order — ``created_at`` is approximate, and ties are
    # unordered once a `parallel` node has several branches writing at once.
    task_stmt = (
        select(Task)
        .where(Task.session_id == session_id)
        .order_by(asc(Task.run_id), asc(Task.seq), asc(Task.created_at))
    )
    task_result = await session.execute(task_stmt)
    tasks = [TaskSummary.model_validate(t) for t in task_result.scalars().all()]

    return SessionDetails(
        session_id=session_id, messages=messages, runs=runs, tasks=tasks
    )
