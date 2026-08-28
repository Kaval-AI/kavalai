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
from uuid import UUID
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, desc, asc, case
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.db import Session, Run, Task, ChatMessage, Agent
from typing import TypedDict, Any


class SessionSummary(BaseModel):
    """Row-level summary of a session for the Conversations list.

    Aggregates a session's owning agent, its run/task/message and error counts,
    and a preview of its first and last messages.
    """

    session_id: UUID
    agent_id: UUID
    agent_name: str
    #: The caller-supplied key for this conversation. Evaluation runs record
    #: ``eval:{tag}:{case}``, so pasting one into the filter lands on exactly
    #: the conversation a failing case produced.
    external_id: str | None = None
    runs_count: int
    tasks_count: int
    messages_count: int
    first_message: str | None
    last_message: str | None
    errors_count: int
    created_at: datetime
    updated_at: datetime


class TaskSummary(BaseModel):
    """Summary of a single task (workflow-node execution) for the Tasks view.

    Exposes the task's inputs, output, name, prompt, any errors and duration.
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
    #: Trajectory columns. ``seq`` is the run's execution order — order by it
    #: and you have the path the run actually took. ``parent_task_name`` is set
    #: on the tool calls an agent node made, so they render indented under it,
    #: and ``tool_uri`` names the tool that ran.
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


class SessionsResponse(TypedDict):
    sessions: list[SessionSummary]
    total_count: int


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
    """List sessions, newest first, with their run/task/message counts.

    ``search`` matches message content; ``external_id`` matches the session's
    caller-supplied key as a prefix. The two are separate because they answer
    different questions — "what did someone ask about" versus "show me this
    exact conversation". Pasting ``eval:pr-412:`` into the latter shows every
    conversation that experiment produced.
    """
    filters = []
    if agent_id:
        filters.append(Session.agent_id == agent_id)
    if start_date:
        filters.append(Session.created_at >= start_date)
    if end_date:
        filters.append(Session.created_at <= end_date)
    if external_id:
        filters.append(Session.external_id.ilike(f"{external_id}%"))
    if search:
        matching_sessions = select(ChatMessage.session_id).where(
            ChatMessage.content.ilike(f"%{search}%")
        )
        filters.append(Session.id.in_(matching_sessions))

    count_stmt = select(func.count()).select_from(
        select(Session.id).where(*filters).subquery()
    )
    total_count = (await session.execute(count_stmt)).scalar() or 0

    runs_count_sub = (
        select(Run.session_id, func.count(Run.id).label("count"))
        .group_by(Run.session_id)
        .subquery()
    )
    tasks_count_sub = (
        select(Task.session_id, func.count(Task.id).label("count"))
        .group_by(Task.session_id)
        .subquery()
    )
    messages_count_sub = (
        select(ChatMessage.session_id, func.count(ChatMessage.id).label("count"))
        .group_by(ChatMessage.session_id)
        .subquery()
    )
    errors_count_sub = (
        select(Task.session_id, func.count(Task.id).label("count"))
        .where(Task.errors.is_not(None))
        .where(
            case(
                (
                    func.jsonb_typeof(Task.errors) == "array",
                    func.jsonb_array_length(Task.errors) > 0,
                ),
                else_=func.jsonb_typeof(Task.errors) != "null",
            )
        )
        .group_by(Task.session_id)
        .subquery()
    )

    stmt = (
        select(
            Session.id.label("session_id"),
            Session.agent_id,
            Agent.name.label("agent_name"),
            Session.external_id,
            func.coalesce(runs_count_sub.c.count, 0).label("runs_count"),
            func.coalesce(tasks_count_sub.c.count, 0).label("tasks_count"),
            func.coalesce(messages_count_sub.c.count, 0).label("messages_count"),
            func.coalesce(errors_count_sub.c.count, 0).label("errors_count"),
            Session.created_at,
            Session.updated_at,
        )
        .join(Agent, Session.agent_id == Agent.id)
        .outerjoin(runs_count_sub, Session.id == runs_count_sub.c.session_id)
        .outerjoin(tasks_count_sub, Session.id == tasks_count_sub.c.session_id)
        .outerjoin(messages_count_sub, Session.id == messages_count_sub.c.session_id)
        .outerjoin(errors_count_sub, Session.id == errors_count_sub.c.session_id)
        .where(*filters)
        .order_by(desc(Session.updated_at))
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).all()

    session_ids = [row.session_id for row in rows]
    first_messages = await _boundary_messages(session, session_ids, asc)
    last_messages = await _boundary_messages(session, session_ids, desc)

    summaries = [
        SessionSummary(
            session_id=row.session_id,
            agent_id=row.agent_id,
            agent_name=row.agent_name,
            external_id=row.external_id,
            runs_count=row.runs_count,
            tasks_count=row.tasks_count,
            messages_count=row.messages_count,
            errors_count=row.errors_count,
            first_message=first_messages.get(row.session_id),
            last_message=last_messages.get(row.session_id),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    return {"sessions": summaries, "total_count": total_count}


async def _boundary_messages(
    session: AsyncSession, session_ids: list[UUID], direction
) -> dict[UUID, str]:
    """The oldest (``asc``) or newest (``desc``) message content per session.

    One ``DISTINCT ON`` query for the whole page instead of one query per
    session.
    """
    if not session_ids:
        return {}
    stmt = (
        select(ChatMessage.session_id, ChatMessage.content)
        .distinct(ChatMessage.session_id)
        .where(ChatMessage.session_id.in_(session_ids))
        .order_by(ChatMessage.session_id, direction(ChatMessage.created_at))
    )
    return dict((await session.execute(stmt)).all())


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
