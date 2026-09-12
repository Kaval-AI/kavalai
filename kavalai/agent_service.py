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

from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, TypedDict
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import asc, case, delete, desc, func, null, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from kavalai.db import (
    Agent,
    ChatMessage,
    ModelCallStat,
    Run,
    Session,
    Task,
    json_array_length,
    json_typeof,
)
from kavalai.resolvers import resolve_path, find_key_recursive
from kavalai.utils import clean_text, to_plain


class SessionSummary(BaseModel):
    """One row of a session list.

    Aggregates a session's owning agent, its run/task/message and error
    counts, and a preview of its first and last messages. ``updated_at`` is
    the time of the session's last run.

    ``external_id`` is the caller-supplied key for the conversation. Evaluation
    runs record ``eval:{tag}:{case}``, so filtering on that prefix lands on
    exactly the conversation a failing case produced.
    """

    session_id: UUID
    agent_id: UUID
    agent_name: str
    external_id: str | None = None
    runs_count: int
    tasks_count: int
    messages_count: int
    first_message: str | None
    last_message: str | None
    errors_count: int
    created_at: datetime
    updated_at: datetime


class SessionsResponse(TypedDict):
    sessions: list[SessionSummary]
    total_count: int


def _orm_stat(stats: Any) -> ModelCallStat:
    """The ``model_call_stats`` row for a stats record of either kind.

    LLM clients report the Pydantic ``ModelCallStat``; a row built by hand is
    passed through.
    """
    if isinstance(stats, ModelCallStat):
        return stats
    # SQL NULL rather than JSON null for a payload that was not recorded, as a
    # purge writes it, so one ``IS NULL`` finds every call without a payload.
    return ModelCallStat(
        call_type=stats.call_type,
        model=stats.model or "",
        request_data=null() if stats.request_data is None else stats.request_data,
        response_data=null() if stats.response_data is None else stats.response_data,
        response_code=stats.response_code,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        total_tokens=stats.total_tokens,
        cached_prompt_tokens=stats.cached_prompt_tokens,
        reasoning_tokens=stats.reasoning_tokens,
        batch_size=stats.batch_size,
        duration_seconds=stats.duration_seconds,
    )


async def summarise_sessions(
    db_session: AsyncSession,
    agent_ids: Optional[Sequence[UUID]] = None,
    *,
    search: Optional[str] = None,
    external_id_prefix: Optional[str] = None,
    exclude_external_id_prefix: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> SessionsResponse:
    """List sessions, most recently active first, with their counts.

    The query behind :meth:`AgentService.list_sessions` and the backoffice
    conversation list, on a session the caller already holds.

    Args:
        agent_ids: Sessions of these agents only. ``None`` lists every
            agent's; an empty list lists nothing.
        search: Case-insensitive substring of any message in the session.
        external_id_prefix: The session's caller-supplied key starts with
            this, case-insensitively. Pasting ``eval:pr-412:`` shows every
            conversation that experiment produced.
        exclude_external_id_prefix: Leave out sessions whose key starts with
            this — a host's preview conversations, for example.
        start: Sessions created at or after this time.
        end: Sessions created at or before this time.
    """
    filters = []
    if agent_ids is not None:
        filters.append(Session.agent_id.in_(list(agent_ids)))
    if start:
        filters.append(Session.created_at >= start)
    if end:
        filters.append(Session.created_at <= end)
    if external_id_prefix:
        filters.append(Session.external_id.ilike(f"{external_id_prefix}%"))
    if exclude_external_id_prefix:
        filters.append(
            or_(
                Session.external_id.is_(None),
                ~Session.external_id.startswith(
                    exclude_external_id_prefix, autoescape=True
                ),
            )
        )
    if search:
        matching_sessions = select(ChatMessage.session_id).where(
            ChatMessage.content.ilike(f"%{search}%")
        )
        filters.append(Session.id.in_(matching_sessions))

    count_stmt = select(func.count()).select_from(
        select(Session.id).where(*filters).subquery()
    )
    total_count = (await db_session.execute(count_stmt)).scalar() or 0

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
                    json_typeof(Task.errors) == "array",
                    json_array_length(Task.errors) > 0,
                ),
                else_=json_typeof(Task.errors) != "null",
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
        .order_by(desc(Session.updated_at), desc(Session.id))
        .limit(limit)
        .offset(offset)
    )
    rows = (await db_session.execute(stmt)).all()

    session_ids = [row.session_id for row in rows]
    first_messages = await _boundary_messages(db_session, session_ids, asc)
    last_messages = await _boundary_messages(db_session, session_ids, desc)

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
    db_session: AsyncSession, session_ids: list[UUID], direction
) -> dict[UUID, str]:
    """The oldest (``asc``) or newest (``desc``) message content per session.

    One query for the whole page instead of one per session. A window
    function rather than ``DISTINCT ON``, which only Postgres has — SQLAlchemy
    would silently degrade it to a plain ``DISTINCT`` on SQLite.
    """
    if not session_ids:
        return {}
    ranked = (
        select(
            ChatMessage.session_id,
            ChatMessage.content,
            func.row_number()
            .over(
                partition_by=ChatMessage.session_id,
                order_by=direction(ChatMessage.created_at),
            )
            .label("rank"),
        )
        .where(ChatMessage.session_id.in_(session_ids))
        .subquery()
    )
    stmt = select(ranked.c.session_id, ranked.c.content).where(ranked.c.rank == 1)
    return dict((await db_session.execute(stmt)).all())


def _update_agent(
    agent: Agent,
    description: Optional[str],
    input_schema: Optional[Dict],
    output_schema: Optional[Dict],
    workflow: Optional[Dict],
) -> bool:
    """Apply the non-``None`` values to ``agent``; return whether any changed.

    ``None`` means "not supplied", so a caller that knows only the name never
    blanks the description or workflow another caller recorded.
    """
    updates = {
        "description": description,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "workflow": workflow,
    }
    changed = False
    for field, value in updates.items():
        if value is not None and getattr(agent, field) != value:
            setattr(agent, field, value)
            changed = True
    return changed


class AgentService:
    """Database operations for the agent runtime.

    Manages the core entities (agents, sessions, runs) as well as the history
    data recorded while they execute (chat messages, tasks, model-call stats).
    Works against Postgres and SQLite alike (the models are dialect-agnostic
    and schema-less; the schema comes from the engine's
    ``schema_translate_map``).
    """

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker

    async def get_or_create_agent(
        self,
        name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict] = None,
        output_schema: Optional[Dict] = None,
        workflow: Optional[Dict] = None,
    ) -> Agent:
        """Finds an agent by name or creates a new one if not found."""
        async with self.session_maker() as session:
            stmt = select(Agent).where(Agent.name == name)
            result = await session.execute(stmt)
            agent = result.scalar_one_or_none()

            if not agent:
                agent = Agent(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    workflow=workflow,
                )
                session.add(agent)
                await session.commit()
                await session.refresh(agent)
            elif _update_agent(
                agent, description, input_schema, output_schema, workflow
            ):
                await session.commit()
                await session.refresh(agent)

            return agent

    async def get_or_create_session(
        self,
        agent_id: UUID,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
    ) -> Optional[Session]:
        """Look up a session by id, or create a new one when no id is given."""
        async with self.session_maker() as session:
            if session_id:
                stmt = select(Session).where(Session.id == session_id)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

            new_session = Session(agent_id=agent_id, external_id=external_id)
            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)
            return new_session

    async def create_run(
        self,
        session_id: UUID,
        input_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Run:
        """Creates a new run entry for a specific session."""
        async with self.session_maker() as session:
            run = Run(
                session_id=session_id,
                input_data=to_plain(input_data),
                context=to_plain(context),
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def initialize_workflow_run(
        self,
        agent_name: str,
        agent_description: Optional[str] = None,
        input_schema: Optional[Dict] = None,
        output_schema: Optional[Dict] = None,
        workflow: Optional[Dict] = None,
        session_id: Optional[UUID] = None,
        external_id: Optional[str] = None,
        input_data: Optional[Dict] = None,
    ) -> tuple[Agent, Session, Run]:
        """Initialize agent, session, and run in a single database transaction.

        One session and one commit instead of three, which matters against a
        remote database.

        ``session_id`` selects an existing session by primary id (raises
        ``ValueError`` if absent). Without it, ``external_id`` reuses the
        agent's most recent session carrying that caller-supplied id — letting
        clients pin a conversation to their own identifier — and a new session
        is created when neither matches. A continued session's ``updated_at``
        is moved to now, so it records the session's last activity.

        Returns:
            tuple of (agent, session, run)
        """
        async with self.session_maker() as db_session:
            # 1. Get or create agent
            stmt = select(Agent).where(Agent.name == agent_name)
            result = await db_session.execute(stmt)
            agent = result.scalar_one_or_none()

            if not agent:
                agent = Agent(
                    name=agent_name,
                    description=agent_description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    workflow=workflow,
                )
                db_session.add(agent)
                await db_session.flush()  # Get agent.id for session creation
            else:
                _update_agent(
                    agent, agent_description, input_schema, output_schema, workflow
                )

            # 2. Get or create session
            if session_id:
                stmt = select(Session).where(Session.id == session_id)
                result = await db_session.execute(stmt)
                session_obj = result.scalar_one_or_none()
                if not session_obj:
                    raise ValueError(f"Session with ID {session_id} not found")
            else:
                session_obj = None
                if external_id:
                    stmt = (
                        select(Session)
                        .where(
                            Session.agent_id == agent.id,
                            Session.external_id == external_id,
                        )
                        .order_by(Session.created_at.desc())
                        .limit(1)
                    )
                    result = await db_session.execute(stmt)
                    session_obj = result.scalar_one_or_none()
                if session_obj is None:
                    session_obj = Session(agent_id=agent.id, external_id=external_id)
                    db_session.add(session_obj)
                    await db_session.flush()  # Get session_obj.id for run creation
            session_obj.updated_at = datetime.now(timezone.utc)

            # 3. Create run
            run = Run(
                session_id=session_obj.id,
                input_data=to_plain(input_data),
                context=None,
            )
            db_session.add(run)

            # Single commit for all operations
            await db_session.commit()

            # Refresh to get created_at timestamps
            await db_session.refresh(agent)
            await db_session.refresh(session_obj)
            await db_session.refresh(run)

            return (agent, session_obj, run)

    async def update_run(
        self,
        run_id: UUID,
        *,
        output_data: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Run:
        """Updates an existing run with final output_data and/or context."""
        async with self.session_maker() as session:
            stmt = select(Run).where(Run.id == run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if not run:
                raise ValueError(f"Run not found: {run_id}")
            if output_data is not None:
                run.output_data = to_plain(output_data)
            if context is not None:
                run.context = to_plain(context)
            await session.commit()
            await session.refresh(run)
            return run

    async def get_history_value(self, session_id: UUID, key: str) -> Optional[Any]:
        """
        Retrieves a value from the context of previous runs in the same session.

        - If `key` is a dotted path (e.g., "output.search_results"), resolves it as such.
        - If `key` is a plain name (e.g., "search_results"), searches recursively for the
          first matching key in the context dicts of previous runs (newest first).

        Returns the most recent value found for the given key.
        """
        resolve = resolve_path if "." in key else find_key_recursive

        async with self.session_maker() as session:
            stmt = (
                select(Run.context)
                .where(Run.session_id == session_id)
                .order_by(Run.created_at.desc())
            )
            result = await session.execute(stmt)
            for context in result.scalars():
                if not context:
                    continue
                val = resolve(context, key)
                if val is not None:
                    return val
            return None

    async def add_chat_message(
        self,
        agent_id: UUID,
        session_id: UUID,
        role: str,
        content: Optional[str],
        run_id: Optional[UUID] = None,
    ) -> ChatMessage:
        """Helper to append messages to the chat history."""
        async with self.session_maker() as session:
            message = ChatMessage(
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                role=role,
                content=clean_text(content or ""),
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message

    async def get_chat_history(
        self, session_id: UUID, limit: int = 50, max_chars: Optional[int] = None
    ) -> List[ChatMessage]:
        """The most recent messages of a session, ordered oldest to newest.

        Args:
            limit: At most this many messages.
            max_chars: At most this many characters of content. Whole messages
                are dropped from the oldest end until the rest fits; a message
                is never cut, and one that does not fit ends the window.
        """
        async with self.session_maker() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            newest_first = list(result.scalars().all())

        if max_chars is not None:
            kept, used = [], 0
            for message in newest_first:
                used += len(message.content or "")
                if used > max_chars:
                    break
                kept.append(message)
            newest_first = kept
        return list(reversed(newest_first))

    async def add_task(
        self,
        session_id: UUID,
        run_id: UUID,
        name: Optional[str] = None,
        agent_id: Optional[UUID] = None,
        inputs: Optional[Dict] = None,
        output: Optional[Dict] = None,
        prompt: Optional[str] = None,
        errors: Optional[list[str]] = None,
        duration_seconds: Optional[float] = None,
        node_type: Optional[str] = None,
        seq: Optional[int] = None,
        parent_task_name: Optional[str] = None,
        tool_uri: Optional[str] = None,
    ) -> Task:
        """Records a specific unit of work (Task) performed within a run.

        ``seq`` is the run's execution order, ``parent_task_name`` names the
        node a tool-call row belongs to, and ``tool_uri`` identifies the tool
        that ran. See :class:`kavalai.db.Task` for what they are for.
        """
        async with self.session_maker() as session:
            task = Task(
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                name=clean_text(name),
                node_type=node_type,
                inputs=to_plain(inputs),
                output=to_plain(output),
                prompt=clean_text(prompt),
                errors=to_plain(errors),
                duration_seconds=duration_seconds,
                seq=seq,
                parent_task_name=clean_text(parent_task_name),
                tool_uri=clean_text(tool_uri),
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

    async def add_model_call_stats(
        self,
        stats: Any,
        agent_id: Optional[UUID] = None,
        *,
        session_id: Optional[UUID] = None,
        run_id: Optional[UUID] = None,
    ) -> ModelCallStat:
        """Record one LLM or embedding call.

        Args:
            stats: The Pydantic ``ModelCallStat`` a client reported, or a
                ``model_call_stats`` row built by hand.
            agent_id: The agent the call is attributed to.
            session_id: The session of the run that made the call.
            run_id: The run that made the call.
        """
        row = _orm_stat(stats)
        async with self.session_maker() as session:
            if agent_id:
                row.agent_id = agent_id
            if session_id:
                row.session_id = session_id
            if run_id:
                row.run_id = run_id
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def get_model_call_stats(
        self,
        call_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        *,
        agent_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        run_id: Optional[UUID] = None,
    ) -> List[ModelCallStat]:
        """Model call records, newest first, optionally filtered.

        Args:
            call_type: ``"llm"`` or ``"embedding"``.
            agent_id: Calls attributed to this agent.
            session_id: Calls made by runs of this session.
            run_id: Calls made by this run.
        """
        async with self.session_maker() as session:
            stmt = (
                select(ModelCallStat)
                .order_by(ModelCallStat.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if call_type:
                stmt = stmt.where(ModelCallStat.call_type == call_type)
            if agent_id:
                stmt = stmt.where(ModelCallStat.agent_id == agent_id)
            if session_id:
                stmt = stmt.where(ModelCallStat.session_id == session_id)
            if run_id:
                stmt = stmt.where(ModelCallStat.run_id == run_id)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_sessions(
        self,
        agent_ids: Optional[Sequence[UUID]] = None,
        *,
        search: Optional[str] = None,
        external_id_prefix: Optional[str] = None,
        exclude_external_id_prefix: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SessionsResponse:
        """Sessions, most recently active first, with their counts.

        ``agent_ids`` scopes the list to several agents at once — the agents
        of one tenant, for example — and an empty list lists nothing. See
        :func:`summarise_sessions` for the other filters.
        """
        async with self.session_maker() as session:
            return await summarise_sessions(
                session,
                agent_ids,
                search=search,
                external_id_prefix=external_id_prefix,
                exclude_external_id_prefix=exclude_external_id_prefix,
                start=start,
                end=end,
                limit=limit,
                offset=offset,
            )

    async def purge_sessions(
        self,
        before: datetime,
        agent_ids: Optional[Sequence[UUID]] = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[list[UUID]]:
        """Delete sessions whose last activity is older than ``before``.

        Deletes in batches of ``batch_size``, one transaction each, and yields
        each batch's session ids once it is committed, so a caller can delete
        rows of its own keyed by session in step. A session's runs, tasks and
        chat messages go with it through the foreign keys. Its model calls
        stay, with their request and response payloads cleared: the token
        counts are a cost record, the payloads are the conversation again.

        Args:
            before: Sessions whose ``updated_at`` (last run) is earlier.
            agent_ids: Only these agents' sessions. ``None`` means every
                agent; an empty list purges nothing.
            batch_size: Sessions per transaction.
        """
        while True:
            async with self.session_maker() as session:
                stmt = select(Session.id).where(Session.updated_at < before)
                if agent_ids is not None:
                    stmt = stmt.where(Session.agent_id.in_(list(agent_ids)))
                stmt = stmt.order_by(Session.updated_at).limit(batch_size)
                ids = list((await session.execute(stmt)).scalars().all())
                if not ids:
                    return
                await self._clear_model_call_payloads(session, ids)
                await session.execute(delete(Session).where(Session.id.in_(ids)))
                await session.commit()
            yield ids

    @staticmethod
    async def _clear_model_call_payloads(
        session: AsyncSession, session_ids: list[UUID]
    ) -> None:
        await session.execute(
            update(ModelCallStat)
            .where(ModelCallStat.session_id.in_(session_ids))
            .values(request_data=null(), response_data=null())
        )

    async def delete_history_for_session(self, session_id: UUID) -> None:
        """Delete a session's chat messages and tasks.

        Its model calls keep their token counts and lose their payloads, as in
        :meth:`purge_sessions`.
        """
        async with self.session_maker() as session:
            await session.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            await session.execute(delete(Task).where(Task.session_id == session_id))
            await self._clear_model_call_payloads(session, [session_id])
            await session.commit()

    async def delete_history_for_agent(self, agent_id: UUID) -> None:
        """Delete all history (chat, tasks, stats) belonging to an agent."""
        async with self.session_maker() as session:
            await session.execute(
                delete(ChatMessage).where(ChatMessage.agent_id == agent_id)
            )
            await session.execute(delete(Task).where(Task.agent_id == agent_id))
            await session.execute(
                delete(ModelCallStat).where(ModelCallStat.agent_id == agent_id)
            )
            await session.commit()
