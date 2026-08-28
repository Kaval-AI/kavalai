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

from typing import Optional, Dict, List, Any
from uuid import UUID

from sqlalchemy import asc, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from kavalai.db import Agent, Session, Run, Task, ChatMessage, ModelCallStat
from kavalai.resolvers import resolve_path, find_key_recursive
from kavalai.utils import clean_text, to_plain


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

    # -- core entities: agents, sessions, runs --------------------------------

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
        is created when neither matches.

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

    # -- chat history ----------------------------------------------------------

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
        self, session_id: UUID, limit: int = 50
    ) -> List[ChatMessage]:
        """
        Retrieves the conversation history for a session,
        ordered from oldest to newest.
        """
        async with self.session_maker() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(asc(ChatMessage.created_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # -- task records ----------------------------------------------------------

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

    # -- model call stats ------------------------------------------------------

    async def add_model_call_stats(
        self, stats: ModelCallStat, agent_id: Optional[UUID] = None
    ) -> ModelCallStat:
        """Records LLM/Embedding call statistics."""
        async with self.session_maker() as session:
            if agent_id:
                stats.agent_id = agent_id
            session.add(stats)
            await session.commit()
            await session.refresh(stats)
            return stats

    async def get_model_call_stats(
        self,
        call_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ModelCallStat]:
        """
        Retrieves paginated model call stats, optionally filtered by call type.
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

            result = await session.execute(stmt)
            return list(result.scalars().all())

    # -- deletion ----------------------------------------------------------------

    async def delete_history_for_session(self, session_id: UUID) -> None:
        """Delete all history (chat, tasks) belonging to a session."""
        async with self.session_maker() as session:
            await session.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            await session.execute(delete(Task).where(Task.session_id == session_id))
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
