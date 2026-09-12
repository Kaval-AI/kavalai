from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select, update

from kavalai.agent_service import AgentService, _orm_stat
from kavalai.db import (
    ChatMessage,
    DatabaseManager,
    ModelCallStat,
    Run,
    Session,
    Task,
)
from kavalai.llm_clients.base_client import ModelCallStat as PydModelCallStat


def _utc(moment: datetime) -> datetime:
    """SQLite returns naive UTC timestamps, Postgres aware ones."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def _set_last_activity(session_maker, session_id, when: datetime) -> None:
    async with session_maker() as db:
        await db.execute(
            update(Session).where(Session.id == session_id).values(updated_at=when)
        )
        await db.commit()


async def _count(session_maker, model, *where) -> int:
    async with session_maker() as db:
        stmt = select(func.count()).select_from(model).where(*where)
        return (await db.execute(stmt)).scalar()


async def _session_with_history(service, session_maker, agent, last_activity):
    """A session with one run, task, message and model call, last active then."""
    _, session, run = await service.initialize_workflow_run(agent_name=agent.name)
    await service.add_chat_message(
        agent.id, session.id, "user", "My card number is 4111.", run_id=run.id
    )
    await service.add_task(session_id=session.id, run_id=run.id, name="answer")
    await service.add_model_call_stats(
        PydModelCallStat(
            call_type="llm",
            model="fake/scripted",
            request_data='{"messages": ["My card number is 4111."]}',
            response_data='"Noted."',
            total_tokens=5,
        ),
        agent.id,
        session_id=session.id,
        run_id=run.id,
    )
    await _set_last_activity(session_maker, session.id, last_activity)
    return session


@pytest.fixture(params=["postgres", "sqlite-compat"])
def session_maker(request, agents_session_maker):
    """The service, run against both session implementations.

    The browser leg matters: under Pyodide there is no greenlet and no
    ``aiosqlite``, so ``AgentService`` runs over ``AsyncSessionShim``, a
    hand-written facade. A method the service starts using but the shim never
    implemented does not fail here at import — it fails for a user, mid-session,
    in the browser. Running the same suite over both keeps the facade honest.
    """
    if request.param == "postgres":
        return agents_session_maker
    return DatabaseManager().get_sqlite_compat_sessionmaker()


@pytest.mark.asyncio
class TestAgentService:
    async def test_get_or_create_agent(self, session_maker):
        service = AgentService(session_maker)

        # Test Creation
        agent = await service.get_or_create_agent(
            name="ResearchAgent",
            description="Tests the agent creation",
            workflow={"steps": ["start", "end"]},
        )
        assert agent.name == "ResearchAgent"
        assert agent.workflow["steps"] == ["start", "end"]

        # Test Retrieval
        existing_agent = await service.get_or_create_agent(name="ResearchAgent")
        assert existing_agent.id == agent.id

    async def test_get_or_create_session_logic(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="SessionTest")

        # 1. Test creation when no session_id is provided
        session = await service.get_or_create_session(agent_id=agent.id)
        assert session.id is not None

        # 2. Test retrieval with existing session_id
        retrieved = await service.get_or_create_session(
            agent_id=agent.id, session_id=session.id
        )
        assert retrieved.id == session.id

        # 3. Test non-existent session_id returns None
        not_found = await service.get_or_create_session(
            agent_id=agent.id, session_id=uuid4()
        )
        assert not_found is None

    async def test_run_and_task_tracking(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="TaskTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Create Run
        run = await service.create_run(
            session_id=session.id, input_data={"user_query": "search for AI news"}
        )
        assert run.id is not None

        # Add Task to Run
        task = await service.add_task(
            session_id=session.id,
            run_id=run.id,
            agent_id=agent.id,
            name="TestTask",
            inputs={"query": "AI news"},
            output={"results": ["result1"]},
            duration_seconds=1.5,
        )
        assert task.run_id == run.id
        assert task.name == "TestTask"
        assert task.output["results"] == ["result1"]
        assert task.duration_seconds == 1.5

    async def test_chat_history_retrieval(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="ChatTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Add a few messages
        await service.add_chat_message(agent.id, session.id, "user", "Message 1")
        await service.add_chat_message(agent.id, session.id, "assistant", "Response 1")

        history = await service.get_chat_history(session.id)

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Message 1"
        assert history[1].role == "assistant"
        assert history[1].content == "Response 1"

    async def test_get_model_call_stats(self, agents_session_maker, agents_db):
        # Postgres only: agents_db truncates the tables so the count assertions
        # below see only the stats created here, whatever the collection order.
        service = AgentService(agents_session_maker)

        # Create some call stats
        async with agents_session_maker() as agents_db:
            for i in range(10):
                stat = ModelCallStat(
                    call_type="llm",
                    model="gpt-4o",
                    response_code=200,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    duration_seconds=0.1,
                    request_data={"query": f"test {i}"},
                    response_data={"answer": f"result {i}"},
                    cached_prompt_tokens=4,
                )
                agents_db.add(stat)
            await agents_db.commit()

        # Test retrieval all
        stats = await service.get_model_call_stats()
        assert len(stats) == 10

        # Test filter by type
        stats = await service.get_model_call_stats(call_type="llm")
        assert len(stats) == 10

        # Test filter by non-existent type
        stats = await service.get_model_call_stats(call_type="embedding")
        assert len(stats) == 0

        # Test pagination
        stats = await service.get_model_call_stats(limit=5, offset=0)
        assert len(stats) == 5

        stats = await service.get_model_call_stats(limit=5, offset=5)
        assert len(stats) == 5

    async def test_get_history_value(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="HistoryTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        # Create Run 1
        run1 = await service.create_run(session_id=session.id)
        await service.update_run(
            run_id=run1.id,
            context={"search_results": "result1", "other": "val1", "nested": {"a": 1}},
        )

        # Create Run 2
        run2 = await service.create_run(session_id=session.id)
        await service.update_run(
            run_id=run2.id,
            context={
                "search_results": "result2",
                "something": "else",
                "nested": {"b": 2},
            },
        )

        # Test single retrieval (most recent)
        val = await service.get_history_value(session.id, "search_results")
        assert val == "result2"

        val_other = await service.get_history_value(session.id, "other")
        assert val_other == "val1"

        # Test path retrieval
        val_nested = await service.get_history_value(session.id, "nested.b")
        assert val_nested == 2

        # Test non-existent key
        val_none = await service.get_history_value(session.id, "non_existent")
        assert val_none is None

    async def test_initialize_workflow_run_new_agent(self, session_maker):
        """Test batch initialization creates new agent, session, and run."""
        service = AgentService(session_maker)

        input_data = {"user_query": "test query"}
        agent, session, run = await service.initialize_workflow_run(
            agent_name="TestWorkflowAgent",
            agent_description="Test description",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            workflow={"tasks": []},
            input_data=input_data,
        )

        # Verify agent was created
        assert agent.id is not None
        assert agent.name == "TestWorkflowAgent"
        assert agent.description == "Test description"
        assert agent.workflow == {"tasks": []}

        # Verify session was created
        assert session.id is not None
        assert session.agent_id == agent.id

        # Verify run was created
        assert run.id is not None
        assert run.session_id == session.id
        assert run.input_data == input_data

    async def test_initialize_workflow_run_existing_agent(self, session_maker):
        """Test batch initialization reuses existing agent."""
        service = AgentService(session_maker)

        # Create agent first
        existing_agent = await service.get_or_create_agent(
            name="ExistingAgent",
            description="Original description",
            workflow={"v": 1},
        )

        # Initialize workflow with same agent name
        agent, session, run = await service.initialize_workflow_run(
            agent_name="ExistingAgent",
            agent_description="Updated description",
            workflow={"v": 2},
            input_data={"query": "test"},
        )

        # Should reuse same agent
        assert agent.id == existing_agent.id
        # Should update description and workflow
        assert agent.description == "Updated description"
        assert agent.workflow == {"v": 2}

        # Should create new session and run
        assert session.id is not None
        assert session.agent_id == agent.id
        assert run.id is not None
        assert run.session_id == session.id

    async def test_initialize_workflow_run_with_existing_session(self, session_maker):
        """Test batch initialization with existing session_id."""
        service = AgentService(session_maker)

        # Create agent and session first
        agent = await service.get_or_create_agent(name="SessionReuseAgent")
        existing_session = await service.get_or_create_session(agent_id=agent.id)

        # Initialize workflow with existing session_id
        agent_result, session_result, run = await service.initialize_workflow_run(
            agent_name="SessionReuseAgent",
            session_id=existing_session.id,
            input_data={"query": "test"},
        )

        # Should reuse same session
        assert session_result.id == existing_session.id
        assert session_result.agent_id == agent.id

        # Should create new run
        assert run.id is not None
        assert run.session_id == existing_session.id

    async def test_initialize_workflow_run_with_external_id(self, session_maker):
        """Test batch initialization with external_id for session."""
        service = AgentService(session_maker)

        agent, session, run = await service.initialize_workflow_run(
            agent_name="ExternalIdAgent",
            external_id="user-123-session",
            input_data={"query": "test"},
        )

        # Verify session has external_id
        assert session.external_id == "user-123-session"
        assert session.agent_id == agent.id

        # The same external_id reuses the session on later runs (new run each
        # time); a different one starts a fresh session.
        _, session2, run2 = await service.initialize_workflow_run(
            agent_name="ExternalIdAgent",
            external_id="user-123-session",
        )
        assert session2.id == session.id
        assert run2.id != run.id

        _, session3, _ = await service.initialize_workflow_run(
            agent_name="ExternalIdAgent",
            external_id="user-456-session",
        )
        assert session3.id != session.id

    async def test_initialize_workflow_run_invalid_session_id(self, session_maker):
        """Test batch initialization with non-existent session_id raises error."""
        service = AgentService(session_maker)

        with pytest.raises(ValueError, match="Session with ID .* not found"):
            await service.initialize_workflow_run(
                agent_name="InvalidSessionAgent",
                session_id=uuid4(),  # Non-existent session
                input_data={"query": "test"},
            )

    async def test_chat_history_windowing(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="ChatWindowTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        for i in range(5):
            await service.add_chat_message(
                agent_id=agent.id,
                session_id=session.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
            )

        messages = await service.get_chat_history(session.id)
        assert [m.content for m in messages] == [f"message {i}" for i in range(5)]

        window = await service.get_chat_history(session.id, limit=2)
        assert [m.content for m in window] == ["message 3", "message 4"]

    async def test_add_model_call_stats_assigns_agent(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="StatAgentTest")

        stat = await service.add_model_call_stats(
            ModelCallStat(call_type="llm", model="test/model"), agent_id=agent.id
        )
        assert stat.agent_id == agent.id

    async def test_delete_history_for_session(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="DeleteSessionTest")
        session = await service.get_or_create_session(agent_id=agent.id)
        other_session = await service.get_or_create_session(agent_id=agent.id)

        for sess in (session, other_session):
            await service.add_chat_message(
                agent_id=agent.id, session_id=sess.id, role="user", content="hello"
            )
            run = await service.create_run(session_id=sess.id)
            await service.add_task(session_id=sess.id, run_id=run.id, name="node-1")

        await service.delete_history_for_session(session.id)
        assert await service.get_chat_history(session.id) == []
        assert len(await service.get_chat_history(other_session.id)) == 1

    async def test_delete_history_for_session_blanks_model_call_payloads(
        self, session_maker
    ):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name=f"Blank-{uuid4()}")
        now = datetime.now(timezone.utc)
        session = await _session_with_history(service, session_maker, agent, now)
        kept = await _session_with_history(service, session_maker, agent, now)

        await service.delete_history_for_session(session.id)

        (call,) = await service.get_model_call_stats(session_id=session.id)
        assert call.total_tokens == 5
        blank = (
            ModelCallStat.request_data.is_(None),
            ModelCallStat.response_data.is_(None),
        )
        assert (
            await _count(
                session_maker,
                ModelCallStat,
                ModelCallStat.session_id == session.id,
                *blank,
            )
            == 1
        )
        assert (
            await _count(
                session_maker,
                ModelCallStat,
                ModelCallStat.session_id == kept.id,
                *blank,
            )
            == 0
        )
        assert await _count(session_maker, Task, Task.session_id == session.id) == 0
        # The session and its run stay; only the history goes.
        assert await _count(session_maker, Run, Run.session_id == session.id) == 1

    async def test_delete_history_for_agent(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="DeleteAgentTest")
        session = await service.get_or_create_session(agent_id=agent.id)

        await service.add_chat_message(
            agent_id=agent.id, session_id=session.id, role="user", content="hello"
        )
        await service.add_model_call_stats(
            ModelCallStat(call_type="llm", model="m"), agent_id=agent.id
        )

        await service.delete_history_for_agent(agent.id)
        assert await service.get_chat_history(session.id) == []
        stats = await service.get_model_call_stats(call_type="llm", limit=100)
        assert not any(s.agent_id == agent.id for s in stats)

    async def test_get_or_create_agent_updates_changed_fields(self, session_maker):
        service = AgentService(session_maker)
        created = await service.get_or_create_agent(
            name="Updatable", description="first"
        )

        updated = await service.get_or_create_agent(
            name="Updatable",
            description="second",
            workflow={"name": "wf"},
        )

        assert updated.id == created.id
        assert updated.description == "second"
        assert updated.workflow == {"name": "wf"}

        # Passing the same values again is a no-op.
        unchanged = await service.get_or_create_agent(
            name="Updatable", description="second"
        )
        assert unchanged.description == "second"

    async def test_update_run_rejects_an_unknown_run(self, session_maker):
        service = AgentService(session_maker)
        with pytest.raises(ValueError, match="Run not found"):
            await service.update_run(run_id=uuid4(), output_data={"a": 1})

    async def test_update_run_stores_output_data(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="OutputBot")
        session = await service.get_or_create_session(agent_id=agent.id)
        run = await service.create_run(session_id=session.id)

        stored = await service.update_run(
            run_id=run.id, output_data={"agent_response": "hi"}
        )

        assert stored.output_data == {"agent_response": "hi"}

    async def test_get_history_value_skips_runs_without_context(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name="SparseHistory")
        session = await service.get_or_create_session(agent_id=agent.id)

        run = await service.create_run(session_id=session.id)
        await service.update_run(run_id=run.id, context={"answer": "kept"})
        # A newer run that never recorded a context must not mask the older one.
        await service.create_run(session_id=session.id)

        assert await service.get_history_value(session.id, "answer") == "kept"

    async def test_a_continued_session_records_its_last_activity(self, session_maker):
        service = AgentService(session_maker)
        name = f"Activity-{uuid4()}"
        _, session, _ = await service.initialize_workflow_run(
            agent_name=name, external_id="visitor-1"
        )
        long_ago = datetime(2020, 1, 1, tzinfo=timezone.utc)

        await _set_last_activity(session_maker, session.id, long_ago)
        _, by_key, _ = await service.initialize_workflow_run(
            agent_name=name, external_id="visitor-1"
        )
        assert by_key.id == session.id
        assert _utc(by_key.updated_at) > long_ago

        await _set_last_activity(session_maker, session.id, long_ago)
        _, by_id, _ = await service.initialize_workflow_run(
            agent_name=name, session_id=session.id
        )
        assert _utc(by_id.updated_at) > long_ago

    async def test_chat_history_character_budget(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name=f"Budget-{uuid4()}")
        session = await service.get_or_create_session(agent_id=agent.id)
        for content in ("aaaa", "bb", "cccccc", "d"):
            await service.add_chat_message(agent.id, session.id, "user", content)

        async def window(**kwargs):
            messages = await service.get_chat_history(session.id, **kwargs)
            return [message.content for message in messages]

        assert await window(max_chars=7) == ["cccccc", "d"]
        # "bb" would fit in what is left, but the window stays contiguous: the
        # first message that does not fit ends it.
        assert await window(max_chars=6) == ["d"]
        assert await window(max_chars=0) == []
        assert await window(max_chars=100) == ["aaaa", "bb", "cccccc", "d"]
        assert await window(limit=2, max_chars=100) == ["cccccc", "d"]

    async def test_model_calls_carry_their_run(self, session_maker):
        service = AgentService(session_maker)
        agent, session, run = await service.initialize_workflow_run(
            agent_name=f"Calls-{uuid4()}"
        )
        _, _, later = await service.initialize_workflow_run(
            agent_name=agent.name, session_id=session.id
        )

        row = await service.add_model_call_stats(
            PydModelCallStat(
                call_type="llm", model="fake/x", total_tokens=7, cached_prompt_tokens=2
            ),
            agent.id,
            session_id=session.id,
            run_id=run.id,
        )
        assert isinstance(row, ModelCallStat)
        assert (row.agent_id, row.session_id, row.run_id) == (
            agent.id,
            session.id,
            run.id,
        )
        assert row.cached_prompt_tokens == 2
        await service.add_model_call_stats(
            PydModelCallStat(call_type="embedding", model="fake/e", total_tokens=3),
            agent.id,
            session_id=session.id,
            run_id=later.id,
        )

        by_run = await service.get_model_call_stats(run_id=run.id)
        assert [call.total_tokens for call in by_run] == [7]
        by_session = await service.get_model_call_stats(session_id=session.id)
        assert sorted(call.total_tokens for call in by_session) == [3, 7]
        embeddings = await service.get_model_call_stats(
            call_type="embedding", agent_id=agent.id
        )
        assert [call.total_tokens for call in embeddings] == [3]

    async def test_list_sessions_across_agents(self, session_maker):
        service = AgentService(session_maker)
        tenant_a, tenant_b, elsewhere = [
            await service.get_or_create_agent(name=f"List-{i}-{uuid4()}")
            for i in range(3)
        ]
        now = datetime.now(timezone.utc)

        async def session_for(agent, external_id, minutes_ago):
            _, session, run = await service.initialize_workflow_run(
                agent_name=agent.name, external_id=external_id
            )
            await service.add_chat_message(
                agent.id, session.id, "user", f"hello from {external_id}", run.id
            )
            await _set_last_activity(
                session_maker, session.id, now - timedelta(minutes=minutes_ago)
            )
            return session

        user = await session_for(tenant_a, "user-1", 40)
        preview = await session_for(tenant_a, "pre_view:1", 30)
        lookalike = await session_for(tenant_b, "prexview:2", 20)
        anonymous = await session_for(tenant_b, None, 10)
        await session_for(elsewhere, "user-2", 0)
        tenant = [tenant_a.id, tenant_b.id]

        def ids(listing):
            return [summary.session_id for summary in listing["sessions"]]

        listed = await service.list_sessions(tenant)
        assert ids(listed) == [anonymous.id, lookalike.id, preview.id, user.id]
        assert listed["total_count"] == 4
        assert listed["sessions"][-1].first_message == "hello from user-1"
        assert listed["sessions"][-1].messages_count == 1

        # The prefix is literal: "_" is not a wildcard, and a session without
        # an external id is not excluded by one.
        kept = await service.list_sessions(
            tenant, exclude_external_id_prefix="pre_view:"
        )
        assert ids(kept) == [anonymous.id, lookalike.id, user.id]

        only = await service.list_sessions(tenant, external_id_prefix="USER-")
        assert ids(only) == [user.id]

        page = await service.list_sessions(tenant, limit=1, offset=1)
        assert ids(page) == [lookalike.id]
        assert page["total_count"] == 4

        assert await service.list_sessions([]) == {"sessions": [], "total_count": 0}

        # Continuing a conversation moves it to the top.
        await service.initialize_workflow_run(
            agent_name=tenant_a.name, external_id="user-1"
        )
        listed = await service.list_sessions(tenant)
        assert listed["sessions"][0].session_id == user.id
        assert listed["sessions"][0].runs_count == 2

    async def test_purge_sessions_in_batches(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name=f"Purge-{uuid4()}")
        other = await service.get_or_create_agent(name=f"Kept-{uuid4()}")
        now = datetime.now(timezone.utc)
        old = [
            await _session_with_history(
                service, session_maker, agent, now - timedelta(days=days)
            )
            for days in (90, 60, 40)
        ]
        recent = await _session_with_history(service, session_maker, agent, now)
        other_old = await _session_with_history(
            service, session_maker, other, now - timedelta(days=90)
        )

        batches = [
            batch
            async for batch in service.purge_sessions(
                now - timedelta(days=30), [agent.id], batch_size=2
            )
        ]

        # Oldest first, one transaction per batch.
        assert batches == [[old[0].id, old[1].id], [old[2].id]]
        purged = [session.id for session in old]
        remaining = await _count(
            session_maker, Session, Session.agent_id.in_([agent.id, other.id])
        )
        assert remaining == 2
        assert await _count(session_maker, Session, Session.id == recent.id) == 1
        assert await _count(session_maker, Session, Session.id == other_old.id) == 1
        for model in (Run, Task, ChatMessage):
            assert await _count(session_maker, model, model.session_id.in_(purged)) == 0

        # The cost record stays; the conversation in its payloads does not.
        calls = await service.get_model_call_stats(agent_id=agent.id, limit=10)
        assert sorted(call.total_tokens for call in calls) == [5, 5, 5, 5]
        blank = (
            ModelCallStat.request_data.is_(None),
            ModelCallStat.response_data.is_(None),
        )
        in_purged = ModelCallStat.session_id.in_(purged)
        assert await _count(session_maker, ModelCallStat, in_purged, *blank) == 3
        assert (
            await _count(
                session_maker,
                ModelCallStat,
                ModelCallStat.session_id == recent.id,
                *blank,
            )
            == 0
        )

    async def test_purge_sessions_scope(self, session_maker):
        service = AgentService(session_maker)
        agent = await service.get_or_create_agent(name=f"Scope-{uuid4()}")
        ancient = await _session_with_history(
            service,
            session_maker,
            agent,
            datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        far_future = datetime.now(timezone.utc) + timedelta(days=1)

        # An empty list of agents purges nothing, however late the cutoff.
        assert [batch async for batch in service.purge_sessions(far_future, [])] == []
        assert await _count(session_maker, Session, Session.id == ancient.id) == 1

        # ``None`` means every agent.
        cutoff = datetime(2001, 1, 1, tzinfo=timezone.utc)
        batches = [batch async for batch in service.purge_sessions(cutoff)]
        assert batches == [[ancient.id]]
        assert await _count(session_maker, Session, Session.id == ancient.id) == 0


def test_orm_stat_converts_and_passes_through():
    pyd = PydModelCallStat(
        call_type="llm",
        model="openai/x",
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
        cached_prompt_tokens=1,
        reasoning_tokens=1,
    )
    orm = _orm_stat(pyd)
    assert isinstance(orm, ModelCallStat)
    assert orm.model == "openai/x" and orm.total_tokens == 5
    # The token-detail columns must survive the conversion too.
    assert orm.cached_prompt_tokens == 1 and orm.reasoning_tokens == 1
    # An ORM stat is returned unchanged.
    assert _orm_stat(orm) is orm


def test_orm_stat_fills_a_missing_model_name():
    # The column is NOT NULL; a failed call may not know its model yet.
    assert _orm_stat(PydModelCallStat(call_type="llm")).model == ""


async def test_a_call_without_payloads_stores_sql_null(session_maker):
    """``IS NULL`` finds a call recorded without payloads, as it finds a purged one."""
    service = AgentService(session_maker)
    model = f"fake/{uuid4()}"
    await service.add_model_call_stats(
        PydModelCallStat(call_type="llm", model=model, total_tokens=1)
    )

    assert (
        await _count(
            session_maker,
            ModelCallStat,
            ModelCallStat.model == model,
            ModelCallStat.request_data.is_(None),
            ModelCallStat.response_data.is_(None),
        )
        == 1
    )
