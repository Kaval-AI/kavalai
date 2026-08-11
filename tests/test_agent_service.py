import pytest
from uuid import uuid4
from kavalai.agent_service import AgentService
from kavalai.db import ModelCallStat


@pytest.mark.asyncio
class TestAgentService:
    async def test_get_or_create_agent(self, agents_session_maker):
        service = AgentService(agents_session_maker)

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

    async def test_get_or_create_session_logic(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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

    async def test_run_and_task_tracking(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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

    async def test_chat_history_retrieval(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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
        # agents_db truncates the tables so the count assertions below see
        # only the stats created here, regardless of test collection order.
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
                    cost=0.001,
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

    async def test_get_history_value(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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

    async def test_initialize_workflow_run_new_agent(self, agents_session_maker):
        """Test batch initialization creates new agent, session, and run."""
        service = AgentService(agents_session_maker)

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

    async def test_initialize_workflow_run_existing_agent(self, agents_session_maker):
        """Test batch initialization reuses existing agent."""
        service = AgentService(agents_session_maker)

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

    async def test_initialize_workflow_run_with_existing_session(
        self, agents_session_maker
    ):
        """Test batch initialization with existing session_id."""
        service = AgentService(agents_session_maker)

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

    async def test_initialize_workflow_run_with_external_id(self, agents_session_maker):
        """Test batch initialization with external_id for session."""
        service = AgentService(agents_session_maker)

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

    async def test_initialize_workflow_run_invalid_session_id(
        self, agents_session_maker
    ):
        """Test batch initialization with non-existent session_id raises error."""
        service = AgentService(agents_session_maker)

        with pytest.raises(ValueError, match="Session with ID .* not found"):
            await service.initialize_workflow_run(
                agent_name="InvalidSessionAgent",
                session_id=uuid4(),  # Non-existent session
                input_data={"query": "test"},
            )

    async def test_chat_history_windowing(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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
        assert len(window) == 2

    async def test_add_model_call_stats_assigns_agent(self, agents_session_maker):
        service = AgentService(agents_session_maker)
        agent = await service.get_or_create_agent(name="StatAgentTest")

        stat = await service.add_model_call_stats(
            ModelCallStat(call_type="llm", model="test/model"), agent_id=agent.id
        )
        assert stat.agent_id == agent.id

    async def test_delete_history_for_session(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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

    async def test_delete_history_for_agent(self, agents_session_maker):
        service = AgentService(agents_session_maker)
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

    async def test_get_or_create_agent_updates_changed_fields(
        self, agents_session_maker
    ):
        service = AgentService(agents_session_maker)
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

    async def test_update_run_rejects_an_unknown_run(self, agents_session_maker):
        service = AgentService(agents_session_maker)
        with pytest.raises(ValueError, match="Run not found"):
            await service.update_run(run_id=uuid4(), output_data={"a": 1})

    async def test_update_run_stores_output_data(self, agents_session_maker):
        service = AgentService(agents_session_maker)
        agent = await service.get_or_create_agent(name="OutputBot")
        session = await service.get_or_create_session(agent_id=agent.id)
        run = await service.create_run(session_id=session.id)

        stored = await service.update_run(
            run_id=run.id, output_data={"agent_response": "hi"}
        )

        assert stored.output_data == {"agent_response": "hi"}

    async def test_get_history_value_skips_runs_without_context(
        self, agents_session_maker
    ):
        service = AgentService(agents_session_maker)
        agent = await service.get_or_create_agent(name="SparseHistory")
        session = await service.get_or_create_session(agent_id=agent.id)

        run = await service.create_run(session_id=session.id)
        await service.update_run(run_id=run.id, context={"answer": "kept"})
        # A newer run that never recorded a context must not mask the older one.
        await service.create_run(session_id=session.id)

        assert await service.get_history_value(session.id, "answer") == "kept"
