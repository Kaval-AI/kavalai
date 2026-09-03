import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kavalai.db import (
    Agent,
    Session,
    Run,
    Task,
    ChatMessage,
    ModelCallStat,
    db_manager,
    ensure_async_scheme,
    is_sqlite_uri,
    sqlite_path_from_uri,
)
from kavalai.crud import insert, delete, get_one


@pytest.mark.asyncio
async def test_session_run_task_flow(agents_db: AsyncSession):
    """Test the hierarchy: Agent -> Session -> Run -> Task."""
    # 1. Setup Agent
    agent = await insert(agents_db, Agent, {"name": "Bot"})

    # 2. Create Session
    session = await insert(
        agents_db, Session, {"agent_id": agent.id, "external_id": "user_123"}
    )
    assert session.agent_id == agent.id

    # 3. Create Run (Execution)
    run = await insert(
        agents_db,
        Run,
        {
            "session_id": session.id,
            "input_data": {"query": "Execute task"},
            "context": {"metadata": "test-context"},
        },
    )
    assert run.session_id == session.id

    # 4. Create Task linked to the run
    task = await insert(
        agents_db,
        Task,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "inputs": {"sub_task": "process_data"},
        },
    )
    assert task.run_id == run.id

    # 5. Create Chat Message linked to run
    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "role": "assistant",
            "content": "Hello there!",
        },
    )
    assert message.run_id == run.id


@pytest.mark.asyncio
async def test_run_set_null_on_delete_for_messages(agents_db: AsyncSession):
    """Ensure deleting a run sets run_id to NULL in messages instead of deleting them."""
    agent = await insert(agents_db, Agent, {"name": "Persistence Test"})
    session = await insert(agents_db, Session, {"agent_id": agent.id})
    run = await insert(agents_db, Run, {"session_id": session.id})

    message = await insert(
        agents_db,
        ChatMessage,
        {
            "agent_id": agent.id,
            "session_id": session.id,
            "run_id": run.id,
            "role": "assistant",
            "content": "I am linked to a run",
        },
    )

    # FIX: Capture the ID before expiring the object
    message_id = message.id

    # Action: Delete ONLY the run
    await delete(agents_db, Run, run.id)

    # This detaches the local 'message' object from its current state
    agents_db.expire_all()

    # Asserts - Use the local message_id variable
    fetched_msg = await get_one(agents_db, ChatMessage, message_id)

    assert fetched_msg is not None
    assert fetched_msg.run_id is None  # Check ON DELETE SET NULL works


@pytest.mark.asyncio
async def test_agent_model_call_stats(agents_db: AsyncSession):
    """Test Agent and ModelCallStat models (no direct FK)."""
    # 1. Create Agent
    agent = await insert(
        agents_db,
        Agent,
        {
            "name": "Test Agent",
        },
    )
    assert agent.name == "Test Agent"

    # 2. Create Call Stat linked to agent (by ID only)
    stat = await insert(
        agents_db,
        ModelCallStat,
        {
            "call_type": "llm",
            "model": "gpt-4",
            "agent_id": agent.id,
            "response_code": 200,
            "cached_prompt_tokens": 12,
        },
    )
    assert stat.agent_id == agent.id

    # 3. Verify no automatic NULLing (since no FK)
    agent_id = agent.id
    stat_id = stat.id

    # Action: Delete the agent
    await delete(agents_db, Agent, agent_id)

    # Clear session to force re-fetch from DB
    agents_db.expire_all()

    assert await get_one(agents_db, Agent, agent_id) is None

    # Re-fetch stat to see if it's still there and STILL HAS agent_id
    # (because there is no FK and no ON DELETE SET NULL)
    fetched_stat = await get_one(agents_db, ModelCallStat, stat_id)
    assert fetched_stat is not None
    assert fetched_stat.agent_id == agent_id


def test_ensure_async_scheme():
    """Test the ensure_async_scheme utility function."""
    # Test converting standard postgresql scheme
    assert (
        ensure_async_scheme("postgresql://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # Test with already correct scheme
    assert (
        ensure_async_scheme("postgresql+asyncpg://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # Test with other postgresql variations
    assert (
        ensure_async_scheme("postgresql+psycopg2://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )

    # SQLite gets its async driver too; an unknown scheme is left alone
    assert ensure_async_scheme("sqlite:///:memory:") == "sqlite+aiosqlite:///:memory:"
    assert ensure_async_scheme("sqlite+aiosqlite:///x.db") == "sqlite+aiosqlite:///x.db"
    assert ensure_async_scheme("mysql://u:p@h/db") == "mysql://u:p@h/db"

    # Test with invalid URI
    assert ensure_async_scheme("not_a_uri") == "not_a_uri"
    assert ensure_async_scheme("") == ""
    assert ensure_async_scheme(None) is None


@pytest.mark.asyncio
async def test_sqlite_compat_sessionmaker_runs_agent_service():
    """The greenlet-free compat shim (sync SQLite engine behind an async
    session surface) supports the full AgentService flow — the stack the
    in-browser playground runs on."""
    from kavalai.agent_service import AgentService
    from kavalai.db import DatabaseManager

    service = AgentService(DatabaseManager().get_sqlite_compat_sessionmaker())

    agent, session, run = await service.initialize_workflow_run(
        agent_name="shim-bot", external_id="chat-1", input_data={"q": "hi"}
    )
    assert agent.id and session.id and run.id
    assert session.external_id == "chat-1"

    # The same external id reuses the session; a new one starts fresh.
    _, session2, run2 = await service.initialize_workflow_run(
        agent_name="shim-bot", external_id="chat-1"
    )
    assert session2.id == session.id and run2.id != run.id
    _, other, _ = await service.initialize_workflow_run(
        agent_name="shim-bot", external_id="chat-2"
    )
    assert other.id != session.id

    await service.add_chat_message(
        agent_id=agent.id, session_id=session.id, role="user", content="hello"
    )
    history = await service.get_chat_history(session.id)
    assert [(m.role, m.content) for m in history] == [("user", "hello")]

    updated = await service.update_run(run.id, output_data={"a": 1})
    assert updated.output_data == {"a": 1}


@pytest.mark.asyncio
async def test_delete_missing_row_reports_false(agents_db: AsyncSession):
    from uuid import uuid4

    assert await delete(agents_db, Agent, uuid4()) is False


def test_sqlite_uri_helpers():
    assert is_sqlite_uri("sqlite:///x.db")
    assert is_sqlite_uri("sqlite+aiosqlite:///x.db")
    assert not is_sqlite_uri("postgresql://u:p@h/db")
    assert not is_sqlite_uri("")
    assert sqlite_path_from_uri("sqlite:////tmp/agents.db") == "/tmp/agents.db"
    assert sqlite_path_from_uri("sqlite:///rel.db") == "rel.db"
    assert sqlite_path_from_uri("sqlite://") == ":memory:"


@pytest.mark.asyncio
async def test_get_sessionmaker_serves_sqlite_uris(tmp_path):
    """A ``sqlite://`` URI lands on the SQLite engine: schema ignored, the
    same shared engine as ``get_sqlite_sessionmaker`` and foreign keys on."""
    db_path = tmp_path / "agents.db"
    session_maker = db_manager.get_sessionmaker(
        uri=f"sqlite:///{db_path}", schema="ignored"
    )
    assert (
        session_maker.kw["bind"]
        is db_manager.get_sqlite_sessionmaker(db_path=str(db_path)).kw["bind"]
    )
    async with session_maker() as session:
        assert (await session.execute(text("PRAGMA foreign_keys"))).scalar() == 1
