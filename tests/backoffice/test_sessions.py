import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from kavalai.db import Agent, Session, Run, Task, ChatMessage, db_manager
from kavalai.backoffice.sessions import get_sessions_summary, get_session_details


@pytest_asyncio.fixture(params=["postgres", "sqlite"])
async def sessions_db(request, agents_db, tmp_path):
    """The agent database on both backends a project can point at.

    The session queries use JSON functions and a per-session ranking that
    differ between the dialects, so every test here runs against both.
    """
    if request.param == "postgres":
        yield agents_db
        return
    db_path = str(tmp_path / "agents.db")
    await db_manager.init_sqlite(db_path=db_path)
    async with db_manager.get_sqlite_sessionmaker(db_path=db_path)() as session:
        yield session


@pytest.mark.asyncio
async def test_get_session_details(sessions_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc)
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)

    r1 = Run(
        id=uuid4(),
        session_id=s1.id,
        input_data={"q": "test"},
        output_data={"ans": "res"},
        created_at=now - timedelta(minutes=5),
    )
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="user",
        content="Hello",
        created_at=now - timedelta(minutes=4),
    )
    t1 = Task(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        inputs={"cmd": "ls"},
        output={"out": "file1"},
        name="test_task",
        created_at=now - timedelta(minutes=3),
    )

    sessions_db.add_all([s1, r1, m1, t1])
    await sessions_db.commit()

    details = await get_session_details(sessions_db, s1.id)

    assert details.session_id == s1.id
    assert len(details.messages) == 1
    assert details.messages[0].content == "Hello"
    assert len(details.runs) == 1
    assert details.runs[0].id == r1.id
    assert details.runs[0].tasks_count == 1
    assert len(details.tasks) == 1
    assert details.tasks[0].id == t1.id
    assert details.tasks[0].name == "test_task"
    assert details.tasks[0].run_id == r1.id


@pytest.mark.asyncio
async def test_get_sessions_summary(sessions_db):
    # 1. Setup data
    agent = Agent(id=uuid4(), name="Test Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc)

    # Session 1: 1 run, 1 task, 2 messages
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    r1 = Run(id=uuid4(), session_id=s1.id, created_at=now)
    t1 = Task(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        created_at=now,
        errors=["Test error"],
    )
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="user",
        content="First message",
        created_at=now - timedelta(minutes=5),
    )
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        run_id=r1.id,
        role="assistant",
        content="Last message",
        created_at=now,
    )

    # Session 2: No runs, no tasks, no messages
    s2 = Session(
        id=uuid4(),
        agent_id=agent.id,
        created_at=now - timedelta(hours=1),
        updated_at=now - timedelta(hours=1),
    )

    # Session 3: Task with non-array errors (to reproduce the issue)
    s3 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    r3 = Run(id=uuid4(), session_id=s3.id, created_at=now)
    # Use a raw insert or a direct assignment if possible to bypass type hints if needed,
    # but SQLAlchemy JSONB might allow it if not strictly checked during assignment.
    t3 = Task(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s3.id,
        run_id=r3.id,
        created_at=now,
        errors="not an array",  # type: ignore
    )

    # Session 4: Task with empty array errors
    s4 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    r4 = Run(id=uuid4(), session_id=s4.id, created_at=now)
    t4 = Task(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s4.id,
        run_id=r4.id,
        created_at=now,
        errors=[],
    )

    sessions_db.add_all([s1, r1, t1, m1, m2, s2, s3, r3, t3, s4, r4, t4])
    await sessions_db.commit()

    # 2. Call the function
    result = await get_sessions_summary(sessions_db)
    summaries = result["sessions"]
    total_count = result["total_count"]

    # 3. Assertions
    assert len(summaries) == 4
    assert total_count == 4

    # Ordered by updated_at desc, so s1, s3, s4 should be before s2
    ids = [s.session_id for s in summaries]
    assert s1.id in ids
    assert s3.id in ids
    assert s4.id in ids
    assert s2.id == ids[3]

    # Find s1 summary
    summary1 = next(s for s in summaries if s.session_id == s1.id)
    assert summary1.errors_count == 1

    # Find s3 summary (non-array string errors -> count as 1 error)
    summary3 = next(s for s in summaries if s.session_id == s3.id)
    assert summary3.errors_count == 1

    # Find s4 summary (empty array -> count as 0 errors)
    summary4 = next(s for s in summaries if s.session_id == s4.id)
    assert summary4.errors_count == 0

    summary2 = summaries[3]
    assert summary2.session_id == s2.id
    assert summary2.runs_count == 0
    assert summary2.tasks_count == 0
    assert summary2.messages_count == 0
    assert summary2.first_message is None
    assert summary2.last_message is None

    # Test filtering by agent_id
    another_agent = Agent(id=uuid4(), name="Another Agent")
    sessions_db.add(another_agent)
    await sessions_db.commit()

    s3 = Session(id=uuid4(), agent_id=another_agent.id, created_at=now, updated_at=now)
    sessions_db.add(s3)
    await sessions_db.commit()

    result_filtered = await get_sessions_summary(sessions_db, agent_id=another_agent.id)
    summaries_filtered = result_filtered["sessions"]
    assert len(summaries_filtered) == 1
    assert result_filtered["total_count"] == 1
    assert summaries_filtered[0].session_id == s3.id


@pytest.mark.asyncio
async def test_get_sessions_summary_with_search(sessions_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc)
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    m1 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s1.id,
        role="user",
        content="Looking for a specific needle in the haystack",
        created_at=now,
    )

    s2 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    m2 = ChatMessage(
        id=uuid4(),
        agent_id=agent.id,
        session_id=s2.id,
        role="user",
        content="Just a normal message",
        created_at=now,
    )

    sessions_db.add_all([s1, m1, s2, m2])
    await sessions_db.commit()

    # Search for "needle"
    result = await get_sessions_summary(sessions_db, search="needle")
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s1.id

    # Search for "MESSAGE" (case insensitive)
    result = await get_sessions_summary(sessions_db, search="MESSAGE")
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s2.id

    # Search for something non-existent
    result = await get_sessions_summary(sessions_db, search="nonexistent")
    assert result["total_count"] == 0
    assert len(result["sessions"]) == 0


@pytest.mark.asyncio
async def test_get_sessions_summary_with_date_range(sessions_db):
    agent = Agent(id=uuid4(), name="Test Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Session 1: Today
    s1 = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)

    # Session 2: 10 days ago
    s2 = Session(
        id=uuid4(),
        agent_id=agent.id,
        created_at=now - timedelta(days=10),
        updated_at=now - timedelta(days=10),
    )

    sessions_db.add_all([s1, s2])
    await sessions_db.commit()

    # Filter for last 7 days
    start_date = now - timedelta(days=7)
    result = await get_sessions_summary(sessions_db, start_date=start_date)
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s1.id

    # Filter for range that includes only s2
    end_date = now - timedelta(days=5)
    result = await get_sessions_summary(sessions_db, end_date=end_date)
    assert result["total_count"] == 1
    assert result["sessions"][0].session_id == s2.id

    # Filter for both
    result = await get_sessions_summary(
        sessions_db,
        start_date=now - timedelta(days=15),
        end_date=now + timedelta(days=1),
    )
    assert result["total_count"] == 2


@pytest.mark.asyncio
async def test_get_sessions_summary_filters_by_external_id(sessions_db):
    """The one backoffice change evaluation needs: find *this* conversation.

    Evaluation runs tag their sessions ``eval:{suite}:{tag}:{case}:{repeat}``,
    so a prefix separates test traffic from production in one predicate, and a
    full id from a failing case in a result file lands on exactly the
    conversation that produced it.
    """
    agent = Agent(id=uuid4(), name="Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc)
    external_ids = [
        "eval:bakery-acceptance:pr-412:vague_quantity:0",
        "eval:bakery-acceptance:pr-412:clean_order:0",
        "eval:bakery-acceptance:main:clean_order:0",
        "chat-session-9931",
        None,
    ]
    sessions_db.add_all(
        [
            Session(
                id=uuid4(),
                agent_id=agent.id,
                external_id=external_id,
                created_at=now,
                updated_at=now,
            )
            for external_id in external_ids
        ]
    )
    await sessions_db.commit()

    everything = await get_sessions_summary(sessions_db)
    assert everything["total_count"] == 5

    # A prefix separates one experiment's traffic from everything else.
    experiment = await get_sessions_summary(
        sessions_db, external_id="eval:bakery-acceptance:pr-412:"
    )
    assert experiment["total_count"] == 2

    # And all evaluation traffic from all real traffic.
    all_evals = await get_sessions_summary(sessions_db, external_id="eval:")
    assert all_evals["total_count"] == 3

    # A full id lands on the single failing conversation.
    exact = await get_sessions_summary(
        sessions_db, external_id="eval:bakery-acceptance:pr-412:vague_quantity:0"
    )
    assert exact["total_count"] == 1
    assert exact["sessions"][0].external_id.endswith("vague_quantity:0")

    assert (await get_sessions_summary(sessions_db, external_id="nope"))[
        "total_count"
    ] == 0


@pytest.mark.asyncio
async def test_session_details_returns_the_trajectory_in_execution_order(sessions_db):
    """``seq`` is what carries the structure; ``created_at`` is approximate."""
    agent = Agent(id=uuid4(), name="Agent")
    sessions_db.add(agent)
    await sessions_db.commit()

    now = datetime.now(timezone.utc)
    session = Session(id=uuid4(), agent_id=agent.id, created_at=now, updated_at=now)
    run = Run(id=uuid4(), session_id=session.id, created_at=now)

    # Written out of order on purpose: task rows are fire-and-forget, so the
    # order they land in is not the order they happened in.
    rows = [
        ("crawl", "tool_call", 2, "research", "python://crawl"),
        ("begin", "start", 0, None, None),
        ("research", "agent", 1, None, None),
        ("finish", "end", 3, None, None),
    ]
    sessions_db.add_all(
        [session, run]
        + [
            Task(
                id=uuid4(),
                agent_id=agent.id,
                session_id=session.id,
                run_id=run.id,
                name=name,
                node_type=node_type,
                seq=seq,
                parent_task_name=parent,
                tool_uri=tool_uri,
                created_at=now,
            )
            for name, node_type, seq, parent, tool_uri in rows
        ]
    )
    await sessions_db.commit()

    details = await get_session_details(sessions_db, session.id)
    assert [t.name for t in details.tasks] == ["begin", "research", "crawl", "finish"]

    crawl = details.tasks[2]
    assert crawl.node_type == "tool_call"
    # This is what the run-tasks view indents under its node.
    assert crawl.parent_task_name == "research"
    assert crawl.tool_uri == "python://crawl"
