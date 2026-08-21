import pytest
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.db import Agent, ModelCallStat
from kavalai.stats import get_summary_stats, get_daily_stats
from kavalai.crud import insert


@pytest.mark.asyncio
async def test_stats_with_tokens_and_models(agents_db: AsyncSession):
    # 1. Setup Agent
    agent = await insert(agents_db, Agent, {"name": "TokenBot"})

    # 2. Create Model Call Stats for LLM
    await insert(
        agents_db,
        ModelCallStat,
        {
            "agent_id": agent.id,
            "call_type": "llm",
            "model": "gpt-4",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_prompt_tokens": 40,
        },
    )

    await insert(
        agents_db,
        ModelCallStat,
        {
            "agent_id": agent.id,
            "call_type": "llm",
            "model": "claude-3",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "reasoning_tokens": 30,
        },
    )

    # 3. Create Model Call Stats for Embedding
    await insert(
        agents_db,
        ModelCallStat,
        {
            "agent_id": agent.id,
            "call_type": "embedding",
            "model": "text-embedding-3-small",
            "prompt_tokens": 500,
            "completion_tokens": 0,
            "total_tokens": 500,
        },
    )

    # Test summary stats
    summary = await get_summary_stats(agents_db, agent_id=agent.id)
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 150
    assert summary["total_embedding_tokens"] == 500

    # Test daily stats
    daily = await get_daily_stats(agents_db, days=7, agent_id=agent.id)

    # Check LLM grouping
    assert "gpt-4" in daily["llm"]
    assert "claude-3" in daily["llm"]

    # Check Embedding grouping
    assert "text-embedding-3-small" in daily["embedding"]

    # Verify values in series
    today_str = datetime.now(timezone.utc).date().isoformat()

    gpt4_today = next(d for d in daily["llm"]["gpt-4"] if d["date"] == today_str)
    assert gpt4_today["prompt_tokens"] == 100
    assert gpt4_today["completion_tokens"] == 50

    embedding_today = next(
        d
        for d in daily["embedding"]["text-embedding-3-small"]
        if d["date"] == today_str
    )
    assert embedding_today["prompt_tokens"] == 500


@pytest.mark.asyncio
async def test_daily_run_series_is_grouped_per_agent(agents_db: AsyncSession):
    from kavalai.db import Run, Session as DbSession

    agent = await insert(agents_db, Agent, {"name": "RunBot"})
    session = await insert(
        agents_db, DbSession, {"agent_id": agent.id, "external_id": "u1"}
    )
    for _ in range(2):
        await insert(agents_db, Run, {"session_id": session.id, "input_data": {}})

    daily = await get_daily_stats(agents_db, days=7, agent_id=agent.id)

    series = daily["runs"]["RunBot"]
    assert len(series) == 7
    # Every day is represented; the runs land on the most recent one.
    assert {entry["count"] for entry in series} == {0, 2}
    assert all("duration_seconds" in entry for entry in series)
