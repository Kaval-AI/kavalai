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

from datetime import datetime, timedelta, timezone
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from kavalai.db import Agent, ChatMessage, ModelCallStat, Run, Session, Task


async def get_summary_stats(session: AsyncSession, agent_id: str | None = None):
    """Totals over the last 30 days, optionally for one agent."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    # Get total sessions
    stmt_sessions = (
        select(func.count(Session.id))
        .select_from(Session)
        .where(Session.created_at >= start_date)
    )
    if agent_id:
        stmt_sessions = stmt_sessions.where(Session.agent_id == agent_id)

    # Get total tokens
    stmt_tokens = (
        select(
            func.sum(ModelCallStat.prompt_tokens),
            func.sum(ModelCallStat.completion_tokens),
        )
        .select_from(ModelCallStat)
        .where(ModelCallStat.call_type == "llm", ModelCallStat.created_at >= start_date)
    )
    if agent_id:
        stmt_tokens = stmt_tokens.where(ModelCallStat.agent_id == agent_id)

    # Get total embedding tokens
    stmt_embedding_tokens = (
        select(func.sum(ModelCallStat.total_tokens))
        .select_from(ModelCallStat)
        .where(
            ModelCallStat.call_type == "embedding",
            ModelCallStat.created_at >= start_date,
        )
    )
    if agent_id:
        stmt_embedding_tokens = stmt_embedding_tokens.where(
            ModelCallStat.agent_id == agent_id
        )

    # Get total tasks
    stmt_tasks = (
        select(func.count(Task.id))
        .select_from(Task)
        .where(Task.created_at >= start_date)
    )
    if agent_id:
        stmt_tasks = (
            stmt_tasks.join(Run, Task.run_id == Run.id)
            .join(Session, Run.session_id == Session.id)
            .where(Session.agent_id == agent_id)
        )

    # Get total workflow runs
    stmt_runs = (
        select(func.count(Run.id))
        .select_from(Run)
        .join(Session, Run.session_id == Session.id)
        .where(Session.created_at >= start_date)
    )
    if agent_id:
        stmt_runs = stmt_runs.where(Session.agent_id == agent_id)

    # Get total messages
    stmt_messages = (
        select(func.count(ChatMessage.id))
        .select_from(ChatMessage)
        .where(ChatMessage.created_at >= start_date)
    )
    if agent_id:
        stmt_messages = stmt_messages.where(ChatMessage.agent_id == agent_id)

    res_sessions = await session.execute(stmt_sessions)
    res_tokens = await session.execute(stmt_tokens)
    res_embedding_tokens = await session.execute(stmt_embedding_tokens)
    res_tasks = await session.execute(stmt_tasks)
    res_messages = await session.execute(stmt_messages)
    res_runs = await session.execute(stmt_runs)

    tokens_row = res_tokens.one()

    return {
        "total_sessions": int(res_sessions.scalar() or 0),
        "total_prompt_tokens": int(tokens_row[0] or 0),
        "total_completion_tokens": int(tokens_row[1] or 0),
        "total_embedding_tokens": int(res_embedding_tokens.scalar() or 0),
        "total_tasks": int(res_tasks.scalar() or 0),
        "total_messages": int(res_messages.scalar() or 0),
        "total_runs": int(res_runs.scalar() or 0),
    }


async def get_daily_stats(
    session: AsyncSession, days: int = 7, agent_id: str | None = None
):
    """Per-day series over the last ``days`` days, optionally for one agent."""
    # Midnight of N days ago.
    end_date = datetime.now(timezone.utc)
    start_date = (end_date - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # We want to return a list of dates from start_date to today
    dates = [(start_date + timedelta(days=i)).date() for i in range(days)]

    async def get_counts_for_model(model):
        if model is Run:
            stmt = (
                select(
                    func.date(model.created_at).label("date"),
                    func.count(model.id).label("count"),
                    Agent.name.label("agent_name"),
                    func.sum(
                        func.extract("epoch", model.updated_at)
                        - func.extract("epoch", model.created_at)
                    ).label("duration_seconds"),
                )
                .select_from(model)
                .join(Session, model.session_id == Session.id)
                .join(Agent, Session.agent_id == Agent.id)
            )
        else:
            stmt = select(
                func.date(model.created_at).label("date"),
                func.count(model.id).label("count"),
            ).select_from(model)

        stmt = stmt.where(model.created_at >= start_date)

        if agent_id:
            if model is Run:
                # Already joined to Session above; Run has no agent_id.
                stmt = stmt.where(Session.agent_id == agent_id)
            elif model is Task:
                stmt = (
                    stmt.join(Run, model.run_id == Run.id)
                    .join(Session, Run.session_id == Session.id)
                    .where(Session.agent_id == agent_id)
                )
            else:
                stmt = stmt.where(model.agent_id == agent_id)

        if model is Run:
            stmt = stmt.group_by(func.date(model.created_at), Agent.name).order_by(
                func.date(model.created_at)
            )
        else:
            stmt = stmt.group_by(func.date(model.created_at)).order_by(
                func.date(model.created_at)
            )
        result = await session.execute(stmt)
        if model is Run:
            runs_by_agent = {}
            for row in result.all():
                runs_by_agent.setdefault(row.agent_name, {})[row.date] = {
                    "count": row.count,
                    "duration_seconds": float(row.duration_seconds or 0),
                }
            return runs_by_agent
        return {row.date: row.count for row in result.all()}

    async def get_model_stats(call_type: str):
        model_name_col = ModelCallStat.model.label("model_name")
        date_col = func.date(ModelCallStat.created_at).label("date")
        stmt = select(
            model_name_col,
            date_col,
            func.count(ModelCallStat.id).label("count"),
            func.sum(ModelCallStat.duration_seconds).label("duration_seconds"),
            func.sum(ModelCallStat.prompt_tokens).label("prompt_tokens"),
            func.sum(ModelCallStat.completion_tokens).label("completion_tokens"),
            func.sum(ModelCallStat.total_tokens).label("total_tokens"),
            func.sum(ModelCallStat.cached_prompt_tokens).label("cached_prompt_tokens"),
            func.sum(ModelCallStat.reasoning_tokens).label("reasoning_tokens"),
            func.sum(ModelCallStat.batch_size).label("batch_size"),
        ).where(
            ModelCallStat.call_type == call_type, ModelCallStat.created_at >= start_date
        )
        if agent_id:
            stmt = stmt.where(ModelCallStat.agent_id == agent_id)

        stmt = stmt.group_by(model_name_col, date_col)
        result = await session.execute(stmt)

        stats_by_model = {}
        for row in result.all():
            stats_by_model.setdefault(row.model_name, {})[row.date] = {
                "count": row.count,
                "duration_seconds": float(row.duration_seconds or 0),
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
                "total_tokens": int(row.total_tokens or 0),
                "cached_prompt_tokens": int(row.cached_prompt_tokens or 0),
                "reasoning_tokens": int(row.reasoning_tokens or 0),
                "batch_size": int(row.batch_size or 0),
            }
        return stats_by_model

    runs_counts = await get_counts_for_model(Run)
    sessions_counts = await get_counts_for_model(Session)
    messages_counts = await get_counts_for_model(ChatMessage)
    tasks_counts = await get_counts_for_model(Task)
    llm_stats = await get_model_stats("llm")
    embedding_stats = await get_model_stats("embedding")

    def format_series(counts):
        return [{"date": d.isoformat(), "count": counts.get(d, 0)} for d in dates]

    def format_run_series(agent_runs):
        series = []
        for d in dates:
            day_stat = agent_runs.get(
                d,
                {
                    "count": 0,
                    "duration_seconds": 0.0,
                },
            )
            series.append({"date": d.isoformat(), **day_stat})
        return series

    def format_model_series(model_stats):
        series = []
        for d in dates:
            day_stat = model_stats.get(
                d,
                {
                    "count": 0,
                    "duration_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                    "batch_size": 0,
                },
            )
            series.append({"date": d.isoformat(), **day_stat})
        return series

    return {
        "runs": {name: format_run_series(stats) for name, stats in runs_counts.items()},
        "sessions": format_series(sessions_counts),
        "messages": format_series(messages_counts),
        "tasks": format_series(tasks_counts),
        "llm": {name: format_model_series(stats) for name, stats in llm_stats.items()},
        "embedding": {
            name: format_model_series(stats) for name, stats in embedding_stats.items()
        },
    }
