"""Regression tests for the business info example.

``examples/business_info_agent.py`` doubles as the script that fills a demo
agent database for the backoffice, so both its workflow shape and the rows the
engine writes for it are worth protecting. The workflow runs here against a
throwaway SQLite database with fake tools and a fake LLM, so no network,
Crawl4AI service or Postgres is involved.
"""

import json
from importlib import import_module

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kavalai import pythontool
from kavalai.agent_service import AgentService
from kavalai.db import Agent, Base, ChatMessage, ModelCallStat, Run, Session, Task
from kavalai.llm_clients.base_client import BaseLlmClient
from kavalai.llm_clients.base_client import ModelCallStat as LlmModelCallStat
from kavalai.tools.webtools.crawl4ai import WebSearchResponse, WebSearchResult
from kavalai.workflow.tasklog.postgres import PostgresTaskLogger

demo = import_module("examples.business_info_agent")

INFO = {
    "name": "Kaval.AI",
    "address": "Tallinn, Estonia",
    "website": "https://kaval.ai",
    "phone": None,
    "owners": "OÜ KAVAL AI",
    "description": "YAML-based AI agent framework.",
    "industry": "Software",
}
PROFILE = {**INFO, "summary": "Kaval.AI builds a YAML-based agent framework."}


# -- fake tools, registered by import path through demo.TOOLS -----------------


@pythontool
async def fake_web_search(query: str, count: int = 10) -> WebSearchResponse:
    """Return canned search results."""
    return WebSearchResponse(
        query=query,
        success=True,
        results=[WebSearchResult(title=query, url="https://example.com", snippet="x")],
    )


@pythontool
async def fake_crawl_url(url: str) -> str:
    """Return canned page content."""
    return f"# {url}\n\nA company page."


class _FakeClient(BaseLlmClient):
    """Answers agent steps and the summary node, and reports token usage.

    Which node is calling is read off the requested ``response_model``: the
    agent loop asks for a step (``tool_calls``), the summary node asks for the
    final profile (``summary``).
    """

    def __init__(self, stats_receiver=None, fail=False):
        super().__init__(model_stats_receiver=stats_receiver)
        self.fail = fail
        self.steps = 0

    def _payload(self, response_model) -> str:
        fields = response_model.model_fields if response_model else {}
        if "summary" in fields:
            return json.dumps(PROFILE)
        if "tool_calls" not in fields:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected response model: {response_model}")

        self.steps += 1
        if self.steps == 1:
            return json.dumps(
                {
                    "instructions": "Crawl the official site.",
                    "tool_calls": [
                        {
                            "name": f"python://{demo.CRAWL_TOOL}",
                            "literal_args": json.dumps({"url": "https://example.com"}),
                            "call_id": "c0",
                        }
                    ],
                    "output": None,
                }
            )
        return json.dumps(
            {"instructions": "Fill in the form.", "tool_calls": [], "output": INFO}
        )

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        if self.fail:
            raise RuntimeError("model unavailable")
        payload = self._payload(response_model)
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        await value_streamer.stream_partial(payload)
        await value_streamer.stream_complete()
        await self._send_model_call_stats(
            LlmModelCallStat(
                call_type="llm",
                model="fake/model",
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                duration_seconds=0.01,
            )
        )


@pytest.fixture
def session_maker(tmp_path):
    """An empty agent database on SQLite."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'agents.db'}")
    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return _create, maker


@pytest.fixture
def demo_env(monkeypatch, session_maker):
    """Point the example at the test database, fake tools and a fake model."""
    create, maker = session_maker
    here = __name__

    monkeypatch.setattr(
        demo,
        "TOOLS",
        {
            demo.SEARCH_TOOL: f"{here}.fake_web_search",
            demo.CRAWL_TOOL: f"{here}.fake_crawl_url",
        },
    )
    monkeypatch.setattr(demo, "open_agent_service", lambda: AgentService(maker))

    state = {"fail": False}
    real_build_engine = demo.build_engine

    def build_engine(**kwargs):
        engine = real_build_engine(**kwargs)
        engine.client_factory = lambda model, parameters, stats: _FakeClient(
            stats, fail=state["fail"]
        )
        return engine

    monkeypatch.setattr(demo, "build_engine", build_engine)
    return create, maker, state


async def _count(maker, model):
    async with maker() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar()


def _engine():
    return demo.build_engine(
        model="fake/model",
        max_steps=4,
        search_results=3,
        agent_service=None,
        task_logger=None,
    )


def test_session_key_is_a_stable_slug():
    assert demo.session_key("Kaval.AI (kaval.ai)") == "demo-kaval-ai-kaval-ai"
    assert demo.session_key("Alphabet (Google)") == "demo-alphabet-google"


def test_default_companies_cover_kavalai_and_faang():
    assert demo.DEFAULT_COMPANIES[0].startswith("Kaval.AI")
    assert len(demo.DEFAULT_COMPANIES) == 6


def test_workflow_is_a_search_research_summarize_pipeline():
    graph = _engine().graph
    assert [(n.name, n.type) for n in graph.nodes] == [
        ("start", "start"),
        ("search", "function"),
        ("research", "agent"),
        ("summarize", "llm"),
        ("end", "end"),
    ]

    nodes = {n.name: n for n in graph.nodes}
    assert nodes["search"].tool == f"python://{demo.SEARCH_TOOL}"
    assert nodes["search"].output == "search_results"
    # The research agent may crawl pages but not re-run the search.
    assert nodes["research"].allowed_tools == [f"python://{demo.CRAWL_TOOL}"]
    assert nodes["research"].output == "business_info"
    # Each company is researched from scratch.
    assert nodes["summarize"].use_history is False
    assert nodes["summarize"].output == "output"


async def test_run_is_recorded_for_every_company(demo_env):
    create, maker, _ = demo_env
    await create()

    await demo.main(["--company", "Kaval.AI (kaval.ai)", "--company", "Netflix"])

    assert await _count(maker, Agent) == 1
    assert await _count(maker, Session) == 2
    assert await _count(maker, Run) == 2
    # One task per executed node: search, research, summarize.
    assert await _count(maker, Task) == 6
    # A user and an assistant message per run.
    assert await _count(maker, ChatMessage) == 4
    # Two agent steps plus the summary call per run.
    assert await _count(maker, ModelCallStat) == 6

    async with maker() as session:
        agent = (await session.execute(select(Agent))).scalars().one()
        sessions = (await session.execute(select(Session))).scalars().all()

    assert agent.name == demo.AGENT_NAME
    # The stored graph is what lets the backoffice draw the workflow.
    assert agent.workflow["nodes"][1]["name"] == "search"
    assert {s.external_id for s in sessions} == {
        "demo-kaval-ai-kaval-ai",
        "demo-netflix",
    }


async def test_run_records_the_profile_tasks_and_tokens(demo_env):
    create, maker, _ = demo_env
    await create()

    await demo.main(["--company", "Kaval.AI (kaval.ai)"])

    async with maker() as session:
        run = (await session.execute(select(Run))).scalars().one()
        tasks = (
            (await session.execute(select(Task).order_by(Task.created_at)))
            .scalars()
            .all()
        )
        chat_stmt = select(ChatMessage).order_by(ChatMessage.created_at)
        messages = (await session.execute(chat_stmt)).scalars().all()

    assert run.input_data["business_query"] == "Kaval.AI (kaval.ai)"
    assert run.output_data["name"] == "Kaval.AI"
    assert run.output_data["summary"].startswith("Kaval.AI builds")

    # Task rows are written fire-and-forget, so compare by name rather than
    # by insertion order.
    by_name = {t.name: t for t in tasks}
    assert {n: t.node_type for n, t in by_name.items()} == {
        "search": "function",
        "research": "agent",
        "summarize": "llm",
    }
    # The search node's results are what the research agent works from.
    assert by_name["search"].output["results"][0]["url"] == "https://example.com"
    assert all(t.duration_seconds is not None for t in tasks)

    # The conversation reads as a request and its answer.
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Research the business 'Kaval.AI (kaval.ai)'."
    assert json.loads(messages[1].content)["summary"].startswith("Kaval.AI builds")

    # The run context keeps every intermediate value, so the backoffice run
    # view can show what each node produced.
    assert set(run.context) >= {"input", "search_results", "business_info", "output"}
    assert run.context["business_info"]["name"] == "Kaval.AI"


async def test_reruns_reuse_the_company_session(demo_env):
    create, maker, _ = demo_env
    await create()

    await demo.main(["--company", "Netflix"])
    await demo.main(["--company", "Netflix"])

    assert await _count(maker, Session) == 1
    assert await _count(maker, Run) == 2


async def test_failed_run_is_recorded_and_does_not_stop_the_batch(demo_env):
    create, maker, state = demo_env
    await create()
    state["fail"] = True

    await demo.main(["--company", "Netflix", "--company", "Apple Inc."])

    # Both companies were attempted and both failures recorded.
    assert await _count(maker, Run) == 2
    async with maker() as session:
        runs = (await session.execute(select(Run))).scalars().all()
    assert all(run.output_data is None for run in runs)


async def test_no_db_flag_skips_persistence(demo_env):
    create, maker, _ = demo_env
    await create()

    await demo.main(["--company", "Netflix", "--no-db"])

    assert await _count(maker, Run) == 0
    assert await _count(maker, Agent) == 0
