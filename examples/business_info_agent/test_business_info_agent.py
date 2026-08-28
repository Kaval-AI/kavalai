"""Regression tests for the business info example, next to what they test.

``business_info.py`` is the workflow, ``research_companies.py`` runs a list of
companies through it, and ``eval_cases.yaml`` grades the server in
``business_info_in_memory.py``. The workflow shape, the rows the engine writes
and the case file are all worth protecting — an example that quietly stopped
working is worse than no example.

These are not the evaluation: they run the workflow against a throwaway SQLite
database with fake tools and a fake LLM, so they need no network, no Crawl4AI
browser and no API key. ``eval_cases.yaml`` is what grades the agent's answers,
and it needs a running server.
"""

import json
from importlib import import_module
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kavalai import pythontool
from kavalai.agent_service import AgentService
from kavalai.db import Agent, Base, ChatMessage, ModelCallStat, Run, Session, Task
from kavalai.eval.eval_runner import load_suite
from kavalai.llm_clients.base_client import BaseLlmClient
from kavalai.llm_clients.base_client import ModelCallStat as LlmModelCallStat
from kavalai.tools.webtools.crawl4ai import WebSearchResponse, WebSearchResult

workflow = import_module("examples.business_info_agent.business_info")
demo = import_module("examples.business_info_agent.research_companies")
server = import_module("examples.business_info_agent.business_info_in_memory")

CASES = Path(__file__).resolve().parent / "eval_cases.yaml"

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
                            "name": f"python://{workflow.CRAWL_TOOL}",
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
        workflow,
        "TOOLS",
        {
            workflow.SEARCH_TOOL: f"{here}.fake_web_search",
            workflow.CRAWL_TOOL: f"{here}.fake_crawl_url",
        },
    )

    async def open_agent_service():
        return AgentService(maker)

    monkeypatch.setattr(demo, "open_agent_service", open_agent_service)

    state = {"fail": False}
    real_build_engine = workflow.build_engine

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
    return workflow.build_engine(
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
    assert nodes["search"].tool == f"python://{workflow.SEARCH_TOOL}"
    assert nodes["search"].output == "search_results"
    # The research agent may crawl pages but not re-run the search.
    assert nodes["research"].allowed_tools == [f"python://{workflow.CRAWL_TOOL}"]
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
    # One row per node visit — start, search, research, summarize, end — plus
    # one per tool call the research agent made. Six per run.
    assert await _count(maker, Task) == 12
    # A user and an assistant message per run.
    assert await _count(maker, ChatMessage) == 4
    # Two agent steps plus the summary call per run.
    assert await _count(maker, ModelCallStat) == 6

    async with maker() as session:
        agent = (await session.execute(select(Agent))).scalars().one()
        sessions = (await session.execute(select(Session))).scalars().all()

    assert agent.name == workflow.AGENT_NAME
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

    # Every node visit wrote a row and ``seq`` is the executed path, including
    # the tool the research agent chose for itself.
    by_seq = sorted(tasks, key=lambda t: t.seq)
    assert [(t.name, t.node_type) for t in by_seq] == [
        ("start", "start"),
        ("search", "function"),
        ("research", "agent"),
        ("webtools.crawl_url", "tool_call"),
        ("summarize", "llm"),
        ("end", "end"),
    ]
    assert [t.seq for t in by_seq] == [0, 1, 2, 3, 4, 5]

    by_name = {t.name: t for t in tasks}
    # The search node's results are what the research agent works from.
    assert by_name["search"].output["results"][0]["url"] == "https://example.com"
    assert all(t.duration_seconds is not None for t in tasks)

    # A function node and an agent's own tool call are both findable by URI,
    # which is what lets one assertion cover "was this tool ever called".
    assert by_name["search"].tool_uri == f"python://{workflow.SEARCH_TOOL}"
    crawl = by_name["webtools.crawl_url"]
    assert crawl.tool_uri == f"python://{workflow.CRAWL_TOOL}"
    # The agent step is a field on the row, not a level of nesting.
    assert crawl.parent_task_name == "research"
    assert crawl.inputs["step"] == 0

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


def test_server_serves_the_same_workflow():
    """The evaluated server is the example workflow, not a copy of it.

    ``kavalai-eval`` discovers the agent's input and output types from the
    served OpenAPI spec, so the spec is the part of the server the case file
    depends on.
    """
    app = server.create_app(_engine())

    assert server.PORT == 25200
    assert app.state.engine.graph.name == workflow.AGENT_NAME

    spec = app.openapi()
    assert {"/run_agent", "/stream_agent"} <= set(spec["paths"])
    schemas = spec["components"]["schemas"]
    assert "business_query" in schemas["BusinessQuery"]["properties"]
    assert "summary" in schemas["BusinessProfile"]["properties"]


def test_eval_cases_fit_the_agents_input_type():
    """Every case can actually be sent to this agent.

    The evaluator validates a case's input against the agent's own input type
    before it sends it, so a mistyped field is a case that never runs. Checking
    it here means the suite is known to be runnable without an agent, an API
    key or a network.
    """
    suite = load_suite(CASES)
    assert suite.name == "business-info-agent"
    assert len(suite.cases) >= 5

    for case in suite.cases:
        workflow.BusinessQuery(**case.input)

    # Judged cases carry a criterion; literal ones name output fields that
    # exist. A matcher on a field the agent never returns always fails.
    fields = set(workflow.BusinessProfile.model_fields)
    for case in suite.cases:
        if case.type == "judge":
            assert isinstance(case.expected, str) and case.expected.strip()
        else:
            assert set(case.expected) <= fields, case.name


def test_eval_cases_grade_kavalai_and_a_refusal():
    """The suite covers the two behaviours worth protecting.

    Facts about Kaval.AI are what the agent must find; leaving a field null for
    a business that does not exist is what it must not paper over. A suite that
    only asked for facts would pass an agent that invents them.
    """
    names = [case.name for case in load_suite(CASES).cases]

    assert any(name.startswith("kavalai_") for name in names)
    assert "unknown_business_is_not_invented" in names
