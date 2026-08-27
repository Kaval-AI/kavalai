"""Example: a workflow that researches businesses with Crawl4AI.

The workflow is a four-step pipeline, one node per step::

    start → search (function) → research (agent) → summarize (llm) → end

1. ``search`` runs a keyless Crawl4AI web search for the company name.
2. ``research`` is an agent node: it reads the search results, crawls the
   pages worth reading — deciding for itself how many steps that takes — and
   fills in the structured company facts. It is restricted to the crawl tool.
3. ``summarize`` writes a short company summary and emits the final record.

Everything else comes from the library: hand the engine an
:class:`~kavalai.agent_service.AgentService` and a
:class:`~kavalai.workflow.tasklog.postgres.PostgresTaskLogger` and it records
the runs itself — the agent (with its graph, so the backoffice can draw the
workflow), a session and run per company, a task per node, the chat history and
a model-call stat per LLM call. That is what fills the backoffice with demo
data; the example only builds the workflow and invokes it once per company.

Usage::

    # Research the default company list, persisting to $KAVALAI_DB_URI.
    uv run --env-file .env python examples/business_info_agent.py

    # Pick the companies, and keep the database out of it.
    uv run --env-file .env python examples/business_info_agent.py \
        --company "Spotify" --no-db

Needs the Crawl4AI service (``docker compose up -d crawl4ai``). Persistence
needs a migrated agent database (``python -m kavalai.migrate_db agents``) reachable
via ``KAVALAI_DB_URI`` / ``KAVALAI_DB_SCHEMA``; without ``KAVALAI_DB_URI`` the
example just runs the workflow and prints the results.
"""

import argparse
import asyncio
import os
import re
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from kavalai.agent_service import AgentService
from kavalai.db import db_manager
from kavalai.tools.webtools.crawl4ai import WebSearchResponse
from kavalai.workflow import WorkflowBuilder, WorkflowEngine
from kavalai.workflow.tasklog.postgres import PostgresTaskLogger

AGENT_NAME = "business-info-agent"
AGENT_DESCRIPTION = (
    "Searches the web for a business, crawls the pages worth reading and "
    "returns a structured company profile with a short summary."
)

# Kaval.AI plus the FAANG five — enough runs to make the backoffice pages look
# like a real deployment.
DEFAULT_COMPANIES = [
    "Kaval.AI (kaval.ai)",
    "Meta Platforms",
    "Apple Inc.",
    "Amazon.com",
    "Netflix",
    "Alphabet (Google)",
]

DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"

SEARCH_TOOL = "webtools.web_search"
CRAWL_TOOL = "webtools.crawl_url"
TOOLS = {
    SEARCH_TOOL: "kavalai.tools.webtools.crawl4ai.web_search",
    CRAWL_TOOL: "kavalai.tools.webtools.crawl4ai.crawl_url",
}

RESEARCH_PROMPT = """
Research the business named in `input.business_query`.

`search_results` holds web search results for it. Crawl the pages that look
most likely to carry company facts — the official site first, then its about,
contact or imprint pages, and reputable profiles elsewhere — and keep crawling
until you can fill in the form or run out of promising pages.

Fill in every field you can support from the crawled pages, and leave a field
null rather than guessing at it.
"""

SUMMARY_PROMPT = """
You are given the researched company facts in `business_info`.

Copy every field of `business_info` into your answer unchanged — do not
correct, reformat or invent values — and add a `summary`: two or three
sentences describing what the company does and who it serves, written for
someone who has never heard of it.
"""


class BusinessQuery(BaseModel):
    """The workflow input."""

    model_config = ConfigDict(extra="forbid")

    business_query: str = Field(description="Name of the business to research.")
    # The engine records a run's `user_message` as the opening chat message, so
    # the backoffice conversation reads like a request rather than a dict dump.
    user_message: Optional[str] = Field(
        default=None, description="Human-readable version of the request."
    )


class BusinessInfo(BaseModel):
    """The company facts the research agent extracts."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The legal or trading name of the business.")
    address: Optional[str] = Field(description="The physical address of the business.")
    website: Optional[str] = Field(description="The official website URL.")
    phone: Optional[str] = Field(description="Contact phone number.")
    owners: Optional[str] = Field(description="The owners of the business.")
    description: str = Field(
        description="A brief description of what the business does."
    )
    industry: Optional[str] = Field(
        description="The industry the business operates in."
    )


class BusinessProfile(BusinessInfo):
    """The workflow output: the researched facts plus a written summary."""

    summary: str = Field(description="A two-to-three sentence summary of the company.")


console = Console()


def build_engine(
    *,
    model: str,
    max_steps: int,
    search_results: int,
    agent_service: Optional[AgentService],
    task_logger: Optional[PostgresTaskLogger],
) -> WorkflowEngine:
    """Assemble the research workflow.

    ``data_model`` registers the Pydantic models themselves — so the field
    descriptions that steer the model survive — while recording their JSON
    schema on the graph for the backoffice.
    """
    builder = WorkflowBuilder(
        AGENT_NAME, description=AGENT_DESCRIPTION, llm_model=model
    )
    builder.data_model("input", BusinessQuery)
    builder.data_model("search_results", WebSearchResponse)
    builder.data_model("business_info", BusinessInfo)
    builder.data_model("output", BusinessProfile)
    for name, path in TOOLS.items():
        builder.python_function(name, path)

    return (
        builder.start("search")
        # 1. Find candidate pages. Crawl4AI scrapes the DuckDuckGo HTML
        #    endpoint, so this needs no search API key.
        .function(
            "search",
            tool=f"python://{SEARCH_TOOL}",
            inputs={
                "query": "input.business_query",
                "count": {"type": "literal", "value": search_results},
            },
            output="search_results",
            next="research",
        )
        # 2. Read the promising ones. The agent decides how many pages to
        #    crawl, one tool call at a time, up to `max_steps`.
        .agent(
            "research",
            prompt=RESEARCH_PROMPT,
            inputs={"input": "input", "search_results": "search_results"},
            output="business_info",
            next="summarize",
            allowed_tools=[f"python://{CRAWL_TOOL}"],
            max_steps=max_steps,
            # Each step's plan is streamed, so the console shows progress.
            stream_instructions=True,
        )
        # 3. Write the summary and emit the final record.
        .llm(
            "summarize",
            prompt=SUMMARY_PROMPT,
            inputs={"business_info": "business_info"},
            output="output",
            next="end",
            # Each company is researched from scratch; earlier runs in the
            # session are not context for this one.
            use_history=False,
        )
        .end()
        .build_engine(agent_service=agent_service, task_logger=task_logger)
    )


def open_agent_service() -> Optional[AgentService]:
    """Build an :class:`AgentService` from the environment, if one is configured.

    Reading the environment is fine here: this script is an entry point, and
    only entry points may do so — library code takes its connection details as
    arguments.
    """
    db_uri = os.environ.get("KAVALAI_DB_URI")
    if not db_uri:
        return None
    session_maker = db_manager.get_sessionmaker(
        uri=db_uri,
        schema=os.environ.get("KAVALAI_DB_SCHEMA") or None,
    )
    return AgentService(session_maker)


def session_key(company: str) -> str:
    """A stable per-company session id, so re-runs land in the same session."""
    slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")
    return f"demo-{slug}"


async def research_company(
    company: str, *, engine: WorkflowEngine
) -> tuple[Optional[BusinessProfile], dict]:
    """Run the workflow for one company, reporting progress as it goes."""
    console.rule(f"[bold]{company}")

    output_data: Optional[dict] = None
    tokens: dict = {}
    try:
        async for event in engine.run_stream(
            {
                "business_query": company,
                "user_message": f"Research the business '{company}'.",
            },
            # Re-runs of the same company continue its session.
            external_id=session_key(company),
        ):
            if event.type == "node_started":
                console.print(f"[dim]node:[/dim] {event.name}")
            elif event.type == "complete" and event.name.endswith("_instructions"):
                console.print(f"  [dim]step:[/dim] {event.value}")
            elif event.type == "workflow_completed":
                output_data = event.output_data
                tokens = event.token_usage or {}
    except Exception as exc:
        # The engine has already recorded the failure on the run; keep the
        # demo going for the other companies.
        logger.exception(f"Research failed for {company}")
        console.print(f"[bold red]{company} failed: {exc}[/bold red]")
        return None, tokens

    if not output_data:
        console.print("[bold red]Workflow produced no output.[/bold red]")
        return None, tokens

    profile = BusinessProfile.model_validate(output_data)
    console.print(JSON(profile.model_dump_json()))
    return profile, tokens


def print_summary(rows: list[tuple[str, Optional[BusinessProfile], dict]]) -> None:
    """Print one line per researched company."""
    table = Table(title="Business research runs")
    table.add_column("Query")
    table.add_column("Name")
    table.add_column("Industry")
    table.add_column("Website")
    table.add_column("Model calls", justify="right")
    table.add_column("Tokens", justify="right")

    for company, profile, tokens in rows:
        table.add_row(
            company,
            profile.name if profile else "[red]failed[/red]",
            (profile.industry or "-") if profile else "-",
            (profile.website or "-") if profile else "-",
            str(tokens.get("model_calls", 0)),
            str(tokens.get("total_tokens", 0)),
        )
    console.print(table)


def parse_args(argv: Optional[list[str]] = None) -> Any:
    parser = argparse.ArgumentParser(
        description=(
            "Research businesses and populate the agent database with demo runs."
        )
    )
    parser.add_argument(
        "--company",
        action="append",
        dest="companies",
        metavar="NAME",
        help="Business to research; repeat for several. Defaults to Kaval.AI + FAANG.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("KAVALAI_DEFAULT_LLM_MODEL", DEFAULT_MODEL),
        help="LLM in 'provider/model' form (default: %(default)s).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum crawl/reasoning steps for the research agent "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--search-results",
        type=int,
        default=8,
        help="How many search results to hand the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Run the workflow without writing anything to the agent database.",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    companies = args.companies or DEFAULT_COMPANIES

    agent_service = None if args.no_db else open_agent_service()
    if not args.no_db and agent_service is None:
        console.print(
            "[yellow]KAVALAI_DB_URI is not set — running without persistence.[/yellow]"
        )
    task_logger = PostgresTaskLogger(agent_service) if agent_service else None

    engine = build_engine(
        model=args.model,
        max_steps=args.max_steps,
        search_results=args.search_results,
        agent_service=agent_service,
        task_logger=task_logger,
    )

    rows: list[tuple[str, Optional[BusinessProfile], dict]] = []
    try:
        for company in companies:
            profile, tokens = await research_company(company, engine=engine)
            rows.append((company, profile, tokens))
    finally:
        if task_logger:
            await task_logger.close()

    print_summary(rows)
    if agent_service:
        console.print(
            f"\nRecorded [bold]{len(rows)}[/bold] runs as agent "
            f"[bold]{AGENT_NAME}[/bold] — open the backoffice to browse them."
        )


if __name__ == "__main__":
    asyncio.run(main())
