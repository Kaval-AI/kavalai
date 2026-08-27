"""Research a list of businesses from the command line, and record the runs.

The workflow itself is `business_info.py`; this module runs it once per company
and prints what came back. Everything that ends up in the database comes from
the library: hand the engine an
:class:`~kavalai.agent_service.AgentService` and a task logger and it records
the runs itself — the agent and its graph, a session and run per company, a
task per node, the chat history and a model-call stat per LLM call. This script
only builds the workflow and invokes it once per company.

That database is in-memory SQLite, so this example needs nothing set up and
leaves nothing behind: the recording is there to be watched, not kept. `--no-db`
turns it off entirely, and the runs still work — which is the point worth
seeing, that persistence is something the engine is handed rather than
something it needs.

Usage::

    # Research the default company list.
    uv run --env-file .env python -m examples.business_info_agent.research_companies

    # Pick the companies, and keep the database out of it.
    uv run --env-file .env python -m examples.business_info_agent.research_companies \
        --company "Spotify" --no-db

Reads the live web through Crawl4AI's browser, so a company takes tens of
seconds.
"""

import argparse
import asyncio
import os
import re
from typing import Any, Optional

from loguru import logger
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from examples.business_info_agent.business_info import (
    AGENT_NAME,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_RESULTS,
    BusinessProfile,
    build_engine,
)
from kavalai.agent_service import AgentService
from kavalai.db import db_manager
from kavalai.workflow import WorkflowEngine

# Named for the database it was written against, but it records through the
# `AgentService` it is handed — here, one over in-memory SQLite.
from kavalai.workflow.tasklog.postgres import PostgresTaskLogger

# In-memory SQLite: the agent, sessions, runs, tasks and model-call stats live
# only as long as the process.
AGENT_DB_PATH = ":memory:"

# Kaval.AI first, then five companies any model already has opinions about:
# researching both in one run shows what the agent found rather than what it
# remembered.
DEFAULT_COMPANIES = [
    "Kaval.AI (kaval.ai)",
    "Meta Platforms",
    "Apple Inc.",
    "Amazon.com",
    "Netflix",
    "Alphabet (Google)",
]

console = Console()


async def open_agent_service() -> AgentService:
    """Create the in-memory agent database and a service over it.

    The tables are created here rather than by Alembic: a database that
    disappears with the process has no migration history to keep.
    """
    await db_manager.init_sqlite(db_path=AGENT_DB_PATH)
    return AgentService(db_manager.get_sqlite_sessionmaker(db_path=AGENT_DB_PATH))


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
        default=DEFAULT_MAX_STEPS,
        help="Maximum crawl/reasoning steps for the research agent "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--search-results",
        type=int,
        default=DEFAULT_SEARCH_RESULTS,
        help="How many search results to hand the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Run the workflow without recording anything at all.",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    companies = args.companies or DEFAULT_COMPANIES

    agent_service = None if args.no_db else await open_agent_service()
    task_logger = PostgresTaskLogger(agent_service) if agent_service else None

    engine = build_engine(
        model=args.model,
        max_steps=args.max_steps,
        search_results=args.search_results,
        agent_service=agent_service,
        task_logger=task_logger,
        # Each step's plan is streamed, so the console shows progress.
        stream_instructions=True,
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
            f"[bold]{AGENT_NAME}[/bold], in a database that ends with this "
            "process."
        )


if __name__ == "__main__":
    asyncio.run(main())
