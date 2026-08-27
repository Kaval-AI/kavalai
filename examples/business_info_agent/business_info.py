"""The business research workflow: search the web, crawl, write a profile.

The graph is a four-step pipeline, one node per step::

    start → search (function) → research (agent) → summarize (llm) → end

1. ``search`` runs a keyless web search for the company name. Crawl4AI scrapes
   the DuckDuckGo HTML endpoint, so no search API key is involved.
2. ``research`` is an agent node: it reads the search results, crawls the pages
   worth reading — deciding for itself how many steps that takes — and fills in
   the structured company facts. It is restricted to the crawl tool, so it
   cannot re-run the search.
3. ``summarize`` writes a short company summary and emits the final record.

This module is only the agent. Where it runs and what it records is decided by
the two entry points beside it: ``business_info_in_memory.py`` serves it over
REST — that is what ``eval_cases.yaml`` is run against — and
``research_companies.py`` runs a list of companies through it from the command
line. Both record into an in-memory SQLite database, so the example needs
nothing set up and leaves nothing behind.

Needs Crawl4AI's browser (``uv sync --extra common`` plus
``playwright install chromium``); the pages it reads are the live web.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from kavalai.agent_service import AgentService
from kavalai.tools.webtools.crawl4ai import WebSearchResponse
from kavalai.workflow import WorkflowBuilder, WorkflowEngine
from kavalai.workflow.tasklog.base import TaskLogger

AGENT_NAME = "business-info-agent"
AGENT_DESCRIPTION = (
    "Searches the web for a business, crawls the pages worth reading and "
    "returns a structured company profile with a short summary."
)

DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"

#: How many search results the research agent is handed.
DEFAULT_SEARCH_RESULTS = 8
#: How many crawl/reasoning steps it may take before it has to answer.
DEFAULT_MAX_STEPS = 8

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
null rather than guessing at it. A page that only mentions the company in
passing is not support for a fact about it.
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


def build_engine(
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    search_results: int = DEFAULT_SEARCH_RESULTS,
    agent_service: Optional[AgentService] = None,
    task_logger: Optional[TaskLogger] = None,
    stream_instructions: bool = False,
) -> WorkflowEngine:
    """Assemble the research workflow.

    ``data_model`` registers the Pydantic models themselves — so the field
    descriptions that steer the model survive — while recording their JSON
    schema on the graph for the backoffice.

    Args:
        model: The LLM in ``provider/model`` form.
        max_steps: Crawl/reasoning steps the research agent may take.
        search_results: How many search results it is handed to start from.
        agent_service: Where sessions, runs and chat history are recorded;
            ``None`` runs the workflow without persistence.
        task_logger: Where per-node tasks are recorded.
        stream_instructions: Stream each agent step's plan, so a console
            caller can show progress. A server has nothing to do with it.
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
            stream_instructions=stream_instructions,
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
