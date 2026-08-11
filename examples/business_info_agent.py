"""Example: a v2 Agent that researches a business using LangSearch + Crawl4AI."""

import asyncio
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.json import JSON

from kavalai import FunctionKernel, make_client
from kavalai.run_context import RunContext
from kavalai.agent import Agent
from kavalai.tools.websearch.langsearch import langsearch_web_search
from kavalai.tools.webtools.crawl4ai import crawl_url


class BusinessInfo(BaseModel):
    """Information about a business found online."""

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


console = Console()


async def main():
    if not os.environ.get("LANGSEARCH_API_KEY"):
        print("Please set LANGSEARCH_API_KEY environment variable.")
        return

    # 1. Register the tools the agent may call (addressed as python://<name>).
    #    Searching finds candidate pages; crawling reads the promising ones.
    kernel = FunctionKernel()
    kernel.register_python_tool(
        "websearch.langsearch_web_search", langsearch_web_search
    )
    kernel.register_python_tool("webtools.crawl_url", crawl_url)

    # 2. Seed the run context with the query the agent will research.
    run_context = RunContext()
    run_context.data["input"] = {"business_query": "Kaval AI"}

    # 3. Build the v2 agent on a capable model and run the multi-step loop.
    agent = Agent(
        llm_client=make_client("gemini/gemini-3.1-pro-preview"),
        kernel=kernel,
        run_context=run_context,
    )

    task = (
        "Find information about the business named in input.business_query using the "
        "web search and crawl tools, then fill in the business information form."
    )
    console.print(f"Running agent for task: [italic]{task}[/italic]")
    console.print("-" * 50)

    result = await agent.prompt(task, response_model=BusinessInfo, max_steps=10)

    if result:
        console.print(
            "\n[bold green]Successfully found business information:[/bold green]"
        )
        console.print(JSON(result.model_dump_json()))
    else:
        console.print(
            "\n[bold red]Agent failed to produce the requested output.[/bold red]"
        )

    await kernel.close()


if __name__ == "__main__":
    asyncio.run(main())
