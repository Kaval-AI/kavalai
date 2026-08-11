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

import asyncio
import sys
from argparse import ArgumentParser

from rich.console import Console
from rich.prompt import Prompt

from kavalai.client import AgentClient

console = Console()


def ask_user() -> str:
    """Read one chat line from the terminal."""
    return Prompt.ask("[bold cyan]You[/bold cyan]")


async def chat_loop(client: AgentClient, ask=ask_user):
    """Relay terminal input to the agent until the user quits.

    ``ask`` supplies each line; it is a parameter so the loop can be driven
    without a terminal.
    """
    while True:
        try:
            user_input = ask()
            if user_input.lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            # Assuming the agent expects a 'user_message' field based on the issue description.
            # If the schema is different, this might need more complex input handling.
            if "user_message" not in client.input_schema.model_fields:
                console.print(
                    f"[bold red]Error: Agent input schema does not have 'user_message' field. Fields: {client.input_schema.model_fields.keys()}[/bold red]"
                )
                break

            data = client.input_schema(user_message=user_input)

            console.print("[bold yellow]Agent:[/bold yellow] ", end="")
            async for line in client.stream_agent(data):
                console.print(line)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


def parse_args(argv=None):
    """Parse the chat tool's command line."""
    parser = ArgumentParser(description="CLI Chat tool for Kaval.AI agents.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Agent server URL with port (e.g. http://localhost:8000)",
    )
    parser.add_argument("--user", type=str, help="Basic auth username")
    parser.add_argument("--password", type=str, help="Basic auth password")

    return parser.parse_args(argv)


async def main(argv=None, client: AgentClient = None, ask=ask_user):
    args = parse_args(argv)

    base_url = args.url.rstrip("/")
    if client is None:
        client = AgentClient(base_url, args.user, args.password)

    console.print(f"[bold green]Connecting to agent at {base_url}...[/bold green]")
    try:
        await client.discover_schemas()
    except Exception as e:
        console.print(
            f"[bold red]Failed to connect or discover schemas: {e}[/bold red]"
        )
        sys.exit(1)

    console.print(
        "[bold blue]Connected! Type 'exit' or 'quit' to end the chat.[/bold blue]"
    )
    console.print(f"Input schema: {client.input_schema.model_fields.keys()}")
    console.print(f"Output schema: {client.output_schema.model_fields.keys()}")

    await chat_loop(client, ask=ask)

    console.print("[bold blue]Goodbye![/bold blue]")


if __name__ == "__main__":  # pragma: no cover - script entry point
    asyncio.run(main())
