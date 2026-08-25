r"""A terminal chat against any running Kaval.AI agent server.

Talk to the server on the default port, http://localhost:10000::

    python -m examples.chat_client.chat_client

Point it at another server, with the basic auth it expects::

    python -m examples.chat_client.chat_client \
        --base-url http://agents.example.com:8000 --auth user:password

Give slow agents more time per reply, and show more of the other fields::

    python -m examples.chat_client.chat_client \
        --timeout 300 --extra-width 1000

Hide the other fields and see only the conversation::

    python -m examples.chat_client.chat_client --extra-width 0

For example, against the Green Village chatbot (start it first; see
``examples/green_village/README.md``)::

    KAVALAI_AGENT_WORKFLOW_PATH=examples/green_village/chatbot.yaml \
    KAVALAI_AGENT_SETUP_MODULE=examples/green_village/eval_setup.py \
        uv run --env-file .env python -m kavalai.server
    python -m examples.chat_client.chat_client

    you: How deep is Lake Miller?
    bot: Lake Miller is 1.2 metres deep at its deepest point.
         {"used_ids": ["fact-09"]}

Works with every agent whose request carries a ``user_message`` field and whose
reply carries an ``agent_response`` field — the shape shared by the chat
examples. Each turn POSTs the message to ``/run_agent``; the ``session_id``
that comes back is sent again on the next turn, so the whole conversation is
one session in the agent database and one conversation in the backoffice.

``agent_response`` is printed as the reply. Whatever else the agent returns
(sources, ids, routing decisions, …) is printed underneath as truncated JSON,
which makes this a handy way to poke at any chat workflow without writing a
client for its particular output type.
"""

import argparse
import json
import sys
import uuid
from typing import Any

import httpx

MESSAGE_FIELD = "user_message"
RESPONSE_FIELD = "agent_response"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:10000",
        help="Where the agent server listens (default: %(default)s).",
    )
    parser.add_argument(
        "--auth",
        metavar="USER:PASSWORD",
        help="HTTP basic auth, if the server has it configured.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for one reply (default: %(default)s).",
    )
    parser.add_argument(
        "--extra-width",
        type=int,
        default=200,
        help="Characters to show of the reply's other fields "
        "(default: %(default)s; 0 hides them).",
    )
    return parser.parse_args(argv)


def truncate(text: str, width: int) -> str:
    """Cut ``text`` to ``width`` characters, marking the cut with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(width - 1, 0)] + "…"


def format_reply(data: dict[str, Any], extra_width: int) -> str:
    """Render one ``data`` payload: the response, then the rest as JSON."""
    extra = {key: value for key, value in data.items() if key != RESPONSE_FIELD}
    if RESPONSE_FIELD in data:
        lines = [f"bot: {data[RESPONSE_FIELD]}"]
    else:
        # Not a chat-shaped reply; show everything so nothing is hidden.
        lines = [f"bot: (no '{RESPONSE_FIELD}' in reply)"]
    if extra and extra_width > 0:
        rendered = json.dumps(extra, ensure_ascii=False, default=str)
        lines.append(f"     {truncate(rendered, extra_width)}")
    return "\n".join(lines)


class ChatSession:
    """One conversation with an agent server, continued turn after turn."""

    def __init__(self, client: httpx.Client, external_id: str | None = None):
        self.client = client
        # A recognisable external id, so the conversation is easy to find on
        # the backoffice sessions page.
        self.external_id = external_id or f"chat-client:{uuid.uuid4().hex[:8]}"
        self.session_id: str | None = None

    def send(self, message: str) -> dict[str, Any]:
        """Send one user message and return the reply's ``data`` payload.

        Raises :class:`httpx.HTTPStatusError` when the server answers with an
        error status.
        """
        response = self.client.post(
            "/run_agent",
            json={
                "session_id": self.session_id,
                "external_id": self.external_id,
                "data": {MESSAGE_FIELD: message},
            },
        )
        response.raise_for_status()
        payload = response.json()
        self.session_id = payload.get("session_id")
        return payload.get("data") or {}


def chat_loop(session: ChatSession, extra_width: int, ask=input) -> None:
    """Relay terminal lines to the agent until the user quits.

    ``ask`` supplies each line; it is a parameter so the loop can be driven
    without a terminal.
    """
    while True:
        try:
            message = ask("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            return

        try:
            data = session.send(message)
        except httpx.HTTPStatusError as error:
            print(f"[{error.response.status_code}] {error.response.text}\n")
            continue
        except httpx.HTTPError as error:
            print(f"Request failed: {error}\n")
            continue
        print(format_reply(data, extra_width))
        print()


def main(argv: list[str] | None = None, ask=input) -> int:
    args = parse_args(argv)
    auth = tuple(args.auth.split(":", 1)) if args.auth else None

    with httpx.Client(
        base_url=args.base_url, auth=auth, timeout=args.timeout
    ) as client:
        try:
            client.get("/liveness").raise_for_status()
        except httpx.HTTPError as error:
            print(f"No agent server at {args.base_url}: {error}")
            print("Start one first — see README.md.")
            return 1

        session = ChatSession(client)
        print(f"Connected to {args.base_url} (session {session.external_id}).")
        print("Ctrl-D, 'quit' or 'exit' ends the chat.\n")
        chat_loop(session, args.extra_width, ask=ask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
