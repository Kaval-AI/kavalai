import pytest
from pydantic import BaseModel

from kavalai.tools import cli_chat
from kavalai.tools.cli_chat import chat_loop, main, parse_args


class ChatInput(BaseModel):
    user_message: str


class ChatOutput(BaseModel):
    agent_response: str


class FakeAgentClient:
    """A stand-in for AgentClient that records what the chat loop sent."""

    def __init__(self, input_schema=ChatInput, replies=("Hello",), fail_discovery=None):
        self.input_schema = input_schema
        self.output_schema = ChatOutput
        self.replies = list(replies)
        self.fail_discovery = fail_discovery
        self.sent = []

    async def discover_schemas(self):
        if self.fail_discovery is not None:
            raise self.fail_discovery

    async def stream_agent(self, data, external_id=None):
        self.sent.append(data)
        for reply in self.replies:
            yield reply


def scripted(*lines):
    """An ``ask`` callable replaying ``lines``, then raising EOF."""
    remaining = list(lines)

    def ask():
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return ask


async def test_chat_loop_streams_each_reply():
    client = FakeAgentClient(replies=("Hel", "Hello"))

    await chat_loop(client, ask=scripted("Hi there", "exit"))

    assert [message.user_message for message in client.sent] == ["Hi there"]


async def test_chat_loop_ignores_blank_input():
    client = FakeAgentClient()

    await chat_loop(client, ask=scripted("", "   ", "quit"))

    assert client.sent == []


async def test_chat_loop_stops_on_end_of_input():
    client = FakeAgentClient()

    await chat_loop(client, ask=scripted())

    assert client.sent == []


async def test_chat_loop_stops_when_the_schema_has_no_user_message():
    class OtherInput(BaseModel):
        question: str

    client = FakeAgentClient(input_schema=OtherInput)

    await chat_loop(client, ask=scripted("hello"))

    assert client.sent == []


async def test_chat_loop_reports_errors_and_keeps_going():
    class ExplodingClient(FakeAgentClient):
        async def stream_agent(self, data, external_id=None):
            raise RuntimeError("stream died")
            yield  # pragma: no cover - unreachable, keeps this an async generator

    client = ExplodingClient()

    # The error is reported, then the next line still gets a turn.
    await chat_loop(client, ask=scripted("first", "exit"))


def test_parse_args_reads_url_and_credentials():
    args = parse_args(
        ["--url", "http://localhost:8000/", "--user", "u", "--password", "p"]
    )
    assert (args.url, args.user, args.password) == ("http://localhost:8000/", "u", "p")


async def test_main_connects_and_runs_the_chat():
    client = FakeAgentClient()

    await main(["--url", "http://localhost:8000/"], client=client, ask=scripted("exit"))

    assert client.sent == []


async def test_main_exits_when_discovery_fails():
    client = FakeAgentClient(fail_discovery=RuntimeError("no server"))

    with pytest.raises(SystemExit) as exit_info:
        await main(["--url", "http://localhost:8000"], client=client)

    assert exit_info.value.code == 1


def test_ask_user_reads_from_the_rich_prompt(monkeypatch):
    monkeypatch.setattr(cli_chat.Prompt, "ask", lambda *args, **kwargs: "typed")
    assert cli_chat.ask_user() == "typed"


async def test_main_builds_a_client_from_the_url(monkeypatch):
    built = {}

    class RecordingClient(FakeAgentClient):
        def __init__(self, base_url, user, password):
            super().__init__()
            built.update(base_url=base_url, user=user, password=password)

    monkeypatch.setattr(cli_chat, "AgentClient", RecordingClient)

    await main(
        ["--url", "http://localhost:8000/", "--user", "u", "--password", "p"],
        ask=scripted("exit"),
    )

    assert built == {"base_url": "http://localhost:8000", "user": "u", "password": "p"}
