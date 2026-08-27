"""``chat_client.py`` — the terminal chat client, tested beside the example."""

import json

import httpx
import pytest

from examples.chat_client import chat_client
from examples.chat_client.chat_client import (
    ChatSession,
    chat_loop,
    format_reply,
    main,
    parse_args,
    truncate,
)


def scripted(*lines):
    """An ``ask`` callable replaying ``lines``, then raising EOF."""
    remaining = list(lines)

    def ask(prompt=""):
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return ask


class FakeServer:
    """Answers ``/liveness`` and ``/run_agent`` and records what it was sent."""

    def __init__(self, reply=None, status=200, alive=True):
        self.reply = reply if reply is not None else {"agent_response": "Hi"}
        self.status = status
        self.alive = alive
        self.requests = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/liveness":
            return httpx.Response(200 if self.alive else 503)
        assert request.url.path == "/run_agent"
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        return httpx.Response(200, json={"session_id": "sess-1", "data": self.reply})

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(self.handle)
        )


def test_truncate_marks_the_cut():
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("abcdef", 0) == "…"


def test_format_reply_prints_response_then_other_fields_as_json():
    text = format_reply({"agent_response": "Yes", "used_ids": ["fact-09"], "n": 1}, 200)
    assert text == 'bot: Yes\n     {"used_ids": ["fact-09"], "n": 1}'


def test_format_reply_truncates_and_can_hide_extra_fields():
    data = {"agent_response": "Yes", "blob": "x" * 500}
    shown = format_reply(data, 20).splitlines()[1].strip()
    assert len(shown) == 20 and shown.endswith("…")
    assert format_reply(data, 0) == "bot: Yes"


def test_format_reply_without_agent_response_shows_everything():
    text = format_reply({"answer": "42"}, 200)
    assert text == 'bot: (no \'agent_response\' in reply)\n     {"answer": "42"}'


def test_session_sends_user_message_and_continues_the_session():
    server = FakeServer()
    with server.client() as client:
        session = ChatSession(client, external_id="chat-client:test")
        assert session.send("hello") == {"agent_response": "Hi"}
        session.send("again")

    first, second = server.requests
    assert first == {
        "session_id": None,
        "external_id": "chat-client:test",
        "data": {"user_message": "hello"},
    }
    assert second["session_id"] == "sess-1"


def test_session_generates_a_recognisable_external_id():
    with FakeServer().client() as client:
        assert ChatSession(client).external_id.startswith("chat-client:")


def test_chat_loop_skips_blank_lines_and_stops_on_quit(capsys):
    server = FakeServer(reply={"agent_response": "Hi", "used_ids": ["a"]})
    with server.client() as client:
        chat_loop(ChatSession(client), 200, ask=scripted("", "  ", "hello", "quit"))

    assert [r["data"]["user_message"] for r in server.requests] == ["hello"]
    assert 'bot: Hi\n     {"used_ids": ["a"]}\n' in capsys.readouterr().out


def test_chat_loop_stops_on_end_of_input(capsys):
    server = FakeServer()
    with server.client() as client:
        chat_loop(ChatSession(client), 200, ask=scripted())
    assert server.requests == []


def test_chat_loop_reports_http_errors_and_keeps_going(capsys):
    server = FakeServer(status=500)
    with server.client() as client:
        chat_loop(ChatSession(client), 200, ask=scripted("one", "two", "exit"))

    assert len(server.requests) == 2
    assert "[500] boom" in capsys.readouterr().out


def test_chat_loop_reports_transport_errors_and_keeps_going(capsys):
    def explode(request):
        raise httpx.ConnectError("gone", request=request)

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(explode)
    ) as client:
        chat_loop(ChatSession(client), 200, ask=scripted("one", "exit"))

    assert "Request failed: gone" in capsys.readouterr().out


def test_parse_args_defaults_and_overrides():
    args = parse_args([])
    assert (args.base_url, args.auth, args.timeout, args.extra_width) == (
        "http://localhost:10000",
        None,
        120.0,
        200,
    )
    args = parse_args(
        [
            "--base-url",
            "http://x",
            "--auth",
            "u:p",
            "--timeout",
            "3",
            "--extra-width",
            "0",
        ]
    )
    assert (args.base_url, args.auth, args.timeout, args.extra_width) == (
        "http://x",
        "u:p",
        3.0,
        0,
    )


@pytest.fixture
def served(monkeypatch):
    """Route ``main``'s ``httpx.Client`` to a fake server, recording its args."""
    built = {}
    real_client = httpx.Client

    def install(server):
        def factory(**kwargs):
            built.update(kwargs)
            return real_client(
                base_url=kwargs["base_url"],
                transport=httpx.MockTransport(server.handle),
            )

        monkeypatch.setattr(chat_client.httpx, "Client", factory)
        return built

    return install


def test_main_connects_chats_and_exits_cleanly(served, capsys):
    server = FakeServer()
    built = served(server)

    code = main(
        ["--base-url", "http://test", "--auth", "u:p"], ask=scripted("hi", "exit")
    )

    assert code == 0
    assert built["auth"] == ("u", "p") and built["timeout"] == 120.0
    assert [r["data"]["user_message"] for r in server.requests] == ["hi"]
    out = capsys.readouterr().out
    assert out.startswith("Connected to http://test (session chat-client:")
    assert "bot: Hi" in out


def test_main_fails_when_no_server_answers(served, capsys):
    served(FakeServer(alive=False))

    assert main(["--base-url", "http://test"], ask=scripted("hi")) == 1
    assert "No agent server at http://test" in capsys.readouterr().out
