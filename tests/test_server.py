import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from sqlalchemy.ext.asyncio import async_sessionmaker

import kavalai.server as server_module
from kavalai.settings import LLM_PARAMETER_VARIABLES
from kavalai.server import (
    create_agent_app,
    create_app_from_env_conf,
    run_agent_server,
    format_sse_event,
    mask_db_uri,
    session_scope,
    stream_sse_events,
    validate_auth,
)
from kavalai.workflow import WorkflowEngine
from kavalai.workflow.models import WorkflowException, WorkflowStreamEvent
from kavalai.llm_clients.base_client import BaseLlmClient

YAML = """
name: srv
description: test server
llm_model: openai/fake
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
nodes:
  - {name: start, type: start, next: reply}
  - name: reply
    type: llm
    prompt: hi
    inputs: {input: {type: context, value: input}}
    output: output
    next: end
    stream_output: true
  - {name: end, type: end, output: output}
"""


class FakeClient(BaseLlmClient):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        text = json.dumps({name: "hello" for name in response_model.model_fields})
        mid = len(text) // 2
        await value_streamer.stream_partial(text[:mid])
        await value_streamer.stream_partial(text[mid:])
        await value_streamer.stream_complete()


class BoomClient(BaseLlmClient):
    def __init__(self, *args, **kwargs):
        super().__init__()

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        raise RuntimeError("llm boom")


def _factory(model, parameters=None, stats_receiver=None):
    return FakeClient()


def make_client_app(client_factory=_factory, auth_dependency=lambda: None):
    engine = WorkflowEngine.from_yaml(YAML, client_factory=client_factory)
    app = create_agent_app(engine=engine, auth_dependency=auth_dependency)
    return TestClient(app)


def parse_sse(body: str) -> list[dict]:
    """Parse SSE frames into event payload dicts (skipping comment pings)."""
    events = []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def test_run_agent_returns_output():
    client = make_client_app()
    resp = client.post("/run_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 200
    assert resp.json()["data"]["agent_response"] == "hello"


def test_get_workflow_returns_v2_graph():
    client = make_client_app()
    resp = client.get("/workflow")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "srv"
    assert any(n["type"] == "llm" for n in body["nodes"])


def test_liveness():
    client = make_client_app()
    assert client.get("/liveness").json() == {"status": "ok"}


def test_stream_agent_event_sequence():
    client = make_client_app()
    resp = client.post("/stream_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "workflow_started"
    assert types[-1] == "workflow_completed"
    assert "partial" in types and "complete" in types

    # The node's content events are named after the node.
    partials = [e for e in events if e["type"] == "partial"]
    assert all(e["name"] == "reply" for e in partials)
    completed = events[-1]
    assert completed["output_data"] == {"agent_response": "hello"}


def test_stream_agent_failure_emits_workflow_failed():
    client = make_client_app(client_factory=lambda *a, **k: BoomClient())
    resp = client.post("/stream_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert events[-1]["type"] == "workflow_failed"
    assert "llm boom" in events[-1]["value"]


def test_stream_agent_requires_auth(monkeypatch):
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_USER", "u")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "p")
    client = make_client_app(auth_dependency=None)  # default basic auth

    resp = client.post("/stream_agent", json={"data": {"user_message": "hi"}})
    assert resp.status_code == 401

    resp = client.post(
        "/stream_agent", json={"data": {"user_message": "hi"}}, auth=("u", "p")
    )
    assert resp.status_code == 200
    assert parse_sse(resp.text)[-1]["type"] == "workflow_completed"


def test_format_sse_event():
    frame = format_sse_event(WorkflowStreamEvent(type="partial", name="n", value="v"))
    assert (
        frame == 'event: partial\ndata: {"type":"partial","name":"n","value":"v"}\n\n'
    )


@pytest.mark.asyncio
async def test_stream_sse_events_pings_during_silence():
    async def slow_events():
        yield WorkflowStreamEvent(type="node_started", name="a")
        await asyncio.sleep(0.1)
        yield WorkflowStreamEvent(type="node_completed", name="a")

    frames = []
    async for frame in stream_sse_events(slow_events(), ping_interval=0.02):
        frames.append(frame)

    assert any(f == ": ping\n\n" for f in frames)
    data_frames = [f for f in frames if f.startswith("event: ")]
    assert len(data_frames) == 2


@pytest.mark.asyncio
async def test_stream_sse_events_ends_quietly_on_workflow_exception():
    async def failing_events():
        yield WorkflowStreamEvent(type="workflow_failed", name="wf", value="boom")
        raise WorkflowException("boom")

    frames = [f async for f in stream_sse_events(failing_events(), ping_interval=1)]
    assert len(frames) == 1
    assert "workflow_failed" in frames[0]


@pytest.mark.asyncio
async def test_stream_sse_events_logs_unexpected_error_and_ends():
    async def broken_events():
        yield WorkflowStreamEvent(type="node_started", name="a")
        raise ValueError("unexpected")

    frames = [f async for f in stream_sse_events(broken_events(), ping_interval=1)]
    assert len(frames) == 1


@pytest.mark.asyncio
async def test_stream_sse_events_early_close_aborts_source():
    """Closing the SSE generator (client disconnect) cancels the event pump,
    which closes the source generator."""
    closed = asyncio.Event()

    async def endless_events():
        try:
            while True:
                yield WorkflowStreamEvent(type="node_started", name="a")
                await asyncio.sleep(0.01)
        finally:
            closed.set()

    gen = stream_sse_events(endless_events(), ping_interval=1)
    first = await gen.__anext__()
    assert first.startswith("event: node_started")
    await gen.aclose()
    await asyncio.wait_for(closed.wait(), timeout=1)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_validate_auth_allows_everything_when_unconfigured(monkeypatch):
    monkeypatch.delenv("KAVALAI_AGENT_BASIC_AUTH_USER", raising=False)
    monkeypatch.delenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", raising=False)
    assert validate_auth(None) is True


def test_validate_auth_accepts_matching_credentials(monkeypatch):
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_USER", "u")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "p")
    credentials = HTTPBasicCredentials(username="u", password="p")
    assert validate_auth(credentials) is True


def test_validate_auth_rejects_missing_credentials(monkeypatch):
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_USER", "u")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "p")
    with pytest.raises(HTTPException) as exc:
        validate_auth(None)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Authentication required"


def test_validate_auth_rejects_wrong_credentials(monkeypatch):
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_USER", "u")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "p")
    with pytest.raises(HTTPException) as exc:
        validate_auth(HTTPBasicCredentials(username="u", password="nope"))
    assert exc.value.detail == "Incorrect username/password"


# ---------------------------------------------------------------------------
# session_scope
# ---------------------------------------------------------------------------


class FakeSession:
    def __init__(self):
        self.executed = []
        self.closed = False

    async def execute(self, statement):
        self.executed.append(statement)

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.close()
        return False


@pytest.mark.asyncio
async def test_session_scope_opens_a_session_from_a_factory():
    session = FakeSession()

    class FakeFactory(async_sessionmaker):
        def __call__(self, **kwargs):
            return session

    factory = FakeFactory.__new__(FakeFactory)

    async with session_scope(factory) as scoped:
        assert scoped is session
    assert session.closed is True


@pytest.mark.asyncio
async def test_session_scope_yields_an_existing_session():
    session = FakeSession()
    async with session_scope(session) as scoped:
        assert scoped is session
    # A borrowed session is not closed by the scope.
    assert session.closed is False


# ---------------------------------------------------------------------------
# health / mask_db_uri
# ---------------------------------------------------------------------------


def test_health_reports_a_connected_database():
    engine = WorkflowEngine.from_yaml(YAML, client_factory=_factory)
    session = FakeSession()
    app = create_agent_app(
        engine=engine, session_provider=session, auth_dependency=lambda: None
    )

    resp = TestClient(app).get("/health")

    assert resp.json() == {"status": "ok", "database": "connected"}


def test_health_reports_a_broken_database():
    class BrokenSession(FakeSession):
        async def execute(self, statement):
            raise RuntimeError("db down")

    engine = WorkflowEngine.from_yaml(YAML, client_factory=_factory)
    app = create_agent_app(
        engine=engine, session_provider=BrokenSession(), auth_dependency=lambda: None
    )

    resp = TestClient(app, raise_server_exceptions=False).get("/health")

    assert resp.status_code == 503


def test_mask_db_uri_hides_the_password():
    masked = mask_db_uri("postgresql+asyncpg://user:secret@localhost:5432/db")
    assert masked == "postgresql+asyncpg://user:***@localhost:5432/db"


def test_mask_db_uri_passes_through_uris_without_credentials():
    assert mask_db_uri("sqlite+aiosqlite:///local.db") == "sqlite+aiosqlite:///local.db"


def test_mask_db_uri_masks_everything_it_cannot_parse():
    assert mask_db_uri("weird@uri") == "***"


def test_mask_db_uri_keeps_a_credential_free_authority():
    # An authority with a user but no password has nothing to mask.
    assert (
        mask_db_uri("postgresql+asyncpg://user@localhost:5432/db")
        == "postgresql+asyncpg://user@localhost:5432/db"
    )


@pytest.fixture
def env_configured_agent(tmp_path, monkeypatch):
    """Environment describing a runnable agent server (no database contact)."""
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(YAML, encoding="utf-8")

    monkeypatch.setenv("KAVALAI_AGENT_WORKFLOW_PATH", str(workflow_path))
    monkeypatch.setenv(
        "KAVALAI_DB_URI", "postgresql+asyncpg://user:secret@localhost:5432/db"
    )
    monkeypatch.setenv("KAVALAI_DB_SCHEMA", "agents")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_USER", "u")
    monkeypatch.setenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "p")
    # A developer's .env may set fleet defaults; the tests decide their own.
    # ``environs`` falls back to the .env file it read at import, so that
    # copy has to go as well.
    monkeypatch.setattr(server_module.env, "_environ", {})
    for name in (
        "KAVALAI_DEFAULT_LLM_MODEL",
        "KAVALAI_EMBEDDING_NORMALIZER_YAML",
        *LLM_PARAMETER_VARIABLES,
    ):
        monkeypatch.delenv(name, raising=False)
    return workflow_path


def test_create_app_from_env_conf_builds_the_agent(env_configured_agent):
    app = create_app_from_env_conf()

    assert app.state.engine.graph.name == "srv"


def test_create_app_from_env_conf_passes_the_llm_defaults_to_the_engine(
    env_configured_agent, monkeypatch
):
    """The engine reads no environment variable; the server hands them over."""
    monkeypatch.setenv("KAVALAI_DEFAULT_LLM_MODEL", "openai/fleet-default")
    monkeypatch.setenv("KAVALAI_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("KAVALAI_LLM_TIMEOUT_SECONDS", "12")

    app = create_app_from_env_conf()

    engine = app.state.engine
    assert engine.default_llm_model == "openai/fleet-default"
    assert engine.default_llm_parameters == {
        "temperature": 0.1,
        "timeout_seconds": 12.0,
    }


def test_create_app_from_env_conf_without_llm_defaults(env_configured_agent):
    app = create_app_from_env_conf()

    assert app.state.engine.default_llm_model is None
    assert app.state.engine.default_llm_parameters == {}


def test_create_app_from_env_conf_installs_the_normalizer(
    env_configured_agent, monkeypatch, tmp_path
):
    import kavalai.normalizer
    from kavalai.normalizer import Normalizer, get_default_normalizer

    path = tmp_path / "normalizer.yaml"
    Normalizer(l1=True, l2=False).save_to_yaml(str(path))
    monkeypatch.setenv("KAVALAI_EMBEDDING_NORMALIZER_YAML", str(path))
    monkeypatch.setattr(kavalai.normalizer, "_default_normalizer", None)

    create_app_from_env_conf()

    assert get_default_normalizer().l1 is True


def test_create_app_from_env_conf_arguments_override_the_environment(
    env_configured_agent, monkeypatch
):
    monkeypatch.delenv("KAVALAI_AGENT_BASIC_AUTH_USER", raising=False)
    monkeypatch.delenv("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", raising=False)

    app = create_app_from_env_conf(
        workflow_path=str(env_configured_agent),
        db_uri="postgresql+asyncpg://user:secret@localhost:5432/other",
        db_schema="public",
        pool_size=1,
        max_overflow=2,
        sql_echo=True,
    )

    assert app.state.engine.graph.name == "srv"


def test_run_agent_server_serves_the_configured_app(env_configured_agent, monkeypatch):
    served = {}

    def fake_run(app, host, port):
        served.update(app=app, host=host, port=port)

    monkeypatch.setattr(server_module.uvicorn, "run", fake_run)
    monkeypatch.setenv("KAVALAI_AGENT_HOST", "127.0.0.1")
    monkeypatch.setenv("KAVALAI_AGENT_PORT", "1234")

    run_agent_server()

    assert served["host"] == "127.0.0.1"
    assert served["port"] == 1234
    assert served["app"].state.engine.graph.name == "srv"


def test_get_workflow_redacts_mcp_server_secrets():
    """`env` is where an API key for a stdio MCP server ends up.

    The endpoint sits behind the same authentication as everything else, which
    means public when none is configured — so the values must not be in the
    payload at all. The keys stay: which variables a server needs is useful,
    what they are set to is not.
    """
    yaml_with_secret = YAML.replace(
        "nodes:",
        """mcp_servers:
  - name: github
    command: mcp-github
    env:
      GITHUB_TOKEN: ghp_supersecret
nodes:""",
        1,
    )
    engine = WorkflowEngine.from_yaml(yaml_with_secret, client_factory=_factory)
    client = TestClient(create_agent_app(engine=engine, auth_dependency=lambda: None))

    resp = client.get("/workflow")
    assert resp.status_code == 200
    assert "ghp_supersecret" not in resp.text

    (server,) = resp.json()["mcp_servers"]
    assert server["env"] == {"GITHUB_TOKEN": "***"}
    assert server["command"] == "mcp-github"


# ------------------------------------------------------- provider module loading
def test_load_provider_modules_imports_named_modules():
    from kavalai.server import load_provider_modules

    assert load_provider_modules("json, os") == ["json", "os"]


def test_load_provider_modules_ignores_blanks():
    from kavalai.server import load_provider_modules

    assert load_provider_modules("") == []
    assert load_provider_modules(" , ") == []


def test_load_provider_modules_fails_loudly_on_a_missing_module():
    from kavalai.server import load_provider_modules

    with pytest.raises(ImportError):
        load_provider_modules("kavalai_no_such_provider_module")


def test_load_provider_modules_verifies_registrations(monkeypatch):
    """A mistyped dotted path fails at start-up, not at the first request."""
    from kavalai.llm_clients import registry
    from kavalai.server import load_provider_modules

    registry.register_llm_provider("broken-at-boot", "no.such.module.Client")
    try:
        with pytest.raises(registry.RegistryError, match="broken-at-boot"):
            load_provider_modules("")
    finally:
        registry.llm_providers.unregister("broken-at-boot")
