import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from kavalai.server import (
    create_agent_app,
    format_sse_event,
    stream_sse_events,
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
