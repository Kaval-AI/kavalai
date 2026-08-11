import json

import httpx
import pytest
from pydantic import BaseModel

from kavalai.client import AgentClient


class MockInput(BaseModel):
    user_message: str


class MockOutput(BaseModel):
    agent_response: str


OPENAPI_SPEC = {
    "paths": {
        "/run_agent": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "properties": {
                                    "data": {
                                        "properties": {
                                            "user_message": {"type": "string"}
                                        },
                                        "type": "object",
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "properties": {
                                        "data": {
                                            "properties": {
                                                "agent_response": {"type": "string"}
                                            },
                                            "type": "object",
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
        }
    }
}

SSE_BODY = (
    "event: partial\n"
    'data: {"type": "partial", "name": "output", "value": "Hel"}\n'
    "\n"
    ": ping\n"
    "\n"
    "event: complete\n"
    'data: {"type": "complete", "name": "output", "value": "Hello"}\n'
    "\n"
)


def make_client(handler, **kwargs):
    """An AgentClient whose requests are served by ``handler``."""
    return AgentClient(
        "http://testserver", transport=httpx.MockTransport(handler), **kwargs
    )


def agent_server(requests=None, run_response=None):
    """A handler standing in for a Kaval.AI agent server."""
    run_response = run_response or {
        "session_id": "test-session",
        "data": {"agent_response": "Hello world"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=OPENAPI_SPEC)
        if request.url.path == "/run_agent":
            return httpx.Response(200, json=run_response)
        if request.url.path == "/stream_agent":
            return httpx.Response(
                200, text=SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_agent_client_discover_schemas():
    client = make_client(agent_server())

    await client.discover_schemas()

    assert "user_message" in client.input_schema.model_fields
    assert "agent_response" in client.output_schema.model_fields


@pytest.mark.asyncio
async def test_agent_client_run_agent():
    requests = []
    client = make_client(agent_server(requests))
    client.input_schema = MockInput
    client.output_schema = MockOutput

    result = await client.run_agent(MockInput(user_message="Hi"))

    assert result.agent_response == "Hello world"
    assert client.session_id == "test-session"

    (run_request,) = requests
    assert json.loads(run_request.content) == {
        "session_id": None,
        "external_id": None,
        "data": {"user_message": "Hi"},
    }


@pytest.mark.asyncio
async def test_run_agent_discovers_schemas_on_first_use():
    requests = []
    client = make_client(agent_server(requests))

    result = await client.run_agent(MockInput(user_message="Hi"))

    assert result.agent_response == "Hello world"
    # The spec was fetched before the run.
    assert [r.url.path for r in requests] == ["/openapi.json", "/run_agent"]


@pytest.mark.asyncio
async def test_run_agent_sends_the_session_and_external_id():
    requests = []
    client = make_client(agent_server(requests))
    client.input_schema = MockInput
    client.output_schema = MockOutput
    client.session_id = "existing-session"

    await client.run_agent(MockInput(user_message="Hi"), external_id="ext-1")

    body = json.loads(requests[0].content)
    assert body["session_id"] == "existing-session"
    assert body["external_id"] == "ext-1"


@pytest.mark.asyncio
async def test_agent_client_stream_agent():
    client = make_client(agent_server())
    client.input_schema = MockInput
    client.output_schema = MockOutput

    chunks = [
        chunk async for chunk in client.stream_agent(MockInput(user_message="Hi"))
    ]

    # Only `data:` payloads are yielded; SSE comments and event lines are not.
    assert chunks == [
        '{"type": "partial", "name": "output", "value": "Hel"}',
        '{"type": "complete", "name": "output", "value": "Hello"}',
    ]


@pytest.mark.asyncio
async def test_stream_agent_discovers_schemas_on_first_use():
    requests = []
    client = make_client(agent_server(requests))

    chunks = [
        chunk async for chunk in client.stream_agent(MockInput(user_message="Hi"))
    ]

    assert len(chunks) == 2
    assert [r.url.path for r in requests] == ["/openapi.json", "/stream_agent"]


@pytest.mark.asyncio
async def test_basic_auth_is_sent_when_both_credentials_are_given():
    requests = []
    client = make_client(agent_server(requests), username="user", password="pass")
    client.input_schema = MockInput
    client.output_schema = MockOutput

    await client.run_agent(MockInput(user_message="Hi"))

    assert requests[0].headers["authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_agent_client_auth_only_username():
    client = AgentClient("http://testserver", username="user")
    assert client.auth is None


@pytest.mark.asyncio
async def test_agent_client_auth_only_password():
    client = AgentClient("http://testserver", password="pass")
    assert client.auth is None


@pytest.mark.asyncio
async def test_server_errors_are_raised():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = make_client(failing)
    client.input_schema = MockInput
    client.output_schema = MockOutput

    with pytest.raises(httpx.HTTPStatusError):
        await client.run_agent(MockInput(user_message="Hi"))
