"""A stand-in agent server, served over an httpx mock transport.

Every test here runs against it rather than a live server, so the suite
needs neither a port nor an API key.
"""

import httpx
import pytest

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
                                        "required": ["user_message"],
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
                                                "agent_response": {"type": "string"},
                                                "used_ids": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
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


def agent_transport(reply=None, requests=None, status_code=200):
    """A transport answering as an agent server would.

    Args:
        reply: The ``data`` payload the agent answers with, or a callable
            taking the request's input data and returning one.
        requests: Optional list collecting every request that was made.
        status_code: Status code for ``/run_agent``, to simulate a server
            that is unhappy.
    """
    reply = reply if reply is not None else {"agent_response": "Hello world"}

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=OPENAPI_SPEC)
        if request.url.path == "/run_agent":
            if status_code != 200:
                return httpx.Response(status_code, json={"detail": "nope"})
            import json

            data = json.loads(request.content)["data"]
            payload = reply(data) if callable(reply) else reply
            return httpx.Response(200, json={"session_id": None, "data": payload})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def transport():
    """The default stand-in agent: it always answers "Hello world"."""
    return agent_transport()


class FakeJudge:
    """A judging model that returns whatever verdict a test asks for.

    Stands in for a :class:`~kavalai.BaseLlmClient` — the evaluator only ever
    calls ``prompt``.
    """

    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error
        self.prompts = []

    async def prompt(self, system_message, response_model=None):
        self.prompts.append(system_message)
        if self.error is not None:
            raise self.error
        return self.verdict
