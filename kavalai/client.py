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

import urllib.parse
from typing import Optional, Type

import httpx
from json_schema_to_pydantic import create_model
from pydantic import BaseModel

from kavalai.tools.openapi_spec_parser import OpenApiSpecParser


class AgentClient:
    """Async HTTP client for invoking a remote Kaval.AI agent server.

    Wraps the agent server's ``/run_agent`` and ``/stream_agent`` endpoints,
    discovering the agent's input/output schemas from its OpenAPI spec and
    transparently maintaining the conversation ``session_id`` across calls so
    successive invocations share the same session. Optional HTTP Basic Auth is
    used when both ``username`` and ``password`` are provided.

    Args:
        base_url: Base URL of the agent server.
        username: Optional HTTP Basic Auth username.
        password: Optional HTTP Basic Auth password.
        timeout: Per-request timeout in seconds.
        transport: Optional httpx transport, e.g. to route requests through a
            proxy or, in tests, to serve them without a network.
    """

    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password) if username and password else None
        self.timeout = timeout
        self.transport = transport
        self.session_id: Optional[str] = None
        self.input_schema: Optional[Type[BaseModel]] = None
        self.output_schema: Optional[Type[BaseModel]] = None

    def _http_client(self) -> httpx.AsyncClient:
        """Build the HTTP client used for one request."""
        return httpx.AsyncClient(
            auth=self.auth, timeout=self.timeout, transport=self.transport
        )

    async def discover_schemas(self):
        """Fetch the server's OpenAPI spec and derive the agent's schemas.

        Populates ``self.input_schema`` and ``self.output_schema`` with the
        Pydantic models for the agent's request and response payloads. Called
        automatically by :meth:`run_agent` and :meth:`stream_agent` on first
        use, but may be invoked directly to inspect the schemas up front.
        """
        async with self._http_client() as client:
            openapi_spec_url = urllib.parse.urljoin(self.base_url + "/", "openapi.json")
            resp = await client.get(openapi_spec_url)
            resp.raise_for_status()
            spec = resp.json()

        parser = OpenApiSpecParser(spec)

        self.input_schema = (
            create_model(parser.get_path_request_schema("/run_agent", "POST"))
            .model_fields["data"]
            .annotation
        )
        self.output_schema = (
            create_model(parser.get_path_response_schema("/run_agent", "POST"))
            .model_fields["data"]
            .annotation
        )

    async def run_agent(
        self, data: BaseModel, external_id: Optional[str] = None
    ) -> BaseModel:
        """Run the agent once and return its complete response.

        Sends ``data`` to the server's ``/run_agent`` endpoint, blocking until
        the run finishes. Updates ``self.session_id`` from the response so the
        next call continues the same conversation.

        Args:
            data: The request payload (an instance matching the agent's input
                schema).
            external_id: Optional caller-side identifier to correlate the
                session with an external system.

        Returns:
            An instance of the agent's output schema with the run's result.
        """
        if self.input_schema is None or self.output_schema is None:
            await self.discover_schemas()

        payload = {
            "session_id": self.session_id,
            "external_id": external_id,
            "data": data.model_dump(),
        }

        url = f"{self.base_url}/run_agent"

        async with self._http_client() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            response_json = resp.json()

        self.session_id = response_json.get("session_id")
        return self.output_schema(**response_json["data"])

    async def stream_agent(self, data: BaseModel, external_id: Optional[str] = None):
        """Run the agent and stream its output incrementally.

        Sends ``data`` to the server's ``/stream_agent`` (Server-Sent Events)
        endpoint and yields each ``data:`` chunk as a string as it arrives,
        letting callers consume partial output before the run completes.

        Args:
            data: The request payload (an instance matching the agent's input
                schema).
            external_id: Optional caller-side identifier to correlate the
                session with an external system.

        Yields:
            str: Successive content chunks from the streamed response.
        """
        if self.input_schema is None or self.output_schema is None:
            await self.discover_schemas()

        payload = {
            "session_id": self.session_id,
            "external_id": external_id,
            "data": data.model_dump(),
        }

        url = f"{self.base_url}/stream_agent"

        async with self._http_client() as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    async for chunk in self._process_stream_line(line):
                        yield chunk

    async def _process_stream_line(self, line: str):
        if line.startswith("data: "):
            yield line[6:]
