"""Shared plumbing for the evaluators: one agent call, one result.

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

from typing import Any, Optional

import httpx
from pydantic import BaseModel

from kavalai.client import AgentClient

DEFAULT_BASE_URL = "http://localhost:10000"


class EvalResult(BaseModel):
    """The verdict on a single case.

    Attributes:
        name: Name of the case, for reporting.
        passed: Whether the agent's answer satisfied the expectation.
        reason: Why it failed; empty when it passed.
        inputs: What was sent to the agent.
        output: What the agent answered, or ``None`` if the run never got
            that far (a connection error, an input that does not fit the
            agent's input type).
    """

    name: str
    passed: bool
    reason: str = ""
    inputs: dict[str, Any] = {}
    output: Optional[dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.passed


class AgentEvaluator:
    """Base class for evaluators that call a *running* agent server.

    It owns one :class:`~kavalai.client.AgentClient`, which discovers the
    agent's input and output types from the server's OpenAPI spec on first
    use. Subclasses only have to decide whether an answer is good.

    Args:
        base_url: Where the agent server listens.
        username: HTTP Basic Auth user, if the server requires one.
        password: HTTP Basic Auth password.
        timeout: Seconds to wait for one agent run.
        transport: Optional httpx transport — in tests, this serves the
            requests without a network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url
        self.client = AgentClient(
            base_url,
            username=username,
            password=password,
            timeout=timeout,
            transport=transport,
        )

    async def run_agent(
        self, inputs: dict[str, Any], external_id: Optional[str] = None
    ) -> BaseModel:
        """Send one input to the agent and return its output.

        Each call starts a fresh session, so cases cannot leak conversation
        history into each other. ``inputs`` is validated against the agent's
        own input type before it is sent, which turns a mistyped field into a
        clear error instead of a puzzling answer.

        Args:
            inputs: Field values for the agent's input type.
            external_id: Optional identifier recorded with the session.

        Returns:
            An instance of the agent's output type.
        """
        if self.client.input_schema is None or self.client.output_schema is None:
            await self.client.discover_schemas()
        data = self.client.input_schema(**inputs)
        self.client.session_id = None
        return await self.client.run_agent(data, external_id=external_id)

    async def evaluate(
        self,
        inputs: dict[str, Any],
        expected: Any = None,
        name: str = "case",
    ) -> EvalResult:
        """Run one case and judge the answer."""
        raise NotImplementedError
