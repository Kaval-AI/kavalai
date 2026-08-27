"""Let a model decide whether an agent's answer is acceptable.

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

import json
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from kavalai.eval.base import DEFAULT_BASE_URL, AgentEvaluator, EvalResult
from kavalai.llm_clients.base_client import BaseLlmClient
from kavalai.workflow.clients import make_client

DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-mini"

JUDGE_PROMPT = """You are grading one response of an AI agent under test.

The agent was given this input:
{inputs}

The agent responded with:
{output}

The response passes if, and only if, it satisfies this criterion:
{criterion}

Judge the criterion above and nothing else — do not add requirements of your
own, and do not reward or punish style, length or politeness unless the
criterion asks about them. Set `passed` to true or false. When it is false,
`reason` must say in one sentence what the response got wrong; when it is
true, leave `reason` empty.
"""


class JudgeVerdict(BaseModel):
    """What the judging model answers."""

    passed: bool
    reason: str = ""


class JudgeEvaluator(AgentEvaluator):
    """Send one input to a running agent and have a model grade the answer.

    Use it where the right answer cannot be written down in advance — an
    explanation, a refusal, a comparison, anything where wording varies but
    the substance must hold. The expectation is a plain-language criterion,
    and a failing case comes back with the judge's reason.

    The judging model is only built when a case is actually judged, so a run
    of purely :class:`~kavalai.eval.simple_evaluator.SimpleEvaluator` cases
    needs no API key.

    Args:
        base_url: Where the agent server listens.
        username: HTTP Basic Auth user, if the server requires one.
        password: HTTP Basic Auth password.
        timeout: Seconds to wait for one agent run.
        transport: Optional httpx transport, used by tests.
        model: ``provider/model`` of the judge.
        llm_client: A ready-made judge client, overriding ``model``.
        prompt: The grading prompt; must accept ``{inputs}``, ``{output}``
            and ``{criterion}``.

    Example:
        .. code-block:: python

            evaluator = JudgeEvaluator("http://localhost:25000")
            result = await evaluator.evaluate(
                {"user_message": "What is the village's annual budget?"},
                "The answer says the information is not available "
                "instead of inventing a number.",
            )
            assert result.passed, result.reason
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        model: str = DEFAULT_JUDGE_MODEL,
        llm_client: Optional[BaseLlmClient] = None,
        prompt: str = JUDGE_PROMPT,
    ):
        super().__init__(
            base_url,
            username=username,
            password=password,
            timeout=timeout,
            transport=transport,
        )
        self.model = model
        self.prompt = prompt
        self._llm_client = llm_client

    @property
    def llm_client(self) -> BaseLlmClient:
        """The judging model, built on first use."""
        if self._llm_client is None:
            self._llm_client = make_client(self.model)
        return self._llm_client

    def build_prompt(
        self, inputs: dict[str, Any], output: dict[str, Any], criterion: str
    ) -> str:
        """Render the grading prompt for one case."""
        return self.prompt.format(
            inputs=json.dumps(inputs, ensure_ascii=False, indent=2, default=str),
            output=json.dumps(output, ensure_ascii=False, indent=2, default=str),
            criterion=criterion,
        )

    async def judge(
        self, inputs: dict[str, Any], output: dict[str, Any], criterion: str
    ) -> JudgeVerdict:
        """Ask the judging model whether one answer meets the criterion."""
        return await self.llm_client.prompt(
            self.build_prompt(inputs, output, criterion),
            response_model=JudgeVerdict,
        )

    async def evaluate(
        self,
        inputs: dict[str, Any],
        expected: Optional[str] = None,
        name: str = "case",
    ) -> EvalResult:
        """Run one case and let the judging model grade the answer.

        Args:
            inputs: Field values for the agent's input type.
            expected: The criterion the answer has to satisfy, in plain
                language.
            name: Case name, carried into the result.

        Returns:
            The :class:`~kavalai.eval.base.EvalResult` for this case, its
            ``reason`` taken from the judge when the case failed.

        Raises:
            ValueError: ``expected`` is missing. A judged case with nothing
                to judge against would pass on anything at all.
        """
        if not expected:
            raise ValueError(
                f"Case '{name}' is judged but states no criterion to judge it by."
            )

        try:
            output = (
                await self.run_agent(inputs, external_id=f"eval:{name}")
            ).model_dump()
        except Exception as e:
            return EvalResult(
                name=name,
                passed=False,
                reason=f"the agent run failed: {e}",
                inputs=inputs,
            )

        try:
            verdict = await self.judge(inputs, output, expected)
        except Exception as e:
            return EvalResult(
                name=name,
                passed=False,
                reason=f"the judge failed: {e}",
                inputs=inputs,
                output=output,
            )

        return EvalResult(
            name=name,
            passed=verdict.passed,
            reason="" if verdict.passed else (verdict.reason or "the judge said no"),
            inputs=inputs,
            output=output,
        )
