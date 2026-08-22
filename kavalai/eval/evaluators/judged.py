"""Evaluators that ask a model, for the things only judgement can score.

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

import hashlib
import json
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from kavalai.eval.evaluators.base import Evaluator, evaluator
from kavalai.eval.models import Case, Score
from kavalai.eval.targets import RunRecord
from kavalai.utils import to_plain

#: Judges are graders, not authors: a judge that samples is a judge whose
#: verdict moves under you between runs.
JUDGE_TEMPERATURE = 0.0

#: The default judge. Deliberately a different provider from the models the
#: example workflows use: a model grading its own output family scores its own
#: habits, not the task.
DEFAULT_JUDGE_MODEL = "gemini/gemini-3.6-flash"


class Verdict(BaseModel):
    """A judge's structured answer. The reason is not decoration — it is what
    makes a failed case actionable without re-running it."""

    passed: bool = Field(description="Whether the answer satisfies the rubric.")
    score: float = Field(
        default=0.0, description="Confidence between 0 and 1, where 1 is fully met."
    )
    reason: str = Field(description="One sentence explaining the verdict.")


_JUDGE_PROMPT = """You are grading the output of an AI system against a rubric.
Be strict and literal: grade only what the rubric asks about, and ignore style
unless the rubric mentions it.

RUBRIC
{rubric}

THE INPUT THE SYSTEM WAS GIVEN
{inputs}
{expected}
THE OUTPUT TO GRADE
{output}

Answer with `passed` (does it satisfy the rubric), `score` (0..1) and a
one-sentence `reason`. If the output is empty or an error, it fails."""


def rubric_sha(rubrics: list[str]) -> str:
    """A stable hash of every rubric used in a run.

    A rubric that changes makes historical scores incomparable, and so does a
    judge model that moves under its alias. Recording both is the only way to
    tell a regression in the workflow from a change in the grader.
    """
    digest = hashlib.sha256()
    for rubric in sorted(rubrics):
        digest.update(rubric.encode("utf-8"))
    return digest.hexdigest()[:12]


class _JudgeBase(Evaluator):
    """Shared client construction, so a judge model is built once per suite."""

    needs_model = True

    def __init__(self, model: Optional[str] = None):
        self.model = model or DEFAULT_JUDGE_MODEL
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from kavalai.llm_clients.base_client import LlmClientParameters
            from kavalai.workflow.clients import make_client

            self._client = make_client(
                self.model, LlmClientParameters(temperature=JUDGE_TEMPERATURE)
            )
        return self._client


@evaluator("llm_judge")
class LLMJudge(_JudgeBase):
    """Grades the output against a rubric written in plain words.

    For the things that genuinely are a matter of judgement — is the reply
    polite, does it explain the rule, does it correct the false premise. Do not
    reach for it where a deterministic evaluator would do: a judge costs money,
    can flake, and is a dependency that moves under you.
    """

    def __init__(
        self, rubric: str, model: Optional[str] = None, threshold: float = 0.5
    ):
        super().__init__(model)
        self.rubric = rubric
        self.threshold = threshold

    async def score(self, case: Case, record: RunRecord) -> Score:
        if not record.ok:
            return Score.boolean(self.name, False, reason=f"run failed: {record.error}")

        expected = ""
        if case.expected is not None:
            expected = (
                f"\nTHE EXPECTED ANSWER (ground truth)\n"
                f"{json.dumps(to_plain(case.expected), ensure_ascii=False)}\n"
            )
        # A multi-turn run is graded on the whole exchange. Judging only the
        # last reply reads every detail established earlier in the thread as
        # invented — which is a judge that is wrong, not a workflow that is.
        if record.chat:
            output = "\n".join(
                f"{'USER' if turn.get('role') == 'user' else 'ASSISTANT'}: "
                f"{turn.get('content')}"
                for turn in record.chat
            )
        else:
            output = record.output_text() or json.dumps(to_plain(record.output))
        prompt = _JUDGE_PROMPT.format(
            rubric=self.rubric,
            inputs=json.dumps(to_plain(case.inputs), ensure_ascii=False),
            expected=expected,
            output=output,
        )
        try:
            verdict: Verdict = await self._get_client().prompt(prompt, Verdict)
        except Exception as exc:
            # A judge that is down is an error, not a failing case.
            logger.error(f"Judge '{self.model}' failed: {exc}")
            return Score(
                name=self.name, passed=None, reason=f"judge unavailable: {exc}"
            )
        return Score(
            name=self.name,
            value=verdict.score,
            passed=verdict.passed and verdict.score >= self.threshold,
            reason=verdict.reason,
            meta={"model": self.model, "rubric_sha": rubric_sha([self.rubric])},
        )


@evaluator("refuses")
class Refuses(_JudgeBase):
    """The answer declines rather than guessing.

    Gets its own evaluator because "I don't know" is a *correct* answer for the
    unanswerable slice, and a suite with no defined right behaviour there
    grades whatever the model happens to do.
    """

    RUBRIC = (
        "The answer says plainly that the information is not available, and "
        "does NOT state a figure, date or fact as if it were known. Saying "
        "what it does not know is a pass; guessing is a fail."
    )

    async def score(self, case: Case, record: RunRecord) -> Score:
        # The cheap check first: a refusal that declares itself in the output
        # schema needs no model call at all.
        output = record.output if isinstance(record.output, dict) else {}
        if "grounded" in output:
            grounded = bool(output.get("grounded"))
            return Score.boolean(
                self.name,
                not grounded,
                reason=None if not grounded else "answered as if the facts were known",
            )
        judge = LLMJudge(self.RUBRIC, model=self.model)
        judge.name = self.name
        return await judge.score(case, record)


@evaluator("semantic_similarity")
class SemanticSimilarity(Evaluator):
    """The answer means the same as the expected one, by embedding distance.

    Reuses the embedding providers the RAG services already ship, so this needs
    no new dependency and runs offline against a local model.
    """

    needs_model = True

    def __init__(
        self,
        threshold: float = 0.8,
        model: str = "openai/text-embedding-3-small",
        field: Optional[str] = None,
    ):
        self.threshold = threshold
        self.model = model
        self.field = field
        self._client: Any = None

    def _expected_text(self, case: Case) -> str:
        expected = case.expected
        if isinstance(expected, dict):
            key = self.field or "answer"
            return str(expected.get(key, expected))
        return "" if expected is None else str(expected)

    async def score(self, case: Case, record: RunRecord) -> Score:
        expected = self._expected_text(case)
        if not expected:
            return Score(name=self.name, reason="no expected answer to compare against")
        if self._client is None:
            from kavalai.llm_clients.embeddings import make_embedding_client

            self._client = make_embedding_client(self.model)

        vectors, _stats = await self._client.compute_embeddings(
            [expected, record.output_text()]
        )
        similarity = _cosine(vectors[0], vectors[1])
        ok = similarity >= self.threshold
        return Score(
            name=self.name,
            value=similarity,
            passed=ok,
            reason=None
            if ok
            else f"similarity {similarity:.2f} < {self.threshold:.2f}",
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
