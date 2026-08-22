"""Simulated users: a model that plays a person talking to your agent.

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

import time
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from kavalai.eval.models import Case, EvaluatorSpecs
from kavalai.eval.targets import RunRecord, Target
from kavalai.eval.trajectory import Trajectory

#: Personas are deliberately *not* run at temperature 0. A simulated user who
#: says exactly the same thing every time tests one path repeatedly; the point
#: of a persona is coverage, and coverage needs variation.
PERSONA_TEMPERATURE = 0.8

#: A different provider from the workflows under test, on purpose. A model
#: playing the user and grading the conversation from the same family as the
#: model being tested is correlated error, not independent measurement.
DEFAULT_PERSONA_MODEL = "gemini/gemini-3.6-flash"


class Traits(BaseModel):
    """How the simulated user behaves. Plain words, because they go in a prompt."""

    temperament: str = "neutral"  # patient | neutral | impatient | hostile
    verbosity: str = "normal"  # terse | normal | rambling
    expertise: str = "medium"  # low | medium | high
    language: str = "en"


class Persona(BaseModel):
    """A simulated user: a goal, a personality, and what they know.

    A file, reviewed in git exactly like a workflow. Nothing edits it from a
    UI, which is what keeps a persona diffable.
    """

    name: str
    goal: str
    traits: Traits = Field(default_factory=Traits)
    #: What this person knows and does not know. The most important field: an
    #: under-specified persona drifts into being a helpful test script rather
    #: than a user.
    knowledge: str = ""
    #: The first message. Written by hand so every run starts identically.
    opening: Optional[str] = None
    #: When to stop, in plain words, judged after each assistant turn.
    stop_when: str = (
        "the assistant has answered, or has asked twice for the same detail"
    )
    max_turns: int = 6
    model: Optional[str] = None
    #: ``chat`` sends ``{user_message: ...}``. ``email`` wraps each turn in an
    #: ``{email: {sender, subject, body}}`` envelope.
    channel: str = "chat"
    sender: Optional[str] = None
    subject: Optional[str] = None
    slice: str = "persona"
    evaluators: EvaluatorSpecs = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Persona":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
        return cls(**data)

    def as_case(self) -> Case:
        """The persona seen as a case, so one runner grades both kinds."""
        return Case(
            name=self.name,
            inputs=self.wrap(self.opening or ""),
            slice=self.slice,
            metadata={"persona": True, "goal": self.goal},
            evaluators=list(self.evaluators),
        )

    def wrap(self, message: str, history: Optional[list[dict]] = None) -> dict:
        """Shape one turn into the workflow's input.

        Two shapes and no template language: a chatbot takes a message, an
        email assistant takes an envelope. Anything else is a ``CallableTarget``
        away.

        For the email channel the previous reply is quoted underneath, the way
        a real mail client does it. That is not decoration: an email thread
        carries its own history, so a multi-turn conversation works against a
        stateless workflow with no session store at all — and it exercises the
        "read the new message, ignore the quote" behaviour that email parsing
        actually has to get right.
        """
        if self.channel != "email":
            return {"user_message": message}

        body = message
        replies = [t for t in (history or []) if t.get("role") == "assistant"]
        if replies:
            quoted = "\n".join(
                f"> {line}" for line in replies[-1]["content"].splitlines()
            )
            body = f"{message}\n\n> On an earlier message, the bakery wrote:\n{quoted}"
        subject = self.subject or self.goal[:60]
        return {
            "email": {
                "sender": self.sender or f"{self.name}@example.test",
                "subject": f"Re: {subject}" if replies else subject,
                "body": body,
            }
        }


class PersonaTurn(BaseModel):
    """What the persona model produces each time it is asked to speak."""

    message: str = Field(description="What the user says next, in their own voice.")
    goal_achieved: bool = Field(
        default=False, description="Has the assistant met the user's goal yet?"
    )
    done: bool = Field(
        default=False, description="Should the conversation stop after this turn?"
    )


_PERSONA_PROMPT = """You are role-playing a real person contacting a company.
Stay in character. You are NOT an assistant and you must never help, explain or
apologise on the company's behalf.

WHO YOU ARE
{knowledge}

YOUR GOAL
{goal}

HOW YOU COME ACROSS
- temperament: {temperament}
- how much you write: {verbosity}
- how much you know about this domain: {expertise}
- you write in: {language}

STOP WHEN
{stop_when}

THE CONVERSATION SO FAR
{transcript}

Write your next message. Rules you must keep:
- Reveal information only when you are asked for it directly. Do not
  volunteer everything at once.
- Never invent facts about yourself that contradict WHO YOU ARE. If you do not
  know something, say so the way this person would.
- Set `goal_achieved` only if the assistant has actually met your goal.
- Set `done` when STOP WHEN is satisfied, or when continuing would be
  pointless."""


class Conversation:
    """One multi-turn exchange between a persona and a target.

    The turns are ordinary workflow runs. When the target has a database the
    session carries over, so the workflow sees its own history; without one
    each turn is independent, which is worth knowing before reading the
    results — the runner says so.
    """

    def __init__(
        self, target: Target, persona: Persona, external_id: Optional[str] = None
    ):
        self.target = target
        self.persona = persona
        self.external_id = external_id
        self.turns: list[dict] = []
        self.records: list[RunRecord] = []
        self.session_id: Optional[str] = None
        #: The persona's own verdict, given while it was still in character.
        self.goal_achieved = False
        self.elapsed = 0.0

    async def send(self, message: str) -> str:
        """Send one user turn, return what the assistant said."""
        case = Case(
            name=self.persona.name, inputs=self.persona.wrap(message, self.turns)
        )
        record = await self.target.run(case, external_id=self.external_id)
        self.session_id = record.session_id or self.session_id
        reply = record.output_text()
        self.turns.append({"role": "user", "content": message})
        self.turns.append({"role": "assistant", "content": reply})
        self.records.append(record)
        return reply

    def transcript(self) -> str:
        if not self.turns:
            return "(nothing yet — you speak first)"
        return "\n".join(
            f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}"
            for t in self.turns
        )

    def as_record(self) -> RunRecord:
        """Fold the conversation into one record the evaluators can read.

        The last turn's output is the outcome; the trajectory and the token
        totals cover every turn, because "did it eventually store the order"
        is a question about the whole conversation, not the final message.
        """
        last = self.records[-1] if self.records else RunRecord()
        merged = Trajectory(
            records=[r for rec in self.records for r in rec.trajectory.records]
        )
        return RunRecord(
            output=last.output,
            status=last.status,
            error=last.error,
            duration_seconds=sum(r.duration_seconds for r in self.records),
            trajectory=merged,
            model_calls=[c for r in self.records for c in r.model_calls],
            chat=list(self.turns),
            external_id=self.external_id,
            session_id=self.session_id,
            sandbox=last.sandbox,
            meta={
                "persona": self.persona.name,
                "goal": self.persona.goal,
                "goal_achieved": self.goal_achieved,
                "user_turns": sum(1 for t in self.turns if t["role"] == "user"),
            },
        )


class PersonaRunner:
    """Alternates turns between a persona model and a target.

    No new execution machinery: a persona is one model call per turn, and the
    system under test is driven exactly as a case is.
    """

    def __init__(
        self,
        persona: Persona,
        target: Target,
        *,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        on_turn: Optional[Any] = None,
    ):
        self.persona = persona
        self.target = target
        self.model = model or persona.model or DEFAULT_PERSONA_MODEL
        self.max_turns = max_turns or persona.max_turns
        self.on_turn = on_turn
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from kavalai.llm_clients.base_client import LlmClientParameters
            from kavalai.workflow.clients import make_client

            self._client = make_client(
                self.model, LlmClientParameters(temperature=PERSONA_TEMPERATURE)
            )
        return self._client

    async def _next_turn(self, conversation: Conversation) -> PersonaTurn:
        persona = self.persona
        prompt = _PERSONA_PROMPT.format(
            knowledge=persona.knowledge or f"Someone who wants: {persona.goal}",
            goal=persona.goal,
            temperament=persona.traits.temperament,
            verbosity=persona.traits.verbosity,
            expertise=persona.traits.expertise,
            language=persona.traits.language,
            stop_when=persona.stop_when,
            transcript=conversation.transcript(),
        )
        return await self._get_client().prompt(prompt, PersonaTurn)

    async def run(self, external_id: Optional[str] = None) -> Conversation:
        """Play the conversation out and return it."""
        conversation = Conversation(self.target, self.persona, external_id)
        goal_achieved = False
        start = time.perf_counter()

        for turn_index in range(self.max_turns):
            if turn_index == 0 and self.persona.opening:
                message, done = self.persona.opening, False
            else:
                try:
                    turn = await self._next_turn(conversation)
                except Exception as exc:
                    logger.error(
                        f"Persona '{self.persona.name}' could not speak: {exc}"
                    )
                    break
                message, done = turn.message, turn.done
                goal_achieved = goal_achieved or turn.goal_achieved
                if done and not message.strip():
                    break

            reply = await conversation.send(message)
            if self.on_turn:
                self.on_turn(message, reply)
            if turn_index > 0 and done:
                break

        conversation.goal_achieved = goal_achieved
        conversation.elapsed = time.perf_counter() - start
        return conversation
