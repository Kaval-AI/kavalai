"""Simulated users: the conversation loop, the two channels, and the stop rule."""

import pytest
import yaml

from kavalai.eval import Case, Persona, PersonaRunner, RunRecord
from kavalai.eval.persona import Conversation, PersonaTurn
from kavalai.eval.targets import Target


class EchoTarget(Target):
    """Replies with a canned script, so the loop is deterministic."""

    observes_trajectory = True

    def __init__(self, replies=None):
        self.replies = list(replies or ["how many would you like?"])
        self.seen: list[dict] = []

    async def setup(self):
        return None

    async def aclose(self):
        return None

    async def run(self, case: Case, external_id=None) -> RunRecord:
        self.seen.append(case.inputs)
        reply = self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]
        return RunRecord(output={"agent_response": reply}, session_id="sess-1")

    def describe(self):
        return {"kind": "echo"}


class ScriptedPersonaClient:
    """Stands in for the persona model. No provider, fully deterministic."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.prompts: list[str] = []

    async def prompt(self, text, response_model=None):
        self.prompts.append(text)
        return self.turns.pop(0)


def runner(persona, target, turns) -> PersonaRunner:
    instance = PersonaRunner(persona, target)
    instance._client = ScriptedPersonaClient(turns)
    return instance


def test_a_persona_loads_from_yaml(tmp_path):
    path = tmp_path / "vague_parent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "goal": "Order a cake",
                "traits": {"temperament": "patient", "verbosity": "rambling"},
                "opening": "hi!",
                "evaluators": ["goal_achieved"],
            }
        )
    )
    persona = Persona.from_yaml(path)
    assert persona.name == "vague_parent"
    assert persona.traits.temperament == "patient"
    assert persona.traits.expertise == "medium"
    assert [e.type for e in persona.evaluators] == ["goal_achieved"]


def test_a_chat_persona_wraps_a_turn_as_a_message():
    persona = Persona(name="p", goal="g")
    assert persona.wrap("hello") == {"user_message": "hello"}


def test_an_email_persona_wraps_a_turn_as_an_envelope():
    persona = Persona(
        name="p",
        goal="Order a cake",
        channel="email",
        sender="p@example.test",
        subject="cake",
    )
    assert persona.wrap("hello") == {
        "email": {"sender": "p@example.test", "subject": "cake", "body": "hello"}
    }


def test_an_email_thread_quotes_the_previous_reply():
    """An email thread carries its own history, so no session store is needed."""
    persona = Persona(name="p", goal="g", channel="email", subject="cake")
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "how many?\nlet us know"},
    ]
    email = persona.wrap("two please", history)["email"]

    assert email["subject"] == "Re: cake"
    assert email["body"].startswith("two please")
    assert "> how many?" in email["body"]
    assert "> let us know" in email["body"]


def test_an_email_persona_defaults_its_address_and_subject():
    email = Persona(name="p", goal="Order a very large cake").wrap("hi")
    assert (
        Persona(name="p", goal="g", channel="email").wrap("hi")["email"]["sender"]
        == "p@example.test"
    )
    assert email == {"user_message": "hi"}


async def test_the_opening_is_used_verbatim_for_the_first_turn():
    """So every run of a persona starts identically."""
    persona = Persona(name="p", goal="g", opening="pond depth?", max_turns=1)
    target = EchoTarget(["1.2 metres"])
    conversation = await runner(persona, target, []).run()

    assert target.seen[0] == {"user_message": "pond depth?"}
    assert conversation.turns == [
        {"role": "user", "content": "pond depth?"},
        {"role": "assistant", "content": "1.2 metres"},
    ]


async def test_the_persona_stops_when_it_says_it_is_done():
    persona = Persona(name="p", goal="g", opening="hi", max_turns=5)
    turns = [
        PersonaTurn(message="two please", goal_achieved=False, done=False),
        PersonaTurn(message="thanks", goal_achieved=True, done=True),
    ]
    conversation = await runner(persona, EchoTarget(), turns).run()

    assert [t["content"] for t in conversation.turns if t["role"] == "user"] == [
        "hi",
        "two please",
        "thanks",
    ]
    assert conversation.goal_achieved is True


async def test_max_turns_is_a_hard_cap():
    persona = Persona(name="p", goal="g", opening="hi", max_turns=3)
    turns = [PersonaTurn(message=f"m{i}") for i in range(5)]
    conversation = await runner(persona, EchoTarget(), turns).run()
    assert len([t for t in conversation.turns if t["role"] == "user"]) == 3


async def test_a_persona_model_failure_ends_the_conversation_rather_than_the_run():
    class Broken:
        async def prompt(self, text, response_model=None):
            raise RuntimeError("provider down")

    persona = Persona(name="p", goal="g", opening="hi", max_turns=3)
    instance = PersonaRunner(persona, EchoTarget())
    instance._client = Broken()
    conversation = await instance.run()

    assert len(conversation.records) == 1


async def test_the_folded_record_covers_the_whole_conversation():
    persona = Persona(name="p", goal="Order a cake", opening="hi", max_turns=2)
    turns = [PersonaTurn(message="two", goal_achieved=True, done=True)]
    conversation = await runner(persona, EchoTarget(["a", "b"]), turns).run()
    record = conversation.as_record()

    assert record.output_text() == "b"  # the outcome is the last turn
    assert len(record.chat) == 4  # but the transcript is all of it
    assert record.meta["goal_achieved"] is True
    assert record.meta["user_turns"] == 2
    assert record.session_id == "sess-1"


async def test_the_prompt_carries_the_traits_and_the_transcript():
    persona = Persona(
        name="p",
        goal="Find the pond depth",
        knowledge="Knows nothing.",
        opening="hi",
        max_turns=2,
    )
    instance = runner(
        persona, EchoTarget(["1.2 m"]), [PersonaTurn(message="ok", done=True)]
    )
    await instance.run()

    prompt = instance._client.prompts[0]
    assert "Find the pond depth" in prompt
    assert "Knows nothing." in prompt
    assert "ASSISTANT: 1.2 m" in prompt


def test_an_empty_transcript_tells_the_persona_to_speak_first():
    conversation = Conversation(EchoTarget(), Persona(name="p", goal="g"))
    assert "you speak first" in conversation.transcript()


async def test_a_persona_is_also_a_case():
    persona = Persona(name="p", goal="g", opening="hi", slice="persona")
    case = persona.as_case()
    assert case.name == "p" and case.slice == "persona"
    assert case.inputs == {"user_message": "hi"}
    assert case.metadata["persona"] is True
