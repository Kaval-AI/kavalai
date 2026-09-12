"""``kavalai.testing``: the scripted and fake clients, and the real engine on them.

Everything here runs offline. The scripted client is checked against the base
client's own machinery — streaming, retries, restarts, the output cap, model
call statistics — and then through the real ``WorkflowEngine`` with a SQLite
RAG index embedded by the fake embedding client.
"""

import asyncio
import json
import math
import subprocess
import sys

import pytest
from loguru import logger
from pydantic import BaseModel, ValidationError

import kavalai.llm_clients.with_retry as retry_module
from kavalai import (
    Agent,
    RunContext,
    WorkflowEngine,
    WorkflowException,
    make_client,
    make_embedding_client,
    register_llm_provider,
)
from kavalai.llm_clients import registry
from kavalai.llm_clients.base_client import (
    ChatHistory,
    ChatMessage,
    LlmClientParameters,
    ModelStatsReceiver,
    OutputTruncatedError,
)
from kavalai.normalizer import get_default_normalizer
from kavalai.rag import SqliteRagService
from kavalai.testing import (
    FakeEmbeddingClient,
    FakeProviders,
    Interrupted,
    ScriptedCall,
    ScriptedLlmClient,
    fake_providers,
)


class Answer(BaseModel):
    answer: str
    confidence: float


class Collector(ModelStatsReceiver):
    def __init__(self):
        self.stats = []

    def receive_model_stats(self, stats):
        self.stats.append(stats)


def tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def transient_error():
    openai = pytest.importorskip("openai")
    return openai.APIConnectionError(request=None)


@pytest.fixture
def instant_retries(monkeypatch):
    """Skip the retry backoff while still yielding to the event loop."""
    real_sleep = asyncio.sleep

    async def no_wait(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr(retry_module.asyncio, "sleep", no_wait)


@pytest.fixture
def warnings():
    """Every loguru warning logged during the test."""
    messages = []
    sink = logger.add(messages.append, level="WARNING")
    yield messages
    logger.remove(sink)


async def collect(streamer):
    return [chunk async for chunk in streamer]


class TestScriptedReplies:
    async def test_replies_are_consumed_in_order(self):
        client = ScriptedLlmClient(["one", "two"], model_stats_receiver=Collector())

        assert await client.prompt("first") == "one"
        assert client.remaining == 1
        assert await client.prompt("second") == "two"
        assert client.remaining == 0

    async def test_a_dict_or_a_model_validates_into_the_response_model(self):
        client = ScriptedLlmClient(
            [
                {"answer": "dict", "confidence": 0.5},
                Answer(answer="model", confidence=1.0),
            ],
            model_stats_receiver=Collector(),
        )

        first = await client.prompt("q", response_model=Answer)
        second = await client.prompt("q", response_model=Answer)

        assert first == Answer(answer="dict", confidence=0.5)
        assert second == Answer(answer="model", confidence=1.0)

    async def test_a_reply_that_does_not_fit_fails_as_a_model_reply_would(self):
        client = ScriptedLlmClient([{"answer": "x"}], model_stats_receiver=Collector())

        with pytest.raises(ValidationError, match="confidence"):
            await client.prompt("q", response_model=Answer)

    async def test_a_json_value_other_than_a_dict_is_serialised(self):
        client = ScriptedLlmClient([[1, 2]], model_stats_receiver=Collector())

        assert await client.prompt("q") == "[1, 2]"

    async def test_an_exhausted_script_fails_the_call(self):
        collector = Collector()
        client = ScriptedLlmClient([], model_stats_receiver=collector)

        with pytest.raises(RuntimeError, match="no reply left for call 1"):
            await client.prompt("q")
        assert collector.stats[0].response_code is None

    @pytest.mark.parametrize(
        "single", ["text", b"bytes", {"a": 1}, Answer(answer="a", confidence=1)]
    )
    def test_a_single_reply_is_refused(self, single):
        with pytest.raises(TypeError, match="wrap a single reply in a list"):
            ScriptedLlmClient(single)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ScriptedLlmClient(["a"], chunk_size=0)

    async def test_a_function_answers_every_call(self):
        seen = []

        def respond(messages, response_model):
            seen.append((messages, response_model))
            return {"answer": messages[0].content, "confidence": 1.0}

        client = ScriptedLlmClient(respond, model_stats_receiver=Collector())

        result = await client.prompt("echo", response_model=Answer)

        assert result.answer == "echo"
        assert seen[0][1] is Answer
        assert client.remaining is None

    async def test_the_function_may_be_a_coroutine_function(self):
        async def respond(messages, response_model):
            return "async"

        client = ScriptedLlmClient(respond, model_stats_receiver=Collector())

        assert await client.prompt("q") == "async"


class TestStreaming:
    async def test_partials_accumulate_in_chunk_size_steps(self):
        client = ScriptedLlmClient(
            ["abcdefgh"], chunk_size=3, model_stats_receiver=Collector()
        )

        chunks = await collect(await client.stream_prompt("q"))

        assert [c.value for c in chunks if c.type == "partial"] == [
            "abc",
            "abcdef",
            "abcdefgh",
        ]
        assert chunks[-1].type == "complete"
        assert chunks[-1].value == "abcdefgh"

    async def test_delta_mode_sends_only_the_new_text(self):
        client = ScriptedLlmClient(
            ["abcdefgh"], chunk_size=3, model_stats_receiver=Collector()
        )
        history = ChatHistory(messages=[ChatMessage(role="user", content="q")])

        streamer = await client.stream_chat_completions(
            chat_history=history, stream_delta=True
        )
        chunks = await collect(streamer)

        assert [c.value for c in chunks if c.type == "partial"] == ["abc", "def", "gh"]
        assert chunks[-1].value is None

    async def test_structured_partials_are_parsed_from_incomplete_json(self):
        client = ScriptedLlmClient(
            [{"answer": "a long streamed answer", "confidence": 0.25}],
            chunk_size=5,
            model_stats_receiver=Collector(),
        )

        streamer = await client.stream_prompt("q", response_model=Answer)
        partials = [c.value for c in await collect(streamer) if c.type == "partial"]

        assert len(partials) > 5
        assert all(isinstance(json.loads(value), dict) for value in partials)
        assert json.loads(partials[-1]) == {
            "answer": "a long streamed answer",
            "confidence": 0.25,
        }

    async def test_an_empty_reply_streams_no_partial(self):
        client = ScriptedLlmClient([""], model_stats_receiver=Collector())

        chunks = await collect(await client.stream_prompt("q"))

        assert [c.type for c in chunks] == ["complete"]


class TestRecording:
    async def test_calls_record_the_request_and_the_reply(self):
        parameters = LlmClientParameters(temperature=0.2, max_output_tokens=100)
        client = ScriptedLlmClient(
            [{"answer": "a", "confidence": 1.0}],
            llm_client_parameters=parameters,
            model_stats_receiver=Collector(),
        )

        await client.prompt("What is the answer?", response_model=Answer)

        (call,) = client.calls
        assert isinstance(call, ScriptedCall)
        assert call.model == "fake/scripted"
        assert call.messages == [
            ChatMessage(role="system", content="What is the answer?")
        ]
        assert call.response_model is Answer
        assert call.parameters is parameters
        assert call.reply == {"answer": "a", "confidence": 1.0}
        assert call.prompt == "What is the answer?"

    async def test_every_call_reports_a_realistic_model_call_stat(self):
        collector = Collector()
        client = ScriptedLlmClient(
            [{"answer": "a", "confidence": 1.0}],
            llm_client_parameters=LlmClientParameters(temperature=0.2),
            model_stats_receiver=collector,
        )

        await client.prompt("What is the answer?", response_model=Answer)

        (stat,) = collector.stats
        reply = '{"answer": "a", "confidence": 1.0}'
        assert stat.call_type == "llm"
        assert stat.model == "fake/scripted"
        assert stat.response_code == 200
        assert stat.response_data == reply
        assert stat.prompt_tokens == tokens("What is the answer?")
        assert stat.completion_tokens == tokens(reply)
        assert stat.total_tokens == stat.prompt_tokens + stat.completion_tokens
        assert stat.duration_seconds >= 0
        request = json.loads(stat.request_data)
        assert request["model"] == "scripted"
        assert request["response_model"] == "Answer"
        assert request["parameters"]["temperature"] == 0.2
        assert request["messages"] == [
            {"role": "system", "content": "What is the answer?"}
        ]


class TestFailures:
    @pytest.mark.parametrize("error", [ValueError("boom"), ValueError])
    async def test_an_exception_reply_fails_the_call_and_is_recorded(self, error):
        collector = Collector()
        client = ScriptedLlmClient([error], model_stats_receiver=collector)

        with pytest.raises(RuntimeError):
            await client.prompt("q")

        (stat,) = collector.stats
        assert stat.response_code is None
        assert stat.completion_tokens is None
        assert len(client.calls) == 1

    async def test_a_transient_error_is_retried_after_a_restart(self, instant_retries):
        collector = Collector()
        client = ScriptedLlmClient(
            [
                Interrupted('{"answer": "disc', transient_error()),
                {"answer": "kept", "confidence": 1.0},
            ],
            chunk_size=4,
            model_stats_receiver=collector,
        )

        chunks = await collect(await client.stream_prompt("q", response_model=Answer))

        types = [c.type for c in chunks]
        restart = types.index("restart")
        assert "partial" in types[:restart]
        assert types[-1] == "complete"
        assert json.loads(chunks[-1].value) == {"answer": "kept", "confidence": 1.0}
        assert len(client.calls) == 2
        assert [s.response_code for s in collector.stats] == [None, 200]

    async def test_an_interrupted_reply_with_a_permanent_error_fails(self):
        client = ScriptedLlmClient(
            [Interrupted("half", KeyError("gone"))], model_stats_receiver=Collector()
        )

        streamer = await client.stream_prompt("q")
        seen = []
        with pytest.raises(RuntimeError, match="gone"):
            async for chunk in streamer:
                seen.append(chunk.value)

        assert seen == ["half"]

    async def test_a_reply_over_the_output_cap_is_truncated(self):
        collector = Collector()
        client = ScriptedLlmClient(
            ["abcdefghijkl"],
            chunk_size=4,
            llm_client_parameters=LlmClientParameters(max_output_tokens=2),
            model_stats_receiver=collector,
        )

        streamer = await client.stream_prompt("q")
        seen = []
        with pytest.raises(RuntimeError, match="cut off at the output cap of 2"):
            async for chunk in streamer:
                seen.append(chunk.value)

        assert seen == ["abcd", "abcdefgh"]
        (stat,) = collector.stats
        assert stat.completion_tokens == 2
        assert stat.prompt_tokens == tokens("q")
        assert "max_output_tokens" in stat.response_data

    async def test_a_reply_within_the_output_cap_completes(self):
        client = ScriptedLlmClient(
            ["abcdefgh"],
            llm_client_parameters=LlmClientParameters(max_output_tokens=2),
            model_stats_receiver=Collector(),
        )

        assert await client.prompt("q") == "abcdefgh"

    async def test_a_scripted_truncation_error_is_raised_as_is(self):
        error = OutputTruncatedError(model="fake/scripted", reason="length")
        client = ScriptedLlmClient([error], model_stats_receiver=Collector())

        with pytest.raises(RuntimeError, match="stop reason 'length'"):
            await client.prompt("q")


class TestFactory:
    def test_calling_the_client_binds_a_provider_model(self):
        root = ScriptedLlmClient(["a"], model_stats_receiver=Collector())
        parameters = LlmClientParameters(temperature=0.0)
        receiver = Collector()

        bound = root("openai/gpt-5", parameters, receiver)

        assert isinstance(bound, ScriptedLlmClient)
        assert bound is not root
        assert bound.stat_model_name() == "openai/gpt-5"
        assert bound.parameters is parameters
        assert bound.model_stats_receiver is receiver
        assert bound.calls is root.calls

    def test_a_name_without_a_provider_keeps_the_clients_provider(self):
        root = ScriptedLlmClient(["a"], provider="scripted")

        assert root("tiny").stat_model_name() == "scripted/tiny"

    def test_bind_inherits_what_it_is_not_given(self):
        parameters = LlmClientParameters(top_p=0.5)
        receiver = Collector()
        root = ScriptedLlmClient(
            ["a"], llm_client_parameters=parameters, model_stats_receiver=receiver
        )

        bound = root.bind("other")

        assert bound.parameters is parameters
        assert bound.model_stats_receiver is receiver
        assert bound.stat_model_name() == "fake/other"

    async def test_bound_clients_share_the_script(self):
        root = ScriptedLlmClient(["first", "second"], model_stats_receiver=Collector())

        assert await root("fake/a").prompt("q") == "first"
        assert await root("fake/b").prompt("q") == "second"
        assert [call.model for call in root.calls] == ["fake/a", "fake/b"]

    def test_the_class_cannot_be_built_from_a_model_name(self):
        with pytest.raises(TypeError, match="fake_providers"):
            ScriptedLlmClient.from_model("tiny")

    def test_registering_the_class_fails_with_the_reason(self):
        register_llm_provider("scripted-class", ScriptedLlmClient)
        try:
            with pytest.raises(registry.RegistryError, match="fake_providers"):
                make_client("scripted-class/tiny")
        finally:
            registry.llm_providers.unregister("scripted-class")


class TestFakeEmbeddingClient:
    def test_vectors_are_deterministic_unit_length_and_sized(self):
        client = FakeEmbeddingClient(dimension=16)

        vector = client.vector("The pond is deep")

        assert len(vector) == 16
        assert math.isclose(cosine(vector, vector), 1.0)
        assert vector == FakeEmbeddingClient(dimension=16).vector("the POND is deep!")

    @pytest.mark.parametrize("text", ["", "?!", "   "])
    def test_text_without_words_still_has_a_direction(self, text):
        vector = FakeEmbeddingClient().vector(text)

        assert math.isclose(cosine(vector, vector), 1.0)

    def test_a_shared_word_brings_texts_together(self):
        client = FakeEmbeddingClient(dimension=64)
        query = client.vector("pond depth")

        near = client.vector("the pond is deep")
        far = client.vector("bakery opening hours")

        assert cosine(query, near) > cosine(query, far)

    def test_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="dimension"):
            FakeEmbeddingClient(dimension=0)

    async def test_compute_embeddings_reports_a_model_call_record(self):
        client = FakeEmbeddingClient("tiny")

        embeddings, stats = await client.compute_embeddings(["abcdef", "ab"])

        assert embeddings == [client.vector("abcdef"), client.vector("ab")]
        assert stats.call_type == "embedding"
        assert stats.model == "fake/tiny"
        assert stats.batch_size == 2
        assert stats.total_tokens == 3
        assert stats.response_code == 200
        assert client.calls == [["abcdef", "ab"]]

    async def test_normalize_applies_the_given_normalizer(self):
        class Doubling:
            def transform(self, embeddings):
                return [[2 * value for value in vector] for vector in embeddings]

        client = FakeEmbeddingClient()

        normalised, _ = await client.compute_embeddings(
            ["pond"], normalize=True, normalizer=Doubling()
        )
        untouched, _ = await client.compute_embeddings(
            ["pond"], normalize=False, normalizer=Doubling()
        )

        assert normalised == [[2 * value for value in client.vector("pond")]]
        assert untouched == [client.vector("pond")]

    async def test_normalize_without_a_normalizer_uses_the_default(self):
        client = FakeEmbeddingClient()

        normalised, _ = await client.compute_embeddings(["pond"], normalize=True)

        assert normalised == get_default_normalizer().transform([client.vector("pond")])

    def test_bind_shares_the_calls(self):
        root = FakeEmbeddingClient(dimension=4)

        bound = root.bind("tiny", provider="mine")

        assert (bound.model, bound.provider, bound.dimension) == ("tiny", "mine", 4)
        assert bound.calls is root.calls
        assert root.bind("other").provider == "fake"


class TestFakeProviders:
    async def test_the_name_resolves_to_the_fakes_inside_the_block(self):
        script = ScriptedLlmClient(["hello"])
        receiver = Collector()

        with fake_providers(llm=script) as fakes:
            client = make_client("fake/tiny", stats_receiver=receiver)
            embedder = make_embedding_client("fake/tiny")
            reply = await client.prompt("q")

        assert isinstance(fakes, FakeProviders)
        assert fakes.llm is script
        assert reply == "hello"
        assert client.stat_model_name() == "fake/tiny"
        assert receiver.stats[0].model == "fake/tiny"
        assert isinstance(embedder, FakeEmbeddingClient)
        assert embedder.calls is fakes.embedding.calls
        assert embedder.model == "tiny"

    def test_without_an_llm_only_the_embedding_provider_is_registered(self):
        with fake_providers() as fakes:
            assert fakes.llm is None
            assert isinstance(fakes.embedding, FakeEmbeddingClient)
            assert "fake" in registry.registered_embedding_providers()
            assert "fake" not in registry.registered_llm_providers()

    def test_a_name_registered_nowhere_is_removed_on_exit(self):
        with fake_providers(llm=ScriptedLlmClient([])):
            pass

        assert "fake" not in registry.registered_llm_providers()
        assert "fake" not in registry.registered_embedding_providers()

    def test_a_previous_registration_is_restored_exactly(self, warnings):
        dotted = "kavalai.testing.FakeEmbeddingClient"
        registry.register_embedding_provider("mycorp", dotted, dimension=4)
        registry.embedding_providers._resolve_target("mycorp")
        store = registry.embedding_providers
        before = (
            store._targets["mycorp"],
            store._defaults["mycorp"],
            store._resolved["mycorp"],
        )
        try:
            with fake_providers(name="mycorp") as fakes:
                assert make_embedding_client("mycorp/x").calls is fakes.embedding.calls

            after = (
                store._targets["mycorp"],
                store._defaults["mycorp"],
                store._resolved["mycorp"],
            )
            assert after == before
            assert make_embedding_client("mycorp/x").dimension == 4
            assert warnings == []
        finally:
            store.unregister("mycorp")

    async def test_a_builtin_provider_can_be_faked(self, warnings):
        store = registry.llm_providers
        before = store._targets["openai"]

        with fake_providers(llm=ScriptedLlmClient(["scripted"]), name="openai"):
            reply = await make_client("openai/gpt-5").prompt("q")

        assert reply == "scripted"
        assert store._targets["openai"] == before
        assert "openai" in store._builtins
        assert warnings == []

    def test_registrations_are_restored_when_the_block_raises(self):
        with pytest.raises(KeyError):
            with fake_providers(llm=ScriptedLlmClient([])):
                raise KeyError("inside")

        assert "fake" not in registry.registered_llm_providers()

    def test_an_invalid_name_is_refused_and_changes_nothing(self):
        llm_before = registry.registered_llm_providers()
        embedding_before = registry.registered_embedding_providers()

        with pytest.raises(registry.RegistryError, match="Invalid"):
            with fake_providers(llm=ScriptedLlmClient([]), name="bad/"):
                pass  # pragma: no cover - never entered

        assert registry.registered_llm_providers() == llm_before
        assert registry.registered_embedding_providers() == embedding_before

    def test_the_module_imports_no_provider_sdk_and_no_pytest(self):
        code = (
            "import sys, kavalai.testing\n"
            "banned = ('pytest', '_pytest', 'openai', 'anthropic', 'ollama', "
            "'fastembed', 'google.genai')\n"
            "print(sorted(m for m in sys.modules "
            "if any(m == b or m.startswith(b + '.') for b in banned)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "[]"


WORKFLOW = """
name: Village desk
llm_model: fake/tiny
rag_collection: village
data_types:
  input:
    type: object
    properties:
      question: {type: string}
  output:
    type: object
    properties:
      answer: {type: string}
      confidence: {type: number}
nodes:
  - {name: begin, type: start, next: retrieve}
  - name: retrieve
    type: rag_query
    query: "{{ context.input.question }}"
    top_k: 1
    store: content
    output: facts
    next: answer
  - name: answer
    type: llm
    prompt: "Answer from the facts: {{ context.facts }}"
    inputs:
      question: {type: context, value: input.question}
    output: output
    next: finish
    stream_output: true
    stream_delta: STREAM_DELTA
  - {name: finish, type: end, output: output}
"""

FACTS = [
    "Green Village has 104 residents.",
    "The village pond, Lake Miller, is 1.2 metres deep.",
    "The bakery opens at seven on weekdays.",
]

REPLY = {"answer": "Lake Miller is 1.2 metres deep.", "confidence": 0.9}


async def indexed(tmp_path) -> SqliteRagService:
    rag = SqliteRagService(str(tmp_path / "facts.db"), model="fake/tiny")
    await rag.index_batch(FACTS, [{} for _ in FACTS], collection_name="village")
    return rag


def workflow(stream_delta: bool = False) -> str:
    return WORKFLOW.replace("STREAM_DELTA", str(stream_delta).lower())


class TestWorkflowEngine:
    async def test_a_workflow_runs_end_to_end_on_registered_fakes(self, tmp_path):
        script = ScriptedLlmClient([REPLY])

        with fake_providers(llm=script) as fakes:
            rag = await indexed(tmp_path)
            engine = WorkflowEngine.from_yaml(workflow(), rag_services=rag)
            events = [
                event
                async for event in engine.run_stream(
                    {"question": "How deep is the pond?"}
                )
            ]
            rag.close()

        completed = events[-1]
        assert completed.type == "workflow_completed"
        assert completed.output_data == REPLY
        partials = [e for e in events if e.type == "partial" and e.name == "answer"]
        assert len(partials) > 1
        assert "Lake Miller" in script.calls[0].prompt
        assert "bakery" not in script.calls[0].prompt
        assert script.calls[0].model == "fake/tiny"
        assert fakes.embedding.calls[-1] == ["How deep is the pond?"]
        assert completed.token_usage["completion_tokens"] == tokens(json.dumps(REPLY))

    async def test_the_client_is_the_engines_client_factory(self, tmp_path):
        script = ScriptedLlmClient([REPLY, REPLY], chunk_size=5)
        rag = await indexed_with_injected_client(tmp_path)
        engine = WorkflowEngine.from_yaml(
            workflow(stream_delta=True), rag_services=rag, client_factory=script
        )

        deltas = []
        async for event in engine.run_stream({"question": "How deep is the pond?"}):
            if event.type == "partial" and event.name == "answer":
                deltas.append(event.value)
        state = await engine.run({"question": "How deep is the pond?"})
        rag.close()

        assert json.loads("".join(deltas)) == REPLY
        assert all(len(delta) <= 5 for delta in deltas)
        assert state.output_data == REPLY
        assert script.remaining == 0
        assert state.trace == ["begin", "retrieve", "answer", "finish"]

    async def test_a_restart_reaches_the_workflow_stream(
        self, tmp_path, instant_retries
    ):
        script = ScriptedLlmClient(
            [Interrupted('{"answer": "lost', transient_error()), REPLY]
        )
        rag = await indexed_with_injected_client(tmp_path)
        engine = WorkflowEngine.from_yaml(
            workflow(), rag_services=rag, client_factory=script
        )

        events = [e async for e in engine.run_stream({"question": "pond?"})]
        rag.close()

        assert "restart" in [e.type for e in events if e.name == "answer"]
        assert events[-1].output_data == REPLY
        assert len(script.calls) == 2

    async def test_a_reply_that_does_not_fit_fails_the_run(self, tmp_path):
        script = ScriptedLlmClient([{"answer": "no confidence"}])
        rag = await indexed_with_injected_client(tmp_path)
        engine = WorkflowEngine.from_yaml(
            workflow(), rag_services=rag, client_factory=script
        )

        with pytest.raises(WorkflowException) as raised:
            await engine.run({"question": "pond?"})
        rag.close()

        assert isinstance(raised.value.__cause__, ValidationError)

    async def test_an_agent_answers_from_the_script(self):
        script = ScriptedLlmClient(
            [
                {
                    "instructions": "Answer directly.",
                    "tool_calls": [],
                    "output": {"answer": "agent", "confidence": 0.5},
                }
            ],
            model_stats_receiver=Collector(),
        )
        agent = Agent(llm_client=script, kernel=None, run_context=RunContext(data={}))

        result = await agent.prompt("q", response_model=Answer)

        assert result == Answer(answer="agent", confidence=0.5)


async def indexed_with_injected_client(tmp_path) -> SqliteRagService:
    """An index whose embedding client is assigned rather than registered."""
    rag = SqliteRagService(str(tmp_path / "facts.db"), model="fake/tiny")
    rag.embedding_client = FakeEmbeddingClient("tiny")
    await rag.index_batch(FACTS, [{} for _ in FACTS], collection_name="village")
    return rag
