import json
import os
from typing import Optional

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, ValidationError

from kavalai import make_client, pythontool
from kavalai.llm_clients.base_client import LlmClientParameters
from kavalai.testing import ScriptedLlmClient

from kavalai.agent import (
    Agent,
    StepStreamDemuxer,
    ToolCall,
    _FAILED,
    _ToolArgumentError,
    _assign_unique_call_ids,
    _parse_json_object,
    _read_refs,
    _resolve_arguments,
    _resolve_ref,
    _unique_call_id,
    _written_arguments,
    get_step_output_type,
)
from kavalai.run_context import RunContext
from kavalai.step_schema import ArgRef
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.base_client import BaseLlmClient, ChatMessage
from jinja2 import Template
from kavalai.llm_clients.streamer import StreamContent


class MockResponse(BaseModel):
    answer: str


class ScriptedClient(BaseLlmClient):
    """Streams pre-scripted step outputs through the real Streamer machinery.

    Each entry is a ``StepOutput`` instance (serialized to JSON) or a raw
    string, streamed in small chunks so the agent's demux is exercised.
    """

    def __init__(self, outputs=None, chunk_size=7):
        super().__init__()
        self.outputs = list(outputs or [])
        self.chunk_size = chunk_size
        self.await_count = 0
        self.seen: list[list[ChatMessage]] = []

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        self.await_count += 1
        self.seen.append(list(chat_history.messages))
        out = self.outputs.pop(0)
        text = out if isinstance(out, str) else out.model_dump_json()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        for i in range(0, len(text), self.chunk_size):
            await value_streamer.stream_partial(text[i : i + self.chunk_size])
        await value_streamer.stream_complete()


@pytest.fixture
def mock_kernel():
    kernel = MagicMock(spec=FunctionKernel)
    kernel.get_tool_descriptions = AsyncMock(return_value="Mock tool descriptions")
    kernel.get_tool_definitions = AsyncMock(return_value={})
    kernel.call_tool = AsyncMock(return_value="Tool result")
    return kernel


@pytest.fixture
def run_context():
    return RunContext(data={})


def make_agent(client, kernel, run_context=None):
    return Agent(
        llm_client=client,
        kernel=kernel,
        run_context=run_context or RunContext(data={}),
    )


@pytest.mark.asyncio
async def test_prompt_tool_call_then_output(mock_kernel, run_context):
    """A tool-calling step followed by a step producing the final output."""
    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="call the test tool",
        tool_calls=[
            ToolCall(name="python://test_tool", args={"val": 1}, call_id="c1")
        ],
    )
    step2 = StepOutput(
        instructions="return the final answer",
        output=MockResponse(answer="Final answer"),
    )
    client = ScriptedClient([step1, step2])
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt(
        "Do something", response_model=MockResponse, max_steps=5
    )

    assert isinstance(result, MockResponse)
    assert result.answer == "Final answer"
    # The tool was executed with the arguments the model wrote.
    mock_kernel.call_tool.assert_awaited_once_with(
        tool_uri="python://test_tool", arguments={"val": 1}
    )
    # Only two LLM calls were needed (stopped once output produced).
    assert client.await_count == 2


@pytest.mark.asyncio
async def test_prompt_plain_string_output(mock_kernel, run_context):
    """Without a response_model the agent returns a plain string output."""
    StepOutput = get_step_output_type(str)
    client = ScriptedClient(
        [StepOutput(instructions="greet the user", output="hello there")]
    )
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt("Greet the user")

    assert result == "hello there"
    mock_kernel.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_respects_max_steps(mock_kernel, run_context):
    """The loop stops after max_steps even if no final output is produced."""
    StepOutput = get_step_output_type(MockResponse)
    # Always request a tool call, never produce an output.
    looping_step = StepOutput(
        instructions="keep looping",
        tool_calls=[ToolCall(name="python://loop", call_id="c1")],
    )
    client = ScriptedClient([looping_step] * 10)
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt(
        "Never finish", response_model=MockResponse, max_steps=3
    )

    assert result is None
    assert client.await_count == 3
    # The last step has no next step to read a result in: its tool calls are not run.
    assert mock_kernel.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_prompt_empty_step_output_stops(mock_kernel, run_context):
    """An empty completion ends the loop with no output."""
    client = ScriptedClient([""])
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt("anything", response_model=MockResponse)

    assert result is None
    assert client.await_count == 1


@pytest.mark.asyncio
async def test_tool_results_are_isolated_across_invocations(mock_kernel, run_context):
    """One invocation's tool results cannot be referenced from the next."""
    StepOutput = get_step_output_type(MockResponse)

    # First invocation: produce a tool result under call_id "c1", then finish.
    client = ScriptedClient(
        [
            StepOutput(
                instructions="run the tool",
                tool_calls=[ToolCall(name="python://t", call_id="c1")],
            ),
            StepOutput(instructions="finish", output=MockResponse(answer="first")),
            # Second invocation references "c1" from the previous run.
            StepOutput(
                instructions="run the tool",
                tool_calls=[
                    ToolCall(
                        name="python://t",
                        refs=[{"arg": "x", "source": "step", "key": "c1"}],
                        call_id="c2",
                    )
                ],
            ),
            StepOutput(instructions="finish", output=MockResponse(answer="second")),
        ]
    )
    agent = make_agent(client, mock_kernel, run_context)

    await agent.prompt("first", response_model=MockResponse)
    mock_kernel.call_tool.reset_mock()
    await agent.prompt("second", response_model=MockResponse)

    # The prior invocation's call_id no longer exists: the call is not run,
    # and the model is told which results it can refer to.
    mock_kernel.call_tool.assert_not_awaited()
    assert "`x` refers to the call_id 'c1', which does not exist" in client.seen[-1][-1].content
    assert "Available: none" in client.seen[-1][-1].content


@pytest.mark.asyncio
async def test_tool_error_is_captured(mock_kernel, run_context):
    """A failing tool call is captured as a result string instead of raising."""
    agent = make_agent(ScriptedClient(), mock_kernel, run_context)
    mock_kernel.call_tool.side_effect = RuntimeError("boom")

    tool_call = ToolCall(name="python://broken", call_id="c1")
    outcome = await agent._run_tool_call(tool_call, {}, {})

    assert outcome.output == "Error: boom"
    assert outcome.failed
    assert outcome.duration_seconds >= 0.0


async def collect(agen):
    return [chunk async for chunk in agen]


@pytest.mark.asyncio
async def test_prompt_stream_instructions_and_output(mock_kernel, run_context):
    """Instructions stream per step; the output field streams as full JSON."""
    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="look things up",
        tool_calls=[ToolCall(name="python://t", call_id="c1")],
    )
    step2 = StepOutput(
        instructions="write the answer",
        output=MockResponse(answer="Final answer"),
    )
    client = ScriptedClient([step1, step2])
    agent = make_agent(client, mock_kernel, run_context)

    chunks = await collect(
        agent.prompt_stream(
            "Do something",
            response_model=MockResponse,
            max_steps=5,
            stream_output=True,
            stream_instructions=True,
        )
    )

    # Instructions: full-buffer partials converge on the step's text, and each
    # step ends with an instructions complete carrying the final text.
    instr = [c for c in chunks if c.name == "instructions"]
    instr_completes = [c for c in instr if c.type == "complete"]
    assert [c.value for c in instr_completes] == [
        "look things up",
        "write the answer",
    ]
    assert instr[0].type == "partial"

    # Output: streamed under "response" as full JSON of the output field.
    out_partials = [c for c in chunks if c.name == "response" and c.type == "partial"]
    assert out_partials
    assert json.loads(out_partials[-1].value) == {"answer": "Final answer"}

    # The final chunk is the authoritative response complete.
    assert chunks[-1].type == "complete"
    assert chunks[-1].name == "response"
    assert json.loads(chunks[-1].value) == {"answer": "Final answer"}


@pytest.mark.asyncio
async def test_prompt_stream_step_partials(mock_kernel, run_context):
    """stream_partials exposes each step's raw output under step<N>."""
    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="tooling",
        tool_calls=[ToolCall(name="python://t", call_id="c1")],
    )
    step2 = StepOutput(instructions="done", output=MockResponse(answer="ok"))
    client = ScriptedClient([step1, step2])
    agent = make_agent(client, mock_kernel, run_context)

    chunks = await collect(
        agent.prompt_stream(
            "Do it", response_model=MockResponse, max_steps=5, stream_partials=True
        )
    )

    step0 = [c for c in chunks if c.name == "step0"]
    step1_chunks = [c for c in chunks if c.name == "step1"]
    assert step0 and step0[-1].type == "complete"
    assert step0[-1].value == step1.model_dump_json()
    assert step1_chunks and step1_chunks[-1].type == "complete"
    # Full-buffer mode: the last partial holds the whole step JSON.
    assert step0[-2].value == step1.model_dump_json()


@pytest.mark.asyncio
async def test_prompt_stream_delta_instructions(mock_kernel, run_context):
    """In delta mode instruction partials are suffix deltas."""
    StepOutput = get_step_output_type(MockResponse)
    step = StepOutput(instructions="abcdef", output=MockResponse(answer="ok"))
    client = ScriptedClient([step], chunk_size=3)
    agent = make_agent(client, mock_kernel, run_context)

    chunks = await collect(
        agent.prompt_stream(
            "Do it",
            response_model=MockResponse,
            max_steps=2,
            stream_instructions=True,
            stream_delta=True,
        )
    )
    partials = [c for c in chunks if c.name == "instructions" and c.type == "partial"]
    assert "".join(c.value for c in partials) == "abcdef"


@pytest.mark.asyncio
async def test_prompt_stream_no_flags_only_final_complete(mock_kernel, run_context):
    """With all flags off the stream carries only the final response chunk."""
    StepOutput = get_step_output_type(MockResponse)
    client = ScriptedClient(
        [StepOutput(instructions="done", output=MockResponse(answer="ok"))]
    )
    agent = make_agent(client, mock_kernel, run_context)

    chunks = await collect(
        agent.prompt_stream("Do it", response_model=MockResponse, max_steps=2)
    )
    assert [(c.type, c.name) for c in chunks] == [("complete", "response")]


class _RestartThenAnswerClient(BaseLlmClient):
    """First attempt streams garbage and restarts (as a retry would); the
    second attempt streams the scripted step. Also emits an auxiliary
    'thought' stream like Gemini."""

    def __init__(self, step_json):
        super().__init__()
        self.step_json = step_json

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        vs = streamer.get_value_streamer("response", response_model=response_model)
        await vs.stream_partial('{"instructions": "gar')
        streamer.reset_active()
        await streamer.stream_restart("attempt 1: transient")

        vs2 = streamer.get_value_streamer("response", response_model=response_model)
        thought = streamer.get_value_streamer("thought")
        await thought.stream_partial("hmm")
        await vs2.stream_partial(self.step_json)
        await vs2.stream_complete()
        await thought.stream_complete()


@pytest.mark.asyncio
async def test_prompt_stream_restart_resets_demux_and_forwards_aux(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(MockResponse)
    step = StepOutput(instructions="done", output=MockResponse(answer="ok"))
    client = _RestartThenAnswerClient(step.model_dump_json())
    agent = make_agent(client, mock_kernel, run_context)

    chunks = await collect(
        agent.prompt_stream(
            "Do it",
            response_model=MockResponse,
            max_steps=2,
            stream_output=True,
            stream_instructions=True,
        )
    )

    restarts = [c for c in chunks if c.type == "restart"]
    assert restarts and "attempt 1" in restarts[0].value
    # The pre-restart garbage instructions never leak into the final state.
    instr_completes = [
        c for c in chunks if c.name == "instructions" and c.type == "complete"
    ]
    assert [c.value for c in instr_completes] == ["done"]
    # Auxiliary provider streams are forwarded when stream_output is on.
    assert any(c.name == "thought" for c in chunks)
    assert json.loads(chunks[-1].value) == {"answer": "ok"}


@pytest.mark.asyncio
async def test_tool_call_without_id_gets_stable_id(mock_kernel, run_context):
    """Tool calls missing a call_id are assigned a deterministic one."""
    StepOutput = get_step_output_type(MockResponse)
    client = ScriptedClient(
        [
            StepOutput(
                instructions="tooling",
                tool_calls=[ToolCall(name="python://t")],  # no call_id
            ),
            StepOutput(instructions="finish", output=MockResponse(answer="ok")),
        ]
    )
    agent = make_agent(client, mock_kernel, run_context)
    result = await agent.prompt("go", response_model=MockResponse)
    assert result.answer == "ok"
    assert mock_kernel.call_tool.await_count == 1


def test_demuxer_restart_on_non_monotonic_instructions():
    """A non-monotonic instructions change in delta mode emits a restart."""
    demux = StepStreamDemuxer(0, stream_instructions=True, stream_delta=True)
    demux._sent_instructions = "previously sent"

    events = demux.on_chunk(
        StreamContent(
            type="partial", name="response", value='{"instructions": "different"}'
        )
    )

    assert [e.type for e in events] == ["restart", "partial"]
    assert events[1].value == "different"
    assert demux._sent_instructions == "different"


def test_demuxer_chunk_routing():
    """Chunk routing: restarts reset state, aux streams are gated by
    stream_output, completes capture the step JSON."""
    demux = StepStreamDemuxer(1, stream_partials=True)

    assert demux.on_chunk(
        StreamContent(type="partial", name="response", value='{"a"')
    ) == [StreamContent(type="partial", name="step1", value='{"a"')]

    # Auxiliary stream is dropped without stream_output.
    assert (
        demux.on_chunk(StreamContent(type="partial", name="thought", value="t")) == []
    )

    # A restart clears the buffer and passes through.
    restart = StreamContent(type="restart", name="response", value="attempt 1")
    assert demux.on_chunk(restart) == [restart]
    assert demux.buffer == ""

    demux.on_chunk(StreamContent(type="partial", name="response", value='{"b": 1}'))
    assert demux.on_chunk(StreamContent(type="complete", name="response")) == []
    assert demux.step_json == '{"b": 1}'


@pytest.mark.asyncio
async def test_agent_creates_its_own_run_context_when_none_is_given(mock_kernel):
    agent = Agent(llm_client=ScriptedClient(), kernel=mock_kernel)
    assert isinstance(agent.run_context, RunContext)
    assert agent.run_context.data == {}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, {}),
        ("", {}),
        ("  \n", {}),
        ("{}", {}),
        ({"a": 1, "b": None}, {"a": 1, "b": None}),
        ('{"a": 1, "b": null}', {"a": 1, "b": None}),
        ('{"filters": {"year": 2026}}', {"filters": {"year": 2026}}),
    ],
)
def test_parse_json_object_reads_an_object_or_a_json_object_in_a_string(value, expected):
    assert _parse_json_object("args", value) == expected


def test_parse_json_object_returns_a_given_object_itself():
    given = {"a": 1}
    assert _parse_json_object("args", given) is given


@pytest.mark.parametrize(
    "value, message",
    [
        ("{not json}", "`args_json` is not valid JSON (Expecting property name"),
        ('{"a": ', "`args_json` is not valid JSON (Expecting value, at character 6)."),
        ("[1, 2]", "`args_json` must hold a JSON object."),
        ('"text"', "`args_json` must hold a JSON object."),
        ("7", "`args_json` must hold a JSON object."),
        ("null", "`args_json` must hold a JSON object."),
    ],
)
def test_parse_json_object_refuses_what_is_not_a_json_object(value, message):
    with pytest.raises(_ToolArgumentError) as error:
        _parse_json_object("args_json", value)
    assert message in str(error.value)


def test_parse_json_object_keeps_the_json_error_as_the_cause():
    with pytest.raises(_ToolArgumentError) as error:
        _parse_json_object("args", "{not json}")
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def call(**fields) -> ToolCall:
    return ToolCall(name="python://test", **fields)


def test_written_arguments_come_from_args_and_args_json():
    assert _written_arguments(call()) == {}
    assert _written_arguments(call(args="", args_json="  ")) == {}
    assert _written_arguments(call(args={"a": 1})) == {"a": 1}
    assert _written_arguments(call(args='{"a": 1}')) == {"a": 1}
    assert _written_arguments(call(args_json='{"a": 1}')) == {"a": 1}
    assert _written_arguments(call(args_json='{"a": 1, "b": 1}', args={"b": 2})) == {
        "a": 1,
        "b": 2,
    }


def test_a_null_in_args_is_not_supplied_and_a_null_in_args_json_is_kept():
    assert _written_arguments(call(args={"a": 1, "b": None})) == {"a": 1}
    assert _written_arguments(call(args_json='{"a": 1, "b": null}')) == {"a": 1, "b": None}


def test_written_arguments_do_not_alter_the_call():
    given = {"a": 1, "b": None}
    assert _written_arguments(call(args=given)) == {"a": 1}
    assert given == {"a": 1, "b": None}


def test_a_call_without_refs_has_none():
    assert _read_refs(call()) == []
    assert _read_refs(call(refs=[])) == []
    assert _read_refs(call(refs={})) == []


def test_refs_are_read_in_the_order_written():
    assert _read_refs(
        call(
            refs=[
                {"arg": "document", "source": "step", "key": "doc"},
                {"arg": "uid", "source": "input", "key": "user_id"},
                ArgRef(arg="other", source="step", key="c2"),
            ]
        )
    ) == [
        ArgRef(arg="document", source="step", key="doc"),
        ArgRef(arg="uid", source="input", key="user_id"),
        ArgRef(arg="other", source="step", key="c2"),
    ]


def test_refs_are_read_from_an_object_keyed_by_argument():
    assert _read_refs(
        call(
            refs={
                "document": {"source": "step", "key": "doc"},
                "uid": {"source": "input", "key": "user_id"},
            }
        )
    ) == [
        ArgRef(arg="document", source="step", key="doc"),
        ArgRef(arg="uid", source="input", key="user_id"),
    ]
    # The key of the object names the argument, whatever the entry says.
    assert _read_refs(call(refs={"a": {"arg": "b", "source": "step", "key": "c1"}})) == [
        ArgRef(arg="a", source="step", key="c1")
    ]


@pytest.mark.parametrize(
    "refs",
    [
        ["c1"],
        [None],
        [{"arg": "a", "source": "memory", "key": "c1"}],
        [{"arg": "a", "source": "step"}],
        [{"source": "step", "key": "c1"}],
        [{"arg": "a", "source": "step", "key": 7}],
        {"a": "c1"},
    ],
)
def test_a_malformed_refs_entry_is_refused(refs):
    with pytest.raises(_ToolArgumentError) as error:
        _read_refs(call(refs=refs))
    assert "is malformed: each needs `arg`, `source`" in str(error.value)
    assert isinstance(error.value.__cause__, ValidationError)


def step_ref(key: str) -> ArgRef:
    return ArgRef(arg="a", source="step", key=key)


def test_a_step_ref_resolves_to_the_result_of_that_call():
    tool_results = {"c1": {"title": "T"}, "c2": None}
    assert _resolve_ref(step_ref("c1"), tool_results, {}) == {"title": "T"}
    # A tool that returned nothing still has an output.
    assert _resolve_ref(step_ref("c2"), tool_results, {}) is None


def test_an_input_ref_resolves_to_the_context_variable():
    ref = ArgRef(arg="a", source="input", key="user_id")
    # A tool result of the same name is not what an input ref names.
    assert _resolve_ref(ref, {"user_id": "a result"}, {"user_id": "u-1"}) == "u-1"


def test_a_ref_to_nothing_is_refused_with_what_is_available():
    with pytest.raises(_ToolArgumentError) as error:
        _resolve_ref(step_ref("nope"), {"c2": 2, "c1": 1}, {"user_id": "u-1"})
    assert str(error.value) == (
        "`a` refers to the call_id 'nope', which does not exist. Available: c1, c2."
    )
    with pytest.raises(_ToolArgumentError) as error:
        _resolve_ref(ArgRef(arg="a", source="input", key="nope"), {"c1": 1}, {"user_id": "u-1"})
    assert str(error.value) == (
        "`a` refers to the context variable 'nope', which does not exist. "
        "Available: user_id."
    )
    with pytest.raises(_ToolArgumentError, match="Available: none\\."):
        _resolve_ref(step_ref("nope"), {}, {})


def test_a_ref_to_a_call_of_the_same_step_is_refused():
    with pytest.raises(_ToolArgumentError) as error:
        _resolve_ref(step_ref("c2"), {"c1": 1}, {}, same_step_ids={"c2", "c3"})
    assert str(error.value) == (
        "`a` refers to the call_id 'c2', a call of this same step. The calls of "
        "one step run together: refer to it from a later step."
    )


def test_a_ref_to_a_failed_call_is_refused_and_the_call_is_not_offered():
    tool_results = {"good": 1, "bad": _FAILED}
    with pytest.raises(_ToolArgumentError) as error:
        _resolve_ref(step_ref("bad"), tool_results, {})
    assert str(error.value) == (
        "`a` refers to the call_id 'bad', which ended in an error and has no output."
    )
    with pytest.raises(_ToolArgumentError, match="Available: good\\."):
        _resolve_ref(step_ref("nope"), tool_results, {})


def test_a_call_id_is_kept_unless_it_is_taken():
    assert _unique_call_id("doc", set()) == "doc"
    assert _unique_call_id("doc", {"doc"}) == "doc_2"
    assert _unique_call_id("doc", {"doc", "doc_2", "doc_3"}) == "doc_4"


def test_every_call_of_a_step_gets_an_id_of_its_own():
    tool_calls = [call(call_id="doc"), call(), call(call_id="doc"), call(call_id="new")]

    renamed = _assign_unique_call_ids(tool_calls, 3, {"doc": "an earlier result"})

    assert [tool_call.call_id for tool_call in tool_calls] == [
        "doc_2",
        "tool_call_3_1",
        "doc_3",
        "new",
    ]
    # Only a call that asked for an id and did not get it is reported.
    assert renamed == {"doc_2": "doc", "doc_3": "doc"}


def test_arguments_are_resolved_from_written_values_and_references():
    tool_call = call(
        args={"mode": "fast", "limit": None, "document": None},
        refs=[
            {"arg": "document", "source": "step", "key": "c1"},
            {"arg": "uid", "source": "input", "key": "user_id"},
        ],
    )

    args, refs = _resolve_arguments(tool_call, {"c1": {"title": "T"}}, {"user_id": "u-1"})

    assert args == {"mode": "fast", "document": {"title": "T"}, "uid": "u-1"}
    assert refs == {
        "document": {"source": "step", "key": "c1"},
        "uid": {"source": "input", "key": "user_id"},
    }


def test_a_written_value_beats_a_reference_to_the_same_argument():
    tool_call = call(args={"x": 1}, refs=[{"arg": "x", "source": "step", "key": "c1"}])

    assert _resolve_arguments(tool_call, {"c1": 7}, {}) == ({"x": 1}, {})
    # The reference is still looked up, so one that names nothing is reported.
    with pytest.raises(_ToolArgumentError, match="call_id 'c1', which does not exist"):
        _resolve_arguments(tool_call, {}, {})


def test_the_later_of_two_references_to_one_argument_counts():
    tool_call = call(
        refs=[
            {"arg": "x", "source": "step", "key": "c1"},
            {"arg": "x", "source": "input", "key": "user_id"},
        ]
    )

    assert _resolve_arguments(tool_call, {"c1": 7}, {"user_id": "u-1"}) == (
        {"x": "u-1"},
        {"x": {"source": "input", "key": "user_id"}},
    )


def test_a_call_without_arguments_resolves_to_none():
    assert _resolve_arguments(call(), {}, {}) == ({}, {})
    assert _resolve_arguments(call(args="", args_json="  ", refs=[]), {}, {}) == ({}, {})


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"args": '{"a": '}, "`args` is not valid JSON"),
        ({"args_json": "[1, 2]"}, "`args_json` must hold a JSON object"),
        ({"refs": ["c1"]}, "`refs` entry 'c1' is malformed"),
        (
            {"refs": [{"arg": "a", "source": "input", "key": "nope"}]},
            "`a` refers to the context variable 'nope', which does not exist. "
            "Available: user_id.",
        ),
        (
            {"refs": [{"arg": "a", "source": "step", "key": "c9"}]},
            "`a` refers to the call_id 'c9', a call of this same step.",
        ),
    ],
)
def test_unusable_arguments_are_refused(fields, message):
    with pytest.raises(_ToolArgumentError) as error:
        _resolve_arguments(call(**fields), {}, {"user_id": "u-1"}, same_step_ids={"c9"})
    assert message in str(error.value)


@pytest.mark.asyncio
async def test_unreadable_arguments_become_the_calls_result(mock_kernel, run_context):
    """The tool is not run with half its arguments; the model is told why."""
    agent = make_agent(ScriptedClient(), mock_kernel, run_context)
    tool_call = ToolCall(name="python://t", args_json="{not json}", call_id="c1")

    outcome = await agent._run_tool_call(tool_call, {}, {})

    assert outcome.args == {} and outcome.refs == {}
    assert outcome.output.startswith("Error: `args_json` is not valid JSON")
    assert outcome.failed
    assert outcome.duration_seconds >= 0.0
    mock_kernel.call_tool.assert_not_awaited()


def test_a_step_with_an_unknown_tool_call_field_is_refused():
    StepOutput = get_step_output_type(str)
    step = StepOutput.model_validate(
        {
            "instructions": "i",
            "tool_calls": [
                {
                    "name": "python://t",
                    "args": {"a": 1},
                    "refs": [{"arg": "b", "source": "step", "key": "c"}],
                }
            ],
        }
    )
    assert step.tool_calls[0].args == {"a": 1}
    with pytest.raises(ValidationError):
        StepOutput.model_validate(
            {"instructions": "i", "tool_calls": [{"name": "python://t", "arguments": {}}]}
        )


@pytest.mark.asyncio
async def test_debug_prints_the_rendered_prompt(mock_kernel, run_context, capsys):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = Agent(
        llm_client=client, kernel=mock_kernel, run_context=run_context, debug=True
    )

    assert await agent.prompt("say something", max_steps=1) == "done"

    printed = capsys.readouterr().out
    assert "say something" in printed
    assert "[STEPS EXECUTED SO FAR]" in printed


@pytest.mark.asyncio
async def test_history_sits_between_the_system_prompt_and_the_step_message(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = make_agent(client, mock_kernel, run_context)
    history = [
        ChatMessage(role="user", content="I am John"),
        ChatMessage(role="assistant", content="Hello John"),
    ]

    assert await agent.prompt("What is my name?", history=history) == "done"

    messages = client.seen[0]
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert "What is my name?" in messages[0].content
    assert messages[1:3] == history
    assert "Analyze the situation" in messages[3].content


@pytest.mark.asyncio
async def test_without_history_the_step_message_follows_the_system_prompt(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = make_agent(client, mock_kernel, run_context)

    await agent.prompt("task")

    assert [m.role for m in client.seen[0]] == ["system", "user"]


@pytest.mark.asyncio
async def test_executed_steps_are_in_the_step_message_not_the_system_prompt(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient(
        [
            StepOutput(
                instructions="look it up",
                tool_calls=[ToolCall(name="python://lookup", call_id="c1")],
            ),
            StepOutput(instructions="answer", output="done"),
        ]
    )
    agent = make_agent(client, mock_kernel, run_context)

    assert await agent.prompt("task") == "done"

    system, step = client.seen[1]
    assert "look it up" not in system.content
    assert "Tool result" not in system.content
    assert "look it up" in step.content
    assert "python://lookup(call_id=c1)" in step.content
    assert "Tool result" in step.content
    assert "current_step=1" in step.content


@pytest.mark.asyncio
async def test_custom_step_template_receives_the_same_variables(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = Agent(
        llm_client=client,
        kernel=mock_kernel,
        run_context=run_context,
        prompt_template=Template("task={{ prompt }}"),
        step_template=Template("step {{ current_step }} of {{ max_steps }}"),
    )

    await agent.prompt("t", max_steps=3)

    system, step = client.seen[0]
    assert system.content == "task=t"
    assert step.content == "step 0 of 3"


@pytest.mark.asyncio
async def test_inputs_replace_the_run_context_as_context_variables(mock_kernel):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = make_agent(
        client, mock_kernel, RunContext(data={"passages": ["a", "b"], "x": 1})
    )

    await agent.prompt("task", inputs={"city": "Turku"})

    system = client.seen[0][0].content
    assert "- city: Turku" in system
    assert "passages" not in system and "['a', 'b']" not in system


@pytest.mark.asyncio
async def test_without_inputs_the_run_context_is_listed(mock_kernel):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = make_agent(client, mock_kernel, RunContext(data={"city": "Turku"}))

    await agent.prompt("task")

    assert "- city: Turku" in client.seen[0][0].content


@pytest.mark.asyncio
async def test_input_references_resolve_against_the_inputs(mock_kernel):
    StepOutput = get_step_output_type(str)
    client = ScriptedClient(
        [
            StepOutput(
                instructions="look",
                tool_calls=[
                    ToolCall(
                        name="python://t",
                        call_id="c1",
                        refs=[{"arg": "q", "source": "input", "key": "city"}],
                    )
                ],
            ),
            StepOutput(instructions="answer", output="done"),
        ]
    )
    agent = make_agent(client, mock_kernel, RunContext(data={"city": "hidden"}))

    await agent.prompt("task", inputs={"city": "Turku"})

    mock_kernel.call_tool.assert_awaited_once_with(
        tool_uri="python://t", arguments={"q": "Turku"}
    )


@pytest.mark.asyncio
async def test_all_tools_are_described_without_an_allow_list(mock_kernel, run_context):
    """``allowed_tools=None`` keeps every registered tool available."""
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = make_agent(client, mock_kernel, run_context)

    await agent.prompt("do it", max_steps=1)

    mock_kernel.get_tool_descriptions.assert_awaited_with(None)


@pytest.mark.asyncio
async def test_allowed_tools_filters_the_described_tools(mock_kernel, run_context):
    """Only the allowed tools are offered to the model."""
    StepOutput = get_step_output_type(str)
    client = ScriptedClient([StepOutput(instructions="answer", output="done")])
    agent = Agent(
        llm_client=client,
        kernel=mock_kernel,
        run_context=run_context,
        allowed_tools=["python://web.crawl"],
    )

    await agent.prompt("do it", max_steps=1)

    mock_kernel.get_tool_descriptions.assert_awaited_with(["python://web.crawl"])


@pytest.mark.asyncio
async def test_disallowed_tool_call_is_refused(mock_kernel, run_context):
    """A tool outside the allow-list is not executed; the model is told why."""
    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="try the forbidden tool",
        tool_calls=[ToolCall(name="python://web.search", call_id="c1")],
    )
    step2 = StepOutput(instructions="answer", output=MockResponse(answer="ok"))
    client = ScriptedClient([step1, step2])
    agent = Agent(
        llm_client=client,
        kernel=mock_kernel,
        run_context=run_context,
        allowed_tools=["python://web.crawl"],
    )

    result = await agent.prompt("do it", response_model=MockResponse, max_steps=5)

    assert result.answer == "ok"
    mock_kernel.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_tool_call_still_runs(mock_kernel, run_context):
    """The allow-list does not get in the way of a permitted tool."""
    StepOutput = get_step_output_type(MockResponse)
    step1 = StepOutput(
        instructions="crawl the page",
        tool_calls=[
            ToolCall(name="python://web.crawl", args={"url": "x"}, call_id="c1")
        ],
    )
    step2 = StepOutput(instructions="answer", output=MockResponse(answer="ok"))
    agent = Agent(
        llm_client=ScriptedClient([step1, step2]),
        kernel=mock_kernel,
        run_context=run_context,
        allowed_tools=["python://web.crawl"],
    )

    await agent.prompt("do it", response_model=MockResponse, max_steps=5)

    mock_kernel.call_tool.assert_awaited_once_with(
        tool_uri="python://web.crawl", arguments={"url": "x"}
    )


@pytest.mark.asyncio
async def test_the_step_message_states_the_request_and_says_when_it_is_the_last(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(MockResponse)
    lookup = StepOutput(
        instructions="look", tool_calls=[ToolCall(name="python://lookup", call_id="c1")]
    )
    client = ScriptedClient(
        [
            lookup,
            lookup,
            StepOutput(instructions="answer", output=MockResponse(answer="done")),
        ]
    )
    agent = make_agent(client, mock_kernel, run_context)

    await agent.prompt(
        "task", response_model=MockResponse, max_steps=3, request="Around 800 euros"
    )

    steps = [seen[-1].content for seen in client.seen]
    assert all(
        "[CURRENT REQUEST]" in step and "Around 800 euros" in step for step in steps
    )
    assert ["This is the last step" in step for step in steps] == [False, False, True]
    assert (
        "Analyze the situation" in steps[0] and "Analyze the situation" not in steps[2]
    )
    # Without a request the step message has no such section, as before.
    plain = ScriptedClient(
        [StepOutput(instructions="answer", output=MockResponse(answer="done"))]
    )
    await make_agent(plain, mock_kernel, run_context).prompt(
        "task", response_model=MockResponse, max_steps=2
    )
    assert "[CURRENT REQUEST]" not in plain.seen[0][-1].content


@pytest.mark.asyncio
async def test_an_answer_on_the_last_step_is_kept_though_its_tool_calls_are_not_run(
    mock_kernel, run_context
):
    StepOutput = get_step_output_type(MockResponse)
    lookup = StepOutput(
        instructions="look", tool_calls=[ToolCall(name="python://lookup", call_id="c1")]
    )
    both = StepOutput(
        instructions="answer anyway",
        tool_calls=[ToolCall(name="python://lookup", call_id="c2")],
        output=MockResponse(answer="from what I have"),
    )
    client = ScriptedClient([lookup, both])
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt("task", response_model=MockResponse, max_steps=2)

    assert result == MockResponse(answer="from what I have")
    assert mock_kernel.call_tool.await_count == 1


class Document(BaseModel):
    title: str
    body: str


@pythontool
def fetch_document(doc_id: str) -> Document:
    """Fetch a document by its id."""
    return Document(title=f"Policy {doc_id}", body="Claims are reported within 72 hours.")


@pythontool
def summarise(document: Document, max_words: int = 100, language: Optional[str] = None) -> str:
    """Summarise a fetched document."""
    return f"{document.title} in {max_words} words ({language})"


@pythontool
def search(query: str, filters: dict) -> str:
    """Search the knowledge base."""
    return f"{query} {sorted(filters)}"


def real_kernel() -> FunctionKernel:
    kernel = FunctionKernel()
    kernel.register_python_tool("fetch_document", fetch_document)
    kernel.register_python_tool("summarise", summarise)
    kernel.register_python_tool("search", search)
    return kernel


def typed_agent(client, **kwargs) -> Agent:
    return Agent(llm_client=client, kernel=real_kernel(), run_context=RunContext(), **kwargs)


async def test_the_model_is_asked_for_typed_tool_calls():
    client = ScriptedLlmClient([{"instructions": "answer", "output": "done"}])
    agent = typed_agent(client, allowed_tools=["python://fetch_document", "python://summarise"])

    assert await agent.prompt("task") == "done"

    schema = client.calls[0].response_model.model_json_schema()
    options = schema["properties"]["tool_calls"]["items"]["anyOf"]
    assert [option["$ref"].rsplit("/", 1)[1] for option in options] == [
        "python_fetch_document_call",
        "python_summarise_call",
    ]
    arguments = schema["$defs"]["python_summarise_args"]["properties"]
    assert set(arguments) == {"document", "max_words", "language"}

    system_prompt = client.calls[0].messages[0].content
    assert "- `args`: the argument values, as an object." in system_prompt
    assert "- `refs`:" in system_prompt
    assert "args_json" not in system_prompt


async def test_typed_calls_run_with_literals_and_references():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch it",
                "tool_calls": [
                    {
                        "name": "python://fetch_document",
                        "args": {"doc_id": None},
                        "refs": [{"arg": "doc_id", "source": "input", "key": "wanted"}],
                        "call_id": "doc",
                    }
                ],
            },
            {
                "instructions": "summarise it",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "args": {"document": None, "max_words": 50, "language": None},
                        "refs": [{"arg": "document", "source": "step", "key": "doc"}],
                        "call_id": "summary",
                    }
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []
    agent = typed_agent(client, on_step=steps.append)

    assert await agent.prompt("summarise the wanted document", inputs={"wanted": "D-9"}) == "done"

    fetch, summary = steps[0]["tool_calls"][0], steps[1]["tool_calls"][0]
    assert fetch["args"] == {"doc_id": "D-9"}
    assert fetch["output"] == {
        "title": "Policy D-9",
        "body": "Claims are reported within 72 hours.",
    }
    # A null argument is not supplied: the tool's own default applies.
    assert summary["args"]["max_words"] == 50 and "language" not in summary["args"]
    assert summary["output"] == {"result": "Policy D-9 in 50 words (None)"}
    assert "Policy D-9 in 50 words (None)" in client.calls[2].messages[-1].content


async def test_a_tool_with_a_free_form_argument_is_called_with_a_json_string():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "search",
                "tool_calls": [
                    {
                        "name": "python://search",
                        "args_json": '{"query": "claims", "filters": {"year": 2026}}',
                        "call_id": "hits",
                    }
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []
    agent = typed_agent(client, allowed_tools=["python://search"], on_step=steps.append)

    assert await agent.prompt("search for claims") == "done"

    call_schema = client.calls[0].response_model.model_json_schema()["$defs"][
        "python_search_call"
    ]
    assert "args_json" in call_schema["properties"] and "args" not in call_schema["properties"]
    system_prompt = client.calls[0].messages[0].content
    assert "these tools take `args_json` in place of `args`" in system_prompt
    assert "written as a JSON string: python://search." in system_prompt
    assert steps[0]["tool_calls"][0]["output"] == {"result": "claims ['year']"}


async def test_a_referenced_value_is_not_repeated_in_the_step_message():
    """The step message names the reference; the value is printed once, as an output."""
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch it",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": "D-9"}, "call_id": "doc"}
                ],
            },
            {
                "instructions": "summarise it",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "args": {"max_words": 50},
                        "refs": [{"arg": "document", "source": "step", "key": "doc"}],
                        "call_id": "summary",
                    }
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []

    assert await typed_agent(client, on_step=steps.append).prompt("task") == "done"

    summary = steps[1]["tool_calls"][0]
    # The record keeps what the tool was called with, and says where it came from.
    assert summary["args"]["document"].title == "Policy D-9"
    assert summary["refs"] == {"document": {"source": "step", "key": "doc"}}
    assert steps[0]["tool_calls"][0]["refs"] == {}
    message = client.calls[2].messages[-1].content
    assert "python://fetch_document(call_id=doc) args={'doc_id': 'D-9'}\n" in message
    assert (
        "python://summarise(call_id=summary) args={'max_words': 50} refs=document=<step:doc>\n"
        in message
    )
    assert message.count("Claims are reported within 72 hours.") == 1


async def test_several_references_are_listed_in_the_step_message():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": "D-1"}, "call_id": "doc"}
                ],
            },
            {
                "instructions": "summarise",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "refs": [
                            {"arg": "document", "source": "step", "key": "doc"},
                            {"arg": "language", "source": "input", "key": "lang"},
                        ],
                    }
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )

    assert await typed_agent(client).prompt("task", inputs={"lang": "et"}) == "done"

    assert (
        "args={} refs=document=<step:doc>, language=<input:lang>\n"
        in client.calls[2].messages[-1].content
    )


async def test_a_failed_call_cannot_be_referenced():
    """Its error message is not handed to the next tool as if it were data."""
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": None}, "call_id": "doc"}
                ],
            },
            {
                "instructions": "summarise",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "refs": [{"arg": "document", "source": "step", "key": "doc"}],
                        "call_id": "summary",
                    }
                ],
            },
            {
                "instructions": "summarise again",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "refs": [{"arg": "document", "source": "step", "key": "summary"}],
                    }
                ],
            },
            {"instructions": "answer", "output": "could not"},
        ]
    )
    steps = []

    assert await typed_agent(client, on_step=steps.append).prompt("task") == "could not"

    assert steps[0]["tool_calls"][0]["output"].startswith("Error: ")
    expected = "Error: `document` refers to the call_id 'doc', which ended in an error and has no output."
    assert steps[1]["tool_calls"][0]["output"] == expected
    assert expected in client.calls[2].messages[-1].content
    # A call that was not run for its arguments has no output either.
    assert steps[2]["tool_calls"][0]["output"] == expected.replace("'doc'", "'summary'")


async def test_a_call_of_a_tool_that_is_not_allowed_cannot_be_referenced():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "search",
                "tool_calls": [
                    {"name": "python://search", "args_json": '{"query": "q"}', "call_id": "hits"}
                ],
            },
            {
                "instructions": "fetch",
                "tool_calls": [
                    {
                        "name": "python://fetch_document",
                        "refs": [{"arg": "doc_id", "source": "step", "key": "hits"}],
                    }
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []
    agent = typed_agent(client, allowed_tools=["python://fetch_document"], on_step=steps.append)

    assert await agent.prompt("task") == "done"

    assert steps[0]["tool_calls"][0]["output"] == "Error: tool python://search is not available."
    assert "'hits', which ended in an error" in steps[1]["tool_calls"][0]["output"]


async def test_a_call_cannot_reference_a_call_of_its_own_step():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch and summarise at once",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": "D-9"}, "call_id": "doc"},
                    {
                        "name": "python://summarise",
                        "refs": [{"arg": "document", "source": "step", "key": "doc"}],
                    },
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []

    assert await typed_agent(client, on_step=steps.append).prompt("task") == "done"

    fetch, summary = steps[0]["tool_calls"]
    assert fetch["output"]["title"] == "Policy D-9"
    assert summary["output"] == (
        "Error: `document` refers to the call_id 'doc', a call of this same step. "
        "The calls of one step run together: refer to it from a later step."
    )
    system_prompt = client.calls[0].messages[0].content
    assert "a tool call from an earlier step" in system_prompt
    assert "a call cannot reference another call of its own step" in system_prompt


async def test_a_reused_call_id_does_not_replace_the_earlier_result():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": "D-1"}, "call_id": "doc"}
                ],
            },
            {
                "instructions": "fetch two more",
                "tool_calls": [
                    {"name": "python://fetch_document", "args": {"doc_id": "D-2"}, "call_id": "doc"},
                    {"name": "python://fetch_document", "args": {"doc_id": "D-3"}, "call_id": "doc"},
                ],
            },
            {
                "instructions": "summarise each",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "args": {"max_words": 5},
                        "refs": [{"arg": "document", "source": "step", "key": key}],
                    }
                    for key in ("doc", "doc_2", "doc_3")
                ],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []

    assert await typed_agent(client, on_step=steps.append).prompt("task") == "done"

    assert steps[0]["tool_calls"][0]["requested_call_id"] is None
    assert [
        (call["call_id"], call["requested_call_id"]) for call in steps[1]["tool_calls"]
    ] == [("doc_2", "doc"), ("doc_3", "doc")]
    assert [call["output"]["result"][:10] for call in steps[2]["tool_calls"]] == [
        "Policy D-1",
        "Policy D-2",
        "Policy D-3",
    ]
    message = client.calls[2].messages[-1].content
    assert "note: the call_id `doc` was already used, so this result is `doc_2`." in message
    assert "note: the call_id `doc` was already used, so this result is `doc_3`." in message
    assert message.count("note:") == 2


async def test_a_generated_call_id_avoids_one_the_model_chose():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "fetch",
                "tool_calls": [
                    {
                        "name": "python://fetch_document",
                        "args": {"doc_id": "D-1"},
                        "call_id": "tool_call_1_0",
                    }
                ],
            },
            {
                "instructions": "fetch",
                "tool_calls": [{"name": "python://fetch_document", "args": {"doc_id": "D-2"}}],
            },
            {"instructions": "answer", "output": "done"},
        ]
    )
    steps = []

    assert await typed_agent(client, on_step=steps.append).prompt("task") == "done"

    assert steps[1]["tool_calls"][0]["call_id"] == "tool_call_1_0_2"


async def test_an_unresolvable_call_is_answered_with_an_error_the_model_reads():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "summarise",
                "tool_calls": [
                    {
                        "name": "python://summarise",
                        "args": {"max_words": 10},
                        "refs": [{"arg": "document", "source": "step", "key": "doc"}],
                    }
                ],
            },
            {"instructions": "answer", "output": "could not"},
        ]
    )
    steps = []
    agent = typed_agent(client, on_step=steps.append)

    assert await agent.prompt("summarise D-9") == "could not"

    call = steps[0]["tool_calls"][0]
    assert call["args"] == {}
    assert call["output"].startswith("Error: `document` refers to the call_id 'doc'")
    assert "Error: `document` refers to the call_id 'doc'" in client.calls[1].messages[-1].content


async def test_arguments_the_tool_refuses_are_answered_with_an_error():
    client = ScriptedLlmClient(
        [
            {
                "instructions": "summarise",
                "tool_calls": [{"name": "python://summarise", "args": {"max_words": 10}}],
            },
            {"instructions": "answer", "output": "could not"},
        ]
    )
    steps = []
    agent = typed_agent(client, on_step=steps.append)

    await agent.prompt("summarise D-9")

    output = steps[0]["tool_calls"][0]["output"]
    assert "argument validation failed" in output and "document" in output


async def test_an_agent_without_tools_is_asked_for_no_tool_calls():
    for kwargs in ({"kernel": None}, {"kernel": real_kernel(), "allowed_tools": []}):
        client = ScriptedLlmClient([{"instructions": "answer", "output": "done"}])
        agent = Agent(llm_client=client, run_context=RunContext(), **kwargs)

        assert await agent.prompt("task") == "done"
        assert list(client.calls[0].response_model.model_fields) == ["instructions", "output"]


TYPED_CALL_MODELS = [
    ("openai/gpt-5.6-luna", "OPENAI_API_KEY"),
    ("gemini/gemini-3.5-flash", "GEMINI_API_KEY"),
    ("scaleway/glm-5.3", "SCW_SECRET_KEY"),
]


@pytest.mark.integration
@pytest.mark.parametrize("model, key", TYPED_CALL_MODELS, ids=lambda value: value.split("/")[0])
async def test_a_provider_accepts_the_typed_step_schema(model, key):
    """Each provider takes the union of call models and the agent finishes the task."""
    if not os.getenv(key):
        pytest.skip(f"{key} is not set")
    steps = []
    agent = Agent(
        llm_client=make_client(model, LlmClientParameters(timeout_seconds=120)),
        kernel=real_kernel(),
        run_context=RunContext(),
        allowed_tools=["python://fetch_document", "python://summarise"],
        on_step=steps.append,
    )

    answer = await agent.prompt(
        "Fetch document D-9, then summarise it in at most 20 words with the "
        "summarise tool. Reply with the summarise tool's result, word for word.",
        max_steps=5,
    )

    calls = [call for step in steps for call in step["tool_calls"]]
    assert [call["name"] for call in calls][:2] == [
        "python://fetch_document",
        "python://summarise",
    ]
    assert calls[0]["args"] == {"doc_id": "D-9"}
    assert calls[1]["args"]["max_words"] == 20
    # The document is passed by reference, not copied out of the step message.
    assert calls[1]["refs"] == {
        "document": {"source": "step", "key": calls[0]["call_id"]}
    }
    assert calls[1]["output"] == {"result": "Policy D-9 in 20 words (None)"}
    assert "Policy D-9 in 20 words" in answer


@pytest.mark.integration
@pytest.mark.parametrize("model, key", TYPED_CALL_MODELS, ids=lambda value: value.split("/")[0])
async def test_a_provider_writes_args_json_for_a_free_form_tool(model, key):
    """Each provider takes a call model with ``args_json`` and fills it with a JSON object."""
    if not os.getenv(key):
        pytest.skip(f"{key} is not set")
    steps = []
    agent = Agent(
        llm_client=make_client(model, LlmClientParameters(timeout_seconds=120)),
        kernel=real_kernel(),
        run_context=RunContext(),
        allowed_tools=["python://search"],
        on_step=steps.append,
    )

    answer = await agent.prompt(
        "Search the knowledge base for the query 'claims' with the filters "
        '{"year": 2026}. Reply with the search tool\'s result, word for word.',
        max_steps=5,
    )

    call = steps[0]["tool_calls"][0]
    assert call["name"] == "python://search"
    assert call["args"] == {"query": "claims", "filters": {"year": 2026}}
    assert call["output"] == {"result": "claims ['year']"}
    assert "claims ['year']" in answer
