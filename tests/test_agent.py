import json

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from kavalai.agent import Agent, StepStreamDemuxer, ToolCall, get_step_output_type
from kavalai.run_context import RunContext
from kavalai.functionkernel import FunctionKernel
from kavalai.llm_clients.base_client import BaseLlmClient
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

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        self.await_count += 1
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
            ToolCall(name="python://test_tool", literal_args='{"val": 1}', call_id="c1")
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
    # The tool was executed with the resolved literal arguments.
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
    assert mock_kernel.call_tool.await_count == 3


@pytest.mark.asyncio
async def test_prompt_empty_step_output_stops(mock_kernel, run_context):
    """An empty completion ends the loop with no output."""
    client = ScriptedClient([""])
    agent = make_agent(client, mock_kernel, run_context)

    result = await agent.prompt("anything", response_model=MockResponse)

    assert result is None
    assert client.await_count == 1


@pytest.mark.asyncio
async def test_resolve_args_merges_sources(mock_kernel):
    """literal/context/input args merge with literal > context > input precedence."""
    agent = make_agent(
        ScriptedClient(), mock_kernel, RunContext(data={"user_id": "u-123"})
    )

    tool_call = ToolCall(
        name="python://test",
        literal_args='{"mode": "fast"}',
        planner_context_args='{"prev": "c1"}',
        input_args='{"uid": "user_id"}',
    )
    planner_context = {"c1": "previous result"}

    args = agent._resolve_args(tool_call, planner_context)

    assert args == {
        "uid": "u-123",
        "prev": "previous result",
        "mode": "fast",
    }


@pytest.mark.asyncio
async def test_planner_context_args_resolve_from_planner_context(mock_kernel):
    """planner_context_args resolves against the per-invocation planner_context."""
    agent = make_agent(
        ScriptedClient(), mock_kernel, RunContext(data={"input_key": "input value"})
    )

    tool_call = ToolCall(
        name="python://test",
        planner_context_args='{"a": "c1", "b": "missing"}',
    )
    planner_context = {"c1": "tool result"}

    args = agent._resolve_args(tool_call, planner_context)

    # "c1" comes from planner_context; an unknown key resolves to None and
    # input data is not reachable through planner_context_args.
    assert args == {"a": "tool result", "b": None}


@pytest.mark.asyncio
async def test_planner_context_isolated_across_invocations(mock_kernel, run_context):
    """Tool results in planner_context do not leak into the next invocation."""
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
                        planner_context_args='{"x": "c1"}',
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

    # The reference to the prior invocation's call_id resolves to None.
    first_args = mock_kernel.call_tool.await_args_list[0].kwargs["arguments"]
    assert first_args == {"x": None}


@pytest.mark.asyncio
async def test_tool_error_is_captured(mock_kernel, run_context):
    """A failing tool call is captured as a result string instead of raising."""
    agent = make_agent(ScriptedClient(), mock_kernel, run_context)
    mock_kernel.call_tool.side_effect = RuntimeError("boom")

    tool_call = ToolCall(name="python://broken", call_id="c1")
    _, _, result = await agent._call_tool(tool_call, {})

    assert result == "Error: boom"


# ------------------------------------------------------------------- streaming
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


# ------------------------------------------------------------------- demuxer
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


@pytest.mark.asyncio
async def test_resolve_args_ignores_unparsable_json(mock_kernel, caplog):
    agent = make_agent(ScriptedClient(), mock_kernel, RunContext(data={}))

    tool_call = ToolCall(
        name="python://test",
        literal_args="{not json}",
        planner_context_args="",
        input_args="",
    )

    assert agent._resolve_args(tool_call, {}) == {}
    assert "Failed to parse literal_args as JSON" in caplog.text


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
