import json
import typing
from uuid import UUID

import pytest
from sqlalchemy import select

from kavalai.agent_service import AgentService
from kavalai.db import DatabaseManager, Run
from kavalai.workflow import WorkflowEngine, SqliteTaskLogger
from kavalai.workflow.models import WorkflowException
from kavalai.functionkernel import pythontool
from kavalai.llm_clients.base_client import BaseLlmClient, ModelCallStat
from pydantic import BaseModel


def make_agent_service() -> AgentService:
    """AgentService over a private in-memory SQLite database, through the
    greenlet-free compat shim — the same stack the browser playground uses."""
    return AgentService(DatabaseManager().get_sqlite_compat_sessionmaker())


def _default_for(annotation):
    if annotation is str:
        return "x"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    return None


def _build(model, value_map):
    kwargs = {}
    for name, field in model.model_fields.items():
        ann = field.annotation
        if name == "tool_calls":
            kwargs[name] = []
        elif name == "output":
            inner = [a for a in typing.get_args(ann) if a is not type(None)]
            kwargs[name] = _build(inner[0], value_map) if inner else None
        elif name == "instructions":
            kwargs[name] = "done"
        elif name in value_map:
            kwargs[name] = value_map[name]
        else:
            kwargs[name] = _default_for(ann)
    return model(**kwargs)


class FakeLLMClient(BaseLlmClient):
    """Deterministic streaming client: fills response models from a value map,
    streams the JSON in two chunks through the real Streamer machinery, and
    emits one ModelCallStat per call (so the StatsBridge path is exercised)."""

    def __init__(self, model, parameters=None, stats_receiver=None, value_map=None):
        super().__init__(parameters, stats_receiver)
        self.value_map = value_map or {}
        self.calls = []

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        self.calls.append(chat_history)
        if self.model_stats_receiver is not None:
            self.model_stats_receiver.receive_model_stats(
                ModelCallStat(
                    call_type="llm",
                    model="fake",
                    total_tokens=1,
                    duration_seconds=0.0,
                )
            )
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        if response_model is not None:
            text = _build(response_model, self.value_map).model_dump_json()
            mid = max(1, len(text) // 2)
            await value_streamer.stream_partial(text[:mid])
            await value_streamer.stream_partial(text[mid:])
        await value_streamer.stream_complete()


def make_factory(value_map=None, raises=False):
    created = []

    def factory(model, parameters=None, stats_receiver=None):
        if raises:
            client = _RaisingClient(model, parameters, stats_receiver)
        else:
            client = FakeLLMClient(
                model, parameters, stats_receiver, value_map=value_map
            )
        created.append(client)
        return client

    factory.created = created
    return factory


class _RaisingClient(BaseLlmClient):
    def __init__(self, model, parameters=None, stats_receiver=None):
        super().__init__(parameters, stats_receiver)

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        raise RuntimeError("llm boom")


DATA_TYPES = {
    "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
    "classification": {
        "type": "object",
        "properties": {"intent": {"type": "string"}},
    },
    "output": {
        "type": "object",
        "properties": {"agent_response": {"type": "string"}},
    },
}


def graph_dict(nodes, **extra):
    return {
        "name": "wf",
        "description": "test workflow",
        "llm_model": "openai/fake",
        "data_types": dict(DATA_TYPES),
        "nodes": nodes,
        **extra,
    }


async def test_linear_llm_workflow_persists_everything():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "respond to {{ context.input.user_message }}",
            "inputs": {"input": {"type": "context", "value": "input"}},
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    service = make_agent_service()
    tlog = SqliteTaskLogger()
    factory = make_factory({"agent_response": "hi there"})
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        agent_service=service,
        task_logger=tlog,
        client_factory=factory,
    )
    state = await engine.run({"user_message": "hello"})

    assert state.status == "completed"
    assert state.trace == ["s", "answer", "e"]
    assert state.output_data == {"agent_response": "hi there"}

    # A short invocation id is assigned and token usage is aggregated.
    assert state.invocation_id and len(state.invocation_id) == 8
    assert state.token_usage == {
        "model_calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 1,
    }

    # The run row holds the output and the resolved run context.
    async with service.session_maker() as db:
        run = (await db.execute(select(Run))).scalars().one()
    assert str(run.id) == state.run_id
    assert run.output_data == {"agent_response": "hi there"}
    assert run.context["output"] == {"agent_response": "hi there"}

    # Chat history captured both turns.
    history = await service.get_chat_history(UUID(state.session_id))
    assert [(m.role, m.content) for m in history] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]

    # Every node visit wrote a row, and ``ORDER BY seq`` is the executed path.
    await tlog.flush()
    conn = await tlog._connect()
    async with conn.execute(
        "SELECT name, node_type, seq FROM tasks ORDER BY seq"
    ) as cur:
        tasks = [tuple(r) for r in await cur.fetchall()]
    assert tasks == [("s", "start", 0), ("answer", "llm", 1), ("e", "end", 2)]
    assert [t[0] for t in tasks] == state.trace
    async with conn.execute("SELECT count(*) FROM model_call_stats") as cur:
        assert (await cur.fetchone())[0] == 1

    await tlog.close()


@pytest.mark.parametrize(
    "message,expected_branch",
    [("hi", "yes_node"), ("bye", "no_node")],
)
async def test_if_branch_routing(message, expected_branch):
    nodes = [
        {"name": "s", "type": "start", "next": "branch"},
        {
            "name": "branch",
            "type": "if",
            "condition": "input.user_message == 'hi'",
            "then": "yes_node",
            "else": "no_node",
        },
        {
            "name": "yes_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {
            "name": "no_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": message})
    assert expected_branch in state.trace
    assert state.status == "completed"


async def test_switch_routing_and_default():
    nodes = [
        {"name": "s", "type": "start", "next": "classify"},
        {
            "name": "classify",
            "type": "llm",
            "prompt": "p",
            "output": "classification",
            "next": "route",
        },
        {
            "name": "route",
            "type": "switch",
            "expr": "classification.intent",
            "cases": {"news": "news_node"},
            "default": "chat_node",
        },
        {
            "name": "news_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {
            "name": "chat_node",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    # intent == news -> news_node
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"intent": "news", "agent_response": "r"}),
    )
    state = await engine.run({"user_message": "x"})
    assert "news_node" in state.trace and "chat_node" not in state.trace

    # unknown intent -> default chat_node
    engine2 = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"intent": "weather", "agent_response": "r"}),
    )
    state2 = await engine2.run({"user_message": "x"})
    assert "chat_node" in state2.trace and "news_node" not in state2.trace


async def test_function_node_executes_tool():
    class Greeting(BaseModel):
        agent_response: str

    @pythontool
    def greet(name: str) -> Greeting:
        return Greeting(agent_response=f"hi {name}")

    nodes = [
        {"name": "s", "type": "start", "next": "call"},
        {
            "name": "call",
            "type": "function",
            "tool": "python://greet",
            "inputs": {"name": {"type": "literal", "value": "Sam"}},
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    engine.kernel.register_python_tool("greet", greet)
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.output_data == {"agent_response": "hi Sam"}
    assert state.trace == ["s", "call", "e"]


async def test_agent_node():
    nodes = [
        {"name": "s", "type": "start", "next": "do"},
        {
            "name": "do",
            "type": "agent",
            "prompt": "do the thing",
            "output": "output",
            "max_steps": 3,
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"agent_response": "agent did it"}),
    )
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.output_data == {"agent_response": "agent did it"}


def _agent_node_graph(allowed_tools=None):
    node = {
        "name": "do",
        "type": "agent",
        "prompt": "do the thing",
        "output": "output",
        "next": "e",
    }
    if allowed_tools is not None:
        node["allowed_tools"] = allowed_tools
    return graph_dict(
        [
            {"name": "s", "type": "start", "next": "do"},
            node,
            {"name": "e", "type": "end", "output": "output"},
        ]
    )


async def _run_and_capture_allowed_tools(graph):
    """Run the graph, returning the allow-list the agent asked the kernel for."""
    engine = WorkflowEngine.from_dict(
        graph, client_factory=make_factory({"agent_response": "done"})
    )
    seen = {}
    original = engine.kernel.get_tool_descriptions

    async def spy(allowed_tools=None):
        seen["allowed_tools"] = allowed_tools
        return await original(allowed_tools)

    engine.kernel.get_tool_descriptions = spy
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    return seen["allowed_tools"]


async def test_non_chat_output_is_still_recorded_in_the_chat_history():
    """Without an ``agent_response`` field the output data is recorded instead."""
    service = make_agent_service()
    nodes = [
        {"name": "s", "type": "start", "next": "do"},
        {
            "name": "do",
            "type": "llm",
            "prompt": "extract",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    graph = graph_dict(nodes)
    graph["data_types"]["output"] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    engine = WorkflowEngine.from_dict(
        graph, agent_service=service, client_factory=make_factory({"city": "Tallinn"})
    )

    state = await engine.run({"user_message": "where?"})

    assert state.output_data == {"city": "Tallinn"}
    messages = await service.get_chat_history(UUID(state.session_id))
    assert [m.role for m in messages] == ["user", "assistant"]
    assert json.loads(messages[1].content) == {"city": "Tallinn"}


async def test_agent_node_forwards_allowed_tools():
    """A node's ``allowed_tools`` restricts the tools its agent may use."""
    allowed = await _run_and_capture_allowed_tools(
        _agent_node_graph(["python://web.crawl"])
    )
    assert allowed == ["python://web.crawl"]


async def test_agent_node_allowed_tools_reach_the_agent_unchanged():
    """Omitted means every tool; ``[]`` means none — the same as in Python."""
    assert await _run_and_capture_allowed_tools(_agent_node_graph()) is None
    assert await _run_and_capture_allowed_tools(_agent_node_graph([])) == []
    assert await _run_and_capture_allowed_tools(_agent_node_graph(["*"])) == ["*"]


async def test_invocation_id_is_unique_per_run_and_tokens_aggregate():
    nodes = [
        {"name": "s", "type": "start", "next": "a"},
        {"name": "a", "type": "llm", "prompt": "p", "output": "output", "next": "b"},
        {"name": "b", "type": "llm", "prompt": "p", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    s1 = await engine.run({"user_message": "x"})
    s2 = await engine.run({"user_message": "y"})

    # Two llm nodes -> two model calls aggregated per run.
    assert s1.token_usage["model_calls"] == 2
    assert s1.token_usage["total_tokens"] == 2
    # Each run gets its own id.
    assert s1.invocation_id != s2.invocation_id
    # Totals do not leak across runs.
    assert s2.token_usage["model_calls"] == 2


async def test_no_persistence_or_logger_still_runs():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.run_id is None  # no agent_service -> no ids


async def test_cycle_guard():
    # if-node that always loops back to itself never reaches the end.
    nodes = [
        {"name": "s", "type": "start", "next": "loop"},
        {
            "name": "loop",
            "type": "if",
            "condition": "True",
            "then": "loop",
            "else": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory(), max_node_visits=10
    )
    with pytest.raises(WorkflowException, match="max node visits"):
        await engine.run({"user_message": "x"})


async def test_switch_no_match_no_default_halts():
    nodes = [
        {"name": "s", "type": "start", "next": "route"},
        {
            "name": "route",
            "type": "switch",
            "expr": "input.user_message",
            "cases": {"hi": "e"},
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    with pytest.raises(WorkflowException, match="no next node"):
        await engine.run({"user_message": "bye"})


async def test_failure_marks_state_failed_and_persists():
    nodes = [
        {"name": "s", "type": "start", "next": "boom"},
        {
            "name": "boom",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    service = make_agent_service()
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        agent_service=service,
        client_factory=make_factory(raises=True),
    )
    with pytest.raises(WorkflowException, match="llm boom"):
        await engine.run({"user_message": "x"})

    # The failure was recorded on the run row.
    async with service.session_maker() as db:
        run = (await db.execute(select(Run))).scalars().one()
    assert run.context["status"] == "failed"
    assert "llm boom" in run.context["error"]


async def test_from_yaml_invalid_raises():
    bad_yaml = """
name: bad
data_types:
  input:
    type: object
nodes:
  - {name: s, type: start, next: ghost}
  - {name: e, type: end}
"""
    with pytest.raises(WorkflowException, match="validation failed"):
        WorkflowEngine.from_yaml(bad_yaml)


def test_resolve_model_missing_raises():
    nodes = [
        {"name": "s", "type": "start", "next": "n"},
        {"name": "n", "type": "llm", "prompt": "p", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = dict(graph_dict(nodes))
    g.pop("llm_model")
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    with pytest.raises(WorkflowException, match="No LLM model configured"):
        engine._resolve_model(None)


def test_resolve_model_falls_back_to_the_engine_default():
    """The engine reads no environment variable; the server passes the default."""
    nodes = [
        {"name": "s", "type": "start", "next": "n"},
        {"name": "n", "type": "llm", "prompt": "p", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = dict(graph_dict(nodes))
    g.pop("llm_model")
    engine = WorkflowEngine.from_dict(
        g, client_factory=make_factory(), default_llm_model="openai/fleet"
    )
    assert engine._resolve_model(None) == "openai/fleet"
    assert engine._resolve_model("openai/node") == "openai/node"


def test_default_llm_parameters_sit_below_graph_and_node():
    """node > graph > engine defaults > provider defaults."""
    from kavalai.run_context import RunContext

    seen = {}

    def factory(model, parameters, stats_receiver):
        seen["parameters"] = parameters
        return make_factory()(model, parameters, stats_receiver)

    nodes = [
        {"name": "s", "type": "start", "next": "n"},
        {
            "name": "n",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
            "llm_kwargs": {"top_p": 0.1},
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = dict(graph_dict(nodes))
    g["llm_kwargs"] = {"temperature": 0.5}
    engine = WorkflowEngine.from_dict(
        g,
        client_factory=factory,
        default_llm_parameters={
            "temperature": 0.9,
            "top_p": 0.9,
            "timeout_seconds": 7.0,
        },
    )
    node = engine.node_map["n"]
    engine._make_llm_client(node.llm_model, node.llm_kwargs, RunContext())
    parameters = seen["parameters"]
    assert parameters.temperature == 0.5  # graph beats the engine default
    assert parameters.top_p == 0.1  # node beats both
    assert parameters.timeout_seconds == 7.0  # nobody else set it


async def test_use_history_includes_prior_messages():
    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
            "use_history": True,
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    service = make_agent_service()
    factory = make_factory({"agent_response": "r"})
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), agent_service=service, client_factory=factory
    )
    # Seed a session with a prior message, then run on the same session.
    agent, session, run = await service.initialize_workflow_run(agent_name="wf")
    await service.add_chat_message(
        agent_id=agent.id,
        session_id=session.id,
        run_id=run.id,
        role="user",
        content="earlier",
    )
    await engine.run({"user_message": "now"}, session_id=str(session.id))

    # The LLM client received the seeded history in its chat_history.
    client = factory.created[0]
    contents = [m.content for m in client.calls[0].messages]
    assert any(c == "earlier" for c in contents)


# The exact workflow the standalone chat-playground.html builds, with the
# in-browser model swapped for the test fake (browser/... only runs in Pyodide).
CHAT_WORKFLOW_YAML = """
name: Browser chat
description: A friendly assistant that remembers the conversation.
llm_model: openai/fake

data_types:
  input:
    type: object
    properties:
      user_message:
        type: string
  output:
    type: object
    properties:
      agent_response:
        type: string

nodes:
  - name: begin
    type: start
    next: reply
  - name: reply
    type: llm
    use_history: true
    prompt: >
      You are a warm, concise assistant chatting with a user. Answer their
      latest message and put your reply in the agent_response field.
    inputs:
      input:
        type: context
        value: input
    output: output
    next: done
  - name: done
    type: end
    output: output
"""


async def test_chat_workflow_remembers_history_in_memory():
    """End-to-end chat-playground scenario over in-memory SQLite via the
    greenlet-free compat shim: a stable client-supplied external id pins the
    session, so each turn's LLM call sees the whole prior conversation
    (no seeding — the engine accumulates it).
    """
    service = make_agent_service()
    factory = make_factory({"agent_response": "ack"})
    engine = WorkflowEngine.from_yaml(
        CHAT_WORKFLOW_YAML, agent_service=service, client_factory=factory
    )
    chat_id = "chat-1"

    session_ids = set()
    for msg in ["hello", "what did I say?", "and before that?"]:
        state = await engine.run({"user_message": msg}, external_id=chat_id)
        assert state.status == "completed"
        assert state.output_data == {"agent_response": "ack"}
        session_ids.add(state.session_id)

    # Every turn ran under the same session, pinned by the external id.
    assert len(session_ids) == 1

    # Turn 3's LLM call must include every earlier user + assistant message.
    third_call = factory.created[2].calls[0]
    contents = [m.content for m in third_call.messages]
    assert "hello" in contents
    assert "what did I say?" in contents
    assert contents.count("ack") == 2  # the two prior assistant replies

    # Persisted history holds all six messages, oldest first.
    history = await service.get_chat_history(UUID(session_ids.pop()))
    assert [(m.role, m.content) for m in history] == [
        ("user", "hello"),
        ("assistant", "ack"),
        ("user", "what did I say?"),
        ("assistant", "ack"),
        ("user", "and before that?"),
        ("assistant", "ack"),
    ]


def test_servers_and_unmarked_python_function_registration():
    # Covers rest/mcp server registration and the pythontool() wrap of an
    # undecorated function in WorkflowEngine.__init__.
    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = graph_dict(
        nodes,
        rest_servers=[{"name": "api", "url": "http://localhost:9999"}],
        mcp_servers=[{"name": "m", "command": "echo"}],
        python_functions=[
            # clean_text is NOT decorated with @pythontool -> exercises the wrap.
            {"name": "clean_text", "path": "kavalai.utils.clean_text"}
        ],
    )
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    assert "clean_text" in engine.kernel.python_tools
    assert "api" in engine.kernel.rest_servers
    assert "m" in engine.kernel.mcp_servers


def test_make_prompt_with_basemodel():
    from kavalai.workflow.engine import make_prompt

    class Payload(BaseModel):
        v: int

    text = make_prompt("base", {"p": Payload(v=1), "q": "lit"})
    assert "INPUT DATA:" in text
    assert '"v":1' in text  # BaseModel serialized as JSON
    assert "q:lit" in text


def test_get_data_type_none():
    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    assert engine.get_data_type(None) is None
    assert engine.get_data_type("output") is not None


def test_next_node_end_returns_none():
    from kavalai.run_context import RunContext

    nodes = [
        {"name": "s", "type": "start", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=make_factory())
    end_node = engine.node_map["e"]
    assert engine._next_node(end_node, RunContext()) == (None, None)


def test_from_yaml_and_from_yaml_path(tmp_path):
    yaml_text = """
name: yamlwf
description: y
llm_model: openai/fake
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
nodes:
  - {name: s, type: start, next: e}
  - {name: e, type: end, output: output}
"""
    engine = WorkflowEngine.from_yaml(yaml_text, client_factory=make_factory())
    assert engine.graph.name == "yamlwf"

    path = tmp_path / "wf.yaml"
    path.write_text(yaml_text)
    engine2 = WorkflowEngine.from_yaml_path(str(path), client_factory=make_factory())
    assert engine2.graph.name == "yamlwf"


def test_from_dict_invalid_raises():
    bad = {
        "name": "bad",
        "data_types": {"input": {"type": "object"}},
        "nodes": [
            {"name": "s", "type": "start", "next": "ghost"},
            {"name": "e", "type": "end"},
        ],
    }
    with pytest.raises(WorkflowException, match="validation failed"):
        WorkflowEngine.from_dict(bad)


async def test_rest_function_node_passes_method():
    from unittest.mock import AsyncMock

    nodes = [
        {"name": "s", "type": "start", "next": "call"},
        {
            "name": "call",
            "type": "function",
            "tool": "rest://api.do",
            "output": "output",
            "method": "post",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    g = graph_dict(nodes, rest_servers=[{"name": "api", "url": "http://localhost"}])
    engine = WorkflowEngine.from_dict(g, client_factory=make_factory())
    out_model = engine.get_data_type("output")
    engine.kernel.call_tool = AsyncMock(return_value=out_model(agent_response="ok"))

    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    # The REST method was forwarded to the kernel call.
    _, kwargs = engine.kernel.call_tool.call_args
    assert kwargs["method"] == "post"
    assert kwargs["tool_uri"] == "rest://api.do"


async def test_engine_data_models_override_parsed_types():
    """data_models supplies ready-made Pydantic models that the engine uses
    verbatim, skipping the SchemaParser for those names while still parsing the
    rest."""
    from pydantic import BaseModel

    class Output(BaseModel):
        agent_response: str

    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "inputs": {"input": {"type": "context", "value": "input"}},
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"agent_response": "hi"}),
        data_models={"output": Output},
    )
    # The provided model is used as-is; "input" is still parser-compiled.
    assert engine.get_data_type("output") is Output
    assert engine.get_data_type("input").__name__ == "input"

    state = await engine.run({"user_message": "hello"})
    assert state.output_data == {"agent_response": "hi"}


STREAM_NODES = [
    {"name": "s", "type": "start", "next": "answer"},
    {
        "name": "answer",
        "type": "llm",
        "prompt": "p",
        "inputs": {"input": {"type": "context", "value": "input"}},
        "output": "output",
        "next": "e",
        "stream_output": True,
    },
    {"name": "e", "type": "end", "output": "output"},
]


async def collect_stream(engine, input_data):
    events = []
    async for event in engine.run_stream(input_data):
        events.append(event)
    return events


async def test_run_stream_llm_event_sequence():
    service = make_agent_service()
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        agent_service=service,
        client_factory=make_factory({"agent_response": "hi there"}),
    )
    events = await collect_stream(engine, {"user_message": "hello"})

    types = [(e.type, e.name) for e in events]
    assert types == [
        ("workflow_started", "wf"),
        ("node_started", "s"),
        ("node_completed", "s"),
        ("node_started", "answer"),
        ("partial", "answer"),
        ("partial", "answer"),
        ("complete", "answer"),
        ("node_completed", "answer"),
        ("node_started", "e"),
        ("node_completed", "e"),
        ("workflow_completed", "wf"),
    ]
    # Lifecycle payloads.
    started = events[0]
    assert started.session_id and started.run_id
    completed = events[-1]
    assert completed.output_data == {"agent_response": "hi there"}
    assert completed.token_usage["model_calls"] == 1
    assert completed.session_id == started.session_id
    # Full-buffer mode: the complete event carries the full JSON.
    assert events[6].value == '{"agent_response": "hi there"}'


async def test_run_stream_without_stream_output_has_no_content_events():
    nodes = [dict(n) for n in STREAM_NODES]
    nodes[1] = {**nodes[1], "stream_output": False}
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    events = await collect_stream(engine, {"user_message": "x"})
    assert all(e.type not in ("partial", "complete") for e in events)
    assert events[-1].type == "workflow_completed"
    assert events[-1].output_data == {"agent_response": "r"}


async def test_run_stream_delta_mode_emits_deltas():
    nodes = [dict(n) for n in STREAM_NODES]
    nodes[1] = {**nodes[1], "stream_delta": True}
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "hi"})
    )
    events = await collect_stream(engine, {"user_message": "x"})
    partials = [e for e in events if e.type == "partial"]
    # Deltas reassemble to the full JSON; the complete chunk carries no value.
    assert "".join(p.value for p in partials) == '{"agent_response":"hi"}'
    completes = [e for e in events if e.type == "complete"]
    assert completes[0].value is None
    # The engine still parsed the output from its own delta buffer.
    assert events[-1].output_data == {"agent_response": "hi"}


class _RestartingClient(BaseLlmClient):
    """Streams garbage, signals a restart (as a retry would), then re-streams."""

    def __init__(self, model=None, parameters=None, stats_receiver=None):
        super().__init__(parameters, stats_receiver)

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        vs = streamer.get_value_streamer("response", response_model=response_model)
        await vs.stream_partial('{"agent_response": "jun')
        streamer.reset_active()
        await streamer.stream_restart("attempt 1: transient")
        vs2 = streamer.get_value_streamer("response", response_model=response_model)
        await vs2.stream_partial('{"agent_response": "ok"}')
        await vs2.stream_complete()


async def test_run_stream_restart_resets_delta_buffer():
    nodes = [dict(n) for n in STREAM_NODES]
    nodes[1] = {**nodes[1], "stream_delta": True}
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=lambda *a, **k: _RestartingClient()
    )
    events = await collect_stream(engine, {"user_message": "x"})
    restarts = [e for e in events if e.type == "restart"]
    assert [e.name for e in restarts] == ["answer"]
    assert "attempt 1" in restarts[0].value
    # The pre-restart garbage was discarded; only the re-sent value survives.
    assert events[-1].output_data == {"agent_response": "ok"}


async def test_run_stream_failure_yields_workflow_failed_then_raises():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory(raises=True)
    )
    events = []
    with pytest.raises(WorkflowException, match="llm boom"):
        async for event in engine.run_stream({"user_message": "x"}):
            events.append(event)
    assert events[-1].type == "workflow_failed"
    assert "llm boom" in events[-1].value


async def test_run_stream_agent_node_streams_instructions_and_output():
    nodes = [
        {"name": "s", "type": "start", "next": "do"},
        {
            "name": "do",
            "type": "agent",
            "prompt": "do the thing",
            "output": "output",
            "max_steps": 3,
            "next": "e",
            "stream_output": True,
            "stream_instructions": True,
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=make_factory({"agent_response": "agent did it"}),
    )
    events = await collect_stream(engine, {"user_message": "x"})

    # Instructions stream under <node>_instructions and complete per step.
    instr = [e for e in events if e.name == "do_instructions"]
    assert instr and instr[-1].type == "complete"
    assert instr[-1].value == "done"
    # The output field streams under the node name; complete is authoritative.
    out = [e for e in events if e.name == "do" and e.type in ("partial", "complete")]
    assert any(e.type == "partial" for e in out)
    assert out[-1].type == "complete"
    assert events[-1].output_data == {"agent_response": "agent did it"}


async def test_run_stream_abort_records_failure():
    service = make_agent_service()
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        agent_service=service,
        client_factory=make_factory({"agent_response": "r"}),
    )
    stream = engine.run_stream({"user_message": "x"})
    started = await stream.__anext__()
    assert started.type == "workflow_started"
    # Simulate the SSE client disconnecting mid-run.
    await stream.aclose()

    async with service.session_maker() as db:
        run = (await db.execute(select(Run))).scalars().one()
    assert run.context["status"] == "failed"
    assert "aborted" in run.context["error"]


def test_parse_streamed_output_edge_cases():
    assert WorkflowEngine._parse_streamed_output(None, None, raw_text=False) is None
    assert WorkflowEngine._parse_streamed_output(None, "raw", raw_text=False) == "raw"

    class Out(BaseModel):
        agent_response: str

    parsed = WorkflowEngine._parse_streamed_output(
        Out, '{"agent_response": "x"}', raw_text=True
    )
    assert parsed.agent_response == "x"


async def test_run_stream_abort_survives_recording_failure(monkeypatch):
    """A failing abort-recording is logged, not raised, during teardown."""
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        agent_service=make_agent_service(),
        client_factory=make_factory({"agent_response": "r"}),
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(engine, "_record_failure", boom)
    stream = engine.run_stream({"user_message": "x"})
    assert (await stream.__anext__()).type == "workflow_started"
    await stream.aclose()  # must not raise


async def test_run_matches_run_stream_result():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"
    assert state.output_data == {"agent_response": "r"}
    assert state.trace == ["s", "answer", "e"]


async def test_concurrent_runs_report_their_own_token_usage():
    """One engine, four overlapping runs: each reports only its own model call.

    The accumulator used to live on the engine and be reset per run, so runs
    that overlapped shared a counter and each reported whatever total happened
    to be there when it finished. ``kavalai.server`` serves one engine to every
    request, so this was the normal case under load, not an edge case.
    """
    import asyncio

    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory({"agent_response": "r"})
    )
    states = await asyncio.gather(
        *(engine.run({"user_message": f"m{i}"}) for i in range(4))
    )

    assert [s.status for s in states] == ["completed"] * 4
    for state in states:
        assert state.token_usage["model_calls"] == 1
        assert state.token_usage["total_tokens"] == 1


async def test_parallel_branches_share_one_run_total():
    """Branches of a parallel node are one run, so their calls land in one total."""
    nodes = [
        {"name": "s", "type": "start", "next": "fan"},
        {
            "name": "fan",
            "type": "parallel",
            "branches": ["a", "b"],
            "next": "e",
        },
        {
            "name": "a",
            "type": "llm",
            "prompt": "a",
            "output": "classification",
            "next": "e",
        },
        {
            "name": "b",
            "type": "llm",
            "prompt": "b",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state = await engine.run({"user_message": "x"})

    assert state.status == "completed"
    assert state.token_usage["model_calls"] == 2


async def test_kernel_survives_a_run_and_closes_with_the_engine():
    """Tool servers belong to the engine: a finished run must not close them."""
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory({"agent_response": "r"})
    )
    closed = []

    async def record_close():
        closed.append(True)

    engine.kernel.close = record_close

    await engine.run({"user_message": "x"})
    assert closed == []

    await engine.aclose()
    assert closed == [True]


async def test_engine_connect_opens_mcp_servers():
    """``connect()`` reaches the kernel so tools are known before the first run."""
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory({"agent_response": "r"})
    )
    connected = []

    async def record_connect():
        connected.append(True)

    engine.kernel.connect_mcp_servers = record_connect

    async with engine:
        assert connected == [True]


async def test_agent_node_allowed_tools_are_passed_through_verbatim():
    """`[]` means no tools in YAML too — it used to mean "all of them".

    The node default was an empty list and the engine turned a falsy value into
    ``None``, so a YAML author had no way to say "this node gets no tools" and
    the same value meant opposite things in YAML and Python.
    """
    from kavalai.workflow.models import AgentNode

    node_default = AgentNode(name="a", prompt="p", output="output", next="e")
    assert node_default.allowed_tools is None

    node_none = AgentNode(
        name="a", prompt="p", output="output", next="e", allowed_tools=[]
    )
    assert node_none.allowed_tools == []


def test_tool_allow_list_understands_the_wildcard():
    from kavalai.functionkernel import _is_tool_allowed

    assert _is_tool_allowed("python://anything", ["*"])
    assert _is_tool_allowed("mcp://github.create_issue", ["mcp://github.*"])
    assert _is_tool_allowed("python://exact", ["python://exact"])
    assert not _is_tool_allowed("python://anything", [])
    assert _is_tool_allowed("python://anything", None)


class RecordingRagService:
    """A stand-in that records calls and refuses anything but ``query``.

    ``rag_query`` is read-only by construction, so the write side is defined
    here to explode rather than being merely absent: the test then proves the
    node never reaches for it, instead of the attribute quietly not existing.
    """

    def __init__(self, hits=None):
        self.hits = hits if hits is not None else []
        self.calls = []

    async def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.hits

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("rag_query must only ever call query()")

    index = index_batch = delete = delete_by_source_id = _forbidden


def rag_hit(content, source_id="s", similarity=0.9):
    from kavalai.rag import RagServiceResult

    return RagServiceResult(
        id=UUID(int=abs(hash(content)) % (2**128)),
        model="fake/embedding",
        collection_name="default",
        source_id=source_id,
        content=content,
        embedding_size=4,
        rag_metadata={},
        similarity=similarity,
    )


def rag_graph(**node_extra):
    """The four-line case: a rag_query node that names nothing."""
    nodes = [
        {"name": "s", "type": "start", "next": "retrieve"},
        {
            "name": "retrieve",
            "type": "rag_query",
            "query": "{{ context.input.user_message }}",
            "output": "docs",
            "next": "e",
            **node_extra,
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    return graph_dict(nodes)


async def test_rag_query_needs_no_configuration_when_there_is_one_service():
    service = RecordingRagService([rag_hit("Green Village has 104 residents.")])
    engine = WorkflowEngine.from_dict(rag_graph(), rag_services=service)

    state = await engine.run({"user_message": "How many residents?"})

    assert service.calls[0]["text"] == "How many residents?"
    # state.data is to_plain(run_context.data), so hits arrive as dicts here
    # while nodes downstream still see RagServiceResult models.
    assert state.data["docs"][0]["content"] == "Green Village has 104 residents."


async def test_rag_query_output_need_not_be_declared_in_data_types():
    # 'docs' is nowhere in DATA_TYPES; the node owns its own output shape.
    assert "docs" not in DATA_TYPES
    WorkflowEngine.from_dict(rag_graph(), rag_services=RecordingRagService())


async def test_rag_query_only_ever_calls_query():
    service = RecordingRagService([rag_hit("a fact")])
    engine = WorkflowEngine.from_dict(rag_graph(), rag_services=service)

    await engine.run({"user_message": "q"})

    assert len(service.calls) == 1


async def test_rag_query_passes_its_parameters_through():
    service = RecordingRagService()
    engine = WorkflowEngine.from_dict(
        rag_graph(top_k=3, source_ids=["policies"], keep_best=True),
        rag_services=service,
    )

    await engine.run({"user_message": "q"})

    call = service.calls[0]
    assert call["top_k"] == 3
    assert call["source_ids"] == ["policies"]
    assert call["keep_best"] is True


async def test_rag_query_renders_templates_in_the_query():
    service = RecordingRagService()
    graph = rag_graph()
    graph["templates"] = [{"name": "suffix", "value": "in Green Village"}]
    graph["nodes"][1]["query"] = (
        "{{ context.input.user_message }} {{ templates.suffix }}"
    )
    engine = WorkflowEngine.from_dict(graph, rag_services=service)

    await engine.run({"user_message": "the bakery"})

    assert service.calls[0]["text"] == "the bakery in Green Village"


async def test_store_content_joins_the_hit_texts():
    service = RecordingRagService([rag_hit("fact one"), rag_hit("fact two")])
    engine = WorkflowEngine.from_dict(rag_graph(store="content"), rag_services=service)

    state = await engine.run({"user_message": "q"})

    assert state.data["docs"] == "fact one\n\nfact two"


async def test_store_results_keeps_scores_reachable():
    service = RecordingRagService([rag_hit("fact", similarity=0.42)])
    engine = WorkflowEngine.from_dict(rag_graph(), rag_services=service)

    state = await engine.run({"user_message": "q"})

    assert state.data["docs"][0]["similarity"] == 0.42


async def test_retrieved_content_reaches_a_following_prompt():
    """The point of the node: hits land in the next node's prompt."""
    service = RecordingRagService([rag_hit("Lake Miller is 1.2 meters deep.")])
    nodes = [
        {"name": "s", "type": "start", "next": "retrieve"},
        {
            "name": "retrieve",
            "type": "rag_query",
            "query": "{{ context.input.user_message }}",
            "output": "docs",
            "next": "answer",
            "store": "content",
        },
        {
            "name": "answer",
            "type": "llm",
            "prompt": "Facts:\n{{ context.docs }}",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    seen = {}

    class CapturingClient(BaseLlmClient):
        async def _run_chat_completions(self, chat_history, response_model, streamer):
            seen["prompt"] = chat_history.messages[0].content
            vs = streamer.get_value_streamer("response", response_model=response_model)
            await vs.stream_partial('{"agent_response": "ok"}')
            await vs.stream_complete()

    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=lambda *a, **k: CapturingClient(),
        rag_services=service,
    )
    await engine.run({"user_message": "how deep is the pond"})

    assert "Lake Miller is 1.2 meters deep." in seen["prompt"]


async def test_node_service_beats_workflow_default():
    node_service, workflow_service = RecordingRagService(), RecordingRagService()
    graph = rag_graph(service="special")
    graph["rag_service"] = "wide"
    engine = WorkflowEngine.from_dict(
        graph, rag_services={"special": node_service, "wide": workflow_service}
    )

    await engine.run({"user_message": "q"})

    assert node_service.calls and not workflow_service.calls


async def test_workflow_service_beats_default():
    named, fallback = RecordingRagService(), RecordingRagService()
    graph = rag_graph()
    graph["rag_service"] = "wide"
    engine = WorkflowEngine.from_dict(
        graph, rag_services={"wide": named, "default": fallback}
    )

    await engine.run({"user_message": "q"})

    assert named.calls and not fallback.calls


async def test_collection_precedence_matches_service_precedence():
    service = RecordingRagService()
    graph = rag_graph(collection="handbook")
    graph["rag_collection"] = "wide"
    engine = WorkflowEngine.from_dict(graph, rag_services=service)

    await engine.run({"user_message": "q"})

    assert service.calls[0]["collection_name"] == "handbook"


async def test_workflow_collection_is_used_when_the_node_is_silent():
    service = RecordingRagService()
    graph = rag_graph()
    graph["rag_collection"] = "handbook"
    engine = WorkflowEngine.from_dict(graph, rag_services=service)

    await engine.run({"user_message": "q"})

    assert service.calls[0]["collection_name"] == "handbook"


async def test_passed_service_beats_a_registered_one_of_the_same_name():
    from kavalai.llm_clients import registry

    registered = RecordingRagService()
    registry.register_rag_service("default", lambda: registered)
    try:
        passed = RecordingRagService()
        engine = WorkflowEngine.from_dict(rag_graph(), rag_services=passed)
        await engine.run({"user_message": "q"})

        assert passed.calls and not registered.calls
    finally:
        registry.rag_services.unregister("default")


async def test_a_registered_service_is_found_when_none_is_passed():
    """The deployment path: nobody constructs the engine, so nothing is passed."""
    from kavalai.llm_clients import registry

    service = RecordingRagService([rag_hit("registered")])
    registry.register_rag_service("default", lambda: service)
    try:
        engine = WorkflowEngine.from_dict(rag_graph())
        state = await engine.run({"user_message": "q"})

        assert state.data["docs"][0]["content"] == "registered"
    finally:
        registry.rag_services.unregister("default")


async def test_unknown_service_fails_at_load_not_at_execution():
    with pytest.raises(WorkflowException) as error:
        WorkflowEngine.from_dict(rag_graph(service="nosuch"))

    message = str(error.value)
    assert "nosuch" in message
    assert "register_rag_service()" in message


async def _trajectory(engine, input_data=None):
    """Run once with a private in-memory logger and return its records."""
    from kavalai.workflow.tasklog import MemoryTaskLogger

    tlog = MemoryTaskLogger()
    state = await engine.run(input_data or {"user_message": "hi"}, task_logger=tlog)
    return state, tlog.records


async def test_every_node_visit_writes_one_row_in_order():
    """``ORDER BY seq`` is the executed path, not the subset that did something."""
    nodes = [
        {"name": "s", "type": "start", "next": "branch"},
        {
            "name": "branch",
            "type": "if",
            "condition": "input.user_message == 'hi'",
            "then": "yes_node",
            "else": "no_node",
        },
        {
            "name": "yes_node",
            "type": "llm",
            "prompt": "y",
            "output": "output",
            "next": "e",
        },
        {
            "name": "no_node",
            "type": "llm",
            "prompt": "n",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    state, records = await _trajectory(engine)

    assert [r.name for r in records] == ["s", "branch", "yes_node", "e"]
    assert [r.seq for r in records] == [0, 1, 2, 3]
    # The trace the state reports and the rows the run wrote agree.
    assert [r.name for r in records] == state.trace


async def test_if_branch_records_the_value_it_routed_on():
    nodes = [
        {"name": "s", "type": "start", "next": "branch"},
        {
            "name": "branch",
            "type": "if",
            "condition": "input.user_message == 'hi'",
            "then": "yes_node",
            "else": "no_node",
        },
        {
            "name": "yes_node",
            "type": "llm",
            "prompt": "y",
            "output": "output",
            "next": "e",
        },
        {
            "name": "no_node",
            "type": "llm",
            "prompt": "n",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    _, records = await _trajectory(engine, {"user_message": "bye"})

    branch = next(r for r in records if r.name == "branch")
    assert branch.node_type == "if"
    assert branch.inputs == {"expr": "input.user_message == 'hi'", "value": False}
    assert branch.output == {"taken": "no_node", "matched": True}


async def test_switch_records_the_unmatched_label_that_fell_to_default(caplog):
    """A model returning a label outside the enum is the usual mis-route cause."""
    nodes = [
        {"name": "s", "type": "start", "next": "route"},
        {
            "name": "route",
            "type": "switch",
            "expr": "input.user_message",
            "cases": {"order": "handle"},
            "default": "handle",
        },
        {
            "name": "handle",
            "type": "llm",
            "prompt": "h",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    _, records = await _trajectory(engine, {"user_message": "Order "})

    route = next(r for r in records if r.name == "route")
    assert route.inputs == {"expr": "input.user_message", "value": "Order "}
    assert route.output == {"taken": "handle", "matched": False}


async def test_function_node_row_carries_its_tool_uri():
    @pythontool
    def echo(text: str) -> str:
        """Echo."""
        return text

    nodes = [
        {"name": "s", "type": "start", "next": "call"},
        {
            "name": "call",
            "type": "function",
            "tool": "python://echo",
            "inputs": {"text": {"type": "literal", "value": "hi"}},
            "output": "classification",
            "next": "answer",
        },
        {
            "name": "answer",
            "type": "llm",
            "prompt": "a",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    engine.kernel.register_python_tool("echo", echo)
    _, records = await _trajectory(engine)

    call = next(r for r in records if r.name == "call")
    assert call.tool_uri == "python://echo"


async def test_parallel_branches_draw_from_one_sequence():
    """One counter per run, so the interleaving is recorded rather than lost."""
    nodes = [
        {"name": "s", "type": "start", "next": "fan"},
        {"name": "fan", "type": "parallel", "branches": ["a", "b"], "next": "e"},
        {
            "name": "a",
            "type": "llm",
            "prompt": "a",
            "output": "classification",
            "next": "e",
        },
        {"name": "b", "type": "llm", "prompt": "b", "output": "output", "next": "e"},
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    _, records = await _trajectory(engine)

    assert {r.name for r in records} == {"s", "fan", "a", "b", "e"}
    # A gapless permutation: every visit took exactly one number.
    assert sorted(r.seq for r in records) == list(range(len(records)))
    fan = next(r for r in records if r.name == "fan")
    assert fan.output == {"branches": ["a", "b"]}
    # The fan row sorts ahead of everything its branches wrote.
    assert fan.seq < min(r.seq for r in records if r.name in {"a", "b"})


async def test_agent_tool_calls_become_child_rows():
    """The gap this closes: what the agent chose, not just what it answered."""
    from kavalai.llm_clients.streamer import StreamContent

    @pythontool
    def lookup(order_id: str) -> str:
        """Look an order up."""
        return "found"

    class ToolCallingClient(BaseLlmClient):
        """Calls ``lookup`` on step 0, answers on step 1."""

        def __init__(self, model, parameters=None, stats_receiver=None):
            super().__init__(parameters, stats_receiver)
            self.step = 0

        async def stream_chat_completions(
            self, chat_history, response_model=None, **kw
        ):
            if self.step == 0:
                payload = {
                    "instructions": "look it up",
                    "tool_calls": [
                        {
                            "name": "python://lookup",
                            "call_id": "c1",
                            "literal_args": '{"order_id": "4471"}',
                        }
                    ],
                    "output": None,
                }
            else:
                payload = {
                    "instructions": "answer",
                    "tool_calls": [],
                    "output": {"agent_response": "found it"},
                }
            self.step += 1

            async def gen():
                yield StreamContent(
                    type="partial", name="response", value=json.dumps(payload)
                )
                yield StreamContent(type="complete", name="response", value=None)

            return gen()

    nodes = [
        {"name": "s", "type": "start", "next": "research"},
        {
            "name": "research",
            "type": "agent",
            "prompt": "find order 4471",
            "output": "output",
            "max_steps": 3,
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes),
        client_factory=lambda model, params=None, stats=None: ToolCallingClient(
            model, params, stats
        ),
    )
    engine.kernel.register_python_tool("lookup", lookup)
    _, records = await _trajectory(engine)

    assert [r.name for r in records] == ["s", "research", "lookup", "e"]
    call = next(r for r in records if r.node_type == "tool_call")
    assert call.tool_uri == "python://lookup"
    assert call.parent_task_name == "research"
    assert call.inputs["args"] == {"order_id": "4471"}
    assert call.inputs["step"] == 0
    # A bare (non-Pydantic) tool result is wrapped as ``.result`` by the kernel.
    assert call.output == {"result": "found"}
    assert call.duration_seconds >= 0.0
    # The child row sorts after its node and before the next one.
    research = next(r for r in records if r.name == "research")
    end = next(r for r in records if r.name == "e")
    assert research.seq < call.seq < end.seq


async def test_per_run_logger_isolates_concurrent_runs():
    """One engine, two runs, two private trajectories — no run_id filtering."""
    import asyncio

    from kavalai.workflow.tasklog import MemoryTaskLogger

    nodes = [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "a",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=make_factory({"agent_response": "r"})
    )
    first, second = MemoryTaskLogger(), MemoryTaskLogger()
    await asyncio.gather(
        engine.run({"user_message": "a"}, task_logger=first),
        engine.run({"user_message": "b"}, task_logger=second),
    )

    assert [r.name for r in first.records] == ["s", "answer", "e"]
    assert [r.name for r in second.records] == ["s", "answer", "e"]


class _SlowClient(BaseLlmClient):
    """Does not answer within any test's patience, so a run can time out."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        import asyncio

        await asyncio.sleep(5)


class _RestartingAgentClient(FakeLLMClient):
    """An agent-step client that signals one restart before it answers."""

    async def _run_chat_completions(self, chat_history, response_model, streamer):
        streamer.reset_active()
        await streamer.stream_restart("attempt 1: transient")
        await super()._run_chat_completions(chat_history, response_model, streamer)


def _llm_nodes(**llm_extra):
    return [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "p",
            "output": "output",
            "next": "e",
            **llm_extra,
        },
        {"name": "e", "type": "end", "output": "output"},
    ]


async def _seeded_session(service, *contents):
    agent, session, run = await service.initialize_workflow_run(agent_name="wf")
    for content in contents:
        await service.add_chat_message(
            agent_id=agent.id,
            session_id=session.id,
            run_id=run.id,
            role="user",
            content=content,
        )
    return session


def _history_seen(factory) -> list[str]:
    """The messages after the system prompt that the first client received."""
    return [m.content for m in factory.created[0].calls[0].messages[1:]]


async def _only_run(service) -> Run:
    async with service.session_maker() as db:
        return (await db.execute(select(Run))).scalars().one()


async def test_history_limit_keeps_the_most_recent_messages():
    service = make_agent_service()
    factory = make_factory({"agent_response": "r"})
    engine = WorkflowEngine.from_dict(
        graph_dict(_llm_nodes(history_limit=2)),
        agent_service=service,
        client_factory=factory,
    )
    session = await _seeded_session(service, "one", "two", "three")

    await engine.run({"user_message": "now"}, session_id=str(session.id))

    assert _history_seen(factory) == ["three", "now"]


async def test_history_max_chars_drops_whole_messages_from_the_oldest_end():
    service = make_agent_service()
    factory = make_factory({"agent_response": "r"})
    engine = WorkflowEngine.from_dict(
        graph_dict(_llm_nodes(history_max_chars=5)),
        agent_service=service,
        client_factory=factory,
    )
    session = await _seeded_session(service, "aaaa", "bb")

    await engine.run({"user_message": "now"}, session_id=str(session.id))

    assert _history_seen(factory) == ["bb", "now"]


async def test_history_limit_zero_sends_no_history():
    service = make_agent_service()
    factory = make_factory({"agent_response": "r"})
    engine = WorkflowEngine.from_dict(
        graph_dict(_llm_nodes(history_limit=0)),
        agent_service=service,
        client_factory=factory,
    )
    session = await _seeded_session(service, "earlier")

    await engine.run({"user_message": "now"}, session_id=str(session.id))

    assert _history_seen(factory) == []


async def test_a_node_that_does_not_stream_emits_no_restart():
    nodes = [dict(n) for n in STREAM_NODES]
    nodes[1] = {**nodes[1], "stream_output": False, "stream_delta": True}
    engine = WorkflowEngine.from_dict(
        graph_dict(nodes), client_factory=lambda *a, **k: _RestartingClient()
    )

    events = await collect_stream(engine, {"user_message": "x"})

    assert [e for e in events if e.type == "restart"] == []
    # The delta buffer is still reset, so only the re-sent value survives.
    assert events[-1].output_data == {"agent_response": "ok"}


async def test_an_agent_node_that_streams_nothing_emits_no_restart():
    nodes = [
        {"name": "s", "type": "start", "next": "do"},
        {
            "name": "do",
            "type": "agent",
            "prompt": "do the thing",
            "output": "output",
            "max_steps": 2,
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ]

    def factory(model, parameters=None, stats_receiver=None):
        return _RestartingAgentClient(
            model, parameters, stats_receiver, value_map={"agent_response": "done"}
        )

    engine = WorkflowEngine.from_dict(graph_dict(nodes), client_factory=factory)

    events = await collect_stream(engine, {"user_message": "x"})

    assert [e for e in events if e.type == "restart"] == []
    assert events[-1].output_data == {"agent_response": "done"}


TONE = [{"name": "tone", "value": "formal"}]


def _tone_engine(**engine_options):
    return WorkflowEngine.from_dict(
        graph_dict(_llm_nodes(prompt="Be {{ templates.tone }}."), templates=TONE),
        **engine_options,
    )


async def test_per_run_templates_override_the_documents_values():
    factory = make_factory({"agent_response": "r"})
    engine = _tone_engine(client_factory=factory)

    await engine.run({"user_message": "x"})
    await engine.run({"user_message": "x"}, templates={"tone": "casual"})

    first, second = (c.calls[0].messages[0].content for c in factory.created)
    assert "Be formal." in first
    assert "Be casual." in second


async def test_a_per_run_template_value_is_never_rendered_again():
    """Rendering is one pass, so text passed per run needs no escaping."""
    factory = make_factory({"agent_response": "r"})
    engine = _tone_engine(client_factory=factory)

    await engine.run(
        {"user_message": "secret"},
        templates={"tone": "{{ context.input.user_message }}"},
    )

    system = factory.created[0].calls[0].messages[0].content
    assert "Be {{ context.input.user_message }}." in system


async def test_an_undeclared_or_non_text_template_is_refused():
    engine = _tone_engine(client_factory=make_factory({"agent_response": "r"}))

    with pytest.raises(WorkflowException, match="not declared"):
        await engine.run({"user_message": "x"}, templates={"mood": "calm"})
    with pytest.raises(WorkflowException, match="not strings"):
        await engine.run({"user_message": "x"}, templates={"tone": 3})


async def test_per_run_templates_are_recorded_with_the_run():
    service = make_agent_service()
    engine = _tone_engine(
        agent_service=service, client_factory=make_factory({"agent_response": "r"})
    )

    await engine.run({"user_message": "x"}, templates={"tone": "casual"})

    assert (await _only_run(service)).context["run_templates"] == {"tone": "casual"}


async def test_a_run_past_its_timeout_is_cancelled_and_recorded():
    from kavalai.workflow.models import WorkflowTimeoutError

    service = make_agent_service()
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        agent_service=service,
        client_factory=lambda *a, **k: _SlowClient(),
    )
    events = []

    with pytest.raises(WorkflowTimeoutError, match="time limit of 0.05 s"):
        async for event in engine.run_stream({"user_message": "x"}, timeout=0.05):
            events.append(event)

    assert events[-1].type == "workflow_failed"
    assert events[-1].run_id == events[0].run_id
    run = await _only_run(service)
    assert run.context["status"] == "failed"
    assert "time limit" in run.context["error"]


async def test_the_engine_run_timeout_is_the_default():
    from kavalai.workflow.models import WorkflowTimeoutError

    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        client_factory=lambda *a, **k: _SlowClient(),
        run_timeout=0.05,
    )

    with pytest.raises(WorkflowTimeoutError):
        await engine.run({"user_message": "x"})


async def test_a_run_within_its_timeout_completes():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        client_factory=make_factory({"agent_response": "hi"}),
        run_timeout=5,
    )

    state = await engine.run({"user_message": "x"})

    assert state.output_data == {"agent_response": "hi"}


async def test_a_failure_inside_a_timed_run_keeps_its_own_error():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=make_factory(raises=True)
    )

    with pytest.raises(WorkflowException, match="llm boom"):
        await engine.run({"user_message": "x"}, timeout=5)


async def test_closing_a_timed_stream_cancels_its_walk():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES), client_factory=lambda *a, **k: _SlowClient()
    )
    stream = engine.run_stream({"user_message": "x"}, timeout=5)
    seen = [await stream.__anext__() for _ in range(4)]
    assert (seen[-1].type, seen[-1].name) == ("node_started", "answer")

    await stream.aclose()


async def test_a_workflow_exception_is_recorded_on_the_run():
    nodes = [
        {"name": "s", "type": "start", "next": "route"},
        {
            "name": "route",
            "type": "switch",
            "expr": "input.user_message",
            "cases": {"yes": "e"},
        },
        {"name": "e", "type": "end", "output": "output"},
    ]
    service = make_agent_service()
    engine = WorkflowEngine.from_dict(graph_dict(nodes), agent_service=service)

    with pytest.raises(WorkflowException, match="halted"):
        await engine.run({"user_message": "no"})

    run = await _only_run(service)
    assert run.context["status"] == "failed"
    assert "halted" in run.context["error"]


async def test_record_context_false_keeps_only_the_input_and_the_output():
    service = make_agent_service()
    engine = _tone_engine(
        agent_service=service,
        client_factory=make_factory({"agent_response": "r"}),
        record_context=False,
    )

    await engine.run({"user_message": "x"}, templates={"tone": "casual"})

    assert (await _only_run(service)).context == {
        "input": {"user_message": "x"},
        "output": {"agent_response": "r"},
        "run_templates": {"tone": "casual"},
    }


async def test_record_context_false_records_a_failure_without_its_data():
    service = make_agent_service()
    engine = _tone_engine(
        agent_service=service,
        client_factory=make_factory(raises=True),
        record_context=False,
    )

    with pytest.raises(WorkflowException):
        await engine.run({"user_message": "x"}, templates={"tone": "casual"})

    context = (await _only_run(service)).context
    assert set(context) == {"status", "error", "run_templates"}


async def test_min_similarity_drops_weak_hits():
    service = RecordingRagService(
        [rag_hit("strong", similarity=0.9), rag_hit("weak", similarity=0.2)]
    )
    engine = WorkflowEngine.from_dict(
        rag_graph(min_similarity=0.5), rag_services=service
    )

    state = await engine.run({"user_message": "q"})

    assert [hit["content"] for hit in state.data["docs"]] == ["strong"]


async def test_rag_query_reports_its_hits_on_node_completed():
    hit = rag_hit("a fact", source_id="doc-1", similarity=0.8)
    engine = WorkflowEngine.from_dict(
        rag_graph(store="content"), rag_services=RecordingRagService([hit])
    )

    events = await collect_stream(engine, {"user_message": "q"})

    completed = {e.name: e for e in events if e.type == "node_completed"}
    assert completed["retrieve"].output_data == {
        "hits": [
            {"id": str(hit.id), "source_id": "doc-1", "similarity": 0.8, "metadata": {}}
        ]
    }
    assert completed["s"].output_data is None


async def test_a_content_rag_query_still_records_its_hits_on_the_task_row():
    hit = rag_hit("a fact", source_id="doc-1")
    engine = WorkflowEngine.from_dict(
        rag_graph(store="content"), rag_services=RecordingRagService([hit])
    )

    _, records = await _trajectory(engine, {"user_message": "q"})

    row = next(r for r in records if r.name == "retrieve")
    assert row.output["content"] == "a fact"
    assert [h["source_id"] for h in row.output["hits"]] == ["doc-1"]
    assert row.inputs["top_k"] == 5
    assert row.inputs["source_ids"] is None
    assert row.inputs["min_similarity"] is None


async def test_rag_query_hands_the_runs_accumulator_to_the_service():
    from kavalai.workflow.tasklog.base import TokenAccumulator

    service = RecordingRagService([rag_hit("x")])
    engine = WorkflowEngine.from_dict(rag_graph(), rag_services=service)

    await engine.run({"user_message": "q"})

    assert isinstance(service.calls[0]["stats_receiver"], TokenAccumulator)


class _PlainRagService:
    """A service written against the interface before stats receivers."""

    def __init__(self):
        self.calls = []

    async def query(
        self,
        text,
        top_k=5,
        collection_name=None,
        source_ids=None,
        keep_best=False,
        include_content=True,
    ):
        self.calls.append(text)
        return []


async def test_a_service_without_a_stats_receiver_parameter_still_works():
    service = _PlainRagService()
    engine = WorkflowEngine.from_dict(rag_graph(), rag_services=service)

    await engine.run({"user_message": "q"})

    assert service.calls == ["q"]


async def test_a_registered_service_is_built_once_per_engine():
    from kavalai.llm_clients import registry

    built = []

    def build():
        service = RecordingRagService()
        built.append(service)
        return service

    registry.register_rag_service("default", build)
    try:
        engine = WorkflowEngine.from_dict(rag_graph())
        await engine.run({"user_message": "a"})
        await engine.run({"user_message": "b"})

        assert len(built) == 1
        assert len(built[0].calls) == 2
    finally:
        registry.rag_services.unregister("default")


async def test_model_calls_carry_the_run_and_session_they_belong_to():
    from kavalai.workflow.tasklog import MemoryTaskLogger

    engine = WorkflowEngine.from_dict(
        graph_dict(_llm_nodes()),
        agent_service=make_agent_service(),
        client_factory=make_factory({"agent_response": "r"}),
    )
    task_logger = MemoryTaskLogger()

    state = await engine.run({"user_message": "x"}, task_logger=task_logger)

    (call,) = task_logger.model_calls
    assert (call.agent_id, call.session_id, call.run_id) == (
        state.agent_id,
        state.session_id,
        state.run_id,
    )


async def test_the_completed_event_carries_the_run_id():
    engine = WorkflowEngine.from_dict(
        graph_dict(STREAM_NODES),
        agent_service=make_agent_service(),
        client_factory=make_factory({"agent_response": "r"}),
    )

    events = await collect_stream(engine, {"user_message": "x"})

    assert events[-1].type == "workflow_completed"
    assert events[-1].run_id == events[0].run_id is not None
