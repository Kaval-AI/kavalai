import pytest

from kavalai.workflow import WorkflowBuilder, WorkflowEngine
from kavalai.workflow.builder import _coerce_inputs, _field_schema
from kavalai.workflow.models import (
    AgentNode,
    FunctionNode,
    IfNode,
    LLMNode,
    SwitchNode,
)
from kavalai.workflow.models import ArgumentInfo


def _base(builder: WorkflowBuilder) -> WorkflowBuilder:
    return builder.data_type("input", {"user_message": str}).data_type(
        "output", {"agent_response": str}
    )


def test_field_schema_variants():
    assert _field_schema(str) == {"type": "string"}
    assert _field_schema(int) == {"type": "integer"}
    assert _field_schema("number") == {"type": "number"}
    raw = {"type": "array", "items": {"type": "string"}}
    assert _field_schema(raw) is raw


def test_field_schema_rejects_bad_type():
    with pytest.raises(TypeError):
        _field_schema(bytes)
    with pytest.raises(TypeError):
        _field_schema(123)


def test_coerce_inputs_variants():
    info = ArgumentInfo(type="literal", value="x")
    out = _coerce_inputs(
        {
            "a": "input.user_message",  # string -> context path
            "b": {"type": "literal", "value": 5},  # dict -> ArgumentInfo
            "c": info,  # ArgumentInfo passthrough
        }
    )
    assert out["a"] == ArgumentInfo(type="context", value="input.user_message")
    assert out["b"] == ArgumentInfo(type="literal", value=5)
    assert out["c"] is info
    assert _coerce_inputs(None) == {}


def test_coerce_inputs_rejects_bad_spec():
    with pytest.raises(TypeError):
        _coerce_inputs({"a": 123})


def test_build_minimal_graph():
    graph = (
        WorkflowBuilder("demo", description="d", llm_model="openai/x")
        .data_type("input", {"user_message": str})
        .data_type("output", {"agent_response": str})
        .start("reply")
        .llm(
            "reply", prompt="hi", inputs={"input": "input"}, output="output", next="end"
        )
        .end()
        .build()
    )
    assert graph.name == "demo"
    assert graph.llm_model == "openai/x"
    assert graph.start == "start"  # entry node (default name); its next is "reply"
    assert graph.node_map["start"].next == "reply"
    reply = graph.node_map["reply"]
    assert isinstance(reply, LLMNode)
    assert reply.inputs["input"] == ArgumentInfo(type="context", value="input")


def test_build_all_node_types():
    graph = (
        WorkflowBuilder("everything")
        .data_type("input", {"user_message": str})
        .data_type("flag", {"go": bool})
        .data_type("output", {"agent_response": str})
        .start("decide")
        .if_("decide", condition="True", then="sw", else_="act")
        .switch("sw", expr="input.user_message", cases={"x": "act"}, default="act")
        .agent("act", prompt="do", output="output", next="call", max_steps=2)
        .function("call", tool="python://noop", output="output", next="end")
        .end()
        .build()
    )
    assert isinstance(graph.node_map["decide"], IfNode)
    assert isinstance(graph.node_map["sw"], SwitchNode)
    assert isinstance(graph.node_map["act"], AgentNode)
    assert graph.node_map["act"].max_steps == 2
    assert isinstance(graph.node_map["call"], FunctionNode)
    assert graph.node_map["decide"].else_ == "act"


def test_data_type_schema_and_ref():
    graph = (
        WorkflowBuilder("dt")
        .data_type("input", {"user_message": str})
        .data_type(
            "feed",
            schema={"type": "object", "properties": {"url": {"type": "string"}}},
        )
        .data_type("output", ref="feed")
        .start("end")
        .end()
        .build()
    )
    assert graph.data_types["feed"]["properties"]["url"]["type"] == "string"
    assert graph.data_types["output"] == {"$ref": "feed"}


def test_resources_registered():
    builder = (
        _base(WorkflowBuilder("res"))
        .start("end")
        .end()
        .python_function("noop", "kavalai.utils.to_plain")
        .rest_server({"name": "api", "url": "http://localhost"})
        .mcp_server({"name": "m", "command": "echo"})
        .template("greeting", "Hello {{ name }}")
    )
    graph = builder.build()
    assert graph.python_functions[0].name == "noop"
    assert graph.rest_servers[0].name == "api"
    assert graph.mcp_servers[0].name == "m"
    assert graph.templates[0].name == "greeting"


def test_explicit_node_name_overrides_default():
    graph = (
        _base(WorkflowBuilder("named"))
        .start("reply", name="begin")
        .llm("reply", prompt="hi", output="output", next="finish")
        .end(name="finish", output="output")
        .build()
    )
    assert graph.start == "begin"
    assert "finish" in graph.node_map


def test_build_engine_returns_engine():
    engine = (
        _base(WorkflowBuilder("eng", llm_model="openai/x"))
        .start("reply")
        .llm("reply", prompt="hi", output="output", next="end")
        .end()
        .build_engine()
    )
    assert isinstance(engine, WorkflowEngine)
    assert engine.graph.name == "eng"


def test_data_model_uses_pydantic_models_directly():
    """data_model registers a Pydantic model used as-is by the engine (no
    re-parsing), and records its JSON schema on the graph for validation."""
    from typing import Literal

    from pydantic import BaseModel

    class Email(BaseModel):
        sender: str
        subject: str
        content: str

    class Classification(BaseModel):
        category: Literal["support_request", "other"]

    builder = (
        WorkflowBuilder("triage", llm_model="openai/fake")
        .data_model("input", Email)
        .data_model("classification", Classification)
        .data_type("output", {"agent_response": str})
        .start("classify")
        .llm(
            "classify",
            prompt="p",
            inputs={"email": "input"},
            output="classification",
            next="end",
        )
        .end()
    )

    # The graph carries each model's JSON schema (so validate_graph accepts the
    # node outputs and storage has a schema).
    graph = builder.build()
    assert graph.data_types["input"]["properties"].keys() >= {
        "sender",
        "subject",
        "content",
    }
    assert "enum" in graph.data_types["classification"]["properties"]["category"]

    # The engine uses the supplied models verbatim, not parser-compiled copies.
    engine = builder.build_engine()
    assert engine.get_data_type("input") is Email
    assert engine.get_data_type("classification") is Classification
    # The non-model data_type is still compiled by the parser.
    assert engine.get_data_type("output").__name__ == "output"


def test_data_model_validates_input_at_runtime():
    import asyncio

    from pydantic import BaseModel

    from kavalai.llm_clients.base_client import BaseLlmClient

    class Email(BaseModel):
        sender: str
        content: str

    class Reply(BaseModel):
        agent_response: str

    class _Fake(BaseLlmClient):
        def __init__(self, *a, **k):
            super().__init__()

        async def chat_completions(self, *, chat_history, response_model=None):
            return response_model(agent_response="ok")

    engine = (
        WorkflowBuilder("e", llm_model="openai/fake")
        .data_model("input", Email)
        .data_model("output", Reply)
        .start("reply")
        .llm(
            "reply", prompt="p", inputs={"email": "input"}, output="output", next="end"
        )
        .end()
        .build_engine(client_factory=lambda *a, **k: _Fake())
    )

    state = asyncio.run(engine.run({"sender": "a@b.c", "content": "hi"}))
    assert state.output_data == {"agent_response": "ok"}

    # A payload that violates the Email model is rejected.
    with pytest.raises(Exception):
        asyncio.run(engine.run({"sender": "a@b.c"}))  # missing 'content'
