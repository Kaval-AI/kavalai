import pytest
import yaml
from pydantic import ValidationError

from kavalai.workflow.models import (
    AgentNode,
    EndNode,
    IfNode,
    LLMNode,
    McpServer,
    RestServer,
    StartNode,
    SwitchNode,
    WorkflowGraph,
    WorkflowStreamEvent,
)

BASE_DATA_TYPES = {
    "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
    "output": {"type": "object", "properties": {"agent_response": {"type": "string"}}},
}


def make_graph(nodes, **extra):
    return WorkflowGraph(
        name="wf", data_types=dict(BASE_DATA_TYPES), nodes=nodes, **extra
    )


def test_minimal_valid_graph():
    graph = make_graph(
        [
            {"name": "s", "type": "start", "next": "e"},
            {"name": "e", "type": "end", "output": "output"},
        ]
    )
    assert graph.start == "s"
    assert isinstance(graph.node_map["s"], StartNode)
    assert isinstance(graph.node_map["e"], EndNode)


def test_discriminated_node_parsing():
    graph = make_graph(
        [
            {"name": "s", "type": "start", "next": "branch"},
            {
                "name": "branch",
                "type": "if",
                "condition": "input.user_message == 'hi'",
                "then": "e",
                "else": "e",
            },
            {
                "name": "sw",
                "type": "switch",
                "expr": "input.user_message",
                "cases": {"hi": "e"},
                "default": "e",
            },
            {"name": "e", "type": "end"},
        ]
    )
    branch = graph.node_map["branch"]
    assert isinstance(branch, IfNode)
    assert branch.else_ == "e"  # 'else' alias maps to else_
    assert isinstance(graph.node_map["sw"], SwitchNode)


def test_missing_start_node():
    with pytest.raises(ValidationError, match="exactly one 'start' node, found 0"):
        make_graph([{"name": "e", "type": "end"}])


def test_missing_end_node():
    with pytest.raises(ValidationError, match="at least one 'end'"):
        make_graph([{"name": "s", "type": "start", "next": "s"}])


def test_duplicate_node_names():
    with pytest.raises(ValidationError, match="Duplicate node names"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "e"},
                {"name": "s", "type": "end"},
                {"name": "e", "type": "end"},
            ]
        )


def test_unknown_transition_target():
    with pytest.raises(ValidationError, match="unknown node 'nope'"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "nope"},
                {"name": "e", "type": "end"},
            ]
        )


def test_switch_unknown_case_target():
    with pytest.raises(ValidationError, match="unknown node"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "sw"},
                {
                    "name": "sw",
                    "type": "switch",
                    "expr": "input.user_message",
                    "cases": {"hi": "ghost"},
                },
                {"name": "e", "type": "end"},
            ]
        )


def test_undeclared_output_data_type():
    with pytest.raises(ValidationError, match="not declared in data_types"):
        make_graph(
            [
                {"name": "s", "type": "start", "next": "n"},
                {
                    "name": "n",
                    "type": "llm",
                    "prompt": "p",
                    "output": "undeclared",
                    "next": "e",
                },
                {"name": "e", "type": "end"},
            ]
        )


def test_multiple_start_nodes_rejected():
    with pytest.raises(ValidationError, match="exactly one 'start' node, found 2"):
        make_graph(
            [
                {"name": "s1", "type": "start", "next": "e"},
                {"name": "s2", "type": "start", "next": "e"},
                {"name": "e", "type": "end"},
            ]
        )


def test_multiple_end_nodes_share_output_type():
    graph = make_graph(
        [
            {"name": "s", "type": "start", "next": "branch"},
            {
                "name": "branch",
                "type": "if",
                "condition": "input.user_message == 'hi'",
                "then": "e1",
                "else": "e2",
            },
            {"name": "e1", "type": "end", "output": "output"},
            {"name": "e2", "type": "end", "output": "output"},
        ]
    )
    assert graph.output_type == "output"


def test_end_nodes_with_different_output_types_rejected():
    with pytest.raises(
        ValidationError, match="same output data type.*'other', 'output'"
    ):
        WorkflowGraph(
            name="wf",
            data_types={
                **BASE_DATA_TYPES,
                "other": {"type": "object", "properties": {}},
            },
            nodes=[
                {"name": "s", "type": "start", "next": "e1"},
                {"name": "e1", "type": "end", "output": "output"},
                {"name": "e2", "type": "end", "output": "other"},
            ],
        )


def test_from_yaml_roundtrip():
    text = """
name: yaml_wf
description: built from yaml
llm_model: openai/x
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
    graph = WorkflowGraph(**yaml.safe_load(text))
    assert graph.name == "yaml_wf"
    assert graph.llm_model == "openai/x"
    assert graph.start == "s"
    assert graph.output_type == "output"


def test_stream_flags_default_off():
    llm = LLMNode(name="n", prompt="p", output="output", next="e")
    assert llm.stream_output is False
    assert llm.stream_delta is False

    agent = AgentNode(name="a", prompt="p", output="output", next="e")
    assert agent.stream_output is False
    assert agent.stream_delta is False
    assert agent.stream_instructions is False
    assert agent.stream_partials is False


def test_stream_flags_parse_from_yaml():
    text = """
name: n
type: agent
prompt: p
output: output
next: e
stream_output: true
stream_instructions: true
stream_partials: true
stream_delta: true
"""
    node = AgentNode(**yaml.safe_load(text))
    assert node.stream_output and node.stream_instructions
    assert node.stream_partials and node.stream_delta


def test_workflow_stream_event_optional_fields_and_types():
    event = WorkflowStreamEvent(type="partial", name="node")
    assert event.value is None and event.session_id is None
    assert event.output_data is None and event.token_usage is None

    completed = WorkflowStreamEvent(
        type="workflow_completed",
        name="wf",
        session_id="s",
        output_data={"a": 1},
        token_usage={"total_tokens": 2},
    )
    assert completed.output_data == {"a": 1}

    with pytest.raises(ValidationError):
        WorkflowStreamEvent(type="not_a_type", name="x")


def test_rest_server_rejects_both_url_and_url_env():
    with pytest.raises(ValidationError, match="Only one of 'url' or 'url_env'"):
        RestServer(name="api", url="http://x", url_env="API_URL")


def test_rest_server_requires_a_url():
    with pytest.raises(ValidationError, match="Either 'url' or 'url_env'"):
        RestServer(name="api")


def test_mcp_server_rejects_stdio_and_http_together():
    with pytest.raises(ValidationError, match="Cannot specify both stdio"):
        McpServer(name="m", command="echo", url="http://x")


def test_mcp_server_requires_a_transport():
    with pytest.raises(ValidationError, match="Either stdio"):
        McpServer(name="m")


def test_mcp_server_rejects_two_stdio_command_sources():
    with pytest.raises(ValidationError, match="Only one of 'command' or 'command_env'"):
        McpServer(name="m", command="echo", command_env="MCP_COMMAND")


def test_mcp_server_rejects_two_http_url_sources():
    with pytest.raises(ValidationError, match="Only one of 'url' or 'url_env'"):
        McpServer(name="m", url="http://x", url_env="MCP_URL")


def test_rest_server_accepts_a_single_url_source():
    assert RestServer(name="api", url="http://x").url == "http://x"
    assert RestServer(name="api", url_env="API_URL").url_env == "API_URL"


def test_mcp_server_accepts_a_single_transport():
    assert McpServer(name="m", command="echo").command == "echo"
    assert McpServer(name="m", url="http://x").url == "http://x"
