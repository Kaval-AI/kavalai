import xml.etree.ElementTree as ET

import pytest

from kavalai.workflow import WorkflowBuilder, render_workflow_svg
from kavalai.workflow.render import _as_dict, _truncate


def _parse(svg: str) -> ET.Element:
    """Assert the SVG is well-formed XML and return its root."""
    return ET.fromstring(svg)


def _linear_graph():
    return (
        WorkflowBuilder("Chatbot", llm_model="openai/x")
        .data_type("input", {"message": str})
        .data_type("output", {"agent_response": str})
        .start("reply")
        .llm("reply", prompt="p", inputs={"m": "input"}, output="output", next="end")
        .end()
        .build()
    )


BRANCHY = {
    "name": "Triage",
    "start": "begin",
    "nodes": [
        {"name": "begin", "type": "start", "next": "classify"},
        {"name": "classify", "type": "llm", "output": "c", "next": "route"},
        {
            "name": "route",
            "type": "switch",
            "expr": "c.intent",
            "cases": {"refund": "refund", "tech": "tech"},
            "default": "other",
        },
        {"name": "refund", "type": "llm", "output": "o", "next": "gate"},
        {"name": "tech", "type": "agent", "output": "o", "next": "gate"},
        {"name": "other", "type": "llm", "output": "o", "next": "gate"},
        {
            "name": "gate",
            "type": "if",
            "condition": "o.ok",
            "then": "done",
            "else": "begin",
        },
        {"name": "done", "type": "end", "output": "o"},
    ],
}


def test_renders_a_workflowgraph():
    svg = render_workflow_svg(_linear_graph())
    _parse(svg)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    # Every node is drawn by name, with its type as a sub-label.
    for name in ("start", "reply", "end"):
        assert f">{name}<" in svg
    assert ">llm<" in svg


def test_renders_a_workflow_dict():
    svg = render_workflow_svg(BRANCHY)
    _parse(svg)
    for name in (
        "begin",
        "classify",
        "route",
        "refund",
        "tech",
        "other",
        "gate",
        "done",
    ):
        assert f">{name}<" in svg


def test_switch_cases_are_labelled_on_arrows():
    svg = render_workflow_svg(BRANCHY)
    # Each case value and the default branch label its own arrow.
    assert ">refund<" in svg and ">tech<" in svg and ">default<" in svg


def test_if_condition_and_else_are_labelled_on_arrows():
    svg = render_workflow_svg(BRANCHY)
    assert ">o.ok<" in svg  # the condition on the `then` arrow
    assert ">else<" in svg  # the `else` arrow


def test_loop_does_not_collapse_the_layout():
    """A back-edge (gate -> begin) must not push every node onto one row."""
    svg = render_workflow_svg(BRANCHY)
    ys = sorted(
        {
            float(r.get("y"))
            for r in _parse(svg).iter()
            if r.tag.endswith("rect") and r.get("y") and r.get("rx") == "10"
        }
    )
    # 8 nodes across several distinct rows (begin/classify/route/branches/gate/done).
    assert len(ys) >= 5


def test_xml_escapes_node_names_and_labels():
    wf = {
        "start": "s",
        "nodes": [
            {"name": "s", "type": "start", "next": "branch"},
            {
                "name": "branch",
                "type": "if",
                "condition": "a < b & c",
                "then": "e",
                "else": "e",
            },
            {"name": "e", "type": "end"},
        ],
    }
    svg = render_workflow_svg(wf)
    _parse(svg)  # would raise if the `<`/`&` were not escaped
    assert "&lt;" in svg and "&amp;" in svg


def test_unreachable_node_is_still_drawn():
    wf = {
        "start": "s",
        "nodes": [
            {"name": "s", "type": "start", "next": "e"},
            {"name": "e", "type": "end"},
            {"name": "orphan", "type": "llm", "next": "e"},  # not reachable from s
        ],
    }
    svg = render_workflow_svg(wf)
    _parse(svg)
    assert ">orphan<" in svg


def test_empty_workflow_renders_without_error():
    svg = render_workflow_svg({"nodes": []})
    _parse(svg)
    assert "<svg" in svg


def test_as_dict_rejects_unsupported_input():
    with pytest.raises(TypeError):
        _as_dict(42)


def test_truncate():
    assert _truncate("short") == "short"
    out = _truncate("x" * 40, limit=10)
    assert len(out) == 10 and out.endswith("…")


def test_transitions_to_unknown_nodes_are_skipped():
    """The backoffice can store a half-edited workflow whose transition points
    at a node that no longer exists; rendering must still work."""
    workflow = {
        "nodes": [
            {"name": "s", "type": "start", "next": "ghost"},
            {"name": "e", "type": "end", "output": "output"},
        ]
    }

    svg = render_workflow_svg(workflow)

    assert "ghost" not in svg
    assert ">s<" in svg and ">e<" in svg
