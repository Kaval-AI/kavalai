"""The ordered view over a run, and the loop ambiguity it has to resolve."""

from kavalai.eval import Trajectory
from kavalai.workflow.tasklog import TaskRecord


def t(name, node_type="llm", seq=0, **kwargs) -> TaskRecord:
    return TaskRecord(name=name, node_type=node_type, seq=seq, **kwargs)


def test_an_unobserved_trajectory_says_so():
    assert Trajectory().observed is False
    assert Trajectory(records=[t("a")]).observed is True


def test_names_are_the_executed_path_without_tool_calls():
    trajectory = Trajectory(
        records=[
            t("begin", "start", 0),
            t("agent", "agent", 1),
            t("crawl", "tool_call", 2, parent_task_name="agent"),
            t("finish", "end", 3),
        ]
    )
    assert trajectory.names() == ["begin", "agent", "finish"]


def test_tools_covers_function_nodes_and_agent_calls_alike():
    trajectory = Trajectory(
        records=[
            t("validate", "function", 0, tool_uri="python://validate"),
            t("crawl", "tool_call", 1, tool_uri="python://crawl"),
            t("answer", "llm", 2),
        ]
    )
    assert trajectory.tool_uris() == ["python://validate", "python://crawl"]
    assert trajectory.called("python://crawl") is True
    assert trajectory.called("python://nope") is False
    assert len(trajectory.calls_to("python://validate")) == 1


def test_branch_lookup_reads_the_decision():
    trajectory = Trajectory(
        records=[
            t(
                "route",
                "switch",
                0,
                inputs={"expr": "x", "value": "order"},
                output={"taken": "validate", "matched": True},
            ),
            t("gate", "if", 1, output={"taken": "store", "matched": True}),
        ]
    )
    assert trajectory.taken("route") == "validate"
    assert trajectory.taken("gate") == "store"
    assert trajectory.taken("absent") is None
    assert len(trajectory.branches()) == 2


def test_agent_steps_counts_from_the_step_field():
    trajectory = Trajectory(
        records=[
            t("agent", "agent", 0),
            t("a", "tool_call", 1, parent_task_name="agent", inputs={"step": 0}),
            t("b", "tool_call", 2, parent_task_name="agent", inputs={"step": 2}),
        ]
    )
    assert trajectory.agent_steps() == 3
    assert trajectory.agent_steps("agent") == 3
    assert trajectory.agent_steps("other") == 0
    assert Trajectory().agent_steps() == 0


def test_children_are_segmented_per_visit_not_per_name():
    """``parent_task_name`` is unique per node, not per visit.

    The engine permits a node to be visited more than once, so a name join
    alone cannot say which visit a tool call belongs to. ``seq`` can: children
    follow their parent and precede the next node, and segmenting on that
    boundary recovers the grouping exactly.
    """
    trajectory = Trajectory(
        records=[
            t("retry", "agent", 0),
            t("first", "tool_call", 1, parent_task_name="retry"),
            t("check", "llm", 2),
            t("retry", "agent", 3),
            t("second", "tool_call", 4, parent_task_name="retry"),
            t("third", "tool_call", 5, parent_task_name="retry"),
        ]
    )
    groups = trajectory.child_groups("retry")
    assert [[r.name for r in g] for g in groups] == [["first"], ["second", "third"]]
    # The convenience accessor takes the first visit.
    assert [r.name for r in trajectory.children_of("retry")] == ["first"]
    assert trajectory.children_of("nowhere") == []


def test_node_accessors():
    trajectory = Trajectory(records=[t("a", seq=0), t("a", seq=1), t("b", seq=2)])
    assert trajectory.node("a").seq == 0
    assert [r.seq for r in trajectory.nodes("a")] == [0, 1]
    assert trajectory.node("z") is None
    assert trajectory.visited("b") is True


def test_as_table_is_printable():
    row = Trajectory(records=[t("a", "llm", 0, duration_seconds=1.23456)]).as_table()[0]
    assert row == {
        "seq": 0,
        "name": "a",
        "type": "llm",
        "tool": None,
        "parent": None,
        "seconds": 1.235,
    }
