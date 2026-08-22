"""Shared fixtures for the evaluation tests.

Nothing here talks to a provider or a database: an evaluation harness that
gates deploys is itself code, and its tests have to be able to run everywhere.
"""

import pytest

from kavalai.eval import Case, RunRecord, Trajectory
from kavalai.workflow.tasklog import TaskRecord


def record(**kwargs) -> RunRecord:
    """A RunRecord with sensible defaults, overridden per test."""
    kwargs.setdefault("output", {"agent_response": "hello"})
    return RunRecord(**kwargs)


def trajectory(*records: TaskRecord) -> Trajectory:
    return Trajectory(records=list(records))


def task(name, node_type="llm", seq=0, **kwargs) -> TaskRecord:
    return TaskRecord(name=name, node_type=node_type, seq=seq, **kwargs)


@pytest.fixture
def case() -> Case:
    return Case(name="c1", inputs={"user_message": "hi"})
