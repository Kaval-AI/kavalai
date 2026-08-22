"""Tests for the in-memory task logger and the payload size cap."""

import pytest

from kavalai.llm_clients.base_client import ModelCallStat
from kavalai.workflow.tasklog import MemoryTaskLogger, truncate_payload
from kavalai.workflow.tasklog.sqlite import SqliteTaskLogger


@pytest.fixture
def task_logger():
    return MemoryTaskLogger()


async def test_records_a_node(task_logger):
    task_logger.log_node(
        run_id="r1",
        session_id="s1",
        agent_id="a1",
        node_name="classify",
        node_type="llm",
        inputs={"x": 1},
        output={"intent": "greet"},
        prompt="classify this",
        duration=0.5,
        seq=0,
    )
    await task_logger.flush()

    (record,) = task_logger.records
    assert record.name == "classify"
    assert record.node_type == "llm"
    assert record.inputs == {"x": 1}
    assert record.output == {"intent": "greet"}
    assert record.prompt == "classify this"
    assert record.duration_seconds == 0.5
    assert record.run_id == "r1"


async def test_records_come_back_in_sequence_order(task_logger):
    # Writes are fire-and-forget, so completion order is not execution order;
    # ``seq`` is what carries the structure.
    for seq in (2, 0, 1):
        task_logger.log_node(
            run_id="r1",
            session_id=None,
            agent_id=None,
            node_name=f"n{seq}",
            node_type="function",
            inputs=None,
            output=None,
            seq=seq,
        )
    await task_logger.flush()
    assert [r.name for r in task_logger.records] == ["n0", "n1", "n2"]


async def test_for_run_separates_concurrent_runs(task_logger):
    for run_id in ("r1", "r2"):
        task_logger.log_node(
            run_id=run_id,
            session_id=None,
            agent_id=None,
            node_name="node",
            node_type="llm",
            inputs=None,
            output=None,
            seq=0,
        )
    await task_logger.flush()

    assert len(task_logger.records) == 2
    assert [r.run_id for r in task_logger.for_run("r2")] == ["r2"]
    assert len(task_logger.for_run(None)) == 2


async def test_child_rows_carry_their_parent(task_logger):
    task_logger.log_node(
        run_id="r1",
        session_id=None,
        agent_id=None,
        node_name="store_order",
        node_type="tool_call",
        inputs={"args": {"n": 2}, "step": 1},
        output={"order_id": "o-1"},
        seq=4,
        parent_task_name="validate",
        tool_uri="python://store_order",
    )
    await task_logger.flush()

    (record,) = task_logger.records
    assert record.parent_task_name == "validate"
    assert record.tool_uri == "python://store_order"
    assert record.inputs["step"] == 1


async def test_model_calls_are_collected(task_logger):
    task_logger.log_model_call(
        ModelCallStat(call_type="llm", model="m", total_tokens=7)
    )
    await task_logger.flush()
    assert [s.total_tokens for s in task_logger.model_calls] == [7]


async def test_clear_drops_everything(task_logger):
    task_logger.log_node(
        run_id="r1",
        session_id=None,
        agent_id=None,
        node_name="n",
        node_type="llm",
        inputs=None,
        output=None,
    )
    task_logger.log_model_call(ModelCallStat(call_type="llm", model="m"))
    await task_logger.flush()

    task_logger.clear()
    assert task_logger.records == []
    assert task_logger.model_calls == []


async def test_payloads_are_not_truncated_by_default(task_logger):
    """Nothing is written to a database, so the writer's size cap does not apply."""
    big = {"blob": "x" * (600 * 1024)}
    task_logger.log_node(
        run_id="r1",
        session_id=None,
        agent_id=None,
        node_name="crawl",
        node_type="function",
        inputs=None,
        output=big,
    )
    await task_logger.flush()
    assert task_logger.records[0].output == big


async def test_oversized_payloads_are_replaced_by_a_marker():
    """The database-backed loggers keep one crawl result from breaking the writer."""
    logger = SqliteTaskLogger()
    logger.max_payload_bytes = 1024
    logger.log_node(
        run_id="r1",
        session_id="s1",
        agent_id=None,
        node_name="crawl",
        node_type="function",
        inputs=None,
        output={"page": "x" * 5000},
    )
    await logger.flush()

    (row,) = await logger.get_tasks()
    assert row["output"]["truncated"] is True
    assert row["output"]["bytes"] > 5000
    assert row["output"]["preview"].startswith('{"page"')
    await logger.close()


def test_truncate_payload_passes_small_values_through():
    assert truncate_payload({"a": 1}, 1024) == {"a": 1}
    assert truncate_payload(None, 1024) is None
    # A cap of zero disables truncation entirely.
    assert truncate_payload({"a": "x" * 100}, 0) == {"a": "x" * 100}
