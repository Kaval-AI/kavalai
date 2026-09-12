"""Tests for the in-memory task logger, the recording options and the payload cap."""

import json
from decimal import Decimal

import pytest

from kavalai.db import ModelCallStat as ModelCallRow
from kavalai.llm_clients.base_client import ModelCallStat
from kavalai.workflow.tasklog import MemoryTaskLogger, truncate_payload
from kavalai.workflow.tasklog.base import (
    StatsBridge,
    TokenAccumulator,
    as_model_call_stat,
    truncate_text_payload,
)
from kavalai.workflow.tasklog.memory import ModelCallRecord, TeeTaskLogger
from kavalai.workflow.tasklog.sqlite import SqliteTaskLogger


def log_one_node(task_logger, run_id="r1"):
    task_logger.log_node(
        run_id=run_id,
        session_id="s1",
        agent_id="a1",
        node_name="answer",
        node_type="llm",
        inputs={"user_message": "my card number is 4111"},
        output={"agent_response": "noted"},
        prompt="Answer the visitor.",
        duration=0.25,
        errors=["late"],
        seq=3,
    )


def a_model_call(**overrides):
    values = dict(
        call_type="llm",
        model="fake/scripted",
        request_data=json.dumps({"messages": ["the whole transcript"]}),
        response_data=json.dumps({"text": "noted"}),
        prompt_tokens=11,
        completion_tokens=4,
        total_tokens=15,
        duration_seconds=0.5,
    )
    values.update(overrides)
    return ModelCallStat(**values)


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


def test_truncate_text_payload_keeps_the_column_a_string():
    assert truncate_text_payload(None, 10) is None
    assert truncate_text_payload('{"a": 1}', 1024) == '{"a": 1}'
    assert truncate_text_payload("x" * 100, 0) == "x" * 100

    marker = json.loads(truncate_text_payload("é" * 3000, 1024))
    assert marker["truncated"] is True
    # Measured in bytes, as the database sees it, not in characters.
    assert marker["bytes"] == 6000
    assert len(marker["preview"].encode("utf-8")) <= 2048


def test_as_model_call_stat_passes_the_pydantic_model_through():
    stats = a_model_call()
    assert as_model_call_stat(stats) is stats


def test_as_model_call_stat_converts_an_orm_row_with_dict_payloads():
    row = ModelCallRow(
        call_type="embedding",
        model="fake/tiny",
        request_data={"texts": ["a", "b"]},
        response_data="already text",
        response_code=200,
        prompt_tokens=2,
        total_tokens=2,
        batch_size=2,
        duration_seconds=Decimal("0.125"),
    )
    stats = as_model_call_stat(row)

    assert isinstance(stats, ModelCallStat)
    assert json.loads(stats.request_data) == {"texts": ["a", "b"]}
    assert stats.response_data == "already text"
    assert stats.batch_size == 2
    assert stats.duration_seconds == 0.125
    assert isinstance(stats.duration_seconds, float)

    bare = as_model_call_stat(ModelCallRow(call_type="llm", model="m"))
    assert bare.request_data is None
    assert bare.duration_seconds is None


async def test_record_nodes_false_keeps_the_model_calls():
    task_logger = MemoryTaskLogger(record_nodes=False)
    log_one_node(task_logger)
    task_logger.log_model_call(a_model_call(), "a1", run_id="r1")
    await task_logger.flush()

    assert task_logger.records == []
    assert [call.total_tokens for call in task_logger.model_calls] == [15]


async def test_record_payloads_false_keeps_names_timings_and_tokens():
    task_logger = MemoryTaskLogger(record_payloads=False)
    log_one_node(task_logger)
    task_logger.log_model_call(a_model_call(), "a1", session_id="s1", run_id="r1")
    await task_logger.flush()

    (node,) = task_logger.records
    assert (node.inputs, node.output, node.prompt) == (None, None, None)
    assert node.name == "answer"
    assert node.duration_seconds == 0.25
    assert node.errors == ["late"]
    assert node.seq == 3

    (call,) = task_logger.model_calls
    assert call.request_data is None and call.response_data is None
    assert (call.prompt_tokens, call.completion_tokens) == (11, 4)
    assert call.duration_seconds == 0.5
    assert (call.agent_id, call.session_id, call.run_id) == ("a1", "s1", "r1")


async def test_the_payload_cap_covers_model_calls():
    task_logger = MemoryTaskLogger(max_payload_bytes=256)
    task_logger.log_model_call(
        a_model_call(request_data=json.dumps({"history": "x" * 5000}))
    )
    await task_logger.flush()

    (call,) = task_logger.model_calls
    marker = json.loads(call.request_data)
    assert marker["truncated"] is True
    assert marker["bytes"] > 5000
    assert marker["preview"].startswith('{"history"')
    # A payload under the cap is stored as it came.
    assert call.response_data == json.dumps({"text": "noted"})


async def test_an_orm_row_is_recorded_as_the_pydantic_model():
    task_logger = MemoryTaskLogger()
    task_logger.log_model_call(
        ModelCallRow(call_type="llm", model="m", request_data={"q": 1}),
        "a1",
    )
    await task_logger.flush()

    (call,) = task_logger.model_calls
    assert isinstance(call, ModelCallRecord)
    assert json.loads(call.request_data) == {"q": 1}
    assert call.agent_id == "a1"


async def test_model_calls_for_run_picks_out_one_run(task_logger):
    task_logger.log_model_call(a_model_call(), "a1", session_id="s1", run_id="r1")
    task_logger.log_model_call(a_model_call(), "a1", session_id="s1", run_id="r2")
    task_logger.log_model_call(a_model_call())
    await task_logger.flush()

    (call,) = task_logger.model_calls_for_run("r2")
    assert (call.agent_id, call.session_id, call.run_id) == ("a1", "s1", "r2")
    assert len(task_logger.model_calls_for_run(None)) == 3
    assert task_logger.model_calls_for_run("unknown") == []
    # A call made outside a recorded run carries no ids.
    assert task_logger.model_calls[-1].run_id is None


async def test_stats_bridge_and_accumulator_forward_the_run(task_logger):
    StatsBridge(task_logger, "a1", session_id="s1", run_id="r1").receive_model_stats(
        a_model_call()
    )
    accumulator = TokenAccumulator(task_logger, "a2", session_id="s2", run_id="r2")
    accumulator.receive_model_stats(a_model_call())
    await task_logger.flush()

    ids = [(c.agent_id, c.session_id, c.run_id) for c in task_logger.model_calls]
    assert sorted(ids) == [("a1", "s1", "r1"), ("a2", "s2", "r2")]
    assert accumulator.summary()["total_tokens"] == 15


async def test_tee_forwards_the_ids_and_keeps_each_loggers_options():
    full = MemoryTaskLogger()
    slim = MemoryTaskLogger(record_nodes=False, record_payloads=False)
    tee = TeeTaskLogger(full, None, slim)
    assert tee.loggers == [full, slim]

    log_one_node(tee)
    tee.log_model_call(a_model_call(), "a1", session_id="s1", run_id="r1")
    await tee.flush()

    assert [record.name for record in full.records] == ["answer"]
    assert slim.records == []
    for memory in (full, slim):
        (call,) = memory.model_calls
        assert (call.agent_id, call.session_id, call.run_id) == ("a1", "s1", "r1")
    assert full.model_calls[0].request_data is not None
    assert slim.model_calls[0].request_data is None
