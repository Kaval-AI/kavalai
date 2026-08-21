import pytest

from kavalai.workflow.tasklog.base import StatsBridge, TokenAccumulator
from kavalai.workflow.tasklog.sqlite import SqliteTaskLogger
from kavalai.llm_clients.base_client import ModelCallStat


@pytest.fixture
async def task_logger():
    tl = SqliteTaskLogger()
    yield tl
    await tl.close()


async def _fetchall(logger, query):
    conn = await logger._connect()
    async with conn.execute(query) as cur:
        return await cur.fetchall()


async def test_log_node(task_logger):
    task_logger.log_node(
        run_id="r1",
        session_id="s1",
        agent_id="a1",
        node_name="classify",
        node_type="llm",
        inputs={"x": 1},
        output={"intent": "greet"},
        prompt="classify",
        duration=0.5,
    )
    await task_logger.flush()
    rows = await _fetchall(
        task_logger, "SELECT name, node_type, inputs, output, prompt FROM tasks"
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "classify"
    assert rows[0]["node_type"] == "llm"
    assert '"intent"' in rows[0]["output"]
    assert '"x"' in rows[0]["inputs"]
    assert rows[0]["prompt"] == "classify"


async def test_log_node_with_none_inputs(task_logger):
    task_logger.log_node(
        run_id=None,
        session_id=None,
        agent_id=None,
        node_name="n",
        node_type="function",
        inputs=None,
        output=None,
    )
    await task_logger.flush()
    rows = await _fetchall(task_logger, "SELECT inputs, output FROM tasks")
    assert rows[0]["inputs"] is None
    assert rows[0]["output"] is None


async def test_log_model_call(task_logger):
    stats = ModelCallStat(
        call_type="llm",
        model="openai/gpt",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        duration_seconds=1.2,
    )
    task_logger.log_model_call(stats, agent_id="a1")
    await task_logger.flush()
    rows = await _fetchall(
        task_logger,
        "SELECT call_type, model, agent_id, total_tokens FROM model_call_stats",
    )
    assert len(rows) == 1
    assert rows[0]["call_type"] == "llm"
    assert rows[0]["model"] == "openai/gpt"
    assert rows[0]["agent_id"] == "a1"
    assert rows[0]["total_tokens"] == 15


async def test_stats_bridge_forwards_to_logger(task_logger):
    bridge = StatsBridge(task_logger, agent_id="agent-x")
    bridge.receive_model_stats(
        ModelCallStat(call_type="llm", model="openai/gpt", total_tokens=3)
    )
    await task_logger.flush()
    rows = await _fetchall(
        task_logger, "SELECT agent_id, total_tokens FROM model_call_stats"
    )
    assert rows[0]["agent_id"] == "agent-x"
    assert rows[0]["total_tokens"] == 3


async def test_token_accumulator_aggregates_and_forwards(task_logger):
    acc = TokenAccumulator(task_logger, agent_id="a1")
    acc.receive_model_stats(
        ModelCallStat(
            call_type="llm", prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
    )
    acc.receive_model_stats(
        ModelCallStat(call_type="llm", prompt_tokens=2, total_tokens=2)
    )  # missing completion_tokens counts as 0
    assert acc.summary() == {
        "model_calls": 2,
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }
    # Each call was also forwarded to the task logger.
    await task_logger.flush()
    rows = await _fetchall(task_logger, "SELECT agent_id FROM model_call_stats")
    assert len(rows) == 2 and all(r["agent_id"] == "a1" for r in rows)


def test_token_accumulator_without_logger():
    acc = TokenAccumulator()
    acc.receive_model_stats(ModelCallStat(call_type="llm", total_tokens=9))
    assert acc.summary()["total_tokens"] == 9
    assert acc.summary()["model_calls"] == 1


async def test_flush_without_tasks_is_safe(task_logger):
    await task_logger.flush()  # nothing scheduled


async def test_base_close_default_flushes():
    # A minimal TaskLogger using the ABC's default close()/flush().
    class MinimalLogger(SqliteTaskLogger.__bases__[0]):
        async def _log_node_impl(self, **kwargs):
            return None

        async def _log_model_call_impl(self, stats, agent_id):
            return None

    logger = MinimalLogger()
    logger.log_model_call(ModelCallStat(call_type="llm"))
    await logger.close()  # default close() awaits flush()
    assert logger._background_tasks == set()


async def test_background_exception_is_swallowed(task_logger, caplog):
    # Force a backend failure by logging a stat after the connection is closed
    # in a way that raises inside the background task; the error is logged, not
    # raised to the caller.
    await task_logger._connect()
    await task_logger._conn.close()
    task_logger._conn = None

    class Boom(SqliteTaskLogger):
        async def _connect(self):
            raise RuntimeError("boom")

    task_logger._connect = Boom._connect.__get__(task_logger)
    task_logger.log_model_call(ModelCallStat(call_type="llm"))
    await task_logger.flush()  # should not raise


@pytest.mark.asyncio
async def test_logged_rows_can_be_read_back_without_sql():
    """Reading the log used to mean `_connect()` and hand-written SQL."""
    logger = SqliteTaskLogger()
    try:
        logger.log_node(
            run_id="run-1",
            session_id="s",
            agent_id="a",
            node_name="classify",
            node_type="llm",
            inputs={"user_message": "hei"},
            output={"intent": "greeting"},
            prompt="classify this",
            duration=0.5,
            errors=None,
        )
        logger.log_node(
            run_id="run-2",
            session_id="s",
            agent_id="a",
            node_name="answer",
            node_type="llm",
            inputs=None,
            output="tere",
            prompt=None,
            duration=0.1,
            errors=["nope"],
        )
        logger.log_model_call(
            ModelCallStat(
                call_type="llm",
                model="openai/gpt-4o-mini",
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                cached_prompt_tokens=8,
                response_code=200,
            ),
            agent_id="a",
        )

        # No explicit flush: the reader awaits the pending background writes,
        # otherwise it races them and returns a short list now and then.
        tasks = await logger.get_tasks()
        assert {task["name"] for task in tasks} == {"classify", "answer"}

        (one,) = await logger.get_tasks(run_id="run-1")
        assert one["inputs"] == {"user_message": "hei"}
        assert one["output"] == {"intent": "greeting"}
        assert one["errors"] is None

        (failed,) = await logger.get_tasks(run_id="run-2")
        assert failed["errors"] == ["nope"]

        (call,) = await logger.get_model_calls()
        assert call["model"] == "openai/gpt-4o-mini"
        assert call["cached_prompt_tokens"] == 8
        assert call["response_code"] == 200
    finally:
        await logger.close()
