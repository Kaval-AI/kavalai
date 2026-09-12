import json
from uuid import UUID

import pytest
from sqlalchemy import select

from kavalai.agent_service import AgentService
from kavalai.db import ModelCallStat, Task
from kavalai.llm_clients.base_client import ModelCallStat as PydModelCallStat
from kavalai.testing import ScriptedLlmClient
from kavalai.workflow import WorkflowEngine
from kavalai.workflow.tasklog.base import StatsBridge
from kavalai.workflow.tasklog.postgres import PostgresTaskLogger

TWO_CALLS = {
    "name": "attribution",
    "description": "Two model calls per run.",
    "llm_model": "openai/scripted",
    "data_types": {
        "input": {
            "type": "object",
            "properties": {"user_message": {"type": "string"}},
        },
        "classification": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
        },
        "output": {
            "type": "object",
            "properties": {"agent_response": {"type": "string"}},
        },
    },
    "nodes": [
        {"name": "s", "type": "start", "next": "classify"},
        {
            "name": "classify",
            "type": "llm",
            "prompt": "Classify {{ context.input.user_message }}",
            "output": "classification",
            "next": "answer",
        },
        {
            "name": "answer",
            "type": "llm",
            "prompt": "Answer the visitor.",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ],
}


def a_model_call(**overrides):
    values = dict(
        call_type="llm",
        model="openai/x",
        request_data=json.dumps({"messages": ["the whole transcript"]}),
        response_data=json.dumps({"text": "noted"}),
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
    )
    values.update(overrides)
    return PydModelCallStat(**values)


def log_one_node(tlog, agent, session_obj, run, output=None):
    tlog.log_node(
        run_id=str(run.id),
        session_id=str(session_obj.id),
        agent_id=str(agent.id),
        node_name="classify",
        node_type="llm",
        inputs={"x": 1},
        output={"intent": "greet"} if output is None else output,
        prompt="classify the intent",
        duration=0.5,
    )


async def all_rows(session_maker, model):
    async with session_maker() as session:
        return (await session.execute(select(model))).scalars().all()


@pytest.mark.asyncio
class TestPostgresTaskLogger:
    async def test_log_node_and_model_call(self, agents_session_maker, agents_db):
        service = AgentService(agents_session_maker)
        agent, session_obj, run = await service.initialize_workflow_run(
            agent_name="tl_wf"
        )

        tlog = PostgresTaskLogger.from_session_maker(agents_session_maker)
        log_one_node(tlog, agent, session_obj, run)
        tlog.log_model_call(
            a_model_call(),
            agent_id=str(agent.id),
            session_id=str(session_obj.id),
            run_id=str(run.id),
        )
        await tlog.flush()

        (task,) = await all_rows(agents_session_maker, Task)
        assert task.name == "classify"
        assert task.node_type == "llm"

        (stat,) = await all_rows(agents_session_maker, ModelCallStat)
        assert stat.call_type == "llm"
        assert stat.total_tokens == 5
        assert (stat.agent_id, stat.session_id, stat.run_id) == (
            agent.id,
            session_obj.id,
            run.id,
        )

    async def test_a_plain_output_is_wrapped(self, agents_session_maker, agents_db):
        service = AgentService(agents_session_maker)
        agent, session_obj, run = await service.initialize_workflow_run(
            agent_name="tl_plain"
        )
        tlog = PostgresTaskLogger(service)
        log_one_node(tlog, agent, session_obj, run, output="tere")
        await tlog.flush()

        (task,) = await all_rows(agents_session_maker, Task)
        assert task.output == {"result": "tere"}

    async def test_log_node_skips_without_run(self, agents_session_maker, agents_db):
        tlog = PostgresTaskLogger.from_session_maker(agents_session_maker)
        # No run/session -> nothing can be persisted; must not raise.
        tlog.log_node(
            run_id=None,
            session_id=None,
            agent_id=None,
            node_name="start",
            node_type="start",
            inputs=None,
            output=None,
            prompt=None,
            duration=0.0,
        )
        await tlog.flush()
        assert await all_rows(agents_session_maker, Task) == []

    async def test_a_call_outside_a_run_has_no_ids(
        self, agents_session_maker, agents_db
    ):
        tlog = PostgresTaskLogger.from_session_maker(agents_session_maker)
        tlog.log_model_call(a_model_call())
        await tlog.flush()

        (stat,) = await all_rows(agents_session_maker, ModelCallStat)
        assert (stat.agent_id, stat.session_id, stat.run_id) == (None, None, None)

    async def test_record_nodes_false_keeps_the_model_calls(
        self, agents_session_maker, agents_db
    ):
        service = AgentService(agents_session_maker)
        agent, session_obj, run = await service.initialize_workflow_run(
            agent_name="tl_no_nodes"
        )
        tlog = PostgresTaskLogger.from_session_maker(
            agents_session_maker, record_nodes=False
        )
        log_one_node(tlog, agent, session_obj, run)
        tlog.log_model_call(a_model_call(), str(agent.id), run_id=str(run.id))
        await tlog.flush()

        assert await all_rows(agents_session_maker, Task) == []
        (stat,) = await all_rows(agents_session_maker, ModelCallStat)
        assert stat.run_id == run.id

    async def test_record_payloads_false_stores_no_payloads(
        self, agents_session_maker, agents_db
    ):
        service = AgentService(agents_session_maker)
        agent, session_obj, run = await service.initialize_workflow_run(
            agent_name="tl_no_payloads"
        )
        tlog = PostgresTaskLogger(service, record_payloads=False)
        log_one_node(tlog, agent, session_obj, run)
        tlog.log_model_call(a_model_call(), str(agent.id), run_id=str(run.id))
        await tlog.flush()

        (task,) = await all_rows(agents_session_maker, Task)
        assert (task.inputs, task.output, task.prompt) == (None, None, None)
        assert task.name == "classify"
        assert float(task.duration_seconds) == 0.5

        (stat,) = await all_rows(agents_session_maker, ModelCallStat)
        assert stat.request_data is None and stat.response_data is None
        assert (stat.prompt_tokens, stat.completion_tokens) == (3, 2)

    async def test_the_payload_cap_covers_model_calls(
        self, agents_session_maker, agents_db
    ):
        tlog = PostgresTaskLogger.from_session_maker(
            agents_session_maker, max_payload_bytes=256
        )
        tlog.log_model_call(
            a_model_call(request_data=json.dumps({"history": "x" * 5000}))
        )
        await tlog.flush()

        (stat,) = await all_rows(agents_session_maker, ModelCallStat)
        marker = json.loads(stat.request_data)
        assert marker["truncated"] is True
        assert marker["bytes"] > 5000

    async def test_stats_bridge_attributes_calls_to_a_run(
        self, agents_session_maker, agents_db
    ):
        service = AgentService(agents_session_maker)
        agent, session_obj, run = await service.initialize_workflow_run(
            agent_name="tl_bridge"
        )
        tlog = PostgresTaskLogger(service)
        bridge = StatsBridge(
            tlog, str(agent.id), session_id=str(session_obj.id), run_id=str(run.id)
        )
        bridge.receive_model_stats(a_model_call(call_type="embedding"))
        await tlog.flush()

        (stat,) = await service.get_model_call_stats(run_id=run.id)
        assert stat.call_type == "embedding"
        assert stat.session_id == session_obj.id

    async def test_an_engine_run_attributes_every_model_call(
        self, agents_session_maker, agents_db
    ):
        service = AgentService(agents_session_maker)
        tlog = PostgresTaskLogger(service)
        client = ScriptedLlmClient(
            [
                {"intent": "greeting"},
                {"agent_response": "Tere!"},
                {"intent": "question"},
                {"agent_response": "Yes, from noon."},
            ]
        )
        engine = WorkflowEngine.from_dict(
            TWO_CALLS, agent_service=service, task_logger=tlog, client_factory=client
        )

        first = await engine.run({"user_message": "Tere"}, external_id="visitor-7")
        second = await engine.run(
            {"user_message": "Open on Sunday?"}, external_id="visitor-7"
        )
        await tlog.flush()

        assert first.session_id == second.session_id
        assert first.run_id != second.run_id

        by_run: dict[str, list[ModelCallStat]] = {}
        for stat in await all_rows(agents_session_maker, ModelCallStat):
            by_run.setdefault(str(stat.run_id), []).append(stat)
        assert set(by_run) == {first.run_id, second.run_id}

        for state in (first, second):
            calls = by_run[state.run_id]
            assert len(calls) == 2
            assert {str(call.session_id) for call in calls} == {state.session_id}
            assert {str(call.agent_id) for call in calls} == {state.agent_id}
            assert (
                sum(call.total_tokens for call in calls)
                == state.token_usage["total_tokens"]
            )

        assert len(await service.get_model_call_stats(run_id=UUID(first.run_id))) == 2
        per_session = await service.get_model_call_stats(
            session_id=UUID(first.session_id)
        )
        assert len(per_session) == 4
