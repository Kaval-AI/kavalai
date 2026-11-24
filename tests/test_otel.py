import asyncio
import json
import logging
import subprocess
import sys

import pytest
from opentelemetry import context as otel_context
from opentelemetry import metrics as metrics_api
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel

import kavalai.otel as otel
from kavalai import ModelCallStat, WorkflowEngine, WorkflowException, pythontool
from kavalai.llm_clients.base_client import LlmClientParameters
from kavalai.otel import OtelTelemetry, configure_sdk
from kavalai.testing import FakeEmbeddingClient, ScriptedLlmClient

DATA_TYPES = {
    "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
    "classification": {"type": "object", "properties": {"intent": {"type": "string"}}},
    "output": {"type": "object", "properties": {"agent_response": {"type": "string"}}},
}

ANSWER = {"agent_response": "the secret answer"}


class Greeting(BaseModel):
    agent_response: str


@pythontool
def greet(name: str) -> Greeting:
    return Greeting(agent_response=f"hi {name}")


@pythontool
def explode(name: str) -> Greeting:
    raise RuntimeError("tool went wrong")


class Recorder:
    """An :class:`OtelTelemetry` over in-memory exporters of its own, so no
    test touches the process-global providers."""

    def __init__(self, **options):
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.reader])
        self.telemetry = OtelTelemetry(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
            **options,
        )

    @property
    def spans(self):
        return self.exporter.get_finished_spans()

    def span(self, name):
        (found,) = [span for span in self.spans if span.name == name]
        return found

    def children(self, parent):
        return sorted(
            span.name
            for span in self.spans
            if span.parent and span.parent.span_id == parent.context.span_id
        )

    def metrics(self):
        data = self.reader.get_metrics_data()
        found = {}
        for resource_metrics in data.resource_metrics if data else []:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    found[metric.name] = metric
        return found


@pytest.fixture
def recorder():
    return Recorder()


def engine_for(nodes, recorder, replies=(), **extra):
    graph = {
        "name": "wf",
        "description": "test workflow",
        "llm_model": "gemini/m",
        "data_types": dict(DATA_TYPES),
        "nodes": nodes,
    }
    engine = WorkflowEngine.from_dict(
        graph,
        client_factory=ScriptedLlmClient(list(replies)),
        telemetry=recorder.telemetry,
        **extra,
    )
    engine.kernel.register_python_tool("greet", greet)
    engine.kernel.register_python_tool("explode", explode)
    return engine


def llm_node(name="answer", next_node="e", output="output"):
    return {
        "name": name,
        "type": "llm",
        "prompt": "say it",
        "output": output,
        "next": next_node,
    }


def function_node(tool="greet", next_node="e"):
    return {
        "name": "call",
        "type": "function",
        "tool": f"python://{tool}",
        "inputs": {"name": {"type": "literal", "value": "Sam"}},
        "output": "output",
        "next": next_node,
    }


START = {"name": "s", "type": "start", "next": "answer"}
END = {"name": "e", "type": "end", "output": "output"}
LINEAR = [START, llm_node(next_node="call"), function_node(), END]


async def test_linear_run_is_one_trace_with_the_documented_tree(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    state = await engine.run({"user_message": "hello"})
    assert state.status == "completed"

    run = recorder.span("invoke_workflow wf")
    assert run.parent is None
    assert run.kind is SpanKind.INTERNAL
    assert run.status.status_code is StatusCode.UNSET
    assert run.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert run.attributes["kavalai.workflow.name"] == "wf"
    assert run.attributes["gen_ai.agent.name"] == "wf"
    assert run.attributes["kavalai.invocation_id"] == state.invocation_id
    assert run.attributes["kavalai.run.status"] == "completed"
    assert run.attributes["gen_ai.usage.input_tokens"] > 0
    assert run.attributes["gen_ai.usage.output_tokens"] > 0
    # No AgentService, so there are no ids to report.
    assert "kavalai.run.id" not in run.attributes
    assert "gen_ai.conversation.id" not in run.attributes

    assert recorder.children(run) == [
        "end e",
        "function call",
        "llm answer",
        "start s",
    ]
    assert {span.context.trace_id for span in recorder.spans} == {run.context.trace_id}

    node = recorder.span("llm answer")
    assert node.attributes["kavalai.node.name"] == "answer"
    assert node.attributes["kavalai.node.type"] == "llm"
    assert node.attributes["kavalai.node.seq"] == 1

    chat = recorder.span("chat m")
    assert recorder.children(node) == ["chat m"]
    assert chat.kind is SpanKind.CLIENT
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert chat.attributes["gen_ai.request.model"] == "m"
    assert chat.attributes["http.response.status_code"] == 200
    assert chat.attributes["gen_ai.usage.output_tokens"] > 0
    assert chat.start_time <= chat.end_time

    tool = recorder.span("execute_tool greet")
    assert recorder.children(recorder.span("function call")) == ["execute_tool greet"]
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "greet"
    assert tool.attributes["kavalai.tool.uri"] == "python://greet"


async def test_recorded_run_carries_its_ids(recorder):
    from kavalai.agent_service import AgentService
    from kavalai.db import DatabaseManager

    service = AgentService(DatabaseManager().get_sqlite_compat_sessionmaker())
    engine = engine_for(
        [dict(START, next="call"), function_node(), END],
        recorder,
        agent_service=service,
    )
    state = await engine.run({"user_message": "x"}, external_id="ticket-7")

    run = recorder.span("invoke_workflow wf")
    assert run.attributes["kavalai.run.id"] == state.run_id
    assert run.attributes["kavalai.agent.id"] == state.agent_id
    assert run.attributes["gen_ai.conversation.id"] == state.session_id
    assert run.attributes["kavalai.session.external_id"] == "ticket-7"


async def test_no_content_leaves_the_process_by_default(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    await engine.run({"user_message": "a private question"})

    exported = json.dumps(
        [
            [span.name, dict(span.attributes), [e.name for e in span.events]]
            for span in recorder.spans
        ],
        default=str,
    )
    assert "private question" not in exported
    assert "secret answer" not in exported
    assert "hi Sam" not in exported
    assert "kavalai.model_call.request" not in exported


async def test_capture_content_attaches_truncated_payloads():
    recorder = Recorder(capture_content=True, max_content_bytes=64)
    engine = engine_for([START, llm_node(), END], recorder, [ANSWER])
    await engine.run({"user_message": "a private question " * 20})

    chat = recorder.span("chat m")
    request = json.loads(chat.attributes["kavalai.model_call.request"])
    assert request["truncated"] is True and request["bytes"] > 64
    assert "secret answer" in chat.attributes["kavalai.model_call.response"]


async def test_agent_node_tool_calls_are_children_of_the_node(recorder):
    nodes = [
        dict(START, next="do"),
        {
            "name": "do",
            "type": "agent",
            "prompt": "greet Sam",
            "output": "output",
            "max_steps": 3,
            "next": "e",
        },
        END,
    ]
    step = {
        "instructions": "call the tool",
        "tool_calls": [
            {"name": "python://greet", "args": {"name": "Sam"}}
        ],
        "output": None,
    }
    final = {"instructions": "done", "tool_calls": [], "output": ANSWER}
    engine = engine_for(nodes, recorder, [step, final])
    state = await engine.run({"user_message": "x"})
    assert state.status == "completed"

    node = recorder.span("agent do")
    assert node.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert node.attributes["gen_ai.agent.name"] == "do"
    assert recorder.children(node) == ["chat m", "chat m", "execute_tool greet"]


async def test_parallel_branches_are_children_of_the_parallel_span(recorder):
    nodes = [
        dict(START, next="fan"),
        {"name": "fan", "type": "parallel", "branches": ["a", "b"], "next": "e"},
        llm_node("a", output="classification"),
        llm_node("b"),
        END,
    ]
    engine = engine_for(nodes, recorder, [{"intent": "x"}, ANSWER])
    await engine.run({"user_message": "x"})

    fan = recorder.span("parallel fan")
    assert recorder.children(fan) == ["llm a", "llm b"]
    assert recorder.children(recorder.span("llm a")) == ["chat m"]
    assert recorder.children(recorder.span("llm b")) == ["chat m"]
    assert recorder.children(recorder.span("invoke_workflow wf")) == [
        "end e",
        "parallel fan",
        "start s",
    ]


async def test_run_with_a_timeout_keeps_the_tree(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER], run_timeout=30)
    await engine.run({"user_message": "x"})

    run = recorder.span("invoke_workflow wf")
    assert len(recorder.children(run)) == 4
    assert recorder.children(recorder.span("llm answer")) == ["chat m"]


async def test_rag_query_embedding_is_a_child_of_the_node(recorder):
    class Hit(BaseModel):
        id: str = "1"
        source_id: str = "s"
        similarity: float = 0.9
        content: str = "passage"
        rag_metadata: dict = {}

    class Service:
        async def query(self, *, text, stats_receiver=None, **_):
            _, stats = await FakeEmbeddingClient().compute_embeddings([text])
            stats_receiver.receive_model_stats(stats)
            return [Hit()]

    nodes = [
        dict(START, next="find"),
        {
            "name": "find",
            "type": "rag_query",
            "query": "{{ input.user_message }}",
            "output": "passages",
            "next": "call",
        },
        function_node(),
        END,
    ]
    engine = engine_for(nodes, recorder, rag_services=Service())
    await engine.run({"user_message": "x"})

    embedding = recorder.span("embeddings hashing")
    assert recorder.children(recorder.span("rag_query find")) == ["embeddings hashing"]
    assert embedding.attributes["gen_ai.operation.name"] == "embeddings"
    assert embedding.attributes["gen_ai.provider.name"] == "fake"
    assert embedding.attributes["kavalai.batch_size"] == 1


async def test_branch_node_reports_where_it_routed(recorder):
    nodes = [
        dict(START, next="route"),
        {
            "name": "route",
            "type": "switch",
            "expr": "input.user_message",
            "cases": {"hello": "call"},
            "default": "call",
        },
        function_node(),
        END,
    ]
    engine = engine_for(nodes, recorder)
    await engine.run({"user_message": "something private"})

    route = recorder.span("switch route")
    assert route.attributes["kavalai.branch.taken"] == "call"
    assert route.attributes["kavalai.branch.matched"] is False
    assert "something private" not in json.dumps(dict(route.attributes))


async def test_concurrent_runs_never_share_a_trace(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER] * 6)
    await asyncio.gather(*(engine.run({"user_message": str(i)}) for i in range(6)))

    runs = [span for span in recorder.spans if span.name == "invoke_workflow wf"]
    assert len({run.context.trace_id for run in runs}) == 6
    by_id = {span.context.span_id: span for span in recorder.spans}
    for span in recorder.spans:
        if span.parent is not None:
            assert by_id[span.parent.span_id].context.trace_id == span.context.trace_id
    for run in runs:
        assert len(recorder.children(run)) == 4


async def test_current_span_becomes_the_parent_of_the_run(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    tracer = recorder.tracer_provider.get_tracer("host")
    with tracer.start_as_current_span("POST /chat") as request:
        await engine.run({"user_message": "x"})

    run = recorder.span("invoke_workflow wf")
    assert run.parent.span_id == request.get_span_context().span_id


async def test_consumer_never_sees_an_engine_span_as_current(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    seen = set()
    async for _ in engine.run_stream({"user_message": "x"}):
        seen.add(trace.get_current_span().get_span_context().is_valid)
    assert seen == {False}


async def test_abandoned_run_ends_its_spans_without_a_detach_error(recorder, caplog):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    before = otel_context.get_current()

    async def consume():
        stream = engine.run_stream({"user_message": "x"})
        async for event in stream:
            if event.type == "node_started" and event.name == "answer":
                break
        await stream.aclose()

    with caplog.at_level(logging.ERROR, logger="opentelemetry"):
        # Abandoned from a task of its own: finalising a generator in a task
        # other than the one that drove it is what a naive context manager
        # around a ``yield`` cannot survive.
        await asyncio.create_task(consume())
        # The abandoned walk is finalised by the event loop, a generator at
        # a time, so the node's span ends shortly after the run's.
        for _ in range(100):
            if any(span.name == "llm answer" for span in recorder.spans):
                break
            await asyncio.sleep(0.01)

    assert "detach" not in caplog.text.lower()
    assert otel_context.get_current() == before
    run = recorder.span("invoke_workflow wf")
    assert run.status.status_code is StatusCode.ERROR
    assert run.attributes["kavalai.run.status"] == "failed"
    assert recorder.span("llm answer").end_time is not None


async def test_failing_tool_marks_tool_node_and_run(recorder):
    engine = engine_for(
        [dict(START, next="call"), function_node("explode"), END], recorder
    )
    with pytest.raises(WorkflowException):
        await engine.run({"user_message": "x"})

    tool = recorder.span("execute_tool explode")
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.attributes["error.type"]
    assert [event.name for event in tool.events] == ["exception"]

    node = recorder.span("function call")
    assert node.status.status_code is StatusCode.ERROR
    assert node.attributes["error.type"] == tool.attributes["error.type"]

    run = recorder.span("invoke_workflow wf")
    assert run.status.status_code is StatusCode.ERROR
    assert "tool went wrong" in run.status.description
    assert run.attributes["kavalai.run.status"] == "failed"

    durations = recorder.metrics()["kavalai.tool.call.duration"].data.data_points
    assert [dict(point.attributes) for point in durations] == [
        {"gen_ai.tool.name": "explode", "error.type": tool.attributes["error.type"]}
    ]


async def test_failed_model_call_is_an_error_span(recorder):
    class Refused(Exception):
        status_code = 400

    engine = engine_for([START, llm_node(), END], recorder, [Refused("no")])
    with pytest.raises(WorkflowException):
        await engine.run({"user_message": "x"})

    chat = recorder.span("chat m")
    assert chat.status.status_code is StatusCode.ERROR
    assert chat.attributes["error.type"] == "400"
    assert chat.attributes["http.response.status_code"] == 400
    assert recorder.span("llm answer").status.status_code is StatusCode.ERROR

    (point,) = recorder.metrics()["gen_ai.client.operation.duration"].data.data_points
    assert point.attributes["error.type"] == "400"
    assert "gen_ai.client.token.usage" not in recorder.metrics()


async def test_truncated_call_keeps_its_token_counts(recorder):
    client = ScriptedLlmClient(
        ["word " * 200],
        llm_client_parameters=LlmClientParameters(max_output_tokens=5),
        model_stats_receiver=recorder.telemetry,
    )
    with pytest.raises(Exception):
        await client.prompt("write a lot")

    chat = recorder.span("chat scripted")
    assert chat.status.status_code is StatusCode.ERROR
    assert chat.attributes["error.type"] == "_OTHER"
    assert chat.attributes["gen_ai.usage.output_tokens"] > 0
    assert "word" not in json.dumps(dict(chat.attributes))


async def test_bare_client_reports_through_the_stats_receiver(recorder):
    client = ScriptedLlmClient(["hello"], model_stats_receiver=recorder.telemetry)
    assert await client.prompt("hi") == "hello"

    chat = recorder.span("chat scripted")
    assert chat.parent is None
    assert chat.attributes["gen_ai.provider.name"] == "fake"


def test_model_call_span_covers_the_reported_duration(recorder):
    recorder.telemetry.model_call(
        ModelCallStat(
            call_type="llm",
            model="local-model",
            response_code=200,
            prompt_tokens=10,
            completion_tokens=3,
            cached_prompt_tokens=4,
            reasoning_tokens=2,
            duration_seconds=1.5,
        )
    )
    chat = recorder.span("chat local-model")
    assert chat.end_time - chat.start_time == 1_500_000_000
    assert "gen_ai.provider.name" not in chat.attributes
    assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 4
    assert chat.attributes["gen_ai.usage.reasoning.output_tokens"] == 2


def test_model_call_without_a_name_or_duration(recorder):
    recorder.telemetry.model_call(ModelCallStat(call_type="embedding", total_tokens=7))
    span = recorder.span("embeddings unknown")
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.start_time == span.end_time


async def test_metrics_units_and_attributes(recorder):
    engine = engine_for(LINEAR, recorder, [ANSWER])
    await engine.run({"user_message": "x"})
    found = recorder.metrics()

    assert {name: metric.unit for name, metric in found.items()} == {
        "gen_ai.client.token.usage": "{token}",
        "gen_ai.client.operation.duration": "s",
        "kavalai.workflow.run.duration": "s",
        "kavalai.workflow.node.duration": "s",
        "kavalai.tool.call.duration": "s",
    }

    def attributes(name):
        return [dict(p.attributes) for p in found[name].data.data_points]

    model = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "gcp.gemini",
        "gen_ai.request.model": "m",
    }
    assert attributes("gen_ai.client.operation.duration") == [model]
    assert sorted(
        point["gen_ai.token.type"] for point in attributes("gen_ai.client.token.usage")
    ) == ["input", "output"]
    assert attributes("kavalai.workflow.run.duration") == [
        {"kavalai.workflow.name": "wf", "kavalai.run.status": "completed"}
    ]
    assert {
        "kavalai.workflow.name": "wf",
        "kavalai.node.name": "answer",
        "kavalai.node.type": "llm",
    } in attributes("kavalai.workflow.node.duration")
    assert attributes("kavalai.tool.call.duration") == [{"gen_ai.tool.name": "greet"}]

    # Cardinality is bounded by the graph and the model list: no id anywhere.
    keys = {key for name in found for point in attributes(name) for key in point}
    assert not {key for key in keys if key.endswith(".id") or "invocation" in key}


def test_end_run_measures_a_run_that_reports_no_duration(recorder):
    span = recorder.telemetry.start_run(
        workflow="wf",
        agent_id=None,
        session_id=None,
        run_id=None,
        external_id=None,
        invocation_id="abc",
    )
    recorder.telemetry.end_run(
        span, status="completed", error=None, token_usage=None, duration_seconds=None
    )
    (point,) = recorder.metrics()["kavalai.workflow.run.duration"].data.data_points
    assert point.count == 1
    assert (
        recorder.span("invoke_workflow wf").attributes["gen_ai.usage.input_tokens"] == 0
    )


def test_node_started_outside_a_run_has_no_workflow_label(recorder):
    span = recorder.telemetry.start_node(name="n", node_type="llm", seq=None)
    recorder.telemetry.end_node(span, attributes={"taken": None, "matched": True})

    node = recorder.span("llm n")
    assert "kavalai.node.seq" not in node.attributes
    assert "kavalai.branch.taken" not in node.attributes
    assert node.attributes["kavalai.branch.matched"] is True
    (point,) = recorder.metrics()["kavalai.workflow.node.duration"].data.data_points
    assert "kavalai.workflow.name" not in point.attributes


async def test_within_relays_a_plain_async_iterator(recorder):
    class Numbers:
        def __init__(self):
            self.left = [1, 2]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.left:
                raise StopAsyncIteration
            return self.left.pop(0)

    span = recorder.telemetry.start_node(name="n", node_type="llm", seq=0)
    assert [n async for n in recorder.telemetry.within(span, Numbers())] == [1, 2]


def test_tool_names():
    assert otel._tool_name("rest://weather.forecast [GET]") == "forecast"
    assert otel._tool_name("python://greet") == "greet"
    assert otel._tool_name("greet") == "greet"


def test_configure_sdk_installs_otlp_providers(monkeypatch):
    installed = {}
    monkeypatch.setattr(
        trace, "set_tracer_provider", lambda p: installed.update(traces=p)
    )
    monkeypatch.setattr(
        metrics_api, "set_meter_provider", lambda p: installed.update(metrics=p)
    )
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    exporters = []
    real_span_exporter = otel._span_exporter
    real_metric_exporter = otel._metric_exporter
    monkeypatch.setattr(
        otel,
        "_span_exporter",
        lambda: exporters.append("span") or InMemorySpanExporter(),
    )
    monkeypatch.setattr(
        otel,
        "_metric_exporter",
        lambda: exporters.append("metric") or real_metric_exporter(),
    )

    configure_sdk(service_name="bakery")

    assert exporters == ["span", "metric"]
    assert installed["traces"].resource.attributes["service.name"] == "bakery"
    assert installed["metrics"]._sdk_config.resource.attributes["service.name"] == (
        "bakery"
    )
    installed["traces"].shutdown()
    installed["metrics"].shutdown(timeout_millis=1)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert isinstance(real_span_exporter(), OTLPSpanExporter)


def test_configure_sdk_respects_the_environment_and_the_switches(monkeypatch):
    installed = {}
    monkeypatch.setattr(
        trace, "set_tracer_provider", lambda p: installed.update(traces=p)
    )
    monkeypatch.setattr(
        metrics_api, "set_meter_provider", lambda p: installed.update(metrics=p)
    )
    monkeypatch.setattr(otel, "_span_exporter", InMemorySpanExporter)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")

    configure_sdk(service_name="bakery", metrics=False)
    assert set(installed) == {"traces"}
    assert installed["traces"].resource.attributes["service.name"] == "from-env"
    installed["traces"].shutdown()

    installed.clear()
    configure_sdk(service_name="bakery", traces=False, metrics=False)
    assert installed == {}


def test_missing_package_names_the_extra():
    code = (
        "import sys\n"
        "sys.modules['opentelemetry'] = None\n"
        "try:\n"
        "    import kavalai.otel\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert 'pip install "kavalai[otel]"' in result.stdout


def test_import_kavalai_imports_no_opentelemetry():
    code = (
        "import sys, kavalai, kavalai.telemetry\n"
        "print(sorted(m for m in sys.modules if m.startswith('opentelemetry')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
