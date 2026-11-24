"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

OpenTelemetry export: a run as a trace, and a small set of metrics.

:class:`OtelTelemetry` implements :class:`kavalai.telemetry.Telemetry` over
the OpenTelemetry API. Passed to a :class:`~kavalai.WorkflowEngine` it emits
one trace per run — run, node, tool call and model call spans — and the
histograms listed below. :func:`configure_sdk` installs process-global
providers that export over OTLP/HTTP; a host that has configured
OpenTelemetry already does not call it.

The module is not imported by ``kavalai/__init__.py`` and needs the ``otel``
extra. It reads no environment variable: the OTLP exporters read the standard
``OTEL_EXPORTER_OTLP_*`` variables themselves.

Names follow the OpenTelemetry GenAI semantic conventions where one exists
and carry a ``kavalai.`` prefix where none does. The conventions have
*Development* status, so every name this module emits is one of the constants
below and a rename is a change to this block. ``SEMCONV_VERSION`` is the
semantic-conventions release they were taken from: 1.41.0, the last one
before the GenAI conventions moved to a repository of their own.

``PROVIDER_NAMES`` maps a Kaval.AI provider prefix onto the convention's
``gen_ai.provider.name`` value where the two differ. ``TOKEN_BUCKETS`` and
``DURATION_BUCKETS`` are the histogram boundaries the conventions advise.
"""

import time
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from opentelemetry import context as otel_context
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "kavalai.otel requires the optional 'opentelemetry-sdk' package. "
        'Install it with: pip install "kavalai[otel]"'
    ) from error

from kavalai.llm_clients.base_client import ModelCallStat, ModelStatsReceiver
from kavalai.telemetry import Telemetry
from kavalai.workflow.tasklog.base import truncate_text_payload

__all__ = ["OtelTelemetry", "configure_sdk", "SEMCONV_VERSION"]

SEMCONV_VERSION = "1.41.0"
INSTRUMENTATION_NAME = "kavalai"

OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
CONVERSATION_ID = "gen_ai.conversation.id"
AGENT_NAME = "gen_ai.agent.name"
TOOL_NAME = "gen_ai.tool.name"
TOKEN_TYPE = "gen_ai.token.type"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
ERROR_TYPE = "error.type"
HTTP_STATUS_CODE = "http.response.status_code"

WORKFLOW_NAME = "kavalai.workflow.name"
AGENT_ID = "kavalai.agent.id"
RUN_ID = "kavalai.run.id"
RUN_STATUS = "kavalai.run.status"
SESSION_EXTERNAL_ID = "kavalai.session.external_id"
INVOCATION_ID = "kavalai.invocation_id"
NODE_NAME = "kavalai.node.name"
NODE_TYPE = "kavalai.node.type"
NODE_SEQ = "kavalai.node.seq"
BRANCH_PREFIX = "kavalai.branch."
TOOL_URI = "kavalai.tool.uri"
BATCH_SIZE = "kavalai.batch_size"
MODEL_CALL_REQUEST = "kavalai.model_call.request"
MODEL_CALL_RESPONSE = "kavalai.model_call.response"

OP_WORKFLOW = "invoke_workflow"
OP_AGENT = "invoke_agent"
OP_TOOL = "execute_tool"
OP_CHAT = "chat"
OP_EMBEDDINGS = "embeddings"
ERROR_OTHER = "_OTHER"

METRIC_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_OPERATION_DURATION = "gen_ai.client.operation.duration"
METRIC_RUN_DURATION = "kavalai.workflow.run.duration"
METRIC_NODE_DURATION = "kavalai.workflow.node.duration"
METRIC_TOOL_DURATION = "kavalai.tool.call.duration"

PROVIDER_NAMES = {"gemini": "gcp.gemini"}

TOKEN_BUCKETS = [4.0**exponent for exponent in range(14)]
DURATION_BUCKETS = [0.01 * 2.0**exponent for exponent in range(14)]

_WORKFLOW_KEY = otel_context.create_key("kavalai.workflow.name")


@dataclass
class _Handle:
    """What the engine holds for an open run or node span.

    ``context`` is the context in which the span is current; ``labels`` are
    the metric attributes recorded when the span ends.
    """

    span: Span
    context: otel_context.Context
    started: float
    labels: dict[str, Any] = field(default_factory=dict)


def _tool_name(tool_uri: str) -> str:
    """``rest://weather.forecast [GET]`` -> ``forecast``."""
    path = tool_uri.partition("://")[2] or tool_uri
    name = path.rsplit(".", 1)[-1]
    return name.split(" [", 1)[0].strip()


def _split_model(name: Optional[str]) -> tuple[Optional[str], str]:
    """``gemini/gemini-2.5-flash`` -> ``("gcp.gemini", "gemini-2.5-flash")``."""
    provider, separator, model = (name or "").partition("/")
    if not separator:
        return None, provider or "unknown"
    return PROVIDER_NAMES.get(provider, provider), model


def _error_type(error: BaseException) -> str:
    return type(error).__qualname__


class OtelTelemetry(Telemetry, ModelStatsReceiver):
    """Reports runs to OpenTelemetry as traces and metrics.

    One trace per run::

        invoke_workflow {workflow}
          {node_type} {node_name}
            chat {model} | embeddings {model}
            execute_tool {function}

    A run's span is a child of whatever span is current when the run starts,
    and everything a node starts — an outbound HTTP request of a tool, a
    provider SDK's own instrumentation — is a child of the node's span.
    Every span of a run carries its position in the trace; the run span
    carries ``kavalai.run.id`` and the session id, which is how a trace is
    found from a recorded run.

    A model-call span is created when the call has finished, from its
    :class:`~kavalai.ModelCallStat`, with the start time computed from the
    duration. Each attempt of a retried call is a span of its own.

    The instance is also a
    :class:`~kavalai.ModelStatsReceiver`, so
    ``make_client(model, stats_receiver=OtelTelemetry())`` traces a client
    used outside a workflow.

    Args:
        tracer_provider: Provider the spans are created from. Defaults to
            the process-global one, as every OpenTelemetry instrumentation
            does.
        meter_provider: Provider the histograms are created from. Defaults
            to the process-global one.
        capture_content: Whether a model call's request and response are
            attached to its span. Off by default: spans then carry names,
            timings, token counts and error types, and no prompt, answer,
            tool argument or tool result. An OTLP endpoint is commonly
            operated by a third party, which is why the default differs
            from the task logger's.
        max_content_bytes: Cap on each captured payload; a larger one is
            replaced by the marker the task logger uses.
    """

    def __init__(
        self,
        *,
        tracer_provider: Optional[trace.TracerProvider] = None,
        meter_provider: Optional[metrics.MeterProvider] = None,
        capture_content: bool = False,
        max_content_bytes: int = 8192,
    ) -> None:
        self.capture_content = capture_content
        self.max_content_bytes = max_content_bytes
        self._tracer = trace.get_tracer(
            INSTRUMENTATION_NAME, tracer_provider=tracer_provider
        )
        meter = metrics.get_meter(INSTRUMENTATION_NAME, meter_provider=meter_provider)
        self._token_usage = meter.create_histogram(
            METRIC_TOKEN_USAGE,
            unit="{token}",
            description="Tokens used by a model call, by direction.",
            explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
        )
        self._operation_duration = meter.create_histogram(
            METRIC_OPERATION_DURATION,
            unit="s",
            description="Duration of a model call.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self._run_duration = meter.create_histogram(
            METRIC_RUN_DURATION,
            unit="s",
            description="Wall-clock duration of a workflow run.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self._node_duration = meter.create_histogram(
            METRIC_NODE_DURATION,
            unit="s",
            description="Duration of one visit to a workflow node.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self._tool_duration = meter.create_histogram(
            METRIC_TOOL_DURATION,
            unit="s",
            description="Duration of a tool call.",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )

    def start_run(
        self,
        *,
        workflow: str,
        agent_id: Optional[str],
        session_id: Optional[str],
        run_id: Optional[str],
        external_id: Optional[str],
        invocation_id: str,
    ) -> _Handle:
        attributes = {
            OPERATION_NAME: OP_WORKFLOW,
            AGENT_NAME: workflow,
            WORKFLOW_NAME: workflow,
            INVOCATION_ID: invocation_id,
            CONVERSATION_ID: session_id,
            AGENT_ID: agent_id,
            RUN_ID: run_id,
            SESSION_EXTERNAL_ID: external_id,
        }
        span = self._tracer.start_span(
            f"{OP_WORKFLOW} {workflow}",
            attributes={k: v for k, v in attributes.items() if v is not None},
        )
        context = otel_context.set_value(
            _WORKFLOW_KEY, workflow, trace.set_span_in_context(span)
        )
        return _Handle(span, context, time.perf_counter(), {WORKFLOW_NAME: workflow})

    def end_run(
        self,
        span: _Handle,
        *,
        status: str,
        error: Optional[str],
        token_usage: Optional[dict],
        duration_seconds: Optional[float],
    ) -> None:
        usage = token_usage or {}
        span.span.set_attribute(RUN_STATUS, status)
        span.span.set_attribute(USAGE_INPUT_TOKENS, usage.get("prompt_tokens", 0))
        span.span.set_attribute(USAGE_OUTPUT_TOKENS, usage.get("completion_tokens", 0))
        if status == "failed":
            span.span.set_status(Status(StatusCode.ERROR, error))
        span.span.end()
        if duration_seconds is None:
            duration_seconds = time.perf_counter() - span.started
        self._run_duration.record(duration_seconds, {**span.labels, RUN_STATUS: status})

    def start_node(self, *, name: str, node_type: str, seq: Optional[int]) -> _Handle:
        attributes: dict[str, Any] = {NODE_NAME: name, NODE_TYPE: node_type}
        if seq is not None:
            attributes[NODE_SEQ] = seq
        if node_type == "agent":
            attributes[OPERATION_NAME] = OP_AGENT
            attributes[AGENT_NAME] = name
        span = self._tracer.start_span(f"{node_type} {name}", attributes=attributes)
        labels = {NODE_NAME: name, NODE_TYPE: node_type}
        workflow = otel_context.get_value(_WORKFLOW_KEY)
        if workflow is not None:
            labels[WORKFLOW_NAME] = workflow
        return _Handle(
            span, trace.set_span_in_context(span), time.perf_counter(), labels
        )

    def end_node(
        self,
        span: _Handle,
        *,
        error: Optional[BaseException] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.span.set_attribute(f"{BRANCH_PREFIX}{key}", value)
        if error is not None:
            span.span.record_exception(error)
            span.span.set_attribute(ERROR_TYPE, _error_type(error))
            span.span.set_status(Status(StatusCode.ERROR, str(error)))
        span.span.end()
        self._node_duration.record(time.perf_counter() - span.started, span.labels)

    async def within(
        self, span: _Handle, events: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Relay ``events`` with ``span`` current during each step only.

        The context is attached and detached around one ``__anext__`` — in
        one resumption of one task — so the consumer never runs with an
        engine span current and a generator finalised from another task has
        nothing left to detach. Tasks the step creates copy the context and
        so inherit the span.
        """
        iterator = events.__aiter__()
        try:
            while True:
                token = otel_context.attach(span.context)
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                finally:
                    otel_context.detach(token)
                yield event
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    def tool_call(self, tool_uri: str) -> AbstractContextManager[Any]:
        return self._tool_call(tool_uri)

    @contextmanager
    def _tool_call(self, tool_uri: str) -> Iterator[Span]:
        name = _tool_name(tool_uri)
        labels: dict[str, Any] = {TOOL_NAME: name}
        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f"{OP_TOOL} {name}",
            attributes={OPERATION_NAME: OP_TOOL, TOOL_NAME: name, TOOL_URI: tool_uri},
        ) as span:
            try:
                yield span
            except BaseException as error:
                labels[ERROR_TYPE] = _error_type(error)
                span.set_attribute(ERROR_TYPE, labels[ERROR_TYPE])
                raise
            finally:
                self._tool_duration.record(time.perf_counter() - started, labels)

    def model_call(self, stats: ModelCallStat) -> None:
        operation = OP_CHAT if stats.call_type == "llm" else OP_EMBEDDINGS
        provider, model = _split_model(stats.model)
        labels: dict[str, Any] = {OPERATION_NAME: operation, REQUEST_MODEL: model}
        if provider is not None:
            labels[PROVIDER_NAME] = provider

        input_tokens = stats.prompt_tokens
        if input_tokens is None and stats.call_type == "embedding":
            input_tokens = stats.total_tokens
        attributes: dict[str, Any] = {
            **labels,
            USAGE_INPUT_TOKENS: input_tokens,
            USAGE_OUTPUT_TOKENS: stats.completion_tokens,
            USAGE_CACHE_READ_TOKENS: stats.cached_prompt_tokens,
            USAGE_REASONING_TOKENS: stats.reasoning_tokens,
            BATCH_SIZE: stats.batch_size,
            HTTP_STATUS_CODE: stats.response_code,
        }
        if self.capture_content:
            attributes[MODEL_CALL_REQUEST] = truncate_text_payload(
                stats.request_data, self.max_content_bytes
            )
            attributes[MODEL_CALL_RESPONSE] = truncate_text_payload(
                stats.response_data, self.max_content_bytes
            )

        duration = stats.duration_seconds or 0.0
        end_time = time.time_ns()
        span = self._tracer.start_span(
            f"{operation} {model}",
            kind=SpanKind.CLIENT,
            attributes={k: v for k, v in attributes.items() if v is not None},
            start_time=end_time - int(duration * 1e9),
        )
        # A failed attempt has a provider status other than 200, or none at
        # all (a truncated answer, a timeout).
        if stats.response_code != 200:
            labels[ERROR_TYPE] = (
                str(stats.response_code) if stats.response_code else ERROR_OTHER
            )
            span.set_attribute(ERROR_TYPE, labels[ERROR_TYPE])
            span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=end_time)

        self._operation_duration.record(duration, labels)
        usage_labels = {k: v for k, v in labels.items() if k != ERROR_TYPE}
        if input_tokens is not None:
            self._token_usage.record(
                input_tokens, {**usage_labels, TOKEN_TYPE: "input"}
            )
        if stats.completion_tokens is not None:
            self._token_usage.record(
                stats.completion_tokens, {**usage_labels, TOKEN_TYPE: "output"}
            )

    def receive_model_stats(self, stats: ModelCallStat) -> None:
        self.model_call(stats)


def configure_sdk(
    *, service_name: str, traces: bool = True, metrics: bool = True
) -> None:
    """Install process-global providers that export over OTLP/HTTP.

    For an entry point that owns the process, such as
    ``python -m kavalai.server``. A host that has configured OpenTelemetry
    for the rest of its service does not call this; it passes
    ``telemetry=OtelTelemetry()`` and the spans go where its own do.

    The exporters read the standard variables themselves —
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` and its per-signal forms,
    ``OTEL_EXPORTER_OTLP_HEADERS`` — and ``OTEL_SERVICE_NAME`` takes
    precedence over ``service_name``. Both providers flush when the
    interpreter exits.

    Args:
        service_name: The ``service.name`` resource attribute, unless the
            environment sets one.
        traces: Whether to install a tracer provider.
        metrics: Whether to install a meter provider.
    """
    from opentelemetry import metrics as metrics_api
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create()
    if str(resource.attributes.get(SERVICE_NAME, "")).startswith("unknown_service"):
        resource = resource.merge(Resource({SERVICE_NAME: service_name}))

    if traces:
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(_span_exporter()))
        trace.set_tracer_provider(tracer_provider)
    if metrics:
        reader = PeriodicExportingMetricReader(_metric_exporter())
        metrics_api.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=[reader])
        )


def _span_exporter() -> Any:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def _metric_exporter() -> Any:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )

    return OTLPMetricExporter()
