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

The seam through which a run is reported to a tracing system.

:class:`Telemetry` is the interface the workflow engine, the function kernel
and the token accumulator call. Every method of the base class does nothing,
so the engine holds one unconditionally and a deployment that passes none
behaves exactly as before. :class:`kavalai.otel.OtelTelemetry` is the
OpenTelemetry implementation; it lives in a module of its own because this one
has to import under Pyodide, where there is no collector to send spans to.

The module imports nothing outside the base install.
"""

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional

from kavalai.llm_clients.base_client import ModelCallStat


class Telemetry:
    """Receives the life cycle of runs, nodes, tool calls and model calls.

    The engine treats a span as an opaque handle: it receives one from
    :meth:`start_run` or :meth:`start_node` and hands it back to the matching
    ``end_*`` method and to :meth:`within`. An implementation holds no per-run
    state of its own, so one instance serves every concurrent run of an
    engine.

    A subclass overrides the methods it needs. None of them may raise for an
    ordinary run: telemetry is an observer, and a run does not fail because
    it could not be observed.
    """

    def start_run(
        self,
        *,
        workflow: str,
        agent_id: Optional[str],
        session_id: Optional[str],
        run_id: Optional[str],
        external_id: Optional[str],
        invocation_id: str,
    ) -> Any:
        """Open the span of one run and return its handle.

        Called once per run, after the run row exists, so the ids are known;
        they are ``None`` for an engine without an ``AgentService``.
        """
        return None

    def end_run(
        self,
        span: Any,
        *,
        status: str,
        error: Optional[str],
        token_usage: Optional[dict],
        duration_seconds: Optional[float],
    ) -> None:
        """Close a run's span. Called exactly once per :meth:`start_run`.

        ``status`` is the run's final ``WorkflowState.status`` and ``error``
        its error text — also for a run that timed out or whose consumer went
        away.
        """

    def start_node(self, *, name: str, node_type: str, seq: Optional[int]) -> Any:
        """Open the span of one node visit and return its handle."""
        return None

    def end_node(
        self,
        span: Any,
        *,
        error: Optional[BaseException] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Close a node's span.

        ``attributes`` carries what a node reports about itself; today that
        is a branch node's ``taken`` and ``matched``.
        """

    def within(self, span: Any, events: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Relay ``events`` with ``span`` current while each one is produced.

        The base class returns ``events`` unchanged. An implementation must
        make the span current only for the duration of one step of the
        iterator and never across a ``yield``: between two events the
        consumer's own code runs, and it must not see the engine's span.
        """
        return events

    def tool_call(self, tool_uri: str) -> AbstractContextManager[Any]:
        """A context manager entered around one tool call."""
        return nullcontext()

    def model_call(self, stats: ModelCallStat) -> None:
        """Report one finished model call, successful or failed."""
