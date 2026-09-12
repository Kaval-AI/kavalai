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

``DEFAULT_MAX_PAYLOAD_BYTES`` is an operational limit, not a privacy control: a
single ``crawl_url`` result or a long prompt can be megabytes, and writing that
into a row hits statement-size limits, grows the write-behind logger's memory
under concurrency and makes the backoffice task list unusable. Payloads above
the cap are replaced by a marker carrying their real size and a preview;
everything below it is stored in full. The privacy controls are
``record_nodes`` and ``record_payloads`` on :class:`TaskLogger`.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from kavalai.llm_clients.base_client import ModelCallStat, ModelStatsReceiver
from kavalai.utils import to_plain

DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024


def truncate_payload(value: Any, max_bytes: int) -> Any:
    """Return ``value`` unchanged, or a marker dict when it is too large.

    Args:
        value: The payload to measure, already reduced to plain types.
        max_bytes: Size limit; ``0`` or less disables truncation.
    """
    if value is None or max_bytes <= 0:
        return value
    try:
        encoded = json.dumps(value, default=str).encode("utf-8")
    except (TypeError, ValueError):  # pragma: no cover - to_plain covers this
        return value
    if len(encoded) <= max_bytes:
        return value
    # Not ``_truncated``: ``to_plain`` drops keys starting with an underscore
    # on the way into a row, which would make the marker indistinguishable
    # from a genuine payload that happened to have a ``bytes`` field.
    return {
        "truncated": True,
        "bytes": len(encoded),
        "preview": encoded[:2048].decode("utf-8", errors="replace"),
    }


def truncate_text_payload(text: Optional[str], max_bytes: int) -> Optional[str]:
    """:func:`truncate_payload` for a payload already encoded as JSON text.

    Model-call payloads travel as strings, so the marker is returned as a
    JSON string too, and the column keeps a single type.
    """
    if text is None or max_bytes <= 0:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return json.dumps(
        {
            "truncated": True,
            "bytes": len(encoded),
            "preview": encoded[:2048].decode("utf-8", errors="replace"),
        }
    )


def as_model_call_stat(stats: Any) -> ModelCallStat:
    """The Pydantic :class:`ModelCallStat` for a stats record of either kind.

    LLM clients report the Pydantic model; older embedding code and callers
    that build rows by hand pass the ORM row, whose payloads may be dicts.
    Loggers receive one type, with payloads as JSON text.
    """
    if isinstance(stats, ModelCallStat):
        return stats

    def as_text(value: Any) -> Optional[str]:
        if value is None or isinstance(value, str):
            return value
        return json.dumps(to_plain(value))

    duration = getattr(stats, "duration_seconds", None)
    return ModelCallStat(
        call_type=stats.call_type,
        model=stats.model,
        request_data=as_text(getattr(stats, "request_data", None)),
        response_data=as_text(getattr(stats, "response_data", None)),
        response_code=getattr(stats, "response_code", None),
        prompt_tokens=getattr(stats, "prompt_tokens", None),
        completion_tokens=getattr(stats, "completion_tokens", None),
        total_tokens=getattr(stats, "total_tokens", None),
        cached_prompt_tokens=getattr(stats, "cached_prompt_tokens", None),
        reasoning_tokens=getattr(stats, "reasoning_tokens", None),
        batch_size=getattr(stats, "batch_size", None),
        duration_seconds=float(duration) if duration is not None else None,
    )


class TaskLogger(ABC):
    """Common interface for storing per-node debugging data and model stats.

    Logging is fire-and-forget: the public ``log_*`` methods schedule a
    background task and return immediately so they never block workflow
    execution. Call :meth:`flush` (e.g. at the end of a run or in tests) to
    await all pending writes.

    A backend implements two hooks, :meth:`write_node` and
    :meth:`write_model_call`. Several loggers compose with
    :class:`~kavalai.workflow.tasklog.memory.TeeTaskLogger` — a metering
    logger beside the database one, for example — so a subclass never needs
    to override the ``log_*`` methods themselves.

    Two options record less, for deployers who must not keep a second copy of
    what a run saw:

    ``record_nodes``
        When false, node executions are not recorded at all. Model calls
        still are.
    ``record_payloads``
        When false, a node's ``inputs``, ``output`` and ``prompt`` and a model
        call's ``request_data`` and ``response_data`` are not stored. Names,
        timings, errors and token counts are. A model call's request is the
        whole prompt — chat history and retrieved passages included — so this
        is the option that keeps transcripts out of the telemetry.

    ``max_payload_bytes`` caps each stored payload, node and model call alike.
    """

    def __init__(
        self,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        *,
        record_nodes: bool = True,
        record_payloads: bool = True,
    ) -> None:
        self._background_tasks: set[asyncio.Task] = set()
        self.max_payload_bytes = max_payload_bytes
        self.record_nodes = record_nodes
        self.record_payloads = record_payloads

    def log_node(
        self,
        *,
        run_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        node_name: str,
        node_type: str,
        inputs: Optional[dict],
        output: Any,
        prompt: Optional[str] = None,
        duration: float = 0.0,
        errors: Optional[list[str]] = None,
        seq: Optional[int] = None,
        parent_task_name: Optional[str] = None,
        tool_uri: Optional[str] = None,
    ) -> None:
        """Record the execution of a single node (fire-and-forget).

        Args:
            seq: Position of this record in the run's execution order.
            parent_task_name: Node that produced this record, set on the
                tool-call rows an agent node emits.
            tool_uri: Tool this record executed, for function nodes and agent
                tool calls alike.
        """
        if not self.record_nodes:
            return
        if not self.record_payloads:
            inputs, output, prompt = None, None, None
        self._spawn(
            self.write_node(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                node_name=node_name,
                node_type=node_type,
                inputs=truncate_payload(inputs, self.max_payload_bytes),
                output=truncate_payload(output, self.max_payload_bytes),
                prompt=prompt,
                duration=duration,
                errors=errors,
                seq=seq,
                parent_task_name=parent_task_name,
                tool_uri=tool_uri,
            )
        )

    def log_model_call(
        self,
        stats: ModelCallStat,
        agent_id: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Record an LLM / embedding model call (fire-and-forget).

        ``session_id`` and ``run_id`` attribute the call to the run that made
        it; the engine supplies both, so cost per run and per session is a
        query over the recorded rows.
        """
        self._spawn(
            self.write_model_call(
                self._prepare_stats(stats),
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
            )
        )

    def _prepare_stats(self, stats: Any) -> ModelCallStat:
        """Apply the payload options to a model call before it is written."""
        stats = as_model_call_stat(stats)
        if not self.record_payloads:
            return stats.model_copy(
                update={"request_data": None, "response_data": None}
            )
        return stats.model_copy(
            update={
                "request_data": truncate_text_payload(
                    stats.request_data, self.max_payload_bytes
                ),
                "response_data": truncate_text_payload(
                    stats.response_data, self.max_payload_bytes
                ),
            }
        )

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():  # pragma: no cover - tasks are not cancelled in practice
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Task logger background write failed: {exc}")

    async def flush(self) -> None:
        """Await all pending background writes."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def close(self) -> None:
        """Flush and release backend resources. Override to add cleanup."""
        await self.flush()

    @abstractmethod
    async def write_node(
        self,
        *,
        run_id: Optional[str],
        session_id: Optional[str],
        agent_id: Optional[str],
        node_name: str,
        node_type: str,
        inputs: Optional[dict],
        output: Any,
        prompt: Optional[str],
        duration: float,
        errors: Optional[list[str]],
        seq: Optional[int] = None,
        parent_task_name: Optional[str] = None,
        tool_uri: Optional[str] = None,
    ) -> None:
        """Persist a node execution record.

        Called in a background task, after the recording options and the
        payload cap have been applied.
        """

    @abstractmethod
    async def write_model_call(
        self,
        stats: ModelCallStat,
        *,
        agent_id: Optional[str],
        session_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        """Persist a model call statistics record.

        Called in a background task, after the recording options and the
        payload cap have been applied. ``stats`` is always the Pydantic
        :class:`~kavalai.llm_clients.base_client.ModelCallStat`.
        """


class StatsBridge(ModelStatsReceiver):
    """Adapter forwarding LLM ``ModelCallStat`` events to a :class:`TaskLogger`.

    Pass it as an LLM or embedding client's stats receiver — or as a RAG
    service's ``stats_receiver`` — to log every call it makes against
    ``agent_id`` (and, when known, a session and a run). The engine itself
    uses :class:`TokenAccumulator`, which also forwards to a logger.
    """

    def __init__(
        self,
        task_logger: TaskLogger,
        agent_id: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.task_logger = task_logger
        self.agent_id = agent_id
        self.session_id = session_id
        self.run_id = run_id

    def receive_model_stats(self, stats: ModelCallStat) -> None:
        self.task_logger.log_model_call(
            stats, self.agent_id, session_id=self.session_id, run_id=self.run_id
        )


class TokenAccumulator(ModelStatsReceiver):
    """Aggregates token usage across a workflow run and optionally forwards each
    ``ModelCallStat`` to a :class:`TaskLogger`.

    The engine wires one accumulator into every LLM client built during a run,
    and hands it to the RAG services the run queries, so that when the run
    ends it can report the total token spend. When a ``task_logger`` is
    supplied each individual call is still logged through it, attributed to
    ``agent_id``, ``session_id`` and ``run_id``, so this fully subsumes
    :class:`StatsBridge`.
    """

    def __init__(
        self,
        task_logger: Optional[TaskLogger] = None,
        agent_id: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.task_logger = task_logger
        self.agent_id = agent_id
        self.session_id = session_id
        self.run_id = run_id
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def receive_model_stats(self, stats: ModelCallStat) -> None:
        self.model_calls += 1
        self.prompt_tokens += stats.prompt_tokens or 0
        self.completion_tokens += stats.completion_tokens or 0
        self.total_tokens += stats.total_tokens or 0
        if self.task_logger is not None:
            self.task_logger.log_model_call(
                stats, self.agent_id, session_id=self.session_id, run_id=self.run_id
            )

    def summary(self) -> dict:
        """Return the aggregated token counts as a plain dict."""
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
