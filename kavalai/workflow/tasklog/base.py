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
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from kavalai.llm_clients.base_client import ModelCallStat, ModelStatsReceiver

# Operational, not a privacy control: a single ``crawl_url`` result can be
# megabytes, and writing that into a row hits statement-size limits, grows the
# write-behind logger's memory under concurrency and makes the backoffice task
# list unusable. Payloads above the cap are replaced by a marker carrying their
# real size and a preview. Everything below it is stored in full.
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


class TaskLogger(ABC):
    """Common interface for storing per-node debugging data and model stats.

    Logging is fire-and-forget: the public ``log_*`` methods schedule a
    background task and return immediately so they never block workflow
    execution. Call :meth:`flush` (e.g. at the end of a run or in tests) to
    await all pending writes.
    """

    def __init__(self, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        self._background_tasks: set[asyncio.Task] = set()
        self.max_payload_bytes = max_payload_bytes

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
        self._spawn(
            self._log_node_impl(
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
        self, stats: ModelCallStat, agent_id: Optional[str] = None
    ) -> None:
        """Record an LLM / embedding model call (fire-and-forget)."""
        self._spawn(self._log_model_call_impl(stats, agent_id))

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
    async def _log_node_impl(
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
        """Persist a node execution record."""

    @abstractmethod
    async def _log_model_call_impl(
        self, stats: ModelCallStat, agent_id: Optional[str]
    ) -> None:
        """Persist a model call statistics record."""


class StatsBridge(ModelStatsReceiver):
    """Adapter forwarding LLM ``ModelCallStat`` events to a :class:`TaskLogger`.

    Pass it as an LLM client's ``model_stats_receiver`` to log every call the
    client makes against ``agent_id``. The engine itself uses
    :class:`TokenAccumulator`, which also forwards to a logger.
    """

    def __init__(self, task_logger: TaskLogger, agent_id: Optional[str] = None):
        self.task_logger = task_logger
        self.agent_id = agent_id

    def receive_model_stats(self, stats: ModelCallStat) -> None:
        self.task_logger.log_model_call(stats, self.agent_id)


class TokenAccumulator(ModelStatsReceiver):
    """Aggregates token usage across a workflow run and optionally forwards each
    ``ModelCallStat`` to a :class:`TaskLogger`.

    The engine wires one accumulator into every LLM client built during a run so
    that, when the run ends, it can report the total token spend. When a
    ``task_logger`` is supplied each individual call is still logged through it,
    so this fully subsumes :class:`StatsBridge`.
    """

    def __init__(
        self,
        task_logger: Optional[TaskLogger] = None,
        agent_id: Optional[str] = None,
    ):
        self.task_logger = task_logger
        self.agent_id = agent_id
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
            self.task_logger.log_model_call(stats, self.agent_id)

    def summary(self) -> dict:
        """Return the aggregated token counts as a plain dict."""
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
