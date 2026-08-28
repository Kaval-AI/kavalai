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

import json
import re
from typing import Optional, Any, Dict
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict

from kavalai.resolvers import resolve_path
from kavalai.utils import to_plain
from kavalai.workflow.models import ArgumentInfo


class RunContext(BaseModel):
    """Runtime data for a single interaction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    data: dict = {}
    templates: Dict[str, str] = {}
    agent_service: Optional[Any] = None
    # The run's TokenAccumulator. Typed loosely for the same reason as
    # ``agent_service``: importing it here would close an import cycle. Every
    # LLM client built during the run reports into this one object, and parallel
    # branches share the parent's, so the totals cover the whole run.
    token_stats: Optional[Any] = None
    # The run's task-sequence counter (an ``itertools.count``). Shared with
    # parallel branches exactly as ``token_stats`` is, so every task row of the
    # run draws from one sequence and the interleaving of concurrent branches is
    # recorded rather than lost. ``current_seq`` is the number handed to the
    # node this context is currently executing, and is per-branch.
    seq_counter: Optional[Any] = None
    current_seq: Optional[int] = None
    # Optional per-run TaskLogger, overriding the engine's for this run only.
    # One engine serves many concurrent runs, so a caller that wants *this*
    # run's trajectory on its own — the evaluation runner, a notebook debugging
    # one call — cannot swap the engine's logger. Typed loosely for the same
    # reason as ``token_stats``.
    task_logger: Optional[Any] = None

    def next_seq(self) -> Optional[int]:
        """Take the next number from the run's task sequence.

        Returns ``None`` when the run has no counter, which is the case for a
        bare ``RunContext()`` built outside the engine.
        """
        if self.seq_counter is None:
            return None
        return next(self.seq_counter)

    def resolve_context_value(self, path: str):
        """Resolve a dotted path like 'input.user_message' from context data."""
        return resolve_path(self.data, path)

    async def resolve_history_value(self, path: str):
        """Resolve a value from session history."""
        if not self.agent_service or not self.session_id:
            logger.error(
                f"Cannot load from history for {path}: agent_service or session_id not set"
            )
            return None
        return await self.agent_service.get_history_value(self.session_id, str(path))

    async def resolve_template_value(self, name: str):
        """Resolve a template value by name."""
        return self.templates.get(name)

    async def render_prompt(self, prompt: str) -> str:
        """
        Render a prompt string by replacing {{ templates.NAME }}, {{ context.PATH }},
        and {{ history.PATH }} with their resolved values.
        """
        pattern = re.compile(r"\{\{\s*(templates|context|history)\.(.+?)\s*\}\}")

        async def replace_match(match):
            prefix = match.group(1)
            path = match.group(2).strip()

            # The pattern only matches these three prefixes.
            if prefix == "templates":
                val = await self.resolve_template_value(path)
            elif prefix == "context":
                val = self.resolve_context_value(path)
            else:
                val = await self.resolve_history_value(path)

            if val is None:
                raise ValueError(f"Could not resolve {prefix}.{path}")

            if isinstance(val, (dict, list, BaseModel)):
                try:
                    return json.dumps(to_plain(val), ensure_ascii=False)
                except Exception as e:
                    logger.opt(exception=True).warning(
                        f"Error serializing template value {path}: {e}"
                    )
                    return str(val)

            return str(val)

        # re.sub cannot take an async replacement, so splice by hand.
        last_pos = 0
        pieces = []
        for match in pattern.finditer(prompt):
            pieces.append(prompt[last_pos : match.start()])
            pieces.append(await replace_match(match))
            last_pos = match.end()
        pieces.append(prompt[last_pos:])

        return "".join(pieces)

    async def resolve_input_info(self, info: ArgumentInfo):
        """Resolve an :class:`ArgumentInfo` to its actual value."""
        if info.type == "literal":
            return info.value
        # ``value`` is the path; ``name`` is the fallback.
        path = info.value or info.name
        if info.type == "history":
            return await self.resolve_history_value(str(path))
        if path:
            return self.resolve_context_value(str(path))
        return None

    async def prepare_tool_inputs(self, task: Any) -> dict:
        """Resolve a task/node's ``inputs`` mapping into plain values."""
        inputs = {}
        for name, info in task.inputs.items():
            if info.value is None and info.name is None:
                info = info.model_copy(update={"value": name})
            value = await self.resolve_input_info(info)
            if isinstance(value, BaseModel):
                value = value.model_dump()
            inputs[name] = value

        return inputs
