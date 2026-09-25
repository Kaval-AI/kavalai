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

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*(templates|context|history)\.(.+?)\s*\}\}")
"""One ``{{ templates.NAME }}``, ``{{ context.PATH }}`` or ``{{ history.PATH }}``
placeholder; the only three prefixes a template may reference."""


class RunContext(BaseModel):
    """Runtime data for a single interaction.

    ``token_stats`` is the run's ``TokenAccumulator``. Every LLM client built
    during the run reports into it, and parallel branches share the parent's,
    so the totals cover the whole run.

    ``seq_counter`` is the run's task-sequence counter, shared with parallel
    branches in the same way so that every task row of the run draws from one
    sequence and the interleaving of concurrent branches is recorded rather than
    lost. ``current_seq`` is the number handed to the node this context is
    currently executing, and is per-branch.

    ``task_logger`` overrides the engine's ``TaskLogger`` for this run only.
    One engine serves many concurrent runs, so a caller that wants *this* run's
    trajectory on its own — the evaluation runner, a notebook debugging one
    call — cannot swap the engine's logger.

    ``template_overrides`` holds the template values passed to this run only
    (``run_stream(templates=...)``); they are already merged into
    ``templates`` and are kept separately so the run record can say which
    values came from the caller rather than from the document.

    ``started_at`` is the monotonic-clock reading taken when the run started,
    shared with parallel branches, from which the run's ``duration_seconds`` is
    measured when it completes or fails.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    data: dict = {}
    templates: Dict[str, str] = {}
    template_overrides: Dict[str, str] = {}
    agent_service: Optional[Any] = None
    token_stats: Optional[Any] = None
    seq_counter: Optional[Any] = None
    started_at: Optional[float] = None
    current_seq: Optional[int] = None
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

    async def resolve_placeholder(self, prefix: str, path: str):
        """The value behind one ``{{ prefix.path }}`` placeholder.

        ``prefix`` is ``templates``, ``context`` or ``history``. A reference
        that resolves to nothing raises ``ValueError``: a prompt or filter
        silently missing a value is worse than a failed run.
        """
        if prefix == "templates":
            val = await self.resolve_template_value(path)
        elif prefix == "context":
            val = self.resolve_context_value(path)
        else:
            val = await self.resolve_history_value(path)
        if val is None:
            raise ValueError(f"Could not resolve {prefix}.{path}")
        return val

    async def render_value(self, template: str):
        """Render a template that may stand for a single typed value.

        A template that is exactly one placeholder resolves to the referenced
        value itself — a number stays a number, a boolean a boolean — so a
        value an earlier node extracted can be compared without passing
        through text. Anything else is rendered with :meth:`render_prompt`.
        """
        whole = PLACEHOLDER_PATTERN.fullmatch(template.strip())
        if whole is None:
            return await self.render_prompt(template)
        return await self.resolve_placeholder(whole.group(1), whole.group(2).strip())

    async def render_prompt(self, prompt: str) -> str:
        """
        Render a prompt string by replacing {{ templates.NAME }}, {{ context.PATH }},
        and {{ history.PATH }} with their resolved values.
        """

        async def replace_match(match):
            path = match.group(2).strip()
            val = await self.resolve_placeholder(match.group(1), path)

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
        for match in PLACEHOLDER_PATTERN.finditer(prompt):
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
