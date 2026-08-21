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
from typing import Any, Optional
from uuid import uuid4

import aiosqlite

from kavalai.utils import to_plain
from kavalai.workflow.tasklog.base import TaskLogger
from kavalai.llm_clients.base_client import ModelCallStat

# Mirrors the Postgres ``tasks`` and ``model_call_stats`` tables (db.py) using
# TEXT UUIDs and JSON-encoded payload columns.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    session_id TEXT,
    run_id TEXT,
    name TEXT,
    node_type TEXT,
    inputs TEXT,
    output TEXT,
    prompt TEXT,
    errors TEXT,
    duration_seconds REAL
);
CREATE TABLE IF NOT EXISTS model_call_stats (
    id TEXT PRIMARY KEY,
    call_type TEXT NOT NULL,
    model TEXT,
    agent_id TEXT,
    request_data TEXT,
    response_data TEXT,
    response_code INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_prompt_tokens INTEGER,
    reasoning_tokens INTEGER,
    batch_size INTEGER,
    duration_seconds REAL
);
"""


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(to_plain(value))


class SqliteTaskLogger(TaskLogger):
    """Task logger storing node executions and model stats in SQLite.

    Defaults to a private ``:memory:`` database. Pass a file ``path`` to keep
    the debugging data across runs.
    """

    def __init__(self, path: str = ":memory:"):
        super().__init__()
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _connect(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
        return self._conn

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
    ) -> None:
        conn = await self._connect()
        await conn.execute(
            "INSERT INTO tasks (id, agent_id, session_id, run_id, name, "
            "node_type, inputs, output, prompt, errors, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                agent_id,
                session_id,
                run_id,
                node_name,
                node_type,
                _dumps(inputs),
                _dumps(output),
                prompt,
                _dumps(errors),
                duration,
            ),
        )
        await conn.commit()

    async def _log_model_call_impl(
        self, stats: ModelCallStat, agent_id: Optional[str]
    ) -> None:
        conn = await self._connect()
        await conn.execute(
            "INSERT INTO model_call_stats (id, call_type, model, agent_id, "
            "request_data, response_data, response_code, prompt_tokens, "
            "completion_tokens, total_tokens, cached_prompt_tokens, "
            "reasoning_tokens, batch_size, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                stats.call_type,
                stats.model,
                agent_id,
                stats.request_data,
                stats.response_data,
                stats.response_code,
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.total_tokens,
                stats.cached_prompt_tokens,
                stats.reasoning_tokens,
                stats.batch_size,
                stats.duration_seconds,
            ),
        )
        await conn.commit()

    async def get_tasks(self, run_id: Optional[str] = None) -> list[dict]:
        """Return logged node executions, newest table order, as plain dicts.

        ``inputs``, ``output`` and ``errors`` come back decoded rather than as
        JSON strings. Reading these used to mean reaching for ``_connect()``
        and writing SQL by hand, which the observability tutorial had to teach.

        Args:
            run_id: Restrict to one run. ``None`` returns every logged task.
        """
        return await self._select("tasks", run_id, ("inputs", "output", "errors"))

    async def get_model_calls(self, run_id: Optional[str] = None) -> list[dict]:
        """Return logged model calls as plain dicts.

        Note that ``model_call_stats`` rows carry no ``run_id`` — the stats
        arrive from the LLM client, which knows the agent but not the run — so
        ``run_id`` filters by agent only when the tasks table can resolve it.
        """
        return await self._select("model_call_stats", None, ())

    async def _select(
        self, table: str, run_id: Optional[str], json_columns: tuple[str, ...]
    ) -> list[dict]:
        # Writes are fire-and-forget background tasks, so a read that does not
        # flush first races them and intermittently returns a short list.
        await self.flush()
        conn = await self._connect()
        query = f"SELECT * FROM {table}"  # nosec B608 - table name is internal
        parameters: tuple = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters = (run_id,)
        async with conn.execute(query, parameters) as cursor:
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            record = dict(row)
            for column in json_columns:
                if record.get(column) is not None:
                    record[column] = json.loads(record[column])
            results.append(record)
        return results

    async def close(self) -> None:
        await self.flush()
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
