"""attribute model calls to runs; sessions.updated_at as last activity; indexes

``model_call_stats`` gains ``session_id`` and ``run_id``: nullable, indexed and
without foreign keys, like ``agent_id``. A cost record outlives the
conversation it belongs to, and with the run on the row, cost per run and per
conversation is a query rather than a join through timestamps.

``sessions.updated_at`` is now moved by every run, so it means "last activity".
Rows written before this revision still carry their creation time; the
backfill sets them to the time of the session's last run, so a retention sweep
run right after the upgrade does not see long-running conversations as idle.
It rewrites ``sessions`` once — far fewer rows than ``chat_messages``.

Indexes serve the queries every turn and every retention sweep make:

``sessions(agent_id, external_id)``
    finding a conversation by the caller's key, on every turn.
``sessions(agent_id, updated_at)``, ``sessions(updated_at)``
    recent-first lists and purges, per agent and across agents.
``chat_messages(session_id, created_at)``
    the history window and transcripts; replaces ``ix_chat_messages_session_id``.
``chat_messages(agent_id, created_at)``, ``model_call_stats(agent_id, created_at)``
    figures per agent and date; replace the single-column ``agent_id`` indexes.

Every index is created with ``IF NOT EXISTS``. On a large PostgreSQL database
an operator can build them beforehand with ``CREATE INDEX CONCURRENTLY`` under
the same names, and this revision then leaves them as they are instead of
blocking writes while it builds them.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from kavalai.db import uuid_column

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_INDEXES = [
    (
        "ix_model_call_stats_agent_id_created_at",
        "model_call_stats",
        ["agent_id", "created_at"],
    ),
    ("ix_model_call_stats_session_id", "model_call_stats", ["session_id"]),
    ("ix_model_call_stats_run_id", "model_call_stats", ["run_id"]),
    ("ix_sessions_agent_id_external_id", "sessions", ["agent_id", "external_id"]),
    ("ix_sessions_agent_id_updated_at", "sessions", ["agent_id", "updated_at"]),
    ("ix_sessions_updated_at", "sessions", ["updated_at"]),
    (
        "ix_chat_messages_session_id_created_at",
        "chat_messages",
        ["session_id", "created_at"],
    ),
    (
        "ix_chat_messages_agent_id_created_at",
        "chat_messages",
        ["agent_id", "created_at"],
    ),
]

_REPLACED_INDEXES = [
    ("ix_model_call_stats_agent_id", "model_call_stats", ["agent_id"]),
    ("ix_sessions_agent_id", "sessions", ["agent_id"]),
    ("ix_chat_messages_session_id", "chat_messages", ["session_id"]),
    ("ix_chat_messages_agent_id", "chat_messages", ["agent_id"]),
]


def _target_schema() -> Union[str, None]:
    """The schema these ops must name explicitly.

    ``batch_alter_table`` reflects the existing table and reflection does not
    consult ``schema_translate_map``, so an ALTER against a translated schema
    has to be told where the table is. See revision 0004.
    """
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def _backfill_session_activity(schema: Union[str, None]) -> None:
    """Set each session's ``updated_at`` to the time of its last run."""
    sessions = sa.table(
        "sessions", sa.column("id"), sa.column("updated_at"), schema=schema
    )
    runs = sa.table(
        "runs", sa.column("session_id"), sa.column("created_at"), schema=schema
    )
    last_run = (
        sa.select(sa.func.max(runs.c.created_at))
        .where(runs.c.session_id == sessions.c.id)
        .scalar_subquery()
    )
    has_runs = sa.exists().where(runs.c.session_id == sessions.c.id)
    op.execute(sessions.update().where(has_runs).values(updated_at=last_run))


def upgrade() -> None:
    schema = _target_schema()
    with op.batch_alter_table("model_call_stats", schema=schema) as batch_op:
        batch_op.add_column(sa.Column("session_id", uuid_column(), nullable=True))
        batch_op.add_column(sa.Column("run_id", uuid_column(), nullable=True))
    for name, table, columns in _NEW_INDEXES:
        op.create_index(
            name, table, columns, unique=False, schema=schema, if_not_exists=True
        )
    for name, table, _ in _REPLACED_INDEXES:
        op.drop_index(name, table_name=table, schema=schema, if_exists=True)
    _backfill_session_activity(schema)


def downgrade() -> None:
    schema = _target_schema()
    for name, table, columns in _REPLACED_INDEXES:
        op.create_index(
            name, table, columns, unique=False, schema=schema, if_not_exists=True
        )
    for name, table, _ in _NEW_INDEXES:
        op.drop_index(name, table_name=table, schema=schema, if_exists=True)
    with op.batch_alter_table("model_call_stats", schema=schema) as batch_op:
        batch_op.drop_column("run_id")
        batch_op.drop_column("session_id")
