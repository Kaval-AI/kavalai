"""record the executed trajectory on tasks: seq, parent_task_name, tool_uri

Before this revision a task row was written only by the side-effecting nodes,
and an agent node produced exactly one row holding its final answer. Everything
the agent actually did — which tools it chose, with what arguments, what came
back — was built during the run and then discarded, which made an agent failure
un-debuggable after the fact.

Three additive, nullable columns close that:

``seq``
    Per-run execution order. This is what carries the structure: ordering by it
    reconstructs the executed path exactly, including the interleaving of
    concurrent ``parallel`` branches, which ``created_at`` cannot (it is
    approximate, and ties are unordered).

``parent_task_name``
    The node that produced the row, set on the tool-call rows an agent node
    emits. A name rather than an id: no parent id has to be allocated up front,
    and the readable column is the one that survives an export to a warehouse.

``tool_uri``
    The tool the row executed, set by both function nodes and agent tool calls,
    so one predicate finds every call to a tool regardless of whether a human
    wired it into the YAML or an agent chose it mid-run.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _target_schema() -> Union[str, None]:
    """The schema these ops must name explicitly.

    ``schema_translate_map`` rewrites compiled DDL, but ``batch_alter_table``
    reflects the existing table first and reflection does not consult the map —
    so an ALTER against a translated schema has to be told where the table is.
    The runner applies the map as ``{None: <schema>}``; read it back from the
    connection.
    """
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def upgrade() -> None:
    schema = _target_schema()
    with op.batch_alter_table("tasks", schema=schema) as batch_op:
        batch_op.add_column(sa.Column("seq", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("parent_task_name", sa.TEXT(), nullable=True))
        batch_op.add_column(sa.Column("tool_uri", sa.TEXT(), nullable=True))
    op.create_index(
        op.f("ix_tasks_parent_task_name"),
        "tasks",
        ["parent_task_name"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        op.f("ix_tasks_tool_uri"), "tasks", ["tool_uri"], unique=False, schema=schema
    )


def downgrade() -> None:
    schema = _target_schema()
    op.drop_index(op.f("ix_tasks_tool_uri"), table_name="tasks", schema=schema)
    op.drop_index(op.f("ix_tasks_parent_task_name"), table_name="tasks", schema=schema)
    with op.batch_alter_table("tasks", schema=schema) as batch_op:
        batch_op.drop_column("tool_uri")
        batch_op.drop_column("parent_task_name")
        batch_op.drop_column("seq")
