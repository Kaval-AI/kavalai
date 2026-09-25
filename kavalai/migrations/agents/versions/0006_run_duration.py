"""record how long each run took

``runs`` gains ``duration_seconds``: the wall-clock time of the invocation,
measured by the engine with a monotonic clock from the start of the run to the
moment it records the result, on success and on failure alike. The value was
until now approximated as ``updated_at - created_at``, which is a difference of
two database timestamps: it moves whenever the row is touched again, and it is
missing for a run that died before its final update. Every node visit already
records its own ``tasks.duration_seconds``; the run-level figure is not their
sum, because parallel branches overlap and the walk between nodes takes time
too.

The column is nullable and not backfilled: a run written before this revision
keeps ``NULL``, and readers fall back to the timestamp difference for it.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _target_schema() -> Union[str, None]:
    """The schema these ops must name explicitly.

    ``batch_alter_table`` reflects the existing table and reflection does not
    consult ``schema_translate_map``, so an ALTER against a translated schema
    has to be told where the table is. See revision 0004.
    """
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def upgrade() -> None:
    with op.batch_alter_table("runs", schema=_target_schema()) as batch_op:
        batch_op.add_column(sa.Column("duration_seconds", sa.Numeric(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("runs", schema=_target_schema()) as batch_op:
        batch_op.drop_column("duration_seconds")
