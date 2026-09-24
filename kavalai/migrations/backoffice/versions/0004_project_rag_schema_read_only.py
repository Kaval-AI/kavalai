"""projects.rag_schema and projects.read_only

A deployment may keep its RAG collections in a schema of their own, apart
from the runtime tables (kavalapp does: ``agents_<mode>`` and
``rag_<mode>``); ``rag_schema`` names it, and ``NULL`` keeps the old reading,
the agent schema. ``read_only`` makes every connection the backoffice opens
to the project's database refuse writes at the database; it is on for every
existing project, since the backoffice only reads.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-24 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _target_schema() -> Union[str, None]:
    """The schema batch mode must name explicitly (reflection ignores the
    ``schema_translate_map`` the runner applies as ``{None: <schema>}``)."""
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def upgrade() -> None:
    with op.batch_alter_table("projects", schema=_target_schema()) as batch_op:
        batch_op.add_column(sa.Column("rag_schema", sa.TEXT(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "read_only",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("projects", schema=_target_schema()) as batch_op:
        batch_op.drop_column("read_only")
        batch_op.drop_column("rag_schema")
