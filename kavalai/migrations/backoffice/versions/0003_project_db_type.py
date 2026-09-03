"""projects.db_type selects the agent database backend

A project used to describe its agent database as host, port, user, password,
database and schema, which only fits PostgreSQL. ``db_type`` names the
backend: ``postgresql`` (the default, so existing rows keep working) reads
those fields as before; ``sqlite`` reads ``db_name`` as the file path.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-03 16:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _target_schema() -> Union[str, None]:
    """The schema batch mode must name explicitly (reflection ignores the
    ``schema_translate_map`` the runner applies as ``{None: <schema>}``)."""
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def upgrade() -> None:
    with op.batch_alter_table("projects", schema=_target_schema()) as batch_op:
        batch_op.add_column(
            sa.Column(
                "db_type",
                sa.TEXT(),
                nullable=False,
                server_default="postgresql",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("projects", schema=_target_schema()) as batch_op:
        batch_op.drop_column("db_type")
