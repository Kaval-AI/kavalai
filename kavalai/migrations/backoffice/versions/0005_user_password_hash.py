"""users.password_hash

Sign-in with an address and a password, beside Google OAuth, which becomes
optional. The column holds a salted scrypt hash (see
``kavalai.backoffice.users``), never the password; ``NULL`` is a user that
signs in through OAuth only, which is every existing user.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-24 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _target_schema() -> Union[str, None]:
    """The schema batch mode must name explicitly (reflection ignores the
    ``schema_translate_map`` the runner applies as ``{None: <schema>}``)."""
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def upgrade() -> None:
    with op.batch_alter_table("users", schema=_target_schema()) as batch_op:
        batch_op.add_column(sa.Column("password_hash", sa.TEXT(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users", schema=_target_schema()) as batch_op:
        batch_op.drop_column("password_hash")
