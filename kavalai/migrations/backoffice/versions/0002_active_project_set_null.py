"""users.active_project_id clears itself when the project is deleted

Without ``ON DELETE SET NULL`` deleting a project either fails on this foreign
key or (where the constraint was never created) leaves users pointing at a
project that no longer exists, which makes every project-scoped endpoint answer
403 with no way to recover.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-20 10:14:22.108431

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FK_NAME = "users_active_project_id_fkey"


def upgrade() -> None:
    op.drop_constraint(FK_NAME, "users", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "users",
        "projects",
        ["active_project_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(FK_NAME, "users", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "users",
        "projects",
        ["active_project_id"],
        ["id"],
    )
