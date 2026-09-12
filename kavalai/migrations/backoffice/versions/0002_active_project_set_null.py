"""users.active_project_id clears itself when the project is deleted

Without ``ON DELETE SET NULL`` deleting a project either fails on this foreign
key or (where the constraint was never created) leaves users pointing at a
project that no longer exists, which makes every project-scoped endpoint answer
403 with no way to recover.

The change runs in batch mode so it also applies to SQLite, which cannot alter
a constraint in place. Revision 0001 left the constraint unnamed; Postgres
named it ``users_active_project_id_fkey`` on its own, and the naming
convention below gives the reflected SQLite constraint the same name so both
dialects drop the same thing.

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
NAMING_CONVENTION = {"fk": "%(table_name)s_%(column_0_name)s_fkey"}


def _target_schema() -> Union[str, None]:
    """The schema batch mode must name explicitly.

    ``schema_translate_map`` rewrites compiled DDL, but ``batch_alter_table``
    reflects the existing table first and reflection does not consult the map.
    The runner applies the map as ``{None: <schema>}``; read it back.
    """
    translate_map = op.get_bind().get_execution_options().get("schema_translate_map")
    return (translate_map or {}).get(None)


def _recreate_foreign_key(ondelete: Union[str, None]) -> None:
    with op.batch_alter_table(
        "users", schema=_target_schema(), naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(FK_NAME, type_="foreignkey")
        batch_op.create_foreign_key(
            FK_NAME,
            "projects",
            ["active_project_id"],
            ["id"],
            referent_schema=_target_schema(),
            ondelete=ondelete,
        )


def upgrade() -> None:
    _recreate_foreign_key("SET NULL")


def downgrade() -> None:
    _recreate_foreign_key(None)
