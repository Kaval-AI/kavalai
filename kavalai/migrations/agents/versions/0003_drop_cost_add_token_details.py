"""drop cost/currency from model_call_stats, add cached and reasoning tokens

Cost was never populated: providers return token counts, not money, so any
figure in these columns would have come from a price table the library would
have to keep current. Cached input tokens are billed at a fraction of fresh
input, so a cost derived from ``prompt_tokens`` alone would have been wrong by
a multiple rather than merely stale.

The two new counts are what make cost computable downstream (e.g. with
``genai-prices``) from a row like this one, and are useful on their own for
seeing how much of a prompt is being served from cache.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
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
    with op.batch_alter_table("model_call_stats", schema=_target_schema()) as batch_op:
        batch_op.add_column(
            sa.Column("cached_prompt_tokens", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("reasoning_tokens", sa.Integer(), nullable=True))
        batch_op.drop_column("cost")
        batch_op.drop_column("currency")


def downgrade() -> None:
    with op.batch_alter_table("model_call_stats", schema=_target_schema()) as batch_op:
        batch_op.add_column(sa.Column("currency", sa.TEXT(), nullable=True))
        batch_op.add_column(sa.Column("cost", sa.Numeric(10, 6), nullable=True))
        batch_op.drop_column("reasoning_tokens")
        batch_op.drop_column("cached_prompt_tokens")
