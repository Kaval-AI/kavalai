"""Alembic environment for the *backoffice* migration set.

Covers the backoffice application tables: ``users``, ``projects``,
``project_memberships``, ``project_cache``.
"""

from kavalai.backoffice.db import Base
from kavalai.migrations.common import agents_render_item, run_migrations

target_metadata = Base.metadata

run_migrations(target_metadata, render_item=agents_render_item)
