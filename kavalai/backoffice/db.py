"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from datetime import datetime, timezone
from enum import Enum as PyEnum
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import Enum, MetaData
from sqlalchemy import TEXT, Boolean, ForeignKey, DateTime, Integer
from sqlalchemy import select, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool

from kavalai.db import ensure_async_scheme


def AsyncBackofficeSession(uri: str | None = None, schema: str | None = None):
    """Create a backoffice DB session.

    ``uri``/``schema`` default to ``KAVALAI_BO_DB_URI``/``KAVALAI_BO_DB_SCHEMA``
    when omitted — the backoffice is an application, so reading its own
    environment at call time is acceptable; library code should always pass
    both explicitly. The schema is applied via ``schema_translate_map``.
    """
    uri = uri or os.environ["KAVALAI_BO_DB_URI"]
    if schema is None:
        schema = os.environ.get("KAVALAI_BO_DB_SCHEMA")
    engine = create_async_engine(
        ensure_async_scheme(uri),
        echo=False,
        poolclass=NullPool,
    )
    if schema:
        engine = engine.execution_options(schema_translate_map={None: schema})
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )()


class Base(DeclarativeBase):
    # Schema-less by design: the target schema is applied per-engine via
    # ``schema_translate_map`` (see ``AsyncBackofficeSession``); no env vars
    # are read at import time.
    metadata = MetaData()


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    email: Mapped[str] = mapped_column(TEXT, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    picture: Mapped[str | None] = mapped_column(TEXT)
    active_project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    memberships: Mapped[list["ProjectMembership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT)

    # New Database Connection Columns
    db_host: Mapped[str | None] = mapped_column(TEXT)
    db_port: Mapped[int | None] = mapped_column(Integer, default=5432)
    db_user: Mapped[str | None] = mapped_column(TEXT)
    db_password: Mapped[str | None] = mapped_column(TEXT)
    db_name: Mapped[str | None] = mapped_column(TEXT)
    db_schema: Mapped[str | None] = mapped_column(TEXT, default="public")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    members: Mapped[list["ProjectMembership"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    cache: Mapped[list["ProjectCache"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class ProjectRole(str, PyEnum):
    owner = "owner"
    viewer = "viewer"


class ProjectMembership(Base):
    __tablename__ = "project_memberships"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[ProjectRole] = mapped_column(
        # Stored as plain text with a CHECK constraint instead of a native
        # Postgres ENUM: keeps the model schema-agnostic (no type to place in
        # a schema) and portable across dialects.
        Enum(ProjectRole, name="project_role", native_enum=False),
        nullable=False,
    )

    # Relationship Links
    user: Mapped["User"] = relationship(back_populates="memberships")
    project: Mapped["Project"] = relationship(back_populates="members")


class ProjectCache(Base):
    __tablename__ = "project_cache"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    value: Mapped[str | None] = mapped_column(TEXT)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="cache")

    __table_args__ = (Index("idx_project_cache_name", "name"),)


async def is_member(db: AsyncSession, user_id: UUID, project_id: UUID) -> bool:
    """Check if a user is any kind of member of a project."""
    stmt = select(ProjectMembership).where(
        ProjectMembership.user_id == user_id, ProjectMembership.project_id == project_id
    )
    result = await db.execute(stmt)
    return result.scalars().first() is not None


async def is_owner(db: AsyncSession, user_id: UUID, project_id: UUID) -> bool:
    """Check if a user has the 'owner' role for a project."""
    stmt = select(ProjectMembership).where(
        ProjectMembership.user_id == user_id,
        ProjectMembership.project_id == project_id,
        ProjectMembership.role == ProjectRole.owner,
    )
    result = await db.execute(stmt)
    return result.scalars().first() is not None


async def resolve_active_project_id(db: AsyncSession, user_id: UUID) -> UUID | None:
    """Return a usable active project for the user, repairing a stale one.

    ``users.active_project_id`` can go stale: the project may have been deleted
    or the user may have lost their membership. Every project-scoped endpoint
    would then answer 403, and nothing would ever clear the bad value. This
    resolves the reference — keeping it when the user is still a member, and
    otherwise falling back to one of their projects (or ``None``) and
    persisting that choice.
    """
    user = await db.get(User, user_id)
    if user is None:
        return None

    if user.active_project_id is not None and await is_member(
        db, user_id, user.active_project_id
    ):
        return user.active_project_id

    stmt = (
        select(ProjectMembership.project_id)
        .join(Project, Project.id == ProjectMembership.project_id)
        .where(ProjectMembership.user_id == user_id)
        .order_by(Project.name)
        .limit(1)
    )
    result = await db.execute(stmt)
    fallback = result.scalars().first()

    if user.active_project_id != fallback:
        logger.info(
            f"Repairing stale active project for user {user_id}: "
            f"{user.active_project_id} -> {fallback}"
        )
        user.active_project_id = fallback
        await db.commit()

    return fallback


async def get_user_projects(db: AsyncSession, user_id: UUID) -> list[dict]:
    """Fetch projects along with the specific user's role and DB details."""
    stmt = (
        select(Project, ProjectMembership.role)
        .join(ProjectMembership, Project.id == ProjectMembership.project_id)
        .where(ProjectMembership.user_id == user_id)
    )

    result = await db.execute(stmt)

    projects_with_roles = []
    for row in result.all():
        project_obj: Project = row[0]
        role: ProjectRole = row[1]

        project_data = {
            "id": str(project_obj.id),
            "name": project_obj.name,
            "description": project_obj.description,
            "db_host": project_obj.db_host,
            "db_port": project_obj.db_port,
            "db_user": project_obj.db_user,
            "db_name": project_obj.db_name,
            "db_schema": project_obj.db_schema,
            "db_password": project_obj.db_password,
            "created_at": project_obj.created_at.isoformat(),
            "updated_at": project_obj.updated_at.isoformat(),
            "role": role.value,
        }
        projects_with_roles.append(project_data)

    return projects_with_roles
