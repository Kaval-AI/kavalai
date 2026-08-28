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

from typing import List, Dict, Any, Optional
from uuid import UUID
from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import func, select, text, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kavalai.backoffice import db
from kavalai.crud import insert, update, delete
from kavalai.db import db_manager


class ProjectService:
    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker

    async def get_user_projects(self, user_id: UUID) -> List[Dict[str, Any]]:
        async with self.session_maker() as session:
            return await db.get_user_projects(session, user_id)

    async def create_project(self, data: Dict[str, Any], owner_id: UUID) -> db.Project:
        async with self.session_maker() as session:
            new_project = await insert(session, db.Project, data)
            # The creator is the project's first owner.
            membership_data = {
                "user_id": owner_id,
                "project_id": new_project.id,
                "role": db.ProjectRole.owner,
            }
            await insert(session, db.ProjectMembership, membership_data)
            return new_project

    async def update_project(
        self, project_id: UUID, data: Dict[str, Any]
    ) -> Optional[db.Project]:
        async with self.session_maker() as session:
            return await update(session, db.Project, project_id, data)

    async def delete_project(self, project_id: UUID) -> bool:
        async with self.session_maker() as session:
            # Detach the project from anyone who has it selected first, so the
            # delete cannot fail on the users.active_project_id foreign key and
            # nobody is left pointing at a project that no longer exists.
            await self._clear_active_project(session, project_id)
            return await delete(session, db.Project, project_id)

    @staticmethod
    async def _clear_active_project(
        session: AsyncSession, project_id: UUID, user_id: Optional[UUID] = None
    ) -> None:
        """Clear ``active_project_id`` for users pointing at ``project_id``."""
        stmt = sa_update(db.User).where(db.User.active_project_id == project_id)
        if user_id is not None:
            stmt = stmt.where(db.User.id == user_id)
        await session.execute(stmt.values(active_project_id=None))
        await session.commit()

    async def get_members(self, project_id: UUID) -> List[Dict[str, Any]]:
        stmt = (
            select(db.User, db.ProjectMembership.role)
            .join(db.ProjectMembership, db.User.id == db.ProjectMembership.user_id)
            .where(db.ProjectMembership.project_id == project_id)
        )
        async with self.session_maker() as session:
            result = await session.execute(stmt)
            return [
                {
                    "id": str(user.id),
                    "name": user.name,
                    "email": user.email,
                    "picture": user.picture,
                    "role": role.value,
                }
                for user, role in result.all()
            ]

    async def add_member(
        self, project_id: UUID, user_id: UUID, role: db.ProjectRole
    ) -> None:
        async with self.session_maker() as session:
            if await db.is_member(session, user_id, project_id):
                raise HTTPException(status_code=400, detail="User is already a member.")

            membership_data = {
                "user_id": user_id,
                "project_id": project_id,
                "role": role,
            }
            await insert(session, db.ProjectMembership, membership_data)

    @staticmethod
    async def _get_membership(
        session: AsyncSession, project_id: UUID, user_id: UUID
    ) -> db.ProjectMembership:
        stmt = select(db.ProjectMembership).where(
            db.ProjectMembership.project_id == project_id,
            db.ProjectMembership.user_id == user_id,
        )
        membership = (await session.execute(stmt)).scalars().first()
        if not membership:
            raise HTTPException(status_code=404, detail="Membership not found.")
        return membership

    @staticmethod
    async def _assert_not_last_owner(
        session: AsyncSession, membership: db.ProjectMembership, detail: str
    ) -> None:
        """Refuse to strip ownership from a project's only owner."""
        if membership.role != db.ProjectRole.owner:
            return
        stmt = (
            select(func.count())
            .select_from(db.ProjectMembership)
            .where(
                db.ProjectMembership.project_id == membership.project_id,
                db.ProjectMembership.role == db.ProjectRole.owner,
            )
        )
        if (await session.execute(stmt)).scalar_one() <= 1:
            raise HTTPException(status_code=400, detail=detail)

    async def update_member_role(
        self, project_id: UUID, user_id: UUID, new_role: db.ProjectRole
    ) -> None:
        async with self.session_maker() as session:
            membership = await self._get_membership(session, project_id, user_id)
            if new_role != db.ProjectRole.owner:
                await self._assert_not_last_owner(
                    session, membership, "Cannot demote the last owner of the project."
                )

            stmt = (
                sa_update(db.ProjectMembership)
                .where(
                    db.ProjectMembership.project_id == project_id,
                    db.ProjectMembership.user_id == user_id,
                )
                .values(role=new_role)
            )
            await session.execute(stmt)
            await session.commit()

    async def remove_member(self, project_id: UUID, user_id: UUID) -> None:
        async with self.session_maker() as session:
            membership = await self._get_membership(session, project_id, user_id)
            await self._assert_not_last_owner(
                session, membership, "Cannot remove the last owner of the project."
            )

            stmt = sa_delete(db.ProjectMembership).where(
                db.ProjectMembership.project_id == project_id,
                db.ProjectMembership.user_id == user_id,
            )
            await session.execute(stmt)
            await session.commit()

            # Without this the removed user keeps the project selected and
            # every project-scoped request answers 403.
            await self._clear_active_project(session, project_id, user_id)

    async def test_connection(self, project: db.Project) -> Dict[str, str]:
        try:
            logger.info(
                f"Testing connection to project database: host={project.db_host}, "
                f"port={project.db_port}, db={project.db_name}, user={project.db_user}"
            )
            project_session_maker = db_manager.get_sessionmaker(
                user=project.db_user,
                password=project.db_password,
                host=project.db_host,
                port=project.db_port,
                db_name=project.db_name,
                schema=project.db_schema,
            )
            async with project_session_maker() as project_session:
                await project_session.execute(text("SELECT 1"))
            return {"status": "success", "message": "Connection successful"}
        except Exception as e:
            logger.error(f"Failed to connect to project database: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to connect: {str(e)}",
            )
