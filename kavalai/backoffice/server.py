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

import asyncio
import base64
import json
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

import numpy as np
import uvicorn
from authlib.integrations.starlette_client import OAuth
from fastapi import Body, FastAPI, HTTPException, Request, status
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from kavalai import stats as agent_stats
from kavalai.agent_service import AgentService
from kavalai.backoffice import db
from kavalai.backoffice import sessions as agent_sessions
from kavalai.backoffice.db import is_member, is_owner
from kavalai.backoffice.embedding_projector import train_pca
from kavalai.backoffice.project_service import ProjectService
from kavalai.crud import delete, get_all, get_one, insert, select, update
from kavalai.db import Agent, db_manager
from kavalai.llm_clients.streamer import StreamContent, Streamer
from kavalai.rag import PostgresRagService

app = FastAPI()


@asynccontextmanager
async def get_backoffice_session():
    """
    Context manager to provide a session for the backoffice database.
    Handles connection errors gracefully.
    """
    try:
        async with db.AsyncBackofficeSession() as session:
            yield session
    except HTTPException:
        # Intentional API errors raised inside the block (e.g. a 403/404 from an
        # access check) must propagate unchanged, not be masked as a 503.
        raise
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to backoffice database: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backoffice database is not connected. Please check your database settings.",
        )


def project_sessionmaker(project: db.Project) -> async_sessionmaker[AsyncSession]:
    """A sessionmaker for the agent database the project points at."""
    return db_manager.get_sessionmaker(
        user=project.db_user,
        password=project.db_password,
        host=project.db_host,
        port=project.db_port,
        db_name=project.db_name,
        schema=project.db_schema,
    )


@asynccontextmanager
async def get_project_session(project: db.Project):
    """
    Context manager to provide a session for a specific project's agent database.
    Handles connection errors gracefully.
    """
    sessionmaker = project_sessionmaker(project)
    try:
        async with sessionmaker() as session:
            yield session
    except HTTPException:
        # Let intentional API errors from inside the block propagate unchanged.
        raise
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to project database for {project.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database is not connected for project '{project.name}'. Please check your database settings.",
        )


oauth = OAuth()


def required_setting(name: str) -> str:
    """A setting the backoffice cannot run without.

    There is no development fallback on purpose: a cookie signed with a
    well-known key, or an OAuth client with no id, is a backoffice that looks
    as if it works right up to the moment it is exposed.
    """
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. The backoffice needs KAVALAI_BO_GOOGLE_CLIENT_ID, "
            "KAVALAI_BO_GOOGLE_CLIENT_SECRET, KAVALAI_BO_SESSION_SECRET_KEY and "
            "KAVALAI_BO_FRONTEND_URL — see .env.example."
        )
    return value


# Trust X-Forwarded-* headers from the reverse proxy / Docker network.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

oauth.register(
    name="google",
    client_id=required_setting("KAVALAI_BO_GOOGLE_CLIENT_ID"),
    client_secret=required_setting("KAVALAI_BO_GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Where a completed sign-in lands. Read once, so a missing value fails at
# start-up rather than at the first login.
FRONTEND_URL = required_setting("KAVALAI_BO_FRONTEND_URL")

app.add_middleware(
    SessionMiddleware,
    secret_key=required_setting("KAVALAI_BO_SESSION_SECRET_KEY"),
    same_site="lax",
    https_only=False,  # Set to True if you are using HTTPS
    domain=None,  # Should be None for localhost/development
    path="/",
)


async def authenticate_and_sync_user(user_info: dict):
    async with get_backoffice_session() as session:
        email = user_info.get("email")
        stmt = select(db.User).where(db.User.email == email)
        user = (await session.execute(stmt)).scalars().first()
        if not user:
            logger.warning(f"Unauthorized login attempt: {email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not registered in the system.",
            )

        # Update picture and name
        user.name = user_info.get("name", user.name)
        user.picture = user_info.get("picture", user.picture)

        await session.commit()
        await session.refresh(user)

        # Drop an active project the user can no longer access, so the login
        # never seeds the session with an id that only yields 403s.
        await db.resolve_active_project_id(session, user.id)
        await session.refresh(user)
        return user


def is_logged_in(request: Request):
    return request.session.get("user_info") is not None


def assert_logged_in(request: Request):
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def assert_is_admin(request: Request):
    user_session = request.session.get("user_info")
    if not user_session.get("is_admin"):
        raise HTTPException(
            status_code=403, detail="Only administrators can create new projects."
        )


async def assert_is_owner(session: db.AsyncSession, request: Request, project_id: UUID):
    if not await is_owner(
        session, UUID(request.session.get("user_info")["id"]), project_id
    ):
        raise HTTPException(status_code=403, detail="Only project owners can do this.")


async def assert_is_admin_or_owner(
    session: db.AsyncSession, request: Request, project_id: UUID
):
    if not request.session.get("user_info").get("is_admin"):
        await assert_is_owner(session, request, project_id)


async def assert_is_member(
    session: db.AsyncSession, request: Request, project_id: UUID
):
    if not await is_member(
        session, UUID(request.session.get("user_info")["id"]), project_id
    ):
        raise HTTPException(status_code=403, detail="Must be a member of the project.")


async def get_project_and_assert_access(
    request: Request, project_id: UUID
) -> db.Project:
    """Fetch project and assert user is a member."""
    async with get_backoffice_session() as session:
        await assert_is_member(session, request, project_id)
        project = await get_one(session, db.Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/logout")
async def logout(request: Request):
    del request.session["user_info"]
    return JSONResponse({"logged_in": is_logged_in(request)})


@app.get("/user/get_details")
async def user_details(request: Request):
    assert_logged_in(request)
    user_info = request.session.get("user_info")

    # The cookie may still carry an active project the user can no longer
    # reach (deleted project, revoked membership), which would make every
    # project-scoped page fail with 403. Re-resolve it on each page load so a
    # session that went stale repairs itself without a re-login.
    async with get_backoffice_session() as session:
        active_project_id = await db.resolve_active_project_id(
            session, UUID(user_info["id"])
        )

    resolved = str(active_project_id) if active_project_id else None
    if user_info.get("active_project_id") != resolved:
        user_info["active_project_id"] = resolved
        request.session["user_info"] = user_info
    return user_info


@app.get("/auth/google/callback")
async def google_auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = await oauth.google.userinfo(token=token)
        db_user = await authenticate_and_sync_user(user_info)
        # Store essential info in the session
        request.session["user_info"] = {
            "id": str(db_user.id),
            "email": db_user.email,
            "name": db_user.name,
            "picture": db_user.picture,
            "is_admin": db_user.is_admin,
            "active_project_id": str(db_user.active_project_id)
            if db_user.active_project_id
            else None,
        }
        # Clear OAuth state from session after successful login
        if "_google_authlib_state_" in request.session:
            del request.session["_google_authlib_state_"]

        return RedirectResponse(url=FRONTEND_URL)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        # If it's a mismatching state error, it might be due to stale session
        if "mismatching_state" in str(e):
            return RedirectResponse(url="/login")
        raise HTTPException(status_code=400, detail="Authentication failed.")


@app.post("/user/set_active_project/{project_id}")
async def set_active_project(project_id: UUID, request: Request):
    assert_logged_in(request)
    user_id = UUID(request.session.get("user_info")["id"])
    async with get_backoffice_session() as session:
        await assert_is_member(session, request, project_id)
        await update(session, db.User, user_id, {"active_project_id": project_id})

        user_info = request.session.get("user_info", {})
        user_info["active_project_id"] = str(project_id)
        request.session["user_info"] = user_info
        return {"status": "ok", "active_project_id": project_id}


@app.post("/projects/create")
async def projects_create(request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    assert_is_admin(request)
    user_session = request.session.get("user_info")
    service = ProjectService(db.AsyncBackofficeSession)
    return await service.create_project(data, UUID(user_session["id"]))


@app.get("/projects/get/{project_id}")
async def projects_get_by_id(project_id: UUID, request: Request):
    assert_logged_in(request)
    return await get_project_and_assert_access(request, project_id)


@app.get("/projects/all")
async def projects_get_all(request: Request):
    assert_logged_in(request)
    user_id = UUID(request.session.get("user_info")["id"])
    service = ProjectService(db.AsyncBackofficeSession)
    return await service.get_user_projects(user_id)


@app.put("/projects/update/{project_id}")
async def projects_update(project_id: UUID, request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    async with get_backoffice_session() as session:
        await assert_is_owner(session, request, project_id)
    service = ProjectService(db.AsyncBackofficeSession)
    updated = await service.update_project(project_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found.")
    return updated


@app.delete("/projects/delete/{project_id}")
async def projects_delete(project_id: UUID, request: Request):
    assert_logged_in(request)
    async with get_backoffice_session() as session:
        await assert_is_owner(session, request, project_id)
    service = ProjectService(db.AsyncBackofficeSession)
    success = await service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


@app.get("/users/all")
async def users_get_all(request: Request):
    assert_logged_in(request)
    assert_is_admin(request)
    async with get_backoffice_session() as session:
        return await get_all(session, db.User)


@app.post("/users/create")
async def users_create(request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    assert_is_admin(request)
    async with get_backoffice_session() as session:
        return await insert(session, db.User, data)


@app.put("/users/update/{user_id}")
async def users_update(user_id: UUID, request: Request, data: dict = Body(...)):
    assert_logged_in(request)
    assert_is_admin(request)
    async with get_backoffice_session() as session:
        updated = await update(session, db.User, user_id, data)
        if not updated:
            raise HTTPException(status_code=404, detail="User not found.")
        return updated


@app.delete("/users/delete/{user_id}")
async def users_delete(user_id: UUID, request: Request):
    assert_logged_in(request)
    assert_is_admin(request)
    if str(user_id) == request.session.get("user_info")["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete yourself.")
    async with get_backoffice_session() as session:
        success = await delete(session, db.User, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "deleted"}


@app.get("/agents/get/{project_id}/{agent_id}")
async def agents_get_by_id(project_id: UUID, agent_id: UUID, request: Request):
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        agent = await get_one(session, Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent


@app.post("/workflows/render-svg")
async def workflows_render_svg(request: Request, data: dict = Body(...)):
    """Render a workflow graph to an SVG diagram.

    Accepts ``{"workflow": {...}}`` (or a bare workflow dict) and returns an
    ``image/svg+xml`` document. The agents page uses this to show an agent's
    workflow as a backend-generated diagram (replacing the client-side build).
    """
    assert_logged_in(request)
    from kavalai.workflow import render_workflow_svg

    workflow = data.get("workflow", data)
    try:
        svg = render_workflow_svg(workflow)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not render workflow: {exc}")
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/agents/all/{project_id}")
async def agents_get_all(project_id: UUID, request: Request):
    """Fetch all agents belonging to a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        return await get_all(session, Agent)


@app.get("/agents/stats/{project_id}")
async def agents_get_stats(
    project_id: UUID, request: Request, days: int = 7, agent_id: UUID | None = None
):
    """Fetch daily stats for agents in a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        return await agent_stats.get_daily_stats(session, days=days, agent_id=agent_id)


@app.get("/agents/summary-stats/{project_id}")
async def agents_get_summary_stats(
    project_id: UUID, request: Request, agent_id: UUID | None = None
):
    """Fetch summary stats (last 30 days) for agents in a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        return await agent_stats.get_summary_stats(session, agent_id=agent_id)


@app.get("/agents/sessions/{project_id}")
async def agents_get_sessions(
    project_id: UUID,
    request: Request,
    agent_id: UUID | None = None,
    search: str | None = None,
    external_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Fetch session summaries for a specific project.

    ``external_id`` matches the session's caller-supplied key as a prefix.
    Evaluation runs record their sessions as ``eval:{tag}:{case}``, so this is
    the click-through from a failing case to the conversation that produced it.
    """
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        return await agent_sessions.get_sessions_summary(
            session,
            agent_id=agent_id,
            search=search,
            external_id=external_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )


@app.get("/agents/sessions/{project_id}/{session_id}/details")
async def agents_get_session_details(
    project_id: UUID,
    session_id: UUID,
    request: Request,
):
    """Fetch all details (messages, runs, tasks) for a specific session."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    async with get_project_session(project) as session:
        return await agent_sessions.get_session_details(session, session_id)


@app.get("/projects/{project_id}/llm-call-stats")
async def projects_get_llm_call_stats(
    project_id: UUID,
    request: Request,
    call_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Fetch paginated LLM call stats for a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    # AgentService opens its own sessions, so connection errors surface here.
    try:
        service = AgentService(project_sessionmaker(project))
        return await service.get_model_call_stats(
            call_type=call_type, limit=limit, offset=offset
        )
    except Exception as e:
        logger.error(f"Failed to connect to project database for {project.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database is not connected for project '{project.name}'. Please check your database settings.",
        )


def _reuse_session(session: AsyncSession):
    """A session factory that hands out one already-open session."""

    @asynccontextmanager
    async def factory():
        yield session

    return factory


async def _cache_value(
    bo_session: db.AsyncSession, project_id: UUID, name: str
) -> str | None:
    stmt = select(db.ProjectCache).where(
        db.ProjectCache.project_id == project_id, db.ProjectCache.name == name
    )
    entry = (await bo_session.execute(stmt)).scalar_one_or_none()
    return entry.value if entry else None


async def _pca_projection(
    project_id: UUID,
    collection_name: str,
    rag_service: PostgresRagService,
    text: str,
    normalizer,
    results,
) -> dict | None:
    """Project the query and its results onto the collection's trained PCA plane.

    Returns ``None`` when no model has been trained for the collection or the
    projection fails — the query results are still worth returning on their
    own.
    """
    async with get_backoffice_session() as bo_session:
        model_data = await _cache_value(
            bo_session, project_id, f"pca_model_{collection_name}"
        )
        if model_data is None:
            return None
        try:
            ipca = pickle.loads(base64.b64decode(model_data))  # nosec B301

            embeddings, _ = await rag_service.embedding_client.compute_embeddings(
                texts=[text], normalizer=normalizer
            )
            query_point = ipca.transform(np.array(embeddings))[0]

            # RagServiceResult carries no embedding; fetch them through the
            # service, which owns the storage.
            id_to_embedding = await rag_service.get_embeddings_by_ids(
                collection_name, [r.id for r in results]
            )
            result_points = []
            for r in results:
                emb = id_to_embedding.get(r.id)
                if emb is None:
                    continue
                try:
                    pt = ipca.transform(np.array([emb]))[0]
                except Exception as e:
                    logger.error(
                        f"Failed to transform embedding for result {r.id}: {e}"
                    )
                    continue
                result_points.append(
                    {"label": r.content[:100], "x": float(pt[0]), "y": float(pt[1])}
                )

            sample_data = await _cache_value(
                bo_session, project_id, f"pca_sample_train_data_{collection_name}"
            )
            return {
                "query": {
                    "label": text,
                    "x": float(query_point[0]),
                    "y": float(query_point[1]),
                },
                "results": result_points,
                "samples": json.loads(sample_data) if sample_data else [],
            }
        except Exception as e:
            logger.error(f"Failed to process PCA data: {e}")
            return None


@app.post("/projects/{project_id}/rag/query")
async def projects_rag_query(
    project_id: UUID,
    request: Request,
    query_data: dict = Body(...),
):
    """Execute a RAG query for a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    model = query_data.get("model")
    text = query_data.get("text")
    collection_name = query_data.get("collection_name")
    if not model or not text:
        raise HTTPException(status_code=400, detail="model and text are required")

    normalizer = None
    if normalizer_yaml := query_data.get("normalizer_yaml"):
        from kavalai.normalizer import Normalizer

        try:
            normalizer = Normalizer.from_yaml(normalizer_yaml)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid normalizer YAML: {str(e)}"
            )

    async with get_project_session(project) as session:
        rag_service = PostgresRagService(
            _reuse_session(session),
            model,
            normalizer=normalizer,
            schema=project.db_schema,
        )
        results = await rag_service.query(
            text=text,
            top_k=query_data.get("top_k", 5),
            collection_name=collection_name,
            source_ids=query_data.get("source_ids"),
            keep_best=query_data.get("keep_best", False),
        )

        pca_data = None
        if collection_name:
            pca_data = await _pca_projection(
                project_id, collection_name, rag_service, text, normalizer, results
            )

        return {"results": results, "pca_data": pca_data}


@app.get("/projects/{project_id}/rag/stats")
async def projects_rag_stats(project_id: UUID, request: Request):
    """Fetch RAG statistics for a specific project."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    # Stats come from the collection registry, so no embedding model is needed.
    async with get_project_session(project) as session:
        rag_service = PostgresRagService(
            _reuse_session(session), model=None, schema=project.db_schema
        )
        return await rag_service.get_stats()


@app.get("/projects/{project_id}/rag/train-pca")
async def projects_train_pca(project_id: UUID, collection_name: str, request: Request):
    """Trigger PCA training for a specific project and collection."""
    assert_logged_in(request)
    project = await get_project_and_assert_access(request, project_id)

    streamer = Streamer(stream_delta=True)

    rag_service = PostgresRagService(
        lambda: get_project_session(project),
        model=None,  # export-only: PCA training never computes embeddings
        schema=project.db_schema,
    )

    async def run_training():
        try:
            await train_pca(
                bo_session_maker=get_backoffice_session,
                rag_service=rag_service,
                project_id=project.id,
                collection_name=collection_name,
                streamer=streamer.get_value_streamer("pca_streamer"),
            )
        except Exception as e:
            logger.error(f"PCA training failed: {e}")
            await streamer.stream_error(e)

    async def event_generator():
        pca_task = asyncio.create_task(run_training())
        try:
            async for chunk in streamer:
                yield {"data": chunk.model_dump_json()}
            await pca_task
        except Exception as e:
            logger.error(f"Error in PCA event generator: {e}")
            yield {
                "data": StreamContent(
                    type="error", name="error", value=str(e)
                ).model_dump_json()
            }

    return EventSourceResponse(event_generator())


@app.post("/projects/test-connection/{project_id}")
async def projects_test_connection(
    project_id: str, request: Request, data: dict = Body(default={})
):
    """Test connection to the project database."""
    assert_logged_in(request)

    if project_id == "new":
        project = db.Project(**data)
    else:
        project = await get_project_and_assert_access(request, UUID(project_id))

    service = ProjectService(db.AsyncBackofficeSession)
    return await service.test_connection(project)


@app.get("/projects/{project_id}/members")
async def projects_get_members(project_id: UUID, request: Request):
    assert_logged_in(request)
    async with get_backoffice_session() as session:
        await assert_is_member(session, request, project_id)
    service = ProjectService(db.AsyncBackofficeSession)
    return await service.get_members(project_id)


@app.post("/projects/{project_id}/members/add")
async def projects_add_member(
    project_id: UUID, request: Request, data: dict = Body(...)
):
    assert_logged_in(request)
    user_id = UUID(data["user_id"])
    role = db.ProjectRole(data["role"])

    async with get_backoffice_session() as session:
        await assert_is_admin_or_owner(session, request, project_id)

    service = ProjectService(db.AsyncBackofficeSession)
    await service.add_member(project_id, user_id, role)
    return {"status": "added"}


@app.put("/projects/{project_id}/members/update")
async def projects_update_member(
    project_id: UUID, request: Request, data: dict = Body(...)
):
    assert_logged_in(request)
    user_id = UUID(data["user_id"])
    new_role = db.ProjectRole(data["role"])

    async with get_backoffice_session() as session:
        await assert_is_admin_or_owner(session, request, project_id)

    service = ProjectService(db.AsyncBackofficeSession)
    await service.update_member_role(project_id, user_id, new_role)
    return {"status": "updated"}


@app.delete("/projects/{project_id}/members/remove/{user_id}")
async def projects_remove_member(project_id: UUID, user_id: UUID, request: Request):
    assert_logged_in(request)
    async with get_backoffice_session() as session:
        await assert_is_admin_or_owner(session, request, project_id)

    service = ProjectService(db.AsyncBackofficeSession)
    await service.remove_member(project_id, user_id)
    return {"status": "removed"}


if __name__ == "__main__":
    # Only the entry point reads the environment; the rest of the module takes
    # its configuration from arguments.
    config = uvicorn.Config(
        "kavalai.backoffice.server:app",
        host=os.environ.get("KAVALAI_BO_HOST", "127.0.0.1"),
        port=int(os.environ.get("KAVALAI_BO_PORT", "8000")),
        log_level="info",
        reload=True,
        access_log=True,
    )
    server = uvicorn.Server(config)
    server.run()
