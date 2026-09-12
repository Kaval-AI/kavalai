"""Launch Kaval.AI agent REST server.

``SSE_PING_INTERVAL_SECONDS`` is how long a stream may stay silent before a
keepalive comment is sent. ``SSE_HEADERS`` are the response headers every SSE
stream carries: no caching, and no buffering by a reverse proxy.
``PUBLIC_FAILURE_MESSAGE`` is what :func:`public_events` sends in place of a
failed run's error text.

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
import importlib
import importlib.util
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Annotated,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)
from uuid import UUID

from environs import Env

try:
    import uvicorn
    from fastapi import Depends
    from fastapi import HTTPException, status, FastAPI, Response, APIRouter
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
except ImportError as exc:
    raise ImportError(
        f"The agent server requires the optional '{exc.name}' package. "
        'Install it with: pip install "kavalai[runtime]"'
    ) from exc
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai.agent_service import AgentService
from kavalai.db import db_manager
from kavalai.rag import rag_service_from_uri
from kavalai.settings import apply_normalizer_from_env, llm_parameters_from_env
from kavalai.llm_clients.registry import register_rag_service
from kavalai.workflow import WorkflowEngine
from kavalai.workflow.models import WorkflowException, WorkflowStreamEvent
from kavalai.workflow.tasklog.postgres import PostgresTaskLogger


SSE_PING_INTERVAL_SECONDS = 15.0

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

PUBLIC_FAILURE_MESSAGE = "The agent could not complete this request."

InputT = TypeVar("InputT", bound=BaseModel)

EventFilter = Callable[[WorkflowStreamEvent], Optional[WorkflowStreamEvent]]


class AgentRequest(BaseModel, Generic[InputT]):
    """The request body of ``POST /run_agent`` and ``POST /stream_agent``.

    ``data`` is the workflow's own input type. ``session_id`` continues a
    session by its id; ``external_id`` continues — or starts — the session
    carrying the caller's own key. A host that serves a workflow from a route
    of its own types the body as ``AgentRequest[MyInput]``, so its clients
    speak the same protocol as the SDK's router.
    """

    session_id: Optional[UUID] = None
    external_id: Optional[str] = None
    data: InputT


def public_events(event: WorkflowStreamEvent) -> Optional[WorkflowStreamEvent]:
    """An event filter for streams served to people other than the operator.

    Forwards the run lifecycle and the streamed content, and withholds what
    only the operator should see:

    * ``node_started`` / ``node_completed`` events, which describe the
      workflow's internals;
    * token usage;
    * the reason attached to a ``restart``;
    * the error text of a failed run, which can carry a provider's message.
      It is replaced by :data:`PUBLIC_FAILURE_MESSAGE` and the run id, so a
      report can be matched to the recorded run; the detail stays in the
      server log and on the run row.

    Which nodes stream content at all is the workflow's ``stream_output``
    setting, not this filter's.
    """
    if event.type in ("node_started", "node_completed"):
        return None
    if event.type == "workflow_failed":
        reference = f" Reference: {event.run_id}." if event.run_id else ""
        return event.model_copy(
            update={"value": f"{PUBLIC_FAILURE_MESSAGE}{reference}"}
        )
    if event.type == "restart":
        return event.model_copy(update={"value": None})
    if event.type == "workflow_completed":
        return event.model_copy(update={"token_usage": None})
    return event


security = HTTPBasic(auto_error=False)

env = Env()
env.read_env()


def validate_auth(credentials: Optional[HTTPBasicCredentials]):
    """Validate HTTP Basic Authentication.

    Authentication is disabled if KAVALAI_AGENT_BASIC_AUTH_USER and
    KAVALAI_AGENT_BASIC_AUTH_PASSWORD are not set in the environment.

    Args:
        credentials: The credentials provided in the request.

    Returns:
        True if authentication is successful or disabled.

    Raises:
        HTTPException: If authentication fails.
    """
    expected_username = env.str("KAVALAI_AGENT_BASIC_AUTH_USER", "")
    expected_password = env.str("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "")

    # Basic auth is disabled
    if not expected_username and not expected_password:
        return True

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    is_correct_username = secrets.compare_digest(
        credentials.username, expected_username
    )
    is_correct_password = secrets.compare_digest(
        credentials.password, expected_password
    )

    if is_correct_username and is_correct_password:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username/password",
        headers={"WWW-Authenticate": "Basic"},
    )


@asynccontextmanager
async def session_scope(session_or_factory):
    """Provide a database session from either a sessionmaker or an existing session.

    This context manager ensures that if a factory is provided, a new session
    is created and closed properly. If an existing session is provided, it
    is used as-is.
    """
    if isinstance(session_or_factory, async_sessionmaker):
        async with session_or_factory() as session:
            yield session
    else:
        yield session_or_factory


async def handle_agent_run(
    engine: WorkflowEngine,
    session_provider: Union[async_sessionmaker, None],
    input_data: dict,
    session_id: Optional[UUID] = None,
    external_id: Optional[str] = None,
):
    """Execute the agent workflow with the provided input.

    This is a standalone handler that can be used directly or wrapped in a FastAPI route.

    Args:
        engine: The v2 WorkflowEngine to execute.
        session_provider: An optional SQLAlchemy async_sessionmaker for database sessions (unused, kept for compatibility).
        input_data: The input data for the workflow (already extracted from request).
        session_id: Optional session ID for continuing a previous session.
        external_id: Optional external identifier for tracking.

    Returns:
        A tuple of (session_id, output_data).
    """
    state = await engine.run(
        input_data=input_data,
        session_id=str(session_id) if session_id else None,
        external_id=external_id,
    )
    return state.session_id, state.output_data


def format_sse_event(event: WorkflowStreamEvent) -> str:
    """Render one WorkflowStreamEvent as an SSE frame."""
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


async def stream_sse_events(
    events: AsyncIterable[WorkflowStreamEvent],
    ping_interval: float = SSE_PING_INTERVAL_SECONDS,
    *,
    event_filter: Optional[EventFilter] = None,
) -> AsyncGenerator[str, None]:
    """Format a workflow event stream as SSE frames with keepalive pings.

    A ``: ping`` comment frame is emitted whenever no event arrives within
    ``ping_interval`` seconds (silent stretches such as long tool calls), so
    proxies don't drop the connection. A ``WorkflowException`` from the engine
    ends the stream quietly — the engine has already emitted the
    ``workflow_failed`` event, and an SSE response cannot change its status
    code after the headers are sent.

    ``event_filter`` sees every event before it is written, and returns the
    event to send, a modified copy, or ``None`` to drop it. See
    :func:`public_events`.

    The event stream is consumed by a single pump task (an async generator's
    frames must run in one task — the engine binds task-scoped log context),
    while this generator races the queue against the ping timer. Closing this
    generator cancels the pump, which aborts the workflow run.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _END = ("end", None)

    async def pump():
        try:
            async for event in events:
                await queue.put(("event", event))
        except WorkflowException as e:
            logger.error(f"Workflow failed during streaming: {e}")
        except Exception:
            logger.exception("Unexpected error while streaming workflow events")
        finally:
            queue.put_nowait(_END)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            try:
                kind, event = await asyncio.wait_for(queue.get(), timeout=ping_interval)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if kind == "end":
                return
            if event_filter is not None:
                event = event_filter(event)
                if event is None:
                    continue
            yield format_sse_event(event)
    finally:
        # On early consumer exit (client disconnect), cancel the pump so the
        # engine generator aborts and records the run.
        pump_task.cancel()
        try:
            await pump_task
        except BaseException:  # noqa: BLE001 - teardown is best-effort
            pass


def sse_response(
    events: AsyncIterable[WorkflowStreamEvent],
    *,
    event_filter: Optional[EventFilter] = None,
    headers: Optional[dict[str, str]] = None,
    ping_interval: float = SSE_PING_INTERVAL_SECONDS,
) -> StreamingResponse:
    """Serve a workflow event stream as a Server-Sent Events response.

    What the router's ``/stream_agent`` returns, available to a host that
    serves a workflow from a route of its own — behind its own admission
    checks — so the frames, the keepalive and the headers stay the SDK's.
    ``headers`` are added to :data:`SSE_HEADERS`.
    """
    return StreamingResponse(
        stream_sse_events(events, ping_interval, event_filter=event_filter),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, **(headers or {})},
    )


def create_default_auth_dependency() -> Callable:
    """Create the default HTTP Basic Auth dependency using environment variables.

    Returns:
        A FastAPI dependency function that validates HTTP Basic Authentication.
    """

    def auth_dependency(
        credentials: Annotated[Optional[HTTPBasicCredentials], Depends(security)],
    ):
        return validate_auth(credentials)

    return auth_dependency


def create_agent_router(
    engine: WorkflowEngine,
    session_provider: Union[async_sessionmaker, None] = None,
    auth_dependency: Optional[Callable] = None,
    *,
    event_filter: Optional[EventFilter] = None,
) -> APIRouter:
    """Create a FastAPI router for a given workflow.

    This function creates a reusable router that can be mounted on existing FastAPI applications,
    allowing for flexible composition and custom routing configurations.

    The router serves ``POST /run_agent`` (blocking) and ``POST /stream_agent``
    (SSE), both typed by the workflow's own input and output data types, plus
    ``GET /workflow``, ``GET /liveness`` and ``GET /health``.

    Args:
        engine: The :class:`~kavalai.WorkflowEngine` instance to serve.
        session_provider: An optional SQLAlchemy async_sessionmaker to provide
            database sessions for agent execution.
        auth_dependency: An optional FastAPI dependency for authentication.
            If None, the default HTTP Basic Auth will be used.
            Pass a custom dependency function to use your own auth, or pass
            ``lambda: None`` to disable authentication.
        event_filter: Applied to every event ``/stream_agent`` sends. Pass
            :func:`public_events` when the stream is served to people other
            than the operator. ``/run_agent`` returns only the output, so it
            is unaffected.

    Returns:
        An APIRouter instance with configured endpoints.

    Example:
        .. code-block:: python

            # Use with custom auth
            def my_auth():
                # Custom auth logic
                pass

            router = create_agent_router(engine, session_provider, auth_dependency=my_auth)
            app.include_router(router, prefix="/agents/my-workflow")

            # Or disable auth entirely
            router = create_agent_router(engine, session_provider, auth_dependency=lambda: None)
    """
    router = APIRouter()

    # Use default auth if none provided
    if auth_dependency is None:
        auth_dependency = create_default_auth_dependency()

    InputDataType = engine.get_data_type("input")
    OutputDataType = engine.get_data_type(engine.graph.output_type)

    # Named, rather than used as AgentRequest[...] directly, so the OpenAPI
    # schema keeps the name clients have always seen.
    class InputType(AgentRequest[InputDataType]):
        pass

    # Define the response body schema.
    class OutputType(BaseModel):
        session_id: Optional[UUID]
        data: OutputDataType

    @router.post("/run_agent", response_model=OutputType)
    async def run_agent(
        input_data: InputType,
        _auth: Annotated[None, Depends(auth_dependency)],
    ) -> OutputType:
        """Execute the agent workflow with the provided input."""
        session_id, data = await handle_agent_run(
            engine=engine,
            session_provider=session_provider,
            input_data=input_data.data.model_dump(),
            session_id=input_data.session_id,
            external_id=input_data.external_id,
        )
        return OutputType(session_id=session_id, data=data)

    @router.post("/stream_agent")
    async def stream_agent(
        input_data: InputType,
        _auth: Annotated[None, Depends(auth_dependency)],
    ) -> StreamingResponse:
        """Execute the agent workflow, streaming progress as SSE.

        Each frame is a ``WorkflowStreamEvent`` JSON payload under
        ``event: <type>``; see that model for the event contract. Browsers
        must consume this with ``fetch()`` streaming — ``EventSource`` cannot
        send a POST body or auth headers. Disconnecting aborts the run.
        """
        events = engine.run_stream(
            input_data=input_data.data.model_dump(),
            session_id=str(input_data.session_id) if input_data.session_id else None,
            external_id=input_data.external_id,
        )
        return sse_response(events, event_filter=event_filter)

    @router.get("/workflow")
    async def get_workflow(
        _auth: Annotated[None, Depends(auth_dependency)],
    ):
        """Retrieve the workflow configuration.

        Behind the same authentication as every other endpoint — which means
        public when none is configured, so the response is redacted: MCP server
        environment values are secrets in practice (that is where an API key
        for a stdio server ends up) and never leave the process.
        """
        return Response(
            content=engine.graph.model_dump_public_json(),
            media_type="application/json",
        )

    @router.get("/liveness")
    async def liveness():
        """Liveness probe for K8s."""
        return {"status": "ok"}

    @router.get("/health")
    async def health():
        """Health probe for K8s. Checks DB connectivity."""
        try:
            async with session_scope(session_provider) as session:
                await session.execute(text("SELECT 1"))
            return {"status": "ok", "database": "connected"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed",
            )

    return router


def create_agent_app(
    engine: WorkflowEngine,
    session_provider: Union[async_sessionmaker, None] = None,
    auth_dependency: Optional[Callable] = None,
    *,
    event_filter: Optional[EventFilter] = None,
) -> FastAPI:
    """Create a FastAPI application for a given workflow.

    The application dynamically generates input and output models based on the
    workflow's schema and provides endpoints to run the agent and retrieve
    its configuration.

    Args:
        engine: The :class:`~kavalai.WorkflowEngine` instance to serve.
        session_provider: An optional SQLAlchemy async_sessionmaker to provide
            database sessions for agent execution.
        auth_dependency: An optional FastAPI dependency for authentication.
            If None, the default HTTP Basic Auth will be used.
        event_filter: Applied to the streamed events; see
            :func:`create_agent_router`.

    Returns:
        A FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        """Own the engine's tool servers for the lifetime of the process.

        MCP servers are connected once here rather than per request: their
        tool lists are what an agent node is told it can call, and starting a
        subprocess on the request path would be a poor trade. Connection
        failures surface at startup, before any request is served.
        """
        await engine.connect()
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(
        title=engine.graph.name,
        description=engine.graph.description,
        version=engine.graph.version,
        lifespan=lifespan,
    )
    app.state.engine = engine

    router = create_agent_router(
        engine=engine,
        session_provider=session_provider,
        auth_dependency=auth_dependency,
        event_filter=event_filter,
    )
    app.include_router(router)

    return app


def mask_db_uri(db_uri: str) -> str:
    """Mask the password in a database URI.

    Args:
        db_uri: The database URI to mask.

    Returns:
        The masked database URI.
    """
    if "@" not in db_uri:
        return db_uri

    try:
        prefix, rest = db_uri.split("://", 1)
        auth, host_path = rest.split("@", 1)
        if ":" in auth:
            user, password = auth.split(":", 1)
            return f"{prefix}://{user}:***@{host_path}"
    except Exception:
        # Fallback if URI parsing fails
        return "***"

    return db_uri


def load_provider_modules(module_names: str) -> list[str]:
    """Import modules that register custom LLM, embedding or RAG backends.

    Each module is expected to call :func:`~kavalai.register_llm_provider`,
    :func:`~kavalai.register_embedding_provider` or
    :func:`~kavalai.register_rag_service` at import time. After they are
    loaded, every dotted registration is resolved, so a mistyped path fails
    here rather than on the first request that happens to reach that node.

    Args:
        module_names: Comma-separated dotted module names; blanks are ignored.

    Returns:
        The module names that were imported, in order.

    Raises:
        ImportError: A named module could not be imported.
        RegistryError: A registration names a path that cannot be resolved.
    """
    from kavalai.llm_clients.registry import verify_registrations

    names = [name.strip() for name in module_names.split(",") if name.strip()]
    for name in names:
        logger.info(f"Importing provider module {name}.")
        importlib.import_module(name)
    verify_registrations()
    return names


def create_app_from_env_conf(
    workflow_path: Optional[str] = None,
    db_uri: Optional[str] = None,
    db_schema: Optional[str] = None,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    sql_echo: Optional[bool] = None,
) -> FastAPI:
    """Create Kavalai server application from environment configuration.

    Optional parameters can override the environment variables.

    The following environment variables are used:

    - KAVALAI_AGENT_WORKFLOW_PATH: Path to the workflow YAML file.
    - KAVALAI_AGENT_SETUP_MODULE: Optional module imported before the workflow
      is loaded, as a dotted name or a ``.py`` path. Registers the Python
      tools and named RAG services the workflow refers to.
    - KAVALAI_DB_URI: Database connection string.
    - KAVALAI_DB_SCHEMA: Database schema name.
    - KAVALAI_DB_POOL_SIZE: Database connection pool size (optional, default: 0).
    - KAVALAI_DB_MAX_OVERFLOW: Database connection pool max overflow (optional, default: 0).
    - KAVALAI_SQL_ECHO: Whether to log SQL queries (optional, default: False).
    - KAVALAI_PROVIDER_MODULES: Comma-separated modules to import before the
      workflow loads, so their backend registrations exist (optional).
    - KAVALAI_DEFAULT_LLM_MODEL: Model used when the workflow and its nodes
      name none, passed to the engine as ``default_llm_model`` (optional).
    - KAVALAI_LLM_TEMPERATURE, KAVALAI_LLM_TOP_P, KAVALAI_LLM_REASONING_EFFORT,
      KAVALAI_LLM_SERVICE_TIER, KAVALAI_LLM_TIMEOUT_SECONDS,
      KAVALAI_LLM_STREAM_TIMEOUT_SECONDS: fleet-wide defaults for every model
      call, passed to the engine as ``default_llm_parameters`` (optional).
    - KAVALAI_EMBEDDING_NORMALIZER_YAML: Normalizer installed as the default
      before the workflow loads (optional).
    - KAVALAI_RAG_MODEL: Embedding model of the ``default`` RAG service. Setting
      it registers that service without a setup module, over the index at
      KAVALAI_RAG_URI (required alongside it) in KAVALAI_RAG_SCHEMA (optional),
      with the normalizer above.
    - KAVALAI_AGENT_PUBLIC_EVENTS: Serve streams through
      :func:`public_events` (optional, default: false).
    - KAVALAI_AGENT_RUN_TIMEOUT_SECONDS: Seconds after which a run is
      cancelled and recorded as failed, passed to the engine as
      ``run_timeout`` (optional, default: no limit).

    Args:
        workflow_path: Path to the workflow YAML file.
        db_uri: Database connection string.
        db_schema: Database schema name.
        pool_size: Database connection pool size.
        max_overflow: Database connection pool max overflow.
        sql_echo: Whether to log SQL queries.

    Returns:
        A FastAPI application instance.
    """
    if workflow_path is None:
        workflow_path = env.str("KAVALAI_AGENT_WORKFLOW_PATH")

    # Custom backends have to be registered before the workflow is loaded: the
    # graph validates its model names at load time, and a `rag_query` node
    # resolves its service the same way. The operator names the modules; the
    # library discovers nothing on its own.
    load_provider_modules(env.str("KAVALAI_PROVIDER_MODULES", ""))

    if db_uri is None:
        db_uri = env("KAVALAI_DB_URI")
    if db_schema is None:
        db_schema = env("KAVALAI_DB_SCHEMA", "public")
    if pool_size is None:
        pool_size = env.int("KAVALAI_DB_POOL_SIZE", 0)
    if max_overflow is None:
        max_overflow = env.int("KAVALAI_DB_MAX_OVERFLOW", 0)
    if sql_echo is None:
        sql_echo = env.bool("KAVALAI_SQL_ECHO", False)

    masked_uri = mask_db_uri(db_uri)

    logger.info(f"Database URI: {masked_uri}")
    logger.info(f"Database Schema: {db_schema}")
    logger.info(f"Database Pool Size: {pool_size}")
    logger.info(f"Database Max Overflow: {max_overflow}")
    logger.info(f"SQL Echo: {sql_echo}")

    auth_user = env.str("KAVALAI_AGENT_BASIC_AUTH_USER", "")
    auth_password = env.str("KAVALAI_AGENT_BASIC_AUTH_PASSWORD", "")

    if auth_user or auth_password:
        logger.info(f"Basic Auth configured for user: {auth_user}")
        if auth_password:
            logger.info("Basic Auth password: ***")
    else:
        logger.warning(
            "Basic Auth is NOT configured: every endpoint on this server is "
            "public, including GET /workflow, which returns the full workflow "
            "definition (prompts, tool servers and their configuration). Set "
            "KAVALAI_AGENT_BASIC_AUTH_USER and KAVALAI_AGENT_BASIC_AUTH_PASSWORD "
            "before exposing this to a network you do not control."
        )

    session_provider = db_manager.get_sessionmaker(
        uri=db_uri,
        schema=db_schema,
        echo=sql_echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )

    agent_service = AgentService(session_provider)
    task_logger = PostgresTaskLogger(agent_service)

    # Imported before the engine is built. A workflow may name a RAG service or
    # a Python tool that only a module can register — and the engine resolves a
    # named RAG service *eagerly*, so without this it cannot even be
    # constructed. Same job as a suite's ``setup:`` key.
    setup_module = env.str("KAVALAI_AGENT_SETUP_MODULE", "")
    if setup_module:
        logger.info(f"Importing agent setup module {setup_module}.")
        _import_setup_module(setup_module)

    normalizer = apply_normalizer_from_env()
    if normalizer is not None:
        logger.info("Default embedding normalizer loaded from the environment.")

    # One index needs no setup module: KAVALAI_RAG_MODEL registers ``default``
    # over KAVALAI_RAG_URI. Both are stated explicitly — the RAG index is not
    # assumed to live in the agent database.
    rag_model = env.str("KAVALAI_RAG_MODEL", "") or None
    if rag_model:
        rag_uri = env.str("KAVALAI_RAG_URI")
        rag_schema = env.str("KAVALAI_RAG_SCHEMA", "") or None
        register_rag_service(
            "default",
            rag_service_from_uri,
            replace=True,
            uri=rag_uri,
            model=rag_model,
            schema=rag_schema,
            normalizer=normalizer,
        )
        logger.info(
            f"Default RAG service: {mask_db_uri(rag_uri)}"
            f" (schema {rag_schema or 'default'}, model {rag_model})"
        )

    default_llm_model = env.str("KAVALAI_DEFAULT_LLM_MODEL", "") or None
    default_llm_parameters = llm_parameters_from_env()
    logger.info(f"Default LLM model: {default_llm_model or '(none)'}")
    logger.info(f"Default LLM parameters: {default_llm_parameters or '(none)'}")

    run_timeout_setting = env.str("KAVALAI_AGENT_RUN_TIMEOUT_SECONDS", "")
    run_timeout = float(run_timeout_setting) if run_timeout_setting else None
    public = env.bool("KAVALAI_AGENT_PUBLIC_EVENTS", False)
    logger.info(f"Run timeout: {f'{run_timeout:g} s' if run_timeout else '(none)'}")
    logger.info(f"Public event stream: {'on' if public else 'off'}")

    logger.info(f"Loading workflow from {workflow_path}.")
    engine = WorkflowEngine.from_yaml_path(
        workflow_path,
        agent_service=agent_service,
        task_logger=task_logger,
        default_llm_model=default_llm_model,
        default_llm_parameters=default_llm_parameters,
        run_timeout=run_timeout,
    )

    return create_agent_app(
        engine=engine,
        session_provider=session_provider,
        event_filter=public_events if public else None,
    )


def _import_setup_module(reference: str) -> None:
    """Import a setup module, by dotted name or by file path.

    A path is accepted because an example (or a customer's workflow directory)
    is usually not an installed package: ``myapp/agent_setup.py`` is
    how you would refer to it from a shell, so it is how you refer to it here.
    The file's own directory goes on ``sys.path`` first, so the module can
    import its siblings.
    """
    path = Path(reference)
    if path.suffix != ".py":
        importlib.import_module(reference)
        return
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"KAVALAI_AGENT_SETUP_MODULE not found: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        f"kavalai_agent_setup_{path.stem}", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def run_agent_server():
    """Start the Kaval.AI agent server using environment configuration."""
    app = create_app_from_env_conf()
    logger.info(f"Starting agent <{app.state.engine.graph.name}>.")
    uvicorn.run(
        app,
        host=env.str("KAVALAI_AGENT_HOST", "0.0.0.0"),
        port=env.int("KAVALAI_AGENT_PORT", 10000),
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    run_agent_server()
