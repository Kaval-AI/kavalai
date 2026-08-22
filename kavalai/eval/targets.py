"""What a suite runs against, and the single record every evaluator reads.

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

import importlib
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from kavalai.eval.models import Case, TargetSpec
from kavalai.eval.trajectory import Trajectory
from kavalai.utils import to_plain
from kavalai.workflow.engine import WorkflowEngine
from kavalai.workflow.tasklog.memory import MemoryTaskLogger, TeeTaskLogger


class ModelCallRecord(BaseModel):
    model: Optional[str] = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0


class RunRecord(BaseModel):
    """Everything an evaluator may look at, from one place.

    The single seam of the design: every evaluator is written against this, so
    none of them knows or cares how the run was executed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = None
    status: str = "completed"
    error: Optional[str] = None
    duration_seconds: float = 0.0
    trajectory: Trajectory = Field(default_factory=Trajectory)
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    #: The conversation, for multi-turn persona cases. One entry per turn.
    chat: list[dict] = Field(default_factory=list)
    external_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    #: Whatever the target's ``sandbox`` hook returned — the stubbed world this
    #: case ran against. Domain evaluators read the side effects from here.
    sandbox: Any = None
    #: Free-form extras a target or runner attached. Persona runs put the
    #: simulated user's own "did I get what I came for" verdict here.
    meta: dict = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.model_calls)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.error is None

    def output_text(self) -> str:
        """The output as text, however it is shaped.

        A chat-shaped workflow answers in ``agent_response``; anything else is
        rendered whole, so a ``contains`` assertion always has something to
        look at rather than silently matching nothing.
        """
        value = self.output
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("agent_response", "answer", "text", "body"):
                if isinstance(value.get(key), str):
                    return value[key]
        return str(to_plain(value))


class Target(Protocol):
    """Runs one case and reports what happened."""

    #: Whether runs against this target carry a trajectory. Trajectory
    #: evaluators refuse to score when it is False, rather than passing blind.
    observes_trajectory: bool

    async def setup(self) -> None: ...

    async def run(self, case: Case, external_id: Optional[str] = None) -> RunRecord: ...

    async def aclose(self) -> None: ...

    def describe(self) -> dict: ...


def import_object(reference: str) -> Any:
    """Import ``package.module:attribute`` (or ``package.module.attribute``)."""
    if ":" in reference:
        module_name, _, attr = reference.partition(":")
    else:
        module_name, _, attr = reference.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class _BaseTarget:
    """Shared sandbox handling.

    Side effects need a sandbox story per case, and it has to be opt-out: a
    workflow does not know it is being evaluated, so nothing stops it writing
    to whatever its tools point at. The hook runs before *every* case and every
    repeat — reset between repeats too, or the second run sees the first one's
    rows.
    """

    def __init__(self, sandbox: Optional[str] = None):
        self._sandbox_ref = sandbox
        self._sandbox_factory: Optional[Callable[[], Any]] = None

    def _load_sandbox(self) -> None:
        if self._sandbox_ref and self._sandbox_factory is None:
            self._sandbox_factory = import_object(self._sandbox_ref)

    async def _new_sandbox(self) -> Any:
        if self._sandbox_factory is None:
            return None
        return await _maybe_await(self._sandbox_factory())


class EngineTarget(_BaseTarget):
    """Runs the workflow in this process, with its full trajectory.

    The default and the one that matters. One engine is built and connected
    once for the whole experiment; each case gets its own
    :class:`~kavalai.workflow.tasklog.MemoryTaskLogger`, which is what makes
    concurrent cases safe and needs no database at all.
    """

    observes_trajectory = True

    def __init__(
        self,
        workflow: Union[str, Path],
        *,
        sandbox: Optional[str] = None,
        agent_service: Any = None,
        rag_services: Any = None,
        task_logger: Any = None,
        **engine_kwargs: Any,
    ):
        super().__init__(sandbox)
        self.workflow_path = Path(workflow)
        self.agent_service = agent_service
        #: Written *in addition to* the per-case memory logger. Without it,
        #: asking for a private trajectory would quietly switch off the
        #: database recording the backoffice reads.
        self.task_logger = task_logger
        self.rag_services = rag_services
        self.engine_kwargs = engine_kwargs
        self.engine: Optional[WorkflowEngine] = None

    async def setup(self) -> None:
        self._load_sandbox()
        kwargs = dict(self.engine_kwargs)
        if self.agent_service is not None:
            kwargs["agent_service"] = self.agent_service
        if self.rag_services is not None:
            kwargs["rag_services"] = self.rag_services
        self.engine = WorkflowEngine.from_yaml_path(str(self.workflow_path), **kwargs)
        # Kernel lifetime is the experiment, not the case: MCP sessions and
        # tool servers must outlive a single run.
        await self.engine.connect()

    async def aclose(self) -> None:
        if self.engine is not None:
            await self.engine.aclose()
            self.engine = None

    async def run(self, case: Case, external_id: Optional[str] = None) -> RunRecord:
        assert self.engine is not None, "call setup() before run()"
        sandbox = await self._new_sandbox()
        task_logger = MemoryTaskLogger()
        run_logger = (
            TeeTaskLogger(task_logger, self.task_logger)
            if self.task_logger is not None
            else task_logger
        )
        start = time.perf_counter()
        try:
            state = await self.engine.run(
                case.inputs, external_id=external_id, task_logger=run_logger
            )
            status, error = state.status, state.error
            output = state.output_data
            session_id, run_id = state.session_id, state.run_id
        except Exception as exc:
            # A case that blows up is an error, not a graded failure. Keeping
            # them apart is what stops "the harness broke" reading as "the
            # workflow is wrong".
            logger.warning(f"Case '{case.name}' raised: {exc}")
            status, error, output = "failed", str(exc), None
            session_id = run_id = None
        duration = time.perf_counter() - start
        await run_logger.flush()

        return RunRecord(
            output=output,
            status=status,
            error=error,
            duration_seconds=duration,
            trajectory=Trajectory(records=task_logger.records),
            model_calls=[
                ModelCallRecord(
                    model=s.model,
                    total_tokens=s.total_tokens or 0,
                    prompt_tokens=s.prompt_tokens or 0,
                    completion_tokens=s.completion_tokens or 0,
                    duration_seconds=s.duration_seconds or 0.0,
                )
                for s in task_logger.model_calls
            ],
            external_id=external_id,
            session_id=session_id,
            run_id=run_id,
            sandbox=sandbox,
        )

    def describe(self) -> dict:
        return {"kind": "engine", "workflow": str(self.workflow_path)}


class RestTarget(_BaseTarget):
    """Drives a deployed agent server over ``POST /run_agent``.

    The pre-deploy acceptance target: it exercises the artefact you are about
    to promote, with its real tools, network and secrets.

    **Output-only.** The runner does not read the remote agent's database, so
    there is no trajectory — and trajectory evaluators raise rather than
    quietly scoring a pass. The report says so in its header.
    """

    observes_trajectory = False

    def __init__(
        self,
        base_url: str,
        *,
        path: str = "/run_agent",
        auth: Optional[tuple[str, str]] = None,
        timeout_seconds: float = 120.0,
        sandbox: Optional[str] = None,
    ):
        super().__init__(sandbox)
        self.base_url = base_url.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.auth = auth
        self.timeout_seconds = timeout_seconds
        self._client: Any = None

    async def setup(self) -> None:
        import httpx

        self._load_sandbox()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            auth=self.auth,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run(self, case: Case, external_id: Optional[str] = None) -> RunRecord:
        assert self._client is not None, "call setup() before run()"
        sandbox = await self._new_sandbox()
        start = time.perf_counter()
        status, error, output, session_id = "completed", None, None, None
        try:
            response = await self._client.post(
                self.path, json={"data": case.inputs, "external_id": external_id}
            )
            response.raise_for_status()
            payload = response.json()
            output = payload.get("data")
            session_id = payload.get("session_id")
        except Exception as exc:
            status, error = "failed", str(exc)
        return RunRecord(
            output=output,
            status=status,
            error=error,
            duration_seconds=time.perf_counter() - start,
            external_id=external_id,
            session_id=session_id,
            sandbox=sandbox,
        )

    def describe(self) -> dict:
        return {"kind": "rest", "base_url": self.base_url, "path": self.path}


class CallableTarget(_BaseTarget):
    """Anything that is not a Kaval.AI workflow.

    Give it ``async fn(inputs) -> output`` and you get the deterministic and
    judged evaluators; you lose the trajectory ones, which is the honest
    trade rather than a silent one.
    """

    observes_trajectory = False

    def __init__(
        self,
        function: Union[str, Callable[..., Any]],
        *,
        sandbox: Optional[str] = None,
    ):
        super().__init__(sandbox)
        self.function = (
            import_object(function) if isinstance(function, str) else function
        )

    async def setup(self) -> None:
        self._load_sandbox()

    async def aclose(self) -> None:
        return None

    async def run(self, case: Case, external_id: Optional[str] = None) -> RunRecord:
        sandbox = await self._new_sandbox()
        start = time.perf_counter()
        status, error, output = "completed", None, None
        try:
            output = await _maybe_await(self.function(case.inputs))
        except Exception as exc:
            status, error = "failed", str(exc)
        return RunRecord(
            output=to_plain(output),
            status=status,
            error=error,
            duration_seconds=time.perf_counter() - start,
            external_id=external_id,
            sandbox=sandbox,
        )

    def describe(self) -> dict:
        name = getattr(self.function, "__qualname__", str(self.function))
        return {"kind": "callable", "function": name}


def build_target(spec: TargetSpec, base_dir: Path, **overrides: Any) -> Target:
    """Build the target a suite file describes.

    ``overrides`` is how the CLI injects things the library must not read for
    itself — an ``agent_service`` built from environment variables, say.
    """
    if spec.kind == "engine":
        if not spec.workflow:
            raise ValueError("target kind 'engine' needs a 'workflow' path")
        workflow = Path(spec.workflow)
        if not workflow.is_absolute():
            workflow = base_dir / workflow
        return EngineTarget(workflow, sandbox=spec.sandbox, **overrides)
    if spec.kind == "rest":
        if not spec.base_url:
            raise ValueError("target kind 'rest' needs a 'base_url'")
        return RestTarget(
            spec.base_url,
            path=spec.path,
            auth=overrides.get("auth"),
            timeout_seconds=spec.timeout_seconds,
            sandbox=spec.sandbox,
        )
    if spec.kind == "callable":
        if not spec.function:
            raise ValueError("target kind 'callable' needs a 'function' reference")
        return CallableTarget(spec.function, sandbox=spec.sandbox)
    raise ValueError(
        f"Unknown target kind '{spec.kind}'; expected engine, rest or callable."
    )
