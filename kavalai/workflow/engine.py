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
import importlib
import json
import os
import time
from typing import Any, AsyncGenerator, Callable, Optional, Type, Union
from uuid import UUID, uuid4

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError

from kavalai.schema_parser import SchemaParser
from kavalai.run_context import RunContext
from kavalai.utils import to_plain
from kavalai.agent import Agent
from kavalai.workflow import clients as client_factory_module
from kavalai.workflow.expressions import evaluate_bool, evaluate_value
from kavalai.workflow.models import (
    AgentNode,
    EndNode,
    FunctionNode,
    IfNode,
    LLMNode,
    Node,
    ParallelNode,
    RagQueryNode,
    SwitchNode,
    WorkflowException,
    WorkflowGraph,
    WorkflowStreamEvent,
)
from kavalai.agent_service import AgentService
from kavalai.workflow.state import WorkflowState
from kavalai.workflow.tasklog.base import TaskLogger, TokenAccumulator
from kavalai.functionkernel import FunctionKernel, pythontool
from kavalai.llm_clients.base_client import BaseLlmClient, ChatHistory, ChatMessage
from kavalai.llm_clients.common import safe_parse_json
from kavalai.llm_clients.streamer import StreamContent

ClientFactory = Callable[..., BaseLlmClient]

#: Name a ``rag_query`` node resolves to when neither it nor the workflow says
#: otherwise. It matches ``BaseRagService``'s own default collection and source
#: identifiers, so the simple case never has to name anything.
DEFAULT_RAG_SERVICE = "default"

DEFAULT_MAX_NODE_VISITS = 1000


class _VisitBudget:
    """Shared node-visit counter for one run.

    A run may walk several paths at once (``parallel`` branches), so the loop
    guard has to be counted per run rather than per walk — otherwise N branches
    each get the full budget.
    """

    __slots__ = ("limit", "used")

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise WorkflowException(
                f"Exceeded max node visits ({self.limit}); "
                "the workflow may contain an infinite loop."
            )


def make_prompt(prompt: str, input_data: dict) -> str:
    """Combine a rendered prompt with resolved input data into a system message."""
    pieces = [prompt]
    if input_data:
        pieces.append("INPUT DATA:")
        for key, value in input_data.items():
            if isinstance(value, BaseModel):
                value = value.model_dump_json()
            pieces.append(f"{key}:{value}")
    return "\n".join(pieces)


class WorkflowEngine:
    """Executes a v2 :class:`WorkflowGraph` as a DAG / state machine.

    The engine walks the graph from the start node, following transitions and
    evaluating branch nodes, until it reaches an end node. Each node's result
    is stored in the run context; per-node debug data flows to ``task_logger``.

    Parameters
    ==========
    graph: WorkflowGraph
        The parsed workflow definition.
    agent_service: Optional[AgentService]
        Persistence for agents/sessions/runs/chat history. ``None`` runs the
        workflow without any persistence (no chat memory across turns).
    task_logger: Optional[TaskLogger]
        Backend for per-node debug data and model statistics.
    client_factory: Optional[ClientFactory]
        Factory ``(model, parameters, stats_receiver) -> BaseLlmClient`` used to
        build LLM clients. Defaults to the provider factory; inject a fake for
        offline testing.
    rag_services: Optional[BaseRagService | dict[str, BaseRagService]]
        Services available to ``rag_query`` nodes. A bare service is registered
        under ``"default"``, which is what a node resolves to when neither it
        nor the workflow names one. Anything not passed here is looked up among
        the services registered with
        :func:`~kavalai.register_rag_service`, so a workflow served by
        ``python -m kavalai.server`` --- where nobody constructs the engine ---
        still works.
    max_node_visits: int
        Safety cap on total node executions to guard against infinite loops.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        *,
        agent_service: Optional[AgentService] = None,
        task_logger: Optional[TaskLogger] = None,
        client_factory: Optional[ClientFactory] = None,
        data_models: Optional[dict[str, type[BaseModel]]] = None,
        rag_services: Optional[Union[Any, dict[str, Any]]] = None,
        max_node_visits: int = DEFAULT_MAX_NODE_VISITS,
    ):
        self.graph = graph
        self.agent_service = agent_service
        self.task_logger = task_logger
        self.client_factory = client_factory or client_factory_module.make_client
        self.max_node_visits = max_node_visits

        # A single service is stored under "default", so the common case --- one
        # index, one collection --- names nothing at all in the YAML.
        if rag_services is None:
            self.rag_services: dict[str, Any] = {}
        elif isinstance(rag_services, dict):
            self.rag_services = dict(rag_services)
        else:
            self.rag_services = {DEFAULT_RAG_SERVICE: rag_services}

        # Data types are usually JSON-schema fragments compiled to Pydantic models
        # by the SchemaParser. ``data_models`` lets callers (e.g. the
        # WorkflowBuilder's ``data_model``) supply ready-made Pydantic models
        # directly; those names are used as-is and skip the parser.
        overrides = data_models or {}
        to_parse = {k: v for k, v in graph.data_types.items() if k not in overrides}
        self.parser = SchemaParser(to_parse)
        self.models = self.parser.parse_all()
        self.models.update(overrides)
        self.node_map = graph.node_map

        # Resolve every rag_query node's service name now, so a typo fails here
        # rather than the first time that branch is taken. Existence only ---
        # the service itself is built on first use.
        self._validate_rag_services()

        # Build the function kernel and register declared servers / tools, reusing
        # the v1 registration approach.
        self.kernel = FunctionKernel()
        for server in graph.rest_servers:
            self.kernel.register_rest_server(server)
        for server in graph.mcp_servers:
            self.kernel.register_mcp_server(server)
        for func_config in graph.python_functions:
            module_path, func_name = func_config.path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            if not getattr(func, "_is_kavalai_tool", False):
                func = pythontool(func)
            self.kernel.register_python_tool(func_config.name, func)

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_yaml(cls, yaml_string: str, **kwargs) -> "WorkflowEngine":
        """Build an engine from a YAML workflow definition string."""
        try:
            data = yaml.load(yaml_string, Loader=yaml.SafeLoader)  # nosec B506
            graph = WorkflowGraph(**data)
        except ValidationError as e:
            raise WorkflowException(f"Workflow validation failed: {e}") from e
        return cls(graph, **kwargs)

    @classmethod
    def from_yaml_path(cls, yaml_path: str, **kwargs) -> "WorkflowEngine":
        """Build an engine from a YAML workflow definition file."""
        with open(yaml_path, "r") as f:
            return cls.from_yaml(f.read(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "WorkflowEngine":
        """Build an engine from a parsed workflow definition dict."""
        try:
            graph = WorkflowGraph(**data)
        except ValidationError as e:
            raise WorkflowException(f"Workflow validation failed: {e}") from e
        return cls(graph, **kwargs)

    # ----------------------------------------------------------------- lifecycle
    async def connect(self) -> "WorkflowEngine":
        """Open the connections the workflow's tool servers need.

        Only MCP servers need this: they are separate processes (or HTTP
        endpoints) whose tool lists are discovered on connect. Calling it up
        front means a misconfigured server fails before any tokens are spent,
        and — more importantly — that an agent node is told about the MCP tools
        it has, instead of being handed an empty list because nothing has called
        one yet.

        Safe to call more than once; already-connected servers are skipped. The
        engine still connects lazily on first use if you never call this, so it
        is an optimisation and a fail-fast check rather than a requirement.

        Returns:
            The engine, so it can be used as ``engine = await
            WorkflowEngine.from_yaml_path(...).connect()``.
        """
        await self.kernel.connect_mcp_servers()
        return self

    async def aclose(self) -> None:
        """Release the tool servers this engine owns.

        Call once when the engine is discarded — from a FastAPI lifespan, or at
        the end of a script. Runs must not do this: the kernel is engine state,
        shared by every run, so closing it mid-flight would tear down MCP
        sessions other runs are still using.
        """
        await self.kernel.close()

    async def __aenter__(self) -> "WorkflowEngine":
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------- helpers
    def get_data_type(self, name: Optional[str]):
        if not name:
            return None
        return self.models.get(name)

    def _resolve_model(self, node_model: Optional[str]) -> str:
        model = (
            node_model
            or self.graph.llm_model
            or os.environ.get("KAVALAI_DEFAULT_LLM_MODEL")
        )
        if not model:
            raise WorkflowException(
                "No LLM model configured (set node.llm_model, graph.llm_model "
                "or KAVALAI_DEFAULT_LLM_MODEL)."
            )
        return model

    def _validate_rag_services(self) -> None:
        """Fail at load if a ``rag_query`` node names an unavailable service."""
        from kavalai.llm_clients import registry

        for node in self.graph.nodes:
            if not isinstance(node, RagQueryNode):
                continue
            name = node.service or self.graph.rag_service or DEFAULT_RAG_SERVICE
            if name in self.rag_services:
                continue
            if registry.rag_services.lookup(name) is not None:
                continue
            passed = sorted(self.rag_services) or "(none)"
            raise WorkflowException(
                f"Node '{node.name}' needs RAG service '{name}', which is "
                f"neither passed to the engine (rag_services={passed}) nor "
                f"registered ({registry.registered_rag_services()}). Pass it "
                "as rag_services=, or call register_rag_service() before the "
                "workflow is loaded."
            )

    def _resolve_rag_service(self, node_service: Optional[str]):
        """Find the RAG service a ``rag_query`` node should use.

        Mirrors :meth:`_resolve_model`: the node wins, then the workflow's
        ``rag_service``, then ``"default"``. Services passed to the engine take
        precedence over registered ones, so a test can hand in a fake without
        touching global state.
        """
        name = node_service or self.graph.rag_service or DEFAULT_RAG_SERVICE
        service = self.rag_services.get(name)
        if service is not None:
            return service

        from kavalai.llm_clients.registry import RegistryError, make_rag_service

        try:
            return make_rag_service(name)
        except RegistryError as error:
            passed = sorted(self.rag_services) or "(none)"
            raise WorkflowException(
                f"No RAG service '{name}'. Passed to the engine: {passed}. {error}"
            ) from error

    def _make_llm_client(
        self, node_model: Optional[str], llm_kwargs: dict, run_context: RunContext
    ) -> BaseLlmClient:
        model = self._resolve_model(node_model)
        merged = dict(self.graph.llm_kwargs)
        merged.update(llm_kwargs or {})
        parameters = client_factory_module.build_parameters(merged)
        # The accumulator belongs to the run, not the engine: one engine serves
        # many concurrent runs (see ``kavalai.server``), and a shared counter
        # would report each run's tokens against whichever run finished next.
        # It tallies the whole run and forwards each call to the task logger.
        return self.client_factory(model, parameters, run_context.token_stats)

    # --------------------------------------------------------------------- nodes
    def _scoped_event(self, node: Node, chunk: StreamContent) -> WorkflowStreamEvent:
        """Rename a client stream chunk to node scope.

        The main ``response`` stream takes the node's name; any other stream
        (e.g. Gemini ``thought``, agent ``instructions``/``step<N>``) is
        prefixed with it.
        """
        name = node.name if chunk.name == "response" else f"{node.name}_{chunk.name}"
        return WorkflowStreamEvent(type=chunk.type, name=name, value=chunk.value)

    @staticmethod
    def _parse_streamed_output(
        output_type: Optional[Type[BaseModel]], raw: Optional[str], *, raw_text: bool
    ):
        """Parse a completed stream's value into the node's output type.

        ``raw_text`` marks a delta-mode buffer of raw model text (safe-parsed
        before validation); otherwise ``raw`` is the streamer's already
        safe-parsed complete value.
        """
        if raw is None:
            return None
        if not output_type:
            return raw
        if raw_text:
            return output_type.model_validate(safe_parse_json(raw))
        return output_type.model_validate_json(raw)

    async def _run_llm_node(
        self, node: LLMNode, run_context: RunContext
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        input_data = await run_context.prepare_tool_inputs(node)
        rendered_prompt = await run_context.render_prompt(node.prompt)
        text = make_prompt(rendered_prompt, input_data)

        messages = [ChatMessage(role="system", content=text)]
        if node.use_history and self.agent_service and run_context.session_id:
            history = await self.agent_service.get_chat_history(run_context.session_id)
            for msg in history:
                messages.append(ChatMessage(role=msg.role, content=msg.content))

        client = self._make_llm_client(node.llm_model, node.llm_kwargs, run_context)
        output_type = self.get_data_type(node.output)

        start = time.perf_counter()
        streamer = await client.stream_chat_completions(
            chat_history=ChatHistory(messages=messages),
            response_model=output_type,
            stream_delta=node.stream_delta,
        )
        # Reviewer: It would be better if demuxing and buffer accumulations happened
        # in a dedicated class/module.
        # In delta mode the complete chunk carries no value, so accumulate the
        # raw deltas ourselves to parse the output from.
        buffer = ""
        response_value: Optional[str] = None
        async for chunk in streamer:
            if chunk.type == "restart":
                buffer = ""
                yield self._scoped_event(node, chunk)
                continue
            if chunk.name == "response":
                if chunk.type == "partial" and node.stream_delta:
                    buffer += chunk.value or ""
                elif chunk.type == "complete":
                    response_value = buffer if node.stream_delta else chunk.value
            if node.stream_output:
                yield self._scoped_event(node, chunk)
        duration = time.perf_counter() - start

        response = self._parse_streamed_output(
            output_type, response_value, raw_text=node.stream_delta
        )
        run_context.data[node.output] = response
        self._log_node(
            run_context,
            node,
            inputs=input_data,
            output=response,
            prompt=text,
            duration=duration,
        )

    async def _run_agent_node(
        self, node: AgentNode, run_context: RunContext
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        input_data = await run_context.prepare_tool_inputs(node)
        rendered_prompt = await run_context.render_prompt(node.prompt)
        client = self._make_llm_client(node.llm_model, node.llm_kwargs, run_context)
        output_type = self.get_data_type(node.output)

        agent = Agent(
            llm_client=client,
            kernel=self.kernel,
            run_context=run_context,
            # Passed through verbatim: absent means every tool, ``[]`` means
            # none, exactly as in the Python API.
            allowed_tools=node.allowed_tools,
        )
        start = time.perf_counter()
        result_value: Optional[str] = None
        async for chunk in agent.prompt_stream(
            prompt=rendered_prompt,
            response_model=output_type,
            max_steps=node.max_steps,
            stream_output=node.stream_output,
            stream_instructions=node.stream_instructions,
            stream_partials=node.stream_partials,
            stream_delta=node.stream_delta,
        ):
            if chunk.name == "response" and chunk.type == "complete":
                result_value = chunk.value
                if node.stream_output:
                    yield self._scoped_event(node, chunk)
            else:
                # The agent already gates its progress streams by the flags.
                yield self._scoped_event(node, chunk)
        duration = time.perf_counter() - start

        result = self._parse_streamed_output(output_type, result_value, raw_text=False)
        run_context.data[node.output] = result
        self._log_node(
            run_context,
            node,
            inputs=input_data,
            output=result,
            prompt=rendered_prompt,
            duration=duration,
        )

    async def _run_function_node(
        self, node: FunctionNode, run_context: RunContext
    ) -> None:
        inputs = await run_context.prepare_tool_inputs(node)
        output_type = self.get_data_type(node.output)

        call_kwargs: dict[str, Any] = {}
        if node.tool.startswith("rest://"):
            call_kwargs["method"] = node.method

        start = time.perf_counter()
        result = await self.kernel.call_tool(
            tool_uri=node.tool,
            arguments=inputs,
            output_type=output_type,
            **call_kwargs,
        )
        duration = time.perf_counter() - start

        run_context.data[node.output] = result
        self._log_node(
            run_context,
            node,
            inputs=inputs,
            output=result,
            duration=duration,
        )

    async def _run_rag_query_node(
        self, node: RagQueryNode, run_context: RunContext
    ) -> None:
        """Query a RAG service and store the hits in the run context.

        Read-only: ``query`` is the only method this reaches for.
        """
        service = self._resolve_rag_service(node.service)
        query = await run_context.render_prompt(node.query)
        collection = node.collection or self.graph.rag_collection

        start = time.perf_counter()
        hits = await service.query(
            text=query,
            top_k=node.top_k,
            collection_name=collection,
            source_ids=node.source_ids,
            keep_best=node.keep_best,
            include_content=True,
        )
        duration = time.perf_counter() - start

        # "results" keeps scores and metadata reachable for routing; "content"
        # is what a following llm node's prompt usually wants, without the
        # UUIDs and timestamps a serialised result list would carry into it.
        if node.store == "content":
            value = "\n\n".join(hit.content or "" for hit in hits)
        else:
            value = hits

        run_context.data[node.output] = value
        self._log_node(
            run_context,
            node,
            inputs={"query": query, "collection": collection},
            output=value,
            duration=duration,
        )

    def _log_node(
        self,
        run_context: RunContext,
        node: Node,
        *,
        inputs: Optional[dict],
        output: Any,
        prompt: Optional[str] = None,
        duration: float,
    ) -> None:
        if not self.task_logger:
            return
        self.task_logger.log_node(
            run_id=str(run_context.run_id) if run_context.run_id else None,
            session_id=str(run_context.session_id) if run_context.session_id else None,
            agent_id=str(run_context.agent_id) if run_context.agent_id else None,
            node_name=node.name,
            node_type=node.type,
            inputs=to_plain(inputs) if inputs else inputs,
            output=to_plain(output) if output is not None else None,
            prompt=prompt,
            duration=duration,
        )

    def _next_node(self, node: Node, run_context: RunContext) -> Optional[str]:
        """Return the name of the next node to execute, or None at an end node."""
        if isinstance(node, EndNode):
            return None
        if isinstance(node, IfNode):
            return (
                node.then
                if evaluate_bool(node.condition, run_context.data)
                else node.else_
            )
        if isinstance(node, SwitchNode):
            value = evaluate_value(node.expr, run_context.data)
            return node.cases.get(value, node.default)
        return node.next

    async def _execute_node(
        self,
        node: Node,
        run_context: RunContext,
        state: WorkflowState,
        budget: "_VisitBudget",
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        """Run a side-effecting node (branch nodes are pure routing)."""
        if isinstance(node, LLMNode):
            async for event in self._run_llm_node(node, run_context):
                yield event
        elif isinstance(node, AgentNode):
            async for event in self._run_agent_node(node, run_context):
                yield event
        elif isinstance(node, FunctionNode):
            await self._run_function_node(node, run_context)
        elif isinstance(node, RagQueryNode):
            await self._run_rag_query_node(node, run_context)
        elif isinstance(node, ParallelNode):
            async for event in self._run_parallel_node(
                node, run_context, state, budget
            ):
                yield event
        # start / if / switch / end nodes have no side effects here.

    # ------------------------------------------------------------------ parallel
    @staticmethod
    def _branch_context(parent: RunContext) -> RunContext:
        """A private context for one branch, seeded with the parent's data.

        The copy is shallow: branches read the same input objects but write
        their own keys, so nothing they produce is visible to a sibling until
        :meth:`_merge_branch_contexts` runs at the join.
        """
        return RunContext(
            agent_id=parent.agent_id,
            session_id=parent.session_id,
            run_id=parent.run_id,
            data=dict(parent.data),
            templates=parent.templates,
            agent_service=parent.agent_service,
            # Shared on purpose: branches are part of one run, so their model
            # calls belong in the same total.
            token_stats=parent.token_stats,
        )

    @staticmethod
    def _merge_branch_contexts(
        parent: RunContext, branch_contexts: list[RunContext]
    ) -> None:
        """Copy each branch's writes back into the parent context.

        Only entries a branch actually replaced are copied — everything else is
        still the identical object the branch was seeded with. The graph
        validator guarantees no two branches write the same key, so the merge
        order cannot matter.
        """
        for context in branch_contexts:
            for key, value in context.data.items():
                if key not in parent.data or parent.data[key] is not value:
                    parent.data[key] = value

    async def _run_parallel_node(
        self,
        node: ParallelNode,
        run_context: RunContext,
        state: WorkflowState,
        budget: "_VisitBudget",
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        """Walk every branch concurrently and rejoin at ``node.next``.

        Events from all branches are interleaved onto one stream as they are
        produced; each carries its own node name, so a client can tell them
        apart. Node traces are kept per branch and appended in branch order, so
        ``state.trace`` stays deterministic even though execution is not.
        """
        contexts = [self._branch_context(run_context) for _ in node.branches]
        traces: list[list[str]] = [[] for _ in node.branches]
        queue: asyncio.Queue = asyncio.Queue()
        limit = (
            asyncio.Semaphore(node.max_concurrency)
            if node.max_concurrency is not None
            else None
        )

        async def walk_branch(index: int, entry: str) -> None:
            try:
                if limit is not None:
                    await limit.acquire()
                try:
                    async for event in self._walk_from(
                        entry,
                        contexts[index],
                        state,
                        budget,
                        trace=traces[index],
                        stop_at=node.next,
                        record_state=False,
                    ):
                        queue.put_nowait(("event", event))
                finally:
                    if limit is not None:
                        limit.release()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # surfaced to the driver below
                queue.put_nowait(("error", exc))
                return
            queue.put_nowait(("done", None))

        tasks = [
            asyncio.create_task(walk_branch(i, entry), name=f"{node.name}:{entry}")
            for i, entry in enumerate(node.branches)
        ]
        logger.debug(
            f"Parallel node '{node.name}' fanning out to {len(tasks)} branch(es): "
            f"{', '.join(node.branches)}"
        )
        try:
            remaining = len(tasks)
            while remaining:
                kind, payload = await queue.get()
                if kind == "event":
                    yield payload
                elif kind == "done":
                    remaining -= 1
                else:
                    # One branch failed: stop the siblings before propagating,
                    # so a long-running branch cannot outlive the run.
                    await self._cancel_all(tasks)
                    raise payload
        finally:
            await self._cancel_all(tasks)

        for trace in traces:
            state.trace.extend(trace)
        self._merge_branch_contexts(run_context, contexts)

    @staticmethod
    async def _cancel_all(tasks: list[asyncio.Task]) -> None:
        """Cancel any still-running branch tasks and wait for them to unwind."""
        pending = [t for t in tasks if not t.done()]
        if not pending:
            return
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    # ----------------------------------------------------------------------- run
    async def run(
        self,
        input_data: dict,
        *,
        session_id: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> WorkflowState:
        """Execute the workflow for ``input_data`` and return the final state.

        Drains :meth:`run_stream` — the single execution path.
        """
        state = WorkflowState(workflow_name=self.graph.name)
        async for _ in self.run_stream(
            input_data, session_id=session_id, external_id=external_id, state=state
        ):
            pass
        return state

    async def run_stream(
        self,
        input_data: dict,
        *,
        session_id: Optional[str] = None,
        external_id: Optional[str] = None,
        state: Optional[WorkflowState] = None,
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        """Execute the workflow, yielding :class:`WorkflowStreamEvent` events.

        Lifecycle events (``workflow_started``, ``node_started`` /
        ``node_completed``, ``workflow_completed`` / ``workflow_failed``)
        frame the run; nodes with streaming enabled contribute ``partial`` /
        ``complete`` / ``restart`` content events in between.

        Closing the generator early (e.g. the SSE client disconnected) aborts
        the run; the abort is recorded on the run row, best-effort. On failure
        a ``workflow_failed`` event is yielded before the
        :class:`WorkflowException` is raised to the caller.

        Args:
            input_data: The workflow input.
            session_id: Optional session to continue.
            external_id: Optional caller-supplied session key.
            state: Optional :class:`WorkflowState` instance populated in
                place, so blocking callers can read the final state after
                draining the stream.
        """
        invocation_id = uuid4().hex[:8]
        # One aggregator per run, carried on the run context, so concurrent runs
        # on the same engine never see each other's tokens.
        token_stats = TokenAccumulator(self.task_logger)

        parsed_input = self.get_data_type("input")(**input_data)
        run_context = RunContext()
        run_context.token_stats = token_stats
        run_context.data["input"] = parsed_input
        run_context.templates = {t.name: t.value for t in self.graph.templates}

        if state is None:
            state = WorkflowState(workflow_name=self.graph.name)
        state.status = "running"
        state.input_data = to_plain(input_data)
        state.invocation_id = invocation_id

        # Bind the invocation id onto every log record emitted during the run —
        # the engine, the agent loop and the LLM clients — so an entire
        # invocation can be grepped out of the logs by its id.
        with logger.contextualize(invocation_id=invocation_id):
            logger.info(f"[{invocation_id}] Starting workflow '{self.graph.name}'")

            if self.agent_service:
                agent, session, run = await self.agent_service.initialize_workflow_run(
                    agent_name=self.graph.name,
                    agent_description=self.graph.description,
                    input_schema=self.graph.data_types.get("input"),
                    output_schema=self.graph.data_types.get(self.graph.output_type),
                    workflow=self.graph.model_dump(),
                    session_id=UUID(session_id) if session_id else None,
                    external_id=external_id,
                    input_data=to_plain(input_data),
                )
                run_context.agent_id = agent.id
                # Model calls are logged against the agent; the accumulator is
                # not shared with any other run, so setting this once is safe.
                token_stats.agent_id = str(agent.id)
                run_context.session_id = session.id
                run_context.run_id = run.id
                # Lets ``history:`` inputs resolve values from previous runs.
                run_context.agent_service = self.agent_service
                state.agent_id = str(agent.id)
                state.session_id = str(session.id)
                state.run_id = str(run.id)

                user_message = getattr(parsed_input, "user_message", str(input_data))
                await self.agent_service.add_chat_message(
                    agent_id=agent.id,
                    session_id=session.id,
                    run_id=run.id,
                    role="user",
                    content=user_message,
                )

            try:
                yield WorkflowStreamEvent(
                    type="workflow_started",
                    name=self.graph.name,
                    session_id=state.session_id,
                    run_id=state.run_id,
                )
                async for event in self._walk(run_context, state):
                    yield event
                state.token_usage = token_stats.summary()
                yield WorkflowStreamEvent(
                    type="workflow_completed",
                    name=self.graph.name,
                    session_id=state.session_id,
                    output_data=state.output_data,
                    token_usage=state.token_usage,
                )
            except (GeneratorExit, asyncio.CancelledError):
                # The consumer went away (client disconnect / task cancel):
                # abort the run and record it — no events may be yielded here,
                # and the recording is best-effort during teardown.
                state.status = "failed"
                state.error = "aborted: client disconnected"
                try:
                    await self._record_failure(run_context, state)
                except BaseException:
                    logger.warning(
                        f"[{invocation_id}] Could not record aborted run "
                        f"{run_context.run_id}"
                    )
                raise
            except WorkflowException as e:
                state.status = "failed"
                state.error = str(e)
                yield WorkflowStreamEvent(
                    type="workflow_failed",
                    name=self.graph.name,
                    session_id=state.session_id,
                    value=state.error,
                )
                raise
            except Exception as e:
                state.status = "failed"
                state.error = str(e)
                await self._record_failure(run_context, state)
                yield WorkflowStreamEvent(
                    type="workflow_failed",
                    name=self.graph.name,
                    session_id=state.session_id,
                    value=state.error,
                )
                raise WorkflowException(e) from e
            finally:
                # The kernel is engine state — tool servers outlive the run and
                # are released by :meth:`aclose`. Closing it here would tear
                # down MCP sessions that other runs are still using.
                # Record and report token usage regardless of success or failure.
                state.token_usage = token_stats.summary()
                if self.task_logger:
                    await self.task_logger.flush()
                self._log_token_usage(invocation_id, token_stats)

    def _log_token_usage(self, invocation_id: str, s: TokenAccumulator) -> None:
        """Log the aggregate model token usage for the run."""
        logger.info(
            f"[{invocation_id}] Workflow '{self.graph.name}' token usage: "
            f"{s.model_calls} model call(s), {s.total_tokens} tokens "
            f"(prompt={s.prompt_tokens}, completion={s.completion_tokens})"
        )

    async def _walk(
        self, run_context: RunContext, state: WorkflowState
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        """Walk the whole graph, from the start node to an end node."""
        async for event in self._walk_from(
            self.graph.start,
            run_context,
            state,
            _VisitBudget(self.max_node_visits),
            trace=state.trace,
        ):
            yield event

    async def _walk_from(
        self,
        start: str,
        run_context: RunContext,
        state: WorkflowState,
        budget: "_VisitBudget",
        *,
        trace: list[str],
        stop_at: Optional[str] = None,
        record_state: bool = True,
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        """Walk the graph from ``start`` until an end node — or ``stop_at``.

        The main path walks with ``stop_at=None`` and ``record_state=True``;
        a ``parallel`` branch walks its own subgraph with ``stop_at`` set to the
        join node and ``record_state`` off, so concurrent branches do not
        overwrite each other's view of ``state.current_node`` / ``state.data``.

        Args:
            start: Node to begin at.
            run_context: Context the walked nodes read and write.
            state: Run state; lifecycle fields are only touched when
                ``record_state`` is set.
            budget: Run-wide node-visit guard, shared by every concurrent walk.
            trace: List each executed node name is appended to.
            stop_at: Node name to stop *before* (the join of a parallel node).
            record_state: Whether to publish progress onto ``state``.
        """
        current: Optional[str] = start

        while current is not None:
            if current == stop_at:
                return
            node = self.node_map[current]
            budget.spend()

            if record_state:
                state.current_node = node.name
            yield WorkflowStreamEvent(type="node_started", name=node.name)
            async for event in self._execute_node(node, run_context, state, budget):
                yield event
            trace.append(node.name)
            if record_state:
                state.data = to_plain(run_context.data)
            yield WorkflowStreamEvent(type="node_completed", name=node.name)

            if isinstance(node, EndNode):
                if not record_state:
                    # The graph validator rejects end nodes inside a parallel
                    # branch; this is the belt-and-braces version.
                    raise WorkflowException(
                        f"End node '{node.name}' was reached inside a parallel "
                        "branch; only the main path may end a run."
                    )
                await self._finish(node, run_context, state)
                return

            current = self._next_node(node, run_context)

        # A non-end node with no outgoing transition (switch with no default match).
        raise WorkflowException(
            f"Workflow halted at node '{node.name}' with no next node "
            "and without reaching an end node."
        )

    async def _finish(
        self, node: EndNode, run_context: RunContext, state: WorkflowState
    ) -> None:
        output_value = run_context.data.get(node.output)
        output_data = to_plain(output_value) if output_value is not None else None
        state.output_data = output_data
        state.status = "completed"

        if self.agent_service and run_context.run_id:
            await self.agent_service.update_run(
                run_context.run_id,
                output_data=output_data,
                context=to_plain(run_context.data),
            )
            # Chat-shaped workflows answer in `agent_response`; for any other
            # output type record the data itself, so the chat history is never
            # blank (mirrors the `user_message` fallback on the input side).
            agent_response = getattr(output_value, "agent_response", None)
            if agent_response is None:
                agent_response = (
                    json.dumps(output_data) if output_data is not None else ""
                )
            await self.agent_service.add_chat_message(
                agent_id=run_context.agent_id,
                session_id=run_context.session_id,
                run_id=run_context.run_id,
                role="assistant",
                content=agent_response,
            )
        logger.info(
            f"[{state.invocation_id}] Workflow '{self.graph.name}' completed "
            f"(session={state.session_id})"
        )

    async def _record_failure(
        self, run_context: RunContext, state: WorkflowState
    ) -> None:
        """Persist a failed run's error and partial data so it shows up in the
        backoffice; best-effort, since the failure may be the database itself."""
        if not (self.agent_service and run_context.run_id):
            return
        try:
            await self.agent_service.update_run(
                run_context.run_id,
                context={
                    "status": state.status,
                    "error": state.error,
                    "data": state.data,
                },
            )
        except Exception:
            logger.warning(
                f"[{state.invocation_id}] Could not persist failure state "
                f"for run {run_context.run_id}"
            )
