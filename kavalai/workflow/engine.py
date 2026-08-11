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
import os
import time
from typing import Any, AsyncGenerator, Callable, Optional, Type
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

DEFAULT_MAX_NODE_VISITS = 1000


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
        max_node_visits: int = DEFAULT_MAX_NODE_VISITS,
    ):
        self.graph = graph
        self.agent_service = agent_service
        self.task_logger = task_logger
        self.client_factory = client_factory or client_factory_module.make_client
        self.max_node_visits = max_node_visits
        # Per-run token aggregator; recreated for each run() so totals don't leak.
        self._token_stats = TokenAccumulator(task_logger)

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

    def _make_llm_client(
        self, node_model: Optional[str], llm_kwargs: dict, agent_id: Optional[str]
    ) -> BaseLlmClient:
        model = self._resolve_model(node_model)
        merged = dict(self.graph.llm_kwargs)
        merged.update(llm_kwargs or {})
        parameters = client_factory_module.build_parameters(merged)
        # The accumulator tallies tokens for the whole run and forwards each call
        # to the task logger (when configured).
        self._token_stats.agent_id = agent_id
        return self.client_factory(model, parameters, self._token_stats)

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

        agent_id = str(run_context.agent_id) if run_context.agent_id else None
        client = self._make_llm_client(node.llm_model, node.llm_kwargs, agent_id)
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
        agent_id = str(run_context.agent_id) if run_context.agent_id else None
        client = self._make_llm_client(node.llm_model, node.llm_kwargs, agent_id)
        output_type = self.get_data_type(node.output)

        agent = Agent(llm_client=client, kernel=self.kernel, run_context=run_context)
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
        self, node: Node, run_context: RunContext
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
        # start / if / switch / end nodes have no side effects here.

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
        # Fresh token aggregator so totals never leak between runs.
        self._token_stats = TokenAccumulator(self.task_logger)

        parsed_input = self.get_data_type("input")(**input_data)
        run_context = RunContext()
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
                state.token_usage = self._token_stats.summary()
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
                await self.kernel.close()
                # Record and report token usage regardless of success or failure.
                state.token_usage = self._token_stats.summary()
                if self.task_logger:
                    await self.task_logger.flush()
                self._log_token_usage(invocation_id)

    def _log_token_usage(self, invocation_id: str) -> None:
        """Log the aggregate model token usage for the run."""
        s = self._token_stats
        logger.info(
            f"[{invocation_id}] Workflow '{self.graph.name}' token usage: "
            f"{s.model_calls} model call(s), {s.total_tokens} tokens "
            f"(prompt={s.prompt_tokens}, completion={s.completion_tokens})"
        )

    async def _walk(
        self, run_context: RunContext, state: WorkflowState
    ) -> AsyncGenerator[WorkflowStreamEvent, None]:
        current: Optional[str] = self.graph.start
        visits = 0

        while current is not None:
            node = self.node_map[current]
            visits += 1
            if visits > self.max_node_visits:
                raise WorkflowException(
                    f"Exceeded max node visits ({self.max_node_visits}); "
                    "the workflow may contain an infinite loop."
                )

            state.current_node = node.name
            yield WorkflowStreamEvent(type="node_started", name=node.name)
            async for event in self._execute_node(node, run_context):
                yield event
            state.trace.append(node.name)
            state.data = to_plain(run_context.data)
            yield WorkflowStreamEvent(type="node_completed", name=node.name)

            if isinstance(node, EndNode):
                await self._finish(node, run_context, state)
                return

            current = self._next_node(node, run_context)

        # A non-end node with no outgoing transition (switch with no default match).
        raise WorkflowException(
            f"Workflow halted at node '{state.current_node}' with no next node "
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
            agent_response = getattr(output_value, "agent_response", "")
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
