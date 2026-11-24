import asyncio
import json
import time
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)
import os

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from kavalai.run_context import RunContext
from kavalai.utils import to_plain
from kavalai.functionkernel import FunctionKernel, _is_tool_allowed
from kavalai.llm_clients.base_client import BaseLlmClient, ChatHistory, ChatMessage
from kavalai.llm_clients.common import safe_parse_json
from kavalai.llm_clients.streamer import StreamContent
from kavalai.step_schema import (
    ArgRef,
    build_step_schema,
    step_output_model,
    untyped_tools,
)
from jinja2 import Template


class ToolCall(BaseModel):
    """A tool call as the agent reads it from a step.

    This is the lenient reading of what the model wrote, not the schema the
    model is asked for (see :mod:`kavalai.step_schema`). Nothing about a
    call's arguments fails the step: a malformed value becomes that call's
    error result, which the model reads on its next step.

    Attributes:
        name: The tool's URI, for instance ``python://mypackage.mytool``.
        args: Literal argument values, as an object. A null value means the
            argument is not supplied. A JSON object in a string is accepted.
        args_json: Literal argument values as a JSON object in a string, the
            form asked of a tool whose arguments cannot be typed.
        refs: Arguments taken from elsewhere, each an :class:`ArgRef`. An
            object mapping the argument name to ``{"source", "key"}`` is
            accepted too.
        call_id: The name under which later steps refer to the result.
        literal_args: Literal argument values as a JSON object in a string.
            With the two fields below, the form of a call before arguments
            were typed; the model is no longer asked for it, and it is still
            read.
        planner_context_args: A JSON object mapping argument names to
            ``call_id``s: the earlier form of a ``"step"`` reference.
        input_args: A JSON object mapping argument names to context variable
            names: the earlier form of an ``"input"`` reference.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: Union[dict[str, Any], str, None] = None
    args_json: Optional[str] = None
    refs: Union[list[Any], dict[str, Any], None] = None
    call_id: Optional[str] = None
    literal_args: Union[str, dict[str, Any], None] = None
    planner_context_args: Union[str, dict[str, Any], None] = None
    input_args: Union[str, dict[str, Any], None] = None


class _ToolArgumentError(ValueError):
    """A tool call's arguments cannot be read or resolved."""


def _as_mapping(field_name: str, value: Union[str, dict[str, Any], None]) -> dict:
    """Read an argument field, which holds an object or a JSON object in a string."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise _ToolArgumentError(
            f"`{field_name}` is not valid JSON ({error.msg}, at character {error.pos})."
        ) from error
    if not isinstance(parsed, dict):
        raise _ToolArgumentError(f"`{field_name}` must hold a JSON object.")
    return parsed


def _references(tool_call: ToolCall) -> tuple[dict[str, Any], dict[str, Any]]:
    """The by-reference arguments of a call, in both forms it can take.

    Returns:
        ``(step_refs, input_refs)``, each mapping an argument name to the key
        its value is looked up under.
    """
    step_refs = dict(_as_mapping("planner_context_args", tool_call.planner_context_args))
    input_refs = dict(_as_mapping("input_args", tool_call.input_args))
    entries: list[Any]
    if isinstance(tool_call.refs, dict):
        entries = [
            {**spec, "arg": arg} if isinstance(spec, dict) else spec
            for arg, spec in tool_call.refs.items()
        ]
    else:
        entries = tool_call.refs or []
    for entry in entries:
        try:
            ref = ArgRef.model_validate(entry)
        except ValidationError as error:
            raise _ToolArgumentError(
                f"`refs` entry {entry!r} is malformed: each needs `arg`, `source` "
                '("step" or "input") and `key`.'
            ) from error
        (step_refs if ref.source == "step" else input_refs)[ref.arg] = ref.key
    return step_refs, input_refs


def _lookup(
    refs: Mapping[str, Any], values: Mapping[str, Any], what: str
) -> dict[str, Any]:
    """Resolve references against the values they name."""
    resolved = {}
    for arg, key in refs.items():
        if not isinstance(key, str) or key not in values:
            raise _ToolArgumentError(
                f"`{arg}` refers to the {what} {key!r}, which does not exist. "
                f"Available: {', '.join(sorted(values)) or 'none'}."
            )
        resolved[arg] = values[key]
    return resolved


def get_step_output_type(ResponseModel: Type[BaseModel]) -> Type[BaseModel]:
    """The model the agent reads a step with.

    Its ``tool_calls`` are the lenient :class:`ToolCall`. The schema sent to
    the model is built by :func:`kavalai.step_schema.build_step_schema`.
    """
    return step_output_model(ResponseModel, ToolCall)


class StepStreamDemuxer:
    """Routes one agent step's raw LLM stream into named sub-streams.

    Consumes the step's :class:`StreamContent` chunks (the step is always
    streamed in raw-delta mode) and returns the chunks to emit, gated by the
    stream flags:

    - the step's raw model output under ``step<N>`` (``stream_partials``),
    - the ``instructions`` field under ``instructions``, with a per-step
      ``complete`` (``stream_instructions``),
    - the ``output`` field under ``response`` as full safe-parsed JSON
      (``stream_output``),
    - pass-through of auxiliary provider streams (e.g. Gemini thoughts) and
      of ``restart`` chunks, which also reset the accumulated state.

    One instance handles exactly one step; the raw text accumulates in
    ``buffer`` and the completed step's JSON is exposed as ``step_json``.
    """

    def __init__(
        self,
        step_idx: int,
        *,
        stream_output: bool = False,
        stream_instructions: bool = False,
        stream_partials: bool = False,
        stream_delta: bool = False,
    ) -> None:
        self.step_idx = step_idx
        self.stream_output = stream_output
        self.stream_instructions = stream_instructions
        self.stream_partials = stream_partials
        self.stream_delta = stream_delta
        self.buffer = ""
        self.step_json: Optional[str] = None
        self._sent_instructions = ""
        self._sent_output_json: Optional[str] = None

    def on_chunk(self, chunk: StreamContent) -> list[StreamContent]:
        """Process one incoming chunk and return the chunks to emit."""
        if chunk.type == "restart":
            self.buffer = ""
            self._sent_instructions = ""
            self._sent_output_json = None
            return [chunk]
        if chunk.name != "response":
            # Auxiliary provider streams (e.g. Gemini thoughts).
            return [chunk] if self.stream_output else []
        if chunk.type == "partial":
            return self._on_partial(chunk)
        if chunk.type == "complete":
            self.step_json = self.buffer
        return []

    def step_completed(self, step_output: Any) -> list[StreamContent]:
        """Per-step closing chunks, once the parsed ``StepOutput`` is known."""
        events: list[StreamContent] = []
        if self.stream_instructions and step_output.instructions:
            events.append(
                StreamContent(
                    type="complete",
                    name="instructions",
                    value=step_output.instructions,
                )
            )
        if self.stream_partials:
            events.append(
                StreamContent(
                    type="complete",
                    name=f"step{self.step_idx}",
                    value=None if self.stream_delta else self.step_json,
                )
            )
        return events

    def _on_partial(self, chunk: StreamContent) -> list[StreamContent]:
        self.buffer += chunk.value or ""
        events: list[StreamContent] = []
        if self.stream_partials:
            events.append(
                StreamContent(
                    type="partial",
                    name=f"step{self.step_idx}",
                    value=chunk.value if self.stream_delta else self.buffer,
                )
            )
        if self.stream_instructions or self.stream_output:
            parsed = safe_parse_json(self.buffer)
            if isinstance(parsed, dict):
                events += self._demux_fields(parsed)
        return events

    def _demux_fields(self, parsed: dict) -> list[StreamContent]:
        """Route the fields of the partially parsed ``StepOutput`` to streams."""
        events: list[StreamContent] = []

        instructions = parsed.get("instructions")
        if (
            self.stream_instructions
            and isinstance(instructions, str)
            and instructions != self._sent_instructions
        ):
            if self.stream_delta and instructions.startswith(self._sent_instructions):
                value = instructions[len(self._sent_instructions) :]
            else:
                if self.stream_delta and self._sent_instructions:
                    # Non-monotonic change: tell the client to start over.
                    events.append(StreamContent(type="restart", name="instructions"))
                value = instructions
            if value:
                events.append(
                    StreamContent(type="partial", name="instructions", value=value)
                )
            self._sent_instructions = instructions

        # The output field streams under the main stream name as full JSON;
        # an output produced alongside tool calls is forwarded as-is and may
        # be superseded by the final complete.
        output_value = parsed.get("output")
        if self.stream_output and output_value is not None:
            out_json = (
                output_value
                if isinstance(output_value, str)
                else json.dumps(output_value)
            )
            if out_json != self._sent_output_json:
                events.append(
                    StreamContent(type="partial", name="response", value=out_json)
                )
                self._sent_output_json = out_json
        return events


def _default_template(filename: str) -> Template:
    with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
        return Template(f.read())


class Agent:
    def __init__(
        self,
        llm_client: BaseLlmClient,
        *,
        kernel: Optional[FunctionKernel] = None,
        run_context: Optional[RunContext] = None,
        prompt_template: Optional[Template] = None,
        step_template: Optional[Template] = None,
        allowed_tools: Optional[list[str]] = None,
        on_step: Optional[Callable[[dict], None]] = None,
        debug: bool = False,
    ) -> None:
        """Build an agent.

        Every step sends one conversation to the model: a system message
        rendered from ``prompt_template`` (the task, the context variables and
        the tool descriptions), then the chat ``history`` handed to
        :meth:`prompt_stream`, then a user message rendered from
        ``step_template`` (the request this invocation answers, the steps
        executed so far and the instruction to produce the next one). Both
        templates receive the same variables: ``prompt``, ``data``,
        ``tool_descriptions``, ``steps``, ``current_step``, ``max_steps``,
        ``last_step`` (whether this is the final step, on which tool calls are
        not run) and ``request``. The defaults are
        ``default_prompt_template.j2`` and ``default_step_template.j2``
        beside this module.

        ``allowed_tools`` restricts the agent to a subset of the kernel's
        tools, given as tool URIs (``python://web.crawl``, or ``rest://api.*``
        for a whole server). The tools outside it are neither described to the
        model nor callable. ``None`` (the default) allows every registered
        tool; an empty list allows none.

        The model is asked for a step whose tool calls are typed: one call
        model per allowed tool, its arguments held to that tool's input schema
        (:mod:`kavalai.step_schema`). A custom ``prompt_template`` has to
        describe that form; it receives ``json_argument_tools``, the URIs of
        the tools that take ``args_json`` instead of ``args``.

        ``on_step`` is an observer called once per completed reasoning step
        with that step's record (index, instructions, tool calls with their
        arguments, results and durations, and the step's output). It exists so
        an agent's internal tool calls can be recorded — the workflow engine
        turns them into task rows — without putting them on the public stream,
        which every SSE consumer sees. It is called synchronously and its
        exceptions are swallowed: a broken observer must never break a run.
        """
        self.debug = debug
        self.kernel = kernel
        self.allowed_tools = allowed_tools
        self.on_step = on_step
        if not run_context:
            run_context = RunContext()
        self.run_context = run_context
        self.llm_client = llm_client
        self.prompt_template = prompt_template or _default_template(
            "default_prompt_template.j2"
        )
        self.step_template = step_template or _default_template(
            "default_step_template.j2"
        )

    async def prompt_stream(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        max_steps: int = 10,
        *,
        stream_output: bool = False,
        stream_instructions: bool = False,
        stream_partials: bool = False,
        stream_delta: bool = False,
        history: Optional[Sequence[ChatMessage]] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        request: Optional[str] = None,
    ) -> AsyncGenerator[StreamContent, None]:
        """Run the agent loop, streaming progress as :class:`StreamContent`.

        The agent iterates up to ``max_steps`` times. On each step the LLM
        returns a ``StepOutput`` with optional ``tool_calls`` and an optional
        final ``output``. Tool calls are executed through the
        ``FunctionKernel`` and their results are fed back into the prompt so
        the model can reason over them on the next step. The loop stops once
        the model returns an ``output`` without requesting further tool calls,
        or when ``max_steps`` is reached.

        The final chunk is always ``complete``/``response`` carrying the final
        output (JSON for structured outputs, the plain string otherwise, or
        ``None`` when no output was produced). The flags gate the optional
        progress streams; their semantics (including the naming and the
        full-JSON structured output stream) are documented on
        :class:`kavalai.workflow.models.AgentNode`. Stream names here are
        unscoped (``response``, ``instructions``, ``step<N>``); the workflow
        engine prefixes them with the node name.

        Args:
            prompt: The task description for the agent.
            response_model: Optional Pydantic model describing the structured
                final output. When omitted, a plain string is produced.
            max_steps: Maximum number of reasoning/tool-calling iterations.
            stream_output: Stream the step's ``output`` field as it is written.
            stream_instructions: Stream each step's ``instructions`` field.
            stream_partials: Stream each step's raw model output.
            stream_delta: Emit deltas instead of full accumulated values on the
                ``instructions`` and ``step<N>`` streams.
            history: Earlier turns of the conversation, sent with every step
                between the system prompt and the step message. The current
                user message belongs in ``prompt``, not here — the model
                would read it twice.
            inputs: The context variables of this invocation: listed in the
                system prompt and the values an ``"input"`` reference of a
                tool call resolves against. Defaults to the run context's ``data``. The
                workflow engine passes the node's resolved ``inputs``, so an
                agent node sees what its author declared and not every value
                the run has produced so far.
            request: The request this invocation answers, stated in every
                step message. ``history`` ends on the turn before it, so
                without it a model reading the conversation answers that turn.
                The workflow engine passes the run's user message.
        """
        StepOutput = get_step_output_type(response_model or str)
        # The schema the model is asked for is stricter than the reading of
        # its reply: StepOutput accepts what a model writes when nothing holds
        # it to the schema.
        tool_models = (
            {
                uri: definition.input_model
                for uri, definition in (
                    await self.kernel.get_tool_definitions(self.allowed_tools)
                ).items()
            }
            if self.kernel
            else {}
        )
        step_schema = build_step_schema(response_model or str, tool_models)
        history_messages = list(history or [])
        variables = dict(inputs) if inputs is not None else self.run_context.data

        # Per-invocation working memory: tool call results keyed by call_id,
        # which a later call's "step" reference names. Created fresh for each
        # invocation (up to `max_steps`) and discarded afterwards, unlike
        # `self.run_context` which is passed in at construction.
        planner_context: dict[str, Any] = {}
        # Record of executed steps, rendered back into the prompt template.
        steps: list[dict] = []
        final_output: Optional[BaseModel] = None

        for step_idx in range(max_steps):
            template_vars = dict(
                prompt=prompt,
                data=variables,
                tool_descriptions=(
                    await self.kernel.get_tool_descriptions(self.allowed_tools)
                    if self.kernel
                    else ""
                ),
                steps=steps,
                current_step=step_idx,
                max_steps=max_steps,
                last_step=step_idx == max_steps - 1,
                request=request,
                json_argument_tools=untyped_tools(tool_models),
            )
            rendered_prompt = self.prompt_template.render(**template_vars)
            rendered_step = self.step_template.render(**template_vars)

            if self.debug:
                print(rendered_prompt)
                print(rendered_step)

            chat_history = ChatHistory(
                messages=[
                    ChatMessage(role="system", content=rendered_prompt),
                    *history_messages,
                    ChatMessage(role="user", content=rendered_step),
                ]
            )

            logger.info(f"Agent step {step_idx}/{max_steps}")
            # Always consume the step in raw-delta mode: the demuxer needs the
            # full raw text to safe-parse, and its buffer doubles as the value
            # to validate on completion.
            streamer = await self.llm_client.stream_chat_completions(
                chat_history=chat_history,
                response_model=step_schema,
                stream_delta=True,
            )
            demux = StepStreamDemuxer(
                step_idx,
                stream_output=stream_output,
                stream_instructions=stream_instructions,
                stream_partials=stream_partials,
                stream_delta=stream_delta,
            )
            async for chunk in streamer:
                for event in demux.on_chunk(chunk):
                    yield event

            if demux.step_json is None or not demux.step_json.strip():
                logger.warning("LLM returned no step output, stopping.")
                break
            step_output = StepOutput.model_validate(safe_parse_json(demux.step_json))

            for event in demux.step_completed(step_output):
                yield event

            # Ensure every tool call has a stable id for context lookups.
            for idx, tool_call in enumerate(step_output.tool_calls):
                if not tool_call.call_id:
                    tool_call.call_id = f"tool_call_{step_idx}_{idx}"

            step_record: dict[str, Any] = {
                "index": step_idx,
                "instructions": step_output.instructions,
                "tool_calls": [],
                "output": to_plain(step_output.output),
            }

            # The last step has no next step to read a result in: its tool calls
            # are not run, and the step message says so, so the model answers.
            if step_idx == max_steps - 1 and step_output.tool_calls:
                logger.warning(
                    f"Agent asked for {len(step_output.tool_calls)} tool call(s) on its last step; not run"
                )
            elif step_output.tool_calls and self.kernel:
                results = await asyncio.gather(
                    *[
                        self._call_tool(tc, planner_context, variables)
                        for tc in step_output.tool_calls
                    ]
                )
                for tool_call, args, result, duration in results:
                    planner_context[tool_call.call_id] = result
                    step_record["tool_calls"].append(
                        {
                            "name": tool_call.name,
                            "args": args,
                            "call_id": tool_call.call_id,
                            "output": to_plain(result),
                            "duration_seconds": duration,
                        }
                    )

            steps.append(step_record)
            self._publish_step(step_record)

            if step_output.output is not None:
                final_output = step_output.output
                # Stop once the model produced an answer without more tool calls.
                if not step_output.tool_calls:
                    break
        else:
            if final_output is None:
                logger.warning(
                    f"Agent used its {max_steps} steps without producing an output"
                )

        value = None
        if final_output is not None:
            value = (
                final_output
                if isinstance(final_output, str)
                else final_output.model_dump_json()
            )
        yield StreamContent(type="complete", name="response", value=value)

    async def prompt(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        max_steps: int = 10,
        *,
        history: Optional[Sequence[ChatMessage]] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        request: Optional[str] = None,
    ) -> str | BaseModel:
        """Run the agent loop and return the final output (blocking wrapper).

        Drains :meth:`prompt_stream` with all progress streams disabled.

        Args:
            prompt: The task description for the agent.
            response_model: Optional Pydantic model describing the structured
                final output. When omitted, a plain string is returned.
            max_steps: Maximum number of reasoning/tool-calling iterations.
            history: Earlier turns of the conversation, as on
                :meth:`prompt_stream`.
            inputs: The context variables, as on :meth:`prompt_stream`.
            request: The request this invocation answers, as on
                :meth:`prompt_stream`.

        Returns:
            The structured ``response_model`` instance, or a string when no
            ``response_model`` is provided. ``None`` if no output was produced.
        """
        final_value = None
        async for chunk in self.prompt_stream(
            prompt,
            response_model=response_model,
            max_steps=max_steps,
            history=history,
            inputs=inputs,
            request=request,
        ):
            if chunk.type == "complete" and chunk.name == "response":
                final_value = chunk.value

        if final_value is None:
            return None
        if response_model:
            return response_model.model_validate_json(final_value)
        return final_value

    def _resolve_args(
        self,
        tool_call: ToolCall,
        planner_context: dict[str, Any],
        variables: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Resolve a ToolCall's argument sources into a single argument dict.

        A literal value beats a ``"step"`` reference, which beats an
        ``"input"`` reference. ``"step"`` references resolve against the
        per-invocation ``planner_context`` (results of previous tool calls);
        ``"input"`` references against ``variables``, the context variables
        the prompt listed (``self.run_context.data`` when not given). A null
        in ``args`` means the argument is not supplied, which is how a model
        held to the typed schema writes it; ``literal_args`` keeps its nulls.

        Raises:
            _ToolArgumentError: An argument field is not a JSON object, a
                ``refs`` entry is malformed, or a reference names a tool
                result or a context variable that does not exist.
        """
        if variables is None:
            variables = self.run_context.data

        literals = {
            **_as_mapping("literal_args", tool_call.literal_args),
            **_as_mapping("args_json", tool_call.args_json),
            **{
                name: value
                for name, value in _as_mapping("args", tool_call.args).items()
                if value is not None
            },
        }
        step_refs, input_refs = _references(tool_call)
        return {
            **_lookup(input_refs, variables, "context variable"),
            **_lookup(step_refs, planner_context, "call_id"),
            **literals,
        }

    def _publish_step(self, step_record: dict) -> None:
        """Hand a completed step to the ``on_step`` observer, if there is one.

        Never raises: an observer that fails is a logging problem, not a
        reason to abandon the run.
        """
        if self.on_step is None:
            return
        try:
            self.on_step(step_record)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Agent on_step observer failed: {e}")

    async def _call_tool(
        self,
        tool_call: ToolCall,
        planner_context: dict[str, Any],
        variables: Mapping[str, Any],
    ) -> tuple[ToolCall, dict, Any, float]:
        """Resolve arguments and execute a single tool call via the kernel.

        Returns ``(tool_call, args, result, duration_seconds)``. Arguments
        that cannot be resolved and execution errors are captured and returned
        as the result so the model can self-correct. The duration is measured here rather than around the
        caller's ``asyncio.gather`` — that would time the slowest call in the
        batch and attribute it to all of them.
        """
        start = time.perf_counter()
        try:
            args = self._resolve_args(tool_call, planner_context, variables)
        except _ToolArgumentError as e:
            # Not run: a tool called with half its arguments answers a
            # question the model did not ask.
            logger.warning(f"Tool {tool_call.name} was not called: {e}")
            return tool_call, {}, f"Error: {e}", time.perf_counter() - start
        if not _is_tool_allowed(tool_call.name, self.allowed_tools):
            # The tool was never described to the model; tell it so it can
            # pick one it is actually allowed to call.
            logger.warning(f"Tool {tool_call.name} is not allowed for this agent")
            return (
                tool_call,
                args,
                f"Error: tool {tool_call.name} is not available.",
                time.perf_counter() - start,
            )
        logger.info(f"Calling tool {tool_call.name} with {args}")
        try:
            result = await self.kernel.call_tool(
                tool_uri=tool_call.name, arguments=args
            )
        except Exception as e:
            logger.error(f"Tool {tool_call.name} failed: {e}")
            result = f"Error: {e}"
        return tool_call, args, result, time.perf_counter() - start


if __name__ == "__main__":  # pragma: no cover - manual demo
    from kavalai.llm_clients.openai_client import OpenAIClient
    from kavalai.functionkernel import FunctionKernel, pythontool
    import datetime

    @pythontool
    def get_time():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    kernel = FunctionKernel()
    kernel.register_python_tool("get_time", get_time)

    llm_client = OpenAIClient("gpt-5.6-luna")
    agent = Agent(llm_client=llm_client, kernel=kernel, debug=True)
    result = asyncio.run(agent.prompt("Greet the user based on current time!"))
    print(result)
