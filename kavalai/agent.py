"""The planning agent: a loop of model steps and tool calls.

One invocation (:meth:`Agent.prompt` or :meth:`Agent.prompt_stream`) runs up
to ``max_steps`` steps. On each step the model returns one structured reply,
a *step*, with three fields:

- ``instructions``: what the step is meant to achieve,
- ``tool_calls``: the tools to run now,
- ``output``: the final answer, once the model has one.

The agent runs the step's tool calls together, writes their results into the
next step's message, and stops when a step has an ``output`` and no tool
calls.

A tool call gives its arguments in two ways. ``args`` (or ``args_json``)
holds the values the model writes out. ``refs`` names values the model does
not retype: the output of a tool call from an earlier step, looked up by its
``call_id``, or one of the context variables. Results are kept by ``call_id``
for one invocation only. A call that fails is kept as ``_FAILED``: it has no
output, so a later call cannot take its error message as an argument.

Nothing about a call's arguments ends a run. Arguments that cannot be read or
resolved, and a tool that raises, become that call's ``Error: …`` output,
which the model reads on its next step.
"""

import asyncio
import json
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
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

_FAILED = object()


class ToolCall(BaseModel):
    """A tool call as the agent reads it from a step.

    The model is asked for a stricter form, with ``args`` typed to the tool's
    input (see :mod:`kavalai.step_schema`). A provider that does not enforce
    that schema can still write a value of the wrong shape, so the fields are
    read leniently here and checked when the call is run: a malformed value
    becomes that call's error output instead of failing the whole step.

    Attributes:
        name: The tool's URI, for instance ``python://mypackage.mytool``.
        args: The argument values the model writes out, as an object. A null
            value means the argument is not supplied. A JSON object in a
            string is accepted.
        args_json: The same values as a JSON object in a string, the form
            asked of a tool whose arguments cannot be typed.
        refs: Arguments taken from elsewhere, each an :class:`ArgRef`. An
            object mapping the argument name to ``{"source", "key"}`` is
            accepted too.
        call_id: The name under which later steps refer to the result.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: Union[dict[str, Any], str, None] = None
    args_json: Optional[str] = None
    refs: Union[list[Any], dict[str, Any], None] = None
    call_id: Optional[str] = None


class _ToolArgumentError(ValueError):
    """A tool call's arguments cannot be read or resolved."""


def _parse_json_object(
    field_name: str, value: Union[str, dict[str, Any], None]
) -> dict[str, Any]:
    """Read an argument field: an object, or a JSON object in a string.

    Args:
        field_name: The field the value came from, for the error message.
        value: The field's value. ``None`` and an empty string are an empty
            object.

    Raises:
        _ToolArgumentError: The string is not JSON, or not a JSON object.
    """
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


def _written_arguments(tool_call: ToolCall) -> dict[str, Any]:
    """The argument values the model wrote out, from ``args`` and ``args_json``.

    A null in ``args`` is left out: the typed schema makes the model name
    every argument, so null is how it says "not supplied". ``args_json`` is
    free-form and keeps its nulls.
    """
    written = dict(_parse_json_object("args_json", tool_call.args_json))
    for name, value in _parse_json_object("args", tool_call.args).items():
        if value is not None:
            written[name] = value
    return written


def _read_refs(tool_call: ToolCall) -> list[ArgRef]:
    """The ``refs`` entries of a call, validated.

    Raises:
        _ToolArgumentError: An entry lacks ``arg``, ``source`` or ``key``, or
            names a source other than ``"step"`` and ``"input"``.
    """
    entries: Any = tool_call.refs or []
    if isinstance(entries, dict):
        # The object form, {"document": {"source": …, "key": …}}: the
        # argument's name is the key of the object.
        entries = [
            {**entry, "arg": arg} if isinstance(entry, dict) else entry
            for arg, entry in entries.items()
        ]

    refs = []
    for entry in entries:
        try:
            refs.append(ArgRef.model_validate(entry))
        except ValidationError as error:
            raise _ToolArgumentError(
                f"`refs` entry {entry!r} is malformed: each needs `arg`, `source` "
                '("step" or "input") and `key`.'
            ) from error
    return refs


def _resolve_ref(
    ref: ArgRef,
    tool_results: Mapping[str, Any],
    variables: Mapping[str, Any],
    same_step_ids: Collection[str] = (),
) -> Any:
    """Look up the value one ``refs`` entry points at.

    Args:
        ref: The entry to resolve.
        tool_results: The results of the tool calls of earlier steps, by
            ``call_id`` (``_FAILED`` for a call that has no output).
        variables: The context variables, by name.
        same_step_ids: The ``call_id``s of the step being run.

    Raises:
        _ToolArgumentError: The entry names a context variable or a
            ``call_id`` that does not exist, a call of the same step, or a
            call that failed.
    """
    if ref.source == "input":
        if ref.key not in variables:
            raise _ToolArgumentError(
                f"`{ref.arg}` refers to the context variable {ref.key!r}, which "
                f"does not exist. Available: {_comma_list(variables)}."
            )
        return variables[ref.key]

    if ref.key in same_step_ids:
        raise _ToolArgumentError(
            f"`{ref.arg}` refers to the call_id {ref.key!r}, a call of this same "
            "step. The calls of one step run together: refer to it from a "
            "later step."
        )
    if ref.key not in tool_results:
        with_output = [
            call_id for call_id, result in tool_results.items() if result is not _FAILED
        ]
        raise _ToolArgumentError(
            f"`{ref.arg}` refers to the call_id {ref.key!r}, which does not "
            f"exist. Available: {_comma_list(with_output)}."
        )
    if tool_results[ref.key] is _FAILED:
        raise _ToolArgumentError(
            f"`{ref.arg}` refers to the call_id {ref.key!r}, which ended in an "
            "error and has no output."
        )
    return tool_results[ref.key]


def _comma_list(names: Collection[str]) -> str:
    return ", ".join(sorted(names)) or "none"


def _resolve_arguments(
    tool_call: ToolCall,
    tool_results: Mapping[str, Any],
    variables: Mapping[str, Any],
    same_step_ids: Collection[str] = (),
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Work out the arguments a tool call is run with.

    A written value beats a reference to the same argument, and of two
    references to one argument the later one counts. Every reference is still
    looked up, so one that names nothing is always reported.

    Args:
        tool_call: The call as the model wrote it.
        tool_results: The results of the tool calls of earlier steps, by
            ``call_id``.
        variables: The context variables, by name.
        same_step_ids: The ``call_id``s of the step this call belongs to.

    Returns:
        ``(args, refs)``. ``args`` is what the tool is called with. ``refs``
        maps each argument whose value came from a reference to that
        reference's ``{"source", "key"}``.

    Raises:
        _ToolArgumentError: The arguments cannot be read or resolved.
    """
    written = _written_arguments(tool_call)
    args: dict[str, Any] = {}
    refs: dict[str, dict[str, str]] = {}
    for ref in _read_refs(tool_call):
        value = _resolve_ref(ref, tool_results, variables, same_step_ids)
        if ref.arg not in written:
            args[ref.arg] = value
            refs[ref.arg] = {"source": ref.source, "key": ref.key}
    args.update(written)
    return args, refs


def _unique_call_id(wanted: str, taken: Collection[str]) -> str:
    """``wanted``, or ``wanted`` with the first free numeric suffix."""
    if wanted not in taken:
        return wanted
    suffix = 2
    while f"{wanted}_{suffix}" in taken:
        suffix += 1
    return f"{wanted}_{suffix}"


def _assign_unique_call_ids(
    tool_calls: Sequence[ToolCall], step_index: int, taken: Collection[str]
) -> dict[str, str]:
    """Give every tool call of a step a ``call_id`` no other call has.

    A ``call_id`` names one result for the whole invocation. A call that asks
    for a taken one is given another, rather than replacing the result that
    references were written against; a call without one is given a generated
    one. The calls are changed in place.

    Args:
        tool_calls: The step's tool calls.
        step_index: The step's index, used in a generated id.
        taken: The ``call_id``s of earlier steps.

    Returns:
        For each call that did not get the id it asked for, its new
        ``call_id`` mapped to the one it asked for.
    """
    taken = set(taken)
    renamed: dict[str, str] = {}
    for position, tool_call in enumerate(tool_calls):
        wanted = tool_call.call_id or f"tool_call_{step_index}_{position}"
        tool_call.call_id = _unique_call_id(wanted, taken)
        taken.add(tool_call.call_id)
        if tool_call.call_id != wanted:
            renamed[tool_call.call_id] = wanted
    return renamed


@dataclass
class _ToolCallOutcome:
    """What running one tool call of a step produced.

    Attributes:
        tool_call: The call as the model wrote it.
        args: The arguments the tool was called with; empty when they could
            not be resolved.
        refs: The arguments taken by reference, each mapped to its
            ``{"source", "key"}``.
        output: The tool's result, or an ``Error: …`` message.
        failed: The call has no output: it was not run, or the tool raised.
        duration_seconds: Time spent resolving and running this call.
    """

    tool_call: ToolCall
    args: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, dict[str, str]] = field(default_factory=dict)
    output: Any = None
    failed: bool = False
    duration_seconds: float = 0.0


def get_step_output_type(response_model: Any) -> Type[BaseModel]:
    """The model the agent reads a step with.

    Its ``tool_calls`` are the lenient :class:`ToolCall`. The schema sent to
    the model is built by :func:`kavalai.step_schema.build_step_schema`.

    Args:
        response_model: The type of the step's final ``output``.
    """
    return step_output_model(response_model, ToolCall)


def _serialise_final_output(output: Any) -> Optional[str]:
    """The final output as the closing stream chunk carries it."""
    if output is None or isinstance(output, str):
        return output
    return output.model_dump_json()


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
        """Add a piece of the step's raw text and emit what it completes."""
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


def _load_default_template(filename: str) -> Template:
    """Load one of the templates that sit beside this module."""
    return Template(Path(__file__).with_name(filename).read_text())


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
        self.prompt_template = prompt_template or _load_default_template(
            "default_prompt_template.j2"
        )
        self.step_template = step_template or _load_default_template(
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
        output_type = response_model or str
        tool_models = await self._allowed_tool_input_models()
        # Two models of one step. `step_schema` is what the model is asked
        # for: one typed call model per tool. `StepOutput` is how the reply
        # is read: leniently, so a malformed call costs a tool error the model
        # can correct instead of a failed run.
        step_schema = build_step_schema(output_type, tool_models)
        StepOutput = get_step_output_type(output_type)

        variables = dict(inputs) if inputs is not None else self.run_context.data
        tool_descriptions = (
            await self.kernel.get_tool_descriptions(self.allowed_tools)
            if self.kernel
            else ""
        )
        # State of this invocation only: the steps run so far, and every tool
        # result by call_id, which a later call's "step" reference names.
        steps: list[dict[str, Any]] = []
        tool_results: dict[str, Any] = {}
        final_output: Any = None

        for step_index in range(max_steps):
            is_last_step = step_index == max_steps - 1
            chat_history = self._render_step_messages(
                history or [],
                prompt=prompt,
                data=variables,
                tool_descriptions=tool_descriptions,
                steps=steps,
                current_step=step_index,
                max_steps=max_steps,
                last_step=is_last_step,
                request=request,
                json_argument_tools=untyped_tools(tool_models),
            )

            logger.info(f"Agent step {step_index}/{max_steps}")
            # Always consume the step in raw-delta mode: the demuxer needs the
            # full raw text to safe-parse, and its buffer doubles as the value
            # to validate on completion.
            streamer = await self.llm_client.stream_chat_completions(
                chat_history=chat_history,
                response_model=step_schema,
                stream_delta=True,
            )
            demux = StepStreamDemuxer(
                step_index,
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

            tool_call_records: list[dict[str, Any]] = []
            if step_output.tool_calls and is_last_step:
                # There is no next step to read a result in, and the step
                # message said so: the calls are not run.
                logger.warning(
                    f"Agent asked for {len(step_output.tool_calls)} tool call(s) "
                    "on its last step; not run"
                )
            elif step_output.tool_calls and self.kernel:
                tool_call_records = await self._run_tool_calls(
                    step_output.tool_calls, step_index, tool_results, variables
                )

            step_record = {
                "index": step_index,
                "instructions": step_output.instructions,
                "tool_calls": tool_call_records,
                "output": to_plain(step_output.output),
            }
            steps.append(step_record)
            self._notify_step_observer(step_record)

            if step_output.output is not None:
                final_output = step_output.output
                # An answer without further tool calls ends the run. One given
                # alongside tool calls is kept in case no later step answers.
                if not step_output.tool_calls:
                    break
        else:
            if final_output is None:
                logger.warning(
                    f"Agent used its {max_steps} steps without producing an output"
                )

        yield StreamContent(
            type="complete",
            name="response",
            value=_serialise_final_output(final_output),
        )

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

    async def _allowed_tool_input_models(self) -> dict[str, Type[BaseModel]]:
        """The input model of every tool this agent may call, by tool URI."""
        if not self.kernel:
            return {}
        definitions = await self.kernel.get_tool_definitions(self.allowed_tools)
        return {uri: definition.input_model for uri, definition in definitions.items()}

    def _render_step_messages(
        self, history: Sequence[ChatMessage], **template_variables: Any
    ) -> ChatHistory:
        """Build the conversation sent to the model for one step.

        The system message is rendered from ``prompt_template``, the last
        user message from ``step_template``, and the chat ``history`` sits
        between them. Both templates receive the same variables.
        """
        system_prompt = self.prompt_template.render(**template_variables)
        step_message = self.step_template.render(**template_variables)
        if self.debug:
            print(system_prompt)
            print(step_message)
        return ChatHistory(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                *history,
                ChatMessage(role="user", content=step_message),
            ]
        )

    async def _run_tool_calls(
        self,
        tool_calls: Sequence[ToolCall],
        step_index: int,
        tool_results: dict[str, Any],
        variables: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Run the tool calls of one step, together.

        Each call's result is stored in ``tool_results`` under its
        ``call_id``, where the calls of later steps can refer to it.

        Returns:
            One record per call, in the order the model wrote them: what the
            step message and the ``on_step`` observer show of it.
        """
        requested_ids = _assign_unique_call_ids(tool_calls, step_index, tool_results)
        same_step_ids = {tool_call.call_id for tool_call in tool_calls}
        outcomes = await asyncio.gather(
            *[
                self._run_tool_call(tool_call, tool_results, variables, same_step_ids)
                for tool_call in tool_calls
            ]
        )

        records = []
        for outcome in outcomes:
            call_id = outcome.tool_call.call_id
            tool_results[call_id] = _FAILED if outcome.failed else outcome.output
            records.append(
                {
                    "name": outcome.tool_call.name,
                    "args": outcome.args,
                    "refs": outcome.refs,
                    "call_id": call_id,
                    "requested_call_id": requested_ids.get(call_id),
                    "output": to_plain(outcome.output),
                    "duration_seconds": outcome.duration_seconds,
                }
            )
        return records

    async def _run_tool_call(
        self,
        tool_call: ToolCall,
        tool_results: Mapping[str, Any],
        variables: Mapping[str, Any],
        same_step_ids: Collection[str] = (),
    ) -> _ToolCallOutcome:
        """Run one tool call and time it.

        The duration is measured per call rather than around the step's
        ``asyncio.gather``, which would time the slowest call of the step and
        attribute it to all of them.
        """
        started = time.perf_counter()
        outcome = await self._execute_tool_call(
            tool_call, tool_results, variables, same_step_ids
        )
        outcome.duration_seconds = time.perf_counter() - started
        return outcome

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        tool_results: Mapping[str, Any],
        variables: Mapping[str, Any],
        same_step_ids: Collection[str],
    ) -> _ToolCallOutcome:
        """Resolve a tool call's arguments and call the tool through the kernel.

        Never raises. Whatever goes wrong becomes the call's output, an
        ``Error: …`` message the model reads on its next step, and the
        outcome is marked ``failed``.
        """
        try:
            args, refs = _resolve_arguments(
                tool_call, tool_results, variables, same_step_ids
            )
        except _ToolArgumentError as error:
            # Not run: a tool called with half its arguments answers a
            # question the model did not ask.
            logger.warning(f"Tool {tool_call.name} was not called: {error}")
            return _ToolCallOutcome(tool_call, output=f"Error: {error}", failed=True)

        if not _is_tool_allowed(tool_call.name, self.allowed_tools):
            # The tool was never described to the model; tell it so it can
            # pick one it is actually allowed to call.
            logger.warning(f"Tool {tool_call.name} is not allowed for this agent")
            return _ToolCallOutcome(
                tool_call,
                args,
                refs,
                output=f"Error: tool {tool_call.name} is not available.",
                failed=True,
            )

        logger.info(f"Calling tool {tool_call.name} with {args}")
        try:
            output = await self.kernel.call_tool(
                tool_uri=tool_call.name, arguments=args
            )
        except Exception as error:
            logger.error(f"Tool {tool_call.name} failed: {error}")
            return _ToolCallOutcome(
                tool_call, args, refs, output=f"Error: {error}", failed=True
            )
        return _ToolCallOutcome(tool_call, args, refs, output=output)

    def _notify_step_observer(self, step_record: dict[str, Any]) -> None:
        """Hand a completed step to the ``on_step`` observer, if there is one.

        Never raises: an observer that fails is a logging problem, not a
        reason to abandon the run.
        """
        if self.on_step is None:
            return
        try:
            self.on_step(step_record)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(f"Agent on_step observer failed: {error}")


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
