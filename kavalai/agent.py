import asyncio
import json
import time
from typing import Any, AsyncGenerator, Callable, Optional, Type
import os

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from kavalai.run_context import RunContext
from kavalai.utils import to_plain
from kavalai.functionkernel import FunctionKernel, _is_tool_allowed
from kavalai.llm_clients.base_client import BaseLlmClient, ChatHistory, ChatMessage
from kavalai.llm_clients.common import safe_parse_json
from kavalai.llm_clients.streamer import StreamContent
from jinja2 import Template


class ToolCall(BaseModel):
    """This data structure represents tool call requests.

    Arguments are expected to be JSON encoded to help LLM models encode the data.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool call name i.e python://mypackage.mytool")
    literal_args: str = Field(
        default="{}",
        description="Literal values to use as arguments for the tool call.",
    )
    planner_context_args: str = Field(
        default="{}",
        description="Map of tool argument names to keys in planner_context.",
    )
    input_args: str = Field(
        default="{}",
        description="Map of tool argument names to keys in input_data.",
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Generate an ID, which represents this result in downstream agent runs.",
    )


def get_step_output_type(ResponseModel: Type[BaseModel]):
    class StepOutput(BaseModel):
        """Data structure that helps passing around information between consecutive agent runs."""

        instructions: str = Field(
            description="Briefly describe the goal of this step (what you intend to achieve with these tool calls).",
        )
        tool_calls: list[ToolCall] = Field(
            default_factory=list,
            description="Add tool call requests here, their output will be available via `call_id` key for next steps.",
        )
        output: Optional[ResponseModel] = None

    return StepOutput


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
    ):
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

    def step_completed(self, step_output) -> list[StreamContent]:
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


class Agent:
    def __init__(
        self,
        llm_client: BaseLlmClient,
        *,
        kernel: Optional[FunctionKernel] = None,
        run_context: Optional[RunContext] = None,
        prompt_template: Optional[Template] = None,
        allowed_tools: Optional[list[str]] = None,
        on_step: Optional[Callable[[dict], None]] = None,
        debug: bool = False,
    ):
        """Build an agent.

        ``allowed_tools`` restricts the agent to a subset of the kernel's
        tools, given as tool URIs (``python://web.crawl``, or ``rest://api.*``
        for a whole server). The tools outside it are neither described to the
        model nor callable. ``None`` (the default) allows every registered
        tool; an empty list allows none.

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
        if prompt_template is None:
            with open(
                os.path.join(os.path.dirname(__file__), "default_prompt_template.j2"),
                "r",
            ) as f:
                prompt_template = Template(f.read())
        self.prompt_template = prompt_template

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
        """
        StepOutput = get_step_output_type(response_model or str)

        # Per-invocation working memory: tool call results keyed by call_id,
        # referenced via `planner_context_args`. Created fresh for each
        # invocation (up to `max_steps`) and discarded afterwards, unlike
        # `self.run_context` which is passed in at construction.
        planner_context: dict[str, Any] = {}
        # Record of executed steps, rendered back into the prompt template.
        steps: list[dict] = []
        final_output: Optional[BaseModel] = None

        for step_idx in range(max_steps):
            rendered_prompt = self.prompt_template.render(
                prompt=prompt,
                data=self.run_context.data,
                tool_descriptions=(
                    await self.kernel.get_tool_descriptions(self.allowed_tools)
                    if self.kernel
                    else ""
                ),
                steps=steps,
                current_step=step_idx,
                max_steps=max_steps,
            )

            if self.debug:
                print(rendered_prompt)

            chat_history = ChatHistory(
                messages=[
                    ChatMessage(role="system", content=rendered_prompt),
                    ChatMessage(
                        role="user",
                        content="Analyze the situation and provide the next step output.",
                    ),
                ]
            )

            logger.info(f"Agent step {step_idx}/{max_steps}")
            # Always consume the step in raw-delta mode: the demuxer needs the
            # full raw text to safe-parse, and its buffer doubles as the value
            # to validate on completion.
            streamer = await self.llm_client.stream_chat_completions(
                chat_history=chat_history,
                response_model=StepOutput,
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

            if step_output.tool_calls and self.kernel:
                results = await asyncio.gather(
                    *[
                        self._call_tool(tc, planner_context)
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
    ) -> str | BaseModel:
        """Run the agent loop and return the final output (blocking wrapper).

        Drains :meth:`prompt_stream` with all progress streams disabled.

        Args:
            prompt: The task description for the agent.
            response_model: Optional Pydantic model describing the structured
                final output. When omitted, a plain string is returned.
            max_steps: Maximum number of reasoning/tool-calling iterations.

        Returns:
            The structured ``response_model`` instance, or a string when no
            ``response_model`` is provided. ``None`` if no output was produced.
        """
        final_value = None
        async for chunk in self.prompt_stream(
            prompt, response_model=response_model, max_steps=max_steps
        ):
            if chunk.type == "complete" and chunk.name == "response":
                final_value = chunk.value

        if final_value is None:
            return None
        if response_model:
            return response_model.model_validate_json(final_value)
        return final_value

    def _resolve_args(
        self, tool_call: ToolCall, planner_context: dict[str, Any]
    ) -> dict:
        """Resolve a ToolCall's argument sources into a single argument dict.

        Arguments are merged with precedence ``literal_args`` >
        ``planner_context_args`` > ``input_args``. ``planner_context_args``
        resolves against the per-invocation ``planner_context`` (results of
        previous tool calls); ``input_args`` against ``self.run_context.data``.
        """

        def parse(field_name: str, value: str) -> dict:
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse {field_name} as JSON: {value}")
                return {}

        literal_args = parse("literal_args", tool_call.literal_args)

        context_keys = parse("planner_context_args", tool_call.planner_context_args)
        context_args = {
            arg_name: planner_context.get(ctx_key)
            for arg_name, ctx_key in context_keys.items()
        }

        input_keys = parse("input_args", tool_call.input_args)
        input_args = {
            arg_name: self.run_context.data.get(input_key)
            for arg_name, input_key in input_keys.items()
        }

        return {**input_args, **context_args, **literal_args}

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
        self, tool_call: ToolCall, planner_context: dict[str, Any]
    ) -> tuple[ToolCall, dict, Any, float]:
        """Resolve arguments and execute a single tool call via the kernel.

        Returns ``(tool_call, args, result, duration_seconds)``. Execution
        errors are captured and returned as the result so the model can
        self-correct. The duration is measured here rather than around the
        caller's ``asyncio.gather`` — that would time the slowest call in the
        batch and attribute it to all of them.
        """
        args = self._resolve_args(tool_call, planner_context)
        start = time.perf_counter()
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

    llm_client = OpenAIClient("gpt-5.4-mini")
    agent = Agent(llm_client=llm_client, kernel=kernel, debug=True)
    result = asyncio.run(agent.prompt("Greet the user based on current time!"))
    print(result)
