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

The schema of an agent step, as it is sent to the model.

An agent step is one structured output: ``instructions``, ``tool_calls`` and
``output``. :func:`build_step_schema` builds the schema of that output with
one call model per tool the agent may use: ``name`` is that tool's URI,
``args`` an object held to the tool's input schema, and ``refs`` a list of the
arguments taken from an earlier tool result or a context variable. A provider
that enforces its structured output cannot then produce a call whose arguments
the tool would refuse.

A tool whose input model has a free-form value (a ``dict`` argument, most MCP
and REST tools) cannot be typed: strict structured output refuses such a
schema. Its call model takes ``args_json``, the arguments as a JSON object in
a string, in place of ``args``.

The schema built here is only what the model is asked for. The agent reads a
reply with the more lenient :class:`kavalai.agent.ToolCall`, so a provider
that does not enforce the schema costs a tool error the model can correct, not
a failed run.

The module imports Pydantic only, so it loads under Pyodide.
"""

import re
from typing import Any, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

_CALL_ID_DESCRIPTION = (
    "Generate an ID, which represents this result in downstream agent runs."
)
_INSTRUCTIONS_DESCRIPTION = (
    "Briefly describe the goal of this step (what you intend to achieve with "
    "these tool calls)."
)
_TOOL_CALLS_DESCRIPTION = (
    "Add tool call requests here, their output will be available via `call_id` "
    "key for next steps."
)
_STEP_DESCRIPTION = (
    "Data structure that helps passing around information between consecutive "
    "agent runs."
)

_CLOSED = ConfigDict(extra="forbid")


class ArgRef(BaseModel):
    """A tool argument taken from elsewhere instead of written out.

    Attributes:
        arg: The name of the tool argument to fill.
        source: ``"step"`` for the output of an earlier tool call of this
            agent invocation, ``"input"`` for one of the agent's context
            variables.
        key: The ``call_id`` of that tool call, or the name of that variable.
    """

    arg: str
    source: Literal["step", "input"]
    key: str


def step_output_model(response_model: Any, call_type: Any) -> type[BaseModel]:
    """Build a step model around a tool call type.

    Args:
        response_model: The type of the step's final ``output``.
        call_type: The type of one entry of ``tool_calls``. ``None`` leaves
            the field out, for an agent that has no tool to call.

    Returns:
        A model named ``StepOutput`` with ``instructions``, ``tool_calls``
        (unless left out) and ``output``.
    """
    fields: dict[str, Any] = {
        "instructions": (str, Field(description=_INSTRUCTIONS_DESCRIPTION)),
    }
    if call_type is not None:
        fields["tool_calls"] = (
            list[call_type],
            Field(default_factory=list, description=_TOOL_CALLS_DESCRIPTION),
        )
    fields["output"] = (Optional[response_model], None)
    return create_model("StepOutput", __doc__=_STEP_DESCRIPTION, **fields)


def has_closed_schema(model: type[BaseModel]) -> bool:
    """Whether every value of a model has a schema that names its fields.

    A free-form value — ``dict``, ``Any``, a bare ``list`` — has none: there
    is nothing to hold the model's output to, and strict structured output
    refuses such a schema. A tool whose input model has one is called with
    its arguments as a JSON string instead (:func:`typed_call_model`).
    """
    schema = model.model_json_schema()
    return _is_closed(schema) and all(
        _is_closed(definition) for definition in schema.get("$defs", {}).values()
    )


def _is_closed(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if "$ref" in node or "enum" in node or "const" in node:
        return True
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in node:
            return all(_is_closed(option) for option in node[keyword])
    kind = node.get("type")
    if kind == "object":
        properties = node.get("properties")
        if properties is None or node.get("additionalProperties") not in (None, False):
            return False
        return all(_is_closed(value) for value in properties.values())
    if kind == "array":
        if "prefixItems" in node:
            return all(_is_closed(item) for item in node["prefixItems"])
        return _is_closed(node.get("items"))
    return kind is not None


def _stem(uri: str) -> str:
    return re.sub(r"\W+", "_", uri).strip("_")


def untyped_tools(tools: Mapping[str, type[BaseModel]]) -> list[str]:
    """The URIs of the tools whose arguments cannot be typed, in order."""
    return [uri for uri, model in tools.items() if not has_closed_schema(model)]


def typed_call_model(
    uri: str, input_model: type[BaseModel], stem: Optional[str] = None
) -> type[BaseModel]:
    """Build the call model of one tool for the step schema.

    Every argument is nullable and optional: an argument passed by reference
    is absent from ``args``, and strict structured output makes every property
    required, so a model writes ``null`` for what it does not supply. The
    agent reads a null as "not supplied"; the tool's own input model, with its
    required fields, is checked when the call is made.

    Args:
        uri: The tool's URI as the tool description shows it, for instance
            ``python://lookup`` or ``rest://crm.find [GET]``.
        input_model: The tool's input model.
        stem: The prefix of the generated models' names. Defaults to the URI
            with every run of other characters replaced by an underscore.

    Returns:
        A closed model with ``name`` fixed to ``uri``, the arguments as
        ``args`` (or ``args_json``, a JSON string, when the input model has a
        free-form value), ``refs`` unless the tool takes no arguments, and
        ``call_id``.
    """
    stem = stem or _stem(uri)
    fields: dict[str, Any] = {"name": (Literal[uri], ...)}
    parameters = list(input_model.model_fields)

    if has_closed_schema(input_model):
        arg_fields: dict[str, Any] = {
            name: (
                Optional[info.annotation],
                Field(default=None, description=info.description),
            )
            for name, info in input_model.model_fields.items()
        }
        fields["args"] = (
            create_model(f"{stem}_args", __config__=_CLOSED, **arg_fields),
            Field(
                description=(
                    "The argument values. Null for an argument that is not supplied."
                )
            ),
        )
    else:
        fields["args_json"] = (
            str,
            Field(
                description=(
                    "The argument values as a JSON object in a string, for "
                    'instance "{\\"query\\": \\"Kaval AI\\"}".'
                )
            ),
        )

    if parameters:
        ref_model = create_model(
            f"{stem}_ref",
            __config__=_CLOSED,
            arg=(
                Literal[tuple(parameters)],
                Field(description="The argument to fill."),
            ),
            source=(
                Literal["step", "input"],
                Field(
                    description="step: a previous tool call. input: a context variable."
                ),
            ),
            key=(
                str,
                Field(description="The call_id, or the context variable's name."),
            ),
        )
        fields["refs"] = (
            list[ref_model],
            Field(
                default_factory=list,
                description=(
                    "Arguments taken from a previous tool call or a context variable."
                ),
            ),
        )
    fields["call_id"] = (
        Optional[str],
        Field(default=None, description=_CALL_ID_DESCRIPTION),
    )
    return create_model(f"{stem}_call", __config__=_CLOSED, **fields)


def build_step_schema(
    response_model: Any, tools: Mapping[str, type[BaseModel]]
) -> type[BaseModel]:
    """Build the step schema an agent sends to the model.

    Args:
        response_model: The type of the agent's final output.
        tools: The input model of every tool the agent may call, keyed by the
            tool's URI as its description shows it.

    Returns:
        The model to pass as ``response_model`` to the LLM client, with one
        call model per tool. Without tools it has no ``tool_calls`` field.
    """
    # Two URIs can reduce to one name (``a.b`` and ``a_b``), and a schema
    # cannot hold two definitions under it.
    stems: dict[str, int] = {}
    call_models = []
    for uri, model in tools.items():
        stem = _stem(uri)
        stems[stem] = stems.get(stem, 0) + 1
        if stems[stem] > 1:
            stem = f"{stem}_{stems[stem]}"
        call_models.append(typed_call_model(uri, model, stem))
    call_type = Union[tuple(call_models)] if call_models else None
    return step_output_model(response_model, call_type)
