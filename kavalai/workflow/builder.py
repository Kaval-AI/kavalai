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

from typing import Any, Optional, Union

from pydantic import BaseModel

from kavalai.workflow.models import (
    AgentNode,
    ArgumentInfo,
    EndNode,
    FunctionNode,
    IfNode,
    LLMNode,
    McpServer,
    ParallelNode,
    PythonFunction,
    RestServer,
    StartNode,
    SwitchNode,
    TemplateModel,
    WorkflowGraph,
)

# Python types accepted as a shorthand for a JSON-schema scalar type.
_TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# A field type may be given as a JSON-schema type name ("string"), a Python type
# (``str``), or a full JSON-schema fragment (``{"type": "array", ...}``).
FieldType = Union[str, type, dict]

# A node input may be an ArgumentInfo, a context path string, or a raw dict.
InputSpec = Union[ArgumentInfo, str, dict]


def _field_schema(value: FieldType) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, type):
        if value not in _TYPE_NAMES:
            raise TypeError(f"Unsupported field type: {value!r}")
        return {"type": _TYPE_NAMES[value]}
    if isinstance(value, str):
        return {"type": value}
    raise TypeError(f"Unsupported field type: {value!r}")


def _coerce_inputs(inputs: Optional[dict[str, InputSpec]]) -> dict[str, ArgumentInfo]:
    """Normalise a node's ``inputs`` into ``{name: ArgumentInfo}``.

    A plain string is treated as a context path; a dict is passed straight to
    :class:`ArgumentInfo`; an :class:`ArgumentInfo` is used as-is.
    """
    result: dict[str, ArgumentInfo] = {}
    for name, spec in (inputs or {}).items():
        if isinstance(spec, ArgumentInfo):
            result[name] = spec
        elif isinstance(spec, str):
            result[name] = ArgumentInfo(type="context", value=spec)
        elif isinstance(spec, dict):
            result[name] = ArgumentInfo(**spec)
        else:
            raise TypeError(f"Unsupported input spec for '{name}': {spec!r}")
    return result


class WorkflowBuilder:
    """A small fluent builder for constructing a :class:`WorkflowGraph` in code.

    Every method returns ``self`` so calls can be chained, and ``build()``
    validates and returns the graph (``build_engine()`` returns a ready
    :class:`WorkflowEngine`). For example::

        graph = (
            WorkflowBuilder("Greeter", llm_model="openai/gpt-4o-mini")
            .data_type("input", {"user_message": str})
            .data_type("output", {"agent_response": str})
            .start("reply")
            .llm("reply", prompt="Greet the user.",
                 inputs={"input": "input"}, output="output", next="end")
            .end()
            .build()
        )
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        version: str = "2.0",
        llm_model: Optional[str] = None,
        llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.version = version
        self.llm_model = llm_model
        self.llm_kwargs = llm_kwargs or {}
        self._data_types: dict[str, dict] = {}
        self._data_models: dict[str, type[BaseModel]] = {}
        self._nodes: list = []
        self._rest_servers: list[RestServer] = []
        self._mcp_servers: list[McpServer] = []
        self._python_functions: list[PythonFunction] = []
        self._templates: list[TemplateModel] = []

    # --------------------------------------------------------------- data types
    def data_type(
        self,
        name: str,
        fields: Optional[dict[str, FieldType]] = None,
        *,
        schema: Optional[dict] = None,
        ref: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Declare a data type.

        Pass ``fields`` for an object of named scalars (``{"intent": str}``),
        ``schema`` for a full JSON-schema fragment, or ``ref`` to alias another
        type.
        """
        if schema is not None:
            self._data_types[name] = schema
        elif ref is not None:
            self._data_types[name] = {"$ref": ref}
        else:
            properties = {f: _field_schema(t) for f, t in (fields or {}).items()}
            self._data_types[name] = {"type": "object", "properties": properties}
        return self

    def data_model(self, name: str, model: type[BaseModel]) -> "WorkflowBuilder":
        """Declare a data type from a Pydantic model class.

        The model is used directly at run time (its fields validate the matching
        node input/output and drive structured LLM output), so you get full
        Pydantic expressiveness — ``Literal`` enums, defaults, validators —
        without writing a JSON-schema fragment. Its JSON schema is also recorded
        on the graph for storage and inspection.

        Build the engine with :meth:`build_engine` (which forwards the models) or
        pass ``data_models=`` to :class:`WorkflowEngine` yourself.
        """
        self._data_models[name] = model
        self._data_types[name] = model.model_json_schema()
        return self

    # -------------------------------------------------------------------- nodes
    def start(self, next: str, *, name: str = "start") -> "WorkflowBuilder":
        self._nodes.append(StartNode(name=name, next=next))
        return self

    def end(self, *, name: str = "end", output: str = "output") -> "WorkflowBuilder":
        self._nodes.append(EndNode(name=name, output=output))
        return self

    def llm(
        self,
        name: str,
        *,
        prompt: str,
        output: str,
        next: str,
        inputs: Optional[dict[str, InputSpec]] = None,
        use_history: bool = True,
        llm_model: Optional[str] = None,
        llm_kwargs: Optional[dict[str, Any]] = None,
        stream_output: bool = False,
        stream_delta: bool = False,
    ) -> "WorkflowBuilder":
        self._nodes.append(
            LLMNode(
                name=name,
                prompt=prompt,
                output=output,
                next=next,
                inputs=_coerce_inputs(inputs),
                use_history=use_history,
                llm_model=llm_model,
                llm_kwargs=llm_kwargs or {},
                stream_output=stream_output,
                stream_delta=stream_delta,
            )
        )
        return self

    def agent(
        self,
        name: str,
        *,
        prompt: str,
        output: str,
        next: str,
        inputs: Optional[dict[str, InputSpec]] = None,
        allowed_tools: Optional[list[str]] = None,
        max_steps: int = 10,
        llm_model: Optional[str] = None,
        llm_kwargs: Optional[dict[str, Any]] = None,
        stream_output: bool = False,
        stream_delta: bool = False,
        stream_instructions: bool = False,
        stream_partials: bool = False,
    ) -> "WorkflowBuilder":
        self._nodes.append(
            AgentNode(
                name=name,
                prompt=prompt,
                output=output,
                next=next,
                inputs=_coerce_inputs(inputs),
                allowed_tools=allowed_tools,
                max_steps=max_steps,
                llm_model=llm_model,
                llm_kwargs=llm_kwargs or {},
                stream_output=stream_output,
                stream_delta=stream_delta,
                stream_instructions=stream_instructions,
                stream_partials=stream_partials,
            )
        )
        return self

    def function(
        self,
        name: str,
        *,
        tool: str,
        output: str,
        next: str,
        inputs: Optional[dict[str, InputSpec]] = None,
        method: str = "get",
    ) -> "WorkflowBuilder":
        self._nodes.append(
            FunctionNode(
                name=name,
                tool=tool,
                output=output,
                next=next,
                inputs=_coerce_inputs(inputs),
                method=method,
            )
        )
        return self

    def parallel(
        self,
        name: str,
        *,
        branches: list[str],
        next: str,
        max_concurrency: Optional[int] = None,
    ) -> "WorkflowBuilder":
        """Fan out to independent branches and rejoin at ``next``.

        Each entry in ``branches`` names the first node of a branch; every
        branch is walked concurrently up to ``next``. Branch subgraphs must be
        disjoint and must not write the same output variable — both are checked
        by :meth:`build`.
        """
        self._nodes.append(
            ParallelNode(
                name=name,
                branches=branches,
                next=next,
                max_concurrency=max_concurrency,
            )
        )
        return self

    def if_(
        self, name: str, *, condition: str, then: str, else_: str
    ) -> "WorkflowBuilder":
        self._nodes.append(
            IfNode(name=name, condition=condition, then=then, else_=else_)
        )
        return self

    def switch(
        self,
        name: str,
        *,
        expr: str,
        cases: Optional[dict[str, str]] = None,
        default: Optional[str] = None,
    ) -> "WorkflowBuilder":
        self._nodes.append(
            SwitchNode(name=name, expr=expr, cases=cases or {}, default=default)
        )
        return self

    # ---------------------------------------------------------------- resources
    def python_function(self, name: str, path: str) -> "WorkflowBuilder":
        """Register a Python tool by import path (e.g. ``pkg.mod.func``)."""
        self._python_functions.append(PythonFunction(name=name, path=path))
        return self

    def rest_server(self, server: Union[RestServer, dict]) -> "WorkflowBuilder":
        self._rest_servers.append(
            server if isinstance(server, RestServer) else RestServer(**server)
        )
        return self

    def mcp_server(self, server: Union[McpServer, dict]) -> "WorkflowBuilder":
        self._mcp_servers.append(
            server if isinstance(server, McpServer) else McpServer(**server)
        )
        return self

    def template(self, name: str, value: str) -> "WorkflowBuilder":
        self._templates.append(TemplateModel(name=name, value=value))
        return self

    # -------------------------------------------------------------------- build
    def build(self) -> WorkflowGraph:
        """Validate and return the :class:`WorkflowGraph`."""
        return WorkflowGraph(
            name=self.name,
            description=self.description,
            version=self.version,
            llm_model=self.llm_model,
            llm_kwargs=self.llm_kwargs,
            data_types=self._data_types,
            nodes=self._nodes,
            rest_servers=self._rest_servers,
            mcp_servers=self._mcp_servers,
            python_functions=self._python_functions,
            templates=self._templates,
        )

    def build_engine(self, **kwargs):
        """Build the graph and wrap it in a ready-to-run :class:`WorkflowEngine`.

        Any Pydantic models registered via :meth:`data_model` are forwarded to
        the engine. Keyword arguments are passed through (``agent_service``,
        ``task_logger``, ``client_factory``, ``data_models``, ``max_node_visits``).
        """
        from kavalai.workflow.engine import WorkflowEngine

        if self._data_models:
            kwargs.setdefault("data_models", self._data_models)
        return WorkflowEngine(self.build(), **kwargs)
