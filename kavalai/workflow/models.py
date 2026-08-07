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

Workflow data models: the shared building blocks (input wiring, server/tool
declarations and the workflow exception) and the v2 workflow graph (nodes and
:class:`WorkflowGraph`).
"""

from typing import Optional, Literal, Union, Annotated, Any

from pydantic import BaseModel, Field, model_validator


class WorkflowException(Exception):
    """Base exception for errors building, validating or running a workflow."""


class ArgumentInfo(BaseModel):
    """Describes input arguments in workflow YAML files.

    The 'type' field describes where the input argument should be retrieved from.
    'literal' - use value as specified
    'context' - retrieve from agent run context
    'history' - retrieve from previous agent run contexts.

    """

    type: Literal["literal", "context", "history"]
    value: Optional[BaseModel | str | int | float | bool] = None
    name: Optional[str] = None


class RestServer(BaseModel):
    """Defines a REST server.

    We also support HTTP Basic Auth for REST server endpoints, which are defined via
    environment variables username_env and password_env.

    Note that url_env can also be read from the env file.
    """

    name: str
    url: Optional[str] = None
    url_env: Optional[str] = None
    username_env: Optional[str] = None
    password_env: Optional[str] = None

    @model_validator(mode="after")
    def check_url_configs(self) -> "RestServer":
        if self.url and self.url_env:
            raise ValueError(
                f"REST server '{self.name}': Only one of 'url' or 'url_env' can be specified."
            )
        if not self.url and not self.url_env:
            raise ValueError(
                f"REST server '{self.name}': Either 'url' or 'url_env' must be specified."
            )
        return self


class McpServer(BaseModel):
    """Defines an MCP server."""

    name: str
    command: Optional[str] = None
    command_env: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: Optional[str] = None
    url_env: Optional[str] = None

    @model_validator(mode="after")
    def check_configs(self) -> "McpServer":
        stdio_configured = bool(self.command or self.command_env)
        http_configured = bool(self.url or self.url_env)

        if stdio_configured and http_configured:
            raise ValueError(
                f"MCP server '{self.name}': Cannot specify both stdio (command/command_env) and HTTP (url/url_env) configurations."
            )
        if not stdio_configured and not http_configured:
            raise ValueError(
                f"MCP server '{self.name}': Either stdio (command/command_env) or HTTP (url/url_env) must be specified."
            )

        if self.command and self.command_env:
            raise ValueError(
                f"MCP server '{self.name}': Only one of 'command' or 'command_env' can be specified for stdio."
            )
        if self.url and self.url_env:
            raise ValueError(
                f"MCP server '{self.name}': Only one of 'url' or 'url_env' can be specified for HTTP."
            )
        return self


class PythonFunction(BaseModel):
    """Declares a Python tool available to a workflow.

    Attributes:
        name: Name the tool is registered and addressed under
            (``python://<name>``).
        path: Import path to the ``@kavalai.pythontool`` decorated function,
            e.g. ``my_package.my_module.my_func``.
    """

    name: str
    path: str


class TemplateModel(BaseModel):
    """A named, reusable text template referenced within a workflow.

    Attributes:
        name: Identifier the template is referenced by.
        value: The template text (e.g. a prompt) to interpolate at run time.
    """

    name: str
    value: str


class BaseNode(BaseModel):
    """Common fields shared by every node in a workflow graph.

    A node is one vertex in the DAG/state-machine. ``name`` uniquely identifies
    the node and is the target referenced by transitions (``next``/``then``/
    ``else``/``cases``/``default``).
    """

    name: str


class StartNode(BaseNode):
    """Interaction start node.

    The caller hands an input to this node; execution begins here and proceeds
    to ``next``.
    """

    type: Literal["start"] = "start"
    next: str


class EndNode(BaseNode):
    """Interaction end node.

    Reaching an end node terminates the interaction. ``output`` names the
    context variable whose value is returned to the caller.
    """

    type: Literal["end"] = "end"
    output: str = "output"


class LLMNode(BaseNode):
    """Single LLM completion node.

    Resolves ``inputs`` from context, renders ``prompt`` and calls the LLM,
    storing the structured result in the ``output`` context variable, then
    transitions to ``next``.

    Streaming (see :class:`WorkflowStreamEvent` for the event contract):

    ``stream_output``
        Stream the completion as ``partial`` events named after this node
        while it is generated; auxiliary provider streams (e.g. Gemini
        thoughts) are streamed as ``<node>_<stream>`` (``<node>_thought``).
    ``stream_delta``
        When True, each ``partial`` carries only the newly generated text and
        the client reassembles the value. When False (default), each
        ``partial`` carries the full accumulated, safe-parsed value so far —
        render-ready with no client-side assembly, at the cost of re-sending
        the whole buffer on every chunk (O(n^2) wire traffic over the stream;
        prefer ``stream_delta: true`` for long outputs).
    """

    type: Literal["llm"] = "llm"
    prompt: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    use_history: bool = True
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    stream_output: bool = False
    stream_delta: bool = False


class AgentNode(BaseNode):
    """Multi-step agent node.

    Runs the v2 :class:`~kavalai.agent.Agent` loop (tool calling) up
    to ``max_steps`` and stores the final result in ``output``.

    Streaming (see :class:`WorkflowStreamEvent` for the event contract):

    ``stream_output``
        Stream the step's ``output`` field as ``partial`` events named after
        this node while the model writes it. The streamed value is always the
        full safe-parsed JSON of the output so far (partial objects are not
        delta-able); the final ``complete`` event is authoritative — a
        provisional output produced alongside tool calls may be superseded.
    ``stream_instructions``
        Stream each step's ``instructions`` as ``<node>_instructions``
        partials, completed once per step — an "ideating…" status line the UI
        replaces each step.
    ``stream_partials``
        Stream each step's raw model output as ``<node>_step<N>`` — a debug
        firehose including tool-call JSON.
    ``stream_delta``
        As on :class:`LLMNode` (including the O(n^2) full-buffer trade-off);
        applies to the ``_instructions`` and ``_step<N>`` streams, not to the
        structured output stream.
    """

    type: Literal["agent"] = "agent"
    prompt: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = 10
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    stream_output: bool = False
    stream_delta: bool = False
    stream_instructions: bool = False
    stream_partials: bool = False


class FunctionNode(BaseNode):
    """Function-call node.

    Invokes a single tool via the :class:`~kavalai.functionkernel.FunctionKernel`
    (``python://`` / ``rest://`` / ``mcp://`` URIs) and stores the result in
    ``output``.
    """

    type: Literal["function"] = "function"
    tool: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    method: str = "get"


class IfNode(BaseNode):
    """Boolean branch node.

    Evaluates the ``condition`` string expression (e.g. ``state.count > 3``)
    against the run context and transitions to ``then`` when truthy, otherwise
    to ``else_`` (authored as ``else`` in YAML).
    """

    type: Literal["if"] = "if"
    condition: str
    then: str
    else_: str = Field(alias="else")

    model_config = {"populate_by_name": True}


class SwitchNode(BaseNode):
    """Multi-way branch node.

    Evaluates the ``expr`` string expression, stringifies the result and looks
    it up in ``cases``; falls back to ``default`` when no case matches.
    """

    type: Literal["switch"] = "switch"
    expr: str
    cases: dict[str, str] = {}
    default: Optional[str] = None


Node = Annotated[
    Union[
        StartNode,
        EndNode,
        LLMNode,
        AgentNode,
        FunctionNode,
        IfNode,
        SwitchNode,
    ],
    Field(discriminator="type"),
]


class WorkflowStreamEvent(BaseModel):
    """One event in a streamed workflow run (the SSE payload of
    ``POST /stream_agent``, yielded by ``WorkflowEngine.run_stream``).

    Event types and their ``name``:

    - ``workflow_started`` / ``workflow_completed`` / ``workflow_failed``:
      run lifecycle; ``name`` is the workflow name. ``workflow_started``
      carries ``session_id``/``run_id``; ``workflow_completed`` carries
      ``output_data`` and ``token_usage``; ``workflow_failed`` carries the
      error message in ``value``.
    - ``node_started`` / ``node_completed``: node lifecycle; ``name`` is the
      node name.
    - ``partial`` / ``complete``: streamed content. The node's own output
      streams under the node name; auxiliary streams are prefixed with it
      (``<node>_thought``, ``<node>_instructions``, ``<node>_step<N>``).
      Whether ``value`` is a delta or the full accumulated content depends on
      the node's ``stream_delta`` setting.
    - ``restart``: the named stream is starting over (an LLM call was retried
      after a transient error; ``value`` describes the attempt). Clients must
      discard content accumulated for streams under this name — it will be
      re-sent.
    """

    type: Literal[
        "partial",
        "complete",
        "restart",
        "node_started",
        "node_completed",
        "workflow_started",
        "workflow_completed",
        "workflow_failed",
    ]
    name: str
    value: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    output_data: Optional[dict] = None
    token_usage: Optional[dict] = None


class WorkflowGraph(BaseModel):
    """A workflow: a directed graph of nodes forming a state machine.

    Attributes:
        name: Workflow / agent name.
        description: Human-readable description.
        version: Schema version.
        llm_model: Default LLM model (``provider/model``); nodes may override.
        llm_kwargs: Default LLM kwargs; nodes may override.
        data_types: JSON-schema data type definitions (parsed by SchemaParser).
        nodes: The graph vertices; exactly one ``start`` node, and every
            ``end`` node returns the same ``output`` data type.
    """

    name: str
    description: str = ""
    version: str = "2.0"
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
    mcp_servers: list[McpServer] = []
    templates: list[TemplateModel] = []
    python_functions: list[PythonFunction] = []
    nodes: list[Node]

    @model_validator(mode="after")
    def validate_graph(self) -> "WorkflowGraph":
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate node names: {sorted(dupes)}")
        name_set = set(names)

        start_nodes = [n for n in self.nodes if isinstance(n, StartNode)]
        end_nodes = [n for n in self.nodes if isinstance(n, EndNode)]
        if len(start_nodes) != 1:
            raise ValueError(
                f"Workflow must define exactly one 'start' node, "
                f"found {len(start_nodes)}."
            )
        if not end_nodes:
            raise ValueError("Workflow must define at least one 'end' node.")

        # All end nodes must return the same output data type, so the
        # workflow's output schema (REST response model, recorded run schema)
        # is well-defined regardless of which end node terminates the run.
        end_outputs = {n.output for n in end_nodes}
        if len(end_outputs) > 1:
            raise ValueError(
                f"All 'end' nodes must return the same output data type, "
                f"found {sorted(end_outputs)}."
            )

        # Validate every transition target references an existing node.
        for node in self.nodes:
            for target in self._transition_targets(node):
                if target not in name_set:
                    raise ValueError(
                        f"Node '{node.name}' transitions to unknown node '{target}'."
                    )

        # Validate that node outputs are declared data types.
        for node in self.nodes:
            output = getattr(node, "output", None)
            if output is not None and output not in self.data_types:
                raise ValueError(
                    f"Node '{node.name}' output '{output}' is not declared in data_types."
                )

        return self

    @staticmethod
    def _transition_targets(node: "Node") -> list[str]:
        """Return the set of node names a node may transition to."""
        if isinstance(node, IfNode):
            return [node.then, node.else_]
        if isinstance(node, SwitchNode):
            targets = list(node.cases.values())
            if node.default is not None:
                targets.append(node.default)
            return targets
        if isinstance(node, EndNode):
            return []
        return [node.next]

    @property
    def start(self) -> str:
        """Name of the workflow's entry point (its single start node)."""
        return next(n.name for n in self.nodes if isinstance(n, StartNode))

    @property
    def output_type(self) -> str:
        """Name of the data type every end node returns."""
        return next(n.output for n in self.nodes if isinstance(n, EndNode))

    @property
    def node_map(self) -> dict[str, "Node"]:
        return {n.name: n for n in self.nodes}
