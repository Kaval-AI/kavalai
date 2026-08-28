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

import json
from itertools import combinations
from typing import Optional, Literal, Union, Annotated, Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from kavalai.utils import to_plain


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

    ``allowed_tools`` restricts what this node may call, and means the same
    here as it does in the Python API:

    .. code-block:: yaml

       allowed_tools: ["*"]                        # every registered tool
       allowed_tools: ["mcp://github.*"]           # one server's tools
       allowed_tools: ["python://lookup_resident"] # one tool
       allowed_tools: []                           # no tools at all
       # key omitted                               # every registered tool
    """

    type: Literal["agent"] = "agent"
    prompt: str
    inputs: dict[str, ArgumentInfo] = {}
    output: str
    next: str
    allowed_tools: Optional[list[str]] = None
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


class ParallelNode(BaseNode):
    """Fan-out node: runs several independent branches concurrently.

    Each name in ``branches`` is the **entry node of a branch** — a subgraph
    walked exactly like the main graph, up to (and not including) the join
    node named by ``next``. All branches start together, and the run continues
    at ``next`` once every branch has reached it.

    Branches each get their own copy of the run context, so a node in one
    branch cannot see another branch's outputs while they run; the outputs are
    merged back into the parent context at the join. Because of that, branches
    must be independent, which the graph validator enforces at load time:

    * branch subgraphs must be disjoint from each other,
    * no two branches may write the same ``output`` variable,
    * a branch may not contain an ``end`` node or re-enter the ``parallel``
      node itself.

    Loops, ``if``/``switch`` routing and nested ``parallel`` nodes inside a
    branch are all fine.

    .. code-block:: yaml

       - name: gather
         type: parallel
         branches: [fetch_weather, fetch_news]
         next: summarise
         max_concurrency: 4

    Attributes:
        branches: Entry node of each branch. Branches run concurrently.
        next: The join node; every branch ends by transitioning to it, and the
            run resumes there once all branches have arrived.
        max_concurrency: Cap on branches running at once (``None`` = all of
            them). Useful when the branches hit a rate-limited provider.
    """

    type: Literal["parallel"] = "parallel"
    branches: list[str]
    next: str
    max_concurrency: Optional[int] = None

    @model_validator(mode="after")
    def check_branches(self) -> "ParallelNode":
        if not self.branches:
            raise ValueError(
                f"Parallel node '{self.name}': 'branches' must name at least one node."
            )
        if len(self.branches) != len(set(self.branches)):
            dupes = {b for b in self.branches if self.branches.count(b) > 1}
            raise ValueError(
                f"Parallel node '{self.name}': duplicate branch entries "
                f"{sorted(dupes)}."
            )
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError(
                f"Parallel node '{self.name}': 'max_concurrency' must be >= 1."
            )
        return self


class RagQueryNode(BaseNode):
    """Retrieval node: queries a RAG service and stores the hits.

    **Read-only by construction.** The node calls exactly one method,
    ``BaseRagService.query``; nothing that writes to an index is reachable from
    a workflow document. Retrieval is idempotent, so a cycle revisiting the
    node is harmless, whereas indexing is a side effect whose blast radius is a
    corrupted index -- that belongs in a script or a tool, where the author is
    holding the service object.

    ``query`` is a template rendered exactly like an ``llm`` node's ``prompt``,
    so it may reference ``{{ context.* }}``, ``{{ templates.* }}`` and
    ``{{ history.* }}``.

    ``service`` and ``collection`` follow the ``llm_model`` pattern: a
    workflow-level default that a node may override. The common case is one
    service and one collection, and that case needs neither field:

    .. code-block:: yaml

       - name: retrieve
         type: rag_query
         query: "{{ context.user_message }}"
         output: docs
         next: answer

    The service is resolved as *node* → *workflow* ``rag_service`` →
    ``"default"``, looked up first among the services passed to the engine and
    then among those registered with
    :func:`~kavalai.register_rag_service`. A workflow names a service; it never
    carries a connection string, because the document is served by
    ``GET /workflow`` and edited in the backoffice.

    Attributes:
        query: Query text, rendered as a template.
        output: Context variable the hits are stored under.
        next: The node to continue at.
        service: Registered service name. Defaults to the workflow's
            ``rag_service``, then ``"default"``.
        collection: Collection to search. Defaults to the workflow's
            ``rag_collection``, then the backend's own default.
        top_k: Maximum number of hits.
        source_ids: Restrict the search to these source identifiers.
        keep_best: Keep only the best hit per ``source_id``. Useful when one
            document was indexed as many chunks.
        store: ``"results"`` (default) stores the full
            :class:`~kavalai.rag.RagServiceResult` list, so scores and metadata
            stay available for routing. ``"content"`` stores just the hit texts
            joined by blank lines, which is what a following ``llm`` node's
            prompt usually wants.
    """

    type: Literal["rag_query"] = "rag_query"
    query: str
    output: str
    next: str

    service: Optional[str] = None
    collection: Optional[str] = None
    top_k: int = 5
    source_ids: Optional[list[str]] = None
    keep_best: bool = False
    store: Literal["results", "content"] = "results"


Node = Annotated[
    Union[
        StartNode,
        EndNode,
        LLMNode,
        AgentNode,
        FunctionNode,
        IfNode,
        SwitchNode,
        ParallelNode,
        RagQueryNode,
    ],
    Field(discriminator="type"),
]


#: Node types that declare their own output model, so the author does not have
#: to name one in ``data_types``. See ``validate_graph``.
_NODES_WITH_OWN_OUTPUT_TYPE = frozenset({RagQueryNode})


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


def _validate_model_name(label: str, model: str) -> None:
    """Check a model name is well formed, and warn if nothing serves it.

    A workflow document is served by ``GET /workflow`` and edited through the
    backoffice, so it is not a trusted place to name Python. A provider
    containing a dot would look like an importable path, and accepting one
    would turn "edit a workflow" into "run code in the agent server"; it is
    rejected explicitly rather than left to fail later as an unknown provider.

    A malformed name and a Python path are hard errors. An unregistered
    provider is only a warning --- see below.
    """
    from kavalai.llm_clients import registry

    if "/" not in model:
        raise ValueError(
            f"{label} is '{model}', which is not in 'provider/model' form."
        )
    provider = model.split("/", maxsplit=1)[0]
    if "." in provider:
        raise ValueError(
            f"{label} names provider '{provider}'. A workflow names a "
            "registered provider, never an importable Python path -- register "
            "the client in code and name the registration here."
        )
    if registry.llm_providers.lookup(model) is None:
        # A warning, not an error. The name only means anything if the default
        # factory resolves it, and callers legitimately override
        # ``WorkflowEngine.client_factory`` --- sometimes after construction ---
        # to run a graph offline against a stub, naming a placeholder provider.
        # Refusing to load would break that. Surfacing it at load still puts a
        # typo in the start-up log rather than on a branch taken once a month.
        logger.warning(
            f"{label}: {registry.llm_providers.unknown_message(model)} "
            "The workflow will still load; this fails when the node runs, "
            "unless a client_factory supplies the client."
        )


class WorkflowGraph(BaseModel):
    """A workflow: a directed graph of nodes forming a state machine.

    This is the root of the YAML document a workflow is loaded from, so every
    field below is also a top-level YAML key.

    ``name``
        Workflow / agent name. Also the agent name runs are recorded under.
    ``description``
        Human-readable description.
    ``version``
        Schema version (``"2.0"``).
    ``llm_model``
        Default LLM model (``provider/model``); nodes may override it.
    ``llm_kwargs``
        Default LLM kwargs; nodes may override them.
    ``rag_service``
        Default RAG service name for ``rag_query`` nodes; nodes may override
        it. Falls back to ``"default"``.
    ``rag_collection``
        Default collection for ``rag_query`` nodes; nodes may override it.
    ``data_types``
        JSON-schema data type definitions, compiled to Pydantic models by
        ``SchemaParser``. ``input`` and ``output`` are the workflow's own
        input and output types.
    ``rest_servers`` / ``mcp_servers``
        Tool servers registered on the kernel before the run
        (see :class:`RestServer`, :class:`McpServer`).
    ``python_functions``
        Python tools to import and register by module path
        (see :class:`PythonFunction`).
    ``templates``
        Named prompt fragments reusable across nodes
        (see :class:`TemplateModel`).
    ``nodes``
        The graph vertices; exactly one ``start`` node, and every ``end``
        node returns the same ``output`` data type.
    """

    name: str
    description: str = ""
    version: str = "2.0"
    llm_model: Optional[str] = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    rag_service: Optional[str] = None
    rag_collection: Optional[str] = None
    data_types: dict[str, dict]
    rest_servers: list[RestServer] = []
    mcp_servers: list[McpServer] = []
    templates: list[TemplateModel] = []
    python_functions: list[PythonFunction] = []
    nodes: list[Node]

    REDACTED: ClassVar[str] = "***"

    def model_dump_public(self) -> dict[str, Any]:
        """Serialise the graph with secrets removed.

        ``McpServer.env`` is a plain mapping merged into the tool server's
        process environment, so in practice it is where an API key for a stdio
        MCP server ends up. Anything that hands a workflow definition to a
        caller — ``GET /workflow``, a UI rendering the graph — should use this
        rather than :meth:`model_dump`, and doing the redaction on the model
        means no such caller can leak by forgetting.

        Keys are kept, values replaced: which variables a server needs is
        useful to see, what they are set to is not.
        """
        data = self.model_dump()
        for server in data.get("mcp_servers", []):
            if server.get("env"):
                server["env"] = {key: self.REDACTED for key in server["env"]}
        return data

    def model_dump_public_json(self) -> str:
        """:meth:`model_dump_public`, JSON-encoded."""
        return json.dumps(to_plain(self.model_dump_public()))

    def _named_models(self) -> list[tuple[str, str]]:
        """Every ``provider/model`` string in the document, with its location."""
        found = []
        if self.llm_model:
            found.append(("workflow 'llm_model'", self.llm_model))
        for node in self.nodes:
            model = getattr(node, "llm_model", None)
            if model:
                found.append((f"node '{node.name}' llm_model", model))
        return found

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

        for node in self.nodes:
            for target in self._transition_targets(node):
                if target not in name_set:
                    raise ValueError(
                        f"Node '{node.name}' transitions to unknown node '{target}'."
                    )

        # Validate that node outputs are declared data types. Nodes whose
        # output shape Kaval.AI owns rather than the author are exempt: a
        # `rag_query` node returns hits or joined text, and no author could
        # reasonably write that as a JSON-schema fragment. The boundary is
        # still typed -- by a model we ship instead of one they wrote.
        for node in self.nodes:
            output = getattr(node, "output", None)
            if output is None or type(node) in _NODES_WITH_OWN_OUTPUT_TYPE:
                continue
            if output not in self.data_types:
                raise ValueError(
                    f"Node '{node.name}' output '{output}' is not declared in data_types."
                )

        # A rag_query node names a service; it never carries a connection.
        # The document is served by GET /workflow and edited in the backoffice,
        # so a DSN here would put credentials somewhere they must not be.
        for node in self.nodes:
            service = getattr(node, "service", None)
            if service and ("://" in service or "@" in service):
                raise ValueError(
                    f"Node '{node.name}' service '{service}' looks like a "
                    "connection string. A workflow names a registered RAG "
                    "service; pass the connection to the service itself, in "
                    "code or at registration."
                )

        # Every model named must resolve to a registered provider. Without
        # this a typo loads, renders and diagrams cleanly, then raises the
        # first time that node runs -- possibly on a branch taken once a month.
        for label, model in self._named_models():
            _validate_model_name(label, model)

        # Branches of a 'parallel' node run concurrently on private copies of
        # the run context, so they have to be independent. That is checkable
        # here, once, instead of racing at run time.
        for node in self.nodes:
            if isinstance(node, ParallelNode):
                self._validate_parallel(node)

        return self

    def _validate_parallel(self, node: "ParallelNode") -> None:
        """Check that a ``parallel`` node's branches can safely run together."""
        node_map = self.node_map
        start_name = self.start
        reach: dict[str, list[str]] = {}

        for entry in node.branches:
            if entry == node.next:
                raise ValueError(
                    f"Parallel node '{node.name}': branch '{entry}' is also the "
                    f"join node ('next'), so the branch is empty."
                )
            branch = self._reachable(entry, stop_at=node.next, node_map=node_map)
            reach[entry] = branch

            if node.name in branch:
                raise ValueError(
                    f"Parallel node '{node.name}': branch '{entry}' loops back "
                    f"into the parallel node itself. A branch must rejoin at "
                    f"'{node.next}'."
                )
            if start_name in branch:
                raise ValueError(
                    f"Parallel node '{node.name}': branch '{entry}' reaches the "
                    f"start node '{start_name}'."
                )
            ends = [n for n in branch if isinstance(node_map[n], EndNode)]
            if ends:
                raise ValueError(
                    f"Parallel node '{node.name}': branch '{entry}' can reach "
                    f"end node '{ends[0]}'. A branch must rejoin at "
                    f"'{node.next}'; only the main path may end the run."
                )

        # Disjoint subgraphs: a node shared by two branches would execute twice,
        # concurrently, against two different contexts.
        for left, right in combinations(node.branches, 2):
            shared = sorted(set(reach[left]) & set(reach[right]))
            if shared:
                raise ValueError(
                    f"Parallel node '{node.name}': branches '{left}' and "
                    f"'{right}' share node(s) {shared}. Branch subgraphs "
                    f"must be disjoint."
                )

        # Disjoint writes: two branches writing one context variable is a race
        # whose winner depends on which branch finishes last.
        writes = {
            entry: {
                output
                for n in names
                if (output := getattr(node_map[n], "output", None)) is not None
            }
            for entry, names in reach.items()
        }
        for left, right in combinations(node.branches, 2):
            clash = sorted(writes[left] & writes[right])
            if clash:
                raise ValueError(
                    f"Parallel node '{node.name}': branches '{left}' and "
                    f"'{right}' both write {clash}. Each branch must write "
                    f"its own output variable."
                )

    def _reachable(
        self, entry: str, *, stop_at: str, node_map: dict[str, "Node"]
    ) -> list[str]:
        """Node names reachable from ``entry`` without passing through ``stop_at``."""
        seen: list[str] = []
        pending = [entry]
        while pending:
            current = pending.pop()
            if current == stop_at or current in seen:
                continue
            seen.append(current)
            pending.extend(self._transition_targets(node_map[current]))
        return seen

    @staticmethod
    def _transition_targets(node: "Node") -> list[str]:
        """Return the node names a node may transition to."""
        if isinstance(node, IfNode):
            return [node.then, node.else_]
        if isinstance(node, SwitchNode):
            targets = list(node.cases.values())
            if node.default is not None:
                targets.append(node.default)
            return targets
        if isinstance(node, ParallelNode):
            return [*node.branches, node.next]
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
