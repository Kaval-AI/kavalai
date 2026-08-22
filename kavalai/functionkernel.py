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
import inspect
import json
from loguru import logger
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Type,
    Union,
    get_args,
    get_origin,
)

import httpx
from pydantic import BaseModel, create_model

# ``mcp`` is an optional extra: it is not pyodide-compatible and pulls in a
# large dependency tree. Import it lazily so the core library imports without
# it; only MCP tool calls require ``pip install kavalai[common]``.
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - exercised in pyodide / minimal installs
    ClientSession = None
    StdioServerParameters = None
    sse_client = None
    stdio_client = None

from kavalai.workflow.models import (
    McpServer,
    RestServer,
    WorkflowException,
)

SSE_CLIENT_TIMEOUT_SECONDS = 30.0


class FunctionKernelException(WorkflowException):
    """Raised by :class:`FunctionKernel` when registering or calling a tool fails.

    Covers errors such as unknown protocols, duplicate or unregistered tools,
    malformed tool URIs, argument validation failures and errors surfaced by
    the underlying REST/MCP/Python tool call.
    """


def pythontool(func: Callable) -> Callable:
    """Mark a function as a Kaval.AI Python tool.

    Sets an internal flag (``_is_kavalai_tool``) on the function without
    altering its behaviour, so the decorated function remains directly callable
    as normal Python. Register it with
    :meth:`FunctionKernel.register_python_tool` to expose it to workflows, after
    which it is addressed by the tool URI ``python://<name>``.

    Args:
        func: The function to mark as a tool.

    Returns:
        The same function, flagged as a Kaval.AI tool.
    """
    func._is_kavalai_tool = True
    return func


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]

    def to_dict(self, name_override: Optional[str] = None) -> Dict[str, Any]:
        description_text = self.description or ""
        try:
            desc_data = json.loads(self.description)
            description_text = desc_data.get("description", "")
        except Exception:
            pass

        return {
            "name": name_override or self.name,
            "description": description_text,
            "inputSchema": _get_schema(self.input_model),
            "outputSchema": _get_schema(self.output_model),
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


class Validator:
    """
    Helps converting and validating tool inputs and outputs.
    Allows only basic data types and pydantic models.
    """

    @staticmethod
    def create_model_from_signature(
        name: str, sig: inspect.Signature, is_input: bool = True
    ) -> Type[BaseModel]:
        if is_input:
            input_fields = {}
            for param_name, p in sig.parameters.items():
                annotation = (
                    p.annotation if p.annotation != inspect.Parameter.empty else Any
                )
                default = p.default if p.default != inspect.Parameter.empty else ...
                input_fields[param_name] = (annotation, default)
            return create_model(f"{name}_input", **input_fields)
        else:
            output_annotation = (
                sig.return_annotation
                if sig.return_annotation != inspect.Signature.empty
                else Any
            )
            if inspect.isclass(output_annotation) and issubclass(
                output_annotation, BaseModel
            ):
                return output_annotation
            else:
                return create_model(f"{name}_output", result=(output_annotation, ...))

    @staticmethod
    def create_model_from_schema(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        return _create_model_from_jsonschema(name, schema)

    @staticmethod
    def validate_arguments(
        model: Type[BaseModel], arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        return model(**arguments).model_dump()

    @staticmethod
    def cast_result(
        result: Any, target_output_type: Optional[Type], context_info: str = ""
    ) -> Any:
        """Cast a tool's return value to the model that describes it.

        Tools that declare a Pydantic return type are validated against it.
        Everything else — ``-> str``, ``-> int``, ``-> dict``, an MCP payload —
        is wrapped in a generated single-field model, so every caller reads the
        value from ``.result`` regardless of what the tool returned.

        Raises:
            FunctionKernelException: if the value cannot satisfy the model. It
                used to be logged and the raw value returned instead, which
                handed the caller a different type than the tool's own
                signature promised, quietly.
        """
        if not (
            target_output_type
            and inspect.isclass(target_output_type)
            and issubclass(target_output_type, BaseModel)
        ):
            return result

        try:
            if isinstance(result, target_output_type):
                return result

            # A generated wrapper takes the value *as* its field. Checking this
            # before the dict branch is what stops a `-> dict` tool having its
            # payload expanded into keyword arguments, which no wrapper can
            # accept and which used to fail validation silently.
            if list(target_output_type.model_fields) == ["result"]:
                return target_output_type(result=result)

            if isinstance(result, dict):
                return target_output_type(**result)
            if isinstance(result, BaseModel):
                return target_output_type(**result.model_dump())

            fields = target_output_type.model_fields
            if len(fields) == 1:
                field_name = next(iter(fields))
                return target_output_type(**{field_name: result})

            return target_output_type(result)
        except Exception as e:
            raise FunctionKernelException(
                f"{context_info} returned a result that does not match "
                f"{target_output_type.__name__}: {e}"
            ) from e


class FunctionKernel:
    """
    Manages registration and execution of tools (REST, MCP, Python).
    Handles conversions and MCP session management.
    """

    def __init__(self):
        self.rest_servers: Dict[str, RestServer] = {}
        self.rest_tool_definitions: Dict[str, Dict[str, ToolDefinition]] = {}
        self.mcp_servers: Dict[str, McpServer] = {}
        self.python_tools: Dict[str, Callable] = {}
        self.python_tool_definitions: Dict[str, ToolDefinition] = {}
        self.mcp_tool_definitions: Dict[str, Dict[str, ToolDefinition]] = {}

        # MCP session management
        self.mcp_sessions: Dict[str, "ClientSession"] = {}
        self.mcp_cleanups: List[Any] = []
        # One lock per server so two concurrent first-calls cannot each spawn a
        # process and leak one of them.
        self._mcp_connect_locks: Dict[str, asyncio.Lock] = {}

    def register_rest_server(self, server: RestServer):
        if server.name in self.rest_servers:
            raise FunctionKernelException(
                f"REST server '{server.name}' is already registered."
            )
        self.rest_servers[server.name] = server

    def register_rest_tool(
        self,
        server_name: str,
        tool_name: str,
        method: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        description: Optional[str] = None,
    ):
        if server_name not in self.rest_tool_definitions:
            self.rest_tool_definitions[server_name] = {}

        InputModel = Validator.create_model_from_schema(
            f"{server_name}_{tool_name}_input", input_schema
        )
        OutputModel = Validator.create_model_from_schema(
            f"{server_name}_{tool_name}_output", output_schema
        )

        self.rest_tool_definitions[server_name][tool_name] = ToolDefinition(
            name=tool_name,
            description=description,
            input_model=InputModel,
            output_model=OutputModel,
        )
        # Store method in a way it can be retrieved
        self.rest_tool_definitions[server_name][tool_name].description = json.dumps(
            {"method": method, "description": description}
        )

    def register_mcp_server(self, server: McpServer):
        if server.name in self.mcp_servers:
            raise FunctionKernelException(
                f"MCP server '{server.name}' is already registered."
            )
        self.mcp_servers[server.name] = server

    def register_python_tool(self, name: str, func: Callable):
        if not getattr(func, "_is_kavalai_tool", False):
            raise FunctionKernelException(
                f"Function '{func.__name__}' must be decorated with @kavalai.pythontool"
            )

        # Normalize name by removing protocol prefix if present
        if "://" in name:
            protocol, name = name.split("://", 1)
            if protocol != "python":
                raise FunctionKernelException(
                    f"Invalid protocol '{protocol}' for Python tool '{name}'"
                )

        if name in self.python_tools:
            raise FunctionKernelException(
                f"Python tool '{name}' is already registered."
            )
        self.python_tools[name] = func
        self.python_tool_definitions[name] = self._generate_python_tool_definition(
            name, func
        )

    def _generate_python_tool_definition(
        self, name: str, func: Callable
    ) -> ToolDefinition:
        sig = inspect.signature(func)

        InputModel = Validator.create_model_from_signature(name, sig, is_input=True)
        OutputModel = Validator.create_model_from_signature(name, sig, is_input=False)

        return ToolDefinition(
            name=name,
            description=inspect.getdoc(func) or "",
            input_model=InputModel,
            output_model=OutputModel,
        )

    async def call_tool(
        self,
        tool_uri: str,
        arguments: Dict[str, Any] = None,
        output_type: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """
        Unified tool call interface.
        Format: protocol://[name|module].function_name
        Example: python://kavalai.mytool.myfunc or rest://myrestserver.restfunction
        """
        if arguments is None:
            arguments = {}
        arguments = _strip_metadata(arguments)
        protocol, path = _parse_tool_uri(tool_uri)

        if protocol == "python":
            return await self._call_python_tool(path, arguments, output_type)

        if protocol == "rest" or protocol == "mcp":
            if "." not in path:
                raise FunctionKernelException(
                    f"Invalid tool path format: '{path}'. Expected [name|module].function_name"
                )
            name_or_module, function_name = path.rsplit(".", 1)
            # Strip HTTP method annotation if present, e.g. "forecast [GET]" → "forecast"
            if " [" in function_name:
                function_name = function_name[: function_name.index(" [")].strip()

            if protocol == "rest":
                return await self._handle_rest_call(
                    name_or_module, function_name, arguments, output_type, **kwargs
                )
            if protocol == "mcp":
                return await self._call_mcp_tool(
                    name_or_module, function_name, arguments, output_type
                )

        raise FunctionKernelException(f"Unsupported protocol: '{protocol}'")

    async def _handle_rest_call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
        **kwargs,
    ) -> Any:
        """Handle REST tool calls with validation if definition exists."""
        method = kwargs.get("method", "get")
        if (
            server_name in self.rest_tool_definitions
            and tool_name in self.rest_tool_definitions[server_name]
        ):
            definition = self.rest_tool_definitions[server_name][tool_name]
            try:
                desc_data = json.loads(definition.description)
                method = desc_data.get("method", method)
            except Exception:
                pass

            # Validate input
            validated_args = Validator.validate_arguments(
                definition.input_model, arguments
            )
            return await self._call_rest_tool(
                server_name,
                tool_name,
                validated_args,
                method,
                output_type or definition.output_model,
            )

        return await self._call_rest_tool(
            server_name, tool_name, arguments, method, output_type
        )

    async def close(self):
        """Cleanup all MCP sessions."""
        for session in reversed(self.mcp_cleanups):
            try:
                # Some are async context managers, some are ClientSessions
                if hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during MCP cleanup: {e}")
        self.mcp_sessions.clear()
        self.mcp_cleanups.clear()

    async def _call_rest_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        method: str = "get",
        output_type: Optional[type] = None,
    ) -> Any:
        if server_name not in self.rest_servers:
            raise FunctionKernelException(
                f"REST server '{server_name}' not registered."
            )

        server = self.rest_servers[server_name]
        url = server.url
        if not url and server.url_env:
            url = os.environ.get(server.url_env)

        if not url:
            raise FunctionKernelException(
                f"URL for REST server '{server_name}' not found."
            )

        auth = None
        if server.username_env and server.password_env:
            username = os.environ.get(server.username_env)
            password = os.environ.get(server.password_env)
            if username and password:
                auth = (username, password)

        async with httpx.AsyncClient(auth=auth) as client:
            kwargs = {
                "params": arguments,
                "timeout": 60.0,
            }
            if method.lower() in ("post", "put", "patch"):
                kwargs["json"] = arguments
                kwargs.pop("params", None)

            full_url = f"{url.rstrip('/')}/{tool.lstrip('/')}"
            logger.info(f"Calling {method.upper()} {full_url}")
            response = await client.request(method.upper(), full_url, **kwargs)
            response.raise_for_status()
            result_data = response.json()

            return Validator.cast_result(
                result_data, output_type, f"REST tool '{server_name}.{tool}'"
            )

    async def connect_mcp_servers(self) -> None:
        """Open a session to every registered MCP server and list its tools.

        MCP servers are the only tool backend whose tools are not known from
        configuration alone — they have to be asked. Until that happens
        :meth:`get_tool_descriptions` cannot mention them, and an agent handed a
        freshly-registered server is told it has no tools and answers without
        them. Connecting up front closes that window, and turns a misconfigured
        server into an error before any model tokens are spent.

        Idempotent: servers that are already connected are left alone.
        """
        for server_name in list(self.mcp_servers):
            if server_name not in self.mcp_sessions:
                await self._get_mcp_session(server_name)

    async def _get_mcp_session(self, server_name: str) -> "ClientSession":
        if ClientSession is None:
            raise FunctionKernelException(
                "MCP support requires the optional 'mcp' dependency. "
                "Install it with: pip install kavalai[common]"
            )

        if server_name in self.mcp_sessions:
            return self.mcp_sessions[server_name]

        if server_name not in self.mcp_servers:
            raise FunctionKernelException(f"MCP server '{server_name}' not registered.")

        lock = self._mcp_connect_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            # Another caller may have connected while we waited for the lock.
            if server_name in self.mcp_sessions:
                return self.mcp_sessions[server_name]
            return await self._open_mcp_session(server_name)

    async def _open_mcp_session(self, server_name: str) -> "ClientSession":
        config = self.mcp_servers[server_name]

        if config.url or config.url_env:
            url = config.url
            if not url and config.url_env:
                url = os.environ.get(config.url_env)

            if not url:
                raise FunctionKernelException(
                    f"URL for MCP server '{server_name}' not found."
                )

            logger.info(f"Connecting to HTTP MCP server {server_name} at {url}")
            aclient = sse_client(url, timeout=SSE_CLIENT_TIMEOUT_SECONDS)
            read, write = await aclient.__aenter__()
            self.mcp_cleanups.append(aclient)
        else:
            command = config.command
            if not command and config.command_env:
                command = os.environ.get(config.command_env)

            if not command:
                raise FunctionKernelException(
                    f"Command for MCP server '{server_name}' not found."
                )

            server_params = StdioServerParameters(
                command=command,
                args=config.args,
                env={**os.environ, **config.env},
            )
            logger.info(f"Connecting to stdio MCP server {server_name}")
            aclient = stdio_client(server_params)
            read, write = await aclient.__aenter__()
            self.mcp_cleanups.append(aclient)

        session = ClientSession(read, write)
        await session.__aenter__()
        self.mcp_cleanups.append(session)
        await session.initialize()
        self.mcp_sessions[server_name] = session

        # Fetch and store tool definitions
        await self._refresh_mcp_tool_definitions(server_name, session)

        return session

    async def _refresh_mcp_tool_definitions(
        self, server_name: str, session: "ClientSession"
    ):
        try:
            tools_result = await session.list_tools()
            definitions = {}
            for tool in tools_result.tools:
                # MCP tool input schema is usually a JSON Schema
                # For now, we store the raw schema and we could dynamically create a Pydantic model
                # But to stay consistent with the "Pydantic models for everything" requirement:
                input_model = Validator.create_model_from_schema(
                    f"{server_name}_{tool.name}_input", tool.inputSchema
                )
                # MCP doesn't strictly define output schema in tool list, so we use a generic one
                output_model = create_model(
                    f"{server_name}_{tool.name}_output", result=(Any, ...)
                )

                definitions[tool.name] = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    input_model=input_model,
                    output_model=output_model,
                )
            self.mcp_tool_definitions[server_name] = definitions
        except Exception as e:
            # Swallowing this used to leave the server connected but toolless,
            # which reaches the model as "you have no tools" — a wrong answer
            # rather than an error.
            raise FunctionKernelException(
                f"Could not list tools for MCP server '{server_name}': {e}"
            ) from e

    async def _call_mcp_tool(
        self,
        server_name: str,
        tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        session = await self._get_mcp_session(server_name)

        # Validate arguments if definition exists
        if (
            server_name in self.mcp_tool_definitions
            and tool in self.mcp_tool_definitions[server_name]
        ):
            definition = self.mcp_tool_definitions[server_name][tool]
            try:
                arguments = Validator.validate_arguments(
                    definition.input_model, arguments
                )
            except Exception as e:
                raise FunctionKernelException(
                    f"MCP tool '{tool}' on server '{server_name}' argument validation failed: {e}"
                )

        logger.info(f"Calling MCP tool {server_name}/{tool}")
        response = await session.call_tool(tool, arguments=arguments)

        if response.isError:
            raise FunctionKernelException(
                f"MCP tool '{tool}' on server '{server_name}' failed: {response.content}"
            )

        result_data = None
        for content in response.content:
            if hasattr(content, "text"):
                try:
                    result_data = json.loads(content.text)
                except (json.JSONDecodeError, TypeError):
                    result_data = content.text
                break

        # Convert output using output_type or definition's output_model
        target_output_type = output_type
        if not target_output_type and server_name in self.mcp_tool_definitions:
            if tool in self.mcp_tool_definitions[server_name]:
                target_output_type = self.mcp_tool_definitions[server_name][
                    tool
                ].output_model

        return Validator.cast_result(
            result_data, target_output_type, f"MCP tool '{server_name}.{tool}'"
        )

    async def _call_python_tool(
        self,
        python_tool: str,
        arguments: Dict[str, Any],
        output_type: Optional[type] = None,
    ) -> Any:
        if python_tool in self.python_tools:
            func = self.python_tools[python_tool]
            definition = self.python_tool_definitions.get(python_tool)
        else:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' not registered."
            )

        # Validate arguments using input model
        try:
            call_args = Validator.validate_arguments(definition.input_model, arguments)

            # Ensure complex Pydantic types are passed as model instances if
            # needed. ``Optional[Model]`` counts: a tool whose parameter may
            # legitimately be absent still wants the model when it is present,
            # and leaving it a dict fails later, inside the tool, with a
            # confusing error about the dict's own methods.
            for param_name, p in inspect.signature(func).parameters.items():
                model = _model_annotation(p.annotation)
                if model is not None and call_args.get(param_name) is not None:
                    value = call_args[param_name]
                    if not isinstance(value, model):
                        call_args[param_name] = model.model_validate(value)
        except Exception as e:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' argument validation failed: {e}"
            )

        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(**call_args)
        except TypeError as e:
            raise FunctionKernelException(
                f"Python tool '{python_tool}' signature mismatch: {e}"
            )

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*bound_args.args, **bound_args.kwargs)
            else:
                result = func(*bound_args.args, **bound_args.kwargs)
        except Exception as e:
            logger.exception(f"Error executing python_tool '{python_tool}'")
            raise FunctionKernelException(
                f"Error executing python_tool '{python_tool}': {e}"
            )

        target_output_type = output_type or definition.output_model
        return Validator.cast_result(
            result, target_output_type, f"Python tool '{python_tool}'"
        )

    def get_tool_definition(self, tool_uri: str) -> ToolDefinition:
        """Resolve a tool URI to its ToolDefinition.

        Format: protocol://[name|module].function_name
        Raises FunctionKernelException if the tool is not registered.
        """
        protocol, path = _parse_tool_uri(tool_uri)

        if protocol == "python":
            definition = self.python_tool_definitions.get(path)
            if definition is None:
                raise FunctionKernelException(f"Python tool '{path}' not registered.")
            return definition

        if protocol in ("rest", "mcp"):
            if "." not in path:
                raise FunctionKernelException(
                    f"Invalid tool path format: '{path}'. Expected [name|module].function_name"
                )
            server_name, function_name = path.rsplit(".", 1)
            definitions = (
                self.rest_tool_definitions
                if protocol == "rest"
                else self.mcp_tool_definitions
            )
            server_tools = definitions.get(server_name)
            if not server_tools or function_name not in server_tools:
                raise FunctionKernelException(
                    f"{protocol.upper()} tool '{function_name}' on server "
                    f"'{server_name}' not registered."
                )
            return server_tools[function_name]

        raise FunctionKernelException(f"Unsupported protocol: '{protocol}'")

    def get_input_model(self, tool_uri: str) -> Type[BaseModel]:
        """Return the input Pydantic model for a registered tool."""
        return self.get_tool_definition(tool_uri).input_model

    def get_output_model(self, tool_uri: str) -> Type[BaseModel]:
        """Return the output Pydantic model for a registered tool."""
        return self.get_tool_definition(tool_uri).output_model

    async def get_tool_descriptions(
        self, allowed_tools: Optional[List[str]] = None
    ) -> str:
        """Returns a string description of all registered tools as a JSON array for prompts."""
        # MCP tools are only known once their server has been asked, so connect
        # anything still unconnected before describing what is available.
        # Without this an agent is told it has no MCP tools and answers without
        # them — silently, and plausibly.
        await self.connect_mcp_servers()

        tools_list = []

        # Python tools
        for name, definition in self.python_tool_definitions.items():
            tool_uri = f"python://{name}"
            if _is_tool_allowed(tool_uri, allowed_tools):
                tools_list.append(definition.to_dict(name_override=tool_uri))

        # REST tools
        for server_name, tools in self.rest_tool_definitions.items():
            for tool_name, definition in tools.items():
                method = "GET"
                try:
                    desc_data = json.loads(definition.description)
                    method = desc_data.get("method", "GET").upper()
                except Exception:
                    pass

                tool_uri = f"rest://{server_name}.{tool_name} [{method}]"
                if _is_tool_allowed(tool_uri, allowed_tools):
                    tools_list.append(definition.to_dict(name_override=tool_uri))

        # MCP tools
        for server_name, tools in self.mcp_tool_definitions.items():
            for tool_name, definition in tools.items():
                tool_uri = f"mcp://{server_name}.{tool_name}"
                if _is_tool_allowed(tool_uri, allowed_tools):
                    tools_list.append(definition.to_dict(name_override=tool_uri))

        return json.dumps(tools_list, indent=2)


def _strip_metadata(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strip metadata from arguments."""
    return {
        k: v for k, v in arguments.items() if k not in ("__line__", "__file_path__")
    }


def _parse_tool_uri(tool_uri: str) -> tuple[str, str]:
    """Parse tool URI into protocol and path."""
    if "://" not in tool_uri:
        raise FunctionKernelException(
            f"Invalid tool URI format: '{tool_uri}'. Expected protocol://[name|module].function_name"
        )
    protocol, path = tool_uri.split("://", 1)
    return protocol, path


def _create_model_from_jsonschema(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Very basic JSON Schema to Pydantic model conversion."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    fields = {}

    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "any")
        python_type = type_map.get(prop_type, Any)
        default = ... if prop_name in required else None
        fields[prop_name] = (python_type, default)

    return create_model(name, **fields)


def _get_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema_dict = model.model_json_schema()
    # Remove pydantic-specific keys to keep it cleaner for LLM
    schema_dict.pop("title", None)
    schema_dict.pop("type", None)
    return schema_dict


def _is_tool_allowed(name: str, allowed_tools: Optional[List[str]]) -> bool:
    """Check whether one tool URI passes an allow-list.

    ``None`` allows everything, ``[]`` allows nothing, and the list itself may
    hold exact URIs (``python://lookup_resident``), a whole server
    (``mcp://github.*``) or ``*`` for every tool. The wildcard is worth spelling
    out even though an absent list means the same: in a review, a permissive
    default and a deleted line look identical.
    """
    if allowed_tools is None:
        return True

    if "*" in allowed_tools:
        return True

    if name in allowed_tools:
        return True

    # Checks for dynamic rest/mcp server and if the server is allowed
    # e.g. rest://server.* or mcp://server.*
    if "." in name:
        server_prefix = name.split(".")[0] + ".*"
        if server_prefix in allowed_tools:
            return True

    return False


def _model_annotation(annotation: Any) -> Optional[Type[BaseModel]]:
    """The Pydantic model in an annotation, unwrapping ``Optional``/``Union``.

    Returns ``None`` when the annotation is not (or does not contain) a model.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _add_tool_to_list(
    tools_list: List[Dict[str, Any]],
    name: str,
    description: str,
    input_model: Type[BaseModel],
    allowed_tools: Optional[List[str]],
):
    """Add tool to list if it's allowed."""
    if _is_tool_allowed(name, allowed_tools):
        tools_list.append(
            {
                "name": name,
                "description": description,
                "inputSchema": _get_schema(input_model),
            }
        )
