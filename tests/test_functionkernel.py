from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import asyncio
import json
import uvicorn
import multiprocessing
import time
import os
import socket


from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model
from typing import Any, Dict, Optional
from kavalai.functionkernel import (
    FunctionKernel,
    pythontool,
    ToolDefinition,
    FunctionKernelException,
)
from kavalai.workflow.models import RestServer, McpServer


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


FREE_PORT = find_free_port()

app = FastAPI()


class SimpleModel(BaseModel):
    name: str
    value: int


@app.get("/get_item")
async def get_item(id: int):
    return {"name": "rest_test", "value": 100}


@app.post("/create_item")
async def create_item(item: Dict[str, Any]):
    return {"name": "rest_test", "value": 100}


@app.put("/update_item")
async def update_item(item: Dict[str, Any]):
    return {"name": "updated", "value": 200}


@app.patch("/patch_item")
async def patch_item(item: Dict[str, Any]):
    return {"name": "patched", "value": 300}


@app.delete("/delete_item")
async def delete_item(id: int):
    return {"status": "deleted"}


@app.get("/auth_check")
async def auth_check(request: Request):
    auth = request.headers.get("Authorization")
    if auth == "Basic dGVzdF91c2VyOnRlc3RfcGFzcw==":  # test_user:test_pass
        return {"status": "authenticated"}
    return Response(status_code=401)


sse_queues: Dict[str, asyncio.Queue] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    session_id = "123"
    q = asyncio.Queue()
    sse_queues[session_id] = q

    async def event_generator():
        # MCP SSE handshake
        # 1. Send endpoint event
        yield f"event: endpoint\ndata: http://127.0.0.1:{FREE_PORT}/messages?session_id=123\n\n"

        # Keep alive and send messages from queue
        while True:
            try:
                if await request.is_disconnected():
                    break

                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield ":\n\n"
            except Exception:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/messages")
async def sse_messages(request: Request, session_id: Optional[str] = None):
    # Handle JSON-RPC over POST
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    response_body = None
    if data.get("method") == "initialize":
        response_body = {
            "jsonrpc": "2.0",
            "id": data["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-sse", "version": "1.0"},
            },
        }
    elif data.get("method") == "notifications/initialized":
        return Response(status_code=202)
    elif data.get("method") == "tools/list":
        response_body = {
            "jsonrpc": "2.0",
            "id": data["id"],
            "result": {
                "tools": [
                    {
                        "name": "hello",
                        "description": "Greet",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }
    elif data.get("method") == "tools/call":
        response_body = {
            "jsonrpc": "2.0",
            "id": data["id"],
            "result": {
                "content": [{"type": "text", "text": json.dumps({"result": "ok"})}],
                "isError": False,
            },
        }

    if response_body and session_id in sse_queues:
        await sse_queues[session_id].put(response_body)
        return Response(status_code=202)

    return {"jsonrpc": "2.0", "id": data.get("id"), "result": {}}


def run_server(port):
    uvicorn.run(app, host="127.0.0.1", port=port)


@pytest.fixture(scope="module")
def rest_server():
    proc = multiprocessing.Process(target=run_server, args=(FREE_PORT,))
    proc.start()
    time.sleep(1)  # Wait for server to start
    yield f"http://127.0.0.1:{FREE_PORT}"
    proc.terminate()


class CustomOutput(BaseModel):
    result: int
    meta: str


# Simple functions for testing
@pythontool
def sync_add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b


@pythontool
async def async_multiply(a: int, b: int) -> int:
    """Multiplies two integers asynchronously."""
    await asyncio.sleep(0.01)
    return a * b


@pythontool
def dict_output(name: str) -> Dict[str, Any]:
    return {"name": name, "value": 42}


@pythontool
def primitive_output(val: int) -> int:
    return val * 10


@pythontool
def nested_dict_output() -> Dict[str, Any]:
    return {"data": {"nested": "value"}}


@pythontool
def list_output() -> list:
    return [1, 2, 3]


@pythontool
def raise_error():
    raise ValueError("Tool execution error")


@pythontool
def custom_model_output(val: int) -> CustomOutput:
    return CustomOutput(result=val * 2, meta="test")


@pytest.mark.asyncio
async def test_register_and_call_python_tool():
    """
    Tests the registration and calling of Python-based tools.
    Covers:
    - Synchronous and asynchronous function registration.
    - Result mapping to default models (with 'result' field).
    - Result mapping to custom Pydantic models (output_type).
    - Mapping primitive outputs to single-field Pydantic models.
    - Handling of mapping failures (returning original results).
    """
    kernel = FunctionKernel()
    kernel.register_python_tool("add", sync_add)
    kernel.register_python_tool("mul", async_multiply)

    # Sync call - returns model with 'result' field
    result = await kernel._call_python_tool("add", {"a": 1, "b": 2})
    assert result.result == 3

    # Async call
    result = await kernel._call_python_tool("mul", {"a": 3, "b": 4})
    assert result.result == 12

    # With output_type
    kernel.register_python_tool("dict_tool", dict_output)
    result = await kernel._call_python_tool(
        "dict_tool", {"name": "test"}, output_type=SimpleModel
    )
    assert isinstance(result, SimpleModel)
    assert result.name == "test"
    assert result.value == 42

    # Return BaseModel directly
    kernel.register_python_tool("custom", custom_model_output)
    result = await kernel._call_python_tool(
        "custom", {"val": 10}, output_type=CustomOutput
    )
    assert isinstance(result, CustomOutput)
    assert result.result == 20

    # Primitive output with single field model
    kernel.register_python_tool("primitive", primitive_output)

    class OneField(BaseModel):
        val: int

    result = await kernel._call_python_tool(
        "primitive", {"val": 5}, output_type=OneField
    )
    assert isinstance(result, OneField)
    assert result.val == 50

    # Incompatible output type (mapping failure)
    result = await kernel._call_python_tool(
        "primitive", {"val": 5}, output_type=SimpleModel
    )
    assert result == 50  # Should return original result on failure

    # List output
    kernel.register_python_tool("list_tool", list_output)
    result = await kernel._call_python_tool("list_tool", {})
    # It returns a model with 'result' field if not specified otherwise
    assert result.result == [1, 2, 3]

    # Python tool mapping exception (line 529)
    @pythontool
    def dict_tool() -> dict:
        return {"a": 1}

    kernel.register_python_tool("dict", dict_tool)

    class MultiField(BaseModel):
        x: int
        y: int

    # Mapping {"a": 1} to MultiField should fail and return original dict
    result = await kernel._call_python_tool("dict", {}, output_type=MultiField)
    assert result == {"a": 1}


@pytest.mark.asyncio
async def test_python_tool_custom_model():
    """
    Tests that Python tools returning custom Pydantic models are correctly handled.
    Verifies that the ToolDefinition captures the output model and that
    calls return the model instance directly.
    """
    kernel = FunctionKernel()
    kernel.register_python_tool("custom", custom_model_output)

    definition = kernel.python_tool_definitions["custom"]
    assert definition.output_model == CustomOutput

    result = await kernel._call_python_tool("custom", {"val": 5})
    assert isinstance(result, CustomOutput)
    assert result.result == 10
    assert result.meta == "test"


@pytest.mark.asyncio
async def test_python_tool_errors():
    """
    Tests error handling for Python tool execution and registration.
    Covers:
    - Attempting to call non-existent tools.
    - Argument validation failures (missing required arguments).
    - Signature mismatch between tool definition and actual function.
    - Enforcement of the @pythontool decorator during registration.
    - Rejection of dynamic loading for undecorated functions.
    """
    kernel = FunctionKernel()

    # Missing tool
    with pytest.raises(FunctionKernelException, match="not registered"):
        await kernel._call_python_tool("non.existent.tool", {})

    # Signature mismatch / validation error
    kernel.register_python_tool("add", sync_add)
    with pytest.raises(FunctionKernelException, match="argument validation failed"):
        await kernel._call_python_tool("add", {"a": 1})  # missing b

    # Signature mismatch at bind
    with pytest.raises(FunctionKernelException, match="signature mismatch"):
        # This is tricky because Pydantic usually catches it first if annotations are right
        # But we can try passing extra args if it's not in the input model
        # Actually, input model is generated from signature, so it should be consistent.
        # Let's mock the definition to force a mismatch
        kernel.python_tool_definitions["add"].input_model = create_model(
            "fake", a=(int, ...), b=(int, ...), c=(int, ...)
        )
        await kernel._call_python_tool("add", {"a": 1, "b": 2, "c": 3})

    # Missing decorator
    def undecorated(x: int) -> int:
        return x

    with pytest.raises(
        FunctionKernelException, match="must be decorated with @kavalai.pythontool"
    ):
        kernel.register_python_tool("undecorated", undecorated)

    with pytest.raises(FunctionKernelException, match="not registered"):
        # Dynamic loading is now disabled
        await kernel._call_python_tool("os.getcwd", {})


@pytest.mark.asyncio
async def test_python_tool_output_mapping_details():
    """
    Tests detailed scenarios of mapping Python tool outputs to Pydantic models.
    Covers:
    - Mapping nested dictionaries to models.
    - Behavior when tool output (e.g., list) is incompatible with multi-field models.
    - Mapping between different Pydantic models (BaseModel to BaseModel).
    """
    kernel = FunctionKernel()

    # Nested dict output to model mapping
    @pythontool
    def nested_tool() -> Dict[str, Any]:
        return {"name": "nested", "value": 42}

    kernel.register_python_tool("nested", nested_tool)
    result = await kernel._call_python_tool("nested", {}, output_type=SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "nested"

    # Non-dict, non-BaseModel output with multi-field model should return original
    @pythontool
    def list_tool() -> list:
        return [1, 2]

    kernel.register_python_tool("list_tool_2", list_tool)
    result = await kernel._call_python_tool("list_tool_2", {}, output_type=SimpleModel)
    assert result == [1, 2]

    # BaseModel to different BaseModel mapping
    class OtherModel(BaseModel):
        name: str
        value: int
        extra: str = "extra"

    @pythontool
    def model_tool() -> SimpleModel:
        return SimpleModel(name="test", value=1)

    kernel.register_python_tool("model_tool", model_tool)
    result = await kernel._call_python_tool("model_tool", {}, output_type=OtherModel)
    assert isinstance(result, OtherModel)
    assert result.name == "test"
    assert result.extra == "extra"


@pytest.mark.asyncio
async def test_python_tool_unregistered_load():
    """
    Tests that dynamic loading is disabled and fails as expected.
    """
    kernel = FunctionKernel()
    # Should no longer work via dynamic loading
    tool_uri = "tests.test_functionkernel.sync_add"
    with pytest.raises(FunctionKernelException, match="not registered"):
        await kernel._call_python_tool(tool_uri, {"a": 10, "b": 20})

    # But works if registered
    kernel.register_python_tool("sync_add", sync_add)
    result = await kernel._call_python_tool("sync_add", {"a": 10, "b": 20})
    assert result.result == 30


@pytest.mark.asyncio
async def test_python_tool_docstring_parsing():
    """
    Tests that docstrings are correctly parsed for tool descriptions.
    """
    kernel = FunctionKernel()

    @pythontool
    def documented_func():
        """
        Summary line.

        Detailed description that should be included
        by inspect.getdoc().
        """
        pass

    kernel.register_python_tool("documented", documented_func)
    definition = kernel.python_tool_definitions["documented"]

    # inspect.getdoc() cleans up indentation and should return the whole docstring
    expected_doc = "Summary line.\n\nDetailed description that should be included\nby inspect.getdoc()."
    assert definition.description == expected_doc


@pytest.mark.asyncio
async def test_call_rest_tool(rest_server):
    """
    Tests calling tools on a remote REST server.
    Covers:
    - GET, POST, PUT, PATCH, and DELETE methods.
    - Output mapping to Pydantic models.
    - Basic Authentication using environment variables.
    - Error handling for missing servers or URLs.
    - Dynamic URL resolution from environment variables.
    """
    kernel = FunctionKernel()
    server = RestServer(name="test_server", url=rest_server)
    kernel.register_rest_server(server)

    # Get request
    result = await kernel._call_rest_tool("test_server", "get_item", {"id": 1})
    assert result == {"name": "rest_test", "value": 100}

    # Post request with model output
    result = await kernel._call_rest_tool(
        "test_server",
        "create_item",
        {"name": "new"},
        method="post",
        output_type=SimpleModel,
    )
    assert isinstance(result, SimpleModel)
    assert result.name == "rest_test"
    assert result.value == 100

    # Put request
    result = await kernel._call_rest_tool(
        "test_server", "update_item", {"id": 1}, method="put"
    )
    assert result == {"name": "updated", "value": 200}

    # Patch request
    result = await kernel._call_rest_tool(
        "test_server", "patch_item", {"id": 1}, method="patch"
    )
    assert result == {"name": "patched", "value": 300}

    # Delete request
    result = await kernel._call_rest_tool(
        "test_server", "delete_item", {"id": 1}, method="delete"
    )
    assert result == {"status": "deleted"}

    # Basic auth with env vars
    os.environ["TEST_USER"] = "test_user"
    os.environ["TEST_PASS"] = "test_pass"
    auth_server = RestServer(
        name="auth_server",
        url=rest_server,
        username_env="TEST_USER",
        password_env="TEST_PASS",
    )
    kernel.register_rest_server(auth_server)
    result = await kernel._call_rest_tool("auth_server", "auth_check", {})
    assert result == {"status": "authenticated"}

    # Basic auth with missing env vars
    os.environ.pop("TEST_USER", None)
    os.environ.pop("TEST_PASS", None)
    # Should call without auth and get 401, but _call_rest_tool uses raise_for_status
    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        await kernel._call_rest_tool("auth_server", "auth_check", {})

    # Error: Missing server
    with pytest.raises(
        FunctionKernelException, match="REST server 'missing' not registered"
    ):
        await kernel._call_rest_tool("missing", "any", {})

    # Error: Missing URL
    # Using a fake URL then clearing it to test the exception in _call_rest_tool
    _ = RestServer(name="no_url", url="http://temp")
    # Actually, we can just test with a missing URL in the env if using url_env
    server_env_missing = RestServer(name="no_env_url", url_env="NON_EXISTENT_URL")
    kernel.register_rest_server(server_env_missing)
    with pytest.raises(
        FunctionKernelException, match="URL for REST server 'no_env_url' not found"
    ):
        await kernel._call_rest_tool("no_env_url", "any", {})

    # URL from env
    os.environ["SERVER_URL"] = rest_server
    server_env_url = RestServer(name="env_url", url_env="SERVER_URL")
    kernel.register_rest_server(server_env_url)
    result = await kernel._call_rest_tool("env_url", "get_item", {"id": 1})
    assert result == {"name": "rest_test", "value": 100}


@pytest.mark.asyncio
async def test_mcp_tool_errors():
    """
    Tests error handling and edge cases for Model Context Protocol (MCP) tools.
    Covers:
    - Missing MCP server registration.
    - Handling of 'isError' flag in MCP responses.
    - Robustness against invalid JSON in MCP tool output.
    - Automatic wrapping of primitive MCP results into single-field models.
    - Real subprocess execution failure scenarios.
    """
    kernel = FunctionKernel()

    # Missing server
    with pytest.raises(
        FunctionKernelException, match="MCP server 'missing' not registered"
    ):
        await kernel._call_mcp_tool("missing", "any", {})

    # Use a simpler approach with patches to reach the exact lines
    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        # Mock stdio_client to return a context manager that returns (read, write)
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__.return_value = (mock_read, mock_write)

        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)

            # 1. Test isError=True in response (line 422)
            mock_response = MagicMock()
            mock_response.isError = True
            mock_content = MagicMock()
            mock_content.text = "Error text"
            # Mock the __str__ or __repr__ if needed, but the error message shows it prints the list
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

            mcp_server = McpServer(name="mcp_err", command="true")
            kernel.register_mcp_server(mcp_server)

            with pytest.raises(FunctionKernelException, match="failed: .*MagicMock"):
                await kernel._call_mcp_tool("mcp_err", "any", {})

            # 2. Test JSON parsing error (line 431)
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = "invalid-json"
            mock_response.content = [mock_content]

            result = await kernel._call_mcp_tool("mcp_err", "any", {})
            assert result == "invalid-json"

            # 3. Test TypeError in parsing (line 431)
            mock_content.text = None  # Should cause TypeError in json.loads
            result = await kernel._call_mcp_tool("mcp_err", "any", {})
            assert result is None

            # 4. Wrap primitive into model with one field (line 448)
            class WrapModel(BaseModel):
                result: int

            mock_content.text = "42"
            result = await kernel._call_mcp_tool(
                "mcp_err", "any", {}, output_type=WrapModel
            )
            assert isinstance(result, WrapModel)
            assert result.result == 42

            # 5. Model with multiple fields (line 529 - Python tool variant)
            # Actually line 453 is returned result_data if no output_type or matches failed.

    # 6. mcp_tool_definitions check (line 437)
    # Already hit by calling without tool def
    helper_path = os.path.join(
        os.path.dirname(__file__), "helpers", "mcp_server_errors.py"
    )

    try:
        server = McpServer(name="mcp_fail", command="python", args=[helper_path])
        kernel.register_mcp_server(server)

        with pytest.raises(
            FunctionKernelException, match="failed: .*Something went wrong"
        ):
            await kernel._call_mcp_tool("mcp_fail", "fail_tool", {})
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_mcp_tool_stdio():
    """
    Tests MCP tool execution using the stdio transport protocol.
    Spawns a mock MCP server in a subprocess to verify:
    - Full handshake (initialize, initialized, tools/list).
    - Execution of a tool and result capture.
    - Session reuse for subsequent tool calls.
    """
    kernel = FunctionKernel()
    # MCP uses JSON-RPC
    # We need to handle list_tools and call_tool
    helper_path = os.path.join(
        os.path.dirname(__file__), "helpers", "mcp_server_stdio.py"
    )

    try:
        server = McpServer(name="mcp_stdio", command="python", args=[helper_path])
        kernel.register_mcp_server(server)

        # First call: initializes session and list tools
        result = await kernel._call_mcp_tool(
            "mcp_stdio", "test_tool", {"arg": "val"}, output_type=SimpleModel
        )

        assert isinstance(result, SimpleModel)
        assert result.name == "mcp_test"
        assert result.value == 200

        # Second call should reuse session
        result2 = await kernel._call_mcp_tool("mcp_stdio", "test_tool", {"arg": "val"})
        assert result2.result == "ok"

    finally:
        # Cleanup
        await kernel.close()


@pytest.mark.asyncio
async def test_mcp_tool_results_mapping():
    """
    Tests specifically how MCP tool results (primitives) are mapped to
    Pydantic models with a single field.
    """
    kernel = FunctionKernel()
    helper_path = os.path.join(
        os.path.dirname(__file__), "helpers", "mcp_server_map.py"
    )

    try:
        server = McpServer(name="mcp_map", command="python", args=[helper_path])
        kernel.register_mcp_server(server)

        class WrapModel(BaseModel):
            result: int

        # Primitive 42 should be wrapped into WrapModel if it has one field
        result = await kernel._call_mcp_tool(
            "mcp_map", "primitive", {}, output_type=WrapModel
        )
        assert isinstance(result, WrapModel)
        assert result.result == 42

    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_mcp_tool_sse(rest_server):
    """
    Tests MCP tool execution using the Server-Sent Events (SSE) transport protocol.
    Interacts with a real FastAPI server acting as an MCP SSE host.
    """
    # Wait for server to be ready
    import httpx
    import time

    max_retries = 20
    for i in range(max_retries):
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{rest_server}/get_item?id=1")
                if resp.status_code == 200:
                    break
        except (httpx.ConnectError, httpx.ConnectTimeout):
            if i == max_retries - 1:
                pytest.fail("Server did not start in time")
            time.sleep(0.5)

    try:
        async with asyncio.timeout(20):
            kernel = FunctionKernel()
            server = McpServer(name="mcp_sse", url=f"{rest_server}/sse")
            kernel.register_mcp_server(server)

            result = await kernel._call_mcp_tool("mcp_sse", "hello", {})
            # It should be a model because _refresh_mcp_tool_definitions creates one
            assert result.result == "ok"

            # Cleanup
            await kernel.close()
    except asyncio.TimeoutError:
        pytest.fail("test_mcp_tool_sse timed out after 20 seconds")


@pytest.mark.asyncio
async def test_tool_descriptions():
    """
    Tests the generation of tool descriptions for all supported protocols.
    Verifies that the kernel correctly lists and formats URIs for:
    - Python tools (python://...)
    - REST tools (rest://...)
    - MCP tools (mcp://...)
    Includes error handling for malformed REST tool definitions.
    """
    kernel = FunctionKernel()
    kernel.register_python_tool("math.add", sync_add)
    kernel.register_rest_server(RestServer(name="my_rest", url="http://api.com"))
    kernel.register_mcp_server(McpServer(name="my_mcp", command="ls"))

    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)

            mock_tool = MagicMock()
            mock_tool.name = "list_files"
            mock_tool.description = "Lists files in directory"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }

            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            # Pre-initialize MCP session to load tools
            await kernel._get_mcp_session("my_mcp")

            allowed_tools = [
                "python://math.add",
                "mcp://my_mcp.*",
            ]
            desc = await kernel.get_tool_descriptions(allowed_tools=allowed_tools)
            tools = json.loads(desc)

            tool_names = [t["name"] for t in tools]
            assert "python://math.add" in tool_names
            assert "mcp://my_mcp.list_files" in tool_names

            # Find math.add tool
            math_add = next(t for t in tools if t["name"] == "python://math.add")
            assert math_add["description"] == "Adds two integers."
            assert "properties" in math_add["inputSchema"]
            assert "a" in math_add["inputSchema"]["properties"]
            assert math_add["inputSchema"]["properties"]["a"]["type"] == "integer"

            # Verify concise format (no pydantic specific titles/types at top level)
            assert "title" not in math_add["inputSchema"]
            assert (
                "type" not in math_add["inputSchema"]
            )  # It should be inside properties, but not at root if we popped it

            # Test ToolDefinition.__str__
            tool_def = kernel.python_tool_definitions["math.add"]
            str_repr = str(tool_def)
            parsed_repr = json.loads(str_repr)
            assert parsed_repr["name"] == "math.add"
            assert parsed_repr["description"] == "Adds two integers."
            assert "inputSchema" in parsed_repr
            assert "a" in parsed_repr["inputSchema"]["properties"]

            # REST server descriptions mapping error
            kernel.register_rest_server(
                RestServer(name="bad_desc", url="http://api.com")
            )
            kernel.rest_tool_definitions["bad_desc"] = {
                "tool": ToolDefinition(
                    name="tool",
                    description="not-json",
                    input_model=SimpleModel,
                    output_model=SimpleModel,
                )
            }
            allowed_tools.append("rest://bad_desc.tool [GET]")
            desc = await kernel.get_tool_descriptions(allowed_tools=allowed_tools)
            tools = json.loads(desc)
            bad_tool = next(t for t in tools if "bad_desc.tool" in t["name"])
            assert bad_tool["name"] == "rest://bad_desc.tool [GET]"
            assert bad_tool["description"] == "not-json"


@pytest.mark.asyncio
async def test_call_tool_unified():
    """
    Tests the unified `call_tool` entry point which routes requests based on URI protocol.
    Verifies routing for:
    - python://
    - rest://
    - mcp://
    Uses mocks to isolate protocol-specific logic.
    """
    kernel = FunctionKernel()

    # Test Python tool via unified call
    kernel.register_python_tool("math.add", sync_add)
    result = await kernel.call_tool("python://math.add", {"a": 10, "b": 20})
    assert result.result == 30

    # Test REST tool via unified call
    server = RestServer(name="test_api", url="http://api.example.com")
    kernel.register_rest_server(server)

    with patch("httpx.AsyncClient.request") as mock_request:
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        result = await kernel.call_tool("rest://test_api.status", {})
        assert result == {"status": "ok"}
        mock_request.assert_called_with(
            "GET", "http://api.example.com/status", params={}, timeout=60.0
        )

    # MCP tool via unified call
    mcp_server = McpServer(name="test_mcp", command="true")
    kernel.register_mcp_server(mcp_server)

    with patch("kavalai.functionkernel.stdio_client") as mock_stdio:
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        with patch("kavalai.functionkernel.ClientSession") as mock_session_cls:
            mock_session = mock_session_cls.return_value
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)

            # Mock list_tools
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "A test tool"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {"arg": {"type": "integer"}},
            }
            mock_tools_result = MagicMock()
            mock_tools_result.tools = [mock_tool]
            mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

            mock_response = MagicMock()
            mock_response.isError = False
            mock_content = MagicMock()
            mock_content.text = json.dumps({"result": {"mcp": "works"}})
            mock_response.content = [mock_content]
            mock_session.call_tool = AsyncMock(return_value=mock_response)

            result = await kernel.call_tool("mcp://test_mcp.test_tool", {"arg": 1})
            assert result.result == {"mcp": "works"}
            mock_session.call_tool.assert_called_with("test_tool", arguments={"arg": 1})


@pytest.mark.asyncio
async def test_call_tool_errors():
    """
    Tests error handling in the unified `call_tool` dispatcher.
    Covers invalid URI formats and unsupported protocols.
    """
    kernel = FunctionKernel()

    with pytest.raises(FunctionKernelException, match="Invalid tool URI format"):
        await kernel.call_tool("invalid_format", {})

    with pytest.raises(FunctionKernelException, match="Invalid tool path format"):
        await kernel.call_tool("rest://nodot", {})

    with pytest.raises(FunctionKernelException, match="Unsupported protocol"):
        await kernel.call_tool("ftp://some.file", {})


@pytest.mark.asyncio
async def test_close_error():
    """
    Tests that the kernel's `close` method handles exceptions during cleanup gracefully
    without raising them.
    """
    kernel = FunctionKernel()

    # Mock a cleanup that raises an error
    mock_cleanup = AsyncMock()
    mock_cleanup.__aexit__ = AsyncMock(side_effect=Exception("Cleanup failed"))
    kernel.mcp_cleanups.append(mock_cleanup)

    # Should not raise exception
    await kernel.close()
    assert len(kernel.mcp_sessions) == 0
    assert len(kernel.mcp_cleanups) == 0


@pytest.mark.asyncio
async def test_registration_conflicts():
    """
    Tests that the kernel prevents duplicate registration of servers and tools.
    Verifies exceptions are raised when attempting to register a duplicate:
    - REST server name.
    - MCP server name.
    - Python tool name.
    """
    kernel = FunctionKernel()

    # Test REST server conflict
    kernel.register_rest_server(RestServer(name="test", url="http://api.com"))
    with pytest.raises(
        FunctionKernelException, match="REST server 'test' is already registered"
    ):
        kernel.register_rest_server(RestServer(name="test", url="http://other.com"))

    # Test MCP server conflict
    kernel.register_mcp_server(McpServer(name="mcp", command="ls"))
    with pytest.raises(
        FunctionKernelException, match="MCP server 'mcp' is already registered"
    ):
        kernel.register_mcp_server(McpServer(name="mcp", command="dir"))

    # Test Python tool conflict
    kernel.register_python_tool("add", sync_add)
    with pytest.raises(
        FunctionKernelException, match="Python tool 'add' is already registered"
    ):
        kernel.register_python_tool("add", async_multiply)


@pytest.mark.asyncio
async def test_register_rest_tool(rest_server):
    """
    Tests explicit registration of individual tools for a REST server.
    Verifies:
    - Manual schema definition for REST endpoints.
    - Calling manually registered REST tools via the unified interface.
    - Correct formatting in tool descriptions.
    """
    kernel = FunctionKernel()
    server = RestServer(name="test_server", url=rest_server)
    kernel.register_rest_server(server)

    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
    }

    kernel.register_rest_tool(
        server_name="test_server",
        tool_name="get_item",
        method="get",
        input_schema=input_schema,
        output_schema=output_schema,
        description="Get item by id",
    )

    # Call via call_tool (unified)
    result = await kernel.call_tool("rest://test_server.get_item", {"id": 1})
    assert isinstance(result, BaseModel)
    assert result.name == "rest_test"
    assert result.value == 100

    # Test POST registration
    post_input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    kernel.register_rest_tool(
        server_name="test_server",
        tool_name="create_item",
        method="post",
        input_schema=post_input_schema,
        output_schema=output_schema,
    )

    result = await kernel.call_tool("rest://test_server.create_item", {"name": "new"})
    assert result.value == 100

    # Verify descriptions
    allowed_tools = ["rest://test_server.*"]
    descriptions = await kernel.get_tool_descriptions(allowed_tools=allowed_tools)
    tools = json.loads(descriptions)

    get_item = next(
        t for t in tools if t["name"] == "rest://test_server.get_item [GET]"
    )
    assert get_item["description"] == "Get item by id"
    assert "id" in get_item["inputSchema"]["properties"]

    create_item = next(
        t for t in tools if t["name"] == "rest://test_server.create_item [POST]"
    )
    assert "name" in create_item["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_tool_descriptions_nested():
    """Tests that tool descriptions correctly expand nested Pydantic models."""
    kernel = FunctionKernel()

    class NestedInput(BaseModel):
        field_a: str
        field_b: int

    class ComplexInput(BaseModel):
        name: str
        nested: NestedInput

    class NestedOutput(BaseModel):
        success: bool
        message: str

    class ComplexOutput(BaseModel):
        code: int
        data: NestedOutput

    @pythontool
    def complex_tool(name: str, nested: NestedInput) -> ComplexOutput:
        """A tool with nested pydantic models."""
        return ComplexOutput(
            code=200, data=NestedOutput(success=True, message=f"Hello {name}")
        )

    kernel.register_python_tool("complex_tool", complex_tool)

    desc = await kernel.get_tool_descriptions(allowed_tools=["python://complex_tool"])
    tools = json.loads(desc)
    complex_tool = next(t for t in tools if t["name"] == "python://complex_tool")

    # Check for description
    assert complex_tool["description"] == "A tool with nested pydantic models."

    # Check for input schema expansion
    schema = complex_tool["inputSchema"]
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"
    assert "nested" in schema["properties"]
    assert "$ref" in schema["properties"]["nested"]
    assert "$defs" in schema
    assert "NestedInput" in schema["$defs"]
