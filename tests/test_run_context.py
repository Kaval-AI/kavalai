import pytest
from types import SimpleNamespace
from uuid import uuid4
from pydantic import BaseModel
from unittest.mock import AsyncMock

from kavalai.run_context import RunContext
from kavalai.workflow.models import ArgumentInfo


class MockModel(BaseModel):
    field: str


def _task(inputs: dict) -> SimpleNamespace:
    """A minimal stand-in for anything with an ``inputs`` mapping."""
    return SimpleNamespace(inputs=inputs)


@pytest.mark.asyncio
async def test_resolve_context_value():
    rc = RunContext(data={"user": {"name": "Alice"}, "items": [1, 2, 3]})

    assert rc.resolve_context_value("user.name") == "Alice"
    assert rc.resolve_context_value("items.0") == 1
    assert rc.resolve_context_value("nonexistent") is None


@pytest.mark.asyncio
async def test_resolve_input_info_literal():
    rc = RunContext()
    info = ArgumentInfo(type="literal", value="hello")
    assert await rc.resolve_input_info(info) == "hello"


@pytest.mark.asyncio
async def test_resolve_input_info_context():
    rc = RunContext(data={"input": {"text": "world"}})

    # Using value as path
    info = ArgumentInfo(type="context", value="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Using name as path
    info = ArgumentInfo(type="context", name="input.text")
    assert await rc.resolve_input_info(info) == "world"

    # Path is None
    info = ArgumentInfo(type="context")
    assert await rc.resolve_input_info(info) is None


@pytest.mark.asyncio
async def test_resolve_input_info_history():
    session_id = uuid4()
    mock_service = AsyncMock()
    mock_service.get_history_value.return_value = "history_val"

    rc = RunContext(session_id=session_id, agent_service=mock_service)

    # Using value
    info = ArgumentInfo(type="history", value="some_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "some_key")

    # Using name
    info = ArgumentInfo(type="history", name="other_key")
    assert await rc.resolve_input_info(info) == "history_val"
    mock_service.get_history_value.assert_called_with(session_id, "other_key")


@pytest.mark.asyncio
async def test_resolve_input_info_history_missing_service(caplog):
    rc = RunContext(session_id=uuid4())
    info = ArgumentInfo(type="history", value="key")
    assert await rc.resolve_input_info(info) is None
    assert (
        "Cannot load from history for key: agent_service or session_id not set"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_prepare_tool_inputs():
    rc = RunContext(data={"val": "from_context"})
    task = _task(
        {
            "lit": ArgumentInfo(type="literal", value="fixed"),
            "ctx": ArgumentInfo(type="context", value="val"),
            "implicit": ArgumentInfo(type="context"),  # should use name "implicit"
        }
    )
    # Patch implicit context to match data
    rc.data["implicit"] = "implicit_val"

    inputs = await rc.prepare_tool_inputs(task)
    assert inputs == {"lit": "fixed", "ctx": "from_context", "implicit": "implicit_val"}


@pytest.mark.asyncio
async def test_prepare_tool_inputs_with_pydantic_model():
    rc = RunContext()
    model_instance = MockModel(field="test")
    task = _task({"mod": ArgumentInfo(type="literal", value=model_instance)})

    inputs = await rc.prepare_tool_inputs(task)
    assert inputs["mod"] == {"field": "test"}
