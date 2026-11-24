import contextlib

from kavalai import Telemetry


async def numbers():
    yield 1


def test_base_telemetry_does_nothing():
    telemetry = Telemetry()
    run = telemetry.start_run(
        workflow="wf",
        agent_id=None,
        session_id=None,
        run_id=None,
        external_id=None,
        invocation_id="abc",
    )
    assert run is None
    assert (
        telemetry.end_run(
            run, status="completed", error=None, token_usage={}, duration_seconds=0.1
        )
        is None
    )
    node = telemetry.start_node(name="n", node_type="llm", seq=0)
    assert node is None
    assert telemetry.end_node(node) is None
    assert telemetry.model_call(object()) is None


def test_within_returns_its_argument_unchanged():
    events = numbers()
    assert Telemetry().within(None, events) is events


def test_tool_call_is_a_context_manager_that_does_nothing():
    manager = Telemetry().tool_call("python://greet")
    assert isinstance(manager, contextlib.AbstractContextManager)
    with manager:
        pass
