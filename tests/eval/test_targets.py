"""Targets: the seam that keeps evaluators independent of how a run happened."""

from pathlib import Path

import pytest
import yaml

from kavalai.eval import Case, CallableTarget, EngineTarget, RestTarget, build_target
from kavalai.eval.models import TargetSpec
from kavalai.eval.targets import RunRecord, import_object

WORKFLOW = {
    "name": "wf",
    "description": "d",
    "llm_model": "openai/fake",
    "data_types": {
        "input": {"type": "object", "properties": {"user_message": {"type": "string"}}},
        "output": {
            "type": "object",
            "properties": {"agent_response": {"type": "string"}},
        },
    },
    "nodes": [
        {"name": "s", "type": "start", "next": "answer"},
        {
            "name": "answer",
            "type": "llm",
            "prompt": "hi",
            "output": "output",
            "next": "e",
        },
        {"name": "e", "type": "end", "output": "output"},
    ],
}


@pytest.fixture
def workflow_file(tmp_path) -> Path:
    path = tmp_path / "wf.yaml"
    path.write_text(yaml.safe_dump(WORKFLOW))
    return path


def fake_client_factory(value="hi there"):
    """Reuse the engine tests' deterministic streaming client."""
    from tests.workflow.test_engine import make_factory

    return make_factory({"agent_response": value})


# ------------------------------------------------------------- output shaping
@pytest.mark.parametrize(
    "output,expected",
    [
        ({"agent_response": "hi"}, "hi"),
        ({"answer": "hi"}, "hi"),
        ({"body": "hi"}, "hi"),
        ("plain", "plain"),
        (None, ""),
    ],
)
def test_output_text_finds_the_answer_whatever_its_shape(output, expected):
    assert RunRecord(output=output).output_text() == expected


def test_output_text_falls_back_to_the_whole_value():
    """A `contains` assertion must always have something to look at."""
    assert "42" in RunRecord(output={"total": 42}).output_text()


def test_total_tokens_sums_the_runs_calls():
    from kavalai.eval.targets import ModelCallRecord

    record = RunRecord(
        model_calls=[ModelCallRecord(total_tokens=3), ModelCallRecord(total_tokens=4)]
    )
    assert record.total_tokens == 7


def test_ok_distinguishes_completed_from_failed():
    assert RunRecord().ok is True
    assert RunRecord(status="failed").ok is False
    assert RunRecord(error="boom").ok is False


# ------------------------------------------------------------- engine target
async def test_engine_target_produces_a_full_trajectory(workflow_file):
    target = EngineTarget(workflow_file, client_factory=fake_client_factory())
    await target.setup()
    try:
        record = await target.run(Case(name="c", inputs={"user_message": "hi"}))
    finally:
        await target.aclose()

    assert record.ok
    assert record.output == {"agent_response": "hi there"}
    assert record.trajectory.names() == ["s", "answer", "e"]
    assert record.trajectory.observed is True
    assert record.total_tokens == 1
    assert target.describe()["kind"] == "engine"


async def test_engine_target_isolates_concurrent_cases(workflow_file):
    """One engine, per-case loggers: no run sees another's trajectory."""
    import asyncio

    target = EngineTarget(workflow_file, client_factory=fake_client_factory())
    await target.setup()
    try:
        records = await asyncio.gather(
            *[
                target.run(Case(name=f"c{i}", inputs={"user_message": "hi"}))
                for i in range(4)
            ]
        )
    finally:
        await target.aclose()
    assert all(r.trajectory.names() == ["s", "answer", "e"] for r in records)


async def test_a_case_that_raises_is_an_error_not_a_failing_grade(workflow_file):
    target = EngineTarget(workflow_file, client_factory=fake_client_factory())
    await target.setup()
    try:
        record = await target.run(Case(name="c", inputs={"nonsense": 1}))
    finally:
        await target.aclose()
    assert record.ok is False and record.error


async def test_the_sandbox_hook_runs_once_per_case(workflow_file, monkeypatch):
    calls = []

    def make():
        calls.append(1)
        return {"world": len(calls)}

    monkeypatch.setattr("tests.eval.test_targets._sandbox_factory", make, raising=False)
    target = EngineTarget(
        workflow_file,
        sandbox="tests.eval.test_targets:_sandbox_factory",
        client_factory=fake_client_factory(),
    )
    await target.setup()
    try:
        first = await target.run(Case(name="a", inputs={"user_message": "hi"}))
        second = await target.run(Case(name="b", inputs={"user_message": "hi"}))
    finally:
        await target.aclose()
    assert first.sandbox == {"world": 1}
    assert second.sandbox == {"world": 2}


def _sandbox_factory():  # replaced by the monkeypatch above
    return None


# ----------------------------------------------------------- callable target
async def test_callable_target_takes_an_async_function():
    async def run(inputs):
        return {"agent_response": inputs["user_message"].upper()}

    target = CallableTarget(run)
    await target.setup()
    record = await target.run(Case(name="c", inputs={"user_message": "hi"}))
    await target.aclose()

    assert record.output_text() == "HI"
    # No trajectory: the honest trade, made explicit rather than silently.
    assert record.trajectory.observed is False
    assert target.observes_trajectory is False


async def test_callable_target_takes_a_sync_function_too():
    target = CallableTarget(lambda inputs: "plain")
    await target.setup()
    assert (await target.run(Case(name="c", inputs={}))).output == "plain"


async def test_callable_target_captures_a_raise():
    def boom(inputs):
        raise RuntimeError("nope")

    target = CallableTarget(boom)
    await target.setup()
    record = await target.run(Case(name="c", inputs={}))
    assert record.ok is False and "nope" in record.error


# --------------------------------------------------------------- rest target
def test_rest_target_is_output_only():
    target = RestTarget("http://localhost:8000/")
    assert target.observes_trajectory is False
    assert target.describe() == {
        "kind": "rest",
        "base_url": "http://localhost:8000",
        "path": "/run_agent",
    }
    assert RestTarget("http://x", path="run").path == "/run"


async def test_rest_target_captures_a_transport_failure():
    target = RestTarget("http://127.0.0.1:1", timeout_seconds=0.2)
    await target.setup()
    try:
        record = await target.run(Case(name="c", inputs={}))
    finally:
        await target.aclose()
    assert record.ok is False and record.error


# --------------------------------------------------------------- build_target
def test_build_target_dispatches_on_kind(workflow_file, tmp_path):
    engine = build_target(
        TargetSpec(kind="engine", workflow=workflow_file.name), tmp_path
    )
    assert isinstance(engine, EngineTarget)
    assert isinstance(
        build_target(TargetSpec(kind="rest", base_url="http://x"), tmp_path), RestTarget
    )
    assert isinstance(
        build_target(
            TargetSpec(
                kind="callable", function="tests.eval.test_targets:_sandbox_factory"
            ),
            tmp_path,
        ),
        CallableTarget,
    )


@pytest.mark.parametrize(
    "spec,message",
    [
        (TargetSpec(kind="engine"), "needs a 'workflow' path"),
        (TargetSpec(kind="rest"), "needs a 'base_url'"),
        (TargetSpec(kind="callable"), "needs a 'function' reference"),
        (TargetSpec(kind="magic"), "Unknown target kind"),
    ],
)
def test_build_target_says_what_is_missing(spec, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        build_target(spec, tmp_path)


def test_import_object_takes_both_spellings():
    assert import_object("kavalai.eval.targets:import_object") is import_object
    assert import_object("kavalai.eval.targets.import_object") is import_object
