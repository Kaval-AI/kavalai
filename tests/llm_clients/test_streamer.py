import pytest
from pydantic import BaseModel
from kavalai.llm_clients.streamer import Streamer, StreamerTimeoutException
import asyncio


class MockModel(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_streamer_basic():
    streamer = Streamer()
    value_streamer = streamer.get_value_streamer("results")
    await value_streamer.stream_partial("12")
    await value_streamer.stream_partial("34")
    await value_streamer.stream_partial("5")
    await value_streamer.stream_complete()
    async for stream_content in streamer:
        print(stream_content)


@pytest.mark.asyncio
async def test_streamer_timeout():
    # We want to test that if we don't put anything in the queue, it times out
    streamer = Streamer(timeout_seconds=0.1)
    _ = streamer.get_value_streamer("v1")
    _ = streamer.get_value_streamer("v2")

    with pytest.raises(StreamerTimeoutException) as excinfo:
        async for _ in streamer:
            pass

    assert "v1" in str(excinfo.value)
    assert "v2" in str(excinfo.value)


@pytest.mark.asyncio
async def test_streamer_partial_timeout():
    streamer = Streamer(timeout_seconds=0.1)
    v1 = streamer.get_value_streamer("v1")
    await v1.stream_partial("hello")

    with pytest.raises(StreamerTimeoutException) as excinfo:
        iterator = streamer.__aiter__()
        # First one should succeed
        content = await iterator.__anext__()
        assert content.value == "hello"
        # Second one should timeout as v1 is not completed
        await iterator.__anext__()

    assert "v1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_streamer_response_model():
    streamer = Streamer()
    # Test with dict
    vs = streamer.get_value_streamer("test", response_model=MockModel)
    await vs.stream_partial('{"name": "test"}')
    await vs.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[0].value == '{"name": "test"}'
    assert contents[1].type == "complete"
    assert contents[1].value == '{"name": "test"}'


@pytest.mark.asyncio
async def test_streamer_response_model_primitive():
    streamer = Streamer()
    # Test with primitive (non-dict/list)
    vs = streamer.get_value_streamer("test", response_model=MockModel)
    await vs.stream_partial("42")
    await vs.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[0].value == "42"
    assert contents[1].value == "42"


@pytest.mark.asyncio
async def test_streamer_double_complete():
    streamer = Streamer()
    vs = streamer.get_value_streamer("test")
    await vs.stream_complete()
    with pytest.raises(RuntimeError, match="already called"):
        await vs.stream_complete()


@pytest.mark.asyncio
async def test_streamer_delta():
    streamer = Streamer(stream_delta=True)
    vs = streamer.get_value_streamer("test")
    await vs.stream_partial("h")
    await vs.stream_partial("e")
    await vs.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[0].value == "h"
    assert contents[1].value == "e"
    assert contents[2].type == "complete"
    assert contents[2].value is None


@pytest.mark.asyncio
async def test_streamer_queue_property():
    streamer = Streamer()
    assert isinstance(streamer.queue, asyncio.Queue)


@pytest.mark.asyncio
async def test_streamer_context_manager():
    async with Streamer() as streamer:
        assert isinstance(streamer, Streamer)
        vs = streamer.get_value_streamer("test")
        await vs.stream_complete()
        async for _ in streamer:
            pass


@pytest.mark.asyncio
async def test_streamer_multiple_values():
    streamer = Streamer()
    v1 = streamer.get_value_streamer("v1")
    v2 = streamer.get_value_streamer("v2")

    await v1.stream_partial("a")
    await v1.stream_complete()

    # Do NOT complete v2 yet.

    contents = []
    iterator = streamer.__aiter__()

    # Get v1 partial
    contents.append(await iterator.__anext__())
    # Get v1 complete
    contents.append(await iterator.__anext__())

    assert contents[0].name == "v1"
    assert contents[1].type == "complete"

    await v2.stream_partial("b")
    await v2.stream_complete()

    # Get v2 partial
    contents.append(await iterator.__anext__())
    # Get v2 complete
    contents.append(await iterator.__anext__())

    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()

    assert len(contents) == 4
    assert contents[0].name == "v1"
    assert contents[1].name == "v1"
    assert contents[1].type == "complete"
    assert contents[2].name == "v2"
    assert contents[3].name == "v2"
    assert contents[3].type == "complete"


@pytest.mark.asyncio
async def test_streamer_response_model_list():
    streamer = Streamer()
    vs = streamer.get_value_streamer("test", response_model=MockModel)
    await vs.stream_partial('[{"name": "a"}, {"name": "b"}]')
    await vs.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[0].value == '[{"name": "a"}, {"name": "b"}]'


@pytest.mark.asyncio
async def test_streamer_value_override_delta():
    streamer = Streamer(stream_delta=False)
    vs = streamer.get_value_streamer("test", stream_delta=True)
    await vs.stream_partial("h")
    await vs.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert contents[0].value == "h"
    assert contents[1].type == "complete"
    assert contents[1].value is None


@pytest.mark.asyncio
async def test_streamer_restart_passes_through_and_reset_active():
    """A retried attempt resets the active accounting and re-registers its
    streams; the restart chunk flows to the consumer and iteration still
    terminates on the new attempt's complete."""
    streamer = Streamer()
    v1 = streamer.get_value_streamer("response")
    await v1.stream_partial("gar")

    # Simulate a retry: forget the failed attempt, announce the restart.
    streamer.reset_active()
    await streamer.stream_restart("attempt 1: boom")
    v2 = streamer.get_value_streamer("response")
    await v2.stream_partial("ok")
    await v2.stream_complete()

    contents = []
    async for content in streamer:
        contents.append(content)

    assert [c.type for c in contents] == ["partial", "restart", "partial", "complete"]
    assert contents[1].name == "response"
    assert contents[1].value == "attempt 1: boom"
    # Without reset_active() the stale name would keep iteration alive.
    assert contents[-1].type == "complete"


@pytest.mark.asyncio
async def test_streamer_stale_complete_after_reset_is_harmless():
    """A streamer completing after reset_active() dropped its name must not
    raise from the accounting callback."""
    streamer = Streamer()
    stale = streamer.get_value_streamer("response")
    streamer.reset_active()

    await stale.stream_complete()  # discard-if-present: no ValueError

    contents = []
    async for content in streamer:
        contents.append(content)
    assert [c.type for c in contents] == ["complete"]
