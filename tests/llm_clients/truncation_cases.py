"""The output-cap cases each provider's integration tests run.

Every provider test file calls these against the real API with the cheap
model its smoke test uses. They assert the contract — a cap that is large
enough returns a valid model within it; one that is too small raises
:class:`OutputTruncatedError` once, never a partial model and never a retry —
and print what the provider reported, so ``pytest -m integration -s`` shows
how each provider truncates.
"""

import pytest
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    ModelCallStat,
    ModelStatsReceiver,
    OutputTruncatedError,
)

GENEROUS_CAP = 1024
TINY_CAP = 40


class Capital(BaseModel):
    city: str
    country: str


class Story(BaseModel):
    title: str
    paragraphs: list[str]


CAPITAL_PROMPT = "What is the capital of France? Respond in JSON."
STORY_PROMPT = (
    "Write a story about a lighthouse keeper in five paragraphs of about "
    "eighty words each. Respond in JSON."
)
LONG_TEXT_PROMPT = "Write a 300-word story about a lighthouse keeper."


class StatsCollector(ModelStatsReceiver):
    def __init__(self):
        self.calls: list[ModelCallStat] = []

    def receive_model_stats(self, stats: ModelCallStat):
        self.calls.append(stats)


def history(text: str) -> ChatHistory:
    return ChatHistory(messages=[ChatMessage(role="user", content=text)])


def count_attempts(client: BaseLlmClient) -> list:
    """Record every provider attempt the retry loop makes."""
    attempts = []
    run = client._run_chat_completions

    async def counted(*args, **kwargs):
        attempts.append(1)
        return await run(*args, **kwargs)

    client._run_chat_completions = counted
    return attempts


def report(client: BaseLlmClient, case: str, error: OutputTruncatedError) -> None:
    print(
        f"\n[{client.stat_model_name()}] {case}: reason={error.reason!r} "
        f"cap={error.max_output_tokens} prompt={error.prompt_tokens} "
        f"completion={error.completion_tokens} "
        f"reasoning={error.reasoning_tokens} "
        f"partial={error.partial_output!r}"
    )


async def structured_answer_fits_the_cap(
    client: BaseLlmClient, stats: StatsCollector, cap: int = GENEROUS_CAP
) -> Capital:
    answer = await client.chat_completions(
        chat_history=history(CAPITAL_PROMPT), response_model=Capital
    )

    assert isinstance(answer, Capital)
    assert "Paris" in answer.city
    (stat,) = stats.calls
    assert stat.response_code == 200
    assert 0 < stat.completion_tokens <= cap
    print(
        f"\n[{client.stat_model_name()}] generous cap {cap}: {answer!r} "
        f"completion={stat.completion_tokens} reasoning={stat.reasoning_tokens}"
    )
    return answer


async def structured_answer_over_the_cap_raises(
    client: BaseLlmClient, stats: StatsCollector, cap: int = TINY_CAP
) -> OutputTruncatedError:
    attempts = count_attempts(client)

    with pytest.raises(OutputTruncatedError) as caught:
        await client.chat_completions(
            chat_history=history(STORY_PROMPT), response_model=Story
        )

    error = caught.value
    report(client, f"structured, cap {cap}", error)
    assert attempts == [1], "a truncated call must not be retried"
    assert error.max_output_tokens == cap
    (stat,) = stats.calls
    assert stat.response_code is None
    assert stat.prompt_tokens > 0
    assert stat.completion_tokens <= cap
    return error


async def streamed_text_over_the_cap_raises(
    client: BaseLlmClient, stats: StatsCollector, cap: int = TINY_CAP
) -> OutputTruncatedError:
    """The consumer sees the partial chunks that arrived, then the error.

    No ``complete`` chunk is sent, so a consumer that treats ``complete`` as
    the answer never receives a truncated one.
    """
    attempts = count_attempts(client)
    streamer = await client.stream_chat_completions(
        chat_history=history(LONG_TEXT_PROMPT)
    )

    seen = []
    with pytest.raises(OutputTruncatedError) as caught:
        async for chunk in streamer:
            seen.append(chunk)

    error = caught.value
    report(client, f"streamed text, cap {cap}, {len(seen)} partials", error)
    assert attempts == [1], "a truncated call must not be retried"
    assert all(chunk.type == "partial" for chunk in seen)
    answer = [chunk for chunk in seen if chunk.name == "response"]
    if answer:
        assert answer[-1].value == error.partial_output
    (stat,) = stats.calls
    assert stat.response_code is None
    return error
