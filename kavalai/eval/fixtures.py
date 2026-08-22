"""Recorded model responses, so a suite runs in CI with no API key.

The tier-zero story: record a suite once against the real models, commit the
responses, and every pull request afterwards re-runs the retrieval, the
routing, the validation, the side effects and the evaluator code itself in a
couple of seconds, for nothing, with no secrets on the runner.

Two things this is deliberately **not**:

* It is not a mock. The recorded text is what the model actually said, so the
  parsing and the branch decisions under test are the real ones.
* It is not replayed production traffic. These are fixtures for cases we
  wrote, which is what keeps customer conversations out of a git repository.

A missing fixture is an error, never a silent pass — a gate that goes green
because it could not find anything to check is worse than no gate.

**The prompts have to be deterministic.** A response is looked up by the exact
prompt that produced it, so anything random reaching a prompt — a UUID, a
timestamp, ``datetime.now()`` — makes every run a cache miss. That is a good
constraint to design under rather than around: see ``examples/bakery/tools.py``,
where the order book numbers its orders sequentially and the clock is pinned.

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

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

from loguru import logger
from pydantic import BaseModel

from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
)
from kavalai.llm_clients.streamer import StreamContent


class MissingFixture(Exception):
    """No recorded response for this call, and recording is off."""


def fixture_key(model: str, chat_history: ChatHistory) -> str:
    """A stable id for one model call.

    The model name plus the exact prompt text. Changing a prompt therefore
    invalidates its fixture, which is the correct behaviour: the recorded
    answer was to a different question.
    """
    digest = hashlib.sha256()
    digest.update((model or "").encode("utf-8"))
    for message in chat_history.messages:
        digest.update(f"\n{message.role}:{message.content}".encode("utf-8"))
    return digest.hexdigest()[:24]


class FixtureStore:
    """The recorded responses for one suite, as a single JSON file."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        self.misses: list[str] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[str]:
        entry = self.entries.get(key)
        return entry.get("response") if entry else None

    def put(self, key: str, model: str, prompt: str, response: str) -> None:
        # The prompt is stored alongside the response so a fixture file is
        # reviewable: a diff shows which question changed, not only a hash.
        self.entries[key] = {"model": model, "prompt": prompt, "response": response}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.entries)


class FixtureLlmClient(BaseLlmClient):
    """Replays a recorded response, or records one by delegating to a real client."""

    provider = "fixture"

    def __init__(
        self,
        model: str,
        store: FixtureStore,
        *,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
        record_with: Optional[BaseLlmClient] = None,
    ):
        super().__init__(parameters, stats_receiver)
        self.model = model
        self.store = store
        self.record_with = record_with

    async def stream_chat_completions(
        self,
        *,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]] = None,
        stream_delta: bool = False,
    ):
        key = fixture_key(self.model, chat_history)
        response = self.store.get(key)

        if response is None:
            if self.record_with is None:
                self.store.misses.append(key)
                prompt = "\n".join(m.content or "" for m in chat_history.messages)
                raise MissingFixture(
                    f"No recorded response for {self.model} ({key}). Re-record the "
                    f"fixtures — the prompt has changed, or this case is new.\n"
                    f"Prompt began: {prompt[:200]!r}"
                )
            response = await self.record_with.chat_completions(
                chat_history=chat_history, response_model=response_model
            )
            if isinstance(response, BaseModel):
                response = response.model_dump_json()
            self.store.put(
                key,
                self.model,
                "\n".join(m.content or "" for m in chat_history.messages),
                response,
            )
            logger.info(f"Recorded fixture {key} for {self.model}")

        await self._send_model_call_stats(
            ModelCallStat(
                call_type="llm",
                model=self.stat_model_name(),
                total_tokens=0,
                duration_seconds=0.0,
            )
        )
        return _replay(response)


async def _replay(response: str):
    """Yield the recorded text the way a live stream would end.

    One partial then one complete: consumers that safe-parse partials and
    consumers that only read the final value both see what they expect.
    """
    yield StreamContent(type="partial", name="response", value=response)
    yield StreamContent(type="complete", name="response", value=response)


def fixture_client_factory(
    path: Union[str, Path], *, record: bool = False
) -> Callable[..., BaseLlmClient]:
    """A ``client_factory`` for :class:`~kavalai.eval.EngineTarget`.

    ``record=True`` calls the real provider and writes what it says; the
    default replays and raises on anything it has not seen.

        target = EngineTarget(
            "assistant.yaml",
            client_factory=fixture_client_factory("eval/fixtures/llm.json"),
        )
    """
    store = FixtureStore(path)

    def factory(
        model: str,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
        **kwargs: Any,
    ) -> BaseLlmClient:
        live = None
        if record:
            from kavalai.workflow.clients import make_client

            live = make_client(model, parameters, stats_receiver)
        return FixtureLlmClient(
            model,
            store,
            parameters=parameters,
            stats_receiver=stats_receiver,
            record_with=live,
        )

    factory.store = store  # so the caller can save() after recording
    return factory
