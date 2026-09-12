"""Fake LLM and embedding clients for testing without a provider.

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

A workflow test should exercise the real engine, the real streamer and the real
validation, and replace only the model. This module provides that replacement:

- :class:`ScriptedLlmClient` answers from a script of replies, streams each
  reply in partials through the base client's machinery, and reports a
  :class:`~kavalai.ModelCallStat` for every call.
- :class:`FakeEmbeddingClient` embeds any text deterministically by hashing
  its words, so texts that share words lie close together.
- :func:`fake_providers` registers both under a provider name for the duration
  of a ``with`` block, so ``llm_model: fake/anything`` in a workflow and
  ``model="fake/tiny"`` for a RAG service resolve to them.

The module is pure Python and part of the base install. It imports neither a
provider SDK nor pytest, so it also runs under Pyodide.

Token counts are estimated as one token per four characters, rounded up. The
estimate is deterministic, which is what a test asserting on
``state.token_usage`` needs; it is not a tokeniser.
"""

import asyncio
import copy
import hashlib
import inspect
import json
import math
import re
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence, Union

from pydantic import BaseModel

from kavalai.db import ModelCallStat as ModelCallRecord
from kavalai.llm_clients import registry
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    ChatHistory,
    ChatMessage,
    LlmClientException,
    LlmClientParameters,
    ModelStatsReceiver,
)
from kavalai.llm_clients.common import create_model_call_stat
from kavalai.llm_clients.embeddings import BaseEmbeddingClient, Embeddings
from kavalai.llm_clients.streamer import Streamer, ValueStreamer
from kavalai.normalizer import Normalizer, get_default_normalizer

__all__ = [
    "ScriptedLlmClient",
    "ScriptedCall",
    "Interrupted",
    "FakeEmbeddingClient",
    "FakeProviders",
    "fake_providers",
]

_WORD = re.compile(r"\w+")


def _estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)


@dataclass
class Interrupted:
    """A reply that streams ``text`` and then fails with ``error``.

    It stands for a provider connection that drops mid-answer. When ``error``
    is one the retry policy treats as transient, the base client emits a
    ``restart`` chunk, consumers discard ``text``, and the next scripted reply
    answers the retry.

    Attributes:
        text: The partial output streamed before the failure.
        error: The exception raised once ``text`` has been streamed.
    """

    text: str
    error: BaseException


Reply = Any
Responder = Callable[[list[ChatMessage], Optional[type[BaseModel]]], Any]


@dataclass
class ScriptedCall:
    """One request a :class:`ScriptedLlmClient` received.

    Every attempt is a call, a retried one included, because every attempt
    consumes a reply.

    Attributes:
        model: The ``provider/model`` name the call was recorded under.
        messages: The chat history the client was given.
        response_model: The structured-output model requested, or ``None``.
        parameters: The parameters the client was built with, exactly as
            passed — sampling, timeouts and anything added later.
        reply: The scripted reply that answered the call.
    """

    model: str
    messages: list[ChatMessage]
    response_model: Optional[type[BaseModel]]
    parameters: LlmClientParameters
    reply: Reply = None

    @property
    def prompt(self) -> str:
        """The text of every message, joined by newlines."""
        return "\n".join(
            message.content for message in self.messages if message.content
        )


def _is_exception(reply: Reply) -> bool:
    return isinstance(reply, BaseException) or (
        isinstance(reply, type) and issubclass(reply, BaseException)
    )


def _reply_text(reply: Reply) -> str:
    """The text a provider would send for ``reply``.

    A string is sent verbatim, so a test can script malformed or truncated
    JSON; everything else is serialised.
    """
    if isinstance(reply, str):
        return reply
    if isinstance(reply, BaseModel):
        return reply.model_dump_json()
    return json.dumps(reply)


class ScriptedLlmClient(BaseLlmClient):
    """An LLM client that answers from a script instead of a provider.

    Only the provider call is replaced. The reply is streamed through the
    base client's :class:`~kavalai.Streamer`, retry and restart handling, so
    ``stream_output``, ``stream_delta`` and the partial-JSON parsing of
    structured output run as they do against a real model.

    The client does not validate a reply. Like a provider, it sends text, and
    the consumer — the engine, :meth:`~kavalai.BaseLlmClient.chat_completions`
    or the agent — validates it into the requested response model. A scripted
    reply that does not fit the model therefore fails with the same error a
    model's reply would.

    A reply is one of:

    - a string, streamed verbatim;
    - a dict, list or other JSON value, or a Pydantic model instance,
      streamed as its JSON;
    - an exception instance or class, raised in place of an answer, and
      recorded as a failed call;
    - an :class:`Interrupted`, which streams its text and then raises.

    An exception the retry policy treats as transient (a provider SDK's
    rate-limit or connection error) is retried after the policy's backoff, and
    the next reply answers the retry. Any other exception fails the call.

    ``max_output_tokens`` in the parameters is honoured as a provider honours
    it: a reply estimated at more tokens than the cap is streamed up to the
    cap, and the call then raises
    :class:`~kavalai.llm_clients.base_client.OutputTruncatedError`.

    The instance is also a client factory with the signature the engine's
    ``client_factory=`` expects: calling it returns a client bound to the
    model, parameters and statistics receiver of that call, which shares this
    client's script. The engine builds a client for every node it executes, so
    the replies are consumed in the order the nodes run, and ``calls`` lists
    every :class:`ScriptedCall` received by this client and the clients bound
    from it, in order.

    Args:
        replies: A sequence of replies consumed in order, or a function
            ``(messages, response_model) -> reply`` called for every call. The
            function may be a coroutine function.
        chunk_size: Characters per streamed partial.
        model: Model name recorded in the statistics.
        provider: Provider prefix recorded in the statistics.
        llm_client_parameters: Parameters recorded with each call.
        model_stats_receiver: Where each call's statistics are reported.

    Raises:
        TypeError: ``replies`` is a single reply rather than a sequence of
            them.
        ValueError: ``chunk_size`` is smaller than one.
    """

    def __init__(
        self,
        replies: Union[Sequence[Reply], Responder],
        *,
        chunk_size: int = 8,
        model: str = "scripted",
        provider: str = "fake",
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
    ):
        super().__init__(llm_client_parameters, model_stats_receiver)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")
        if callable(replies):
            self._responder: Optional[Responder] = replies
            self._replies: Optional[deque] = None
        elif isinstance(replies, (str, bytes, dict, BaseModel)):
            raise TypeError(
                "replies must be a sequence of replies or a function; wrap a "
                "single reply in a list."
            )
        else:
            self._responder = None
            self._replies = deque(replies)
        self.chunk_size = chunk_size
        self.model = model
        self.provider = provider
        self.calls: list[ScriptedCall] = []

    @classmethod
    def from_model(cls, model: str, *args: Any, **defaults: Any):
        """Refuse construction from a model name alone.

        A scripted client is nothing without its script, which a registry
        cannot supply. Register an instance with :func:`fake_providers`.
        """
        raise TypeError(
            "ScriptedLlmClient needs a script, so the class cannot be "
            "registered; register an instance with fake_providers(llm=client)."
        )

    @property
    def remaining(self) -> Optional[int]:
        """Replies not yet consumed; ``None`` when a function answers."""
        return None if self._replies is None else len(self._replies)

    def __call__(
        self,
        model: str,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
    ) -> "ScriptedLlmClient":
        """Build a client for ``model``, as the engine's factory does.

        ``model`` is a ``provider/model`` identifier. Its provider part is
        recorded as the provider, so a workflow naming ``openai/gpt-5`` has its
        calls recorded under that name.
        """
        provider, separator, name = model.partition("/")
        if not separator:
            provider, name = self.provider, model
        return self.bind(name, parameters, stats_receiver, provider=provider)

    def bind(
        self,
        model: str,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
        *,
        provider: Optional[str] = None,
    ) -> "ScriptedLlmClient":
        """A client sharing this one's script and calls, for another model.

        Parameters and the statistics receiver not given are this client's.
        """
        client = copy.copy(self)
        BaseLlmClient.__init__(
            client,
            parameters or self.parameters,
            self.model_stats_receiver if stats_receiver is None else stats_receiver,
        )
        client.model = model
        client.provider = provider or self.provider
        return client

    async def _next_reply(
        self, messages: list[ChatMessage], response_model: Optional[type[BaseModel]]
    ) -> Reply:
        if self._responder is not None:
            reply = self._responder(messages, response_model)
            return await reply if inspect.isawaitable(reply) else reply
        if not self._replies:
            raise LlmClientException(
                f"ScriptedLlmClient has no reply left for call {len(self.calls)}; "
                "every reply in the script has been consumed."
            )
        return self._replies.popleft()

    async def _stream_text(
        self,
        streamer: Streamer,
        response_model: Optional[type[BaseModel]],
        text: str,
    ) -> ValueStreamer:
        """Push ``text`` into a fresh value streamer, one chunk at a time.

        Control is yielded between chunks, as it is while a provider's stream
        is read, so the events of concurrent nodes interleave.
        """
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )
        for start in range(0, len(text), self.chunk_size):
            await value_streamer.stream_partial(text[start : start + self.chunk_size])
            await asyncio.sleep(0)
        return value_streamer

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[type[BaseModel]],
        streamer: Streamer,
    ):
        started = time.perf_counter()
        messages = list(chat_history.messages)
        call = ScriptedCall(
            model=self.stat_model_name(),
            messages=messages,
            response_model=response_model,
            parameters=self.parameters,
        )
        self.calls.append(call)
        call.reply = reply = await self._next_reply(messages, response_model)

        if isinstance(reply, Interrupted):
            await self._stream_text(streamer, response_model, reply.text)
            raise reply.error
        if _is_exception(reply):
            raise reply

        text = _reply_text(reply)
        request_data = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "response_model": response_model.__name__ if response_model else None,
            "parameters": self.parameters.model_dump(exclude_none=True),
        }
        prompt_tokens = _estimate_tokens(call.prompt)
        cap = self.parameters.max_output_tokens
        if cap is not None and _estimate_tokens(text) > cap:
            partial = text[: cap * 4]
            await self._stream_text(streamer, response_model, partial)
            raise self._output_truncated(
                "max_output_tokens",
                partial,
                request_data=request_data,
                prompt_tokens=prompt_tokens,
                completion_tokens=cap,
            )

        value_streamer = await self._stream_text(streamer, response_model, text)
        await value_streamer.stream_complete()
        await self._record_completed_call(
            request_data=request_data,
            response_data=text,
            started=started,
            prompt_tokens=prompt_tokens,
            completion_tokens=_estimate_tokens(text),
        )


class FakeEmbeddingClient(BaseEmbeddingClient):
    """An embedding client that hashes words into a fixed number of dimensions.

    Each word, lowercased, is hashed to one dimension and counted there; the
    count vector is scaled to unit length, as provider embeddings are. The
    vectors are deterministic across processes and exist for any text, and a
    query that shares words with a document lies closer to it than to one
    that does not. They are lexical, not semantic: a paraphrase with no word in
    common is not recognised.

    Normalisation follows the real clients: with ``normalize=True`` the
    vectors pass through ``normalizer``, or the default normaliser when none
    is given.

    ``calls`` lists the texts of every batch embedded by this client and the
    clients bound from it, in order.

    Args:
        model: Model name recorded in the statistics.
        dimension: Length of every vector.
        provider: Provider prefix recorded in the statistics.

    Raises:
        ValueError: ``dimension`` is smaller than one.
    """

    def __init__(
        self, model: str = "hashing", dimension: int = 8, *, provider: str = "fake"
    ):
        super().__init__(model)
        if dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}.")
        self.dimension = dimension
        self.provider = provider
        self.calls: list[list[str]] = []

    def bind(
        self, model: str, *, provider: Optional[str] = None
    ) -> "FakeEmbeddingClient":
        """A client sharing this one's dimension and calls, for another model."""
        client = copy.copy(self)
        client.model = model
        client.provider = provider or self.provider
        return client

    def vector(self, text: str) -> list[float]:
        """The unit-length embedding of ``text``.

        A text without a word character is hashed whole, so no text maps to
        the zero vector, whose cosine distance is undefined.
        """
        counts = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()) or [text]:
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            counts[int.from_bytes(digest, "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in counts))
        return [value / norm for value in counts]

    async def compute_embeddings(
        self,
        texts: list[str],
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        **kwargs,
    ) -> tuple[Embeddings, ModelCallRecord]:
        started = time.perf_counter()
        texts = list(texts)
        self.calls.append(texts)
        embeddings = [self.vector(text) for text in texts]
        if normalize:
            embeddings = (normalizer or get_default_normalizer()).transform(embeddings)
        stats = create_model_call_stat(
            call_type="embedding",
            model=f"{self.provider}/{self.model}",
            duration_seconds=time.perf_counter() - started,
            batch_size=len(texts),
            total_tokens=sum(_estimate_tokens(text) for text in texts),
        )
        return embeddings, stats


class FakeProviders(NamedTuple):
    """The clients :func:`fake_providers` registered.

    Attributes:
        llm: The scripted LLM client, or ``None`` when none was registered.
        embedding: The embedding client.
    """

    llm: Optional[ScriptedLlmClient]
    embedding: FakeEmbeddingClient


def _swap_registration(
    target_registry: registry.Registry, name: str, target: Callable[..., Any]
) -> Callable[[], None]:
    """Register ``target`` as ``name`` and return the function that undoes it.

    The previous entry, its registration defaults and its resolved-import cache
    are put back exactly. Unregistering before registering is what keeps the
    swap from logging the re-registration warning a replaced name produces.
    """
    previous = {
        store: getattr(target_registry, store)[name]
        for store in ("_targets", "_defaults", "_resolved")
        if name in getattr(target_registry, store)
    }
    target_registry.unregister(name)
    target_registry.register(name, target)

    def restore() -> None:
        target_registry.unregister(name)
        for store, value in previous.items():
            getattr(target_registry, store)[name] = value

    return restore


@contextmanager
def fake_providers(
    llm: Optional[ScriptedLlmClient] = None,
    embedding: Optional[FakeEmbeddingClient] = None,
    name: str = "fake",
) -> Iterator[FakeProviders]:
    """Register fake clients as the provider ``name`` inside a ``with`` block.

    Inside the block ``name/<model>`` resolves to ``llm`` wherever an LLM model
    is named — a workflow's ``llm_model``, :func:`~kavalai.make_client` — and
    to ``embedding`` wherever an embedding model is, such as a RAG service's
    ``model``. Each resolution returns a client bound to the named model that
    shares the original's script and calls.

    On exit the registries are restored exactly: a name that was registered
    before, a built-in included, gets its previous registration back, and one
    that was not is removed. ``name="openai"`` therefore runs an unchanged
    workflow that names ``openai/...`` against the script.

    The function is also the body of a pytest fixture::

        @pytest.fixture
        def providers():
            with fake_providers(llm=ScriptedLlmClient([...])) as fakes:
                yield fakes

    Args:
        llm: The LLM client to register. ``None`` registers no LLM provider.
        embedding: The embedding client to register. ``None`` registers a new
            :class:`FakeEmbeddingClient`.
        name: The provider name to register both under.

    Yields:
        The registered clients.

    Raises:
        RegistryError: ``name`` is not a valid provider name.
    """
    embedding = embedding or FakeEmbeddingClient()

    def scripted_llm(
        model: str,
        parameters: Optional[LlmClientParameters] = None,
        stats_receiver: Optional[ModelStatsReceiver] = None,
    ) -> ScriptedLlmClient:
        return llm.bind(model, parameters, stats_receiver, provider=name)

    def fake_embedding(model: str) -> FakeEmbeddingClient:
        return embedding.bind(model, provider=name)

    restores = []
    try:
        if llm is not None:
            restores.append(
                _swap_registration(registry.llm_providers, name, scripted_llm)
            )
        restores.append(
            _swap_registration(registry.embedding_providers, name, fake_embedding)
        )
        yield FakeProviders(llm, embedding)
    finally:
        for restore in reversed(restores):
            restore()
