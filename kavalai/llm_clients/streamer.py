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
import io
import json
from typing import Optional, Type

from pydantic import BaseModel

from kavalai.llm_clients.common import safe_parse_json


class StreamerTimeoutException(Exception):
    """Raised when no stream chunk arrives within the configured timeout.

    Reported by :class:`Streamer` while waiting on its queue when a
    ``timeout_seconds`` is set; ``names`` lists the streamers still active when
    the timeout elapsed.
    """

    def __init__(self, names: list[str], timeout_seconds: float):
        self.names = names
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Streamer timed out after {timeout_seconds}s. Active streamers: {', '.join(names)}"
        )


class StreamContent(BaseModel):
    """
    ``StreamContent`` represents a streamed message from a ``Streamer``.

    Attributes:
        type: The type of stream message (e.g., 'partial', 'complete').
        name: The identifier for the stream source or target.
        value: The actual content string.
    """

    type: str
    name: str
    value: Optional[str] = None


class ValueStreamer:
    """
    A helper class to manage and push streaming content to an asyncio queue.

    Attributes:
        name: Default name for the stream chunks.
        queue: The asyncio.Queue where messages are placed.
    """

    def __init__(
        self,
        name: str,
        queue: asyncio.Queue,
        response_model: Optional[Type[BaseModel]] = None,
        stream_delta: bool = False,
        on_complete_callback: Optional[callable] = None,
    ):
        """
        Initialize the Streamer.

        Args:
            name: Name/label of the value.
            queue: Target queue for the JSON-serialized StreamContent.
        """
        self._name = name
        self._queue = queue
        self._response_model = response_model
        self._stream_delta = stream_delta
        self._buffer = io.StringIO()  # Will stay empty if stream_delta is True
        self._completed = False
        self._on_complete_callback = on_complete_callback

    def get_safe_value(self) -> str:
        """
        Safely parse and return the buffered content as JSON string if response_model is set, otherwise return as string.
        """
        if self._response_model:
            parsed = safe_parse_json(self._buffer.getvalue())
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed)
            return str(parsed)
        return self._buffer.getvalue()

    async def stream_partial(self, value: str):
        """
        Push a 'partial' chunk to the queue.

        Args:
            value: The partial content to stream.
        """
        if not self._stream_delta:
            self._buffer.write(value)
            value = self.get_safe_value()

        await self._queue.put(
            StreamContent(
                type="partial", name=self._name, value=value
            ).model_dump_json()
        )

    async def stream_complete(self):
        """
        Push a 'complete' chunk to the queue, indicating the stream has finished.

        In delta mode the chunk carries no value; otherwise it carries the full
        accumulated (safe-parsed) content.
        """
        if self._completed:
            raise RuntimeError(f"stream_complete() already called for '{self._name}'")

        self._completed = True
        if self._stream_delta:
            stream_content = StreamContent(
                type="complete", name=self._name
            ).model_dump_json()
        else:
            stream_content = StreamContent(
                type="complete", name=self._name, value=self.get_safe_value()
            ).model_dump_json()
        await self._queue.put(stream_content)

        if self._on_complete_callback:
            self._on_complete_callback()


class Streamer:
    def __init__(
        self, stream_delta: bool = False, timeout_seconds: Optional[float] = None
    ):
        self._stream_delta = stream_delta
        self._timeout_seconds = timeout_seconds
        self._queue = asyncio.Queue()
        self._active_streamer_names = []
        self._stop_iteration = False

    @property
    def _active_streamers(self) -> int:
        return len(self._active_streamer_names)

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    def get_value_streamer(
        self,
        name: str,
        stream_delta: Optional[bool] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> ValueStreamer:
        self._active_streamer_names.append(name)

        def on_complete():
            # ``reset_active()`` may have dropped the name already (a retried
            # attempt re-registers its value streamers), so discard-if-present.
            if name in self._active_streamer_names:
                self._active_streamer_names.remove(name)

        return ValueStreamer(
            name,
            self._queue,
            response_model,
            stream_delta or self._stream_delta,
            on_complete,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Optionally handle cleanup or ensure the queue is properly closed
        return False

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def stream_error(self, error: Exception):
        """
        Push an 'error' chunk to the queue.
        """
        await self._queue.put(
            StreamContent(
                type="error", name="error", value=str(error)
            ).model_dump_json()
        )

    def reset_active(self) -> None:
        """Forget all active value streamers.

        Called between retry attempts: the failed attempt's value streamers
        never complete, and without a reset their names would keep the
        completion accounting from ever reaching zero.
        """
        self._active_streamer_names.clear()

    async def stream_restart(self, value: Optional[str] = None):
        """Push a 'restart' chunk to the queue.

        Signals that the in-flight completion is starting over (e.g. a retry
        after a transient error): consumers should discard any partial content
        accumulated so far; it will be re-sent.
        """
        await self._queue.put(
            StreamContent(
                type="restart", name="response", value=value
            ).model_dump_json()
        )

    async def __anext__(self) -> StreamContent:
        """
        Async iterator protocol: get the next stream chunk from the queue.

        Returns:
            StreamContent: The next stream content chunk.

        Raises:
            StopAsyncIteration: When a 'complete' message is received or queue is done.
            StreamerTimeoutException: When the queue get times out.
        """
        if self._stop_iteration:
            raise StopAsyncIteration
        if self._timeout_seconds is not None:
            try:
                data = await asyncio.wait_for(
                    self._queue.get(), timeout=self._timeout_seconds
                )
            except asyncio.TimeoutError:
                raise StreamerTimeoutException(
                    self._active_streamer_names, self._timeout_seconds
                )
        else:
            data = await self._queue.get()
        stream_content = StreamContent.model_validate_json(data)
        if stream_content.type == "error":
            self._stop_iteration = True
            raise RuntimeError(stream_content.value)
        if stream_content.type == "complete" and self._active_streamers == 0:
            self._stop_iteration = True
        return stream_content
