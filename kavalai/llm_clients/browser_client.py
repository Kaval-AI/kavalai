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

import json
import time
from typing import Optional, Type

from pydantic import BaseModel

from kavalai.idb import is_pyodide
from kavalai.llm_clients.base_client import (
    ensure_user_turn,
    BaseLlmClient,
    ChatHistory,
    LlmClientException,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
)
from kavalai.llm_clients.streamer import Streamer

# Name of the JavaScript bridge the host page must expose on the global scope
# (``window.kavalBrowserLLM``). See ``webwidget/`` for a WebLLM-backed
# reference implementation of the bridge.
BRIDGE_GLOBAL = "kavalBrowserLLM"


def get_browser_bridge():
    """Return the page's JS bridge object (``window.kavalBrowserLLM``).

    Shared by the browser LLM client (``.chat``) and the browser embedding
    client (``.embed``). Raises a helpful :class:`LlmClientException` when not
    running under Pyodide, or when the page has not loaded a bridge.
    """
    if not is_pyodide():
        raise LlmClientException(
            "The browser bridge only works inside a Pyodide/browser runtime. "
            "Use an 'openai/', 'gemini/' or 'ollama/' model outside the browser."
        )

    import js  # type: ignore[import-not-found]  # provided by Pyodide

    bridge = getattr(js, BRIDGE_GLOBAL, None)
    if bridge is None:
        raise LlmClientException(
            f"No in-browser LLM engine found (window.{BRIDGE_GLOBAL} is "
            "undefined). Load a WebLLM bridge in the page before using a "
            "'browser/...' model — see webwidget/ for an example."
        )
    return bridge


class BrowserLLMClient(BaseLlmClient):
    """LLM client that runs entirely in the browser, with no network calls.

    Inference happens inside the page through a tiny JavaScript bridge exposed
    on ``window.kavalBrowserLLM``, typically backed by a WebGPU engine such as
    `WebLLM <https://github.com/mlc-ai/web-llm>`_. This makes Kaval.AI's LLM
    nodes usable inside Pyodide with **no API key, no provider account and no
    CORS constraints** — the model is downloaded once and cached by the browser.

    Use it through ``make_client("browser/<model-id>")`` or construct it
    directly. ``<model-id>`` is passed verbatim to the bridge (e.g. a WebLLM
    model id like ``Llama-3.2-1B-Instruct-q4f32_1-MLC``).

    The bridge contract is a single async function::

        window.kavalBrowserLLM.chat(requestJson) -> Promise<resultJson>

    where ``requestJson`` is a JSON string of ``{model, messages, temperature,
    top_p, response_format?}`` and ``resultJson`` is a JSON string of either
    ``{content, usage}`` or ``{error}``. Exchanging plain JSON strings keeps the
    Python<->JS boundary free of proxy-conversion surprises.
    """

    provider = "browser"

    def __init__(
        self,
        model: str,
        llm_client_parameters: Optional[LlmClientParameters] = None,
        model_stats_receiver: Optional[ModelStatsReceiver] = None,
    ):
        """Initialize the browser client.

        Args:
            model: The bridge model id (e.g. a WebLLM id like
                ``Llama-3.2-1B-Instruct-q4f32_1-MLC``).
            llm_client_parameters: Optional sampling/timeout parameters.
            model_stats_receiver: Optional receiver for model call statistics.
        """
        super().__init__(llm_client_parameters, model_stats_receiver)
        self.model = model

    def _get_bridge(self):
        """Return the JS bridge object, or raise a helpful error if absent."""
        return get_browser_bridge()

    def _build_request(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
    ) -> dict:
        """Translate the chat history + options into the bridge request dict."""
        messages = [
            {"role": msg.role or "user", "content": msg.content or ""}
            for msg in ensure_user_turn(chat_history.messages)
        ]
        # WebLLM requires the final message to come from the user (or a tool);
        # a conversation that ends on a system message is rejected. ``prompt()``
        # produces exactly that — a lone system message — so relabel a trailing
        # system message as ``user`` to keep single-instruction prompts working.
        if messages and messages[-1]["role"] == "system":
            messages[-1]["role"] = "user"
        request: dict = {"model": self.model, "messages": messages}

        if self.parameters:
            if self.parameters.temperature is not None:
                request["temperature"] = self.parameters.temperature
            if self.parameters.top_p is not None:
                request["top_p"] = self.parameters.top_p

        if response_model is not None:
            # WebLLM (and OpenAI-compatible engines) constrain generation to the
            # given JSON schema when a json_object response_format is supplied.
            request["response_format"] = {
                "type": "json_object",
                "schema": response_model.model_json_schema(),
            }

        return request

    async def _run_chat_completions(
        self,
        chat_history: ChatHistory,
        response_model: Optional[Type[BaseModel]],
        streamer: Streamer,
    ):
        """Run the in-browser completion and stream the result back."""
        start_time = time.perf_counter()
        value_streamer = streamer.get_value_streamer(
            "response", response_model=response_model
        )

        bridge = self._get_bridge()
        request = self._build_request(chat_history, response_model)

        try:
            # ``bridge.chat`` resolves to a JS Promise; awaiting it in Pyodide
            # yields the resolved JSON string.
            raw = await bridge.chat(json.dumps(request))
        except Exception as exc:  # JsException or anything the bridge throws.
            raise LlmClientException(f"In-browser LLM call failed: {exc}") from exc

        data = json.loads(raw)
        if data.get("error"):
            raise LlmClientException(f"In-browser LLM error: {data['error']}")

        content = data.get("content") or ""
        await value_streamer.stream_partial(content)
        await value_streamer.stream_complete()

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

        duration = time.perf_counter() - start_time
        stats = ModelCallStat(
            call_type="llm",
            model=self.stat_model_name(),
            request_data=json.dumps(request, default=str),
            response_data=content,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_code=200,
        )
        await self._send_model_call_stats(stats)
