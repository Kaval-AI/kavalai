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

# Kaval.AI public API.
#
# The names below are the supported, stable import surface. Everything a user
# typically needs is reachable as ``from kavalai import X``. The headline
# entry points are:
#
#   * ``WorkflowEngine`` / ``WorkflowBuilder`` -- define and run workflows.
#   * ``Agent``                                -- the multi-step tool-calling agent.
#   * ``FunctionKernel`` / ``pythontool``      -- register and call tools.
#   * ``OpenAIClient`` / ``GeminiClient`` / ``AnthropicClient`` / ``OllamaClient`` -- LLM backends.
#   * ``PostgresRagService``                   -- index and query embeddings.
#
# The persistence-layer ORM table classes (Agent row, Run, Task, ...) live in
# ``kavalai.db`` to keep the runtime names (``Agent`` the agent,
# ``ModelCallStat`` the stats model) unambiguous at the top level.

# --- Workflow engine -------------------------------------------------------
from kavalai.workflow import (
    WorkflowEngine,
    WorkflowBuilder,
    WorkflowState,
    WorkflowGraph,
    Node,
    StartNode,
    EndNode,
    LLMNode,
    AgentNode,
    FunctionNode,
    IfNode,
    SwitchNode,
    evaluate_expression,
    evaluate_bool,
    evaluate_value,
    ExpressionError,
    TaskLogger,
    StatsBridge,
    TokenAccumulator,
    SqliteTaskLogger,
)
from kavalai.workflow.clients import make_client

# --- Agent & tools ---------------------------------------------------------
from kavalai.agent import Agent, ToolCall
from kavalai.run_context import RunContext
from kavalai.workflow_model import (
    RestServer,
    McpServer,
    PythonFunction,
    TemplateModel,
    ArgumentInfo,
    WorkflowException,
)
from kavalai.functionkernel import (
    FunctionKernel,
    FunctionKernelException,
    pythontool,
)

# --- LLM & embedding clients ----------------------------------------------
from kavalai.llm_clients.base_client import (
    BaseLlmClient,
    LlmClientParameters,
    ChatHistory,
    ChatMessage,
    ModelCallStat,
    ModelStatsReceiver,
    ModelStatsLogger,
)
from kavalai.llm_clients.embeddings import (
    make_embedding_client,
    BrowserEmbeddingClient,
)

# The in-browser clients are part of the pyodide-compatible core (they only
# bridge to a JS engine at call time) so they are imported eagerly, unlike the
# provider clients below which pull in optional SDKs.
from kavalai.llm_clients.browser_client import BrowserLLMClient

# The provider LLM clients (``OpenAIClient`` / ``GeminiClient`` /
# ``AnthropicClient`` / ``OllamaClient``) pull in optional SDKs that are not
# part of the pyodide-compatible core (``openai`` / ``google-genai`` /
# ``anthropic`` / ``ollama``). They are resolved lazily via ``__getattr__``
# below so that ``import kavalai`` works in lightweight / pyodide environments
# where only the core dependencies are installed. Install the matching extra
# (e.g. ``kavalai[openai]``) to use them.
_LAZY_CLIENTS = {
    "OpenAIClient": ("kavalai.llm_clients.openai_client", "openai"),
    "GeminiClient": ("kavalai.llm_clients.gemini_client", "gemini"),
    "AnthropicClient": ("kavalai.llm_clients.anthropic_client", "anthropic"),
    "OllamaClient": ("kavalai.llm_clients.ollama_client", "ollama"),
}


def __getattr__(name: str):
    """Lazily import optional provider clients (PEP 562)."""
    target = _LAZY_CLIENTS.get(name)
    if target is not None:
        module_path, extra = target
        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires the optional '{extra}' dependency. "
                f"Install it with: pip install kavalai[{extra}]"
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module 'kavalai' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


# --- Streaming -------------------------------------------------------------
from kavalai.llm_clients.streamer import (
    Streamer,
    StreamContent,
    ValueStreamer,
    StreamerTimeoutException,
)

# --- RAG, normalization, persistence --------------------------------------
from kavalai.rag import (
    BaseRagService,
    PostgresRagService,
    RagServiceResult,
    SqliteRagService,
)
from kavalai.normalizer import Normalizer
from kavalai.db import EngineOptionsConflictError, db_manager

__all__ = [
    # Workflow engine
    "WorkflowEngine",
    "WorkflowBuilder",
    "WorkflowState",
    "WorkflowGraph",
    "Node",
    "StartNode",
    "EndNode",
    "LLMNode",
    "AgentNode",
    "FunctionNode",
    "IfNode",
    "SwitchNode",
    "evaluate_expression",
    "evaluate_bool",
    "evaluate_value",
    "ExpressionError",
    "TaskLogger",
    "StatsBridge",
    "TokenAccumulator",
    "SqliteTaskLogger",
    "make_client",
    # Agent & tools
    "Agent",
    "ToolCall",
    "RunContext",
    "RestServer",
    "McpServer",
    "PythonFunction",
    "TemplateModel",
    "ArgumentInfo",
    "WorkflowException",
    "FunctionKernel",
    "FunctionKernelException",
    "pythontool",
    # LLM & embedding clients
    "BaseLlmClient",
    "LlmClientParameters",
    "ChatHistory",
    "ChatMessage",
    "ModelCallStat",
    "ModelStatsReceiver",
    "ModelStatsLogger",
    "OpenAIClient",
    "GeminiClient",
    "AnthropicClient",
    "OllamaClient",
    "BrowserLLMClient",
    "make_embedding_client",
    "BrowserEmbeddingClient",
    # Streaming
    "Streamer",
    "StreamContent",
    "ValueStreamer",
    "StreamerTimeoutException",
    # RAG, normalization, persistence
    "BaseRagService",
    "PostgresRagService",
    "RagServiceResult",
    "SqliteRagService",
    "Normalizer",
    "db_manager",
    "EngineOptionsConflictError",
]
