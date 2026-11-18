"""A program written against Kaval.AI's public surface, as a user would write it.

Never executed: ``tests/test_typing.py`` runs ``mypy --strict`` on it, which is
what a consumer's own type check does once the package ships ``py.typed``. A
public function that loses its annotations, or a name that stops being
exported, fails that check here rather than in a consumer's CI. Extend it when
the public surface grows.
"""

from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import kavalai
from kavalai import (
    Agent,
    AnthropicClient,
    BaseEmbeddingClient,
    BaseLlmClient,
    BaseRagService,
    ChatHistory,
    ChatMessage,
    FunctionKernel,
    GeminiClient,
    LlmClientParameters,
    ModelCallStat,
    ModelStatsReceiver,
    OllamaClient,
    OpenAIClient,
    RagServiceResult,
    RunContext,
    SqliteRagService,
    StreamContent,
    WorkflowBuilder,
    WorkflowEngine,
    WorkflowException,
    WorkflowGraph,
    WorkflowState,
    db_manager,
    make_client,
    make_embedding_client,
    make_rag_service,
    pythontool,
    register_embedding_provider,
    register_llm_provider,
    register_rag_service,
    registered_llm_providers,
)
from kavalai.agent_service import AgentService
from kavalai.client import AgentClient
from kavalai.eval import (
    EvalResult,
    EvalSuite,
    JudgeEvaluator,
    SimpleEvaluator,
    load_suite,
    run_suite,
)
from kavalai.rag import CollectionRagService, rag_service_from_uri
from kavalai.server import (
    AgentRequest,
    create_agent_app,
    create_agent_router,
    public_events,
    sse_response,
)
from kavalai.testing import FakeEmbeddingClient, ScriptedLlmClient, fake_providers
from kavalai.text import Chunk, ParsedPage, chunk_markdown, parse_html
from kavalai.widget import asset_path, widget_dir
from kavalai.workflow import render_workflow_svg
from kavalai.workflow.models import WorkflowStreamEvent
from kavalai.workflow.tasklog import (
    MemoryTaskLogger,
    ModelCallRecord,
    PostgresTaskLogger,
    SqliteTaskLogger,
    TaskLogger,
    TaskRecord,
    TeeTaskLogger,
)


class Message(BaseModel):
    user_message: str


class Reply(BaseModel):
    agent_response: str


@pythontool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Meter(TaskLogger):
    """A custom task logger, as in the observability guide."""

    def __init__(self) -> None:
        super().__init__(record_nodes=False, record_payloads=False)
        self.tokens = 0

    async def write_node(self, **record: Any) -> None:
        """Not called: the meter records no nodes."""

    async def write_model_call(
        self,
        stats: ModelCallStat,
        *,
        agent_id: Optional[str],
        session_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        self.tokens += stats.total_tokens or 0


class Printer(ModelStatsReceiver):
    def receive_model_stats(self, stats: ModelCallStat) -> None:
        print(stats.model, stats.total_tokens)


def build(scripted: ScriptedLlmClient, tasklog: TaskLogger) -> WorkflowEngine:
    builder: WorkflowBuilder = (
        WorkflowBuilder("Village guide", llm_model="openai/gpt-5.6-luna")
        .data_model("input", Message)
        .data_model("output", Reply)
        .python_function("add", "consumer.add")
        .start("reply")
        .llm(
            "reply",
            prompt="Be concise.",
            inputs={"m": "input"},
            output="output",
            next="end",
        )
        .end()
    )
    graph: WorkflowGraph = builder.build()
    svg: str = render_workflow_svg(graph)
    print(len(svg))
    engine: WorkflowEngine = builder.build_engine(
        task_logger=tasklog, client_factory=scripted, run_timeout=30.0
    )
    return engine


async def run(engine: WorkflowEngine) -> Optional[str]:
    memory = MemoryTaskLogger()
    state: WorkflowState = await engine.run(
        {"user_message": "hi"}, external_id="user-1", task_logger=memory
    )
    records: list[TaskRecord] = memory.records
    calls: list[ModelCallRecord] = memory.model_calls_for_run(state.run_id)
    print(len(records), len(calls), state.token_usage, state.duration_seconds)

    stream: AsyncGenerator[WorkflowStreamEvent, None] = engine.run_stream(
        {"user_message": "hi"}, session_id=state.session_id
    )
    try:
        async for event in stream:
            kind: str = event.type
            print(kind, event.name)
    except WorkflowException as error:
        print(str(error))
    return state.session_id


async def lifecycle(yaml_text: str) -> None:
    async with WorkflowEngine.from_yaml(yaml_text) as engine:
        await engine.run({"user_message": "hi"})
    other: WorkflowEngine = await WorkflowEngine.from_yaml_path("a.yaml").connect()
    await other.aclose()


async def clients() -> None:
    parameters = LlmClientParameters(temperature=0.0, max_output_tokens=200)
    client: BaseLlmClient = make_client("openai/gpt-5.6-luna", parameters, Printer())
    history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])
    answer = await client.chat_completions(chat_history=history, response_model=Reply)
    print(answer)
    streamer = await client.stream_chat_completions(chat_history=history)
    async for chunk in streamer:
        content: StreamContent = chunk
        print(content.type, content.value)

    native: tuple[type[BaseLlmClient], ...] = (
        OpenAIClient,
        GeminiClient,
        AnthropicClient,
        OllamaClient,
    )
    print(native)

    embedder: BaseEmbeddingClient = make_embedding_client(
        "openai/text-embedding-3-small"
    )
    vectors = await embedder.compute_embeddings(["a", "b"])
    print(len(vectors))

    register_llm_provider("mycorp", "mycorp.clients.MyClient")
    register_embedding_provider("mycorp", FakeEmbeddingClient)
    register_rag_service("docs", rag_service_from_uri, uri="sqlite:///docs.db")
    names: list[str] = sorted(registered_llm_providers())
    print(names)


async def agent(client: BaseLlmClient) -> None:
    kernel = FunctionKernel()
    kernel.register_python_tool("add", add)
    result = await kernel.call_tool("python://add", {"a": 1, "b": 2})
    descriptions = await kernel.get_tool_descriptions(allowed_tools=["python://add"])
    print(result, descriptions)

    worker = Agent(client, kernel=kernel, run_context=RunContext(), allowed_tools=["*"])
    output = await worker.prompt("Add 1 and 2.", response_model=Reply, max_steps=3)
    print(output)
    async for chunk in worker.prompt_stream("Add 1 and 2.", stream_output=True):
        print(chunk.name, chunk.type)
    await kernel.close()


async def rag() -> None:
    service: CollectionRagService = rag_service_from_uri(
        "sqlite:///docs.db", model="fake/hashing"
    )
    sqlite = SqliteRagService("docs.db", model="fake/hashing", read_only=True)
    sqlite.embedding_client = FakeEmbeddingClient()
    entry: dict[str, Any] = await service.index(
        "Open from noon.", {"lang": "en"}, collection_name="faq", source_id="hours"
    )
    hits: list[RagServiceResult] = await service.query(
        "When is it open?", top_k=3, collection_name="faq", match={"lang": "en"}
    )
    for hit in hits:
        print(hit.similarity, hit.source_id, hit.content, entry["id"])
    base: BaseRagService = make_rag_service("docs")
    if base.supports("count_entries"):
        count: int = await base.count_entries("faq")
        print(count)
    await service.delete_by_source_id("faq", "hours")


async def persistence() -> None:
    await db_manager.init_sqlite()
    service = AgentService(db_manager.get_sqlite_sessionmaker())
    sqlite_log = SqliteTaskLogger("tasklog.db", record_payloads=False)
    postgres_log = PostgresTaskLogger(service, max_payload_bytes=1024)
    both = TeeTaskLogger(sqlite_log, postgres_log, Meter())
    await both.flush()
    sessions = await service.list_sessions(agent_ids=[])
    print(sessions)


def serve(engine: WorkflowEngine) -> FastAPI:
    app: FastAPI = create_agent_app(engine, event_filter=public_events)
    app.include_router(create_agent_router(engine))

    async def custom(request: AgentRequest[Message]) -> Any:
        return sse_response(engine.run_stream(request.data.model_dump()))

    print(custom, widget_dir(), asset_path("kaval-chatbot.js"))
    return app


async def evaluate() -> None:
    simple = SimpleEvaluator("http://localhost:25000", tag="pr-1")
    result: EvalResult = await simple.evaluate(
        {"user_message": "hi"}, {"agent_response": {"contains": "hello"}}
    )
    if not result.passed:
        print(result.reason)
    judge = JudgeEvaluator("http://localhost:25000")
    verdict: EvalResult = await judge.evaluate(
        {"user_message": "hi"}, expected="Is polite."
    )
    print(bool(verdict))
    suite: EvalSuite = load_suite("eval_cases.yaml")
    results: list[EvalResult] = await run_suite(suite, "http://localhost:25000")
    print(len(results))

    remote = AgentClient("http://localhost:25000", username="u", password="p")
    reply: BaseModel = await remote.run_agent(Message(user_message="hi"))
    print(reply)


def offline() -> None:
    scripted = ScriptedLlmClient([{"agent_response": "From noon."}])
    with fake_providers(llm=scripted, embedding=FakeEmbeddingClient()) as fakes:
        print(fakes, scripted.calls, scripted.remaining)


def text() -> None:
    page: ParsedPage = parse_html("<h1>Title</h1><p>Body</p>")
    chunks: list[Chunk] = chunk_markdown(page.markdown, target_chars=800)
    print(page.title, [chunk.text for chunk in chunks], kavalai.__all__)
