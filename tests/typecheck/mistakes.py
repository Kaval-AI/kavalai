"""Calls into Kaval.AI that a type checker must reject.

Never executed. ``tests/test_typing.py`` asserts that mypy reports an error on
every line marked ``# expect: <code>`` and nowhere else, which is the proof
that the public surface is typed rather than ``Any``.
"""

from kavalai import (
    ChatHistory,
    ChatMessage,
    FunctionKernel,
    OpenAIClient,
    WorkflowBuilder,
    WorkflowEngine,
    WorkflowState,
    make_client,
)
from kavalai import WorkflowEngin  # expect: attr-defined
from kavalai.eval import SimpleEvaluator
from kavalai.workflow.tasklog import MemoryTaskLogger


async def mistakes(engine: WorkflowEngine) -> None:
    state: WorkflowState = await engine.run("hi")  # expect: arg-type
    print(state.sesion_id)  # expect: attr-defined
    await engine.run({"user_message": "hi"}, timeout="soon")  # expect: arg-type
    async for event in engine.run_stream({"user_message": "hi"}):
        print(event.kind)  # expect: attr-defined

    client = make_client(42)  # expect: arg-type
    history = ChatHistory(messages=[ChatMessage(role="user", content="hi")])
    answer = await client.chat_completions(chat_history=history)
    text: int = answer  # expect: assignment
    print(text, OpenAIClient.no_such_method)  # expect: attr-defined

    kernel = FunctionKernel()
    kernel.register_python_tool(1, print)  # expect: arg-type
    WorkflowBuilder("x").llm("reply", prompt="p")  # expect: call-arg
    MemoryTaskLogger().for_run(run_id=3)  # expect: arg-type
    SimpleEvaluator()  # expect: call-arg
