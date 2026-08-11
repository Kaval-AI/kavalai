<img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an opinionated, elegant, production-grade Python library for building LLM-powered workflows,
chatbots and agents, and connecting them to databases and tooling — with a focus on observability and debuggability.


Features:
- Supports common commercial LLM providers such OpenAI, Google Gemini etc.
- Supports open source LLMs via Ollama backend.
- Agents can be run client-side in browser via WebLLM and Pyodide.
- Structured responses using Pydantic semantics.
- Streaming responses.
- Retrieval augmented generation (RAG) index on various storage engines.
- Fully featured workflow engine using conditional processing and tool calling.
- Workflows can be defined in YAML for better readability.
- Python tools, REST server endpoints (supports basic auth) and MCP support.

## Install

```
pip install kavalai
```

The core install is deliberately small. Pull in what you need with extras —
`kavalai[openai]`, `[gemini]`, `[anthropic]`, `[ollama]`, `[rag]`, `[postgres]`,
`[mcp]`, `[server]` — or take everything with `kavalai[all]`:

```
pip install "kavalai[openai,rag]"
```

## Getting started

### Call a model

Every provider sits behind one small async interface. `make_client` builds the
right client from a `provider/model` id and reads the matching API key
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) from the
environment:

```python
from kavalai import make_client

client = make_client("openai/gpt-5.4-mini")

answer = await client.prompt("Say hello in Estonian.")
```

Pass a Pydantic `response_model` and you get a validated object back instead of
a string to parse — with any provider:

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    fun_fact: str

city = await client.prompt("Describe Tallinn.", response_model=City)
print(city.country, city.fun_fact)
```

### A chatbot grounded in your own documents

A retrieval-augmented chatbot is a `RagService` for retrieval plus a chat loop.
Here the index is a single SQLite file (`SqliteRagService`); swap in
`PostgresRagService` for a pgvector-backed one — the interface is the same:

```python
import asyncio

from kavalai import ChatHistory, ChatMessage, make_client
from kavalai.rag import SqliteRagService

SYSTEM_PROMPT = """You are a helpful support assistant.
Answer the user using only the context below. If the answer is not in the
context, say that you don't know.

Context:
{context}
"""


async def main() -> None:
    # A RAG index you filled earlier with your own documents.
    rag = SqliteRagService("handbook.db", model="fastembed/BAAI/bge-small-en-v1.5")
    client = make_client("openai/gpt-5.4-mini")

    history = ChatHistory(messages=[])
    while question := input("you > ").strip():
        # 1. Retrieve the passages that best match the question.
        hits = await rag.query(question, top_k=3, keep_best=True)
        context = "\n".join(f"- {hit.content}" for hit in hits)

        # 2. Answer, grounded in that context plus the conversation so far.
        history.messages.append(ChatMessage(role="user", content=question))
        answer = await client.chat_completions(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(
                        role="system", content=SYSTEM_PROMPT.format(context=context)
                    ),
                    *history.messages,
                ]
            )
        )
        history.messages.append(ChatMessage(role="assistant", content=answer))
        print(f"bot > {answer}\n")


asyncio.run(main())
```

Filling that index is just as short — embed once, query many times:

```python
await rag.index_batch(
    texts=["Refunds are issued within 14 days of purchase.", ...],
    metadata_list=[{"page": 12}, ...],
    source_ids=["refund-policy", ...],
)
```

### Orchestrate the steps in YAML

When one call is not enough, describe the graph — LLM calls, tools, agents and
branches — in YAML and let the engine run it, checkpointing state and recording
token statistics along the way:

```python
from kavalai import WorkflowEngine

WORKFLOW = """
name: Greeter
description: Greets the user by name.
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
nodes:
  - {name: start, type: start, next: reply}
  - name: reply
    type: llm
    llm_model: openai/gpt-5.4-mini
    prompt: |
      You are a warm, concise greeter. Read the user's message and write a
      one-sentence friendly greeting that uses their name.
    inputs:
      input: {type: context, value: input}
    output: output
    next: end
  - {name: end, type: end, output: output}
"""

engine = WorkflowEngine.from_yaml(WORKFLOW)
state = await engine.run({"user_message": "Hi, I'm Timo!"})

print(state.output_data["agent_response"])  # -> "Hi Timo! Nice to meet you!"
print(state.trace)                          # -> ['start', 'reply', 'end']
```

## Documentation

For full documentation, please visit [Kaval.AI Documentation](https://docs.kaval.ai/) —
streaming, tool calling, agents, MCP servers, observability and running in the
browser are all covered there with runnable examples.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
