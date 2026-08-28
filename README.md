<img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an opinionated Python library for building well-defined, testable and robust
agentic workflows, chatbots and tools.

One small interface for every model
provider, typed inputs and outputs everywhere, and a workflow engine that
records every run in a database you own.

- **Any model, one interface** — OpenAI, Google, Anthropic, Ollama and
  in-browser WebLLM
  ([LLM clients](https://docs.kaval.ai/tutorials/llm_clients.html),
  [running in the browser](https://docs.kaval.ai/tutorials/run_in_browser.html)).
- **Typed end to end** — inputs, outputs and tool calls are Pydantic models
  ([typed inputs and outputs](https://docs.kaval.ai/tutorials/agents.html#typed-inputs-and-outputs)).
- **Retrieval built in** — RAG over SQLite or PostgreSQL/pgvector, with local
  or hosted embeddings ([RAG](https://docs.kaval.ai/tutorials/rag.html)).
- **Workflows as graphs** — conditional routing, parallel fan-out, agents and
  tool calls, in Python or YAML
  ([workflows](https://docs.kaval.ai/tutorials/workflow.html)).
- **Tools your way** — Python functions, REST endpoints and MCP servers
  ([tools](https://docs.kaval.ai/guides/tools.html)).
- **Streaming** from a single model call to a whole workflow
  ([streaming](https://docs.kaval.ai/tutorials/streamer.html)).
- **Full observability** — every session, run, node and model call is recorded
  and browsable in the backoffice UI
  ([data model](https://docs.kaval.ai/guides/data_model.html)).

The design, and the reasoning behind it, is set out in
[Architecture](https://docs.kaval.ai/tutorials/architecture.html).

## Install

```
pip install "kavalai[common]"
```

`common` brings the provider SDKs, RAG, MCP and the servers. The bare
`kavalai` package stays small enough to run in the browser. See
[Installation](https://docs.kaval.ai/tutorials/installation.html).

## Getting started

### Call a model

Name a model as `provider/model` and `make_client` does the rest, reading the
API key from the environment:

```python
import os

os.environ["OPENAI_API_KEY"] = '...'
os.environ["GEMINI_API_KEY"] = '...'
os.environ["ANTHROPIC_API_KEY"] = '...'
```

```python
from kavalai import make_client

client = make_client("openai/gpt-5.4-mini")
answer = await client.prompt("What is the capital of Estonia?")
print (answer)
```

Response:
```
The capital of Estonia is Tallinn.
```

Ask for a [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/)
model and you get a validated object back, not a string to parse — with any
provider:

```python
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str
    fun_fact: str

city = await client.prompt("Describe Tallinn.", response_model=City)
print(city)
```

Response:
```
name='Tallinn' country='Estonia' fun_fact='Medieval Old Town of Tallinn is one of the best-preserved in Northern Europe.'
```

[Streaming](https://docs.kaval.ai/tutorials/llm_clients.html#streaming-responses),
[timeouts and retries](https://docs.kaval.ai/tutorials/llm_clients.html#timeouts-and-retries)
and [token statistics](https://docs.kaval.ai/tutorials/llm_clients.html#model-statistics-and-observability)
come with the same client. Tool calling and multi-step loops are in
[Agents & tools](https://docs.kaval.ai/tutorials/agents.html).

### Using retrieval-augmented generation (RAG)

RAG lets a model answer from your own data: index it once, retrieve the
closest passages for each question, and put them in the prompt. Kaval.AI
ships a `RagService` that does the indexing and the retrieval.

Take some facts about a fictional village:
```python
FACTS = """\
Green Village has 104 residents.
Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.
President of Green Village is Thomas Cook (born 12.04.1994).
Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).
The annual Turnip Festival takes place every year on the third Saturday of October.
The village bakery, run by Greta Lindqvist (born 27.11.1968), sells exactly 340 loaves every week.
Green Village's football team, FC Green Rovers, has won the regional cup twice (1997 and 2013).
Green Village's only pub, The Rusty Anchor, has been operating since 1923.
""".splitlines()
```

Index them in SQLite — in production, the same code runs on
PostgreSQL/pgvector. The embedding model is named `provider/model` too, and
`fastembed` runs locally with no API key:

```python
from kavalai.rag import SqliteRagService

rag = SqliteRagService(":memory:", model="fastembed/BAAI/bge-small-en-v1.5")
await rag.index_batch(
    texts=FACTS,
    metadata_list=[{"village": "Green Village"}] * len(FACTS),
    source_ids=[f"fact-{i}" for i in range(len(FACTS))],
)
```

Query it and the closest facts come back, ranked by similarity:

```python
question = "How old was the Green Village's oldest resident on 2025 Turnip Festival?"

hits = await rag.query(question, top_k=5)
for hit in hits:
    print(f"{hit.similarity:.2f}  {hit.content}")

```

```
0.79  Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).
0.70  Green Village has 104 residents.
0.68  President of Green Village is Thomas Cook (born 12.04.1994).
0.67  Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.
0.62  The annual Turnip Festival takes place every year on the third Saturday of October.
```

Put the hits in a prompt and the model answers from them:

```python
PROMPT = """Answer the question using only the facts below.

Facts:
{context}

Question: {question}
"""

context = "\n".join(f"- {hit.content}" for hit in hits)
client = make_client("openai/gpt-5.4-mini")
print(await client.prompt(PROMPT.format(context=context, question=question)))
```

Response:
```
Agnes Whitlow was born on 02.06.1929.
The Turnip Festival in 2025 took place on the third Saturday of October, which was **18 October 2025**.

So her age on that date was **96 years old**.
```

The [RAG tutorial](https://docs.kaval.ai/tutorials/rag.html) goes on to
[chunked documents](https://docs.kaval.ai/tutorials/rag.html#chunked-documents-and-keep-best),
[batched queries](https://docs.kaval.ai/tutorials/rag.html#batched-queries) and
[similarity matrices](https://docs.kaval.ai/tutorials/rag.html#similarity-matrix),
and an index can even be pre-built and shipped to the browser.

### Building a workflow

The loop above is two steps and a line of glue. A **workflow** turns those
steps into a graph that Kaval.AI executes, records and serves. This one
retrieves the closest facts for the user's message, then writes the reply from
them — typed at both ends by Pydantic models:

```python
from pydantic import BaseModel

from kavalai.workflow import WorkflowBuilder


class Message(BaseModel):
    user_message: str


class Reply(BaseModel):
    agent_response: str


engine = (
    WorkflowBuilder("Green Village support", llm_model="openai/gpt-5.4-mini")
    .data_model("input", Message)
    .data_model("output", Reply)
    .start("get_related_facts")
    .rag_query(
        "get_related_facts",
        query="{{ context.input.user_message }}",
        output="facts",
        top_k=5,
        store="content",
        next="reply",
    )
    .llm(
        "reply",
        prompt=(
            "You are the assistant of the Green Village tourist "
            "information centre. Answer using only these facts:\n"
            "{{ context.facts }}"
        ),
        inputs={"input": "input", "facts": "facts"},
        output="output",
        next="end",
    )
    .end()
    .build_engine(rag_services=rag)
)

state = await engine.run(
    {"user_message": "Who runs the bakery, and where can I get a pint?"}
)
print(state.output_data)
print(state.status, state.token_usage)
```

Response:
```
{'agent_response': 'The bakery is run by Greta Lindqvist. You can get a pint '
                   'at Green Village’s only pub, The Rusty Anchor.'}
completed {'model_calls': 1, 'prompt_tokens': 278,
           'completion_tokens': 39, 'total_tokens': 317}
```

The same graph can be written as YAML, served over REST with one call, and
evaluated against a file of test cases — `examples/green_village/` is exactly
this workflow as a chatbot, with 64 evaluation cases beside it. Conditional
routing, parallel fan-out, agents and tools are in the
[workflows tutorial](https://docs.kaval.ai/tutorials/workflow.html);
[Serving](https://docs.kaval.ai/tutorials/serving.html) and
[Evaluation](https://docs.kaval.ai/guides/evaluation.html) cover the rest.


## The backoffice UI

Everything a workflow does is recorded, and the backoffice UI lets you see
it: sessions, runs and model-call statistics, a step-by-step task debugger,
and a RAG explorer for browsing your indexes.

<a href="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png">
  <img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png" alt="Kaval.AI backoffice project page with database details and activity charts" width="400px"/>
</a>

For details, see
[Using the backoffice UI](https://docs.kaval.ai/ui/index.html).

## Documentation

The full documentation is at [docs.kaval.ai](https://docs.kaval.ai/) — every
tutorial is a runnable notebook.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
