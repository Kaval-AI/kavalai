<img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an opinionated and elegant Python library for building production-grade agentic workflows, chatbots and tools.


Features:
- Support for commercial (OpenAI, Google, Anthropic) and open-source LLM
  providers
  (see [LLM clients](https://docs.kaval.ai/tutorials/llm_clients.html)).
- Client-side execution in the browser through WebLLM and Pyodide
  (see [running in the browser](https://docs.kaval.ai/tutorials/run_in_browser.html)).
- Retrieval-augmented generation (RAG) over several storage engines
  (see [RAG](https://docs.kaval.ai/tutorials/rag.html)).
- Structured inputs, outputs, tool calls and responses expressed with Pydantic
  semantics
  (see [typed inputs and outputs](https://docs.kaval.ai/tutorials/agents.html#typed-inputs-and-outputs)).
- Streaming responses
  (see [streaming](https://docs.kaval.ai/tutorials/streamer.html)).
- A complete workflow engine with conditional routing, parallel fan-out and
  tool calling
  (see [workflows tutorial](https://docs.kaval.ai/tutorials/workflow.html),
  [workflow concepts](https://docs.kaval.ai/guides/workflows.html)).
- Python tools, REST server endpoints (with basic authentication) and MCP
  support
  (see [tools](https://docs.kaval.ai/guides/tools.html),
  [agent server API](https://docs.kaval.ai/api/server.html)).
- Every session, run, node and model call recorded in a database you own
  (see [architecture](https://docs.kaval.ai/tutorials/architecture.html),
  [data model](https://docs.kaval.ai/guides/data_model.html)).

See the [full documentation](https://docs.kaval.ai) for a more detailed
reference. The design of the library, and the reasoning behind it, is set out
in [Architecture](https://docs.kaval.ai/tutorials/architecture.html).

## Install

```
pip install "kavalai[common]"
```

`common` is the normal install: the provider SDKs, RAG and embeddings, the
Postgres drivers, MCP and the REST/SSE servers. The bare `kavalai` package stays
small and Pyodide-compatible, and `kavalai[common_web]` is its browser
counterpart. See
[Installation](https://docs.kaval.ai/tutorials/installation.html) for the full
table and provider configuration.

## Getting started

### Call a model

Every provider sits behind one small async interface. `make_client` builds the
right client from a `provider/model` id and reads the matching API key
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) from the
environment — see
[Provider clients](https://docs.kaval.ai/tutorials/llm_clients.html#provider-clients-openai-gemini-anthropic-and-ollama)
for the full list, Ollama included:

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

Pass a [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) `response_model` and you get a validated object back instead of
a string to parse — with any provider:

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

More on calling models: the
[LLM clients tutorial](https://docs.kaval.ai/tutorials/llm_clients.html) covers
[streaming responses](https://docs.kaval.ai/tutorials/llm_clients.html#streaming-responses),
[timeouts and retries](https://docs.kaval.ai/tutorials/llm_clients.html#timeouts-and-retries)
and [token usage statistics](https://docs.kaval.ai/tutorials/llm_clients.html#model-statistics-and-observability),
with a browser playground you can run without an API key. Tool calling and
multi-step loops live in [Agents & tools](https://docs.kaval.ai/tutorials/agents.html).

### Using retrieval-augmented generation (RAG)

RAG is a technique for injecting relevant information in the context of an LLM
so it could answer questions or perform tasks dependent on information external
from its training dataset.  A common way this is done involves embedding the information in a vector space and later
querying it using similarity search.

Kaval.AI provides a `RagService` that can index (embed) this information and also query it.
The [RAG tutorial](https://docs.kaval.ai/tutorials/rag.html) is a runnable
notebook that walks through the same loop in depth.

For instance, let's define some "facts" about a fictional Green Village
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

Then, we'll use a in-memory Sqlite database backend to index these facts.
In production, you can use PostgreSQL pgvector extension or other
supported vector search databases. The embedding model is named
`provider/model` too — `fastembed` runs locally with no API key, `openai`,
`gemini` and `ollama` are also supported; see
[Choosing an embedding model](https://docs.kaval.ai/tutorials/rag.html#choosing-an-embedding-model).

```python
from kavalai.rag import SqliteRagService

rag = SqliteRagService(":memory:", model="fastembed/BAAI/bge-small-en-v1.5")
await rag.index_batch(
    texts=FACTS,
    metadata_list=[{"village": "Green Village"}] * len(FACTS),
    source_ids=[f"fact-{i}" for i in range(len(FACTS))],
)
```

RagService can be used independently to query the closest matches for a query string.

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

In prompt engineering, you would then inject these results in the prompt to give LLM


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

Every hit is a `RagServiceResult` carrying `content`, `similarity`, `source_id`
and the `rag_metadata` you indexed it with, so retrieval can be filtered
(`collection_name=`, `source_ids=`) or inspected on its own — the model call is
optional.

From here the [RAG tutorial](https://docs.kaval.ai/tutorials/rag.html) covers
[collapsing chunks of one document with `keep_best`](https://docs.kaval.ai/tutorials/rag.html#chunked-documents-and-keep-best),
[batched queries](https://docs.kaval.ai/tutorials/rag.html#batched-queries) and
[similarity matrices](https://docs.kaval.ai/tutorials/rag.html#similarity-matrix)
against a PostgreSQL/`pgvector` index. A RAG index can also be pre-built and
shipped to the browser — see
[Running in the browser](https://docs.kaval.ai/tutorials/run_in_browser.html).


## The backoffice UI

Kaval.AI has a UI for debugging sessions and workflows.

* Session, run, node and model call stats.
* Projects/agents overview
* Detailed task debugger
* RAG explorer

<a href="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png">
  <img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png" alt="Kaval.AI backoffice project page with database details and activity charts" width="400px"/>
</a>

For details, see
[Using the backoffice UI](https://docs.kaval.ai/ui/index.html).

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
