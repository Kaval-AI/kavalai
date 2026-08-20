<img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an opinionated and elegant Python library for building production-grade agentic workflows, chatbots and tools.


Features:
- Support commercial (OpenAI, Google, Anthropic) and open source LLM providers.
- Runs in browser via WebLLM and Pyodide.
- Excellent support for retrieval-augmented generation (RAG) on various storage engines.
- Structured inputs, outputs, tool calls and responses using Pydantic semantics.
- Streaming responses.
- Fully featured workflow engine using conditional processing and tool calling.
- Python tools, REST server endpoints (supports basic auth) and MCP support.

See the [full documentation](https://docs.kaval.ai) for more detailed reference.

## Install

```
pip install kavalai[common]
```

## Getting started

### Call a model

Every provider sits behind one small async interface. `make_client` builds the
right client from a `provider/model` id and reads the matching API key
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) from the
environment:

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

### Using retrieval-augmented generation (RAG)

RAG is a technique for injecting relevant information in the context of an LLM
so it could answer questions or perform tasks dependent on information external
from its training dataset.  A common way this is done involves embedding the information in a vector space and later
querying it using similarity search.

Kaval.AI provides a `RagService` that can index (embed) this information and also query it.

For instance, let's define some "facts" about a fictional Green Village
```python
FACTS = """\
President of Green Village is Thomas Cook (born 12.04.1994).
Green Village has 104 residents.
Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.
The tallest building in Green Village is the Old Grain Tower at 23 meters.
Green Village's official flower is the marsh marigold.
The village bakery, run by Greta Lindqvist (born 27.11.1968), sells exactly 340 loaves every week.
Green Village has one school with 14 pupils and 2 teachers.
The annual Turnip Festival takes place every year on the third Saturday of October.
Green Village's fire brigade consists of 7 volunteers and one dalmatian named Pepper.
The village pond, Lake Miller, is 1.2 meters deep at its deepest point.
Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).
The village has 3 streets: Main Road, Willow Lane, and Cobbler's Path.
Green Village's football team, FC Green Rovers, has won the regional cup twice (1997 and 2013).
The local church bell weighs 412 kilograms and was cast in 1901.
Green Village produces 8 tons of honey per year from its 26 beehives.
The village library owns 1,847 books and is open on Tuesdays and Fridays.
Green Village's mayor before Thomas Cook was Henrietta Voss (served 2009-2021).
The speed limit everywhere in Green Village is 30 km/h.
Green Village's only pub, The Rusty Anchor, has been operating since 1923.
Every resident of Green Village receives a free pumpkin on their birthday, a tradition started in 1954.
""".splitlines()
```

Then, we'll use a in-memory Sqlite database backend to index these facts.
In production, you can use PostgreSQL pgvector extension or other
supported vector search databases.

```python
from kavalai.rag import SqliteRagService

rag = SqliteRagService(":memory", model="fastembed/BAAI/bge-small-en-v1.5")
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


```Python
# 2. Or hand those passages to a model and let it answer.
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
