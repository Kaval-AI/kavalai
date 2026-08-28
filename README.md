<img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/frontend/public/assets/images/iconlogo.svg" alt="Kaval.AI Logo" width="400" height="100"/>

[![CI](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaval-AI/kaval.ai/actions/workflows/ci.yml)

Kaval.AI is an opinionated Python library for building well-defined, testable and robust
agentic workflows, chatbots and tools.

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

Check out [docs.kaval.ai](https://docs.kaval.ai/) for examples and in-depth tutorials.

## Install

```
pip install "kavalai[common]"
```

`common` brings the provider SDKs, RAG, MCP and the servers.
See
[Installation](https://docs.kaval.ai/tutorials/installation.html) for more details.

## Getting started

### Call a model

Name a model as `provider/model` and `make_client` does the rest.
```python
from kavalai import make_client

client = make_client("openai/gpt-5.6-luna")
answer = await client.prompt("What is the capital of Estonia?")
print (answer)
```

Response:
```
The capital of Estonia is **Tallinn**.
```

Use structured responses by passing a [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/)
model:

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
name='Tallinn' country='Estonia'
fun_fact='Tallinn’s remarkably well-preserved medieval Old Town is a UNESCO
          World Heritage Site, and the city is widely regarded as one of the
          world’s most digitally advanced capitals.'
```

### Tools and agents

Kaval.AI has built-in agent loop that supports tool calling:

```python
from datetime import date

from pydantic import BaseModel

from kavalai import Agent, FunctionKernel, make_client, pythontool
from kavalai.tools.webtools.crawl4ai import web_search


@pythontool
def today() -> str:
    """Return today's date in ISO format."""
    return date.today().isoformat()


class Answer(BaseModel):
    answer: str
    sources: list[str]


kernel = FunctionKernel()
kernel.register_python_tool("today", today)
kernel.register_python_tool("web_search", web_search)

agent = Agent(llm_client=make_client("openai/gpt-5.6-luna"), kernel=kernel)
result = await agent.prompt(
    "When is the next Tallinn Marathon, and how many days away is it?",
    response_model=Answer,
    max_steps=5,
)
print(result.answer)
for url in result.sources:
    print(url)
```

Response:
```
The next Tallinn Marathon is scheduled for **Sunday, September 13, 2026**.
From today, **August 28, 2026**, it is **16 days away**.

Sources:
- https://marathonscout.com/races/swedbank-tallinn-marathon
- https://www.jooks.ee/en/tallinn-marathon/
https://marathonscout.com/races/swedbank-tallinn-marathon
https://www.jooks.ee/en/tallinn-marathon/
```

See using [Agents & tools](https://docs.kaval.ai/tutorials/agents.html) for more.

### Using retrieval-augmented generation (RAG)

Retrieval-agumented generation allows the model to operate with data
it was not trained with:

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

Index the data the way you want

```python
from kavalai.rag import SqliteRagService

rag = SqliteRagService(":memory:", model="fastembed/BAAI/bge-small-en-v1.5")
await rag.index_batch(
    texts=FACTS,
    metadata_list=[{"village": "Green Village"}] * len(FACTS),
    source_ids=[f"fact-{i}" for i in range(len(FACTS))],
)
```

Query the dataset

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

Check out the [RAG tutorial](https://docs.kaval.ai/tutorials/rag.html).

### Building a workflow

Turn the Green Village index above into a chatbot: a `rag_query` node
fetches the closest facts for the user's message, and an `llm` node answers
from them.

```python
from pydantic import BaseModel

from kavalai.workflow import WorkflowBuilder


class Message(BaseModel):
    user_message: str


class Reply(BaseModel):
    agent_response: str


engine = (
    WorkflowBuilder("Green Village support", llm_model="openai/gpt-5.6-luna")
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

state = await engine.run({"user_message": question})
print(state.output_data)
print(state.status, state.token_usage)
```

Response:
```
{'agent_response': 'Agnes Whitlow was 96 years old at the 2025 Turnip '
                   'Festival, held on 18 October 2025.'}
completed {'model_calls': 1, 'prompt_tokens': 239,
           'completion_tokens': 109, 'total_tokens': 348}
```

See [Workflows tutorial](https://docs.kaval.ai/tutorials/workflow.html) for more info.


## The backoffice UI

Use the [Backoffice UI](https://docs.kaval.ai/ui/index.html). to observe and debug chat sessions, workflow runs and inspect RAG.

<a href="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png">
  <img src="https://raw.githubusercontent.com/Kaval-AI/kaval.ai/main/docs/ui/projectinfopage.png" alt="Kaval.AI backoffice project page with database details and activity charts" width="400px"/>
</a>


## License

[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)
