---
name: kavalai
description: Kaval.AI agent framework — install and extras, calling a model, structured output, the Agent, and which Kaval.AI skill to load next. Use when working in a project that depends on `kavalai`, when the user mentions Kaval.AI, or when you see `WorkflowEngine`, `@pythontool`, `FunctionKernel`, or a `provider/model` id like `openai/gpt-5.4-mini`.
---

# Kaval.AI

Official skill for Kaval.AI, a YAML-based framework for predictable, observable
AI pipelines. Start here, then load the skill for the job in hand.

## Which skill to load

| You are… | Load |
|---|---|
| writing or debugging a workflow graph (YAML or `WorkflowBuilder`) | `kavalai-workflows` |
| giving an agent tools — Python, REST, MCP | `kavalai-tools` |
| serving, deploying, persisting or monitoring a workflow | `kavalai-serving` |
| indexing documents and retrieving them | `kavalai-rag` |
| writing eval cases, or testing a workflow without a model | `kavalai-eval` |

## Install

```bash
pip install "kavalai[common]"
```

Python 3.12+. The **base** install is deliberately small and Pyodide-compatible,
so it contains no provider SDKs. `common` is the normal install for anything
that is not running in a browser.

| Extra | What it adds |
|---|---|
| `common` | Provider SDKs, RAG/embeddings, Postgres drivers, MCP, the REST/SSE servers, the bundled web tools |
| `common_web` | What the core needs additionally under Pyodide/WebLLM (`pyodide-http`) |
| `gpu` | `fastembed-gpu` for local embedding on an NVIDIA GPU |
| `test` / `docs` | Test tooling; Sphinx and the notebook kernel |

`gpu` is **not** additive to `common`: `fastembed-gpu` is the same import name
as `fastembed` built against `onnxruntime-gpu`, so the two cannot coexist. You
replace the CPU package rather than adding to it:

```bash
uv pip uninstall fastembed
uv pip install "kavalai[common,gpu]"
```

No code change follows — FastEmbed defaults to `cuda=Device.AUTO` and picks the
CUDA execution provider when there is one.

When an import fails with *"requires the optional 'openai' package. Install it
with: pip install kavalai[common]"*, the message is right: install the extra.
Provider clients are resolved lazily through `__getattr__` precisely so that
`import kavalai` works where no SDK is installed. Do not "fix" it by importing
the SDK directly.

## Call a model

`make_client` builds a client from a `provider/model` id and finds the matching
key in the environment.

```python
from kavalai import make_client

client = make_client("openai/gpt-5.4-mini")
answer = await client.prompt("What is the capital of Estonia?")
```

**`provider/model` is the id format everywhere in the framework** — in
`make_client`, in a workflow's `llm_model`, in a node override, in
`KAVALAI_DEFAULT_LLM_MODEL`, in an eval suite's `judge_model`. A bare model
name raises `Model must be in 'provider/model' form`.

Providers: `openai/…`, `gemini/…`, `anthropic/…`, `ollama/…`, `browser/…`.
Switching provider is a change of one string; nothing else moves. Keys come
from `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (note the `_API_`
— `ANTHROPIC_KEY` is not read) and `OLLAMA_HOST`. Each client also accepts
`api_key=` / `host=` directly, which wins over the environment.

The clients are async: `asyncio.run(main())` in a script, bare `await` in a
notebook.

## Ask for a type, not for prose

The single highest-value idiom in the library. Pass a Pydantic model and you
get a validated object, with any provider:

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    population: int

city = await client.prompt("Describe Tallinn.", response_model=City)
city.population   # an int, not a string to parse
```

Do not write output parsing, regex extraction or "respond only with JSON"
instructions around a Kaval.AI client. Declare the shape instead. The same
applies inside a workflow, where `data_types` does the job.

## The Agent

`Agent` is the multi-step, tool-calling loop. Use it when the model should
decide *which* tool to call; call the tool yourself when you already know.

```python
from kavalai import Agent, FunctionKernel, make_client

agent = Agent(
    llm_client=make_client("openai/gpt-5.4-mini"),
    kernel=FunctionKernel(),      # optional
    allowed_tools=None,           # None = every registered tool
    debug=False,
)
result = await agent.prompt("Summarise the latest filings",
                            response_model=MySchema, max_steps=10)
```

`max_steps` (default 10) is the bound that stops a runaway loop; there is no
implicit capability beyond the tools registered on the kernel. See
`kavalai-tools`.

## Import surface

Everything supported is importable from the top level — `from kavalai import X`
— and `kavalai/__init__.py`'s `__all__` is the list. The ORM row classes
(`Agent` the table, `Run`, `Task`, …) deliberately live in `kavalai.db` so the
top-level `Agent` unambiguously means the agent.

## House rules for code you write with this framework

- Use `loguru` for logging and f-strings for formatting, matching the library.
- Library code never reads environment variables; only entry points do. Pass
  configuration explicitly and your code stays testable.
- Type every boundary. The framework validates what it is given — give it
  something to validate.
