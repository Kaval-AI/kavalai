---
name: kavalai-eval
description: Write and run Kaval.AI evaluation cases — the `eval_cases.yaml` format, matchers, the `kavalai-eval` command, and the `run_suite` / `check_output` Python API — and test workflows deterministically without a model. Use when writing eval cases, grading an agent's answers, adding agent checks to CI, or writing pytest tests for a workflow.
---

# Evaluating a Kaval.AI agent

`kavalai.eval` grades an agent server that is **already running**. It speaks
HTTP and discovers the agent's input and output types from the server's OpenAPI
spec, so it knows nothing about the engine, the workflow file or the database.
Start the server first; the evaluator is a client.

## The case file

One file, one suite, validated in full before a single case runs.

```yaml
name: green-village
judge_model: openai/gpt-5.6-luna      # optional

cases:
  - name: president
    input:
      user_message: Who is the president of Green Village?
    expected:
      agent_response: {contains: [Thomas Cook]}

  - name: no_budget
    type: judge
    input:
      user_message: What is the village's annual budget?
    expected: >-
      The answer says the information is not available instead of
      inventing a figure.
```

| Key | Meaning |
|---|---|
| `name` | Names the suite in the run header. Required |
| `judge_model` | `provider/model` grading judged cases. Defaults to `DEFAULT_JUDGE_MODEL`; `--judge-model` overrides for one run |
| `cases` | Run in the order written |

Per case: `name` (required, and what its session is recorded under), `type`
(`simple` default, or `judge`), `input` (validated against the agent's input
type before the call), `expected`.

**There is deliberately no `base_url` key.** Which agent a suite grades is a
property of the run, not of the cases — the same file points at a laptop, at
staging, or at two model versions in turn. Do not add one.

**Unknown keys are refused**, in the suite and in every case. A silently
ignored key is a case that never ran — so a mistyped key fails loudly rather
than weakening the suite.

Two combinations are refused outright, because both would pass on any answer:

```text
Case 'x' is judged, so `expected` must be a plain-language criterion.
Case 'y' is simple, so `expected` must map output fields to expected values.
```

## Matchers

A simple case maps an output field to a literal (shorthand for `equals`) or to
a mapping of matchers:

```yaml
expected:
  status: needs_details                 # equals
  order_id: {equals: ''}
  missing: {equals: ['items[0].quantity']}
  agent_response:
    contains: ["1.2"]
    not_contains: ["approximately"]
```

| Matcher | Passes when |
|---|---|
| `equals` | The field equals it exactly |
| `contains` | Text: every argument appears as a substring, **case-insensitively**. List/tuple/set: every argument is a member. Mapping: every argument is a key. Any other type never contains anything |
| `not_contains` | No argument is contained, by the same rule |
| `regex` | `re.search` finds the pattern in the field rendered as text |
| `one_of` | The field is one of the listed values |

Every matcher on a field is checked and each failure is reported separately.
Fields the expectation does not mention are ignored, so a case states what it
cares about and nothing more. **Naming a field the output does not have is a
failure, not a skip.**

A mapping is read as matchers only when *every* key is a matcher name, so an
agent that genuinely answers with a dictionary still compares cleanly:

```yaml
expected:
  totals: {cases: 26, passed: 26}       # a literal dict, not matchers
```

An empty or absent `expected` asserts only that the agent answered.

## Which cases to make literal, and which to judge

This is the decision that determines whether a suite is worth running.

- **Literal** whatever your Python decided. If the workflow stamps `status`,
  `order_id` or `missing` onto the answer, those are values to compare, and the
  comparison gives the same verdict every time with no model and no API key.
- **Judge** only what a literal comparison cannot settle: whether a reply
  promised a price, whether it took an instruction from an email, whether it
  refused to invent a figure it does not have.

A suite that judges everything is slow, costly and non-deterministic for no
gain. Shape the workflow so more of the answer is decidable — then assert on it.

A judged case's criterion must be a plain-language string, and one with no
criterion is refused: judging against nothing passes on any answer at all.

## Running it

```bash
kavalai-eval examples/bakery/eval_cases.yaml --port 25100 --tag baseline
```

| Flag | Meaning |
|---|---|
| `suite` | Path to the case file. Positional, required |
| `--port` | **Required** — which agent is being evaluated is never left to a default |
| `--host` | Default `localhost` |
| `--tag` | Names this run inside each case's `external_id` — a model version, a prompt variant, a build. Without it, two runs' sessions cannot be told apart afterwards |
| `--auth USER:PASSWORD` | When the server has basic auth configured |
| `--judge-model` | Overrides the suite's `judge_model` |
| `--timeout` | Seconds to wait for one agent run. Default `120` |

| Exit | Constant | Meaning |
|---|---|---|
| `0` | `EXIT_PASSED` | Every case passed |
| `1` | `EXIT_FAILED` | At least one case failed |
| `2` | `EXIT_ERROR` | The run never reached a verdict — the file would not load, or the run broke |

**CI must distinguish 1 from 2**, or "the suite is broken" reads as "the agent
is wrong". Do not write `kavalai-eval … || echo failed`. The constants live in
`kavalai.eval.eval_runner`, so a test can name the meaning rather than the
number.

A failing *agent call* is not an `EXIT_ERROR`: it fails its own case with the
error as the reason and the run continues, so one unreachable case cannot end a
suite.

Judged cases build a provider client, so they need a key —
`dotenv run kavalai-eval …`. A run of purely literal cases needs nothing at
all; nothing is built until a case is actually judged.

## From Python

The console script is a thin wrapper over four public pieces.

```python
from kavalai.eval import load_suite, run_suite

suite = load_suite("eval_cases.yaml")
results = await run_suite(
    suite, base_url="http://localhost:25100", tag="ci",
    on_result=lambda r: print(r.name, "ok" if r else r.reason),
)
```

`run_suite` takes `base_url`, `username`, `password`, `timeout`, `judge_model`,
`tag`, `transport` and `on_result`, and returns one `EvalResult` per case in
file order.

```python
from kavalai.eval import SimpleEvaluator, JudgeEvaluator

simple = SimpleEvaluator("http://localhost:25000", tag="ci")
judge = JudgeEvaluator("http://localhost:25000", tag="ci", model="openai/gpt-5.6-luna")

result = await simple.evaluate(inputs, expected, name="president")
assert result.passed, result.reason
```

Both take `base_url` (no default), `username`, `password`, `timeout`, `tag` and
`transport`; `JudgeEvaluator` adds `model`, `llm_client` and `prompt` (which
must accept `{inputs}`, `{output}` and `{criterion}`). **`transport` is an
`httpx` transport — that is what lets a test serve the requests with no network
at all.**

`evaluate` raises only for a judged case with no criterion; everything else — a
refused connection, a rejected input, a judge that fell over — comes back as a
failed result with the reason attached.

`EvalResult` carries `name`, `passed`, `reason`, `inputs`, `output`.
`bool(result)` is `passed`, so `if not result:` reads correctly.

The matcher engine is exported on its own, for asserting on a payload you
already have — no server involved:

```python
from kavalai.eval import check_output

failures = check_output({"status": "needs_details", "order_id": ""},
                        {"status": "needs_details", "order_id": {"equals": ""}})
assert not failures, failures
```

## Sessions

Each case runs in a fresh session, recorded under:

```text
eval:{tag}:{case}      # with --tag
eval:{case}            # without
```

Sessions are written only when the agent server has an `AgentService`; the
evaluators behave identically when it does not. `eval:` is the reserved prefix
the backoffice Conversations page filters on, and `--tag` is what makes two
runs comparable afterwards — always pass one in CI.

`kavalai.eval` reads **no environment variables** of its own. Base URL, auth,
judging model and timeout are all arguments, so a suite runs from a notebook or
a test with no hidden dependency on the shell. What does read the environment
is the provider client a judged case builds.

## Testing a workflow without a model

Evaluation grades a running agent. To test the *graph* — deterministically, in
CI, with no API key — inject a `client_factory` into the engine and return a
fake client. Then assert on `state.trace` (the exact path taken) as well as
`state.output_data`: a workflow that reaches the right answer down the wrong
branch is a workflow that will break next week.

Keep an example's tests beside the example rather than in the library's own
test directory, and let them stand on their own.
