"""Run a YAML file of cases against a running agent server.

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

The file looks like this::

    name: green-village
    base_url: http://localhost:25000   # optional; --host/--port override it
    judge_model: openai/gpt-5.4-mini   # optional
    cases:
      - name: president
        input: {user_message: Who is the president of Green Village?}
        expected:
          agent_response: {contains: Thomas Cook}
      - name: no_budget
        type: judge
        input: {user_message: What is the village's annual budget?}
        expected: The answer says it does not know instead of inventing one.

Run it against a server that is already up::

    uv run --env-file .env kavalai-eval cases.yaml --host localhost --port 25000
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from kavalai.eval.base import DEFAULT_BASE_URL, EvalResult
from kavalai.eval.judge_evaluator import DEFAULT_JUDGE_MODEL, JudgeEvaluator
from kavalai.eval.simple_evaluator import SimpleEvaluator

#: Exit code for a run whose cases all passed.
EXIT_PASSED = 0
#: Exit code for a run with at least one failing case.
EXIT_FAILED = 1
#: Exit code for a run that never got as far as a verdict.
EXIT_ERROR = 2


class EvalCase(BaseModel):
    """One input, and what its answer has to look like.

    Attributes:
        name: How the case is reported.
        type: ``simple`` compares the answer with expected values;
            ``judge`` asks a model whether the answer is acceptable.
        input: Field values for the agent's input type.
        expected: A mapping of output field to expected value or matcher for
            a simple case; a plain-language criterion for a judged one.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["simple", "judge"] = "simple"
    input: dict[str, Any]
    expected: Optional[Union[dict[str, Any], str]] = None

    @model_validator(mode="after")
    def _expectation_fits_the_type(self) -> "EvalCase":
        """Reject an expectation the chosen evaluator cannot read.

        A judged case without a criterion, or one whose criterion was written
        as a mapping, would otherwise pass on any answer whatsoever.
        """
        if self.type == "judge" and not isinstance(self.expected, str):
            raise ValueError(
                f"Case '{self.name}' is judged, so `expected` must be a "
                "plain-language criterion."
            )
        if self.type == "simple" and isinstance(self.expected, str):
            raise ValueError(
                f"Case '{self.name}' is simple, so `expected` must map output "
                "fields to expected values. Use `type: judge` to grade a "
                "plain-language criterion."
            )
        return self


class EvalSuite(BaseModel):
    """A named list of cases, plus the defaults they run under."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: Optional[str] = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    cases: list[EvalCase]


def load_suite(path: Union[str, Path]) -> EvalSuite:
    """Read and validate a suite file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed :class:`EvalSuite`.

    Raises:
        pydantic.ValidationError: The file does not describe a suite.
    """
    with open(path, "r", encoding="utf-8") as f:
        return EvalSuite.model_validate(yaml.safe_load(f))


async def run_suite(
    suite: EvalSuite,
    base_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: float = 120.0,
    judge_model: Optional[str] = None,
    transport: Optional[Any] = None,
    on_result: Optional[Callable[[EvalResult], None]] = None,
) -> list[EvalResult]:
    """Run every case in ``suite``, in order, against a running agent.

    The cases run one at a time: an evaluation that is easy to read while it
    runs is worth more than one that finishes a few seconds sooner.

    Args:
        suite: The cases to run.
        base_url: Where the agent server listens; falls back to the suite's
            own ``base_url``, then to ``http://localhost:10000``.
        username: HTTP Basic Auth user, if the server requires one.
        password: HTTP Basic Auth password.
        timeout: Seconds to wait for one agent run.
        judge_model: ``provider/model`` of the judge, overriding the suite's.
        transport: Optional httpx transport, used by tests.
        on_result: Called with each result as it arrives, for progress
            output.

    Returns:
        One :class:`~kavalai.eval.base.EvalResult` per case, in file order.
    """
    target = base_url or suite.base_url or DEFAULT_BASE_URL
    connection = dict(
        base_url=target,
        username=username,
        password=password,
        timeout=timeout,
        transport=transport,
    )
    evaluators = {
        "simple": SimpleEvaluator(**connection),
        "judge": JudgeEvaluator(**connection, model=judge_model or suite.judge_model),
    }

    results = []
    for case in suite.cases:
        result = await evaluators[case.type].evaluate(
            case.input, case.expected, name=case.name
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def format_result(result: EvalResult) -> str:
    """One result as a single report line."""
    status = "PASS" if result.passed else "FAIL"
    return f"{status}  {result.name}" + (
        f"  — {result.reason}" if result.reason else ""
    )


def format_summary(results: list[EvalResult]) -> str:
    """The closing line of a run."""
    passed = sum(1 for r in results if r.passed)
    return f"{passed}/{len(results)} passed"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="kavalai-eval",
        description="Run a YAML file of cases against a running agent server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("suite", help="Path to the YAML file of cases.")
    parser.add_argument(
        "--host", help="Agent server host (default: taken from the suite file)."
    )
    parser.add_argument(
        "--port", type=int, help="Agent server port (default: taken from the suite)."
    )
    parser.add_argument(
        "--auth",
        metavar="USER:PASSWORD",
        help="HTTP basic auth, if the server has it configured.",
    )
    parser.add_argument(
        "--judge-model",
        help="Model grading the judged cases, overriding the suite's setting.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for one agent run (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_base_url(host: Optional[str], port: Optional[int]) -> Optional[str]:
    """Build a base URL from the host and port flags.

    Returns ``None`` when neither was given, so the suite's own ``base_url``
    still applies.
    """
    if host is None and port is None:
        return None
    return f"http://{host or 'localhost'}:{port or 10000}"


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point of the ``kavalai-eval`` console script.

    Returns:
        ``0`` when every case passed, ``1`` when a case failed, and ``2``
        when the run never reached a verdict — a CI job needs the third to
        tell "the suite is broken" from "the agent is wrong".
    """
    args = parse_args(argv)

    try:
        suite = load_suite(args.suite)
    except Exception as e:
        print(f"Cannot run {args.suite}: {e}", file=sys.stderr)
        return EXIT_ERROR

    username, _, password = (args.auth or "").partition(":")
    base_url = resolve_base_url(args.host, args.port)

    print(
        f"{suite.name}: {len(suite.cases)} cases against "
        f"{base_url or suite.base_url or DEFAULT_BASE_URL}\n"
    )
    try:
        results = asyncio.run(
            run_suite(
                suite,
                base_url=base_url,
                username=username or None,
                password=password or None,
                timeout=args.timeout,
                judge_model=args.judge_model,
                on_result=lambda result: print(format_result(result)),
            )
        )
    except Exception as e:
        print(f"The run broke: {e}", file=sys.stderr)
        return EXIT_ERROR

    print(f"\n{format_summary(results)}")
    return EXIT_PASSED if all(r.passed for r in results) else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
