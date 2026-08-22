"""``kavalai-eval`` and ``kavalai-persona``: the whole product surface.

The only module in :mod:`kavalai.eval` that reads environment variables. The
library itself never does — that rule is what lets a suite be run from a
notebook, a test or a CI job without a hidden dependency on the shell.

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
"""

import argparse
import asyncio
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from rich.console import Console

from kavalai.eval.models import ExperimentResult, Suite
from kavalai.eval.persona import Persona, PersonaRunner
from kavalai.eval.report import (
    comment_body,
    print_diff,
    print_report,
    write_junit,
)
from kavalai.eval.runner import BudgetExceeded, Experiment, load_setup
from kavalai.eval.targets import build_target

#: 0 the gate passed, 1 it failed, 2 the run itself could not complete. CI
#: needs the third: "the harness broke" and "the workflow is wrong" call for
#: different people.
EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Optional[str]) -> Optional[str]:
    """Expand ``${VAR}`` in a suite field. CLI-only, by design."""
    if value is None:
        return None

    def replace(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise SystemExit(
                f"The suite refers to ${{{name}}}, which is not set. "
                f"Set it, or load your .env first."
            )
        return resolved

    return _ENV_PATTERN.sub(replace, value)


#: The sink this CLI installed, so repeated calls replace it rather than
#: stacking, and so nothing else's sinks are disturbed.
_LOG_SINK: Optional[int] = None


def _quiet_logs(verbose: bool) -> None:
    """Keep the report readable: the run's own logs go to stderr at WARNING.

    Only loguru's default sink and our own are touched. A bare
    ``logger.remove()`` would take out whatever the host application — or the
    test runner — had installed, which is not a CLI's business.
    """
    global _LOG_SINK
    for handler_id in (0, _LOG_SINK):
        if handler_id is not None:
            try:
                logger.remove(handler_id)
            except ValueError:
                pass
    _LOG_SINK = logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")


def build_agent_service(uri: Optional[str], schema: Optional[str]) -> Any:
    """An ``AgentService`` over the configured database, or ``None``.

    This is the "run personas on the database" path: with it, each simulated
    conversation becomes an ordinary session the backoffice already renders,
    tagged so ``eval:`` separates it from production traffic.
    """
    uri = uri or os.environ.get("KAVALAI_DB_URI")
    if not uri:
        return None
    from kavalai.agent_service import AgentService
    from kavalai.db import DatabaseManager

    schema = schema or os.environ.get("KAVALAI_DB_SCHEMA")
    return AgentService(DatabaseManager().get_sessionmaker(uri=uri, schema=schema))


def _fixture_factory(suite: Suite, args: argparse.Namespace):
    """The recorded-response client factory, when the run asked for one."""
    if not (
        getattr(args, "fixtures", False) or getattr(args, "record_fixtures", False)
    ):
        return None
    from kavalai.eval.fixtures import fixture_client_factory

    path = suite.resolve(suite.target.fixtures)
    if args.record_fixtures:
        Console().print(f"  recording model responses to {path}")
    elif not path.exists():
        raise SystemExit(
            f"No fixtures at {path}. Record them first:\n"
            f"  kavalai-eval {args.suite} --record-fixtures"
        )
    return fixture_client_factory(path, record=args.record_fixtures)


def _target_overrides(suite: Suite, args: argparse.Namespace) -> dict:
    """Everything the library must not read for itself."""
    overrides: dict[str, Any] = {}
    factory = _fixture_factory(suite, args)
    if factory is not None:
        if suite.target.kind != "engine":
            raise SystemExit("Fixtures only apply to target kind 'engine'.")
        overrides["client_factory"] = factory
    if suite.target.kind == "engine" and args.persist_sessions:
        service = build_agent_service(args.db_uri, args.db_schema)
        if service is None:
            raise SystemExit(
                "--persist-sessions needs a database: set KAVALAI_DB_URI "
                "(your .env has one) or pass --db-uri."
            )
        from kavalai.workflow.tasklog import PostgresTaskLogger

        overrides["agent_service"] = service
        # Sessions and chat come from the agent service; the task rows behind
        # them come from this. Both, or the backoffice shows a conversation
        # with no trace under it.
        overrides["task_logger"] = PostgresTaskLogger(service)
    if suite.target.kind == "rest":
        suite.target.base_url = expand_env(args.base_url or suite.target.base_url)
        user = os.environ.get("KAVALAI_AGENT_BASIC_AUTH_USER")
        password = os.environ.get("KAVALAI_AGENT_BASIC_AUTH_PASSWORD")
        if user and password:
            overrides["auth"] = (user, password)
    return overrides


def _load_suite(path: str, args: argparse.Namespace) -> Suite:
    suite = Suite.from_yaml(path)
    if args.target:
        suite.target.kind = args.target
    if getattr(args, "repeats", None):
        suite.repeats = args.repeats
    if getattr(args, "concurrency", None):
        suite.concurrency = args.concurrency
    return suite


# ------------------------------------------------------------------ commands
async def _run(args: argparse.Namespace) -> int:
    console = Console()
    suite = _load_suite(args.suite, args)
    overrides = _target_overrides(suite, args)
    experiment = Experiment(
        suite,
        tag=args.tag,
        target_overrides=overrides,
        persist_sessions=args.persist_sessions,
        include_personas=args.personas,
        only_personas=args.only_personas,
        # Replaying recorded responses and then calling a judge over the
        # network would defeat the point of a keyless tier, so fixtures imply
        # the deterministic set unless the run explicitly asks otherwise.
        skip_model_evaluators=args.no_judges or (args.fixtures and not args.judges),
        skip_trajectory_evaluators=args.skip_trajectory_evaluators,
    )
    try:
        result = await experiment.run()
    except BudgetExceeded as exc:
        console.print(f"[red]budget exceeded[/red] {exc}")
        return EXIT_ERROR

    factory = overrides.get("client_factory")
    if args.record_fixtures and factory is not None:
        factory.store.save()
        console.print(f"  recorded {len(factory.store)} model responses")

    print_report(result, console, verbose=args.verbose)

    json_path = suite.result_path(args.tag)
    result.to_json(json_path)
    junit_path = write_junit(result, suite.result_path(args.tag, ".junit.xml"))
    console.print(f"\n  wrote {json_path}\n  wrote {junit_path}")

    if args.comment:
        Path(args.comment).write_text(
            comment_body(suite.load_baseline(), result), encoding="utf-8"
        )
        console.print(f"  wrote {args.comment}")
    return EXIT_OK if result.gate.passed else EXIT_GATE_FAILED


async def _persona(args: argparse.Namespace) -> int:
    """Run one persona against one target and print the conversation live."""
    console = Console()
    persona = Persona.from_yaml(args.persona)

    if args.suite:
        suite = _load_suite(args.suite, args)
        if suite.setup:
            load_setup(suite.resolve(suite.setup))
        target_spec, base_dir = suite.target, suite.base_dir
        overrides = _target_overrides(suite, args)
    else:
        from kavalai.eval.models import TargetSpec

        if not args.workflow:
            raise SystemExit("Pass --workflow <file.yaml> or --suite <suite.yaml>.")
        if args.setup:
            load_setup(Path(args.setup))
        target_spec = TargetSpec(kind="engine", workflow=str(Path(args.workflow).name))
        base_dir = Path(args.workflow).parent
        overrides = {}
        if args.persist_sessions:
            service = build_agent_service(args.db_uri, args.db_schema)
            if service is None:
                raise SystemExit("--persist-sessions needs KAVALAI_DB_URI or --db-uri.")
            from kavalai.workflow.tasklog import PostgresTaskLogger

            overrides["agent_service"] = service
            overrides["task_logger"] = PostgresTaskLogger(service)

    target = build_target(target_spec, base_dir, **overrides)
    await target.setup()

    def show(message: str, reply: str) -> None:
        console.print(f"[bold cyan]{persona.name}[/bold cyan]: {message}")
        console.print(f"[bold green]assistant[/bold green]: {reply}\n")

    external_id = (
        f"eval:persona:{args.tag}:{persona.name}:0" if args.persist_sessions else None
    )
    try:
        console.print(f"[bold]{persona.name}[/bold] — {persona.goal}\n")
        runner = PersonaRunner(persona, target, max_turns=args.turns, on_turn=show)
        conversation = await runner.run(external_id=external_id)
    finally:
        await target.aclose()

    console.print(
        f"  {len([t for t in conversation.turns if t['role'] == 'user'])} turns · "
        f"goal achieved: {conversation.goal_achieved} · "
        f"{conversation.elapsed:.1f}s"
    )
    if external_id:
        console.print(f"  session: {external_id}")
    if args.transcript:
        Path(args.transcript).write_text(conversation.transcript(), encoding="utf-8")
        console.print(f"  wrote {args.transcript}")
    return EXIT_OK


def _diff(args: argparse.Namespace) -> int:
    baseline = ExperimentResult.from_json(args.baseline)
    current = ExperimentResult.from_json(args.result)
    print_diff(baseline, current)
    return EXIT_OK


def _accept(args: argparse.Namespace) -> int:
    """Promote a result file to the committed baseline.

    Deliberately a separate, explicit step that writes a file you then commit —
    never something a passing run does for you, or the gate quietly erases
    itself.
    """
    result = ExperimentResult.from_json(args.result)
    suite = Suite.from_yaml(args.suite) if args.suite else None
    destination = (
        Path(args.output)
        if args.output
        else (suite.baseline_path() if suite else Path("baseline.json"))
    )
    shutil.copyfile(args.result, destination)
    console = Console()
    console.print(f"Baseline updated from {args.result} -> {destination}")
    console.print(
        f"  {result.totals.passed}/{result.totals.cases} passing "
        f"({result.totals.pass_rate:.0%})"
    )
    console.print(
        "  Commit it with a message saying what changed and why: accepting a "
        "baseline is accepting new behaviour."
    )
    return EXIT_OK


def _list_evaluators(_args: argparse.Namespace) -> int:
    from kavalai.eval import evaluators as _  # noqa: F401  (registers built-ins)
    from kavalai.eval.evaluators.base import REGISTRY

    console = Console()
    for name in sorted(REGISTRY):
        doc = (REGISTRY[name].__doc__ or "").strip().splitlines()
        console.print(f"  [bold]{name:<22}[/bold] {doc[0] if doc else ''}")
    return EXIT_OK


# --------------------------------------------------------------------- parser
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default="local", help="Names this run's result file.")
    parser.add_argument(
        "--persist-sessions",
        action="store_true",
        help="Write each run to the agent database under an eval: external id, "
        "so the backoffice can show the conversation that failed.",
    )
    parser.add_argument("--db-uri", help="Overrides KAVALAI_DB_URI.")
    parser.add_argument("--db-schema", help="Overrides KAVALAI_DB_SCHEMA.")
    parser.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kavalai-eval",
        description="Run an acceptance suite against a Kaval.AI workflow.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a suite (the default command).")
    run.add_argument("suite", help="Path to suite.yaml")
    run.add_argument("--target", choices=["engine", "rest", "callable"])
    run.add_argument("--base-url", help="For --target rest; ${VAR} is expanded.")
    run.add_argument("--repeats", type=int)
    run.add_argument("--concurrency", type=int)
    run.add_argument(
        "--personas", action="store_true", help="Also run the suite's personas."
    )
    run.add_argument(
        "--only-personas", action="store_true", help="Run only the personas."
    )
    run.add_argument("--comment", help="Write a plain-words summary to this file.")
    run.add_argument(
        "--fixtures",
        action="store_true",
        help="Replay recorded model responses instead of calling a provider. "
        "No API key needed; implies --no-judges.",
    )
    run.add_argument(
        "--judges",
        action="store_true",
        help="Run model-backed evaluators even with --fixtures (needs a key).",
    )
    run.add_argument(
        "--record-fixtures",
        action="store_true",
        help="Call the real provider and record what it says, for --fixtures.",
    )
    run.add_argument(
        "--no-judges",
        action="store_true",
        help="Skip evaluators that call a model. The report says which.",
    )
    run.add_argument(
        "--skip-trajectory-evaluators",
        action="store_true",
        help="For a target that cannot observe a trajectory (rest, callable): "
        "run the output-only checks and report which assertions were dropped. "
        "Without it, such a run refuses to start.",
    )
    _add_common(run)
    run.set_defaults(handler=_run, is_async=True)

    persona = sub.add_parser("persona", help="Run one persona and watch it talk.")
    persona.add_argument("persona", help="Path to a persona YAML file")
    persona.add_argument("--suite", help="Take the target and setup from this suite.")
    persona.add_argument("--workflow", help="Or point straight at a workflow YAML.")
    persona.add_argument("--setup", help="Setup module to import first.")
    persona.add_argument("--turns", type=int, help="Override the persona's max_turns.")
    persona.add_argument("--transcript", help="Write the conversation to this file.")
    persona.add_argument("--target", choices=["engine", "rest"])
    persona.add_argument("--base-url")
    _add_common(persona)
    persona.set_defaults(handler=_persona, is_async=True)

    diff = sub.add_parser("diff", help="Compare two result files.")
    diff.add_argument("baseline")
    diff.add_argument("result")
    diff.set_defaults(handler=_diff, is_async=False, verbose=False)

    accept = sub.add_parser("accept", help="Promote a result to the baseline.")
    accept.add_argument("result")
    accept.add_argument("--suite", help="Write to this suite's baseline path.")
    accept.add_argument("-o", "--output", help="Write the baseline here instead.")
    accept.set_defaults(handler=_accept, is_async=False, verbose=False)

    evaluators = sub.add_parser("evaluators", help="List the evaluators available.")
    evaluators.set_defaults(handler=_list_evaluators, is_async=False, verbose=False)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``kavalai-eval path/to/suite.yaml`` is the command people actually type,
    # so make ``run`` the default rather than making them say it.
    known = {"run", "persona", "diff", "accept", "evaluators", "-h", "--help"}
    if argv and argv[0] not in known:
        argv.insert(0, "run")

    args = build_parser().parse_args(argv)
    if not getattr(args, "handler", None):
        build_parser().print_help()
        return EXIT_ERROR
    _quiet_logs(args.verbose)
    try:
        if args.is_async:
            return asyncio.run(args.handler(args))
        return args.handler(args)
    except SystemExit:
        raise
    except Exception as exc:
        Console().print(f"[red]error[/red] {exc}")
        if args.verbose:
            raise
        return EXIT_ERROR


def persona_main(argv: Optional[list[str]] = None) -> int:
    """``kavalai-persona <file>`` — the persona command as its own entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return main(["persona"] + argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
