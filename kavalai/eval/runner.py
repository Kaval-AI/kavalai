"""Runs a suite: cases, repeats, concurrency, aggregation and the gate.

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

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from kavalai.eval.evaluators.base import (
    EvaluationError,
    Evaluator,
    build_evaluator,
)
from kavalai.eval.evaluators.judged import rubric_sha
from kavalai.eval.models import (
    Case,
    CaseResult,
    CaseVerdict,
    Dataset,
    EvaluatorSpec,
    ExperimentResult,
    GateResult,
    Score,
    SliceResult,
    Suite,
    Totals,
)
from kavalai.eval.persona import Persona, PersonaRunner
from kavalai.eval.targets import RunRecord, Target, build_target


class BudgetExceeded(Exception):
    """The experiment hit its token ceiling and stopped rather than billing on."""


def load_setup(path: Path) -> Any:
    """Import the suite's setup module by file path.

    A suite is a directory you can copy anywhere, so its setup module is
    addressed by path rather than by an importable package name. Importing it
    is what registers the ``python://`` tools, the named RAG services and any
    custom evaluators the workflow and dataset refer to — and the engine
    resolves RAG services *eagerly*, so without this a non-trivial workflow
    cannot even be constructed.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Suite setup module not found: {path}")
    # The example packages import their own siblings; putting the parent of the
    # example directory on the path is what makes ``examples.bakery.tools``
    # resolve the same way it would from a shell.
    for candidate in (path.parent, path.parent.parent, path.parent.parent.parent):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    module_name = f"kavalai_eval_setup_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.info(f"Loaded suite setup from {path}")
    return module


def external_id_for(suite: str, tag: str, case: str, repeat: int) -> str:
    """``eval:{suite}:{tag}:{case}:{repeat}``.

    A structured prefix, unique per (case, repeat), so every graded run gets
    its own session and ``LIKE 'eval:%'`` separates test traffic from
    production in one predicate. Nothing new is stored — existing rows just
    become filterable, and a failing case in the result file becomes a
    click-through into the backoffice.
    """
    return f"eval:{suite}:{tag}:{case}:{repeat}"


class Experiment:
    """One run of one suite.

    Built to be usable from a notebook as readily as from the CLI: construct
    it, ``await run()``, read the :class:`ExperimentResult`.
    """

    def __init__(
        self,
        suite: Suite,
        *,
        tag: str = "local",
        target: Optional[Target] = None,
        target_overrides: Optional[dict] = None,
        persist_sessions: bool = False,
        include_personas: bool = False,
        only_personas: bool = False,
        skip_model_evaluators: bool = False,
        skip_trajectory_evaluators: bool = False,
        progress: Optional[Callable[[CaseResult], None]] = None,
    ):
        self.suite = suite
        self.tag = tag
        self.target_overrides = target_overrides or {}
        self._target = target
        #: When set, each case runs under its own ``eval:`` external id so the
        #: agent's own session rows can be found later. Off by default: CI does
        #: not need a database, and most suites never touch one.
        self.persist_sessions = persist_sessions
        #: Persona conversations cost turns x 2 models x cases, which is the
        #: most expensive thing here. Nightly and before a release, not on
        #: every commit — so they are opt-in.
        self.include_personas = include_personas or only_personas
        self.only_personas = only_personas
        #: The free tier: deterministic and trajectory assertions only, no
        #: provider call from an evaluator. What it skipped goes in the report.
        self.skip_model_evaluators = skip_model_evaluators
        #: Set when the target cannot observe a trajectory and the caller has
        #: said, explicitly, to run the rest anyway.
        self.skip_trajectory_evaluators = skip_trajectory_evaluators
        self._skipped: set[str] = set()
        self.progress = progress
        self._evaluator_cache: dict[tuple, Evaluator] = {}
        self._rubrics: list[str] = []
        self._tokens_spent = 0

    # ------------------------------------------------------------- building
    def _build_target(self) -> Target:
        if self._target is not None:
            return self._target
        return build_target(
            self.suite.target, self.suite.base_dir, **self.target_overrides
        )

    def _evaluator(self, spec: EvaluatorSpec) -> Evaluator:
        """One instance per distinct spec, reused across cases.

        Judges hold an LLM client, so rebuilding them per case would open a
        client per case for no reason.
        """
        key = (spec.type, tuple(sorted((k, str(v)) for k, v in spec.options.items())))
        if key not in self._evaluator_cache:
            instance = build_evaluator(spec)
            rubric = spec.options.get("rubric")
            if isinstance(rubric, str):
                self._rubrics.append(rubric)
            self._evaluator_cache[key] = instance
        return self._evaluator_cache[key]

    # -------------------------------------------------------------- running
    async def run(self) -> ExperimentResult:
        dataset = (
            Dataset(name=self.suite.name)
            if self.only_personas
            else self.suite.load_dataset()
        )
        personas = self._load_personas()
        if not dataset.cases and not personas:
            raise ValueError(
                f"Suite '{self.suite.name}' has no cases; "
                f"checked {[str(p) for p in self.suite.dataset_paths()]}"
            )
        if self.suite.setup:
            load_setup(self.suite.resolve(self.suite.setup))

        target = self._build_target()
        notes: list[str] = []
        if not getattr(target, "observes_trajectory", False):
            self._check_trajectory_support(dataset, notes)

        notes.extend(self._persona_notes(personas))

        await target.setup()
        started = datetime.now(timezone.utc)
        try:
            results = await self._run_all(dataset, target)
            results += await self._run_personas(personas, target)
        finally:
            await target.aclose()

        result = self._aggregate(dataset, target, results, started, notes)
        if self.skip_model_evaluators:
            models = sorted(
                name
                for name in self._skipped
                if name in self._evaluator_cache_names(needs="model")
            )
            if models:
                result.notes.append(
                    f"Model-backed evaluators were skipped: {', '.join(models)}. "
                    "Those assertions did not run, so this result is not a full "
                    "pass."
                )
        self._apply_gate(result)
        return result

    def _check_trajectory_support(self, dataset: Dataset, notes: list[str]) -> None:
        """Fail fast when the target cannot answer what the suite asks it.

        Said once, before anything runs, rather than as the same paragraph
        repeated under every case. Refusing to start is deliberate: silently
        passing a ``tool_not_called`` safety assertion that never ran would be
        the worst outcome available, and quietly failing all of them would bury
        the one thing the operator needs to read.
        """
        specs = list(self.suite.evaluators) + list(dataset.evaluators)
        specs += [s for sl in self.suite.slices.values() for s in sl.evaluators]
        specs += [s for case in dataset.cases for s in case.evaluators]
        needing = sorted({s.type for s in specs if self._evaluator(s).needs_trajectory})
        if not needing:
            return
        if self.skip_trajectory_evaluators:
            self._skipped.update(needing)
            notes.append(
                "This target produces no trajectory, so these assertions did "
                f"not run: {', '.join(needing)}. Output-only evaluators only."
            )
            return
        raise ValueError(
            f"Suite '{self.suite.name}' asserts on the trajectory "
            f"({', '.join(needing)}), but target kind "
            f"'{self.suite.target.kind}' cannot observe one — only an "
            "in-process 'engine' target can.\n"
            "Either run it with target kind 'engine', or pass "
            "--skip-trajectory-evaluators to run the output-only checks and "
            "have the report say which assertions were dropped."
        )

    def _persona_notes(self, personas: list[Persona]) -> list[str]:
        """Warn when a chat persona will be talking to a workflow with no memory.

        An ``email`` persona quotes the previous reply into its next message, so
        the thread carries its own history and a stateless workflow behaves
        correctly. A ``chat`` persona has no such channel: without a database
        each turn is an independent run, which reads as an assistant that
        forgets everything — worth saying, because it looks like a bug in the
        workflow rather than a missing ``--persist-sessions``.
        """
        if self.persist_sessions:
            return []
        stateless = sorted({p.name for p in personas if p.channel != "email"})
        if not stateless:
            return []
        return [
            "These personas ran without a database, so each turn was an "
            f"independent run with no history: {', '.join(stateless)}. Pass "
            "--persist-sessions against a configured database for true "
            "multi-turn behaviour."
        ]

    def _load_personas(self) -> list[Persona]:
        if not self.include_personas:
            return []
        return [Persona.from_yaml(self.suite.resolve(p)) for p in self.suite.personas]

    async def _run_personas(
        self, personas: list[Persona], target: Target
    ) -> list[CaseResult]:
        """Play each persona out and grade the conversation as one case."""
        if not personas:
            return []
        semaphore = asyncio.Semaphore(max(1, self.suite.concurrency))
        jobs = [
            self._run_persona(persona, repeat, target, semaphore)
            for persona in personas
            for repeat in range(max(1, self.suite.repeats))
        ]
        return await asyncio.gather(*jobs)

    async def _run_persona(
        self,
        persona: Persona,
        repeat: int,
        target: Target,
        semaphore: asyncio.Semaphore,
    ) -> CaseResult:
        case = persona.as_case()
        async with semaphore:
            external_id = (
                external_id_for(self.suite.name, self.tag, persona.name, repeat)
                if self.persist_sessions
                else None
            )
            self._check_budget()
            runner = PersonaRunner(persona, target)
            conversation = await runner.run(external_id=external_id)
            record = conversation.as_record()
            self._tokens_spent += record.total_tokens
            # A persona is graded by its own evaluators and its slice's, not by
            # the suite's golden-case ones: "retrieval hit the right fact" is
            # not a question you can ask of an eight-turn conversation.
            scores = await self._score(case, record, self._persona_specs(persona))

        result = CaseResult(
            case=persona.name,
            repeat=repeat,
            slice=persona.slice,
            status=_status_of(record, scores),
            scores=scores,
            output=record.output,
            error=record.error,
            duration_seconds=record.duration_seconds,
            total_tokens=record.total_tokens,
            external_id=external_id,
            trace=record.trajectory.names(),
        )
        if self.progress:
            self.progress(result)
        return result

    async def _run_all(self, dataset: Dataset, target: Target) -> list[CaseResult]:
        semaphore = asyncio.Semaphore(max(1, self.suite.concurrency))
        jobs = [
            self._run_one(case, repeat, dataset, target, semaphore)
            for case in dataset.cases
            for repeat in range(max(1, self.suite.repeats))
        ]
        return await asyncio.gather(*jobs)

    async def _run_one(
        self,
        case: Case,
        repeat: int,
        dataset: Dataset,
        target: Target,
        semaphore: asyncio.Semaphore,
    ) -> CaseResult:
        async with semaphore:
            external_id = (
                external_id_for(self.suite.name, self.tag, case.name, repeat)
                if self.persist_sessions
                else None
            )
            self._check_budget()
            record = await target.run(case, external_id=external_id)
            self._tokens_spent += record.total_tokens
            scores = await self._score(
                case, record, self.suite.evaluators_for(case, dataset)
            )

        result = CaseResult(
            case=case.name,
            repeat=repeat,
            slice=case.slice,
            status=_status_of(record, scores),
            scores=scores,
            output=record.output,
            error=record.error,
            duration_seconds=record.duration_seconds,
            total_tokens=record.total_tokens,
            external_id=external_id,
            trace=record.trajectory.names(),
        )
        if self.progress:
            self.progress(result)
        return result

    def _check_budget(self) -> None:
        limit = self.suite.gate.max_tokens
        if limit is not None and self._tokens_spent >= limit:
            raise BudgetExceeded(
                f"Experiment spent {self._tokens_spent:,} tokens, at or over the "
                f"{limit:,} ceiling in the suite's gate. Nothing was truncated: "
                "the run stopped instead of billing on."
            )

    def _persona_specs(self, persona: Persona) -> list[EvaluatorSpec]:
        """The evaluators for one persona.

        A persona that declares its own replaces the slice's rather than adding
        to them. Some personas exist precisely to *fail* their stated goal —
        the customer who refuses to say what "the usual" is — and a slice
        default of ``goal_achieved`` would mark the correct outcome as a
        failure. Declaring evaluators means taking control of them.
        """
        if persona.evaluators:
            return list(persona.evaluators)
        slice_spec = self.suite.slices.get(persona.slice)
        return list(slice_spec.evaluators if slice_spec else [])

    async def _score(
        self, case: Case, record: RunRecord, specs: list[EvaluatorSpec]
    ) -> list[Score]:
        scores: list[Score] = []
        for spec in specs:
            try:
                evaluator = self._evaluator(spec)
                if (self.skip_model_evaluators and evaluator.needs_model) or (
                    self.skip_trajectory_evaluators and evaluator.needs_trajectory
                ):
                    self._skipped.add(spec.type)
                    continue
                produced = await evaluator.score(case, record)
            except EvaluationError as exc:
                # Cannot score is not the same as scores badly. Surfaced as a
                # failed score with the reason, never as a silent pass.
                scores.append(Score.boolean(spec.type, False, reason=str(exc)))
                continue
            except Exception as exc:
                logger.exception(f"Evaluator '{spec.type}' raised on '{case.name}'")
                scores.append(
                    Score.boolean(spec.type, False, reason=f"evaluator error: {exc}")
                )
                continue
            scores.extend(produced if isinstance(produced, list) else [produced])
        return scores

    # ---------------------------------------------------------- aggregation
    def _aggregate(
        self,
        dataset: Dataset,
        target: Target,
        results: list[CaseResult],
        started: datetime,
        notes: list[str],
    ) -> ExperimentResult:
        by_case: dict[str, list[CaseResult]] = {}
        for result in results:
            by_case.setdefault(result.case, []).append(result)

        verdicts = [
            _verdict(name, sorted(runs, key=lambda r: r.repeat))
            for name, runs in by_case.items()
        ]
        verdicts.sort(key=lambda v: v.case)

        totals = Totals(
            cases=len(verdicts),
            passed=sum(1 for v in verdicts if v.status == "passed"),
            failed=sum(1 for v in verdicts if v.status == "failed"),
            errors=sum(1 for v in verdicts if v.status == "error"),
            flaky=sum(1 for v in verdicts if v.status == "flaky"),
            total_tokens=sum(r.total_tokens for r in results),
            duration_seconds=sum(r.duration_seconds for r in results),
        )
        # Flaky counts as a pass for the rate: it did work, unreliably. It is
        # reported separately so the unreliability is never invisible.
        good = totals.passed + totals.flaky
        totals.pass_rate = good / totals.cases if totals.cases else 0.0

        return ExperimentResult(
            suite=self.suite.name,
            tag=self.tag,
            started_at=started,
            target=target.describe(),
            models=self._models(),
            judge_prompt_sha=rubric_sha(self._rubrics) if self._rubrics else None,
            totals=totals,
            slices=self._slice_results(verdicts),
            verdicts=verdicts,
            notes=notes,
        )

    def _evaluator_cache_names(self, needs: str) -> set[str]:
        """Names of built evaluators that need a model, or a trajectory."""
        attribute = "needs_model" if needs == "model" else "needs_trajectory"
        return {
            instance.name
            for instance in self._evaluator_cache.values()
            if getattr(instance, attribute, False)
        }

    def _models(self) -> dict:
        judges = sorted(
            {
                getattr(e, "model")
                for e in self._evaluator_cache.values()
                if getattr(e, "model", None)
            }
        )
        return {"judges": judges} if judges else {}

    def _slice_results(self, verdicts: list[CaseVerdict]) -> list[SliceResult]:
        names = sorted({v.slice for v in verdicts if v.slice})
        results = []
        for name in names:
            members = [v for v in verdicts if v.slice == name]
            passed = sum(1 for v in members if v.status in ("passed", "flaky"))
            rate = passed / len(members) if members else 0.0
            spec = self.suite.slices.get(name)
            threshold = spec.min_pass_rate if spec else None
            results.append(
                SliceResult(
                    name=name,
                    cases=len(members),
                    passed=passed,
                    pass_rate=rate,
                    min_pass_rate=threshold,
                    ok=threshold is None or rate >= threshold,
                )
            )
        return results

    # ----------------------------------------------------------------- gate
    def _apply_gate(self, result: ExperimentResult) -> None:
        gate = self.suite.gate
        outcome = GateResult()

        if result.totals.pass_rate < gate.min_pass_rate:
            outcome.passed = False
            outcome.reasons.append(
                f"pass rate {result.totals.pass_rate:.2f} < {gate.min_pass_rate:.2f}"
            )
        if result.totals.errors:
            outcome.passed = False
            outcome.reasons.append(
                f"{result.totals.errors} case(s) errored rather than being graded"
            )
        for slice_result in result.slices:
            if not slice_result.ok:
                outcome.passed = False
                outcome.reasons.append(
                    f"slice '{slice_result.name}' {slice_result.pass_rate:.2f} < "
                    f"{slice_result.min_pass_rate:.2f}"
                )
        for name in gate.required_evaluators:
            offenders = sorted(
                {
                    r.case
                    for r in result.results
                    for s in r.scores
                    if s.name == name and s.passed is False
                }
            )
            if offenders:
                outcome.passed = False
                outcome.reasons.append(
                    f"required evaluator '{name}' failed on: {', '.join(offenders)}"
                )

        baseline = self.suite.load_baseline()
        if baseline is not None:
            outcome.regressions, outcome.fixes = diff_against(baseline, result)
            limit = gate.max_regressions_vs_baseline
            if limit is not None and len(outcome.regressions) > limit:
                outcome.passed = False
                outcome.reasons.append(
                    f"{len(outcome.regressions)} regression(s) vs baseline "
                    f"(allowed {limit}): {', '.join(outcome.regressions)}"
                )
        result.gate = outcome


def _status_of(record: RunRecord, scores: list[Score]) -> str:
    if not record.ok:
        return "error"
    return "failed" if any(s.passed is False for s in scores) else "passed"


def _verdict(name: str, runs: list[CaseResult]) -> CaseVerdict:
    """Fold a case's repeats into one outcome.

    Majority vote: a judged case that fails once in three is reported as
    ``flaky`` and does not block, because a gate that cries wolf gets bypassed
    within a fortnight. An *error* in any repeat is never voted away — the run
    itself broke, and that is not something a majority can excuse.
    """
    passes = sum(1 for r in runs if r.status == "passed")
    total = len(runs)
    if any(r.status == "error" for r in runs):
        status = "error"
    elif passes == total:
        status = "passed"
    elif passes == 0:
        status = "failed"
    elif passes * 2 > total:
        status = "flaky"
    else:
        status = "failed"
    return CaseVerdict(
        case=name,
        slice=runs[0].slice if runs else None,
        status=status,
        passes=passes,
        total=total,
        results=runs,
    )


def diff_against(
    baseline: ExperimentResult, current: ExperimentResult
) -> tuple[list[str], list[str]]:
    """Cases that used to pass and now do not, and the reverse.

    A named case that used to pass and now fails is a far better signal than an
    aggregate crossing a line: it is specific, it is actionable, and it does
    not move just because the suite grew.
    """
    regressions, fixes = [], []
    for verdict in current.verdicts:
        was = baseline.status_of(verdict.case)
        if was is None:
            continue
        was_good = was in ("passed", "flaky")
        is_good = verdict.status in ("passed", "flaky")
        if was_good and not is_good:
            regressions.append(verdict.case)
        elif not was_good and is_good:
            fixes.append(verdict.case)
    return sorted(regressions), sorted(fixes)


async def run_suite(
    suite_path: Path, *, tag: str = "local", **kwargs: Any
) -> ExperimentResult:
    """Load a suite file and run it. The one-liner behind the CLI."""
    return await Experiment(Suite.from_yaml(suite_path), tag=tag, **kwargs).run()


def assert_suite_passes(result: ExperimentResult) -> None:
    """Raise :class:`AssertionError` unless the gate passed.

    For teams who prefer their acceptance tests to be tests::

        async def test_acceptance():
            suite = Suite.from_yaml("examples/bakery/eval/suite.yaml")
            assert_suite_passes(await Experiment(suite, tag="ci").run())
    """
    if result.gate.passed:
        return
    lines = [f"Suite '{result.suite}' did not pass its gate:"]
    lines += [f"  - {reason}" for reason in result.gate.reasons]
    for verdict in result.verdicts:
        if verdict.status in ("failed", "error"):
            reasons = {
                s.reason for r in verdict.results for s in r.failed_scores if s.reason
            }
            detail = f" ({'; '.join(sorted(reasons))})" if reasons else ""
            lines.append(f"  * {verdict.case}: {verdict.status}{detail}")
    raise AssertionError("\n".join(lines))
