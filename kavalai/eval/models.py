"""Data model for evaluation: cases, datasets, suites and results.

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

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from kavalai.workflow.tasklog.memory import TaskRecord


class EvaluatorSpec(BaseModel):
    """One evaluator, named and configured from YAML.

    ``{type: contains, text: "60 days"}`` becomes ``type="contains"`` with
    ``{"text": "60 days"}`` as options. Keeping it declarative is what makes a
    dataset reviewable in a pull request.
    """

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def options(self) -> dict:
        """The evaluator's keyword arguments — everything but ``type``."""
        return dict(self.model_extra or {})

    @classmethod
    def coerce(cls, value: Union[str, dict, "EvaluatorSpec"]) -> "EvaluatorSpec":
        """Accept ``no_error``, ``{type: no_error}`` or an existing spec.

        The bare-string form is there because most evaluators take no options
        and ``- no_error`` reads better in a list than ``- {type: no_error}``.
        """
        if isinstance(value, EvaluatorSpec):
            return value
        if isinstance(value, str):
            return cls(type=value)
        return cls(**value)


def _coerce_specs(value: Any) -> Any:
    """Accept the shorthand forms wherever a list of evaluators is expected."""
    if isinstance(value, list):
        return [EvaluatorSpec.coerce(v) for v in value]
    return value


#: A list of evaluators, written in YAML as ``- no_error`` or
#: ``- {type: contains, text: "60 days"}``. Coercion happens *before*
#: validation, so both forms are accepted everywhere one of these appears.
EvaluatorSpecs = Annotated[list[EvaluatorSpec], BeforeValidator(_coerce_specs)]


class Case(BaseModel):
    """One input the system under test is asked to handle, and how to grade it."""

    name: str
    #: The workflow input. For a chat workflow this is usually
    #: ``{"user_message": "..."}``.
    inputs: dict = Field(default_factory=dict)
    #: Ground truth, when there is any. Evaluators decide what they read from
    #: it; ``equals_expected`` compares it to the output as a whole.
    expected: Optional[Any] = None
    #: Which slice of the suite this case belongs to. Slices carry their own
    #: pass-rate thresholds, because one number over a mixed corpus hides
    #: exactly the movements worth seeing.
    slice: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    #: Evaluators for this case only, appended to the dataset's and suite's.
    evaluators: EvaluatorSpecs = Field(default_factory=list)


class Dataset(BaseModel):
    """A named list of cases, plus the evaluators every one of them gets."""

    name: str
    cases: list[Case] = Field(default_factory=list)
    evaluators: EvaluatorSpecs = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Dataset":
        """Load a dataset file. The file name is the default dataset name."""
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write the dataset back out, ready to be committed and reviewed."""
        Path(path).write_text(
            yaml.safe_dump(
                self.model_dump(exclude_none=True, exclude_defaults=True),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    def slices(self) -> set[str]:
        return {c.slice for c in self.cases if c.slice}


class Score(BaseModel):
    """One evaluator's verdict on one run.

    ``passed`` is deliberately three-valued: ``None`` means *measured, not
    asserted* — a number worth recording that nobody has set a threshold on.
    """

    name: str
    value: float = 0.0
    passed: Optional[bool] = None
    #: Why. Judges explain themselves here, and so do threshold evaluators:
    #: ``"4,812 tokens > 3,000"`` is a self-explanatory JUnit failure.
    reason: Optional[str] = None
    meta: dict = Field(default_factory=dict)

    @classmethod
    def boolean(
        cls, name: str, ok: bool, reason: Optional[str] = None, **meta: Any
    ) -> "Score":
        return cls(
            name=name, value=1.0 if ok else 0.0, passed=ok, reason=reason, meta=meta
        )


class CaseResult(BaseModel):
    """What happened when one case was run once."""

    case: str
    repeat: int = 0
    slice: Optional[str] = None
    #: ``passed`` — every asserted score passed. ``failed`` — at least one did
    #: not. ``error`` — the run itself blew up, which is not the same thing and
    #: must never be reported as a graded failure.
    status: str = "passed"
    scores: list[Score] = Field(default_factory=list)
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    total_tokens: int = 0
    #: ``eval:{suite}:{tag}:{case}:{repeat}`` — paste it into the backoffice
    #: session filter to look at the conversation that failed.
    external_id: Optional[str] = None
    trace: list[str] = Field(default_factory=list)

    @property
    def failed_scores(self) -> list[Score]:
        return [s for s in self.scores if s.passed is False]


class CaseVerdict(BaseModel):
    """One case's outcome across all its repeats.

    A judged case can flake; ``repeats`` with a majority vote is what keeps a
    stochastic grader from turning a gate into noise. A case that neither
    passed nor failed cleanly is reported as ``flaky`` and does not block.
    """

    case: str
    slice: Optional[str] = None
    status: str = "passed"
    passes: int = 0
    total: int = 0
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.total


class Totals(BaseModel):
    cases: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    flaky: int = 0
    pass_rate: float = 0.0
    total_tokens: int = 0
    duration_seconds: float = 0.0


class SliceResult(BaseModel):
    name: str
    cases: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    min_pass_rate: Optional[float] = None
    ok: bool = True


class GateResult(BaseModel):
    """Why the run passed or failed, in terms a person can act on."""

    passed: bool = True
    reasons: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """The result file: one run of one suite, at one tag."""

    suite: str
    tag: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target: dict = Field(default_factory=dict)
    models: dict = Field(default_factory=dict)
    #: Hash of the judge rubrics used. A judge model or rubric that moves makes
    #: historical scores incomparable; recording this is the only way to tell a
    #: regression in the workflow from a change in the grader.
    judge_prompt_sha: Optional[str] = None
    totals: Totals = Field(default_factory=Totals)
    slices: list[SliceResult] = Field(default_factory=list)
    verdicts: list[CaseVerdict] = Field(default_factory=list)
    gate: GateResult = Field(default_factory=GateResult)
    #: Set when trajectory evaluators cannot run against this target, so a
    #: report never reads as green because it could not see anything.
    notes: list[str] = Field(default_factory=list)

    @property
    def results(self) -> list[CaseResult]:
        return [r for v in self.verdicts for r in v.results]

    def status_of(self, case: str) -> Optional[str]:
        for verdict in self.verdicts:
            if verdict.case == case:
                return verdict.status
        return None

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            self.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentResult":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------- suites
class TargetSpec(BaseModel):
    """What the suite runs against."""

    model_config = ConfigDict(extra="allow")

    #: ``engine`` runs the workflow in this process and sees its full
    #: trajectory. ``rest`` drives a deployed agent server and sees only its
    #: output. ``callable`` is the escape hatch for anything else.
    kind: str = "engine"
    #: For ``kind: engine`` — the workflow YAML, relative to the suite file.
    workflow: Optional[str] = None
    #: For ``kind: rest`` — where the agent server is. ``${VAR}`` is expanded
    #: from the environment by the CLI, never by the library.
    base_url: Optional[str] = None
    path: str = "/run_agent"
    timeout_seconds: float = 120.0
    #: ``module:function`` called before every case (and every repeat) to reset
    #: the world a side-effecting workflow writes to. Whatever it returns is
    #: handed to the evaluators as ``record.sandbox``.
    sandbox: Optional[str] = None
    #: For ``kind: callable`` — ``module:function`` taking the case inputs.
    function: Optional[str] = None
    #: Where recorded model responses live, so the suite can run with no API
    #: key. Used only when the CLI is asked for them (``--fixtures``).
    fixtures: str = "fixtures/llm.json"

    @property
    def options(self) -> dict:
        return dict(self.model_extra or {})


class SliceSpec(BaseModel):
    """Extra evaluators and a threshold for one named slice of the dataset."""

    evaluators: EvaluatorSpecs = Field(default_factory=list)
    min_pass_rate: Optional[float] = None


class GateSpec(BaseModel):
    """What makes the run exit non-zero.

    Both thresholds matter and neither is enough alone: the absolute floor lets
    quality ratchet down one case at a time, and the regression check alone
    lets a permanently-broken case stay broken.
    """

    min_pass_rate: float = 0.0
    max_regressions_vs_baseline: Optional[int] = 0
    #: Any failure of these evaluators fails the run outright, whatever the
    #: pass rate says.
    required_evaluators: list[str] = Field(default_factory=list)
    #: Abort and mark the experiment ``budget_exceeded`` rather than silently
    #: running up a bill.
    max_tokens: Optional[int] = None


class Suite(BaseModel):
    """One acceptance suite: a dataset, a target, evaluators and a threshold.

    Every path in the file is relative to the file itself, so a suite is a
    directory you can copy anywhere.
    """

    name: str
    #: One dataset file, or several.
    dataset: Union[str, list[str]] = Field(default_factory=list)
    baseline: str = "baseline.json"
    results_dir: str = "results"
    #: Imported before the run: registers the ``python://`` tools, RAG services
    #: and custom evaluators the workflow and the dataset name. Without it
    #: ``EngineTarget`` cannot even construct a non-trivial workflow, because
    #: the engine resolves named RAG services eagerly.
    setup: Optional[str] = None
    target: TargetSpec = Field(default_factory=TargetSpec)
    repeats: int = 1
    concurrency: int = 4
    evaluators: EvaluatorSpecs = Field(default_factory=list)
    slices: dict[str, SliceSpec] = Field(default_factory=dict)
    #: Persona files run as extra, conversational cases (``--personas``).
    personas: list[str] = Field(default_factory=list)
    gate: GateSpec = Field(default_factory=GateSpec)

    #: Directory the file was loaded from; every relative path resolves against
    #: it. Excluded from a dump so a suite round-trips unchanged.
    base_dir: Path = Field(default=Path("."), exclude=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Suite":
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.parent.name)
        data["base_dir"] = path.parent
        return cls(**data)

    def resolve(self, relative: Union[str, Path]) -> Path:
        """Resolve a path from the suite file against the suite's directory."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.base_dir / candidate)

    def dataset_paths(self) -> list[Path]:
        names = [self.dataset] if isinstance(self.dataset, str) else list(self.dataset)
        return [self.resolve(n) for n in names]

    def load_dataset(self) -> Dataset:
        """Load and merge every dataset the suite names."""
        datasets = [Dataset.from_yaml(p) for p in self.dataset_paths()]
        if len(datasets) == 1:
            return datasets[0]
        merged = Dataset(name=self.name)
        for dataset in datasets:
            for case in dataset.cases:
                case.evaluators = list(dataset.evaluators) + list(case.evaluators)
                merged.cases.append(case)
        return merged

    def baseline_path(self) -> Path:
        return self.resolve(self.baseline)

    def load_baseline(self) -> Optional[ExperimentResult]:
        path = self.baseline_path()
        return ExperimentResult.from_json(path) if path.exists() else None

    def result_path(self, tag: str, suffix: str = ".json") -> Path:
        return self.resolve(self.results_dir) / f"{tag}{suffix}"

    def evaluators_for(self, case: Case, dataset: Dataset) -> list[EvaluatorSpec]:
        """Every evaluator that applies to ``case``, suite-wide first.

        Three layers, most general first: the suite's, the dataset's, the
        slice's, then the case's own. Nothing is deduplicated — an evaluator
        named twice runs twice, which is visible in the report rather than
        silently swallowed.
        """
        specs = list(self.evaluators) + list(dataset.evaluators)
        if case.slice and case.slice in self.slices:
            specs += list(self.slices[case.slice].evaluators)
        return specs + list(case.evaluators)


__all__ = [
    "Case",
    "CaseResult",
    "CaseVerdict",
    "Dataset",
    "EvaluatorSpec",
    "EvaluatorSpecs",
    "ExperimentResult",
    "GateResult",
    "GateSpec",
    "Score",
    "SliceResult",
    "SliceSpec",
    "Suite",
    "TargetSpec",
    "TaskRecord",
    "Totals",
]
