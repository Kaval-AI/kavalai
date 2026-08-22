"""Rendering a result: a console table, a JSON file, JUnit XML and a diff.

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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.table import Table

from kavalai.eval.models import CaseVerdict, ExperimentResult

_STATUS_STYLE = {
    "passed": "[green]pass[/green]",
    "failed": "[red]FAIL[/red]",
    "error": "[red]ERROR[/red]",
    "flaky": "[yellow]flaky[/yellow]",
}


def print_report(
    result: ExperimentResult, console: Optional[Console] = None, verbose: bool = False
) -> None:
    """Print the human-readable report the CLI shows."""
    console = console or Console()
    target = result.target.get("workflow") or result.target.get("base_url") or ""
    console.print()
    console.print(
        f"[bold]{result.suite}[/bold] · {result.target.get('kind')} {target} · "
        f"{result.totals.cases} cases"
    )
    for note in result.notes:
        console.print(f"[yellow]note[/yellow] {note}")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("case")
    table.add_column("slice")
    table.add_column("verdict")
    table.add_column("tokens", justify="right")
    table.add_column("seconds", justify="right")
    for verdict in result.verdicts:
        runs = verdict.results
        tokens = sum(r.total_tokens for r in runs)
        seconds = max((r.duration_seconds for r in runs), default=0.0)
        table.add_row(
            verdict.case,
            verdict.slice or "",
            _STATUS_STYLE.get(verdict.status, verdict.status)
            + (f" {verdict.passes}/{verdict.total}" if verdict.total > 1 else ""),
            f"{tokens:,}" if tokens else "",
            f"{seconds:.1f}",
        )
    console.print(table)

    _print_failures(console, result, verbose)
    _print_slices(console, result)
    _print_totals(console, result)


def _print_failures(console: Console, result: ExperimentResult, verbose: bool) -> None:
    """Show why each failing case failed, right under the table.

    The reason is the whole value of a failing gate: a red row nobody can act
    on is a red row people learn to ignore.
    """
    for verdict in result.verdicts:
        if verdict.status == "passed" and not verbose:
            continue
        reasons = _reasons(verdict)
        if not reasons:
            continue
        console.print(f"  [bold]{verdict.case}[/bold]")
        for name, reason in reasons:
            console.print(f"    {name}: {reason}")
        external_ids = {r.external_id for r in verdict.results if r.external_id}
        for external_id in sorted(external_ids):
            console.print(f"    session: {external_id}")


def _reasons(verdict: CaseVerdict) -> list[tuple[str, str]]:
    seen, reasons = set(), []
    for result in verdict.results:
        for score in result.scores:
            if score.passed is False and score.reason and score.reason not in seen:
                seen.add(score.reason)
                reasons.append((score.name, score.reason))
        if result.error and result.error not in seen:
            seen.add(result.error)
            reasons.append(("error", result.error))
    return reasons


def _print_slices(console: Console, result: ExperimentResult) -> None:
    if not result.slices:
        return
    console.print()
    for slice_result in result.slices:
        threshold = (
            f" (gate {slice_result.min_pass_rate:.2f})"
            if slice_result.min_pass_rate is not None
            else ""
        )
        mark = "[green]ok[/green]" if slice_result.ok else "[red]below gate[/red]"
        console.print(
            f"  {slice_result.name:<18} {slice_result.pass_rate:.2f}{threshold}  {mark}"
        )


def _print_totals(console: Console, result: ExperimentResult) -> None:
    totals = result.totals
    console.print()
    parts = [f"pass rate {totals.pass_rate:.2f}"]
    if totals.failed:
        parts.append(f"{totals.failed} failed")
    if totals.errors:
        parts.append(f"{totals.errors} errored")
    if totals.flaky:
        parts.append(f"{totals.flaky} flaky")
    if totals.total_tokens:
        parts.append(f"{totals.total_tokens:,} tokens")
    console.print("  " + " · ".join(parts))

    gate = result.gate
    if gate.regressions:
        console.print(
            f"  [red]regressions vs baseline:[/red] {', '.join(gate.regressions)}"
        )
    if gate.fixes:
        console.print(f"  [green]newly passing:[/green] {', '.join(gate.fixes)}")
    if gate.passed:
        console.print("  [green]gate passed[/green]")
    else:
        console.print("  [red]gate failed[/red]")
        for reason in gate.reasons:
            console.print(f"    - {reason}")


def print_diff(
    baseline: ExperimentResult,
    current: ExperimentResult,
    console: Optional[Console] = None,
) -> None:
    """Print what changed between two result files, regressions first."""
    from kavalai.eval.runner import diff_against

    console = console or Console()
    regressions, fixes = diff_against(baseline, current)
    console.print(
        f"[bold]{current.suite}[/bold] {current.tag} vs baseline {baseline.tag}"
    )
    console.print(
        f"  pass rate {baseline.totals.pass_rate:.2f} -> {current.totals.pass_rate:.2f}"
    )
    if regressions:
        console.print(f"  [red]{len(regressions)} now failing[/red]")
        for case in regressions:
            console.print(f"    - {case}")
    if fixes:
        console.print(f"  [green]{len(fixes)} now passing[/green]")
        for case in fixes:
            console.print(f"    + {case}")
    if not regressions and not fixes:
        console.print("  no change")


def comment_body(
    baseline: Optional[ExperimentResult], current: ExperimentResult
) -> str:
    """A plain-words summary for a pull-request comment.

    A baseline commit is easy to wave through: a diff showing three cases
    flipping from pass to fail, buried in a pull request that also changes a
    prompt, gets approved by someone who read the prompt. Stating it in words
    is what makes a behaviour change visible rather than merely committed.
    """
    from kavalai.eval.runner import diff_against

    lines = [
        f"**{current.suite}** ({current.tag}) — "
        f"{'passed' if current.gate.passed else 'FAILED'} the gate."
    ]
    totals = current.totals
    lines.append(
        f"{totals.passed}/{totals.cases} cases passed "
        f"({totals.pass_rate:.0%}), {totals.total_tokens:,} tokens."
    )
    if baseline is not None:
        regressions, fixes = diff_against(baseline, current)
        if regressions:
            lines.append(
                f"{len(regressions)} case(s) now fail that previously passed: "
                + ", ".join(f"`{c}`" for c in regressions)
            )
        if fixes:
            lines.append(
                f"{len(fixes)} case(s) now pass that previously failed: "
                + ", ".join(f"`{c}`" for c in fixes)
            )
        if not regressions and not fixes:
            lines.append("No case changed verdict against the baseline.")
    for reason in current.gate.reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def write_junit(result: ExperimentResult, path: Union[str, Path]) -> Path:
    """Write JUnit XML so GitHub Actions renders per-case failures natively."""
    suites = ET.Element(
        "testsuites",
        name=result.suite,
        tests=str(result.totals.cases),
        failures=str(result.totals.failed),
        errors=str(result.totals.errors),
    )
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=result.suite,
        tests=str(result.totals.cases),
        failures=str(result.totals.failed),
        errors=str(result.totals.errors),
        time=f"{result.totals.duration_seconds:.3f}",
    )
    for verdict in result.verdicts:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"{result.suite}.{verdict.slice or 'default'}",
            name=verdict.case,
            time=f"{max((r.duration_seconds for r in verdict.results), default=0.0):.3f}",
        )
        message = "; ".join(f"{n}: {r}" for n, r in _reasons(verdict))
        if verdict.status == "error":
            ET.SubElement(case, "error", message=message or "run failed").text = message
        elif verdict.status == "failed":
            ET.SubElement(
                case, "failure", message=message or "assertion failed"
            ).text = message
        elif verdict.status == "flaky":
            # Reported, not failed: it passed a majority of its repeats.
            ET.SubElement(
                case, "system-out"
            ).text = (
                f"flaky: passed {verdict.passes}/{verdict.total} repeats. {message}"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)
    return path
