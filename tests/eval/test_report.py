"""What the run prints, writes and compares."""

import xml.etree.ElementTree as ET

from rich.console import Console

from kavalai.eval import (
    ExperimentResult,
    comment_body,
    print_diff,
    print_report,
    write_junit,
)
from kavalai.eval.models import (
    CaseResult,
    CaseVerdict,
    GateResult,
    Score,
    SliceResult,
    Totals,
)


def result(**kwargs) -> ExperimentResult:
    kwargs.setdefault("suite", "s")
    kwargs.setdefault("tag", "t")
    return ExperimentResult(**kwargs)


def verdict(case, status, reason=None, **kwargs) -> CaseVerdict:
    scores = [Score(name="judge", value=0.0, passed=status == "passed", reason=reason)]
    return CaseVerdict(
        case=case,
        status=status,
        passes=1 if status == "passed" else 0,
        total=1,
        results=[CaseResult(case=case, status=status, scores=scores, **kwargs)],
    )


def render(fn, *args) -> str:
    console = Console(record=True, width=100, force_terminal=False)
    fn(*args, console)
    return console.export_text()


def test_the_report_shows_why_a_case_failed():
    """A red row nobody can act on is a red row people learn to ignore."""
    experiment = result(
        totals=Totals(cases=2, passed=1, failed=1, pass_rate=0.5, total_tokens=41802),
        verdicts=[
            verdict("ok_case", "passed"),
            verdict(
                "prompt_injection_ignored",
                "failed",
                reason="the assistant restated its instructions",
                external_id="eval:s:pr-412:prompt_injection_ignored:0",
            ),
        ],
        gate=GateResult(passed=False, reasons=["pass rate 0.50 < 0.95"]),
        target={"kind": "engine", "workflow": "assistant.yaml"},
    )
    text = render(print_report, experiment)

    assert "prompt_injection_ignored" in text
    assert "restated its instructions" in text
    # The click-through into the backoffice.
    assert "eval:s:pr-412:prompt_injection_ignored:0" in text
    assert "41,802 tokens" in text
    assert "gate failed" in text


def test_the_report_warns_when_the_target_could_not_see_a_trajectory():
    text = render(
        print_report, result(notes=["This target does not produce a trajectory"])
    )
    assert "does not produce a trajectory" in text


def test_the_report_shows_slice_thresholds_and_regressions():
    experiment = result(
        slices=[
            SliceResult(
                name="direct",
                cases=2,
                passed=1,
                pass_rate=0.5,
                min_pass_rate=1.0,
                ok=False,
            )
        ],
        gate=GateResult(passed=False, regressions=["a"], fixes=["b"], reasons=["nope"]),
    )
    text = render(print_report, experiment)
    assert "direct" in text and "below gate" in text
    assert "regressions vs baseline: a" in text
    assert "newly passing: b" in text


def test_a_flaky_case_is_shown_with_its_tally():
    experiment = result(
        totals=Totals(cases=1, flaky=1, pass_rate=1.0),
        verdicts=[
            CaseVerdict(
                case="c",
                status="flaky",
                passes=2,
                total=3,
                results=[CaseResult(case="c", status="passed")],
            )
        ],
    )
    text = render(print_report, experiment)
    assert "flaky 2/3" in text
    assert "1 flaky" in text


def test_junit_marks_failures_errors_and_flakes(tmp_path):
    experiment = result(
        totals=Totals(cases=3, passed=1, failed=1, errors=1),
        verdicts=[
            verdict("passing", "passed"),
            verdict("failing", "failed", reason="wrong answer"),
            CaseVerdict(
                case="broken",
                status="error",
                total=1,
                results=[CaseResult(case="broken", status="error", error="boom")],
            ),
            CaseVerdict(
                case="wobbly",
                status="flaky",
                passes=2,
                total=3,
                results=[CaseResult(case="wobbly", status="passed")],
            ),
        ],
    )
    path = write_junit(experiment, tmp_path / "out" / "r.junit.xml")
    root = ET.parse(path).getroot()
    cases = {c.get("name"): c for c in root.iter("testcase")}

    assert cases["passing"].find("failure") is None
    assert "wrong answer" in cases["failing"].find("failure").get("message")
    assert cases["broken"].find("error") is not None
    # Flaky is reported, not failed: it passed a majority of its repeats.
    assert cases["wobbly"].find("failure") is None
    assert "flaky: passed 2/3" in cases["wobbly"].find("system-out").text


def test_the_diff_names_what_changed():
    baseline = result(
        tag="baseline",
        totals=Totals(pass_rate=1.0),
        verdicts=[verdict("a", "passed"), verdict("b", "failed")],
    )
    current = result(
        tag="pr-1",
        totals=Totals(pass_rate=0.5),
        verdicts=[verdict("a", "failed"), verdict("b", "passed")],
    )

    text = render(print_diff, baseline, current)
    assert "1 now failing" in text and "- a" in text
    assert "1 now passing" in text and "+ b" in text

    same = render(print_diff, baseline, baseline)
    assert "no change" in same


def test_the_pull_request_comment_states_the_change_in_words():
    """A behaviour change should have to be stated, not merely committed."""
    baseline = result(tag="baseline", verdicts=[verdict("a", "passed")])
    current = result(
        tag="pr-1",
        totals=Totals(cases=1, passed=0, pass_rate=0.0, total_tokens=1234),
        verdicts=[verdict("a", "failed")],
        gate=GateResult(passed=False, reasons=["pass rate 0.00 < 0.95"]),
    )
    body = comment_body(baseline, current)

    assert "FAILED the gate" in body
    assert "1 case(s) now fail that previously passed: `a`" in body
    assert "1,234 tokens" in body
    assert "pass rate 0.00 < 0.95" in body


def test_the_comment_works_without_a_baseline():
    body = comment_body(None, result(totals=Totals(cases=1, passed=1, pass_rate=1.0)))
    assert "passed the gate" in body
    assert "now fail" not in body


def test_the_comment_says_when_nothing_changed():
    baseline = result(tag="b", verdicts=[verdict("a", "passed")])
    current = result(tag="c", verdicts=[verdict("a", "passed")])
    assert "No case changed verdict" in comment_body(baseline, current)
