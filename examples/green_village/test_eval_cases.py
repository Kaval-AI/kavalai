"""The shipped Green Village cases parse, and fit the chatbot they grade.

A suite file is only useful if it loads: a case with a key nobody recognises,
or a judged case with no criterion, is refused by ``load_suite`` rather than
run. Checking that here means the 64 cases are known to be runnable without a
server, an API key or a network — the run itself is
``kavalai-eval examples/green_village/eval_cases.yaml --port 25000``.
"""

from pathlib import Path

from examples.green_village.green_village_support_in_memory import Message
from kavalai.eval.eval_runner import load_suite

CASES = Path(__file__).resolve().parent / "eval_cases.yaml"


def test_the_shipped_cases_are_a_valid_suite():
    suite = load_suite(CASES)

    assert suite.cases
    # Both kinds are used: literal matchers for the facts, a judge for the
    # answers whose wording is free but whose substance is not.
    assert {case.type for case in suite.cases} == {"simple", "judge"}


def test_every_case_fits_the_chatbots_input_type():
    """A mistyped field is a case that never runs, so it is caught here."""
    for case in load_suite(CASES).cases:
        Message(**case.input)
