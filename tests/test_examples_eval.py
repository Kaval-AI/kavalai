"""The shipped example suites, run end to end with no API key.

This is what tier zero looks like in practice: the retrieval, the routing, the
validation rules, the side effects and every deterministic evaluator are
exercised on each pull request, in a couple of seconds, with no secrets on the
runner. The model responses are replayed from committed fixtures — recorded
from the real models with ``kavalai-eval <suite> --record-fixtures``.

A live run against real providers is a separate, key-gated test at the bottom.
"""

import sys
from pathlib import Path

import pytest

from kavalai.eval import Experiment, Suite, assert_suite_passes
from kavalai.eval.fixtures import fixture_client_factory
from kavalai.eval.runner import load_setup

REPO = Path(__file__).resolve().parent.parent
GREEN_VILLAGE = REPO / "examples" / "green_village"
BAKERY = REPO / "examples" / "bakery"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def restore_registries():
    """Undo what the examples' setup modules register.

    ``eval_setup.py`` registers a RAG service under ``"default"`` and several
    evaluators by name — process-global state, by design, because that is how a
    workflow refers to them. Left in place it leaks into every test that runs
    afterwards, so this snapshots and restores it.
    """
    from kavalai.eval.evaluators.base import REGISTRY
    from kavalai.llm_clients.registry import rag_services

    rag_before = dict(rag_services._targets)
    rag_defaults = dict(rag_services._defaults)
    evaluators_before = dict(REGISTRY)
    try:
        yield
    finally:
        rag_services._targets.clear()
        rag_services._targets.update(rag_before)
        rag_services._defaults.clear()
        rag_services._defaults.update(rag_defaults)
        REGISTRY.clear()
        REGISTRY.update(evaluators_before)


def suite_for(directory: Path) -> Suite:
    return Suite.from_yaml(directory / "eval" / "suite.yaml")


async def run_offline(directory: Path) -> "ExperimentResult":  # noqa: F821
    """Run a suite from its committed fixtures, with judges off."""
    suite = suite_for(directory)
    fixtures = suite.resolve(suite.target.fixtures)
    assert fixtures.exists(), (
        f"{fixtures} is missing. Record it with:\n"
        f"  kavalai-eval {suite.resolve('suite.yaml')} --record-fixtures"
    )
    experiment = Experiment(
        suite,
        tag="pytest",
        target_overrides={"client_factory": fixture_client_factory(fixtures)},
        skip_model_evaluators=True,
    )
    return await experiment.run()


# --------------------------------------------------------- the corpus itself
def test_the_green_village_index_matches_the_facts():
    """A stale index grades new questions against an old corpus and passes."""
    import json

    sys.path.insert(0, str(GREEN_VILLAGE))
    from facts import FACTS, corpus_fingerprint

    stamp = json.loads((GREEN_VILLAGE / "green_village.index.json").read_text())
    assert stamp["facts"] == len(FACTS)
    assert stamp["fingerprint"] == corpus_fingerprint(), (
        "facts.py has changed since the index was built. Rebuild it:\n"
        "  uv run python examples/green_village/build_index.py"
    )


def test_every_seed_email_has_an_expectation():
    """An email nobody wrote ground truth for is an email nobody grades."""
    sys.path.insert(0, str(BAKERY))
    from archetypes import SEEDS

    on_disk = {p.name for p in (BAKERY / "inbox").glob("*.eml")}
    assert on_disk == set(
        SEEDS
    ), f"inbox and archetypes.py disagree: {on_disk ^ set(SEEDS)}"


# ------------------------------------------------------------- the rules
@pytest.mark.parametrize(
    "items,delivery,name,expected_missing",
    [
        ([("kringle", 4)], "2026-09-12", "Mari", []),
        ([("kringle", None)], "2026-09-12", "Mari", ["items[0].quantity"]),
        ([("kringle", 4)], None, "Mari", ["delivery_date"]),
        ([("kringle", 4)], "2026-09-12", None, ["customer_name"]),
        ([("unicorn", 4)], "2026-09-12", "Mari", ["items[0].product"]),
        # Below the batch minimum, and above what email may take unattended.
        ([("kringle", 1)], "2026-09-12", "Mari", ["items[0].quantity"]),
        ([("birthday cake", 500)], "2026-09-12", "Mari", ["items[0].quantity"]),
        # Too soon: a birthday cake needs three days' notice.
        ([("birthday cake", 1)], "2026-09-02", "Mari", ["delivery_date"]),
        # One incomplete item makes the whole order incomplete.
        (
            [("rye loaf", 2), ("cinnamon bun", None)],
            "2026-09-12",
            "Mari",
            ["items[1].quantity"],
        ),
    ],
)
def test_the_bakery_validator_is_deterministic(items, delivery, name, expected_missing):
    """The rules live in Python so a model upgrade cannot move them."""
    sys.path.insert(0, str(BAKERY))
    import tools
    from models import Order, OrderItem

    tools.new_workspace()
    order = Order(
        customer_name=name,
        delivery_date=delivery,
        items=[OrderItem(product=p, quantity=q) for p, q in items],
    )
    result = tools.validate_order(order=order)
    assert result.missing_fields == expected_missing
    assert result.ok is (not expected_missing)


def test_a_validated_order_is_normalised_to_catalogue_names():
    sys.path.insert(0, str(BAKERY))
    import tools
    from models import Order, OrderItem

    tools.new_workspace()
    result = tools.validate_order(
        order=Order(
            customer_name="Mari",
            delivery_date="2026-09-12",
            items=[OrderItem(product="sourdough loaves", quantity=2)],
        )
    )
    assert result.order.items[0].product == "sourdough loaf"
    assert result.order.items[0].unit == "loaf"


def test_the_bakery_workspace_isolates_its_world(tmp_path):
    sys.path.insert(0, str(BAKERY))
    import tools
    from models import Order, OrderItem

    first = tools.new_workspace()
    tools.store_order(
        order=Order(
            customer_name="A",
            delivery_date="2026-09-12",
            items=[OrderItem(product="kringle", quantity=2)],
        )
    )
    tools.send_reply(to="a@example.test", subject="s", body="b")
    assert len(first.orders()) == 1
    assert len(first.sent_mail()) == 1
    # Order ids are sequential, which is what lets a recorded fixture match.
    assert first.orders()[0]["id"] == "ord-0001"

    second = tools.new_workspace()
    assert second.orders() == [] and second.sent_mail() == []
    first.cleanup()
    second.cleanup()


def test_the_tools_refuse_to_guess_where_to_write():
    sys.path.insert(0, str(BAKERY))
    import tools

    tools._WORKSPACE.set(None)
    with pytest.raises(RuntimeError, match="refuse to guess"):
        tools.current_workspace()


# ---------------------------------------------------- the suites, keyless
def test_the_example_suites_only_name_evaluators_that_exist():
    """A typo in a suite file should fail here, not two minutes into a run."""
    from kavalai.eval import build_evaluator

    for directory in (GREEN_VILLAGE, BAKERY):
        suite = suite_for(directory)
        load_setup(suite.resolve(suite.setup))
        dataset = suite.load_dataset()
        specs = list(suite.evaluators)
        specs += [s for slice_ in suite.slices.values() for s in slice_.evaluators]
        specs += [s for case in dataset.cases for s in case.evaluators]
        for spec in specs:
            build_evaluator(spec)


@pytest.mark.parametrize(
    "directory", [GREEN_VILLAGE, BAKERY], ids=["green_village", "bakery"]
)
async def test_the_example_suite_passes_offline(directory):
    result = await run_offline(directory)
    # The report has to admit which assertions did not run.
    assert any("skipped" in note for note in result.notes)
    assert_suite_passes(result)


@pytest.mark.parametrize(
    "directory", [GREEN_VILLAGE, BAKERY], ids=["green_village", "bakery"]
)
async def test_the_offline_run_produces_a_full_trajectory(directory):
    """Trajectory assertions are the point; prove they had something to read."""
    result = await run_offline(directory)
    traces = [r.trace for r in result.results]
    assert all(len(trace) >= 3 for trace in traces)
    assert all(trace[0] == "begin" for trace in traces)


async def test_the_bakery_never_stores_an_incomplete_order():
    """The failure this whole example exists to catch, asserted directly."""
    result = await run_offline(BAKERY)
    for verdict in result.verdicts:
        if verdict.slice in ("order_incomplete", "not_an_order"):
            assert verdict.status == "passed", verdict.case
