"""The shipped examples, checked without an API key.

This is what tier zero looks like: the bakery's business rules, the graph in
``assistant.yaml`` and both example case files are exercised on every pull
request, in under a second, with no secrets on the runner. Grading the agents
themselves needs a running server and a model, and that is
``kavalai-eval examples/<example>/eval_cases.yaml --port …`` — a separate job.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from examples.bakery import bakery
from examples.bakery.bakery import (
    CATALOGUE,
    InMemoryOrderBook,
    Order,
    OrderItem,
    ValidationResult,
    compose_reply,
    resolve_product,
    validate_order,
)
from kavalai.agent_service import AgentService
from kavalai.db import db_manager
from kavalai.eval import load_suite

REPO = Path(__file__).resolve().parent.parent
GREEN_VILLAGE = REPO / "examples" / "green_village"
BAKERY = REPO / "examples" / "bakery"


@pytest.fixture(autouse=True)
def restore_order_book():
    """The active order book is module state, so put back what was there."""
    before = bakery._ORDER_BOOK
    try:
        yield
    finally:
        bakery._ORDER_BOOK = before


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
        # No items at all is a different gap from an item with no quantity.
        ([], "2026-09-12", "Mari", ["items"]),
    ],
)
def test_the_validator_is_deterministic(items, delivery, name, expected_missing):
    """The rules live in Python so a model upgrade cannot move them."""
    order = Order(
        customer_name=name,
        delivery_date=delivery,
        items=[OrderItem(product=p, quantity=q) for p, q in items],
    )
    result = validate_order(order=order)
    assert result.missing_fields == expected_missing
    assert result.ok is (not expected_missing)


def test_an_absent_order_is_reported_rather_than_crashing():
    result = validate_order(order=None)
    assert result.ok is False
    assert result.missing_fields == ["order"]


@pytest.mark.parametrize(
    "written,expected",
    [
        ("sourdough loaves", "sourdough loaf"),
        ("Rye Bread", "rye loaf"),
        ("kringles", "kringle"),
        # Estonian inflects; the forms customers write are in the alias table.
        ("kaneelirulli", "cinnamon bun"),
        ("rukkileiba", "rye loaf"),
        # A misspelling is not an alias: it stays a question to the customer.
        ("cinamon buns", None),
        ("", None),
    ],
)
def test_products_resolve_only_through_the_alias_table(written, expected):
    assert resolve_product(written) == expected


def test_a_validated_order_is_normalised_to_catalogue_names():
    """The order book stores what the bakery calls it, not what was typed."""
    result = validate_order(
        order=Order(
            customer_name="Mari",
            delivery_date="2026-09-12",
            items=[OrderItem(product="sourdough loaves", quantity=2)],
        )
    )
    assert result.order.items[0].product == "sourdough loaf"
    assert result.order.items[0].unit == "loaf"


@pytest.mark.parametrize("written", ["2026-09-12", "12.09.2026", "12/09/2026"])
def test_the_date_formats_customers_actually_write(written):
    order = Order(
        customer_name="Mari",
        delivery_date=written,
        items=[OrderItem(product="kringle", quantity=4)],
    )
    assert validate_order(order=order).ok is True


def test_the_clock_is_pinned_so_the_cases_do_not_rot():
    """A wall clock would expire every shipped case as its dates pass."""
    assert bakery.TODAY == date(2026, 9, 1)
    lead = CATALOGUE["birthday cake"].lead_time_days
    earliest = (bakery.TODAY + timedelta(days=lead)).isoformat()
    order = Order(
        customer_name="Mari",
        delivery_date=earliest,
        items=[OrderItem(product="birthday cake", quantity=1)],
    )
    assert validate_order(order=order).ok is True


# ------------------------------------------------------------ the outcome
def test_compose_reply_derives_the_outcome_from_what_python_decided():
    """Three branches, told apart by what is absent rather than by wording."""
    complete = ValidationResult(ok=True)
    incomplete = ValidationResult(ok=False, missing_fields=["delivery_date"])

    stored = compose_reply(intent="order", validation=complete, order_id="ord-0001")
    assert (stored.status, stored.order_id, stored.missing) == (
        "stored",
        "ord-0001",
        [],
    )

    needs = compose_reply(intent="order", validation=incomplete)
    assert (needs.status, needs.order_id, needs.missing) == (
        "needs_details",
        "",
        ["delivery_date"],
    )

    other = compose_reply(intent="complaint")
    assert (other.status, other.order_id, other.missing) == ("not_an_order", "", [])


# --------------------------------------------------------- the order book
async def test_the_in_memory_order_book_numbers_orders_sequentially():
    book = InMemoryOrderBook()
    order = Order(
        customer_name="A",
        delivery_date="2026-09-12",
        items=[OrderItem(product="kringle", quantity=2)],
    )
    assert await book.store(order) == "ord-0001"
    assert await book.store(order) == "ord-0002"
    assert len(await book.orders()) == 2


def test_the_tools_refuse_to_guess_where_to_write():
    """Nothing else stops a demo writing into the real order book."""
    bakery._ORDER_BOOK = None
    with pytest.raises(RuntimeError, match="refuse to guess"):
        bakery.current_order_book()


# --------------------------------------------------- the shipped documents
def test_the_bakery_workflow_loads():
    """A typo in assistant.yaml should fail here, not on the first request."""
    bakery.use_order_book(InMemoryOrderBook())
    session_maker = db_manager.get_sqlite_sessionmaker(db_path=":memory:")
    engine = bakery.build_engine(AgentService(session_maker))

    names = {node.name for node in engine.graph.nodes}
    assert {"parse", "validate", "is_complete", "store", "compose"} <= names

    # Every intent the parse step may produce is a case in the switch, so
    # reaching `default` would mean the model answered outside the enum.
    # Leaving the non-order labels to `default` would make a real classifier
    # bug indistinguishable from ordinary routing.
    graph = yaml.safe_load(bakery.WORKFLOW_PATH.read_text(encoding="utf-8"))
    intents = graph["data_types"]["parsed"]["properties"]["intent"]["enum"]
    assert set(engine.graph.node_map["route"].cases) == set(intents)


@pytest.mark.parametrize(
    "directory", [GREEN_VILLAGE, BAKERY], ids=["green_village", "bakery"]
)
def test_the_example_case_files_are_valid(directory):
    """A malformed case file should fail here, not two minutes into a run."""
    suite = load_suite(directory / "eval_cases.yaml")
    assert suite.cases
    for case in suite.cases:
        assert case.input, f"{case.name} sends the agent nothing"
        if case.type == "judge":
            # Refused by EvalCase too; asserted here because a judged case
            # with no criterion passes on any answer at all.
            assert isinstance(case.expected, str) and case.expected.strip()
