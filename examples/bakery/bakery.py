"""Lindqvist Bakery Workshop — the Python half of the email assistant.

The agent itself is :file:`assistant.yaml`. This module holds only what that
graph calls into: the catalogue, the shapes the workflow passes around, an
order book, and three tools.

**The model extracts, Python decides.** The `parse` node turns prose into
fields and does nothing else; :func:`validate_order` is ordinary code and is
the only thing that says whether an order is complete. That is what lets
:file:`eval_cases.yaml` grade most of this agent by comparing values, and what
stops a model upgrade quietly redefining a valid order.

``TODAY`` is the date the bakery believes it is. It is pinned so that "we need
two days' notice" does not expire the shipped cases as their dates fall into
the past.

``CATALOGUE`` is what the bakery sells. Its key is the identifier the parse
step must write into ``product`` — one underscored token, which a model is far
less likely to get subtly wrong than a phrase. Mapping "sourdough bread" or
"kringles" onto one of these keys is the model's job; the same list is named
in the parse prompt in :file:`assistant.yaml`.

``ORDER_BOOK`` holds every order the assistant accepted, for as long as the
process lives. An example does not need a second database — the one in
:file:`bakery_real_db.py` is there for the agent's *sessions*, not for this.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from kavalai import pythontool

TODAY = date(2026, 9, 1)


@dataclass(frozen=True)
class Product:
    """A catalogue entry and the rules an order of it must satisfy.

    Below ``minimum_quantity`` the bakery cannot bake economically, so the
    order is incomplete. ``lead_time_days`` is the notice needed before the
    delivery date. Above ``maximum_quantity`` a human has to be involved,
    whoever is asking.
    """

    unit: str
    minimum_quantity: int
    lead_time_days: int
    maximum_quantity: int


CATALOGUE: dict[str, Product] = {
    #                       unit,  min, lead, max
    "sourdough_loaf": Product("loaf", 1, 1, 60),
    "rye_loaf": Product("loaf", 1, 1, 60),
    "kringle": Product("piece", 2, 2, 40),
    "cinnamon_bun": Product("piece", 6, 1, 200),
    "birthday_cake": Product("cake", 1, 3, 5),
}


def product_name(key: str) -> str:
    """`sourdough_loaf` as a customer would read it."""
    return key.replace("_", " ")


def catalogue_text() -> str:
    """The product list, for a reply that has to offer options."""
    return "\n".join(
        f"- {product_name(key)} (per {p.unit}, {p.minimum_quantity}-"
        f"{p.maximum_quantity}, {p.lead_time_days} day(s) notice)"
        for key, p in CATALOGUE.items()
    )


class OrderItem(BaseModel):
    """One line of an order.

    Every field is optional on purpose: extraction must be able to say "the
    customer did not state a quantity" rather than inventing one. A required
    field forces a guess, and a guessed quantity is the failure this example
    exists to catch.
    """

    product: Optional[str] = Field(default=None, description="A CATALOGUE key.")
    quantity: Optional[float] = Field(
        default=None,
        description="How many, ONLY if the customer stated a number.",
    )
    unit: Optional[str] = None


class Order(BaseModel):
    customer_name: Optional[str] = None
    delivery_date: Optional[str] = Field(
        default=None, description="ISO date (YYYY-MM-DD), only if stated."
    )
    items: list[OrderItem] = Field(default_factory=list)
    notes: Optional[str] = None


class ValidationResult(BaseModel):
    """What :func:`validate_order` decided.

    ``missing_fields`` lists the field paths still to be supplied, e.g.
    ``items[0].quantity``; the clarification reply is graded against exactly
    this list. ``problems`` lists the rules the order broke, in words a
    customer can act on.
    """

    ok: bool
    order: Optional[Order] = None
    missing_fields: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class StoredOrder(BaseModel):
    order_id: str


class Draft(BaseModel):
    """The reply text a model wrote. Only the words come from the model."""

    subject: str
    body: str


class AssistantReply(BaseModel):
    """What the agent answers over HTTP."""

    status: str = Field(description="stored, needs_details or not_an_order.")
    order_id: str = ""
    missing: list[str] = Field(default_factory=list)
    subject: str
    body: str


ORDER_BOOK: list[dict] = []


@pythontool
def validate_order(order: Optional[Order] = None) -> ValidationResult:
    """Check an extracted order against the catalogue and the bakery's rules.

    Deterministic, and the only thing that decides whether an order is
    complete. Returns the exact field paths still missing, so a case can assert
    on them and the clarification reply can be graded without a judge.
    """
    if order is None:
        return ValidationResult(ok=False, missing_fields=["order"])

    missing: list[str] = []
    problems: list[str] = []

    if not order.items:
        missing.append("items")

    for index, item in enumerate(order.items):
        product = CATALOGUE.get(item.product or "")
        if product is None:
            missing.append(f"items[{index}].product")
            problems.append(
                f"We do not have '{item.product or 'that'}' on the list. "
                f"We bake:\n{catalogue_text()}"
            )
            continue
        item.unit = product.unit

        if item.quantity is None:
            missing.append(f"items[{index}].quantity")
        elif item.quantity < product.minimum_quantity:
            missing.append(f"items[{index}].quantity")
            problems.append(
                f"We bake {product_name(item.product)} in batches of at least "
                f"{product.minimum_quantity}; you asked for {item.quantity:g}."
            )
        elif item.quantity > product.maximum_quantity:
            # The control a persuasive email cannot argue with, because it is
            # a rule in code rather than a sentence in a prompt.
            missing.append(f"items[{index}].quantity")
            problems.append(
                f"We can take at most {product.maximum_quantity} "
                f"{product_name(item.product)}(s) by email; {item.quantity:g} "
                "needs to be arranged with us directly."
            )

    delivery = _parse_date(order.delivery_date)
    notice = max(
        (
            CATALOGUE[i.product].lead_time_days
            for i in order.items
            if i.product in CATALOGUE
        ),
        default=0,
    )
    if delivery is None:
        missing.append("delivery_date")
    elif delivery < TODAY + timedelta(days=notice):
        missing.append("delivery_date")
        problems.append(
            f"We need {notice} day(s) notice, so the earliest we can manage is "
            f"{(TODAY + timedelta(days=notice)).isoformat()}."
        )

    if not order.customer_name:
        missing.append("customer_name")

    # One incomplete item makes the whole order incomplete: storing the good
    # half would leave the customer with an order they never placed.
    return ValidationResult(
        ok=not missing,
        order=order if not missing else None,
        missing_fields=sorted(set(missing)),
        problems=problems,
    )


def _parse_date(value: Optional[str]) -> Optional[date]:
    """The date formats customers actually write, and nothing else."""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime((value or "").strip(), fmt).date()
        except ValueError:
            continue
    return None


@pythontool
def store_order(order: Optional[Order] = None) -> StoredOrder:
    """Write a validated order into the order book."""
    order_id = f"ord-{len(ORDER_BOOK) + 1:04d}"
    ORDER_BOOK.append({"order_id": order_id, **(order or Order()).model_dump()})
    return StoredOrder(order_id=order_id)


@pythontool
def compose_reply(
    intent: str = "",
    draft: Optional[Draft] = None,
    validation: Optional[ValidationResult] = None,
    order_id: Optional[str] = None,
) -> AssistantReply:
    """Assemble the answer from what Python decided and what a model wrote.

    Every branch of the workflow ends here. `validation` and `order_id` are
    absent on the branches that never reached them, and that absence is what
    tells the three outcomes apart — so the outcome is a field with three known
    values rather than something a reader has to infer from the wording.
    """
    draft = draft or Draft(subject="", body="")
    if intent != "order":
        status = "not_an_order"
    elif order_id:
        status = "stored"
    else:
        status = "needs_details"
    return AssistantReply(
        status=status,
        order_id=order_id or "",
        missing=list(validation.missing_fields) if validation else [],
        subject=draft.subject,
        body=draft.body,
    )
