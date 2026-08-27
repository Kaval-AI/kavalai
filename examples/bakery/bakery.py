"""Lindqvist Bakery Workshop — everything the email assistant shares.

The workflow itself is :file:`assistant.yaml`; this module is the Python half
it reaches into. It holds four things:

* the **catalogue** — what the bakery sells and the rules an order must meet;
* the **models** the workflow passes around;
* the **order book**, in two interchangeable implementations;
* the three ``@pythontool`` functions ``assistant.yaml`` calls by name.

The design decision worth stealing is **the model extracts, Python decides**.
The ``parse`` node in the YAML pulls structured fields out of prose and does
nothing else; :func:`validate_order` below is ordinary code and is the only
thing that says whether an order is complete. That is what makes the whole
suite gradeable without a judge — and what stops a model upgrade quietly
changing the definition of a valid order.

Nothing here knows which database it is running against. The server module
picks an :class:`OrderBook` and calls :func:`use_order_book`; see
:file:`bakery_in_memory.py` and :file:`bakery_real_db.py`.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from kavalai import pythontool
from kavalai.agent_service import AgentService
from kavalai.server import create_agent_router
from kavalai.workflow import WorkflowEngine

#: The workflow definition both servers load. Keeping the graph in YAML is the
#: point of this example: the two Python modules differ only in which database
#: they hand it, never in what the assistant does.
WORKFLOW_PATH = Path(__file__).with_name("assistant.yaml")

#: The date the bakery believes it is. Pinned rather than read from the wall
#: clock so that "we need two days' notice" does not silently invalidate every
#: evaluation case the moment its delivery dates fall into the past.
TODAY = date(2026, 9, 1)


# --------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class Product:
    """One thing the bakery sells, and the rules that come with it."""

    name: str
    unit: str
    #: Below this we cannot bake economically, so the order is incomplete.
    minimum_quantity: int
    #: Days of notice needed before the delivery date.
    lead_time_days: int
    #: Above this a human has to be involved, whoever is asking. A
    #: deterministic control, and that is the point: it is the one rule in
    #: this workflow that a persuasive email cannot talk its way past.
    maximum_quantity: int


CATALOGUE: dict[str, Product] = {
    "sourdough loaf": Product("sourdough loaf", "loaf", 1, 1, 60),
    "rye loaf": Product("rye loaf", "loaf", 1, 1, 60),
    "kringle": Product("kringle", "piece", 2, 2, 40),
    "cinnamon bun": Product("cinnamon bun", "piece", 6, 1, 200),
    "birthday cake": Product("birthday cake", "cake", 1, 3, 5),
}

#: Words customers actually use, mapped to catalogue names. Deliberately small
#: and explicit: an alias table you can read beats a fuzzy matcher you cannot
#: predict, and an unrecognised product must become a question to the customer
#: rather than a guess.
ALIASES = {
    "sourdough": "sourdough loaf",
    "sourdough bread": "sourdough loaf",
    "rye": "rye loaf",
    "rye bread": "rye loaf",
    "cinnamon buns": "cinnamon bun",
    "buns": "cinnamon bun",
    "cake": "birthday cake",
    "loaves": "sourdough loaf",
    # Green Village is in Estonia and half the customers write in Estonian, so
    # these are product names rather than translations. A word the bakery
    # genuinely sells belongs here; a misspelling does not — that stays a
    # question to the customer.
    # Estonian inflects, so the partitive forms a customer writes when they
    # order "8 of them" are listed alongside the dictionary form.
    "kringel": "kringle",
    "kringlit": "kringle",
    "kringlid": "kringle",
    "rukkileib": "rye loaf",
    "rukkileiba": "rye loaf",
    "juuretisega leib": "sourdough loaf",
    "juuretisega leiba": "sourdough loaf",
    "kaneelirull": "cinnamon bun",
    "kaneelirulli": "cinnamon bun",
    "saiake": "cinnamon bun",
    "saiakest": "cinnamon bun",
    "sünnipäevatort": "birthday cake",
    "sünnipäevatorti": "birthday cake",
    "tort": "birthday cake",
    "torti": "birthday cake",
}


def resolve_product(name: str) -> Optional[str]:
    """Catalogue name for what the customer wrote, or ``None`` if unknown.

    Deliberately narrow: an exact name, a listed alias, or the same with a
    trailing plural removed. Anything else is unknown, and an unknown product
    becomes a question to the customer rather than a guess — which is the whole
    reason this rule lives in code instead of in a prompt.
    """
    if not name:
        return None
    for candidate in _variants(name):
        if candidate in CATALOGUE:
            return candidate
        if candidate in ALIASES:
            return ALIASES[candidate]
    return None


def _variants(name: str) -> list[str]:
    key = " ".join(name.strip().casefold().split())
    variants = [key]
    # "sourdough loaves" -> "sourdough loaf": customers pluralise the noun and
    # the catalogue is written in the singular.
    for plural, singular in (("loaves", "loaf"), ("es", ""), ("s", "")):
        if key.endswith(plural):
            variants.append(key[: -len(plural)] + singular)
    return [variant.strip() for variant in variants if variant.strip()]


def catalogue_text() -> str:
    """The product list, for a reply that has to offer options."""
    return "\n".join(
        f"- {p.name} (per {p.unit}, {p.minimum_quantity}-{p.maximum_quantity}, "
        f"{p.lead_time_days} day(s) notice)"
        for p in CATALOGUE.values()
    )


# ------------------------------------------------------------------ models
class OrderItem(BaseModel):
    """One line of an order.

    Every field is optional on purpose: extraction must be able to say *"the
    customer did not state a quantity"* rather than inventing one. A required
    field would force the model to guess, and a guessed quantity is the exact
    failure this example exists to catch.
    """

    product: Optional[str] = Field(
        default=None, description="The product as the customer wrote it."
    )
    quantity: Optional[float] = Field(
        default=None,
        description="How many, ONLY if the customer stated a number. Never guess.",
    )
    unit: Optional[str] = Field(default=None, description="loaf, piece, cake …")


class Order(BaseModel):
    """What the customer is trying to buy, as extracted from their email."""

    customer_name: Optional[str] = None
    delivery_date: Optional[str] = Field(
        default=None, description="ISO date (YYYY-MM-DD), only if stated."
    )
    items: list[OrderItem] = Field(default_factory=list)
    notes: Optional[str] = None


class ValidationResult(BaseModel):
    """The verdict of :func:`validate_order` — deterministic, and final."""

    ok: bool
    order: Optional[Order] = None
    #: Field paths the customer still has to supply, e.g.
    #: ``items[0].quantity``. The clarification reply is graded against exactly
    #: this list, which is what makes that branch checkable without a judge.
    missing_fields: list[str] = Field(default_factory=list)
    #: Rules the order broke, in words a customer can act on.
    problems: list[str] = Field(default_factory=list)


class StoredOrder(BaseModel):
    """What the order book hands back once an order is written down."""

    order_id: str


class Draft(BaseModel):
    """The reply text a model wrote. Only the words come from the model."""

    subject: str
    body: str


class AssistantReply(BaseModel):
    """What the agent answers over HTTP.

    ``status``, ``order_id`` and ``missing`` are assembled by
    :func:`compose_reply` from what Python decided, never by a model. That is
    what lets :file:`eval_cases.yaml` grade most of this agent with literal
    comparisons instead of a judge.
    """

    status: str = Field(
        description="stored, needs_details or not_an_order.",
    )
    order_id: str = ""
    missing: list[str] = Field(default_factory=list)
    subject: str
    body: str


# -------------------------------------------------------------- order book
class OrderBook(ABC):
    """Where confirmed orders are written down.

    Two implementations ship with this example — a list in memory and a table
    in Postgres — and the workflow cannot tell them apart. Swapping one for the
    other is the only difference between :file:`bakery_in_memory.py` and
    :file:`bakery_real_db.py`.
    """

    @abstractmethod
    async def store(self, order: Order) -> str:
        """Write one order down and return its reference."""

    @abstractmethod
    async def orders(self) -> list[dict]:
        """Every order stored so far, oldest first."""


class InMemoryOrderBook(OrderBook):
    """An order book that lives for as long as the process does.

    Right for a demo and for tests; wrong for anything a customer relies on,
    which is what :class:`PostgresOrderBook` is for.
    """

    def __init__(self):
        self._orders: list[dict] = []
        self._lock = asyncio.Lock()

    async def store(self, order: Order) -> str:
        async with self._lock:
            order_id = f"ord-{len(self._orders) + 1:04d}"
            self._orders.append({"order_id": order_id, **order.model_dump()})
        return order_id

    async def orders(self) -> list[dict]:
        return list(self._orders)


class PostgresOrderBook(OrderBook):
    """The order book as a table, sharing the agent database's connections.

    The table is not part of Kaval.AI's Alembic sets — it belongs to the
    bakery, not to the framework — so this class owns its own DDL and creates
    it on :meth:`create_table`. Note that the schema is written into the SQL
    explicitly: raw SQL bypasses SQLAlchemy's ``schema_translate_map``, so a
    statement that relies on it would quietly go to ``public``.
    """

    TABLE = "bakery_orders"

    def __init__(self, session_maker: async_sessionmaker, schema: str = "public"):
        self.session_maker = session_maker
        self.schema = schema

    @property
    def qualified_table(self) -> str:
        return f"{self.schema}.{self.TABLE}"

    async def create_table(self) -> None:
        """Create the order book if it is not there yet."""
        async with self.session_maker() as session:
            await session.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self.qualified_table} ("
                    "  id BIGSERIAL PRIMARY KEY,"
                    "  customer_name TEXT,"
                    "  delivery_date TEXT,"
                    "  items JSONB NOT NULL,"
                    "  notes TEXT,"
                    "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                    ")"
                )
            )
            await session.commit()

    async def store(self, order: Order) -> str:
        # ``RETURNING id`` rather than counting the rows first: the database
        # hands out the number, so two orders arriving at once cannot be given
        # the same reference.
        async with self.session_maker() as session:
            row = await session.execute(
                text(
                    f"INSERT INTO {self.qualified_table} "
                    "(customer_name, delivery_date, items, notes) "
                    "VALUES (:customer_name, :delivery_date, :items, :notes) "
                    "RETURNING id"
                ),
                {
                    "customer_name": order.customer_name,
                    "delivery_date": order.delivery_date,
                    "items": json.dumps([i.model_dump() for i in order.items]),
                    "notes": order.notes,
                },
            )
            await session.commit()
        return f"ord-{row.scalar_one():04d}"

    async def orders(self) -> list[dict]:
        async with self.session_maker() as session:
            rows = await session.execute(
                text(
                    "SELECT id, customer_name, delivery_date, items, notes "
                    f"FROM {self.qualified_table} ORDER BY id"
                )
            )
            # The same shape InMemoryOrderBook returns: the two are meant to be
            # interchangeable, so the row number stays behind the reference.
            return [
                {
                    "order_id": f"ord-{row.id:04d}",
                    "customer_name": row.customer_name,
                    "delivery_date": row.delivery_date,
                    "items": row.items,
                    "notes": row.notes,
                }
                for row in rows
            ]


#: The order book the tools are currently writing to. A module-level binding
#: rather than a tool argument, because ``assistant.yaml`` names the tool and
#: not its dependencies — the server decides where orders go, and it decides it
#: once, at startup.
_ORDER_BOOK: Optional[OrderBook] = None


def use_order_book(book: OrderBook) -> OrderBook:
    """Point :func:`store_order` at ``book``. Called once, by the server."""
    global _ORDER_BOOK
    _ORDER_BOOK = book
    return book


def current_order_book() -> OrderBook:
    """The active order book, or a clear error rather than a guess."""
    if _ORDER_BOOK is None:
        raise RuntimeError(
            "No order book is configured. Call use_order_book() before serving "
            "the workflow — the tools deliberately refuse to guess where to "
            "write an order."
        )
    return _ORDER_BOOK


# ------------------------------------------------------------------- tools
@pythontool
def validate_order(order: Optional[Order] = None) -> ValidationResult:
    """Check an extracted order against the catalogue and the bakery's rules.

    Deterministic, and the only thing that decides whether an order is
    complete. Returns the exact field paths that are still missing, so the
    clarification reply can be graded against them and a case can assert on
    them directly.

    Args:
        order: What the parse step extracted, or ``None`` if it found nothing.

    Returns:
        A :class:`ValidationResult`. ``ok`` is true only when nothing is
        missing: one incomplete item makes the whole order incomplete, because
        storing the good half would leave the customer with an order they never
        placed.
    """
    if order is None:
        return ValidationResult(ok=False, missing_fields=["order"])

    missing: list[str] = []
    problems: list[str] = []

    if not order.items:
        missing.append("items")

    for index, item in enumerate(order.items):
        product = resolve_product(item.product or "")
        if product is None:
            missing.append(f"items[{index}].product")
            problems.append(
                f"We do not have '{item.product or 'that'}' on the list. "
                f"We bake:\n{catalogue_text()}"
            )
            continue
        # Store what the bakery calls it, not what the customer typed, so the
        # order book stays queryable and an assertion about a stored row is not
        # really an assertion about the customer's spelling.
        item.product = product
        item.unit = CATALOGUE[product].unit

        if item.quantity is None:
            missing.append(f"items[{index}].quantity")
            continue
        minimum = CATALOGUE[product].minimum_quantity
        maximum = CATALOGUE[product].maximum_quantity
        if item.quantity < minimum:
            missing.append(f"items[{index}].quantity")
            problems.append(
                f"We bake {product} in batches of at least {minimum}; "
                f"you asked for {item.quantity:g}."
            )
        elif item.quantity > maximum:
            missing.append(f"items[{index}].quantity")
            problems.append(
                f"We can take at most {maximum} {product}(s) by email; "
                f"{item.quantity:g} needs to be arranged with us directly."
            )

    delivery = _parse_date(order.delivery_date)
    if delivery is None:
        missing.append("delivery_date")
    else:
        notice = max(
            (
                CATALOGUE[name].lead_time_days
                for name in (resolve_product(i.product or "") for i in order.items)
                if name
            ),
            default=0,
        )
        earliest = TODAY + timedelta(days=notice)
        if delivery < earliest:
            missing.append("delivery_date")
            problems.append(
                f"We need {notice} day(s) notice, so the earliest we can manage "
                f"is {earliest.isoformat()}."
            )

    if not order.customer_name:
        missing.append("customer_name")

    return ValidationResult(
        ok=not missing,
        order=order if not missing else None,
        missing_fields=sorted(set(missing)),
        problems=problems,
    )


def _parse_date(value: Optional[str]) -> Optional[date]:
    """The date formats customers actually write, and nothing else."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


@pythontool
async def store_order(order: Optional[Order] = None) -> StoredOrder:
    """Write a validated order into whichever order book is configured.

    Args:
        order: The order :func:`validate_order` passed. Reached only from the
            branch that has one, so ``None`` here is a wiring mistake.

    Returns:
        The reference the customer is told to quote.
    """
    if order is None:
        raise ValueError("store_order was reached without a validated order.")
    return StoredOrder(order_id=await current_order_book().store(order))


@pythontool
def compose_reply(
    intent: str = "",
    draft: Optional[Draft] = None,
    validation: Optional[ValidationResult] = None,
    order_id: Optional[str] = None,
) -> AssistantReply:
    """Assemble the agent's answer from what Python decided and what a model wrote.

    Every branch of the workflow ends here, which is why the outcome is one
    field with three known values rather than something a reader has to infer
    from the wording. ``validation`` and ``order_id`` are absent on the
    branches that never reached them, and that absence is exactly what
    distinguishes the three outcomes.

    Args:
        intent: What the parse step classified the email as.
        draft: The subject and body a model wrote for this branch.
        validation: The validator's verdict, on the order branches.
        order_id: The order book's reference, when an order was stored.

    Returns:
        The :class:`AssistantReply` the HTTP caller receives.
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


# ------------------------------------------------------------------ server
def build_engine(agent_service: AgentService) -> WorkflowEngine:
    """Load :file:`assistant.yaml` and give it somewhere to record its runs."""
    return WorkflowEngine.from_yaml_path(
        str(WORKFLOW_PATH), agent_service=agent_service
    )


def create_app(
    engine: WorkflowEngine,
    session_maker: async_sessionmaker,
    lifespan=None,
) -> FastAPI:
    """Serve ``engine`` over ``POST /run_agent`` and ``POST /stream_agent``.

    Args:
        engine: The workflow to serve.
        session_maker: Sessions for the agent database the runs are recorded in.
        lifespan: Optional FastAPI lifespan, for whatever the chosen database
            has to do before the first request.
    """
    app = FastAPI(
        title=engine.graph.name,
        description=engine.graph.description,
        version=engine.graph.version,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.include_router(create_agent_router(engine, session_provider=session_maker))
    return app
