"""The bakery's own world: a workspace, a validator, an order book, an outbox.

Two things here are worth copying into a real workflow.

**The model extracts, Python decides.** ``validate_order`` is ordinary code over
a Pydantic ``Order``. Because the decision is deterministic, the clarification
branch is gradeable without a judge (the reply must name the fields the
validator reported), a case can assert the exact ``missing_fields`` list, and a
model upgrade cannot silently change what counts as a complete order. Ask the
LLM to judge completeness instead and every one of those properties disappears.

**The world is a workspace, not a global.** Every evaluation case gets a fresh
:class:`BakeryWorkspace` — its own SQLite order book, its own outbox directory
and a pinned clock. Nothing about running a workflow in a suite otherwise stops
it writing to the real order book: the workflow does not know it is being
evaluated. Getting that wrong once, against a production tool, turns the eval
suite into the incident.
"""

import json
import shutil
import sqlite3
import tempfile
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from kavalai import pythontool

from catalogue import CATALOGUE, catalogue_text, resolve_product
from models import Order, SentMail, StoredOrder, ValidationResult

#: The date the bakery believes it is. Pinned rather than read from the wall
#: clock so "delivery at least 2 days out" does not silently expire every
#: fixture the moment the dates fall into the past.
TODAY = date(2026, 9, 1)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_name TEXT,
    delivery_date TEXT,
    items TEXT,
    notes TEXT
);
"""


@dataclass
class BakeryWorkspace:
    """One isolated copy of the bakery's world."""

    root: Path
    today: date = TODAY
    _owned: bool = field(default=True, repr=False)

    @property
    def db_path(self) -> Path:
        return self.root / "orders.sqlite"

    @property
    def outbox(self) -> Path:
        path = self.root / "outbox"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.executescript(_SCHEMA)
        return connection

    def orders(self) -> list[dict]:
        """Every order stored so far — the side-effect assertion reads this."""
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM orders ORDER BY id").fetchall()
        return [
            {**dict(row), "items": json.loads(row["items"] or "[]")} for row in rows
        ]

    def sent_mail(self) -> list[Path]:
        return sorted(self.outbox.glob("*.eml"))

    def cleanup(self) -> None:
        if self._owned:
            shutil.rmtree(self.root, ignore_errors=True)


#: The workspace the tools are currently writing to. A context variable rather
#: than a module global: cases run concurrently, and each asyncio task gets its
#: own copy of the context, so two cases cannot see each other's orders.
_WORKSPACE: ContextVar[Optional[BakeryWorkspace]] = ContextVar(
    "bakery_workspace", default=None
)


def new_workspace() -> BakeryWorkspace:
    """Create a fresh workspace and make the tools use it.

    This is the suite's ``sandbox:`` hook. It runs before every case *and every
    repeat* — reset between repeats too, or the second run sees the first one's
    rows and "exactly one order stored" starts failing for the wrong reason.
    """
    workspace = BakeryWorkspace(Path(tempfile.mkdtemp(prefix="bakery-")))
    _WORKSPACE.set(workspace)
    return workspace


def use_workspace(workspace: BakeryWorkspace) -> BakeryWorkspace:
    """Point the tools at an existing workspace (a real deployment does this)."""
    _WORKSPACE.set(workspace)
    return workspace


def current_workspace() -> BakeryWorkspace:
    workspace = _WORKSPACE.get()
    if workspace is None:
        raise RuntimeError(
            "No bakery workspace is active. Call new_workspace() first — the "
            "tools deliberately refuse to guess where to write."
        )
    return workspace


# --------------------------------------------------------------------- tools
@pythontool
def validate_order(order: Optional[Order] = None) -> ValidationResult:
    """Check an extracted order against the catalogue and the bakery's rules.

    Deterministic, and the only thing that decides whether an order is
    complete. Returns the exact field paths that are missing, so the
    clarification reply can be graded against them.
    """
    workspace = _WORKSPACE.get()
    today = workspace.today if workspace else TODAY
    missing: list[str] = []
    problems: list[str] = []

    if order is None:
        return ValidationResult(ok=False, missing_fields=["order"], problems=[])

    if not order.items:
        missing.append("items")

    for index, item in enumerate(order.items):
        product = resolve_product(item.product or "")
        if product is not None:
            # Store what the bakery calls it, not what the customer typed, so
            # the order book is queryable and an assertion about a stored row
            # is not really an assertion about the customer's spelling.
            item.product = product
            item.unit = CATALOGUE[product].unit
        if product is None:
            missing.append(f"items[{index}].product")
            problems.append(
                f"We do not have '{item.product or 'that'}' on the list. "
                f"We bake:\n{catalogue_text()}"
            )
            continue
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
            # The control that a persuasive email cannot argue with. An order
            # this size may be perfectly genuine — it still goes to a person,
            # and it is a rule rather than an instruction precisely so that
            # nothing written in an email can move it.
            missing.append(f"items[{index}].quantity")
            problems.append(
                f"We can take at most {maximum} {product}(s) by email; "
                f"{item.quantity:g} needs to be arranged with us directly."
            )

    delivery = _parse_date(order.delivery_date)
    if delivery is None:
        missing.append("delivery_date")
    else:
        needed = max(
            (
                CATALOGUE[p].lead_time_days
                for p in (resolve_product(i.product or "") for i in order.items)
                if p
            ),
            default=0,
        )
        if delivery < today + timedelta(days=needed):
            missing.append("delivery_date")
            problems.append(
                f"We need {needed} day(s) notice, so the earliest we can manage "
                f"is {(today + timedelta(days=needed)).isoformat()}."
            )

    if not order.customer_name:
        missing.append("customer_name")

    # One incomplete item makes the whole order incomplete. Storing the good
    # half would leave the customer with an order they never placed.
    return ValidationResult(
        ok=not missing,
        order=order if not missing else None,
        missing_fields=sorted(set(missing)),
        problems=problems,
    )


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


@pythontool
def store_order(order: Optional[Order] = None) -> StoredOrder:
    """Write a validated order to the bakery's order book."""
    workspace = current_workspace()
    order = order or Order()
    with workspace.connect() as connection:
        # Sequential, not a UUID. An order book numbers its orders — and the
        # id ends up in the acknowledgement prompt, so a random one would make
        # every prompt unique and no recorded fixture would ever match it
        # again. Anything that reaches a prompt has to be deterministic if you
        # want the suite to replay.
        count = connection.execute("SELECT count(*) FROM orders").fetchone()[0]
        order_id = f"ord-{count + 1:04d}"
        connection.execute(
            "INSERT INTO orders (id, customer_name, delivery_date, items, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                order_id,
                order.customer_name,
                order.delivery_date,
                json.dumps([item.model_dump() for item in order.items]),
                order.notes,
            ),
        )
    return StoredOrder(order_id=order_id)


@pythontool
def send_reply(to: str = "", subject: str = "", body: str = "") -> SentMail:
    """Write the reply into the outbox as a real ``.eml`` file.

    No mail service and no new dependency: the outbox is a directory, so "how
    many mails were sent" is ``len(workspace.sent_mail())`` — a genuine,
    deterministic side effect to assert on rather than a mock to trust.
    """
    workspace = current_workspace()
    message_id = f"<{uuid.uuid4().hex}@lindqvist.example>"
    message = EmailMessage()
    message["From"] = "orders@lindqvist.example"
    message["To"] = to
    message["Subject"] = subject
    message["Message-ID"] = message_id
    message["Date"] = f"{workspace.today.isoformat()} 09:00:00 +0000"
    message.set_content(body)

    path = workspace.outbox / f"{message_id.strip('<>').split('@')[0]}.eml"
    path.write_bytes(bytes(message))
    return SentMail(
        message_id=message_id,
        to=to,
        subject=subject,
        body=body,
        outbox_path=str(path),
    )


@pythontool
def list_orders() -> dict:
    """Every order in the book. Handy from a notebook; not used by the workflow."""
    return {"orders": current_workspace().orders()}
