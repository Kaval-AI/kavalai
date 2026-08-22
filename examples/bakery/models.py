"""The shapes the workflow passes around.

``OrderItem`` fields are all optional on purpose: the extraction step must be
able to say *"the customer did not state a quantity"* rather than inventing
one. A required field would force the model to guess, and a guessed quantity is
the exact failure this example exists to catch.
"""

from typing import Optional

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    product: Optional[str] = Field(
        default=None, description="The product as the customer wrote it."
    )
    quantity: Optional[float] = Field(
        default=None,
        description="How many, ONLY if the customer stated a number. Never guess.",
    )
    unit: Optional[str] = Field(default=None, description="loaf, piece, cake …")


class Order(BaseModel):
    customer_name: Optional[str] = None
    delivery_date: Optional[str] = Field(
        default=None, description="ISO date (YYYY-MM-DD), only if stated."
    )
    items: list[OrderItem] = Field(default_factory=list)
    notes: Optional[str] = None


class EmailParse(BaseModel):
    intent: str = Field(description="One of: order, question, complaint, other.")
    order: Optional[Order] = None


class ValidationResult(BaseModel):
    ok: bool
    order: Optional[Order] = None
    #: Field paths the customer still has to supply, e.g.
    #: ``items[0].quantity``. The clarification reply is graded against exactly
    #: this list, which is what makes that branch checkable without a judge.
    missing_fields: list[str] = Field(default_factory=list)
    #: Rules the order broke, in words a customer can act on.
    problems: list[str] = Field(default_factory=list)


class Reply(BaseModel):
    subject: str
    body: str


class SentMail(BaseModel):
    message_id: str
    to: str
    subject: str
    body: str
    outbox_path: str


class StoredOrder(BaseModel):
    order_id: str
