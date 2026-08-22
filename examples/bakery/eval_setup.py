"""Everything the bakery suite needs registered before it can run.

Two jobs: point the tools at a fresh, isolated world for each case, and add the
evaluators that know what "the order book is correct" means. Both belong in the
example rather than the library — a side-effect assertion is domain knowledge,
and the plugin point exists so a customer never has to fork the evaluator
module to express one.
"""

from kavalai.eval import Evaluator, Score, evaluator

# Imported for their side effect: the `@pythontool` functions have to be
# importable under the names the workflow's `python_functions` block gives.
import tools  # noqa: F401
from tools import new_workspace  # noqa: F401  (the suite's `sandbox:` hook)

#: Field paths the validator reports, and the plain words a customer-facing
#: reply may use for each. The clarification branch is graded against these,
#: which is only possible because a deterministic validator — not the model —
#: decided what was missing.
#:
#: Be honest about the limit: this is an English keyword list, so it cannot
#: grade a reply written in another language. The multilingual slice uses a
#: judge for exactly that reason, and this evaluator is not applied there.
FIELD_WORDS = {
    "quantity": ["how many", "quantity", "number of", "how much"],
    "delivery_date": ["date", "when", "day"],
    "customer_name": ["name", "who"],
    "product": [
        "which",
        "what kind",
        "what would you like",
        "what you would like",
        "product",
        "item",
        "we bake",
        "we make",
        "options",
    ],
    "items": ["what", "which"],
}


def _words_for(field_path: str) -> list[str]:
    for key, words in FIELD_WORDS.items():
        if field_path.endswith(key) or field_path == key:
            return words
    return []


def _expected(case) -> dict:
    return case.expected if isinstance(case.expected, dict) else {}


@evaluator("orders_stored", replace=True)
class OrdersStored(Evaluator):
    """The order book holds exactly the number of orders it should.

    The assertion that catches what text metrics cannot: a helpful model that
    resolved "a few loaves" to 3 writes a perfectly polite confirmation of an
    order the customer never placed.
    """

    async def score(self, case, record) -> Score:
        expected = _expected(case).get("orders_after")
        if expected is None:
            return Score(name=self.name, reason="no expectation for this case")
        if record.sandbox is None:
            return Score.boolean(
                self.name, False, reason="no sandbox: the suite has no `sandbox:` hook"
            )
        actual = record.sandbox.orders()
        ok = len(actual) == expected
        return Score.boolean(
            self.name,
            ok,
            reason=None
            if ok
            else f"expected {expected} order(s) in the book, found {len(actual)}: "
            f"{[o['items'] for o in actual]}",
        )


@evaluator("stored_order_matches", replace=True)
class StoredOrderMatches(Evaluator):
    """The stored order matches the spec — product, quantity and date.

    Compares against the *spec the email was written from*, never against
    anything a model produced, which is what keeps the ground truth honest.
    """

    async def score(self, case, record) -> Score:
        expected = _expected(case).get("stored")
        if not expected:
            return Score(name=self.name, reason="no stored order expected")
        if record.sandbox is None:
            return Score.boolean(self.name, False, reason="no sandbox configured")

        orders = record.sandbox.orders()
        if len(orders) != 1:
            return Score.boolean(
                self.name,
                False,
                reason=f"expected exactly 1 order, found {len(orders)}",
            )

        order = orders[0]
        problems = []
        for field in ("customer_name", "delivery_date"):
            if field in expected and order.get(field) != expected[field]:
                problems.append(
                    f"{field} = {order.get(field)!r}, expected {expected[field]!r}"
                )

        wanted_items = expected.get("items") or []
        got = {
            (item.get("product"), float(item.get("quantity") or 0))
            for item in order.get("items", [])
        }
        for item in wanted_items:
            key = (item["product"], float(item["quantity"]))
            if key not in got:
                problems.append(f"missing item {key}; stored {sorted(got)}")
        if len(got) != len(wanted_items):
            problems.append(f"stored {len(got)} item(s), expected {len(wanted_items)}")

        return Score.boolean(
            self.name, not problems, reason="; ".join(problems) or None
        )


@evaluator("exactly_one_email_sent", replace=True)
class ExactlyOneEmailSent(Evaluator):
    """One reply left the outbox — not none, and not two.

    The outbox is a directory of real ``.eml`` files, so this counts files
    rather than trusting a mock.
    """

    async def score(self, case, record) -> Score:
        if record.sandbox is None:
            return Score.boolean(self.name, False, reason="no sandbox configured")
        sent = record.sandbox.sent_mail()
        return Score.boolean(
            self.name,
            len(sent) == 1,
            reason=None if len(sent) == 1 else f"{len(sent)} mail(s) in the outbox",
        )


@evaluator("missing_fields_match", replace=True)
class MissingFieldsMatch(Evaluator):
    """``validate_order`` reported exactly the gaps the spec says are there.

    Asserting on the validator's output rather than the reply's wording:
    deterministic, and it localises a failure to extraction rather than to
    phrasing.
    """

    async def score(self, case, record) -> Score:
        expected = _expected(case).get("missing")
        if not expected:
            return Score(name=self.name, reason="no missing fields expected")
        node = record.trajectory.node("validate")
        if node is None:
            return Score.boolean(self.name, False, reason="the validate node never ran")
        actual = (node.output or {}).get("missing_fields") or []
        missing = [field for field in expected if field not in actual]
        return Score.boolean(
            self.name,
            not missing,
            reason=None
            if not missing
            else f"validator reported {actual}, expected {expected}",
        )


@evaluator("reply_names_missing_fields", replace=True)
class ReplyNamesMissingFields(Evaluator):
    """The clarification reply asks, in plain words, for what is missing.

    Gradeable without a judge only because Python decided what was missing.
    Had the model been asked to judge completeness, this would have been a
    rubric and a bill.
    """

    async def score(self, case, record) -> Score:
        expected = _expected(case).get("missing")
        if not expected:
            return Score(name=self.name, reason="no clarification expected")
        body = record.output_text().casefold()
        unasked = [field for field in expected if not self._asks_for(field, body)]
        return Score.boolean(
            self.name,
            not unasked,
            reason=None if not unasked else f"the reply never asks for {unasked}",
        )

    @staticmethod
    def _asks_for(field_path: str, body: str) -> bool:
        """Did the reply ask for this field, in words a customer would use?

        For a missing *product* there are two honest ways to ask: a question
        word, or simply listing what the bakery sells. A reply that offers the
        catalogue is asking which one they want, and an evaluator that only
        recognised the first phrasing failed a perfectly good answer — which is
        how a gate teaches people to ignore it.
        """
        words = _words_for(field_path)
        if not words:
            return True
        if any(word in body for word in words):
            return True
        if field_path.endswith("product"):
            from catalogue import CATALOGUE

            return sum(name in body for name in CATALOGUE) >= 2
        return False


@evaluator("reply_mentions", replace=True)
class ReplyMentions(Evaluator):
    """The reply states each of the case's required words."""

    async def score(self, case, record) -> Score:
        words = _expected(case).get("reply_mentions") or []
        if not words:
            return Score(name=self.name, reason="nothing required for this case")
        body = record.output_text().casefold()
        missing = [w for w in words if str(w).casefold() not in body]
        return Score.boolean(
            self.name,
            not missing,
            reason=None if not missing else f"never says {missing}",
        )


@evaluator("no_invented_quantity", replace=True)
class NoInventedQuantity(Evaluator):
    """Nothing reached the order book with a quantity nobody supplied.

    A blunt safety net across every slice, not just the incomplete ones: if the
    case says no order should be stored, none may be — whatever the reply says.
    """

    async def score(self, case, record) -> Score:
        expected = _expected(case)
        if expected.get("orders_after") != 0 or record.sandbox is None:
            return Score(name=self.name, passed=None)
        stored = record.sandbox.orders()
        return Score.boolean(
            self.name,
            not stored,
            reason=None
            if not stored
            else "stored an order the customer never completed: "
            f"{[o['items'] for o in stored]}",
        )
