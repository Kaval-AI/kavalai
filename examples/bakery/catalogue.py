"""What Lindqvist Bakery Workshop sells, and the rules an order must satisfy.

Ordinary data in ordinary Python. Everything here is a *business rule*, which
is why none of it lives in a prompt: a model upgrade must not be able to change
what counts as a complete order.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    name: str
    unit: str
    #: Below this we cannot bake economically, so the order is incomplete.
    minimum_quantity: int
    #: Days of notice needed before the delivery date.
    lead_time_days: int
    #: Above this a human has to be involved, whoever is asking. This is a
    #: deterministic control, and that is the point: it is the one thing in
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
#: and explicit: an alias table you can read is better than a fuzzy matcher you
#: cannot predict, and an unrecognised product must become a question to the
#: customer rather than a guess.
ALIASES = {
    "sourdough": "sourdough loaf",
    "sourdough bread": "sourdough loaf",
    "rye": "rye loaf",
    "rye bread": "rye loaf",
    "kringel": "kringle",
    "kringles": "kringle",
    "cinnamon buns": "cinnamon bun",
    "buns": "cinnamon bun",
    "cake": "birthday cake",
    "birthday cakes": "birthday cake",
    "loaves": "sourdough loaf",
    # Green Village is in Estonia and half the customers write in Estonian, so
    # these are product names, not translations. A word the bakery genuinely
    # sells belongs in the catalogue; a misspelling does not — that stays a
    # question to the customer.
    "rukkileib": "rye loaf",
    "rukkileiba": "rye loaf",
    "juuretisega leib": "sourdough loaf",
    "saiake": "cinnamon bun",
    "kaneelirull": "cinnamon bun",
    "kringel": "kringle",
    "sünnipäevatort": "birthday cake",
    "tort": "birthday cake",
}


def resolve_product(name: str) -> str | None:
    """Catalogue name for what the customer wrote, or ``None`` if unknown.

    Deliberately narrow: an exact name, a listed alias, or the same with a
    trailing plural removed. Anything else is unknown, and an unknown product
    becomes a question to the customer rather than a guess — which is the
    whole point of having the rule in code instead of in a prompt.
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
    # "sourdough loaves" -> "sourdough loaf": customers pluralise the noun, and
    # the catalogue is written in the singular.
    for plural, singular in (("loaves", "loaf"), ("es", ""), ("s", "")):
        if key.endswith(plural):
            variants.append(key[: -len(plural)] + singular)
    return [v.strip() for v in variants if v.strip()]


def catalogue_text() -> str:
    """The product list, for a reply that has to offer options."""
    return "\n".join(
        f"- {p.name} (per {p.unit}, {p.minimum_quantity}-{p.maximum_quantity}, "
        f"{p.lead_time_days} day(s) notice)"
        for p in CATALOGUE.values()
    )
