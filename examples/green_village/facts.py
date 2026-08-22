"""The Green Village corpus — one source of truth for notebook, docs and suite.

Green Village is fictional, and that is exactly why it makes a good evaluation
corpus: no model can answer "how deep is Lake Miller?" from pretraining, so a
correct answer is *proof that retrieval worked* rather than a lucky prior. Most
of the facts are numeric, so most of the suite grades by exact match and never
pays for a judge.
"""

FACTS = [
    "President of Green Village is Thomas Cook (born 12.04.1994).",
    "Green Village has 104 residents.",
    "Green Village was founded on 03.09.1887 by shepherd Elias Thornbury.",
    "The tallest building in Green Village is the Old Grain Tower at 23 metres.",
    "Green Village's official flower is the marsh marigold.",
    "The village bakery, run by Greta Lindqvist, sells exactly 340 loaves every week.",
    "Green Village has one school with 14 pupils and 2 teachers.",
    "The annual Turnip Festival takes place every year on the third "
    "Saturday of October.",
    "Green Village's fire brigade consists of 7 volunteers and one "
    "dalmatian named Pepper.",
    "The village pond, Lake Miller, is 1.2 metres deep at its deepest point.",
    "Green Village's oldest resident is Agnes Whitlow (born 02.06.1929).",
    "The village has 3 streets: Main Road, Willow Lane, and Cobbler's Path.",
    "The local church bell weighs 412 kilograms and was cast in 1901.",
    "Green Village produces 8 tons of honey per year from its 26 beehives.",
    "The village library owns 1,847 books and is open on Tuesdays and Fridays.",
    "The speed limit everywhere in Green Village is 30 km/h.",
    "Green Village's only pub, The Rusty Anchor, has been operating since 1923.",
]

TOPICS = [
    "people",
    "people",
    "history",
    "buildings",
    "nature",
    "business",
    "school",
    "events",
    "safety",
    "nature",
    "people",
    "streets",
    "buildings",
    "business",
    "culture",
    "traffic",
    "business",
]

COLLECTION = "green_village"

#: Where the checked-in index lives, relative to this file.
INDEX_FILENAME = "green_village.sqlite"

#: A local model, so building and querying the index costs nothing and needs no
#: API key. That is what lets the retrieval half of the suite run on every pull
#: request rather than only when someone has a key configured.
EMBEDDING_MODEL = "fastembed/BAAI/bge-small-en-v1.5"


def source_ids() -> list[str]:
    """``fact-00`` … one per fact. This is what the suite asserts retrieval on."""
    return [f"fact-{i:02d}" for i in range(len(FACTS))]


def fact_by_source_id() -> dict[str, str]:
    return dict(zip(source_ids(), FACTS))


def corpus_fingerprint() -> str:
    """A hash of the corpus, stored beside the index.

    A stale index grades new questions against an old corpus and passes, which
    is the kind of failure nobody notices. Comparing this makes it loud.
    """
    import hashlib

    digest = hashlib.sha256()
    for fact in FACTS:
        digest.update(fact.encode("utf-8"))
    return digest.hexdigest()[:12]
