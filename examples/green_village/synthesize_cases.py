"""Turn the corpus into an evaluation dataset.

    uv run --env-file .env python examples/green_village/synthesize_cases.py

Writes ``eval/cases/qa.yaml``. The generated file is checked in, so nobody
needs an API key to *run* the suite — only to regenerate the paraphrase slice.

What is generated and what is not:

===============  ==================================================
``direct``       Hand-authored question, hand-authored answer.
``paraphrase``   Question **phrasing** from a model; the answer is
                 still the hand-authored one. Needs ``--llm``.
``unanswerable`` Hand-authored. Expected behaviour: refusal.
``adversarial``  Hand-authored false premises, hand-checked.
``multi_hop``    Hand-authored; graded on retrieving both sources.
===============  ==================================================

**Read what comes out.** Sample a fifth of any generated slice by hand. The
failure mode is silent: an ambiguous question whose expected answer is
defensible either way becomes a permanently red case that people learn to
ignore, and a suite people ignore is worse than no suite at all.
"""

import argparse
import asyncio
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class _NoAliasDumper(yaml.SafeDumper):
    """Write the dataset out without YAML anchors.

    Two cases sharing an ``expected`` list is an implementation detail of the
    generator; ``&id001`` / ``*id001`` in a file people read in a pull request
    is not. Repeating four characters is cheaper than explaining them.
    """

    def ignore_aliases(self, data):
        return True


from facts import fact_by_source_id
from questions import ADVERSARIAL, ANY_OF, DIRECT, MULTI_HOP, UNANSWERABLE

HERE = Path(__file__).parent
OUTPUT = HERE / "eval" / "cases" / "qa.yaml"

#: Deliberately not the model under test: a model rewriting questions for
#: itself produces questions phrased the way it likes to be asked.
GENERATOR_MODEL = "gemini/gemini-3.6-flash"


class Phrasings(BaseModel):
    questions: list[str] = Field(
        description="Different ways a real person might ask the same thing."
    )


async def paraphrase(fact: str, question: str, count: int) -> list[str]:
    from kavalai import make_client

    client = make_client(GENERATOR_MODEL)
    result: Phrasings = await client.prompt(
        f"A fact: {fact}\n"
        f"A question it answers: {question}\n\n"
        f"Write {count} other ways a real person might ask for the same "
        f"information — vaguer, chattier, or more clipped. Do not answer the "
        f"question. Do not include the answer in the question.",
        Phrasings,
    )
    return result.questions[:count]


def expectation(source_id: str, contains: list[str]) -> dict:
    """What the answer must state, in the two forms the grader understands.

    ``contains`` is every value the answer must state; ``contains_any`` is one
    value written several ways. Keeping them apart is what stops "states both
    numbers" and "states the number, however it is spelled" from being the
    same assertion.
    """
    expected: dict = {"source_ids": [source_id]}
    if contains:
        expected["contains"] = list(contains)
    if source_id in ANY_OF:
        expected["contains_any"] = list(ANY_OF[source_id])
    return expected


def direct_cases() -> list[dict]:
    return [
        {
            "name": f"direct_{source_id}",
            "slice": "direct",
            "inputs": {"user_message": question},
            "expected": expectation(source_id, contains),
        }
        for source_id, question, contains in DIRECT
    ]


async def paraphrase_cases(count: int) -> list[dict]:
    facts = fact_by_source_id()
    cases = []
    for source_id, question, contains in DIRECT:
        for index, phrasing in enumerate(
            await paraphrase(facts[source_id], question, count)
        ):
            cases.append(
                {
                    "name": f"paraphrase_{source_id}_{index}",
                    "slice": "paraphrase",
                    "inputs": {"user_message": phrasing},
                    "expected": expectation(source_id, contains),
                }
            )
    return cases


def unanswerable_cases() -> list[dict]:
    return [
        {
            "name": f"unanswerable_{index:02d}",
            "slice": "unanswerable",
            "inputs": {"user_message": question},
        }
        for index, question in enumerate(UNANSWERABLE)
    ]


def adversarial_cases() -> list[dict]:
    return [
        {
            "name": f"adversarial_{source_id}",
            "slice": "adversarial",
            "inputs": {"user_message": question},
            "expected": expectation(source_id, contains),
        }
        for source_id, question, contains in ADVERSARIAL
    ]


def multi_hop_cases() -> list[dict]:
    return [
        {
            "name": f"multi_hop_{index:02d}",
            "slice": "multi_hop",
            "inputs": {"user_message": question},
            "expected": {"source_ids": source_ids},
        }
        for index, (source_ids, question) in enumerate(MULTI_HOP)
    ]


async def build(use_llm: bool, per_fact: int) -> dict:
    cases = direct_cases()
    if use_llm:
        cases += await paraphrase_cases(per_fact)
    cases += multi_hop_cases()
    cases += unanswerable_cases()
    cases += adversarial_cases()
    return {"name": "green_village_qa", "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also generate the paraphrase slice (needs GEMINI_API_KEY).",
    )
    parser.add_argument("--per-fact", type=int, default=2)
    args = parser.parse_args()

    dataset = asyncio.run(build(args.llm, args.per_fact))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        yaml.dump(dataset, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {len(dataset['cases'])} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
