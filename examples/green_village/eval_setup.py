"""Everything the suite needs registered before it can run.

The suite's ``setup:`` key points here. It is not optional: ``WorkflowEngine``
resolves a named RAG service **eagerly, at construction**, so without this the
chatbot workflow cannot even be built — it fails with *"Node 'retrieve' needs
RAG service 'default', which is neither passed to the engine nor registered"*.
"""

from pathlib import Path

from kavalai import register_rag_service
from kavalai.eval import Evaluator, Score, evaluator
from kavalai.rag import SqliteRagService

from facts import EMBEDDING_MODEL, INDEX_FILENAME, fact_by_source_id


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, (str, int, float)) else list(value)


HERE = Path(__file__).parent

# Registered under "default", which is the name a `rag_query` node resolves to
# when neither it nor the workflow says otherwise — so the workflow YAML never
# has to mention a filename or a connection string.
register_rag_service(
    "default",
    SqliteRagService,
    replace=True,
    filename=str(HERE / INDEX_FILENAME),
    model=EMBEDDING_MODEL,
)


@evaluator("answers_with_fact", replace=True)
class AnswersWithFact(Evaluator):
    """The answer contains the key figure from the fact it should have used.

    A domain evaluator, in the example rather than the library, because only
    this corpus knows that "how deep is Lake Miller" must produce ``1.2``. It
    is also the cheapest possible correctness check: no judge, no embedding, no
    ambiguity — which is what the numeric corpus was chosen for.
    """

    def __init__(self):
        self.facts = fact_by_source_id()

    async def score(self, case, record) -> Score:
        expected = case.expected if isinstance(case.expected, dict) else {}
        needles = _as_list(expected.get("contains"))
        # `1,847` and `1847` are the same figure written two ways, so they are
        # alternatives rather than two separate requirements. Keeping the two
        # ideas in one key would have made "states both numbers" and "states
        # the number, however spelled" indistinguishable.
        alternatives = _as_list(expected.get("contains_any"))
        if not needles and not alternatives:
            return Score(name=self.name, reason="no expected value for this case")

        answer = record.output_text().casefold()
        missing = [n for n in needles if str(n).casefold() not in answer]
        if alternatives and not any(str(a).casefold() in answer for a in alternatives):
            missing.append(" or ".join(str(a) for a in alternatives))
        return Score.boolean(
            self.name,
            not missing,
            reason=None if not missing else f"the answer never states {missing}",
        )
