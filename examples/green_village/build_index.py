"""Build the checked-in fact index. Run this whenever ``facts.py`` changes.

    uv run python examples/green_village/build_index.py

Writes ``green_village.sqlite`` next to this file — a single ordinary SQLite
file, which is what lets the whole suite run offline and deterministically with
no Postgres and no embedding API.
"""

import asyncio
import json
from pathlib import Path

from kavalai.rag import SqliteRagService

from facts import (
    COLLECTION,
    EMBEDDING_MODEL,
    FACTS,
    INDEX_FILENAME,
    TOPICS,
    corpus_fingerprint,
    source_ids,
)

HERE = Path(__file__).parent
INDEX_PATH = HERE / INDEX_FILENAME
#: Written beside the index so a test can prove the two agree.
FINGERPRINT_PATH = HERE / "green_village.index.json"


async def build() -> None:
    INDEX_PATH.unlink(missing_ok=True)
    service = SqliteRagService(str(INDEX_PATH), model=EMBEDDING_MODEL)
    rows = await service.index_batch(
        texts=FACTS,
        metadata_list=[{"topic": topic} for topic in TOPICS],
        source_ids=source_ids(),
        collection_name=COLLECTION,
    )
    FINGERPRINT_PATH.write_text(
        json.dumps({"facts": len(FACTS), "fingerprint": corpus_fingerprint()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"indexed {len(rows)} facts into {INDEX_PATH}")


if __name__ == "__main__":
    asyncio.run(build())
