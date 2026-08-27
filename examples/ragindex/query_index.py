"""Query a RAG collection built by ``index_csv.py``.

The counterpart to the indexing script: same ``--index`` / ``--collection``
arguments, and the same embedding model has to be used, since a query is only
comparable to vectors produced by the model that made them.

Query the local SQLite index::

    uv run python -m examples.ragindex.query_index \
        "a song about a bird with a job" --index songs.db

Query the Postgres collection the backoffice RAG explorer shows::

    uv run --env-file .env python -m examples.ragindex.query_index \
        "office life is quietly destroying me" \
        --index postgres --collection songs --top-k 5

Similarity is cosine, **higher is better**, and identical across backends for
the same model --- ``1.0`` is a perfect match.
"""

import argparse
import asyncio
import textwrap

from loguru import logger

from examples.ragindex.index_csv import (
    DEFAULT_COLLECTION,
    EMBEDDING_MODEL,
    make_rag_service,
    split_columns,
)
from kavalai.rag.base import RagServiceResult


def format_result(result: RagServiceResult, rank: int, content_chars: int) -> str:
    """Render one hit as an indented, human-readable block.

    Args:
        result: The hit to render.
        rank: Its 1-based position in the result list.
        content_chars: How much of the indexed text to show (0 hides it).

    Returns:
        str: The formatted block, without a trailing newline.
    """
    metadata = ", ".join(f"{key}={value}" for key, value in result.rag_metadata.items())
    lines = [
        f"{rank:>2}. {result.similarity:.4f}  [{result.source_id}] {metadata}",
    ]
    if content_chars and result.content:
        excerpt = result.content[:content_chars].strip()
        lines.append(textwrap.indent(excerpt, "      "))
        if len(result.content) > content_chars:
            lines.append("      ...")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command line of the query script."""
    parser = argparse.ArgumentParser(
        description="Query a Kaval.AI RAG collection.",
    )
    parser.add_argument("text", help="What to search for")
    parser.add_argument(
        "--index",
        default="songs.db",
        help=(
            "'postgres' for the database in KAVALAI_DB_URI, a database URI, "
            "or a SQLite file path (default: songs.db)"
        ),
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Postgres schema holding the RAG tables (default: KAVALAI_DB_SCHEMA)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection to search (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=(
            "Embedding model; must be the one the collection was indexed with "
            f"(default: {EMBEDDING_MODEL})"
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="How many hits to return (default: 5)"
    )
    parser.add_argument(
        "--source-ids",
        default="",
        help="Comma-separated source ids to restrict the search to",
    )
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Return only the best hit per source id",
    )
    parser.add_argument(
        "--content-chars",
        type=int,
        default=200,
        help="How much of the indexed text to print, 0 for none (default: 200)",
    )
    return parser


async def run(args: argparse.Namespace) -> list[RagServiceResult]:
    """Run the query described by ``args`` and print the hits."""
    rag = make_rag_service(args.index, args.model, args.schema)
    logger.info(f"Querying {args.collection!r} in {args.index} for {args.text!r}")
    results = await rag.query(
        text=args.text,
        top_k=args.top_k,
        collection_name=args.collection,
        source_ids=split_columns(args.source_ids) or None,
        keep_best=args.keep_best,
    )
    if not results:
        print(f"No hits in collection {args.collection!r}.")
        return results
    print(f"\nTop {len(results)} for {args.text!r}:\n")
    for rank, result in enumerate(results, start=1):
        print(format_result(result, rank, args.content_chars))
        print()
    return results


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    try:
        results = asyncio.run(run(args))
    except (ValueError, KeyError, FileNotFoundError) as error:
        logger.error(error)
        return 2
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
