"""Index any CSV file into a Kaval.AI RAG collection.

The script knows nothing about songs. You tell it which columns make up the
text to embed, which columns to keep as metadata, and where the index lives;
everything else is the same for a catalogue, a ticket export or a lyrics
dump. It streams the file row by row, so the source may be far larger than
memory.

Index the bundled 100 made-up songs into a local SQLite file::

    uv run python -m examples.ragindex.index_csv \
        examples/ragindex/songs.csv --index songs.db

Index them into the Postgres database from ``.env`` instead, which is what
the backoffice RAG explorer reads::

    uv run --env-file .env python -m examples.ragindex.index_csv \
        examples/ragindex/songs.csv --index postgres --collection songs

Take a slice of a much bigger file --- here the first 2000 English rows of a
lyrics dump --- and put it in its own collection::

    uv run --env-file .env python -m examples.ragindex.index_csv \
        local_data/song_lyrics.csv --index postgres \
        --collection lyrics_sample --where language=en --limit 2000

``--index`` decides the backend: ``postgres`` reads ``KAVALAI_DB_URI`` and
``KAVALAI_DB_SCHEMA`` from the environment, anything containing ``://`` is
used as a database URI verbatim, and anything else is a SQLite file path.
"""

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

from loguru import logger

from kavalai.rag import PostgresRagService, SqliteRagService
from kavalai.rag.base import BaseRagService

# See https://qdrant.github.io/fastembed/examples/Supported_Models/ for the
# full list. This one is small, local and needs no API key.
EMBEDDING_MODEL = "fastembed/BAAI/bge-small-en-v1.5"

DEFAULT_CSV = "examples/ragindex/songs.csv"
DEFAULT_COLLECTION = "songs"

# What the bundled songs.csv wants. Putting the title and artist in front of
# the lyrics costs nothing at query time and makes the indexed text readable
# on its own --- the backoffice embedding projector labels each point with
# the first 100 characters of the content, so the title is what you see.
DEFAULT_TEXT_COLUMNS = "title,artist,lyrics"
DEFAULT_METADATA_COLUMNS = "title,artist,tag,year"
DEFAULT_SOURCE_ID_COLUMN = "id"

# bge-small truncates at 512 tokens, so embedding a whole novel of a field
# wastes time and tells you nothing the first paragraphs did not.
DEFAULT_MAX_CHARS = 2000


@dataclass
class IndexRow:
    """One CSV row, reduced to what the RAG service is given."""

    source_id: str
    text: str
    metadata: dict


def split_columns(value: str) -> list[str]:
    """Split a comma-separated column list, ignoring blanks and spacing."""
    return [name.strip() for name in value.split(",") if name.strip()]


def parse_where(clauses: list[str]) -> dict[str, str]:
    """Turn ``["language=en", "tag=pop"]`` into a column → value filter.

    Args:
        clauses: ``COLUMN=VALUE`` strings as given on the command line.

    Returns:
        The filter as a dictionary.

    Raises:
        ValueError: If a clause has no ``=``.
    """
    filters = {}
    for clause in clauses:
        column, separator, value = clause.partition("=")
        if not separator or not column.strip():
            raise ValueError(f"--where expects COLUMN=VALUE, got {clause!r}")
        filters[column.strip()] = value
    return filters


def build_text(row: dict, text_columns: list[str], max_chars: int) -> str:
    """Join the chosen columns into the text that will be embedded.

    Empty columns are dropped rather than left as blank lines, and the result
    is truncated to ``max_chars``.
    """
    parts = [str(row.get(name) or "").strip() for name in text_columns]
    text = "\n".join(part for part in parts if part)
    return text[:max_chars] if max_chars > 0 else text


def build_metadata(row: dict, metadata_columns: list[str]) -> dict:
    """Keep the chosen columns as metadata, as plain strings."""
    return {
        name: str(row[name])
        for name in metadata_columns
        if row.get(name) not in (None, "")
    }


def matches(row: dict, filters: dict[str, str]) -> bool:
    """Whether a row satisfies every ``--where`` clause."""
    return all(str(row.get(column) or "") == value for column, value in filters.items())


def read_rows(
    csv_path: str,
    text_columns: list[str],
    metadata_columns: list[str],
    source_id_column: Optional[str] = None,
    filters: Optional[dict[str, str]] = None,
    limit: Optional[int] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Iterator[IndexRow]:
    """Stream a CSV file as :class:`IndexRow` objects.

    The file is read lazily and never held in memory, so a nine-gigabyte
    export costs the same as a small one. Rows whose text ends up empty are
    skipped --- there is nothing to embed.

    Args:
        csv_path: Path of the CSV file.
        text_columns: Columns joined to form the embedded text.
        metadata_columns: Columns kept as metadata.
        source_id_column: Column used as the source identifier. When missing,
            the row's position in the file is used.
        filters: Optional ``COLUMN=VALUE`` filter, all clauses must match.
        limit: Stop after this many matching rows.
        max_chars: Truncate the text at this many characters (0 disables).

    Yields:
        IndexRow: One per row that passed the filter and has text.

    Raises:
        ValueError: If a requested column is not in the file's header.
    """
    filters = filters or {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        wanted = set(text_columns) | set(metadata_columns) | set(filters)
        if source_id_column:
            wanted.add(source_id_column)
        missing = sorted(wanted - set(header))
        if missing:
            raise ValueError(
                f"{csv_path} has no column(s) {', '.join(missing)}; "
                f"it has {', '.join(header)}"
            )

        yielded = 0
        for position, row in enumerate(reader):
            if limit is not None and yielded >= limit:
                return
            if not matches(row, filters):
                continue
            text = build_text(row, text_columns, max_chars)
            if not text:
                continue
            source_id = (
                str(row[source_id_column]) if source_id_column else str(position)
            )
            yielded += 1
            yield IndexRow(
                source_id=source_id,
                text=text,
                metadata=build_metadata(row, metadata_columns),
            )


def make_rag_service(index: str, model: str, schema: Optional[str]) -> BaseRagService:
    """Build the RAG backend named by ``--index``.

    ``postgres`` takes the connection from ``KAVALAI_DB_URI`` /
    ``KAVALAI_DB_SCHEMA`` (the same variables the agent server reads), a URI
    is used as given, and anything else is a SQLite file.

    Raises:
        KeyError: If ``postgres`` was asked for without ``KAVALAI_DB_URI``.
    """
    if index == "postgres":
        uri = os.environ["KAVALAI_DB_URI"]
        schema = schema or os.environ.get("KAVALAI_DB_SCHEMA", "public")
        return PostgresRagService.from_uri(uri, model, schema=schema)
    if "://" in index:
        return PostgresRagService.from_uri(index, model, schema=schema)
    return SqliteRagService(index, model)


async def index_rows(
    rag: BaseRagService,
    rows: Iterator[IndexRow],
    collection_name: str,
    batch_size: int = 32,
    replace: bool = False,
) -> int:
    """Embed and store rows in batches, reporting progress as it goes.

    Args:
        rag: The RAG backend to write to.
        rows: Rows to index, consumed lazily.
        collection_name: Collection the rows are added to.
        batch_size: How many rows are embedded per call.
        replace: Delete any existing entries carrying the same source ids
            first, which makes re-running the script idempotent instead of
            doubling the collection.

    Returns:
        int: How many rows were indexed.
    """
    total = 0
    batch: list[IndexRow] = []

    async def flush() -> None:
        nonlocal total
        if not batch:
            return
        if replace:
            await rag.delete_by_source_id(collection_name, [r.source_id for r in batch])
        await rag.index_batch(
            texts=[r.text for r in batch],
            metadata_list=[r.metadata for r in batch],
            source_ids=[r.source_id for r in batch],
            collection_name=collection_name,
        )
        total += len(batch)
        logger.info(f"Indexed {total} rows into {collection_name!r}")
        batch.clear()

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            await flush()
    await flush()
    return total


def build_parser() -> argparse.ArgumentParser:
    """Command line of the indexing script."""
    parser = argparse.ArgumentParser(
        description="Index a CSV file into a Kaval.AI RAG collection.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"CSV file to index (default: {DEFAULT_CSV})",
    )
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
        help=f"Collection to write to (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"Embedding model (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--text-columns",
        default=DEFAULT_TEXT_COLUMNS,
        help=f"Columns joined into the embedded text (default: {DEFAULT_TEXT_COLUMNS})",
    )
    parser.add_argument(
        "--metadata-columns",
        default=DEFAULT_METADATA_COLUMNS,
        help=f"Columns kept as metadata (default: {DEFAULT_METADATA_COLUMNS})",
    )
    parser.add_argument(
        "--source-id-column",
        default=DEFAULT_SOURCE_ID_COLUMN,
        help=(
            "Column used as the source id; pass an empty string to number the "
            f"rows instead (default: {DEFAULT_SOURCE_ID_COLUMN})"
        ),
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Only index rows matching this; may be given more than once",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many matching rows",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rows embedded per batch (default: 32)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Truncate the text at this length, 0 to keep it whole "
        f"(default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete entries with the same source ids first (idempotent re-run)",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    """Index the file described by ``args`` and return the row count."""
    rag = make_rag_service(args.index, args.model, args.schema)
    rows = read_rows(
        csv_path=args.csv_path,
        text_columns=split_columns(args.text_columns),
        metadata_columns=split_columns(args.metadata_columns),
        source_id_column=args.source_id_column or None,
        filters=parse_where(args.where),
        limit=args.limit,
        max_chars=args.max_chars,
    )
    logger.info(
        f"Indexing {args.csv_path} into {args.index} "
        f"(collection {args.collection!r}, model {args.model})"
    )
    total = await index_rows(
        rag,
        rows,
        collection_name=args.collection,
        batch_size=args.batch_size,
        replace=args.replace,
    )
    logger.info(f"Done: {total} rows in collection {args.collection!r}")
    return total


def main() -> int:
    """Entry point. Returns a process exit code."""
    # Lyrics, articles and log dumps routinely exceed the default field size.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    args = build_parser().parse_args()
    try:
        total = asyncio.run(run(args))
    except (ValueError, KeyError, FileNotFoundError) as error:
        logger.error(error)
        return 2
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
