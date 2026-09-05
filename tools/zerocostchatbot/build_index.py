"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Build a RAG index from a scraped pages database (see ``scrape.py``).

Index the scraped Kaval.AI docs into ``docs.kaval.ai.rag.db``::

    python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db

The two databases are separate on purpose: the pages database is the heavy
crawl artefact (raw HTML), the RAG index is the small servable one. The
collection is dropped and rebuilt on every run — the pages database is the
source of truth, and a rebuild of a small site takes seconds — so the
script is idempotent.

Each page's markdown is chunked at heading boundaries, every chunk prefixed
with the page title and heading path ("Kaval.AI Docs › Quickstart ›
Install") so short chunks stay retrievable on their own. Chunk metadata
carries ``url``, ``title``, ``heading`` and ``crawled_at`` — enough to cite
the source page in an answer.

``--index`` decides the backend, exactly as in ``examples/ragindex``:
``postgres`` reads ``KAVALAI_DB_URI`` / ``KAVALAI_DB_SCHEMA`` from the
environment, anything containing ``://`` is used as a database URI, and
anything else is a SQLite file path. The default embedding model is local
fastembed — no API key, no per-page cost.
"""

import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from kavalai.rag import SqliteRagService, rag_service_from_uri
from kavalai.rag.base import BaseRagService
from kavalai.settings import apply_normalizer_from_env
from tools.zerocostchatbot.pages_db import PagesDatabase

# Small, local and key-free; see the fastembed model list for alternatives.
DEFAULT_MODEL = "fastembed/BAAI/bge-small-en-v1.5"

# bge-small truncates at 512 tokens, so a much longer chunk would embed only
# its beginning anyway.
DEFAULT_MAX_CHARS = 2000

DEFAULT_BATCH_SIZE = 32

HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
PAGES_SUFFIX = ".pages.db"


@dataclass
class Chunk:
    """One indexable piece of a page."""

    text: str
    heading: str
    position: int


def split_to_size(text: str, max_chars: int) -> list[str]:
    """Split text into pieces of at most ``max_chars``, at paragraph breaks.

    A single paragraph longer than the cap is split mid-text — embedding it
    whole would silently truncate instead.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(paragraph) > max_chars:
            parts.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        current = paragraph
    if current:
        parts.append(current)
    return parts


def chunk_markdown(
    markdown: str, title: str = "", max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    """Chunk a page's markdown at heading boundaries.

    Each chunk is prefixed with the page title and the heading path leading
    to it, so a chunk retrieved on its own still says what it is about.
    """
    sections: list[tuple[str, str]] = []
    path: dict[int, str] = {}
    heading = ""
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((heading, body))
        lines.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match[1])
            path = {k: v for k, v in path.items() if k < level}
            path[level] = match[2].strip()
            heading = " › ".join(path[k] for k in sorted(path))
        else:
            lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for section_heading, body in sections:
        prefix = " › ".join(p for p in (title, section_heading) if p)
        for part in split_to_size(body, max_chars):
            text = f"{prefix}\n\n{part}" if prefix else part
            chunks.append(
                Chunk(text=text, heading=section_heading, position=len(chunks))
            )
    return chunks


@dataclass
class IndexReport:
    """What one indexing run did."""

    pages: int = 0
    chunks: int = 0
    skipped: int = 0


async def build_rag_index(
    pages: PagesDatabase,
    rag: BaseRagService,
    collection_name: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IndexReport:
    """Rebuild ``collection_name`` from every page that holds markdown.

    The collection is dropped first, so the index always mirrors the pages
    database exactly; pages without content (failures, robots skips) are
    counted as skipped.
    """
    if any(c["name"] == collection_name for c in await rag.list_collections()):
        await rag.drop_collection(collection_name)

    report = IndexReport()
    texts: list[str] = []
    metadata: list[dict] = []
    source_ids: list[str] = []

    async def flush() -> None:
        if not texts:
            return
        await rag.index_batch(
            texts=list(texts),
            metadata_list=list(metadata),
            source_ids=list(source_ids),
            collection_name=collection_name,
        )
        report.chunks += len(texts)
        logger.info(f"Indexed {report.chunks} chunks into {collection_name!r}")
        texts.clear()
        metadata.clear()
        source_ids.clear()

    for row in pages.iter_pages():
        chunks = chunk_markdown(row.markdown or "", row.title or "", max_chars)
        if not chunks:
            report.skipped += 1
            continue
        report.pages += 1
        for chunk in chunks:
            texts.append(chunk.text)
            metadata.append(
                {
                    key: value
                    for key, value in {
                        "url": row.url,
                        "title": row.title,
                        "heading": chunk.heading,
                        "crawled_at": row.last_crawled_at,
                        "chunk": chunk.position,
                    }.items()
                    if value not in (None, "")
                }
            )
            source_ids.append(row.url)
            if len(texts) >= batch_size:
                await flush()
    await flush()
    return report


def make_rag_service(index: str, model: str, schema: Optional[str]) -> BaseRagService:
    """The RAG backend named by ``--index`` — same contract as ``examples/ragindex``.

    Raises:
        KeyError: If ``postgres`` was asked for without ``KAVALAI_DB_URI``.
    """
    if index == "postgres":
        uri = os.environ["KAVALAI_DB_URI"]
        schema = schema or os.environ.get("KAVALAI_DB_SCHEMA", "public")
        return rag_service_from_uri(uri, model, schema=schema)
    if "://" in index:
        return rag_service_from_uri(index, model, schema=schema)
    return SqliteRagService(index, model)


def default_index_path(pages_path: str) -> str:
    """``docs.kaval.ai.pages.db`` → ``docs.kaval.ai.rag.db``."""
    if pages_path.endswith(PAGES_SUFFIX):
        return pages_path[: -len(PAGES_SUFFIX)] + ".rag.db"
    return pages_path + ".rag.db"


def default_collection(pages: PagesDatabase) -> str:
    """The scraped site's host, read from the first page — e.g. ``docs.kaval.ai``."""
    for row in pages.iter_pages():
        host = urlparse(row.url).netloc
        if host:
            return host
    return "site"


async def run(args: argparse.Namespace) -> IndexReport:
    """Build the index the CLI arguments describe and return what was done."""
    if not os.path.exists(args.pages):
        raise FileNotFoundError(f"Pages database not found: {args.pages}")
    index = args.index or default_index_path(args.pages)
    rag = make_rag_service(index, args.model, args.schema)
    with PagesDatabase(args.pages) as pages:
        collection = args.collection or default_collection(pages)
        logger.info(
            f"Indexing {args.pages} into {index}"
            f" (collection {collection!r}, model {args.model})"
        )
        report = await build_rag_index(
            pages,
            rag,
            collection,
            max_chars=args.max_chars,
            batch_size=args.batch_size,
        )
    logger.info(
        f"Done: {report.chunks} chunks from {report.pages} pages"
        f" in {collection!r}"
        + (f", {report.skipped} pages without content" if report.skipped else "")
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Command line of the index builder."""
    parser = argparse.ArgumentParser(
        description="Build a RAG index from a scraped pages database.",
        epilog=(
            "Example: python -m tools.zerocostchatbot.build_index"
            " docs.kaval.ai.pages.db"
        ),
    )
    parser.add_argument("pages", help="Pages database produced by scrape.py")
    parser.add_argument(
        "--index",
        default=None,
        help=(
            "'postgres' for the database in KAVALAI_DB_URI, a database URI,"
            " or a SQLite file path (default: <pages>.rag.db beside the input)"
        ),
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Postgres schema holding the RAG tables (default: KAVALAI_DB_SCHEMA)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection to rebuild (default: the scraped site's host)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=(
            "Split chunks longer than this many characters, 0 to keep whole"
            f" sections (default: {DEFAULT_MAX_CHARS})"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks embedded per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    try:
        apply_normalizer_from_env()
        report = asyncio.run(run(args))
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        logger.error(error)
        return 2
    return 0 if report.chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
