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
"""

"""
This tool indexes large CSV files into a RAG (Retrieval-Augmented Generation) collection.
It handles batching, metadata field selection, and allows ignoring specific columns.

Example usage:
python -m kavalai.tools.index_csv local_data/song_lyrics.csv \
    --collection-name lyrics \
    --metadata-fields id title artist \
    --index-fields lyrics \
    --source-field id \
    --mode full \
    --replace
    --limit 100

Modes:
- full: (Default) The entire field is indexed as a single RAG entry.
- lines: The field is split into lines and each line is indexed separately.
"""

import argparse
import asyncio
import csv
from loguru import logger
import os
import sys
from typing import List, Optional, Generator, Dict, Any

from kavalai.llm_clients.registry import make_rag_service
from kavalai.rag import BaseRagService, PostgresRagService


def csv_row_generator(
    csv_path: str, limit: Optional[int] = None
) -> Generator[Dict[str, str], None, None]:
    """Generator that yields rows from a CSV file up to a limit."""
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if limit is not None and count >= limit:
                break
            yield row
            count += 1


def content_splitter_generator(
    rows: Generator[Dict[str, str], None, None],
    index_fields: List[str],
    metadata_fields: List[str],
    source_field: str,
    mode: str,
) -> Generator[Dict[str, any], None, None]:
    """Generator that splits row content based on the mode."""
    for row in rows:
        row_meta = {col: val for col, val in row.items() if col in metadata_fields}
        source_id = row.get(source_field, "default")

        for field in index_fields:
            if field not in row:
                continue

            content = row[field]
            yield from _split_content(content, row_meta, source_id, mode)


def _split_content(
    content: str, row_meta: Dict[str, Any], source_id: str, mode: str
) -> Generator[Dict[str, Any], None, None]:
    if mode == "lines":
        for line in content.splitlines():
            if line.strip():
                yield {
                    "text": line.strip(),
                    "meta": row_meta,
                    "source_id": source_id,
                }
    else:  # full
        yield {"text": content, "meta": row_meta, "source_id": source_id}


def rag_service_from_env(model: Optional[str] = None) -> BaseRagService:
    """Build the RAG service this script indexes into.

    Defaults to Postgres, configured from the environment. Set
    ``KAVALAI_RAG_SERVICE`` to the name of any service registered with
    :func:`~kavalai.register_rag_service` to index somewhere else; the
    registration carries that backend's own settings, so no environment
    variable here has to know about them.
    """
    service_name = os.environ.get("KAVALAI_RAG_SERVICE")
    if service_name:
        return make_rag_service(service_name)

    model = model if model else os.environ["KAVALAI_DEFAULT_EMBEDDING_MODEL"]
    return PostgresRagService.from_uri(
        uri=os.environ["KAVALAI_DB_URI"],
        model=model,
        schema=os.environ.get("KAVALAI_DB_SCHEMA"),
    )


async def index_csv(
    *,
    csv_path: str,
    collection_name: str,
    model: str = None,
    metadata_fields: List[str],
    index_fields: List[str],
    source_field: str,
    mode: str,
    limit: Optional[int],
    replace: bool = False,
    batch_size: int = 10,
    rag_service: Optional[BaseRagService] = None,
):
    """Index a CSV into a RAG collection.

    ``rag_service`` defaults to the Postgres service described by the
    environment; pass one explicitly to index into a different backend.
    """
    if rag_service is None:
        rag_service = rag_service_from_env(model)

    rows_processed = 0
    total_chunks = 0

    rows_gen = csv_row_generator(csv_path, limit)

    while True:
        batch_rows = []
        try:
            for _ in range(batch_size):
                batch_rows.append(next(rows_gen))
        except StopIteration:
            pass

        if not batch_rows:
            break

        texts = []
        metas = []
        source_ids = []

        for entry in content_splitter_generator(
            iter(batch_rows), index_fields, metadata_fields, source_field, mode
        ):
            texts.append(entry["text"])
            metas.append(entry["meta"])
            source_ids.append(entry["source_id"])

        if texts:
            if replace:
                # Collect unique source_ids in this batch to delete them
                unique_source_ids = list(set(source_ids))
                await rag_service.delete_by_source_id(
                    collection_name, unique_source_ids
                )

            rows_processed += len(batch_rows)
            total_chunks += len(texts)
            logger.info(
                f"Indexing batch of {len(batch_rows)} rows generating {len(texts)} chunks (Total rows: {rows_processed}, total chunks: {total_chunks})..."
            )
            await rag_service.index_batch(
                texts=texts,
                metadata_list=metas,
                collection_name=collection_name,
                source_ids=source_ids,
            )
        else:
            rows_processed += len(batch_rows)

    logger.info(
        f"Finished indexing {rows_processed} rows ({total_chunks} chunks) into collection '{collection_name}'."
    )


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the indexer command line."""
    parser = argparse.ArgumentParser(description="Index a CSV file into RAG.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--collection-name", required=True, help="RAG collection name")
    parser.add_argument(
        "--metadata-fields", nargs="*", default=[], help="Fields to store in metadata"
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Embedding model name e.g openai/text-embedding-3-small",
    )
    parser.add_argument(
        "--index-fields",
        nargs="+",
        required=True,
        help="Fields to include in indexed content",
    )
    parser.add_argument(
        "--source-field",
        required=True,
        help="Field to store in source_id of the rag_index table",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete matching rows (collection_name, source_id) before updating",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "lines"],
        default="full",
        help="How to index content (full: entire field as one entry, lines: each line as an entry)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of rows to index")

    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for indexing"
    )

    return parser.parse_args(argv)


def main(argv=None, rag_service: Optional[BaseRagService] = None):
    """Index the CSV named on the command line."""
    args = parse_args(argv)

    if not os.path.exists(args.csv_path):
        logger.error(f"Error: CSV file '{args.csv_path}' not found.")
        sys.exit(1)

    asyncio.run(
        index_csv(
            csv_path=args.csv_path,
            collection_name=args.collection_name,
            model=args.model,
            metadata_fields=args.metadata_fields,
            index_fields=args.index_fields,
            source_field=args.source_field,
            mode=args.mode,
            limit=args.limit,
            replace=args.replace,
            batch_size=args.batch_size,
            rag_service=rag_service,
        )
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
