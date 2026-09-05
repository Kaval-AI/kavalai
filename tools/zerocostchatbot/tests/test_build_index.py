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

import hashlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from kavalai.llm_clients.common import create_model_call_stat
from kavalai.rag import SqliteRagService
from tools.zerocostchatbot import build_index
from tools.zerocostchatbot.build_index import (
    build_parser,
    build_rag_index,
    chunk_markdown,
    default_collection,
    default_index_path,
    make_rag_service,
    split_to_size,
)
from tools.zerocostchatbot.pages_db import PagesDatabase

MARKDOWN = """Intro paragraph before any heading.

# Getting started

Install the package.

## Details

First detail paragraph.

Second detail paragraph.

# Reference

Reference text.
"""


def fake_embedding_client():
    """Deterministic vectors derived from the text, so no model is needed."""

    async def compute_embeddings(texts, **kwargs):
        vectors = [
            [byte / 255 for byte in hashlib.sha256(t.encode()).digest()[:8]]
            for t in texts
        ]
        stats = create_model_call_stat(
            call_type="embedding",
            model="fake/embedding-model",
            duration_seconds=0.0,
            batch_size=len(texts),
            total_tokens=len(texts),
        )
        return vectors, stats

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=compute_embeddings)
    return client


def make_rag(tmp_path):
    service = SqliteRagService(
        str(tmp_path / "index.rag.db"), model="fake/embedding-model"
    )
    service.embedding_client = fake_embedding_client()
    return service


def make_pages(tmp_path):
    return PagesDatabase(str(tmp_path / "site.pages.db"))


def add_page(db, url, title, markdown):
    db.add_urls([url])
    db.record_success(
        url,
        status_code=200,
        fetch_mode="http",
        title=title,
        html="<html/>",
        markdown=markdown,
    )


def test_chunk_markdown_tracks_the_heading_path():
    chunks = chunk_markdown(MARKDOWN, title="Kaval Docs")
    texts = [chunk.text for chunk in chunks]
    assert texts[0].startswith("Kaval Docs\n\nIntro paragraph")
    assert texts[1].startswith("Kaval Docs › Getting started\n\n")
    assert texts[2].startswith("Kaval Docs › Getting started › Details\n\n")
    # A new top-level heading resets the path.
    assert texts[3].startswith("Kaval Docs › Reference\n\n")
    assert [chunk.position for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[2].heading == "Getting started › Details"


def test_chunk_markdown_without_title_or_headings():
    (chunk,) = chunk_markdown("Just a paragraph.")
    assert chunk.text == "Just a paragraph."
    assert chunk.heading == ""
    assert chunk_markdown("") == []
    # A heading with no body under it yields no empty chunk.
    assert chunk_markdown("# Lonely heading") == []


def test_chunk_markdown_splits_long_sections():
    body = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(10))
    chunks = chunk_markdown(f"# Big\n\n{body}", title="T", max_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 + len("T › Big\n\n") for chunk in chunks)
    assert all(chunk.text.startswith("T › Big") for chunk in chunks)


def test_split_to_size():
    assert split_to_size("short", 100) == ["short"]
    assert split_to_size("a" * 50, 0) == ["a" * 50]
    # An oversized single paragraph is hard-split rather than truncated.
    assert split_to_size("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]
    # Paragraphs pack together up to the cap, then a new part starts.
    assert split_to_size("aaa\n\nbbb\n\nccc", 8) == ["aaa\n\nbbb", "ccc"]


def test_default_index_path():
    assert default_index_path("docs.kaval.ai.pages.db") == "docs.kaval.ai.rag.db"
    assert default_index_path("scrape.sqlite") == "scrape.sqlite.rag.db"


def test_default_collection(tmp_path):
    with make_pages(tmp_path) as db:
        assert default_collection(db) == "site"
        db.add_urls(["https://docs.kaval.ai/"])
        assert default_collection(db) == "docs.kaval.ai"


def test_make_rag_service(tmp_path, monkeypatch):
    path_service = make_rag_service(str(tmp_path / "a.db"), "fake/model", None)
    assert isinstance(path_service, SqliteRagService)
    uri_service = make_rag_service(f"sqlite:///{tmp_path}/b.db", "fake/model", None)
    assert isinstance(uri_service, SqliteRagService)
    monkeypatch.delenv("KAVALAI_DB_URI", raising=False)
    with pytest.raises(KeyError):
        make_rag_service("postgres", "fake/model", None)


async def test_build_rag_index_end_to_end(tmp_path):
    rag = make_rag(tmp_path)
    with make_pages(tmp_path) as pages:
        add_page(pages, "https://docs.kaval.ai/", "Home", MARKDOWN)
        add_page(pages, "https://docs.kaval.ai/a", "Page A", "# A\n\nAlpha text.")
        # A failed page holds no markdown and is skipped.
        pages.add_urls(["https://docs.kaval.ai/broken"])
        pages.record_failure(
            "https://docs.kaval.ai/broken", status_code=500, fetch_error="HTTP 500"
        )

        report = await build_rag_index(pages, rag, "docs.kaval.ai", batch_size=2)
        assert (report.pages, report.chunks, report.skipped) == (2, 5, 1)

        # The fake embeddings are exact-text hashes, so querying with a
        # chunk's precise text must rank that chunk first.
        results = await rag.query(
            "Home › Getting started › Details\n\n"
            "First detail paragraph.\n\nSecond detail paragraph.",
            collection_name="docs.kaval.ai",
            top_k=1,
        )
        top = results[0]
        assert top.rag_metadata["url"] == "https://docs.kaval.ai/"
        assert top.rag_metadata["heading"] == "Getting started › Details"
        assert top.rag_metadata["title"] == "Home"
        assert "crawled_at" in top.rag_metadata
        # All of a page's chunks share the page URL as their source id; the
        # chunk position lives in the metadata instead.
        assert top.source_id == "https://docs.kaval.ai/"
        assert top.rag_metadata["chunk"] == 2

        # A rebuild replaces the collection instead of doubling it.
        again = await build_rag_index(pages, rag, "docs.kaval.ai", batch_size=2)
        assert again.chunks == 5
        assert await rag.count_entries("docs.kaval.ai") == 5


async def test_run_uses_defaults_from_the_pages_file(tmp_path, monkeypatch):
    pages_path = tmp_path / "docs.kaval.ai.pages.db"
    with PagesDatabase(str(pages_path)) as pages:
        add_page(pages, "https://docs.kaval.ai/", "Home", "# H\n\nBody text.")

    captured = {}

    def fake_service(index, model, schema):
        captured["index"], captured["model"] = index, model
        service = SqliteRagService(str(tmp_path / "out.rag.db"), model="fake/model")
        service.embedding_client = fake_embedding_client()
        return service

    monkeypatch.setattr(build_index, "make_rag_service", fake_service)
    args = build_parser().parse_args([str(pages_path)])
    report = await build_index.run(args)
    assert report.chunks == 1
    assert captured["index"] == str(tmp_path / "docs.kaval.ai.rag.db")
    assert captured["model"].startswith("fastembed/")


async def test_run_refuses_a_missing_pages_file(tmp_path):
    args = build_parser().parse_args([str(tmp_path / "absent.pages.db")])
    with pytest.raises(FileNotFoundError):
        await build_index.run(args)


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_index.py", "x.pages.db"])

    async def ok_run(args):
        return build_index.IndexReport(pages=1, chunks=3)

    monkeypatch.setattr(build_index, "run", ok_run)
    assert build_index.main() == 0

    async def empty_run(args):
        return build_index.IndexReport()

    monkeypatch.setattr(build_index, "run", empty_run)
    assert build_index.main() == 1

    async def missing_run(args):
        raise FileNotFoundError("Pages database not found")

    monkeypatch.setattr(build_index, "run", missing_run)
    assert build_index.main() == 2


async def test_build_rag_index_of_an_empty_pages_database(tmp_path):
    rag = make_rag(tmp_path)
    with make_pages(tmp_path) as pages:
        report = await build_rag_index(pages, rag, "empty")
    assert (report.pages, report.chunks, report.skipped) == (0, 0, 0)


def test_make_rag_service_postgres_reads_the_environment(tmp_path, monkeypatch):
    #` Any URI counts; a SQLite one keeps the test free of a Postgres driver.
    monkeypatch.setenv("KAVALAI_DB_URI", f"sqlite:///{tmp_path}/env.rag.db")
    service = make_rag_service("postgres", "fake/model", None)
    assert isinstance(service, SqliteRagService)
