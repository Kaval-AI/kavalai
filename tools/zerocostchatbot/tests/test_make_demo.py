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

import json
import sys

import pytest

from tools.zerocostchatbot import make_demo
from tools.zerocostchatbot.build_index import build_rag_index
from tools.zerocostchatbot.make_demo import (
    build_parser,
    chunks_from_pages,
    compile_demo,
    inject_base_tag,
    read_site,
)
from tools.zerocostchatbot.pages_db import PagesDatabase
from tools.zerocostchatbot.tests.test_build_index import (
    add_page,
    fake_embedding_client,
    make_rag,
)

HOME_MARKDOWN = "# Welcome\n\nAcme makes fine anvils for discerning coyotes."
PRICING_MARKDOWN = "# Pricing\n\nAnvils cost ten dollars each, shipping included."


def make_site(tmp_path, screenshot=True):
    path = str(tmp_path / "acme.com.pages.db")
    with PagesDatabase(path) as db:
        add_page(db, "https://acme.com/", "Acme", HOME_MARKDOWN)
        add_page(db, "https://acme.com/pricing", "Pricing", PRICING_MARKDOWN)
        db.add_urls(["https://acme.com/broken"])
        if screenshot:
            db.save_screenshot("https://acme.com/", b"png-bytes")
    return path


def test_read_site_takes_the_first_row_as_the_start_page(tmp_path):
    site = read_site(make_site(tmp_path))
    assert site.site_url == "https://acme.com/"
    assert site.title == "Acme"
    assert site.screenshot == b"png-bytes"
    assert site.homepage_html == "<html/>"
    assert [row.url for row in site.pages] == [
        "https://acme.com/",
        "https://acme.com/pricing",
    ]


def test_read_site_refuses_an_empty_database(tmp_path):
    path = str(tmp_path / "empty.pages.db")
    PagesDatabase(path).close()
    with pytest.raises(ValueError, match="scrape first"):
        read_site(path)


def test_inject_base_tag():
    html = '<html><head lang="en"><title>t</title></head></html>'
    injected = inject_base_tag(html, "https://acme.com/")
    assert '<head lang="en"><base href="https://acme.com/"><title>' in injected
    # A fragment without a head still gets anchored.
    assert inject_base_tag("<p>x</p>", "https://a/").startswith(
        '<base href="https://a/">'
    )


async def test_compile_demo_with_screenshot_and_rag_index(tmp_path):
    pages = make_site(tmp_path)
    rag = make_rag(tmp_path)
    with PagesDatabase(pages) as db:
        await build_rag_index(db, rag, "acme.com")
    out = str(tmp_path / "demo")

    report = await compile_demo(
        pages,
        out,
        index_path=str(tmp_path / "index.rag.db"),
        suggestions=["Pricing?"],
    )
    assert report.backdrop == "screenshot"
    assert report.chunks == 2

    index_html = (tmp_path / "demo" / "index.html").read_text()
    assert 'src="screenshot.png"' in index_html
    assert '"Pricing?"' in index_html
    assert '"endpoint": null' in index_html
    assert (tmp_path / "demo" / "kaval-chatbot.js").exists()
    assert (tmp_path / "demo" / "kaval-chatbot.css").exists()
    assert (tmp_path / "demo" / "screenshot.png").read_bytes() == b"png-bytes"

    chunks = json.loads((tmp_path / "demo" / "chunks.json").read_text())
    assert {chunk["url"] for chunk in chunks} == {
        "https://acme.com/",
        "https://acme.com/pricing",
    }
    assert all(chunk["text"] for chunk in chunks)


async def test_compile_demo_falls_back_to_html_and_markdown(tmp_path):
    pages = make_site(tmp_path, screenshot=False)
    out = str(tmp_path / "demo")

    # No RAG index at the default path either, so chunks come from markdown.
    report = await compile_demo(pages, out)
    assert report.backdrop == "html"
    assert report.chunks == 2

    original = (tmp_path / "demo" / "original.html").read_text()
    assert '<base href="https://acme.com/">' in original
    assert 'src="original.html"' in (tmp_path / "demo" / "index.html").read_text()
    assert not (tmp_path / "demo" / "screenshot.png").exists()


async def test_compile_demo_endpoint_mode_ships_no_chunks(tmp_path):
    out = str(tmp_path / "demo")
    await compile_demo(
        make_site(tmp_path), out, endpoint="http://host:25000/api/chat/stream_agent"
    )
    assert not (tmp_path / "demo" / "chunks.json").exists()
    index_html = (tmp_path / "demo" / "index.html").read_text()
    assert '"endpoint": "http://host:25000/api/chat/stream_agent"' in index_html


async def test_compile_demo_refuses_an_ambiguous_index(tmp_path):
    pages = make_site(tmp_path)
    rag = make_rag(tmp_path)
    with PagesDatabase(pages) as db:
        await build_rag_index(db, rag, "one")
        await build_rag_index(db, rag, "two")
    with pytest.raises(ValueError, match="--collection"):
        await compile_demo(
            pages, str(tmp_path / "demo"), index_path=str(tmp_path / "index.rag.db")
        )
    # Naming one of them works; naming a wrong one does not.
    report = await compile_demo(
        pages,
        str(tmp_path / "demo"),
        index_path=str(tmp_path / "index.rag.db"),
        collection="one",
    )
    assert report.chunks == 2
    with pytest.raises(ValueError, match="no collection"):
        await compile_demo(
            pages,
            str(tmp_path / "demo"),
            index_path=str(tmp_path / "index.rag.db"),
            collection="three",
        )


def test_chunks_from_pages_carries_metadata(tmp_path):
    site = read_site(make_site(tmp_path))
    chunks = chunks_from_pages(site)
    assert chunks[0]["url"] == "https://acme.com/"
    assert chunks[0]["title"] == "Acme"
    assert chunks[1]["heading"] == "Pricing"


async def test_compile_demo_without_content_fails(tmp_path):
    path = str(tmp_path / "bare.pages.db")
    with PagesDatabase(path) as db:
        db.add_urls(["https://acme.com/"])
        db.record_skipped("https://acme.com/", "disallowed by robots.txt")
    with pytest.raises(ValueError, match="scrape first|No content"):
        await compile_demo(path, str(tmp_path / "demo"))


def test_main_exit_codes(tmp_path, monkeypatch):
    missing = str(tmp_path / "absent.pages.db")
    monkeypatch.setattr(sys, "argv", ["make_demo.py", missing])
    assert make_demo.main() == 2

    pages = make_site(tmp_path)
    out = str(tmp_path / "demo")
    monkeypatch.setattr(sys, "argv", ["make_demo.py", pages, "--out", out])
    assert make_demo.main() == 0
    assert (tmp_path / "demo" / "index.html").exists()

    empty = str(tmp_path / "empty.pages.db")
    PagesDatabase(empty).close()
    monkeypatch.setattr(sys, "argv", ["make_demo.py", empty])
    assert make_demo.main() == 2


def test_parser_defaults():
    args = build_parser().parse_args(["acme.pages.db"])
    assert args.out is None and args.endpoint is None and args.suggestion == []


async def test_compile_demo_plain_backdrop_when_nothing_stored(tmp_path):
    path = str(tmp_path / "nohtml.pages.db")
    with PagesDatabase(path) as db:
        db.add_urls(["https://acme.com/"])
        db.record_success(
            "https://acme.com/",
            status_code=200,
            fetch_mode="http",
            title="A",
            html=None,
            markdown=HOME_MARKDOWN,
        )
    report = await compile_demo(path, str(tmp_path / "demo"))
    assert report.backdrop == "none"
    index_html = (tmp_path / "demo" / "index.html").read_text()
    assert "screenshot.png" not in index_html and "original.html" not in index_html
