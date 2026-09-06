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
import re
import sys

import pytest

from tools.zerocostchatbot import make_demo
from tools.zerocostchatbot.make_demo import (
    build_parser,
    compile_demo,
    render_index,
    site_url_of,
)
from tools.zerocostchatbot.pages_db import PagesDatabase
from tools.zerocostchatbot.tests.test_build_index import add_page

HOME_MARKDOWN = "# Welcome\n\nAcme makes fine anvils for discerning coyotes."
PRICING_MARKDOWN = "# Pricing\n\nAnvils cost ten dollars each, shipping included."


def make_site(tmp_path):
    path = str(tmp_path / "acme.com.pages.db")
    with PagesDatabase(path) as db:
        db.add_urls(["https://acme.com/skipped"])
        db.record_skipped("https://acme.com/skipped", "disallowed by robots.txt")
        add_page(db, "https://acme.com/", "Acme", HOME_MARKDOWN)
        add_page(db, "https://acme.com/pricing", "Pricing", PRICING_MARKDOWN)
    return path


def make_index(tmp_path):
    path = tmp_path / "acme.com.rag.db"
    path.write_bytes(b"sqlite bytes")
    return str(path)


def defaults_of(index_html):
    found = re.search(r"window\.KavalArchiveDefaults = (.*?);</script>", index_html)
    assert found, "index.html defines no defaults"
    return json.loads(found.group(1))


def test_site_url_of_skips_rows_without_html(tmp_path):
    assert site_url_of(make_site(tmp_path)) == "https://acme.com/"


def test_site_url_of_refuses_an_empty_database(tmp_path):
    path = str(tmp_path / "empty.pages.db")
    with PagesDatabase(path) as db:
        db.add_urls(["https://acme.com/"])
    with pytest.raises(ValueError, match="scrape first"):
        site_url_of(path)


def test_render_index_localises_the_widget_and_defines_defaults():
    html = (
        '<link href="../../chatbotwidget/kaval-chatbot.css">'
        '<script src="../../chatbotwidget/kaval-chatbot.js"></script>'
        '<script src="archive.js"></script>'
    )
    out = render_index(html, {"db": "a.pages.db", "title": "Ärimees"})
    assert "../../chatbotwidget/" not in out
    assert '<link href="kaval-chatbot.css">' in out
    assert defaults_of(out) == {"db": "a.pages.db", "title": "Ärimees"}
    assert out.index("KavalArchiveDefaults") < out.index('src="archive.js"')


def test_render_index_refuses_a_page_without_the_marker():
    with pytest.raises(ValueError, match="archive.js"):
        render_index("<html></html>", {})


def test_compile_demo_ships_the_databases_and_the_viewer(tmp_path):
    pages = make_site(tmp_path)
    make_index(tmp_path)
    out = tmp_path / "demo"

    report = compile_demo(
        pages, str(out), suggestions=["Pricing?"], title="Acme bot", model="m"
    )
    assert report.site_url == "https://acme.com/"
    assert report.pages_file == "acme.com.pages.db"
    assert report.rag_file == "acme.com.rag.db"
    assert report.endpoint is None

    for name in (
        "index.html",
        "archive.js",
        "kaval-chatbot.js",
        "kaval-chatbot.css",
        "acme.com.pages.db",
        "acme.com.rag.db",
    ):
        assert (out / name).exists(), name
    assert (out / "acme.com.rag.db").read_bytes() == b"sqlite bytes"

    index_html = (out / "index.html").read_text()
    assert "../../chatbotwidget/" not in index_html
    assert defaults_of(index_html) == {
        "db": "acme.com.pages.db",
        "rag": "acme.com.rag.db",
        "model": "m",
        "title": "Acme bot",
        "suggestions": ["Pricing?"],
    }


def test_compile_demo_without_an_index_ships_no_chatbot(tmp_path):
    pages = make_site(tmp_path)
    out = tmp_path / "demo"
    report = compile_demo(pages, str(out))
    assert report.rag_file is None
    assert not (out / "acme.com.rag.db").exists()
    assert defaults_of((out / "index.html").read_text()) == {"db": "acme.com.pages.db"}


def test_compile_demo_takes_a_named_index_and_collection(tmp_path):
    pages = make_site(tmp_path)
    index = tmp_path / "other.rag.db"
    index.write_bytes(b"other")
    out = tmp_path / "demo"
    report = compile_demo(pages, str(out), index_path=str(index), collection="one")
    assert report.rag_file == "other.rag.db"
    assert (out / "other.rag.db").read_bytes() == b"other"
    defaults = defaults_of((out / "index.html").read_text())
    assert defaults["rag"] == "other.rag.db" and defaults["collection"] == "one"


def test_compile_demo_refuses_a_database_uri_as_index(tmp_path):
    with pytest.raises(ValueError, match="SQLite file"):
        compile_demo(
            make_site(tmp_path),
            str(tmp_path / "demo"),
            index_path="postgresql://kavalai@localhost/kavalai",
        )


def test_compile_demo_endpoint_mode_ships_no_index(tmp_path):
    pages = make_site(tmp_path)
    make_index(tmp_path)
    out = tmp_path / "demo"
    report = compile_demo(
        pages, str(out), endpoint="http://host:25000/api/chat/stream_agent"
    )
    assert report.rag_file is None
    assert not (out / "acme.com.rag.db").exists()
    assert defaults_of((out / "index.html").read_text()) == {
        "db": "acme.com.pages.db",
        "endpoint": "http://host:25000/api/chat/stream_agent",
    }


def test_compile_demo_defaults_the_output_folder(tmp_path):
    report = compile_demo(make_site(tmp_path))
    assert report.out_dir == str(tmp_path / "acme.com.demo")
    assert (tmp_path / "acme.com.demo" / "index.html").exists()


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
    assert args.index is None and args.model is None
