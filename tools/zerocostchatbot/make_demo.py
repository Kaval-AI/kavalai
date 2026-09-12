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

Compile a client demo from a scraped site: a static folder holding the
archive viewer (``archive.html``), the pages database and the RAG index,
so opening the folder's ``index.html`` browses the scraped site with the
chat widget answering from its own content — entirely in the browser.

    python -m tools.zerocostchatbot.make_demo docs.kaval.ai.pages.db
    python -m http.server -d docs.kaval.ai.demo

The folder is self-contained static files, servable from any bucket. The
``index.html`` is ``archive.html`` with the widget paths made local and
``window.KavalArchiveDefaults`` defined, so the page loads the two database
files beside it without ``?db=``. What answers the chat depends on
``--endpoint``: without it, WebLLM runs an in-browser model over the RAG
index (quoting passages when WebGPU is missing); with ``--endpoint URL`` the
widget talks to a running kavalai agent server and no index is shipped.
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from kavalai.widget import WIDGET_FILES, widget_dir
from tools.zerocostchatbot.build_index import default_index_path
from tools.zerocostchatbot.pages_db import PagesDatabase

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
WIDGET_PATH_PREFIX = "../../kavalai/widget/"


@dataclass
class DemoReport:
    """What one compile produced: the folder, the site, the file names
    inside the folder (``rag_file`` is None in endpoint mode or when no index
    exists) and the endpoint, if any."""

    out_dir: str
    site_url: str
    pages_file: str
    rag_file: Optional[str]
    endpoint: Optional[str]


def site_url_of(pages_path: str) -> str:
    """The crawl's start URL — the first row, since seeds are inserted before
    any discovered link."""
    with PagesDatabase(pages_path) as db:
        for row in db.iter_pages():
            if row.html:
                return row.url
    raise ValueError(f"{pages_path} holds no fetched pages; scrape first")


def render_index(archive_html: str, defaults: dict) -> str:
    """``archive.html`` as the demo's ``index.html``: the widget files sit
    beside it, and the defaults are defined before ``archive.js`` loads."""
    script = (
        "<script>window.KavalArchiveDefaults = "
        + json.dumps(defaults, ensure_ascii=False)
        + ";</script>\n  "
    )
    marker = '<script src="archive.js">'
    if marker not in archive_html:
        raise ValueError("archive.html no longer loads archive.js where expected")
    return archive_html.replace(WIDGET_PATH_PREFIX, "").replace(marker, script + marker)


def default_out_dir(pages_path: str, site_url: str) -> str:
    """``<host>.demo`` next to the pages database."""
    host = urlparse(site_url).netloc
    return os.path.join(os.path.dirname(pages_path) or ".", f"{host}.demo")


def compile_demo(
    pages_path: str,
    out_dir: Optional[str] = None,
    *,
    index_path: Optional[str] = None,
    collection: Optional[str] = None,
    endpoint: Optional[str] = None,
    title: Optional[str] = None,
    suggestions: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> DemoReport:
    """Assemble the demo folder and return what was built."""
    site_url = site_url_of(pages_path)
    out_dir = out_dir or default_out_dir(pages_path, site_url)
    os.makedirs(out_dir, exist_ok=True)

    for name in WIDGET_FILES:
        shutil.copy(widget_dir() / name, os.path.join(out_dir, name))
    shutil.copy(
        os.path.join(TOOL_DIR, "archive.js"), os.path.join(out_dir, "archive.js")
    )

    pages_file = os.path.basename(pages_path)
    shutil.copy(pages_path, os.path.join(out_dir, pages_file))

    rag_file = None
    if not endpoint:
        index_path = index_path or default_index_path(pages_path)
        if "://" in index_path:
            raise ValueError(
                "The demo ships the RAG index as a file the browser loads;"
                f" build it into a SQLite file, not {index_path}"
            )
        if os.path.exists(index_path):
            rag_file = os.path.basename(index_path)
            shutil.copy(index_path, os.path.join(out_dir, rag_file))
        else:
            logger.warning(
                f"No RAG index at {index_path}; the demo browses the site"
                " without a chatbot (run build_index first)"
            )

    defaults = {"db": pages_file}
    if rag_file:
        defaults["rag"] = rag_file
    for key, value in (
        ("collection", collection),
        ("model", model),
        ("endpoint", endpoint),
        ("title", title),
        ("suggestions", suggestions or None),
    ):
        if value:
            defaults[key] = value

    with open(os.path.join(TOOL_DIR, "archive.html"), encoding="utf-8") as f:
        archive_html = f.read()
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_index(archive_html, defaults))

    logger.info(
        f"Demo compiled to {out_dir}: pages={pages_file},"
        + (f" endpoint={endpoint}" if endpoint else f" rag={rag_file}")
        + f". Serve with: python -m http.server -d {out_dir}"
    )
    return DemoReport(
        out_dir=out_dir,
        site_url=site_url,
        pages_file=pages_file,
        rag_file=rag_file,
        endpoint=endpoint,
    )


def build_parser() -> argparse.ArgumentParser:
    """Command line of the demo compiler."""
    parser = argparse.ArgumentParser(
        description="Compile a static chatbot demo from a scraped site.",
        epilog=(
            "Example: python -m tools.zerocostchatbot.make_demo"
            " docs.kaval.ai.pages.db && python -m http.server -d docs.kaval.ai.demo"
        ),
    )
    parser.add_argument("pages", help="Pages database produced by scrape.py")
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder (default: <host>.demo next to the pages database)",
    )
    parser.add_argument(
        "--index",
        default=None,
        help=(
            "RAG index SQLite file shipped with the demo (default: the"
            " <site>.rag.db beside the input; missing means no chatbot)"
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection inside --index (default: its only collection)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="WebLLM chat model id the demo loads (default: the page's first choice)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        metavar="URL",
        help=(
            "Agent-server stream endpoint (e.g. http://host:25000/api/chat/"
            "stream_agent); the widget talks to it instead of an in-browser model"
        ),
    )
    parser.add_argument(
        "--title", default=None, help="Chat window title (default: host)"
    )
    parser.add_argument(
        "--suggestion",
        action="append",
        default=[],
        metavar="TEXT",
        help="Suggested question chip; may be given more than once",
    )
    return parser


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    if not os.path.exists(args.pages):
        logger.error(f"Pages database not found: {args.pages}")
        return 2
    try:
        compile_demo(
            args.pages,
            args.out,
            index_path=args.index,
            collection=args.collection,
            endpoint=args.endpoint,
            title=args.title,
            suggestions=args.suggestion,
            model=args.model,
        )
    except ValueError as error:
        logger.error(error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
