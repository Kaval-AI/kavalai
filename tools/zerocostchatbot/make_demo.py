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

Compile a client demo from a scraped site: a static folder showing the
original page with the Kaval.AI chat widget floating over it.

    python -m tools.zerocostchatbot.make_demo docs.kaval.ai.pages.db
    python -m http.server -d docs.kaval.ai.demo

The folder is self-contained static files — servable from a bucket. The
backdrop is the homepage screenshot when the scrape captured one, else the
stored homepage HTML with a ``<base>`` tag injected so styles and images
load from the live site.

What answers the chat depends on ``--endpoint``:

- Without it, the demo ships the site's chunks (from the RAG index when one
  exists beside the pages database, else re-chunked from the stored
  markdown) and answers client-side: a small lexical ranker picks the most
  relevant passages and replies with them, source links included. No
  backend, no API key, no model download — it demonstrates the widget and
  the site's own content, not an LLM.
- With ``--endpoint URL`` the widget talks to a running kavalai agent
  server (``create_agent_router``) — the full LLM-backed bot.
"""

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from tools.zerocostchatbot.build_index import (
    chunk_markdown,
    default_index_path,
    make_rag_service,
)
from tools.zerocostchatbot.pages_db import PagesDatabase, PageRow

WIDGET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "chatbotwidget")
)
WIDGET_FILES = ("kaval-chatbot.js", "kaval-chatbot.css")

# Enough for a few hundred pages of passages while keeping chunks.json in the
# low megabytes; ranking quality does not improve past that in a demo.
DEFAULT_MAX_CHUNKS = 800


@dataclass
class DemoReport:
    """What one compile produced."""

    out_dir: str
    site_url: str
    backdrop: str
    chunks: int


@dataclass
class SitePages:
    """What the demo needs from the pages database."""

    site_url: str
    title: str
    screenshot: Optional[bytes]
    homepage_html: Optional[str]
    pages: list[PageRow]


def read_site(pages_path: str) -> SitePages:
    """The site as the pages database recorded it.

    The first row is the crawl's start URL — seeds are inserted before any
    discovered link, in order.
    """
    with PagesDatabase(pages_path) as db:
        rows = [row for row in db.iter_pages()]
        if not rows:
            raise ValueError(f"{pages_path} holds no pages; scrape first")
        start = rows[0]
        return SitePages(
            site_url=start.url,
            title=start.title or urlparse(start.url).netloc,
            screenshot=start.screenshot,
            homepage_html=start.html,
            pages=[row for row in rows if row.markdown],
        )


async def chunks_from_index(index: str, collection: Optional[str]) -> list[dict]:
    """The chunks a built RAG index holds, as chunks.json entries.

    ``index`` is what ``build_index --index`` takes — a database URI or a
    SQLite file path; reading needs no embedding model.
    """
    rag = make_rag_service(index, None, None)
    if not rag.supports("iter_entries"):
        raise ValueError(f"{type(rag).__name__} cannot list its entries")
    collections = await rag.list_collections()
    names = [c["name"] for c in collections]
    if collection is None:
        if len(names) != 1:
            raise ValueError(
                f"{index} holds collections {names}; pick one with --collection"
            )
        collection = names[0]
    elif collection not in names:
        raise ValueError(f"{index} has no collection {collection!r} (has {names})")

    chunks = []
    async for entry in rag.iter_entries(collection):
        metadata = entry["rag_metadata"] or {}
        chunks.append(
            {
                "text": entry["content"],
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "heading": metadata.get("heading", ""),
            }
        )
    return chunks


def chunks_from_pages(site: SitePages, max_chars: int = 2000) -> list[dict]:
    """Chunks re-derived from the stored markdown, when no RAG index exists."""
    chunks = []
    for row in site.pages:
        for chunk in chunk_markdown(row.markdown, row.title or "", max_chars):
            chunks.append(
                {
                    "text": chunk.text,
                    "url": row.url,
                    "title": row.title or "",
                    "heading": chunk.heading,
                }
            )
    return chunks


def inject_base_tag(html: str, site_url: str) -> str:
    """Anchor a stored page's relative URLs to the live site.

    The stored homepage references styles and images by relative path; a
    ``<base>`` right after ``<head>`` makes the browser fetch them from the
    original site, so the saved page renders like the live one.
    """
    base = f'<base href="{site_url}">'
    lowered = html.lower()
    at = lowered.find("<head")
    if at != -1:
        end = lowered.find(">", at)
        if end != -1:
            return html[: end + 1] + base + html[end + 1 :]
    return base + html


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__ — Kaval.AI chatbot demo</title>
  <link rel="stylesheet" href="kaval-chatbot.css" />
  <style>
    html, body { height: 100%; margin: 0; }
    body { font-family: system-ui, sans-serif; }
    .demo-backdrop { position: fixed; inset: 0; border: none; width: 100%; height: 100%; }
    img.demo-backdrop { object-fit: cover; object-position: top; }
    .demo-badge {
      position: fixed; left: 16px; bottom: 16px; z-index: 1200;
      background: #273e47; color: #f4f4f6; font-size: 0.8rem;
      padding: 8px 14px; border-radius: 999px; box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    .demo-badge a { color: #acc12f; text-decoration: none; font-weight: 600; }
  </style>
</head>
<body>
  __BACKDROP__
  <div class="demo-badge">
    Chatbot demo by <a href="https://kaval.ai" target="_blank" rel="noopener">Kaval.AI</a>
    — built from <a href="__SITE_URL__" target="_blank" rel="noopener">this site</a>'s
    public pages. Nothing was installed.
  </div>

  <script src="kaval-chatbot.js"></script>
  <script>
    var DEMO = __CONFIG_JSON__;

    /* Lexical passage retrieval: enough to demonstrate the widget answering
       from the site's own content with source links, with no backend at all.
       An agent-server endpoint in the config replaces it with the real bot. */
    function retrievalConnector(chunks) {
      var docFreq = {};
      var indexed = chunks.map(function (chunk) {
        var terms = tokenize(chunk.text);
        var seen = {};
        terms.forEach(function (t) { seen[t] = true; });
        Object.keys(seen).forEach(function (t) { docFreq[t] = (docFreq[t] || 0) + 1; });
        return { chunk: chunk, terms: terms, length: terms.length || 1 };
      });

      function tokenize(text) {
        return (text.toLowerCase().match(/[a-z0-9]{2,}/g) || []);
      }

      function score(queryTerms, doc) {
        var counts = {};
        doc.terms.forEach(function (t) { counts[t] = (counts[t] || 0) + 1; });
        var total = 0;
        queryTerms.forEach(function (t) {
          var tf = counts[t] || 0;
          if (!tf) { return; }
          var idf = Math.log(1 + indexed.length / (docFreq[t] || 1));
          total += idf * tf / (tf + 1.2 * (0.25 + 0.75 * doc.length / 120));
        });
        return total;
      }

      function snippet(text) {
        var body = text.split("\\n\\n").slice(1).join(" ").replace(/\\s+/g, " ").trim() || text;
        return body.length > 320 ? body.slice(0, 320) + "…" : body;
      }

      return {
        send: function (text) {
          var queryTerms = tokenize(text);
          var ranked = indexed
            .map(function (doc) { return { doc: doc, score: score(queryTerms, doc) }; })
            .filter(function (hit) { return hit.score > 0; })
            .sort(function (a, b) { return b.score - a.score; })
            .slice(0, 3);

          if (ranked.length === 0) {
            return Promise.resolve({
              text: "I could not find anything about that on this site. Try one of the suggestions below.",
              choices: DEMO.suggestions,
            });
          }
          var lines = ["Here is what this site says about that:", ""];
          var seenUrls = {};
          ranked.forEach(function (hit) {
            var c = hit.doc.chunk;
            var label = c.heading || c.title || c.url;
            lines.push("**" + label + "** — " + snippet(c.text));
            if (!seenUrls[c.url]) {
              lines.push("[Read more](" + c.url + ")");
              seenUrls[c.url] = true;
            }
            lines.push("");
          });
          lines.push("*This preview quotes the site directly; the full product answers in natural language.*");
          return Promise.resolve({ text: lines.join("\\n"), choices: [] });
        },
        reset: function () {},
      };
    }

    function start(connector) {
      KavalChatbot.mount({
        connector: connector,
        mode: "floating",
        title: DEMO.title,
        greeting: DEMO.greeting,
        emptyMessage: DEMO.emptyMessage,
        suggestions: DEMO.suggestions,
        theme: DEMO.theme,
      });
    }

    if (DEMO.endpoint) {
      start(KavalChatbot.agentConnector({ url: DEMO.endpoint }));
    } else {
      fetch("chunks.json")
        .then(function (r) { return r.json(); })
        .then(function (chunks) { start(retrievalConnector(chunks)); });
    }
  </script>
</body>
</html>
"""


def render_index(site: SitePages, backdrop: str, config: dict) -> str:
    """The demo's index.html."""
    if backdrop == "screenshot":
        backdrop_html = '<img class="demo-backdrop" src="screenshot.png" alt="" />'
    elif backdrop == "html":
        backdrop_html = '<iframe class="demo-backdrop" src="original.html" title="Original site"></iframe>'
    else:
        backdrop_html = ""
    return (
        INDEX_TEMPLATE.replace("__TITLE__", site.title)
        .replace("__SITE_URL__", site.site_url)
        .replace("__BACKDROP__", backdrop_html)
        .replace("__CONFIG_JSON__", json.dumps(config, indent=2))
    )


def default_out_dir(pages_path: str, site: SitePages) -> str:
    """``<host>.demo`` next to the pages database."""
    host = urlparse(site.site_url).netloc
    return os.path.join(os.path.dirname(pages_path) or ".", f"{host}.demo")


async def compile_demo(
    pages_path: str,
    out_dir: Optional[str] = None,
    *,
    index_path: Optional[str] = None,
    collection: Optional[str] = None,
    endpoint: Optional[str] = None,
    title: Optional[str] = None,
    suggestions: Optional[list[str]] = None,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> DemoReport:
    """Assemble the demo folder and return what was built."""
    site = read_site(pages_path)
    out_dir = out_dir or default_out_dir(pages_path, site)
    os.makedirs(out_dir, exist_ok=True)

    for name in WIDGET_FILES:
        shutil.copy(os.path.join(WIDGET_DIR, name), os.path.join(out_dir, name))

    if site.screenshot:
        with open(os.path.join(out_dir, "screenshot.png"), "wb") as f:
            f.write(site.screenshot)
        backdrop = "screenshot"
    elif site.homepage_html:
        with open(os.path.join(out_dir, "original.html"), "w", encoding="utf-8") as f:
            f.write(inject_base_tag(site.homepage_html, site.site_url))
        backdrop = "html"
    else:
        backdrop = "none"
        logger.warning(
            "No screenshot or stored HTML for the start page; plain backdrop"
        )

    chunk_count = 0
    if not endpoint:
        index_path = index_path or default_index_path(pages_path)
        missing_file = "://" not in index_path and not os.path.exists(index_path)
        if missing_file:
            logger.info(f"No RAG index at {index_path}; chunking the stored markdown")
            chunks = chunks_from_pages(site)
        else:
            chunks = await chunks_from_index(index_path, collection)
        chunks = chunks[:max_chunks]
        chunk_count = len(chunks)
        if not chunk_count:
            raise ValueError("No content to answer from; index or re-scrape first")
        with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    host = urlparse(site.site_url).netloc
    config = {
        "title": title or host,
        "greeting": "Hello! Ask me about this site.",
        "emptyMessage": f"Hi! Ask me anything about {host}.",
        "suggestions": suggestions or [],
        "theme": {},
        "endpoint": endpoint,
    }
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_index(site, backdrop, config))

    logger.info(
        f"Demo compiled to {out_dir}: backdrop={backdrop},"
        + (f" chunks={chunk_count}" if not endpoint else f" endpoint={endpoint}")
        + f". Serve with: python -m http.server -d {out_dir}"
    )
    return DemoReport(
        out_dir=out_dir, site_url=site.site_url, backdrop=backdrop, chunks=chunk_count
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
            "RAG index whose chunks feed the client-side retrieval preview, as"
            " a database URI or SQLite path (default: the <site>.rag.db beside"
            " the input; a missing file falls back to re-chunked markdown)"
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection inside --index (default: its only collection)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        metavar="URL",
        help=(
            "Agent-server stream endpoint (e.g. http://host:25000/api/chat/"
            "stream_agent); replaces the retrieval preview with the real bot"
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
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=f"Cap on chunks shipped to the browser (default: {DEFAULT_MAX_CHUNKS})",
    )
    return parser


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    if not os.path.exists(args.pages):
        logger.error(f"Pages database not found: {args.pages}")
        return 2
    try:
        asyncio.run(
            compile_demo(
                args.pages,
                args.out,
                index_path=args.index,
                collection=args.collection,
                endpoint=args.endpoint,
                title=args.title,
                suggestions=args.suggestion,
                max_chunks=args.max_chunks,
            )
        )
    except (ValueError, ImportError) as error:
        logger.error(error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
