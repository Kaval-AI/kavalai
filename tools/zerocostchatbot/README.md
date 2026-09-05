# Zero-cost chatbot toolkit

Turns a website into a ready-to-serve RAG index in two steps, each a
standalone CLI over a SQLite file (see `chatbotplan.md` at the repository
root for the product plan):

1. `scrape.py` crawls the site into a **pages database** — URL, HTTP
   status, raw HTML and extracted markdown per page.
2. `build_index.py` chunks and embeds those pages into a **RAG index** —
   a `CollectionRagService` collection, by default in its own SQLite file.

Content and index are separate files on purpose: the pages database is the
heavy crawl artefact that never leaves your infra, the RAG index is the
small servable one (a 4.6 MB crawl of docs.kaval.ai yields a 0.7 MB
index), rebuildable at any time with a different chunker or embedding
model without re-crawling.

## Usage

```bash
# From the repository root; httpx is the only required dependency.
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 50
```

The database defaults to `<host>.pages.db` — the command above writes
`docs.kaval.ai.pages.db` — and `--pages` names it explicitly. Re-running
the same command **resumes**: the database is the crawl state, every completed page is
committed in one transaction together with the links it revealed, and a
run killed at any moment loses at most the page that was in flight.
`--refresh` re-queues every known URL for a fresh pass (stored content is
kept until the page is fetched again).

## Schema

The pages database is a plain SQLite file (open it with any SQLite tool);
`pages_db.py` creates this on first use:

```sql
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    discovered_at TEXT NOT NULL,
    last_crawled_at TEXT,
    status_code INTEGER,
    fetch_mode TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    fetch_error TEXT,
    title TEXT,
    html TEXT,
    markdown TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_pending
    ON pages (last_crawled_at, attempts);
```

| Column | Meaning |
|--------|---------|
| `url` | Normalised absolute URL; the primary key is what deduplicates discovery |
| `discovered_at` | When the URL entered the frontier (ISO-8601 UTC, like all timestamps here) |
| `last_crawled_at` | NULL while the URL is still pending — this column *is* the work queue |
| `status_code` | Last HTTP status seen, kept for failures too |
| `fetch_mode` | `http` or `browser` — which path produced the stored content |
| `attempts` | Fetch attempts so far; rows at `--max-attempts` are parked |
| `fetch_error` | Last error message, or the skip reason (`disallowed by robots.txt`); NULL after a success |
| `title` | Page title |
| `html` | Raw HTML as fetched (browser path: rendered DOM) |
| `markdown` | Extracted markdown, the input for the indexing step |

## Fetch strategy

Each page is fetched with a plain HTTP request first; the HTML is reduced
to markdown by `htmlmd.py` (stdlib only — navigation, footer and sidebar
text is dropped, their links are kept for discovery). The fetch escalates
to crawl4ai's headless browser only when the response is not usable
content: a JS-app shell, too little visible text, or a 403/503 bot
challenge. Three consecutive escalations memoise the site as JS-rendered
and later pages skip the doomed HTTP attempt. A server-side-rendered site
therefore needs no browser stack at all; the browser paths need
`kavalai[common]` (crawl4ai) installed.

## Options

| Flag | Effect |
|------|--------|
| `--pages PATH` | Pages database file (default `<host>.pages.db`) |
| `--max-pages N` | Stop once N pages hold content, resume included (default 200) |
| `--delay S` | Seconds between fetches (default 0.5) |
| `--timeout S` | Per-request timeout (default 30) |
| `--max-attempts N` | Park a URL after N failed fetches (default 3) |
| `--user-agent X` | `kavalai` (default), `browser`, `googlebot`, or a verbatim string |
| `--ignore-robots` | Do not honour the site's robots.txt |
| `--force-http` / `--force-browser` | Pin the fetch mode |
| `--refresh` | Re-queue every known URL before crawling |
| `--screenshot PATH` | Save a PNG of the start page (browser stack required) |

`robots.txt` is honoured by default and matched with the same identity the
requests carry: `KavalaiBot` for the default agent, `Googlebot` for the
`googlebot` preset (for sites whose robots.txt allows only Google), `*`
otherwise. URL discovery seeds from `sitemap.xml` (one level of sitemap
index) and continues over same-site links; assets and other file
extensions are skipped.

## Building the RAG index

```bash
python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db
```

Defaults: the index lands beside the input (`docs.kaval.ai.rag.db`), the
collection is named after the scraped host (`docs.kaval.ai`), and the
embedding model is local fastembed (`fastembed/BAAI/bge-small-en-v1.5`) —
no API key, no per-page cost; it needs `kavalai[common]` (or
`pip install fastembed sqliteai-vector`). `--model` selects any registered
embedding provider instead, `--index` a different backend (`postgres`
reads `KAVALAI_DB_URI`/`KAVALAI_DB_SCHEMA`, a `...://...` URI is used
verbatim, anything else is a SQLite file path — the same contract as
`examples/ragindex`).

The collection is **dropped and rebuilt** each run, so the index always
mirrors the pages database. Markdown is chunked at heading boundaries
(`--max-chars` splits long sections), each chunk prefixed with the page
title and heading path ("Kaval AI docs › Quickstart › Install"), and
chunk metadata carries `url`, `title`, `heading` and `crawled_at` — enough
to cite sources. The result is queryable with
`examples/ragindex/query_index.py` and browsable in the backoffice RAG
explorer.

## Inspecting a pages database

```python
from tools.zerocostchatbot.pages_db import PagesDatabase

with PagesDatabase("docs.kaval.ai.pages.db") as db:
    print(db.stats())
    for row in db.iter_pages():
        print(row.url, row.status_code, row.fetch_mode, row.title)
```

Tests live in `tests/` beside the code (`pytest tools/zerocostchatbot`) and need no
network, no crawl4ai and no embedding model.
