# Zero-cost chatbot toolkit

Turns a website into a ready-to-serve RAG index in two steps, each a
standalone CLI over a SQLite file (see `chatbotplan.md` at the repository
root for the product plan):

1. `scrape.py` crawls the site into a **pages database** — URL, HTTP
   status, raw HTML and extracted markdown per page.
2. `build_index.py` chunks and embeds those pages into a **RAG index** —
   a `CollectionRagService` collection, by default in its own SQLite file.

Content and index are separate on purpose: the pages database is the
crawl artefact — one self-contained file holding everything the scrape
produced (raw HTML, markdown, the start page's screenshot) that never
leaves your infra — and the RAG index is the small servable file,
rebuildable at any time with a different chunker or embedding model
without re-crawling.

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

### More examples

Re-scrape a site you crawled before and capture a homepage screenshot on
the way (the crawl4ai container must be running for the screenshot):

```bash
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai \
    --refresh --screenshot
```

Crawl a site you own as fast as the server allows — no politeness delay,
eight parallel workers:

```bash
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai \
    --delay 0 --concurrency 8 --max-pages 500
```

A JavaScript-rendered site, skipping the doomed HTTP attempts outright:

```bash
python -m tools.zerocostchatbot.scrape https://app.example.com --force-browser
```

A site whose robots.txt allows only Google (see the user-agent notes
below):

```bash
python -m tools.zerocostchatbot.scrape https://example.com --user-agent googlebot
```

Scrape and build the RAG index in sequence, then try a query against it:

```bash
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 100
python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db
python -m examples.ragindex.query_index "how do I install kavalai" \
    --index docs.kaval.ai.rag.db --collection docs.kaval.ai \
    --model fastembed/snowflake/snowflake-arctic-embed-s
```

Rebuild the same index into Postgres (read from `KAVALAI_DB_URI`), where
the backoffice RAG explorer can browse it:

```bash
dotenv run python -m tools.zerocostchatbot.build_index \
    docs.kaval.ai.pages.db --index postgres
```

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
    markdown TEXT,
    screenshot BLOB
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
| `screenshot` | PNG capture of the start page when `--screenshot` took one; NULL elsewhere |

## Fetch strategy

Each page is fetched with a plain HTTP request first; the HTML is reduced
to markdown by `htmlmd.py` (stdlib only — navigation, footer and sidebar
text is dropped, their links are kept for discovery). The fetch escalates
to a headless browser only when the response is not usable content: a
JS-app shell, too little visible text, or a 403/503 bot challenge. Three
consecutive escalations memoise the site as JS-rendered and later pages
skip the doomed HTTP attempt.

Rendering always happens in the crawl4ai REST container from
`docker-compose.yml` (`docker compose up crawl4ai`); `--crawl4ai-url`
points elsewhere when it is not on `http://localhost:11235`. Nothing
browser-related is installed on this machine, and a server-side-rendered
site never contacts the container at all. Fetches run on `--concurrency`
parallel workers over one shared HTTP client — the container's browser
pool renders in parallel server-side — while `--delay` stays a
*site-wide* rate cap: parallelism overlaps the waiting (server latency,
rendering), not the request rate, so raising it never makes the crawl
less polite.

## Options

| Flag | Effect |
|------|--------|
| `--pages PATH` | Pages database file (default `<host>.pages.db`) |
| `--max-pages N` | Stop once N pages hold content, resume included (default 200) |
| `--delay S` | Minimum seconds between request starts, site-wide (default 0.5) |
| `--concurrency N` | Pages fetched in parallel (default 4) |
| `--timeout S` | Per-request timeout (default 30) |
| `--max-attempts N` | Park a URL after N failed fetches (default 3) |
| `--user-agent X` | `kavalai` (default), `browser`, `googlebot`, or a verbatim string |
| `--ignore-robots` | Do not honour the site's robots.txt |
| `--crawl4ai-url URL` | The crawl4ai container rendering pages (default `http://localhost:11235`) |
| `--force-http` / `--force-browser` | Pin the fetch mode |
| `--refresh` | Re-queue every known URL before crawling |
| `--screenshot` | Store a PNG of the start page on its row (needs the container) |

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
embedding model is local fastembed (`fastembed/snowflake/
snowflake-arctic-embed-s`) — no API key, no per-page cost, and the same
model family the browser widget embeds queries with, so the index stays
usable from a fully client-side demo; it needs `kavalai[common]` (or
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

## Compiling a client demo

```bash
python -m tools.zerocostchatbot.make_demo docs.kaval.ai.pages.db
python -m http.server -d docs.kaval.ai.demo
```

The output folder is self-contained static files (bucket-servable): the
homepage screenshot — or the stored HTML anchored to the live site — as the
backdrop, with the production chat widget (`chatbotwidget/`) floating over
it and a "demo by Kaval.AI" badge. By default the chat answers client-side
from the site's own chunks (the RAG index beside the pages database, else
re-chunked markdown) with a lexical ranker and source links — no backend,
no keys. `--endpoint URL` points the widget at a running agent server for
the full LLM-backed bot instead; `--suggestion` adds question chips,
`--title` names the window.

## The demo page: browsing the archive with the chatbot

`archive.html` is the client-facing demo — a Wayback-style viewer of the
scraped site with the chat widget answering from its RAG index, entirely
in the browser:

```bash
python -m http.server            # from the repo root
# open http://localhost:8000/tools/zerocostchatbot/archive.html?db=/docs.kaval.ai.pages.db
```

`?db=` names the pages database (sql.js loads the SQLite file in the
browser); the RAG index is taken from `?rag=`, defaulting to the
`.rag.db` beside it. Without `?db=` the page offers file pickers for both,
which also work from `file://`. The header reads "DEMO chatbot" so the
page is not mistaken for the live site.

**Browsing.** The collapsible sidebar (☰) lists every fetched page with a
filter; the address bar, back/forward and the links inside the pages
navigate the archive — links whose target was not crawled are shown dotted
and, when clicked, report "Not in the archive" instead of leaving. Styles
and images load from the live site through an injected `<base>` tag; the
site's own scripts are stripped unless **Run page scripts** is ticked, so
a snapshot cannot phone home or navigate on its own.

**Chatting.** When a RAG index loads, the production widget
(`chatbotwidget/`) opens over the page. In a WebGPU browser it runs
WebLLM: the embedding model matching `build_index.py`'s default
(snowflake-arctic-embed-s) and a chat model (`?model=`, or the header
picker; Qwen2.5-1.5B by default) load together, with download progress
shown in the widget's header, and each question is embedded, matched
against the index by cosine and answered from the passages with a source
list. Without WebGPU — or if the model fails to load — the widget quotes
the best-matching passages instead (a lexical ranking), so the demo still
answers. The header's skin picker switches the widget between a few
themes at runtime.

The DOM-free logic (URL matching, page preparation, the RAG index and
ranking, prompt building) lives in `archive.js`, tested with
`node --test tools/zerocostchatbot/tests/archive.test.js`.

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
