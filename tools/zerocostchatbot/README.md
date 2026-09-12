# Zero-cost chatbot toolkit

Turns a website into a chatbot demo in three steps, each a standalone CLI
run from the repository root:

1. `scrape.py` crawls the site into a **pages database** — one SQLite file
   holding URL, HTTP status, raw HTML and extracted markdown per page.
2. `build_index.py` chunks and embeds those pages into a **RAG index** — a
   `CollectionRagService` collection, by default a second SQLite file.
3. `make_demo.py` compiles both into a **static demo folder** that browses
   the scraped site with the chat widget answering from it, entirely in
   the browser.

The pages database is the crawl artefact and never leaves your infra; the
index is small, servable, and rebuildable with another chunker or embedding
model without re-crawling.

```bash
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 100
python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db
python -m tools.zerocostchatbot.make_demo docs.kaval.ai.pages.db
python -m http.server -d docs.kaval.ai.demo
```

## Scraping

```bash
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 50
```

The database defaults to `<host>.pages.db`; `--pages` names it explicitly.
Re-running the same command **resumes**: every completed page is committed
together with the links it revealed, so a run killed at any moment loses at
most the page in flight. `--refresh` re-queues every known URL for a fresh
pass.

Each page is fetched with a plain HTTP request first and reduced to text
and markdown by `kavalai.text.parse_html` (standard library only;
navigation, footer and sidebar text is dropped, their links kept for
discovery). The fetch escalates to a
headless browser only when the response is not usable content — a JS-app
shell, too little visible text, or a 403/503 bot challenge — and three
escalations in a row memoise the site as JS-rendered. Rendering happens in
the crawl4ai container from `docker-compose.yml` (`docker compose up
crawl4ai`), which stores the rendered DOM, stylesheets and scripts
included; a server-side-rendered site never contacts it.

`robots.txt` is honoured by default and matched with the identity the
requests carry (`KavalaiBot`, or `Googlebot` for the `googlebot` preset).
Discovery seeds from `sitemap.xml` and continues over same-site links.

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
| `--crawl4ai-url URL` | The rendering container (default `http://localhost:11235`) |
| `--force-http` / `--force-browser` | Pin the fetch mode |
| `--refresh` | Re-queue every known URL before crawling |
| `--screenshot` | Store a PNG of the start page on its row (needs the container) |

The database is a plain SQLite file with one `pages` table, created by
`pages_db.py` on first use: the URL, discovery and crawl timestamps, status
code, fetch mode (`http` or `browser`), attempt count and last error,
title, raw HTML, markdown and the optional screenshot. `last_crawled_at`
is NULL while a URL is still pending, so that column *is* the work queue.

```python
from tools.zerocostchatbot.pages_db import PagesDatabase

with PagesDatabase("docs.kaval.ai.pages.db") as db:
    print(db.stats())
    for row in db.iter_pages():
        print(row.url, row.status_code, row.fetch_mode, row.title)
```

## Building the RAG index

```bash
python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db
```

The index lands beside the input (`docs.kaval.ai.rag.db`), the collection
is named after the host, and the embedding model is local fastembed
(`fastembed/snowflake/snowflake-arctic-embed-s`) — no API key, no per-page
cost, and the model family the browser demo embeds queries with. It needs
`kavalai[common]`. `--model` selects any registered embedding provider,
`--index` another location as a database URI or SQLite path (a Postgres
index is browsable in the backoffice RAG explorer, but only a SQLite one
can ship in a demo).

The collection is **dropped and rebuilt** each run. Markdown is chunked at
heading boundaries (`--max-chars` splits long sections), each chunk
prefixed with the page title and heading path, and chunk metadata carries
`url`, `title`, `heading` and `crawled_at`. `examples/ragindex/
query_index.py` queries the result from the terminal.

## Compiling a demo

```bash
python -m tools.zerocostchatbot.make_demo docs.kaval.ai.pages.db
python -m http.server -d docs.kaval.ai.demo
```

The output folder (`<host>.demo`) is self-contained static files, servable
from any bucket: `index.html`, the viewer script, the production chat
widget (`kavalai/widget/`), the pages database and the RAG index. Its
`index.html` is `archive.html` with the two database files named in
`window.KavalArchiveDefaults`, so it opens straight into the site.
`--index` names another SQLite index, `--collection` one inside it,
`--model` the WebLLM chat model, `--title` the chat window and
`--suggestion` adds question chips. `--endpoint URL` points the widget at
a running agent server instead of an in-browser model, and ships no index.

## The archive viewer

`archive.html` is the page the demo compiles: a Wayback-style viewer of the
scraped site with the chat widget answering from its RAG index. It also
runs from the repository:

```bash
python -m http.server            # from the repo root
# open http://localhost:8000/tools/zerocostchatbot/archive.html?db=/docs.kaval.ai.pages.db
```

`?db=` names the pages database (sql.js loads it in the browser), `?rag=`
the index (default: the `.rag.db` beside it), `?model=` a WebLLM chat
model. Without `?db=` the page offers file pickers, which also work from
`file://`.

**Browsing.** The sidebar lists every fetched page with a filter; the
address bar, back/forward and the links inside the pages navigate the
archive, and a link whose target was not crawled reports "Not in the
archive". Styles and images load from the live site through an injected
`<base>` tag; the site's own scripts are stripped unless **Run page
scripts** is ticked.

**Chatting.** In a WebGPU browser the widget runs WebLLM: the embedding
model matching `build_index.py`'s default and a chat model (Qwen2.5-1.5B by
default) load together, each question is embedded, matched against the
index by cosine and answered from the passages with a source list. Without
WebGPU, or if the model fails to load, the widget quotes the best-matching
passages instead. The header's skin picker switches the widget's theme.

The DOM-free logic (URL matching, page preparation, the index and ranking,
prompt building) lives in `archive.js`, tested with
`node --test tools/zerocostchatbot/tests/archive.test.js`.

## Tests

`pytest tools/zerocostchatbot` runs the Python tests beside the code; they
need no network, no crawl4ai and no embedding model.
