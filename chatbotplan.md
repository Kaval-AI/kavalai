# Website-to-Chatbot Product Plan

Goal: a prospect gives us their website URL, we crawl it, build a RAG index,
and hand back a ready-to-embed chatbot — with as little friction as possible
on their side. Ideally the trial loop is: *paste URL → wait a few minutes →
paste one `<script>` tag into their site*.

The pipeline has three stages, and we already own most of the parts:

| Stage | What exists today | What is missing |
|-------|-------------------|-----------------|
| Crawl | `crawl_url` / `web_search` in `kavalai/tools/webtools/crawl4ai.py`; the `crawl4ai` docker service (REST on 11235) | Site-wide crawling (link following, sitemap, dedup, politeness) |
| Index | `CollectionRagService` (Postgres + SQLite backends), `examples/ragindex/index_csv.py`, backoffice RAG explorer | Markdown-aware chunking; a crawl→index CLI; per-client collection management |
| Serve | `WorkflowEngine` + `rag_query` node, agent-server docker image, `webwidget/` (`KavalChat`, Pyodide+WebLLM bridge), eval framework | A support-bot workflow template; the embed snippet product; per-client provisioning |

---

## 1. Crawling the website

### Option A — crawl4ai deep crawl (recommended)

We already depend on crawl4ai for `crawl_url`, and `docker-compose.yml`
already ships the `unclecode/crawl4ai` REST service. crawl4ai ≥0.5 has a
built-in deep-crawl mode (BFS/DFS with depth limits, URL filters, scoring)
and returns **clean markdown per page** — exactly the input a RAG chunker
wants. It renders JavaScript via Playwright, so SPA marketing sites work.

- **Pros**: already integrated and dockerised; JS rendering; markdown output
  (no HTML-cleaning stage to build); built-in link filtering/dedup; one
  stack for the crawler tool the agents use and the indexing crawler.
- **Cons**: heavy (Playwright/Chromium) — not something a client runs
  locally, so crawling is a *hosted* step; slower per page than a plain
  HTTP fetcher; deep-crawl config needs tuning (depth, `robots.txt`,
  rate limits) which we must wrap.

### Option B — Scrapy

- **Pros**: mature, fast, battle-tested politeness (auto-throttle,
  robots.txt), great for very large sites, low memory.
- **Cons**: no JS rendering without bolting on Splash/Playwright (at which
  point we have rebuilt option A); Twisted's reactor fights our
  asyncio/FastAPI world; output is raw HTML — we would still need a
  readability/markdown extraction stage; a second framework dependency the
  team must learn. Overkill for the target workload (typical customer site:
  50–2 000 pages, crawled once + periodic refresh).

### Option C — Own minimal crawler (httpx + selectolax/BeautifulSoup)

Fetch `sitemap.xml` (most business sites have one), fall back to BFS over
same-domain links, extract main content with `trafilatura` or `readability`.

- **Pros**: tiny, fully async, no browser, trivially embeddable in the
  library (and pyodide-compatible in principle); easiest to unit-test.
- **Cons**: no JS rendering (fails on JS-heavy sites — precisely the modern
  marketing sites our prospects have); we own edge cases forever (encoding,
  redirects, canonical URLs, traps); content extraction quality is the whole
  product and generic extractors are mediocre.

### Option D — Hosted crawl API (Firecrawl, Jina Reader, Apify)

- **Pros**: zero crawl infrastructure; best-in-class extraction; fastest to
  a demo.
- **Cons**: per-page cost on every trial signup; client data flows through a
  third party (a sales objection for the self-hosted pitch); an external
  dependency in the core funnel; we already run crawl4ai, so it buys little.

### Recommendation

**A with a slice of C**: sitemap-first seeding (cheap, gives page count and
`lastmod` for refreshes), then crawl4ai deep-crawl for fetching/rendering/
markdown. Wrap it as a `kavalai-crawl` console script + a `crawl_site`
function so the same code serves the CLI, the backoffice "Add website"
button, and tests (which mock the crawl4ai endpoint). Store per-page
markdown + URL + title + `lastmod` — the URL becomes the citation link in
answers, `lastmod` drives incremental re-crawls.

---

## 2. Building the RAG index

The storage question is already settled by the codebase:
`CollectionRagService` with one collection per client site, on either
backend — **Postgres/pgvector** for the hosted product, a **SQLite vector
file** for anything portable. That dual backend is a real product asset:
*the index is a single `.db` file we can ship* — to a client's own
deployment, or (in principle) to the browser.

What needs building:

1. **Chunking** — crawl4ai gives markdown, so chunk on heading boundaries
   (H1/H2/H3) with a token cap and small overlap; prepend the page title +
   heading path to each chunk ("Acme › Pricing › Enterprise") so retrieval
   works on short chunks. Keep `url`, `title`, `heading`, `crawl_ts` in
   chunk metadata → citations for free, and the collection is browsable in
   the backoffice RAG explorer.
2. **Embeddings** — two profiles, both already supported by the registry:
   - `fastembed` local models: no API key, no per-trial cost, and the same
     model family has WebLLM-side equivalents (`KAVAL_BROWSER_EMBED_MODEL`)
     for the in-browser option. Default for trials.
   - Provider embeddings (OpenAI/Gemini) as a paid-tier upgrade.
   The embedding model name must be pinned in collection metadata — query
   and index must embed with the same model.
3. **Pipeline CLI** — `kavalai-crawl <url> --index sqlite:///acme.db`
   mirroring `examples/ragindex/index_csv.py`: crawl → chunk → embed →
   upsert, idempotent per URL (re-crawl replaces a page's chunks). Refresh =
   re-run; sitemap `lastmod` limits it to changed pages.
4. **Quality gate** — generate a small eval suite per site (a handful of
   judged cases from the indexed content) and run it with `kavalai-eval`
   before handing the bot over; "the bot passed N/N checks on your own
   content" is also a good onboarding screen.

No new framework needed (no LlamaIndex/LangChain): chunking is ~100 lines,
and everything below it exists.

---

## 3. Deployment — minimal client setup

### Option A — JS snippet / iframe onto a hosted agent-server (recommended for the trial)

We host one agent-server (docker image exists) running a support-bot
workflow template (`rag_query` → agent → answer with citations), one RAG
collection per client. The client pastes:

```html
<script src="https://cdn.kaval.ai/kaval-chat.js" data-bot="acme-a1b2c3"></script>
```

`KavalChat` is deliberately backend-agnostic (`send` callback), so pointing
it at the agent-server's REST/SSE endpoint is a small loader script, not new
widget work. An `<iframe>` variant is the fallback for CSP-restricted sites
and for "preview your bot" on *our* site before they embed anything.

- **Pros**: friction is one copy-paste; works on every browser and CMS; we
  control model choice, upgrades, logging (sessions land in the agent DB →
  backoffice analytics, a retention hook); instant "try it on our page
  first" demo without touching their site at all.
- **Cons**: we run infrastructure and pay for inference on trials (cap:
  small model + message limits); needs multi-tenant hardening on the agent
  server (per-bot auth token, rate limiting, CORS allowlist per customer
  domain).

### Option B — Pyodide + WebLLM, fully client-side

The `webwidget/` stack already does this end-to-end: kavalai wheel in
Pyodide, `browser/…` models over WebGPU, in-browser SQLite for history, and
a browser embedding model for query-side vectors. The missing piece is
loading a pre-built index in the browser (ship the SQLite vector file, or a
JSON chunk dump with brute-force cosine — fine at a few thousand chunks).

- **Pros**: zero inference cost to us — trials scale for free; total
  privacy (nothing leaves the visitor's machine) — a differentiator no
  competitor snippet offers; already ~80 % built.
- **Cons**: WebGPU-capable browsers only; first load downloads a
  multi-hundred-MB model (acceptable on a demo page, hostile on a
  customer's live site); small in-browser models answer noticeably worse;
  index must be public (it is downloaded by every visitor).
- **Verdict**: not the primary product, but a superb *lead magnet*: "watch
  your website answer questions, running entirely in your browser" on our
  landing page, with zero marginal cost per trial. Conversion path: "want
  this quality-of-answer on every browser? — flip to hosted."

### Option C — WordPress plugin

- **Pros**: ~43 % of the web; a plugin-directory listing is a discovery
  channel; "Install plugin → paste bot ID" is even lower friction for
  non-technical owners; WP auth could later push private content (docs
  behind login) into the index.
- **Cons**: it is only a wrapper around option A's snippet — no new
  capability; PHP maintenance + plugin-review process; only worth it once
  the snippet works.
- **Verdict**: phase 3 distribution play, not architecture.

### Option D — Self-hosted docker (existing images)

`docker compose up` with agent-server + migrations + a shipped SQLite index
file. Not a trial channel — it is the *upsell* for the privacy-sensitive
client, and it costs us nothing because the images exist.

### Recommendation

**A is the product, B is the demo, C and D are channels.** All four serve
the same workflow template and the same index format, so nothing is built
twice.

---

## Iterations

### Iteration 1 — the database-preparation tool (`tools/zerocostchatbot/`)

Two CLI scripts and an intermediate **pages database**, so scraping and
indexing are separate, composable steps:

```bash
# 1. Scrape: crawl the site into a pages database (url + raw HTML +
#    markdown + the start page's screenshot, all in one SQLite file)
#    BUILT — tools/zerocostchatbot/ (scrape.py, pages_db.py, htmlmd.py + tests)
python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 200

# 2. Index: build the RAG collection from the pages database
#    BUILT — tools/zerocostchatbot/build_index.py (+ tests); index path,
#    collection name (site host) and fastembed model are defaulted
python -m tools.zerocostchatbot.build_index docs.kaval.ai.pages.db
```

**Two SQLite files, deliberately.** The pages database is the crawl
artefact — one self-contained file with the raw HTML, the markdown and
the start page's screenshot — and never leaves our infra. The RAG index is
the servable artefact: small, shippable (including to a
browser later), rebuildable from the pages database at any time with a
different chunker or embedding model, without re-crawling.

#### The pages database is the crawl state

One `pages` table serves as both the URL frontier and the content store,
which is what makes the scraper kill-safe:

| Column | Meaning |
|--------|---------|
| `url` (PK) | Normalised URL, deduplicated on insert |
| `discovered_at` | When the URL entered the frontier |
| `last_crawled_at` | NULL = still pending — this *is* the work queue |
| `status_code` | Last HTTP status (also kept for failures) |
| `fetch_mode` | `http` or `browser` — how the content was obtained |
| `attempts` | Fetch attempts so far, capped to avoid poison URLs |
| `fetch_error` | Last error message, NULL on success |
| `title`, `markdown` | The content (NULL until fetched) |
| `html`, `screenshot` | Raw HTML per page; the start page's PNG capture as a BLOB |

Crawl loop: seed the start URL + sitemap URLs with `INSERT OR IGNORE`,
then let a small pool of parallel workers each pick a pending row, fetch
it, and in **one transaction per page** write the result and
`INSERT OR IGNORE` the newly discovered links; the fetch *rate* stays
capped site-wide by the politeness delay, parallelism only overlapping the
waiting.
Killed at any moment, the database is consistent and a restart resumes
from the pending rows — at worst one page is fetched twice. A completed
run (`--refresh` re-queues pages older than a given age) is how re-crawls
work; sitemap `lastmod` can narrow that later.

#### HTTP first, browser only when needed

Most business sites are server-side rendered, and a plain `httpx` GET is
10–50× cheaper than a Playwright page load. Per page:

1. Fetch with `httpx`. Convert the HTML to markdown without any browser.
2. Escalate to the headless browser (crawl4ai) when the cheap result is
   not usable content: the visible text (scripts/styles stripped) is
   below a threshold, the body is a JS-app shell (an empty `root`/
   `__next`-style mount node, a `<noscript>` "enable JavaScript" notice),
   or the response is a bot challenge (403/503 with challenge markers).
3. Record which mode succeeded in `fetch_mode`; once a few pages in a row
   have escalated, the site is treated as JS-rendered and later pages go
   straight to the browser (and the reverse keeps HTTP-only sites away
   from Playwright entirely). `--force-http` / `--force-browser`
   override the heuristic. The browser side always runs in the
   docker-compose `crawl4ai` REST container (`--crawl4ai-url`, default
   `http://localhost:11235`) — no Playwright on the scraping machine.

This also keeps the dependency story clean: an SSR site can be scraped
with the base install alone; crawl4ai/Playwright is only exercised when a
site actually needs rendering.

#### User agent presets

`--user-agent` takes a preset or a verbatim string, applied to both fetch
paths (the `httpx` headers and the crawl4ai browser config) **and** to the
`robots.txt` evaluation — the rules are matched against the same agent
token the requests carry:

- `kavalai` (default): `KavalaiBot/1.0 (+https://kaval.ai/bot)` — honest
  self-identification, matched as `KavalaiBot` in robots.txt.
- `browser`: a current Chrome user-agent string, matched as `*`.
- `googlebot`: Googlebot's own string, matched as `Googlebot` — for sites
  whose robots.txt allows only Google without the owner knowing it.
- Anything else is used verbatim (matched as `*`).

Caveat worth remembering: presenting as Googlebot is spoofing — acceptable
for a crawl the site owner asked for, and some sites verify Googlebot by
reverse DNS anyway, so the preset may still get challenged or served
different content.

#### Indexing (`build_index.py`)

- Heading-based markdown chunking (page title + heading path prefixed to
  each chunk), embedded with the given model — by default local fastembed
  `snowflake/snowflake-arctic-embed-s`, no API key, and the same model
  family the browser widget embeds queries with, keeping the index usable
  from a fully client-side demo — into a `CollectionRagService` collection.
- `--index` accepts a SQLite file path, a database URI, or `postgres`
  (from `KAVALAI_DB_URI`/`KAVALAI_DB_SCHEMA`), same as `examples/ragindex`.
- The collection is **dropped and rebuilt** each run: the pages database is
  the source of truth, chunk counts shift between crawls, and a full
  rebuild of a small site takes seconds — incremental patching is not
  worth its bookkeeping yet. Indexing happens after the crawl, never
  interleaved with it; the pages database is the checkpoint between the
  two steps.
- Chunk metadata carries `url`, `title`, `heading`, `crawled_at` →
  citations later, and the collection is browsable in the backoffice RAG
  explorer and queryable with `examples/ragindex/query_index.py` today.

**Convenience, not services**: `scrape.py --index …` runs the index build
in the same process after the crawl. Internally these are two plain
functions over the pages database — the seam to cut along if crawling and
embedding ever become separate services.

Optional `--screenshot` saves a homepage capture for the later demo page.
Tests: injectable fetcher (no crawl4ai needed in the test run) + a fake
embedding client, both backends via the existing conformance pattern.

### Iteration 2 — serving the database

All serving options consume the same collection, so nothing in iteration 1
is throwaway. In rough order of value:

1. **Support-bot workflow template** (`rag_query` → agent → cited answer)
   + agent-server, per-collection — the hosted product core.
2. **Embed snippet / iframe**, plus per-bot token, domain allowlist,
   message caps. STARTED — `chatbotwidget/` holds the production widget
   (a framework-free port of the kaval.ai website chatbot: floating or
   inline, `--kcb-*` theming, markdown without `innerHTML`, an
   agent-server SSE connector, and any `send` callback — the WebLLM
   bridge included — behind the same connector shape). `webwidget/`
   remains the developer playground.
3. **Demo page** for the outreach funnel — BUILT:
   `tools/zerocostchatbot/make_demo.py` compiles a static, bucket-servable
   folder from the iteration-1 artefacts (screenshot or stored-HTML
   backdrop + the `chatbotwidget/` widget). Without a backend it answers
   client-side from the site's chunks with source links; `--endpoint`
   swaps in a running agent server. Generated on demand — no pre-crawl of
   the full prospect list required.
   `tools/zerocostchatbot/archive.html` is the richer, fully client-side
   demo: browses the pages database Wayback-style (sql.js in the browser,
   navigation limited to crawled pages) with the widget answering from the
   RAG index via WebLLM — embedding + chat model in-browser, progress in
   the widget header, lexical fallback without WebGPU — with skin and
   model pickers.
4. **WebLLM landing-page demo**, WordPress plugin, self-hosted compose
   bundle — channels, as in the deployment section above.

## Open questions

- Trial economics: which hosted model for free trials, and what message cap?
- Multi-tenancy: one agent-server process with per-bot workflow instances,
  or one container per client? (One process + per-bot collections scales
  the trial tier far cheaper; isolation per container for paid.)
- Crawl boundaries: same-domain only by default? Max pages per trial
  (e.g. 200) to bound cost?
- Anti-abuse: crawling arbitrary URLs on request is an SSRF/abuse surface —
  block private IP ranges, honour robots.txt, verify domain ownership
  before a bot goes live on the paid tier?
