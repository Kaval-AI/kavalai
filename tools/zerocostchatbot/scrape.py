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

Scrape a website into a pages database (see ``pages_db.PagesDatabase``).

Scrape the Kaval.AI docs into ``docs.kaval.ai.pages.db``::

    python -m tools.zerocostchatbot.scrape https://docs.kaval.ai --max-pages 50

Resume is automatic: the pages database is the crawl state, so re-running
the same command continues where a killed run stopped. ``--refresh``
re-queues every known URL for a fresh crawl.

Each page is fetched with a plain HTTP request first and escalated to a
headless browser (crawl4ai) only when the response is not usable content —
a JS-app shell, a bot challenge, or too little visible text. After a few
consecutive escalations the site is treated as JS-rendered and later pages
go straight to the browser. ``--force-http`` / ``--force-browser`` override
the heuristic; a server-side-rendered site is scraped with the base install
alone, no browser stack needed.
"""

import argparse
import asyncio
import base64
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from loguru import logger

from tools.zerocostchatbot.htmlmd import parse_html
from tools.zerocostchatbot.pages_db import PagesDatabase

USER_AGENTS = {
    "kavalai": ("KavalaiBot/1.0 (+https://kaval.ai/bot)", "KavalaiBot"),
    "browser": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "*",
    ),
    "googlebot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible;"
        " Googlebot/2.1; +http://www.google.com/bot.html)",
        "Googlebot",
    ),
}

SKIP_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".json",
    ".xml",
    ".rss",
    ".atom",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".gz",
    ".tar",
    ".rar",
    ".7z",
    ".whl",
    ".exe",
    ".dmg",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".wav",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

# Below this much extracted text a 200 response is treated as a JS-app shell
# and escalated to the browser. A false positive only costs one browser fetch.
MIN_CONTENT_CHARS = 200

JS_WARNINGS = (
    "enable javascript",
    "javascript is required",
    "javascript to run this app",
    "turn on javascript",
)

# Bot walls answer these while still serving real content to a browser.
CHALLENGE_STATUSES = {403, 503}

# After this many consecutive escalations the site is JS-rendered: skip the
# doomed HTTP attempt on every later page.
ESCALATION_MEMO = 3

DEFAULT_MAX_PAGES = 200
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 3
SITEMAP_FILE_LIMIT = 10


def resolve_user_agent(value: str) -> tuple[str, str]:
    """The header string and robots.txt token for a ``--user-agent`` value.

    A known preset name maps to its pair; anything else is used verbatim as
    the header and matched as ``*`` in robots.txt.
    """
    if value in USER_AGENTS:
        return USER_AGENTS[value]
    return value, "*"


def normalize_url(href: str, base: Optional[str] = None) -> Optional[str]:
    """A canonical absolute URL, or None for links that cannot be crawled.

    Fragments are dropped, the host is lowercased and an empty path becomes
    ``/`` so the same page cannot enter the frontier twice.
    """
    href = (href or "").strip()
    if not href:
        return None
    absolute = urljoin(base, href) if base else href
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )
    )


def site_host(url: str) -> str:
    """The host a crawl is confined to, with any ``www.`` prefix dropped."""
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def same_site(url: str, host: str) -> bool:
    """Whether a URL belongs to the crawled site (``www.`` ignored)."""
    return site_host(url) == host


def is_crawlable_path(url: str) -> bool:
    """Whether the URL's path looks like a page rather than an asset."""
    path = urlparse(url).path.lower()
    dot = path.rfind(".")
    return dot == -1 or path[dot:] not in SKIP_EXTENSIONS


def page_links(links: list[str], base: str, host: str) -> list[str]:
    """The frontier-worthy subset of a page's links, normalized and deduplicated."""
    seen: dict[str, None] = {}
    for href in links:
        url = normalize_url(href, base)
        if url and same_site(url, host) and is_crawlable_path(url):
            seen.setdefault(url)
    return list(seen)


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """The page URLs and nested sitemap URLs a sitemap file lists."""
    root = ET.fromstring(xml_text)
    locations = [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text
    ]
    if root.tag.endswith("sitemapindex"):
        return [], locations
    return locations, []


async def sitemap_urls(
    client: httpx.AsyncClient, start_url: str, limit: int
) -> list[str]:
    """Page URLs from the site's ``sitemap.xml``, empty when there is none.

    A sitemap index is followed one level deep, a few files at most — the
    sitemap only seeds the frontier, link discovery finds the rest.
    """
    parsed = urlparse(start_url)
    queue = [f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"]
    pages: list[str] = []
    fetched = 0
    while queue and len(pages) < limit and fetched < SITEMAP_FILE_LIMIT:
        sitemap_url = queue.pop(0)
        fetched += 1
        try:
            response = await client.get(sitemap_url)
            if response.status_code != 200:
                continue
            found, nested = parse_sitemap(response.text)
        except (httpx.HTTPError, ET.ParseError) as error:
            logger.debug(f"Sitemap {sitemap_url} unusable: {error}")
            continue
        pages.extend(found)
        queue.extend(nested)
    if pages:
        logger.info(f"Sitemap seeded {len(pages)} URLs")
    return pages[:limit]


async def load_robots(
    client: httpx.AsyncClient, start_url: str
) -> robotparser.RobotFileParser:
    """The site's parsed ``robots.txt``; everything is allowed when it is absent."""
    parsed = urlparse(start_url)
    robots = robotparser.RobotFileParser()
    try:
        response = await client.get(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    except httpx.HTTPError as error:
        logger.debug(f"robots.txt unreachable ({error}); allowing everything")
        response = None
    if response is not None and response.status_code == 200:
        robots.parse(response.text.splitlines())
    else:
        # A parser that never saw rules answers False to everything; an
        # absent robots.txt means the opposite.
        robots.allow_all = True
    return robots


@dataclass
class FetchOutcome:
    """What one fetch attempt produced."""

    success: bool
    status_code: Optional[int] = None
    title: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    links: list[str] = field(default_factory=list)
    error: Optional[str] = None
    needs_browser: bool = False


Fetch = Callable[[str], Awaitable[FetchOutcome]]


def looks_like_shell(html: str, markdown: str) -> bool:
    """Whether a 200 response is a JS-app shell rather than rendered content."""
    if len(markdown) < MIN_CONTENT_CHARS:
        return True
    lowered = html.lower()
    return any(warning in lowered for warning in JS_WARNINGS)


class HttpFetcher:
    """The cheap path: a plain HTTP request plus stdlib HTML extraction."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def fetch(self, url: str) -> FetchOutcome:
        try:
            response = await self.client.get(url)
        except httpx.HTTPError as error:
            return FetchOutcome(success=False, error=f"{type(error).__name__}: {error}")

        status = response.status_code
        if status in CHALLENGE_STATUSES:
            return FetchOutcome(
                success=False,
                status_code=status,
                error=f"HTTP {status} (possible bot challenge)",
                needs_browser=True,
            )
        if status != 200:
            return FetchOutcome(
                success=False, status_code=status, error=f"HTTP {status}"
            )
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type:
            return FetchOutcome(
                success=False, status_code=status, error=f"not HTML ({content_type})"
            )

        parsed = parse_html(response.text, str(response.url))
        return FetchOutcome(
            success=True,
            status_code=status,
            title=parsed.title or None,
            html=response.text,
            markdown=parsed.markdown,
            links=parsed.links,
            needs_browser=looks_like_shell(response.text, parsed.markdown),
        )


class BrowserFetcher:
    """The rendering path: crawl4ai's headless browser, started on first use.

    crawl4ai is imported lazily, so a crawl that never escalates works without
    it installed; when it is missing, every browser fetch fails with an
    install hint instead of raising.
    """

    def __init__(self, user_agent: str, timeout: float = DEFAULT_TIMEOUT):
        self.user_agent = user_agent
        self.timeout = timeout
        self._crawler = None

    async def _ensure_crawler(self):
        if self._crawler is None:
            from crawl4ai import AsyncWebCrawler, BrowserConfig

            self._crawler = AsyncWebCrawler(
                config=BrowserConfig(headless=True, user_agent=self.user_agent)
            )
            await self._crawler.__aenter__()
        return self._crawler

    async def _run(self, url: str, **config_kwargs):
        from crawl4ai import CacheMode, CrawlerRunConfig

        crawler = await self._ensure_crawler()
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=int(self.timeout * 1000),
            **config_kwargs,
        )
        return await crawler.arun(url=url, config=config)

    async def fetch(self, url: str) -> FetchOutcome:
        try:
            result = await self._run(url)
        except ImportError:
            return FetchOutcome(
                success=False,
                error="crawl4ai is not installed (pip install kavalai[common])",
            )
        except Exception as error:
            return FetchOutcome(success=False, error=f"{type(error).__name__}: {error}")

        if not result.success:
            return FetchOutcome(
                success=False,
                status_code=result.status_code,
                error=result.error_message or "browser fetch failed",
            )

        markdown = result.markdown
        if markdown is not None:
            markdown = getattr(markdown, "raw_markdown", str(markdown))
        html = result.cleaned_html or ""
        links = [
            item.get("href") if isinstance(item, dict) else item
            for group in (result.links or {}).values()
            for item in group
        ]
        title = (result.metadata or {}).get("title")
        return FetchOutcome(
            success=True,
            status_code=result.status_code,
            title=title,
            html=html,
            markdown=markdown or parse_html(html, url).markdown,
            links=[link for link in links if link],
        )

    async def screenshot(self, url: str) -> Optional[bytes]:
        """A PNG capture of the page, None when the browser cannot provide one."""
        try:
            result = await self._run(url, screenshot=True)
        except ImportError:
            logger.error("Screenshot needs crawl4ai (pip install kavalai[common])")
            return None
        except Exception as error:
            logger.error(f"Screenshot of {url} failed: {error}")
            return None
        if not result.success or not result.screenshot:
            return None
        return base64.b64decode(result.screenshot)

    async def aclose(self) -> None:
        if self._crawler is not None:
            await self._crawler.__aexit__(None, None, None)
            self._crawler = None


@dataclass
class CrawlReport:
    """What one crawl run did."""

    fetched: int = 0
    failed: int = 0
    disallowed: int = 0
    escalated: int = 0


async def crawl_site(
    db: PagesDatabase,
    start_url: str,
    http_fetch: Optional[Fetch],
    browser_fetch: Optional[Fetch],
    *,
    allowed: Callable[[str], bool] = lambda url: True,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay: float = DEFAULT_DELAY,
) -> CrawlReport:
    """Crawl pending URLs until the page budget or the frontier is exhausted.

    ``http_fetch`` is tried first and ``browser_fetch`` is the escalation;
    passing one of them as None forces the other mode. Every completed page is
    committed together with its discovered links before the next URL is
    picked, so the crawl can be killed and resumed at any point.

    ``max_pages`` caps the pages *holding content in the database*, not the
    pages fetched by this run — a resumed crawl keeps the original budget.
    """
    host = site_host(start_url)
    report = CrawlReport()
    tried: set[str] = set()
    consecutive_escalations = 0

    while db.fetched_count() < max_pages:
        url = db.next_pending(skip=tried, max_attempts=max_attempts)
        if url is None:
            break
        tried.add(url)

        if not allowed(url):
            db.record_skipped(url, "disallowed by robots.txt")
            report.disallowed += 1
            continue

        site_is_js = consecutive_escalations >= ESCALATION_MEMO
        outcome: Optional[FetchOutcome] = None
        mode = "browser"
        if http_fetch is not None and not site_is_js:
            outcome = await http_fetch(url)
            mode = "http"
        if browser_fetch is not None and (outcome is None or outcome.needs_browser):
            escalating = outcome is not None
            browser_outcome = await browser_fetch(url)
            if escalating:
                report.escalated += 1
                consecutive_escalations += 1
            # A failed escalation keeps the (thin but real) HTTP content
            # rather than losing the page entirely.
            if browser_outcome.success or outcome is None or not outcome.success:
                outcome, mode = browser_outcome, "browser"
        if outcome is not None and outcome.success and mode == "http":
            consecutive_escalations = 0

        if outcome is not None and outcome.success:
            links = page_links(outcome.links, url, host)
            db.record_success(
                url,
                status_code=outcome.status_code,
                fetch_mode=mode,
                title=outcome.title,
                html=outcome.html,
                markdown=outcome.markdown,
                links=links,
            )
            report.fetched += 1
            logger.info(
                f"[{db.fetched_count()}/{max_pages}] {url} ({mode}, {len(links)} links)"
            )
        else:
            error = outcome.error if outcome else "no fetcher available"
            status = outcome.status_code if outcome else None
            db.record_failure(url, status_code=status, fetch_error=error)
            report.failed += 1
            logger.warning(f"Failed {url}: {error}")

        if delay > 0:
            await asyncio.sleep(delay)
    return report


async def run(
    args: argparse.Namespace,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    browser_fetcher: Optional[BrowserFetcher] = None,
) -> dict:
    """Scrape as the CLI arguments describe; returns the final database stats.

    ``transport`` and ``browser_fetcher`` exist for tests, which substitute
    an ``httpx.MockTransport`` and a stub browser for the real network.
    """
    start_url = normalize_url(args.url if "://" in args.url else f"https://{args.url}")
    if start_url is None:
        raise ValueError(f"Not a crawlable URL: {args.url!r}")
    host = site_host(start_url)
    pages_path = args.pages or f"{host}.pages.db"
    user_agent, robots_token = resolve_user_agent(args.user_agent)

    with PagesDatabase(pages_path) as db:
        if args.refresh:
            logger.info(f"Re-queued {db.requeue_all()} known URLs")

        async with httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=args.timeout,
            transport=transport,
        ) as client:
            seeds = [start_url] + await sitemap_urls(client, start_url, args.max_pages)
            db.add_urls(page_links(seeds, start_url, host))

            if args.ignore_robots:

                def allowed(url: str) -> bool:
                    return True

            else:
                robots = await load_robots(client, start_url)

                def allowed(url: str) -> bool:
                    return robots.can_fetch(robots_token, url)

            browser = browser_fetcher
            if browser is None and not args.force_http:
                browser = BrowserFetcher(user_agent=user_agent, timeout=args.timeout)
            http_fetcher = HttpFetcher(client) if not args.force_browser else None

            try:
                report = await crawl_site(
                    db,
                    start_url,
                    http_fetcher.fetch if http_fetcher else None,
                    browser.fetch if browser else None,
                    allowed=allowed,
                    max_pages=args.max_pages,
                    max_attempts=args.max_attempts,
                    delay=args.delay,
                )
                if args.screenshot and browser:
                    image = await browser.screenshot(start_url)
                    if image:
                        saved = db.save_screenshot(start_url, image)
                        logger.info(f"Screenshot saved to {saved}")
                elif args.screenshot:
                    logger.warning("--screenshot needs the browser; skipped")
            finally:
                if browser is not None and browser is not browser_fetcher:
                    await browser.aclose()

        stats = db.stats()
    logger.info(
        f"Run: {report.fetched} fetched, {report.failed} failed,"
        f" {report.disallowed} disallowed, {report.escalated} escalated."
        f" Database {pages_path}: {stats['fetched']} pages,"
        f" {stats['pending']} pending."
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Command line of the scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape a website into a pages database.",
        epilog=(
            "Example: python -m tools.zerocostchatbot.scrape"
            " https://docs.kaval.ai --max-pages 50"
        ),
    )
    parser.add_argument("url", help="Site to scrape, e.g. https://docs.kaval.ai")
    parser.add_argument(
        "--pages",
        default=None,
        help=(
            "Pages database file to write to and resume from"
            " (default: <host>.pages.db, e.g. docs.kaval.ai.pages.db)"
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Stop once this many pages hold content (default: {DEFAULT_MAX_PAGES})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between fetches (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Park a URL after this many failed fetches (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--user-agent",
        default="kavalai",
        help=(
            "'kavalai' (default), 'browser', 'googlebot', or a verbatim"
            " User-Agent string; robots.txt is matched with the same identity"
        ),
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Do not honour the site's robots.txt",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force-http",
        action="store_true",
        help="Never start the headless browser",
    )
    mode.add_argument(
        "--force-browser",
        action="store_true",
        help="Fetch every page with the headless browser",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-queue every known URL before crawling (a fresh pass)",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help=(
            "Save a PNG of the start page into the files directory"
            " (needs the browser stack)"
        ),
    )
    return parser


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    try:
        stats = asyncio.run(run(args))
    except ValueError as error:
        logger.error(error)
        return 2
    return 0 if stats["fetched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
