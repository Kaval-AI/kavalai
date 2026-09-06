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

import asyncio
import base64
import json
import sys
import httpx
import pytest

from tools.zerocostchatbot import scrape
from tools.zerocostchatbot.pages_db import PagesDatabase
from tools.zerocostchatbot.scrape import (
    FetchOutcome,
    HttpFetcher,
    build_parser,
    crawl_site,
    is_crawlable_path,
    load_robots,
    looks_like_shell,
    normalize_url,
    page_links,
    parse_sitemap,
    resolve_user_agent,
    same_site,
    sitemap_urls,
)

RICH_TEXT = "Kaval.AI turns a YAML file into an agent. " * 20


def html_page(title, body, links=()):
    anchors = "".join(f'<a href="{href}">{href}</a>' for href in links)
    return f"<html><head><title>{title}</title></head><body><p>{body}</p>{anchors}</body></html>"


def rich_page(title, links=()):
    return html_page(title, RICH_TEXT, links)


def outcome_ok(links=(), markdown=RICH_TEXT, needs_browser=False):
    return FetchOutcome(
        success=True,
        status_code=200,
        title="T",
        html="<html/>",
        markdown=markdown,
        links=list(links),
        needs_browser=needs_browser,
    )


def fetcher_from(pages):
    """An async fetch over a dict, recording the URLs it was asked for."""

    async def fetch(url):
        fetch.calls.append(url)
        return pages.get(
            url, FetchOutcome(success=False, status_code=404, error="HTTP 404")
        )

    fetch.calls = []
    return fetch


def make_db(tmp_path):
    return PagesDatabase(str(tmp_path / "pages.db"))


def test_resolve_user_agent():
    header, token = resolve_user_agent("kavalai")
    assert header.startswith("KavalaiBot/") and token == "KavalaiBot"
    assert resolve_user_agent("googlebot")[1] == "Googlebot"
    assert resolve_user_agent("browser")[1] == "*"
    assert resolve_user_agent("MyBot/2.0") == ("MyBot/2.0", "*")


def test_normalize_url():
    assert (
        normalize_url("../a?x=1#frag", "https://Kaval.AI/docs/page")
        == "https://kaval.ai/a?x=1"
    )
    assert normalize_url("https://kaval.ai") == "https://kaval.ai/"
    assert normalize_url("mailto:hi@kaval.ai") is None
    assert normalize_url("javascript:void(0)", "https://kaval.ai/") is None
    assert normalize_url("") is None
    assert normalize_url("ftp://kaval.ai/x") is None


def test_same_site_ignores_www():
    assert same_site("https://www.kaval.ai/x", "kaval.ai")
    assert same_site("https://kaval.ai/x", "kaval.ai")
    assert not same_site("https://docs.kaval.ai/x", "kaval.ai")


def test_is_crawlable_path():
    assert is_crawlable_path("https://kaval.ai/pricing")
    assert is_crawlable_path("https://kaval.ai/release-1.0.4")
    assert not is_crawlable_path("https://kaval.ai/logo.png")
    assert not is_crawlable_path("https://kaval.ai/app.JS")


def test_page_links_filters_and_deduplicates():
    links = page_links(
        [
            "/a",
            "/a#section",
            "/logo.png",
            "https://other.example/x",
            "mailto:x@kaval.ai",
            "/b",
        ],
        base="https://kaval.ai/",
        host="kaval.ai",
    )
    assert links == ["https://kaval.ai/a", "https://kaval.ai/b"]


def test_parse_sitemap_urlset_and_index():
    ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    pages, nested = parse_sitemap(
        f"<urlset {ns}><url><loc>https://kaval.ai/a</loc></url></urlset>"
    )
    assert pages == ["https://kaval.ai/a"] and nested == []
    pages, nested = parse_sitemap(
        f"<sitemapindex {ns}><sitemap><loc>https://kaval.ai/s1.xml</loc></sitemap>"
        f"</sitemapindex>"
    )
    assert pages == [] and nested == ["https://kaval.ai/s1.xml"]


def test_looks_like_shell():
    assert looks_like_shell("<div id=root></div>", "Loading…")
    assert looks_like_shell(
        "<noscript>Please enable JavaScript to continue</noscript>", RICH_TEXT
    )
    assert not looks_like_shell(rich_page("t"), RICH_TEXT)


def site_transport(responses, raise_for=()):
    """A MockTransport serving a dict of url → (status, content_type, body)."""

    def handler(request):
        url = str(request.url)
        if url in raise_for:
            raise httpx.ConnectError("unreachable", request=request)
        if url not in responses:
            return httpx.Response(404, text="missing")
        status, content_type, body = responses[url]
        return httpx.Response(status, headers={"content-type": content_type}, text=body)

    return httpx.MockTransport(handler)


def client_for(responses, **kwargs):
    return httpx.AsyncClient(
        transport=site_transport(responses, **kwargs), follow_redirects=True
    )


@pytest.mark.asyncio
async def test_sitemap_urls_follows_one_level_of_index():
    ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    responses = {
        "https://kaval.ai/sitemap.xml": (
            200,
            "application/xml",
            f"<sitemapindex {ns}><sitemap><loc>https://kaval.ai/s1.xml</loc>"
            f"</sitemap></sitemapindex>",
        ),
        "https://kaval.ai/s1.xml": (
            200,
            "application/xml",
            f"<urlset {ns}><url><loc>https://kaval.ai/a</loc></url>"
            f"<url><loc>https://kaval.ai/b</loc></url></urlset>",
        ),
    }
    async with client_for(responses) as client:
        assert await sitemap_urls(client, "https://kaval.ai/", limit=10) == [
            "https://kaval.ai/a",
            "https://kaval.ai/b",
        ]
        assert await sitemap_urls(client, "https://kaval.ai/", limit=1) == [
            "https://kaval.ai/a"
        ]


@pytest.mark.asyncio
async def test_sitemap_urls_absent_or_broken_is_empty():
    async with client_for({}) as client:
        assert await sitemap_urls(client, "https://kaval.ai/", limit=10) == []
    responses = {"https://kaval.ai/sitemap.xml": (200, "application/xml", "not xml")}
    async with client_for(responses) as client:
        assert await sitemap_urls(client, "https://kaval.ai/", limit=10) == []


@pytest.mark.asyncio
async def test_load_robots_matches_the_agent_token():
    responses = {
        "https://kaval.ai/robots.txt": (
            200,
            "text/plain",
            "User-agent: *\nDisallow: /\n\nUser-agent: Googlebot\nAllow: /\n",
        )
    }
    async with client_for(responses) as client:
        robots = await load_robots(client, "https://kaval.ai/")
    assert not robots.can_fetch("KavalaiBot", "https://kaval.ai/x")
    assert robots.can_fetch("Googlebot", "https://kaval.ai/x")


@pytest.mark.asyncio
async def test_load_robots_missing_or_unreachable_allows_everything():
    async with client_for({}) as client:
        robots = await load_robots(client, "https://kaval.ai/")
        assert robots.can_fetch("KavalaiBot", "https://kaval.ai/x")
    async with client_for({}, raise_for={"https://kaval.ai/robots.txt"}) as client:
        robots = await load_robots(client, "https://kaval.ai/")
        assert robots.can_fetch("KavalaiBot", "https://kaval.ai/x")


@pytest.mark.asyncio
async def test_http_fetcher_success_and_shell_detection():
    responses = {
        "https://kaval.ai/": (
            200,
            "text/html; charset=utf-8",
            rich_page("Kaval", links=["/a"]),
        ),
        "https://kaval.ai/shell": (200, "text/html", "<div id='root'></div>"),
        "https://kaval.ai/blocked": (403, "text/html", "checking your browser"),
        "https://kaval.ai/gone": (410, "text/html", "gone"),
        "https://kaval.ai/data": (200, "application/json", "{}"),
    }
    async with client_for(responses) as client:
        fetcher = HttpFetcher(client)
        page = await fetcher.fetch("https://kaval.ai/")
        assert page.success and not page.needs_browser
        assert page.title == "Kaval"
        assert "https://kaval.ai/a" in page.links

        shell = await fetcher.fetch("https://kaval.ai/shell")
        assert shell.success and shell.needs_browser

        blocked = await fetcher.fetch("https://kaval.ai/blocked")
        assert not blocked.success and blocked.needs_browser

        gone = await fetcher.fetch("https://kaval.ai/gone")
        assert not gone.success and not gone.needs_browser
        assert gone.error == "HTTP 410"

        data = await fetcher.fetch("https://kaval.ai/data")
        assert not data.success and "not HTML" in data.error


@pytest.mark.asyncio
async def test_http_fetcher_transport_error_does_not_escalate():
    async with client_for({}, raise_for={"https://kaval.ai/x"}) as client:
        outcome = await HttpFetcher(client).fetch("https://kaval.ai/x")
    assert not outcome.success and not outcome.needs_browser
    assert "ConnectError" in outcome.error


@pytest.mark.asyncio
async def test_crawl_site_discovers_links_breadth_first(tmp_path):
    http = fetcher_from(
        {
            "https://kaval.ai/": outcome_ok(links=["https://kaval.ai/b"]),
            "https://kaval.ai/b": outcome_ok(links=["https://kaval.ai/c"]),
            "https://kaval.ai/c": outcome_ok(),
        }
    )
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/"])
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, max_pages=10
        )
        assert report.fetched == 3 and report.failed == 0
        assert db.fetched_count() == 3
        assert {row.fetch_mode for row in db.iter_pages()} == {"http"}


@pytest.mark.asyncio
async def test_crawl_site_budget_counts_the_database_not_the_run(tmp_path):
    http = fetcher_from(
        {"https://kaval.ai/": outcome_ok(), "https://kaval.ai/b": outcome_ok()}
    )
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/", "https://kaval.ai/b"])
        db.record_success(
            "https://kaval.ai/",
            status_code=200,
            fetch_mode="http",
            title="t",
            html="x",
            markdown="x",
        )
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, max_pages=2
        )
        # One page was already in the database, so the budget allows one more.
        assert report.fetched == 1
        assert db.next_pending() is None or db.fetched_count() == 2


@pytest.mark.asyncio
async def test_crawl_site_respects_robots(tmp_path):
    http = fetcher_from({"https://kaval.ai/": outcome_ok()})
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/private", "https://kaval.ai/"])
        report = await crawl_site(
            db,
            "https://kaval.ai/",
            http,
            None,
            allowed=lambda url: "/private" not in url,
            delay=0,
        )
        assert report.disallowed == 1 and report.fetched == 1
        assert http.calls == ["https://kaval.ai/"]


@pytest.mark.asyncio
async def test_crawl_site_escalates_and_memoizes_a_js_site(tmp_path):
    urls = [f"https://kaval.ai/p{i}" for i in range(5)]
    http = fetcher_from({url: outcome_ok(needs_browser=True) for url in urls})
    browser = fetcher_from({url: outcome_ok(markdown="rendered") for url in urls})
    with make_db(tmp_path) as db:
        db.add_urls(urls)
        report = await crawl_site(db, "https://kaval.ai/", http, browser, delay=0)
        assert report.fetched == 5
        # After three consecutive escalations the HTTP attempt is skipped.
        assert len(http.calls) == 3
        assert len(browser.calls) == 5
        assert {row.fetch_mode for row in db.iter_pages()} == {"browser"}


@pytest.mark.asyncio
async def test_crawl_site_keeps_http_content_when_escalation_fails(tmp_path):
    http = fetcher_from(
        {"https://kaval.ai/": outcome_ok(markdown="thin", needs_browser=True)}
    )

    async def broken_browser(url):
        return FetchOutcome(success=False, error="crawl4ai is not installed")

    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/"])
        report = await crawl_site(
            db, "https://kaval.ai/", http, broken_browser, delay=0
        )
        assert report.fetched == 1
        (row,) = db.iter_pages()
        assert row.fetch_mode == "http" and row.markdown == "thin"


@pytest.mark.asyncio
async def test_crawl_site_failure_is_retried_on_the_next_run_only(tmp_path):
    http = fetcher_from({})
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/missing"])
        report = await crawl_site(db, "https://kaval.ai/", http, None, delay=0)
        assert report.failed == 1
        assert http.calls == ["https://kaval.ai/missing"]

        # The next run tries again, until the attempt cap parks the URL.
        second = await crawl_site(db, "https://kaval.ai/", http, None, delay=0)
        third = await crawl_site(db, "https://kaval.ai/", http, None, delay=0)
        fourth = await crawl_site(db, "https://kaval.ai/", http, None, delay=0)
        assert second.failed == third.failed == 1 and fourth.failed == 0
        assert len(http.calls) == 3


class StubBrowser:
    """Stands in for BrowserFetcher in run() tests."""

    def __init__(self):
        self.fetch = fetcher_from({})
        self.screenshots = []

    async def screenshot(self, url):
        self.screenshots.append(url)
        return b"png-bytes"


def kaval_site():
    ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    return {
        "https://kaval.ai/robots.txt": (
            200,
            "text/plain",
            "User-agent: *\nDisallow: /private\n",
        ),
        "https://kaval.ai/sitemap.xml": (
            200,
            "application/xml",
            f"<urlset {ns}><url><loc>https://kaval.ai/docs</loc></url></urlset>",
        ),
        "https://kaval.ai/": (
            200,
            "text/html",
            rich_page(
                "Home", links=["/a", "/private", "/logo.png", "https://other.example/"]
            ),
        ),
        "https://kaval.ai/a": (200, "text/html", rich_page("A", links=["/"])),
        "https://kaval.ai/docs": (200, "text/html", rich_page("Docs")),
    }


def parse_run_args(tmp_path, *extra):
    return build_parser().parse_args(
        [
            "https://kaval.ai",
            "--pages",
            str(tmp_path / "kaval.pages.db"),
            "--delay",
            "0",
            *extra,
        ]
    )


@pytest.mark.asyncio
async def test_run_scrapes_a_server_side_rendered_site(tmp_path):
    args = parse_run_args(tmp_path, "--screenshot")
    browser = StubBrowser()
    stats = await scrape.run(
        args, transport=site_transport(kaval_site()), browser_fetcher=browser
    )
    assert stats["fetched"] == 3

    with PagesDatabase(str(tmp_path / "kaval.pages.db")) as db:
        rows = {row.url: row for row in db.iter_pages()}
    assert rows["https://kaval.ai/"].title == "Home"
    assert rows["https://kaval.ai/docs"].fetch_mode == "http"
    assert rows["https://kaval.ai/private"].fetch_error == "disallowed by robots.txt"
    assert "https://kaval.ai/logo.png" not in rows

    # An SSR site never needed the browser — except for the screenshot,
    # which is stored on the start page's row.
    assert browser.fetch.calls == []
    assert rows["https://kaval.ai/"].screenshot == b"png-bytes"
    assert rows["https://kaval.ai/a"].screenshot is None


@pytest.mark.asyncio
async def test_run_resumes_and_refreshes(tmp_path):
    args = parse_run_args(tmp_path, "--max-pages", "1")
    transport = site_transport(kaval_site())
    browser = StubBrowser()
    first = await scrape.run(args, transport=transport, browser_fetcher=browser)
    assert first["fetched"] == 1 and first["pending"] >= 1

    # A re-run continues with the remaining budget raised.
    args = parse_run_args(tmp_path)
    resumed = await scrape.run(args, transport=transport, browser_fetcher=browser)
    assert resumed["fetched"] == 3 and resumed["pending"] == 0

    refreshed = await scrape.run(
        parse_run_args(tmp_path, "--refresh"),
        transport=transport,
        browser_fetcher=browser,
    )
    assert refreshed["fetched"] == 3 and refreshed["pending"] == 0


@pytest.mark.asyncio
async def test_run_force_browser_uses_only_the_browser(tmp_path):
    args = parse_run_args(tmp_path, "--force-browser", "--max-pages", "2")
    browser = StubBrowser()
    browser.fetch = fetcher_from(
        {
            "https://kaval.ai/": outcome_ok(links=["https://kaval.ai/a"]),
            "https://kaval.ai/a": outcome_ok(),
            "https://kaval.ai/docs": outcome_ok(),
        }
    )
    stats = await scrape.run(
        args, transport=site_transport(kaval_site()), browser_fetcher=browser
    )
    assert stats["fetched"] == 2
    assert all(call.startswith("https://kaval.ai/") for call in browser.fetch.calls)


@pytest.mark.asyncio
async def test_run_rejects_an_uncrawlable_url():
    with pytest.raises(ValueError):
        await scrape.run(build_parser().parse_args(["ftp://kaval.ai/x"]))


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scrape.py", "https://kaval.ai"])

    async def fake_run(args):
        return {"fetched": 3, "pending": 0, "errors": 0, "total": 3}

    monkeypatch.setattr(scrape, "run", fake_run)
    assert scrape.main() == 0

    async def empty_run(args):
        return {"fetched": 0, "pending": 0, "errors": 1, "total": 1}

    monkeypatch.setattr(scrape, "run", empty_run)
    assert scrape.main() == 1

    async def bad_run(args):
        raise ValueError("Not a crawlable URL")

    monkeypatch.setattr(scrape, "run", bad_run)
    assert scrape.main() == 2


@pytest.mark.asyncio
async def test_crawl_site_sleeps_between_fetches(tmp_path):
    http = fetcher_from({"https://kaval.ai/": outcome_ok()})
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/"])
        report = await crawl_site(db, "https://kaval.ai/", http, None, delay=0.001)
    assert report.fetched == 1


@pytest.mark.asyncio
async def test_run_ignore_robots_with_force_http(tmp_path):
    site = kaval_site()
    site["https://kaval.ai/private"] = (200, "text/html", rich_page("Private"))
    args = parse_run_args(
        tmp_path,
        "--ignore-robots",
        "--force-http",
        "--screenshot",
    )
    stats = await scrape.run(args, transport=site_transport(site))
    assert stats["fetched"] == 4
    with PagesDatabase(str(tmp_path / "kaval.pages.db")) as db:
        rows = {row.url: row for row in db.iter_pages()}
    assert rows["https://kaval.ai/private"].title == "Private"
    # --force-http means no browser, so no screenshot was produced.
    assert all(row.screenshot is None for row in rows.values())


@pytest.mark.asyncio
async def test_run_never_contacts_the_container_for_an_ssr_site(tmp_path):
    args = parse_run_args(tmp_path)
    # The transport has no /crawl route, so any container call would fail
    # loudly; an SSR site must not make one.
    stats = await scrape.run(args, transport=site_transport(kaval_site()))
    assert stats["fetched"] == 3


def slow_fetcher(pages, seconds):
    """A fetch that takes real event-loop time, to force worker overlap."""

    async def fetch(url):
        fetch.calls.append(url)
        await asyncio.sleep(seconds)
        return pages.get(
            url, FetchOutcome(success=False, status_code=404, error="HTTP 404")
        )

    fetch.calls = []
    return fetch


@pytest.mark.asyncio
async def test_crawl_site_workers_fetch_in_parallel(tmp_path):
    urls = [f"https://kaval.ai/p{i}" for i in range(8)]
    http = slow_fetcher({url: outcome_ok() for url in urls}, 0.05)
    with make_db(tmp_path) as db:
        db.add_urls(urls)
        started = asyncio.get_running_loop().time()
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, concurrency=4
        )
        elapsed = asyncio.get_running_loop().time() - started
    assert report.fetched == 8
    # Sequentially this is 8 × 0.05 s; four workers need about two rounds.
    assert elapsed < 0.05 * 8 * 0.75


@pytest.mark.asyncio
async def test_crawl_site_idle_workers_wait_for_discovered_links(tmp_path):
    children = [f"https://kaval.ai/c{i}" for i in range(3)]
    http = slow_fetcher(
        {"https://kaval.ai/": outcome_ok(links=children)}
        | {url: outcome_ok() for url in children},
        0.02,
    )
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/"])
        # Only one URL is claimable at first; the other three workers must
        # wait for its commit instead of exiting early.
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, concurrency=4
        )
    assert report.fetched == 4


@pytest.mark.asyncio
async def test_crawl_site_budget_holds_under_concurrency(tmp_path):
    urls = [f"https://kaval.ai/p{i}" for i in range(10)]
    http = slow_fetcher({url: outcome_ok() for url in urls}, 0.02)
    with make_db(tmp_path) as db:
        db.add_urls(urls)
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, max_pages=3, concurrency=4
        )
        assert report.fetched == 3
        assert db.fetched_count() == 3


@pytest.mark.asyncio
async def test_request_spacer_caps_the_site_wide_rate():
    spacer = scrape.RequestSpacer(0.05)
    started = asyncio.get_running_loop().time()
    await asyncio.gather(spacer.wait(), spacer.wait(), spacer.wait())
    elapsed = asyncio.get_running_loop().time() - started
    # Three starts spaced 0.05 s apart need at least 0.1 s in total.
    assert elapsed >= 0.1
    # A zero interval never sleeps.
    assert await scrape.RequestSpacer(0).wait() is None


@pytest.mark.asyncio
async def test_idle_worker_survives_a_slow_page(tmp_path):
    # One page takes longer than the idle-wait timeout, so the waiting worker
    # times out, re-checks, and still picks up the link it eventually commits.
    http = slow_fetcher(
        {
            "https://kaval.ai/": outcome_ok(links=["https://kaval.ai/b"]),
            "https://kaval.ai/b": outcome_ok(),
        },
        0.25,
    )
    with make_db(tmp_path) as db:
        db.add_urls(["https://kaval.ai/"])
        report = await crawl_site(
            db, "https://kaval.ai/", http, None, delay=0, concurrency=2
        )
    assert report.fetched == 2


def remote_result(**overrides):
    result = {
        "success": True,
        "status_code": 200,
        "cleaned_html": "<h1>Rendered</h1>",
        "markdown": {"raw_markdown": "# Rendered"},
        "links": {
            "internal": [{"href": "https://kaval.ai/a"}],
            "external": [],
        },
        "metadata": {"title": "Rendered"},
        "error_message": None,
        "screenshot": base64.b64encode(b"png-bytes").decode(),
    }
    result.update(overrides)
    return result


def crawl4ai_transport(result, requests_seen):
    def handler(request):
        assert request.url.path == "/crawl"
        requests_seen.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "results": [result]})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_remote_browser_fetcher_maps_the_server_result():
    seen = []
    async with httpx.AsyncClient(
        transport=crawl4ai_transport(remote_result(), seen)
    ) as client:
        fetcher = scrape.RemoteBrowserFetcher(
            "http://localhost:11235/", client, "KavalaiBot/1.0", timeout=20
        )
        outcome = await fetcher.fetch("https://kaval.ai/")
        image = await fetcher.screenshot("https://kaval.ai/")

    assert outcome.success
    assert outcome.markdown == "# Rendered"
    assert outcome.title == "Rendered"
    assert outcome.links == ["https://kaval.ai/a"]
    assert image == b"png-bytes"

    fetch_payload, shot_payload = seen
    assert fetch_payload["urls"] == ["https://kaval.ai/"]
    assert fetch_payload["browser_config"]["params"]["user_agent"] == "KavalaiBot/1.0"
    run_params = fetch_payload["crawler_config"]["params"]
    assert run_params["cache_mode"] == "bypass"
    assert run_params["page_timeout"] == 20000
    assert "screenshot" not in run_params
    assert shot_payload["crawler_config"]["params"]["screenshot"] is True


@pytest.mark.asyncio
async def test_remote_browser_fetcher_failures():
    seen = []
    failed = remote_result(
        success=False, error_message="net::ERR_NAME_NOT_RESOLVED", screenshot=None
    )
    async with httpx.AsyncClient(transport=crawl4ai_transport(failed, seen)) as client:
        fetcher = scrape.RemoteBrowserFetcher("http://x", client, "ua")
        outcome = await fetcher.fetch("https://kaval.ai/")
        assert not outcome.success
        assert outcome.error == "net::ERR_NAME_NOT_RESOLVED"
        assert await fetcher.screenshot("https://kaval.ai/") is None

    def server_error(request):
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(server_error)) as client:
        fetcher = scrape.RemoteBrowserFetcher("http://x", client, "ua")
        outcome = await fetcher.fetch("https://kaval.ai/")
        assert not outcome.success and "HTTPStatusError" in outcome.error
        assert await fetcher.screenshot("https://kaval.ai/") is None

    def empty_body(request):
        return httpx.Response(200, json={"success": True, "results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(empty_body)) as client:
        fetcher = scrape.RemoteBrowserFetcher("http://x", client, "ua")
        outcome = await fetcher.fetch("https://kaval.ai/")
        assert not outcome.success and "no result" in outcome.error


@pytest.mark.asyncio
async def test_run_uses_the_crawl4ai_container_when_asked(tmp_path):
    site = kaval_site()

    def handler(request):
        url = str(request.url)
        if request.url.path == "/crawl":
            target = json.loads(request.content)["urls"][0]
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "results": [
                        remote_result(
                            cleaned_html=rich_page("Remote"),
                            markdown={"raw_markdown": RICH_TEXT},
                            links={"internal": [], "external": []},
                            metadata={"title": f"Remote {target}"},
                        )
                    ],
                },
            )
        if url not in site:
            return httpx.Response(404, text="missing")
        status, content_type, body = site[url]
        return httpx.Response(status, headers={"content-type": content_type}, text=body)

    args = parse_run_args(tmp_path, "--force-browser", "--max-pages", "2")
    stats = await scrape.run(args, transport=httpx.MockTransport(handler))
    assert stats["fetched"] == 2
    with PagesDatabase(str(tmp_path / "kaval.pages.db")) as db:
        assert all(
            row.fetch_mode == "browser" for row in db.iter_pages() if row.markdown
        )


@pytest.mark.asyncio
async def test_remote_browser_fetcher_hints_when_the_container_is_down():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        fetcher = scrape.RemoteBrowserFetcher("http://localhost:11235", client, "ua")
        outcome = await fetcher.fetch("https://kaval.ai/")
    assert not outcome.success
    assert "docker compose up crawl4ai" in outcome.error
