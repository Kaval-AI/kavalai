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
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from kavalai import net
from kavalai.tools.webtools.crawl4ai import (
    crawl_url,
    make_crawl_url,
    web_search,
    Crawl4aiResponse,
    WebSearchResponse,
)
from tests.tools.local_http import fake_resolver


@pytest.fixture(autouse=True)
def no_real_dns(request, monkeypatch):
    """The URL pre-check resolves names; unit tests must not reach real DNS."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr(net, "resolve_host", fake_resolver)


def _make_crawler_mock(result):
    """Build a mock AsyncWebCrawler usable as an async context manager."""
    crawler = MagicMock()
    crawler.arun = AsyncMock(return_value=result)
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    crawler_cls = MagicMock(return_value=crawler)
    return crawler_cls, crawler


@pytest.mark.asyncio
async def test_crawl_url_success():
    markdown = MagicMock()
    markdown.raw_markdown = "# Example"

    result = MagicMock()
    result.url = "https://example.com"
    result.redirected_url = None
    result.success = True
    result.markdown = markdown
    result.cleaned_html = "<h1>Example</h1>"
    result.status_code = 200
    result.metadata = {"title": "Example"}
    result.error_message = None

    crawler_cls, crawler = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert isinstance(response, Crawl4aiResponse)
    assert response.url == "https://example.com"
    assert response.success is True
    assert response.markdown == "# Example"
    # HTML omitted by default.
    assert response.html is None
    assert response.status_code == 200
    assert response.metadata == {"title": "Example"}
    crawler.arun.assert_awaited_once()


@pytest.mark.asyncio
async def test_crawl_url_include_html_and_bypass_cache():
    result = MagicMock()
    result.url = "https://example.com"
    result.redirected_url = None
    result.success = True
    result.markdown = None
    result.cleaned_html = "<h1>Example</h1>"
    result.status_code = 200
    result.metadata = None
    result.error_message = None

    crawler_cls, crawler = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(
            url="https://example.com", include_html=True, bypass_cache=True
        )

    assert response.markdown is None
    assert response.html == "<h1>Example</h1>"

    # Bypass cache must be reflected in the run config.
    from crawl4ai import CacheMode

    _, kwargs = crawler.arun.call_args
    assert kwargs["config"].cache_mode == CacheMode.BYPASS


@pytest.mark.asyncio
async def test_crawl_url_failure():
    result = MagicMock()
    result.url = "https://example.com"
    result.redirected_url = None
    result.success = False
    result.markdown = None
    result.cleaned_html = None
    result.status_code = 404
    result.metadata = None
    result.error_message = "Not found"

    crawler_cls, _ = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.success is False
    assert response.markdown is None
    assert response.error_message == "Not found"


@pytest.mark.asyncio
async def test_crawl_url_markdown_without_raw_attr():
    # Some crawl4ai versions return a plain string for markdown.
    result = MagicMock()
    result.url = "https://example.com"
    result.redirected_url = None
    result.success = True
    result.markdown = "plain markdown"
    result.cleaned_html = None
    result.status_code = 200
    result.metadata = None
    result.error_message = None

    crawler_cls, _ = _make_crawler_mock(result)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.markdown == "plain markdown"


def _page(url="https://example.com", redirected_url=None):
    result = MagicMock()
    result.url = url
    result.redirected_url = redirected_url
    result.success = True
    result.markdown = "secret page"
    result.cleaned_html = "<p>secret page</p>"
    result.status_code = 200
    result.metadata = None
    result.error_message = None
    return result


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://internal.test/",
        "file:///etc/passwd",
    ],
)
async def test_crawl_url_refuses_a_private_target(url):
    crawler_cls, crawler = _make_crawler_mock(_page(url))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url=url)

    assert response.success is False
    assert response.url == url
    assert response.error_message.startswith("Refused: ")
    assert response.markdown is None
    crawler_cls.assert_not_called()


async def test_crawl_url_withholds_a_page_redirected_to_a_private_target():
    page = _page(redirected_url="http://10.0.0.1/admin")
    crawler_cls, _ = _make_crawler_mock(page)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.success is False
    assert response.url == "http://10.0.0.1/admin"
    assert response.status_code == 200
    assert response.markdown is None
    assert response.html is None
    assert response.error_message.startswith("Redirected to http://10.0.0.1/admin.")
    assert "10.0.0.1 is a non-public address" in response.error_message


async def test_crawl_url_checks_the_reported_url_too():
    crawler_cls, _ = _make_crawler_mock(_page(url="http://internal.test/"))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.success is False
    assert response.url == "http://internal.test/"


async def test_crawl_url_keeps_a_page_redirected_to_a_public_target():
    page = _page(redirected_url="https://www.example.com/")
    crawler_cls, _ = _make_crawler_mock(page)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await crawl_url(url="https://example.com")

    assert response.success is True
    assert response.markdown == "secret page"


async def test_allow_private_networks_crawls_an_intranet():
    intranet_crawl = make_crawl_url(allow_private_networks=True)
    page = _page(url="http://127.0.0.1:8080/", redirected_url="http://10.0.0.1/")
    crawler_cls, crawler = _make_crawler_mock(page)

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await intranet_crawl(url="http://127.0.0.1:8080/")

    assert response.success is True
    assert response.markdown == "secret page"
    crawler.arun.assert_awaited_once()
    assert intranet_crawl._is_kavalai_tool is True
    assert intranet_crawl.__name__ == "crawl_url"


@pytest.mark.parametrize("call", [crawl_url, web_search])
async def test_missing_crawl4ai_names_the_extra(monkeypatch, call):
    monkeypatch.setitem(sys.modules, "crawl4ai", None)

    with pytest.raises(ImportError, match=r'pip install "kavalai\[webtools\]"'):
        await call("https://example.com")


def _make_search_result(extracted_content, success=True, error_message=None):
    result = MagicMock()
    result.success = success
    result.extracted_content = extracted_content
    result.error_message = error_message
    return result


@pytest.mark.asyncio
async def test_web_search_success():
    extracted = json.dumps(
        [
            {
                "title": "Kaval AI",
                # DuckDuckGo wraps result links in a protocol-relative redirect.
                "url": "//duckduckgo.com/l/?uddg=https%3A%2F%2Fkaval.ai%2F&rut=abc",
                "snippet": "YAML-based AI agent framework.",
            },
            {
                "title": "Kaval AI on LinkedIn",
                "url": "https://www.linkedin.com/company/kaval-ai",
                "snippet": "",
            },
        ]
    )
    crawler_cls, crawler = _make_crawler_mock(_make_search_result(extracted))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await web_search(query="Kaval AI")

    assert isinstance(response, WebSearchResponse)
    assert response.success is True
    assert response.query == "Kaval AI"
    assert [r.url for r in response.results] == [
        "https://kaval.ai/",
        "https://www.linkedin.com/company/kaval-ai",
    ]
    assert response.results[0].snippet == "YAML-based AI agent framework."
    # Empty snippets become None.
    assert response.results[1].snippet is None

    # The query must be URL-encoded into the DuckDuckGo HTML endpoint.
    _, kwargs = crawler.arun.call_args
    assert kwargs["url"] == "https://duckduckgo.com/html/?q=Kaval+AI"


@pytest.mark.asyncio
async def test_web_search_respects_count():
    extracted = json.dumps(
        [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": "s"}
            for i in range(5)
        ]
    )
    crawler_cls, _ = _make_crawler_mock(_make_search_result(extracted))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await web_search(query="example", count=2)

    assert [r.title for r in response.results] == ["Result 0", "Result 1"]


@pytest.mark.asyncio
async def test_web_search_skips_unusable_results():
    extracted = json.dumps(
        [
            # Unresolvable redirect (no 'uddg' parameter).
            {"title": "Redirect", "url": "//duckduckgo.com/l/?rut=abc", "snippet": "s"},
            # Non-HTTP scheme.
            {"title": "Mail", "url": "mailto:info@example.com", "snippet": "s"},
            # Missing href.
            {"title": "No link", "url": None, "snippet": "s"},
            # Missing title.
            {"title": "", "url": "https://example.com", "snippet": "s"},
            {"title": "Good", "url": "https://example.com", "snippet": "s"},
        ]
    )
    crawler_cls, _ = _make_crawler_mock(_make_search_result(extracted))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await web_search(query="example")

    assert [r.title for r in response.results] == ["Good"]


@pytest.mark.asyncio
async def test_web_search_without_extracted_content():
    crawler_cls, _ = _make_crawler_mock(_make_search_result(None))

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await web_search(query="example")

    assert response.success is True
    assert response.results == []


@pytest.mark.asyncio
async def test_web_search_failure():
    crawler_cls, _ = _make_crawler_mock(
        _make_search_result(None, success=False, error_message="Timeout")
    )

    with patch("crawl4ai.AsyncWebCrawler", crawler_cls):
        response = await web_search(query="example")

    assert response.success is False
    assert response.results == []
    assert response.error_message == "Timeout"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("CRAWL4AI_INTEGRATION"),
    reason="CRAWL4AI_INTEGRATION not defined",
)
@pytest.mark.asyncio
async def test_crawl_url_integration():
    """
    Real integration test for crawl_url tool.
    Only runs if CRAWL4AI_INTEGRATION is defined and a browser is installed.
    """
    response = await crawl_url(url="https://example.com")

    assert isinstance(response, Crawl4aiResponse)
    assert response.success is True
    assert response.markdown is not None
    assert "Example Domain" in response.markdown


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("CRAWL4AI_INTEGRATION"),
    reason="CRAWL4AI_INTEGRATION not defined",
)
@pytest.mark.asyncio
async def test_web_search_integration():
    """
    Real integration test for the web_search tool.
    Only runs if CRAWL4AI_INTEGRATION is defined and a browser is installed.
    """
    response = await web_search(query="Kaval AI agent framework", count=5)

    assert isinstance(response, WebSearchResponse)
    assert response.success is True
    assert len(response.results) <= 5
    assert all(r.url.startswith("http") for r in response.results)
