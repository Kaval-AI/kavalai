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
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from kavalai.tools.webtools.crawl4ai import (
    crawl_url,
    web_search,
    Crawl4aiResponse,
    WebSearchResponse,
)


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
