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

The ``crawl_url`` and ``web_search`` tools, driving a headless browser through
crawl4ai.

``crawl_url`` refuses a URL that fails :func:`kavalai.net.ensure_public_url`,
and withholds the content when the page the browser ended on fails it. The
browser runs in its own process and resolves names itself, so these are
pre- and post-checks, not a pinned connection: a redirect through a private
address in between, or a DNS answer that changes after the check, is not
seen. Full protection needs network egress control around the browser. An agent
meant to crawl an intranet is given ``make_crawl_url(allow_private_networks=True)``.

``web_search`` fetches a fixed DuckDuckGo endpoint and is not checked.
"""

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

from loguru import logger
from pydantic import BaseModel

from kavalai.functionkernel import pythontool
from kavalai.net import UnsafeUrlError, ensure_public_url

CRAWL4AI_INSTALL_HINT = (
    'crawl_url and web_search need crawl4ai: pip install "kavalai[webtools]"'
)

DUCKDUCKGO_HTML_ENDPOINT = "https://duckduckgo.com/html/"

# CSS extraction schema for the DuckDuckGo HTML result list.
DUCKDUCKGO_RESULT_SCHEMA = {
    "name": "DuckDuckGo results",
    "baseSelector": "div.result__body",
    "fields": [
        {"name": "title", "selector": "a.result__a", "type": "text"},
        {
            "name": "url",
            "selector": "a.result__a",
            "type": "attribute",
            "attribute": "href",
        },
        {"name": "snippet", "selector": "a.result__snippet", "type": "text"},
    ],
}


class Crawl4aiResponse(BaseModel):
    url: str
    success: bool
    markdown: Optional[str] = None
    html: Optional[str] = None
    status_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None


class WebSearchResponse(BaseModel):
    query: str
    success: bool
    results: List[WebSearchResult] = []
    error_message: Optional[str] = None


def _import_crawl4ai():
    """The crawl4ai package, imported on first use.

    crawl4ai is an optional dependency, so this module has to import without
    it; the install hint is raised only when a tool is actually called.
    """
    try:
        import crawl4ai
    except ImportError as error:
        raise ImportError(CRAWL4AI_INSTALL_HINT) from error
    return crawl4ai


def _resolve_result_url(href: Optional[str]) -> Optional[str]:
    """Turn a DuckDuckGo result href into the target page URL."""
    if not href:
        return None

    # Protocol-relative links (//duckduckgo.com/l/?uddg=...) need a scheme to parse.
    if href.startswith("//"):
        href = f"https:{href}"

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        # Redirect link: the real URL sits in the (already decoded) 'uddg' parameter.
        target = parse_qs(parsed.query).get("uddg")
        return target[0] if target else None

    return href if parsed.scheme in ("http", "https") else None


def _parse_search_results(extracted_content: Optional[str]) -> List[WebSearchResult]:
    """Convert the extracted DuckDuckGo JSON into search result models."""
    if not extracted_content:
        return []

    results = []
    for item in json.loads(extracted_content):
        url = _resolve_result_url(item.get("url"))
        title = (item.get("title") or "").strip()
        if not url or not title:
            continue
        snippet = (item.get("snippet") or "").strip() or None
        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
    return results


async def _refusal(url: str) -> Optional[str]:
    """Why ``url`` may not be crawled, or ``None`` when it may."""
    try:
        await ensure_public_url(url)
    except UnsafeUrlError as error:
        return f"Refused: {error}"
    return None


def _final_urls(result: Any, requested: str) -> List[str]:
    """The URLs crawl4ai reports ending on, other than the one requested."""
    reported = (getattr(result, "redirected_url", None), result.url)
    return [url for url in dict.fromkeys(reported) if url and url != requested]


def make_crawl_url(
    *, allow_private_networks: bool = False
) -> Callable[..., Awaitable[Crawl4aiResponse]]:
    """Build a ``crawl_url`` tool.

    The module-level :data:`crawl_url` is ``make_crawl_url()``.

    Args:
        allow_private_networks: Let the tool crawl private, loopback and
            link-local addresses, for an agent meant to read an intranet.
            ``False`` by default, which answers such a URL with
            ``success=False`` and a ``Refused: …`` error message.

    Returns:
        An ``@pythontool`` coroutine function, ready for
        ``FunctionKernel.register_python_tool``.
    """

    @pythontool
    async def crawl_url(
        url: str,
        include_html: bool = False,
        bypass_cache: bool = False,
        timeout: float = 60.0,
    ) -> Crawl4aiResponse:
        """
        Crawl a web page and return its content as clean Markdown using Crawl4AI.

        Args:
            url: The website URL to crawl.
            include_html: Whether to also return the cleaned HTML (default False).
            bypass_cache: Whether to bypass the crawler cache and fetch fresh content (default False).
            timeout: Page load timeout in seconds (default 60.0).
        """
        crawl4ai = _import_crawl4ai()
        if not allow_private_networks:
            refusal = await _refusal(url)
            if refusal:
                return Crawl4aiResponse(url=url, success=False, error_message=refusal)

        browser_config = crawl4ai.BrowserConfig(headless=True)
        run_config = crawl4ai.CrawlerRunConfig(
            cache_mode=(
                crawl4ai.CacheMode.BYPASS
                if bypass_cache
                else crawl4ai.CacheMode.ENABLED
            ),
            page_timeout=int(timeout * 1000),
        )

        logger.info(f"Crawling URL with Crawl4AI: {url}")
        async with crawl4ai.AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

        if not allow_private_networks:
            for final_url in _final_urls(result, url):
                refusal = await _refusal(final_url)
                if refusal:
                    return Crawl4aiResponse(
                        url=final_url,
                        success=False,
                        status_code=result.status_code,
                        error_message=f"Redirected to {final_url}. {refusal}",
                    )

        markdown = None
        if result.markdown is not None:
            markdown = getattr(result.markdown, "raw_markdown", str(result.markdown))

        return Crawl4aiResponse(
            url=result.url,
            success=result.success,
            markdown=markdown,
            html=result.cleaned_html if include_html else None,
            status_code=result.status_code,
            metadata=result.metadata,
            error_message=result.error_message,
        )

    return crawl_url


crawl_url = make_crawl_url()


@pythontool
async def web_search(
    query: str,
    count: int = 10,
    timeout: float = 60.0,
) -> WebSearchResponse:
    """
    Search the web with Crawl4AI by scraping the DuckDuckGo result page.

    Args:
        query: The search query.
        count: Maximum number of results to return (default 10).
        timeout: Page load timeout in seconds (default 60.0).
    """
    crawl4ai = _import_crawl4ai()

    search_url = f"{DUCKDUCKGO_HTML_ENDPOINT}?q={quote_plus(query)}"
    browser_config = crawl4ai.BrowserConfig(headless=True)
    run_config = crawl4ai.CrawlerRunConfig(
        cache_mode=crawl4ai.CacheMode.ENABLED,
        page_timeout=int(timeout * 1000),
        extraction_strategy=crawl4ai.JsonCssExtractionStrategy(
            DUCKDUCKGO_RESULT_SCHEMA
        ),
    )

    logger.info(f"Searching the web with Crawl4AI: {query}")
    async with crawl4ai.AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=search_url, config=run_config)

    if not result.success:
        return WebSearchResponse(
            query=query, success=False, error_message=result.error_message
        )

    results = _parse_search_results(result.extracted_content)
    logger.debug(f"Found {len(results)} search results for: {query}")
    return WebSearchResponse(query=query, success=True, results=results[:count])
