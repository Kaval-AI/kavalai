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

import os
from unittest.mock import patch, MagicMock
import pytest
import httpx
from kavalai.tools.websearch.langsearch import langsearch_web_search, SearchResponse


def test_langsearch_web_search_success():
    mock_response = {
        "_type": "SearchResponse",
        "queryContext": {"originalQuery": "test query"},
        "webPages": {
            "webSearchUrl": "https://api.langsearch.com/search?q=test+query",
            "totalEstimatedMatches": 1,
            "value": [
                {
                    "id": "1",
                    "name": "Test Page",
                    "url": "https://example.com",
                    "displayUrl": "example.com",
                    "snippet": "This is a test snippet",
                    "summary": "This is a test summary",
                    "datePublished": "2023-01-01",
                    "dateLastCrawled": "2023-01-02",
                }
            ],
            "someResultsRemoved": False,
        },
    }

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = MagicMock()

        result = langsearch_web_search(
            query="test query",
            api_key="test_key",
            summary=True,
            count=5,
            freshness="oneDay",
        )

        assert isinstance(result, SearchResponse)
        assert result.queryContext.originalQuery == "test query"
        assert len(result.webPages.value) == 1
        assert result.webPages.value[0].name == "Test Page"
        assert result.webPages.value[0].summary == "This is a test summary"

        # Verify request parameters
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["query"] == "test query"
        assert kwargs["json"]["summary"] is True
        assert kwargs["json"]["count"] == 5
        assert kwargs["json"]["freshness"] == "oneDay"
        assert kwargs["headers"]["Authorization"] == "Bearer test_key"


def test_langsearch_web_search_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="LangSearch API key not provided"):
            langsearch_web_search(query="test")


def test_langsearch_web_search_api_error():
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_post.return_value
        )

        with pytest.raises(httpx.HTTPStatusError):
            langsearch_web_search(query="test", api_key="invalid_key")


def test_langsearch_web_search_env_api_key():
    mock_response = {
        "_type": "SearchResponse",
        "queryContext": {"originalQuery": "test"},
        "webPages": {"totalEstimatedMatches": 0, "value": []},
    }
    with patch.dict(os.environ, {"LANGSEARCH_API_KEY": "env_key"}):
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(spec=httpx.Response)
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response

            langsearch_web_search(query="test")

            args, kwargs = mock_post.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer env_key"


@pytest.mark.skipif(
    not os.environ.get("LANGSEARCH_API_KEY"), reason="LANGSEARCH_API_KEY not provided"
)
def test_langsearch_web_search_integration():
    """Real API call test if key is available."""
    result = langsearch_web_search(query="kaval.ai", summary=True, count=5)

    assert isinstance(result, SearchResponse)
    # totalEstimatedMatches might be None or int
    assert result.webPages.value is not None
    if result.webPages.value:
        assert result.webPages.value[0].url.startswith("http")
        assert result.webPages.value[0].name
    for page in result.webPages.value:
        print(page.name, page.url, page.summary)


def test_langsearch_web_search_unwraps_data_envelope():
    """Some responses wrap the payload in a top-level "data" object."""
    payload = {
        "queryContext": {"originalQuery": "wrapped"},
        "webPages": {"value": []},
    }

    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(spec=httpx.Response)
        mock_post.return_value.json.return_value = {"code": 200, "data": payload}
        mock_post.return_value.raise_for_status = MagicMock()

        result = langsearch_web_search(query="wrapped", api_key="k")

    assert isinstance(result, SearchResponse)
    assert result.queryContext.originalQuery == "wrapped"
