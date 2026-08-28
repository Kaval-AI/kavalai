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
import httpx
from kavalai.tools.webtools.http_client import http_request


def test_http_request_sends_basic_auth_only_when_both_parts_are_given():
    """Half a credential is no credential: a user without a password must not
    become an ``auth`` tuple with an empty secret."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = ""

    with patch("httpx.Client") as mock_client_class:
        mock_client_class.return_value.__enter__.return_value.request.return_value = (
            mock_response
        )
        http_request("get", "https://example.com", auth_user="u", auth_password="p")
        assert mock_client_class.call_args.kwargs["auth"] == ("u", "p")

        http_request("get", "https://example.com", auth_user="u")
        assert mock_client_class.call_args.kwargs["auth"] is None


def test_http_request_proxy():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "proxy ok"

    with patch.dict(
        os.environ,
        {"KAVALAI_TOR_PROXY_HOST": "proxy.host", "KAVALAI_TOR_PROXY_PORT": "9999"},
    ):
        with patch("httpx.Client") as mock_client_class:
            mock_client_instance = mock_client_class.return_value.__enter__.return_value
            mock_client_instance.request.return_value = mock_response

            response = http_request(
                method="GET", url="https://example.com", use_proxy=True
            )

            assert response.status_code == 200

            mock_client_class.assert_called_once()
            args, kwargs = mock_client_class.call_args
            assert kwargs["proxy"] == "http://proxy.host:9999"


def test_http_request_invalid_json():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "invalid json"
    mock_response.json.side_effect = ValueError("Invalid JSON")

    with patch("httpx.Client.request") as mock_request:
        mock_request.return_value = mock_response

        response = http_request(method="GET", url="https://example.com")

        assert response.status_code == 200
        assert response.json_data is None
        assert response.text == "invalid json"
