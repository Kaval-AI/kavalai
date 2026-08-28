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
from kavalai.tools.webtools.http_client import http_request, HttpResponse


def test_http_request_get_success():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.text = '{"status": "ok"}'
    mock_response.json.return_value = {"status": "ok"}

    with patch("httpx.Client.request") as mock_request:
        mock_request.return_value = mock_response

        response = http_request(
            method="GET",
            url="https://example.com/api",
            params={"q": "test"},
            headers={"X-Test": "value"},
        )

        assert isinstance(response, HttpResponse)
        assert response.status_code == 200
        assert response.json_data == {"status": "ok"}
        assert response.headers["Content-Type"] == "application/json"

        mock_request.assert_called_once_with(
            method="GET",
            url="https://example.com/api",
            params={"q": "test"},
            headers={"X-Test": "value"},
            json=None,
            content=None,
        )


def test_http_request_post_json():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 201
    mock_response.headers = {}
    mock_response.text = '{"id": 1}'
    mock_response.json.return_value = {"id": 1}

    with patch("httpx.Client.request") as mock_request:
        mock_request.return_value = mock_response

        response = http_request(
            method="POST", url="https://example.com/api", json_body={"name": "test"}
        )

        assert response.status_code == 201
        assert response.json_data == {"id": 1}

        mock_request.assert_called_once_with(
            method="POST",
            url="https://example.com/api",
            params=None,
            headers=None,
            json={"name": "test"},
            content=None,
        )


def test_http_request_basic_auth():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.text = "auth success"
    mock_response.json.side_effect = Exception("not json")

    with patch("httpx.Client.request") as mock_request:
        mock_request.return_value = mock_response

        # We need to mock Client's __init__ but also let it be a context manager
        # Easier to mock the whole Client context manager?
        with patch("httpx.Client") as mock_client_class:
            mock_client_instance = mock_client_class.return_value.__enter__.return_value
            mock_client_instance.request.return_value = mock_response

            response = http_request(
                method="GET",
                url="https://example.com/protected",
                auth_user="user1",
                auth_password="password1",
            )

            assert response.status_code == 200

            # Check if Client was instantiated with auth
            mock_client_class.assert_called_once()
            args, kwargs = mock_client_class.call_args
            assert kwargs["auth"] == ("user1", "password1")


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
