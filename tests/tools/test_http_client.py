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

import base64
import inspect

import pytest

from kavalai import FunctionKernel
from kavalai import net
from kavalai.functionkernel import FunctionKernelException
from kavalai.net import UnsafeUrlError
from kavalai.tools.webtools.http_client import (
    HttpResponse,
    _tor_proxy_url,
    http_request,
    make_http_request,
)
from tests.tools.local_http import fake_resolver, local_server

intranet_request = make_http_request(allow_private_networks=True)


@pytest.fixture(autouse=True)
def no_real_dns(monkeypatch):
    monkeypatch.setattr(net, "resolve_host", fake_resolver)


async def test_refuses_a_loopback_server_by_default():
    with local_server() as server:
        with pytest.raises(UnsafeUrlError, match="127.0.0.1"):
            await http_request("GET", f"http://127.0.0.1:{server.port}/")
    assert server.requests == []


async def test_allow_private_networks_reaches_it():
    with local_server() as server:
        response = await intranet_request(
            "post",
            f"http://127.0.0.1:{server.port}/orders",
            params={"page": 2},
            json_body={"item": "rye"},
        )
    assert isinstance(response, HttpResponse)
    assert response.status_code == 200
    assert response.json_data["method"] == "POST"
    assert response.json_data["path"] == "/orders?page=2"
    assert response.json_data["body"] == '{"item":"rye"}'


async def test_sends_basic_auth_only_when_both_parts_are_given():
    """Half a credential is no credential: a user without a password must not
    become an ``auth`` tuple with an empty secret."""
    with local_server() as server:
        url = f"http://127.0.0.1:{server.port}/"
        both = await intranet_request("get", url, auth_user="u", auth_password="p")
        half = await intranet_request("get", url, auth_user="u")
    expected = "Basic " + base64.b64encode(b"u:p").decode()
    assert both.json_data["authorization"] == expected
    assert half.json_data["authorization"] is None


async def test_text_that_is_not_json_leaves_json_data_empty():
    with local_server() as server:
        response = await intranet_request(
            "GET", f"http://127.0.0.1:{server.port}/text", data_body="x"
        )
    assert response.text == "invalid json"
    assert response.json_data is None


async def test_proxy_is_allowed_on_a_private_address(monkeypatch):
    """The proxy resolves the name, so the target passes a pre-check only."""
    with local_server() as proxy:
        monkeypatch.setenv("KAVALAI_TOR_PROXY_HOST", "127.0.0.1")
        monkeypatch.setenv("KAVALAI_TOR_PROXY_PORT", str(proxy.port))
        response = await http_request("GET", "http://public.test/page", use_proxy=True)
    assert response.status_code == 200
    assert proxy.requests[0]["path"] == "http://public.test/page"


async def test_proxy_target_is_pre_checked(monkeypatch):
    with local_server() as proxy:
        monkeypatch.setenv("KAVALAI_TOR_PROXY_HOST", "127.0.0.1")
        monkeypatch.setenv("KAVALAI_TOR_PROXY_PORT", str(proxy.port))
        with pytest.raises(UnsafeUrlError):
            await http_request("GET", "http://internal.test/", use_proxy=True)
        assert proxy.requests == []
        response = await intranet_request(
            "GET", "http://internal.test/", use_proxy=True
        )
    assert response.status_code == 200
    assert proxy.requests[0]["path"] == "http://internal.test/"


def test_proxy_defaults(monkeypatch):
    monkeypatch.delenv("KAVALAI_TOR_PROXY_HOST", raising=False)
    monkeypatch.delenv("KAVALAI_TOR_PROXY_PORT", raising=False)
    assert _tor_proxy_url() == "http://localhost:8118"


def test_the_model_cannot_switch_the_guard_off():
    """The opt-out is a registration argument, never a tool argument."""
    for tool in (http_request, intranet_request):
        assert tool._is_kavalai_tool is True
        assert tool.__name__ == "http_request"
        assert "allow_private_networks" not in inspect.signature(tool).parameters
    assert intranet_request is not make_http_request(allow_private_networks=True)


async def test_through_the_kernel():
    kernel = FunctionKernel()
    kernel.register_python_tool("http.request", http_request)
    kernel.register_python_tool("intranet.request", intranet_request)
    with local_server() as server:
        url = f"http://127.0.0.1:{server.port}/"
        with pytest.raises(FunctionKernelException, match="non-public address"):
            await kernel.call_tool(
                "python://http.request", {"method": "GET", "url": url}
            )
        response = await kernel.call_tool(
            "python://intranet.request", {"method": "GET", "url": url}
        )
    assert response.status_code == 200
