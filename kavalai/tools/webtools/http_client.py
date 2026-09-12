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

The ``http_request`` tool: one HTTP request to a URL the model chooses.

``http_request`` refuses private, loopback, link-local and cloud metadata
targets: it connects through :class:`kavalai.net.PublicOnlyTransport`, which
checks every address it dials, on every redirect hop, and dials the address it
checked. An agent meant to call an intranet API is given a tool built by
``make_http_request(allow_private_networks=True)`` instead. The switch belongs
to whoever registers the tool, not to the model, which is why it is a factory
argument rather than a tool argument.

The guarded client connects directly and ignores the ``HTTP_PROXY`` /
``HTTPS_PROXY`` variables, since a proxy would resolve the name itself.

With ``use_proxy=True`` the request goes to the proxy at
``KAVALAI_TOR_PROXY_HOST`` / ``KAVALAI_TOR_PROXY_PORT``, which is allowed
whatever its address. The proxy resolves the target name, so only the pre-check
:func:`kavalai.net.ensure_public_url` applies: the name is resolved locally
once, and the proxy may receive a different answer. The local lookup also means
the name is visible to the local resolver, not only to Tor.
"""

import os
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import httpx
from pydantic import BaseModel

from kavalai.functionkernel import pythontool
from kavalai.net import PublicOnlyTransport, ensure_public_url


class HttpResponse(BaseModel):
    status_code: int
    headers: Dict[str, str]
    text: str
    json_data: Optional[Union[Dict[str, Any], list]] = None


def _tor_proxy_url() -> str:
    host = os.environ.get("KAVALAI_TOR_PROXY_HOST", "localhost")
    port = os.environ.get("KAVALAI_TOR_PROXY_PORT", "8118")
    return f"http://{host}:{port}"


async def _open_client(
    url: httpx.URL,
    *,
    use_proxy: bool,
    allow_private_networks: bool,
    timeout: float,
    auth: Optional[tuple[str, str]],
) -> httpx.AsyncClient:
    """The client for one request, guarded unless the tool allows private targets.

    Raises:
        kavalai.net.UnsafeUrlError: ``use_proxy`` is set and ``url`` fails the
            pre-check.
    """
    if use_proxy:
        if not allow_private_networks:
            await ensure_public_url(str(url))
        return httpx.AsyncClient(timeout=timeout, auth=auth, proxy=_tor_proxy_url())
    if allow_private_networks:
        return httpx.AsyncClient(timeout=timeout, auth=auth)
    return httpx.AsyncClient(
        timeout=timeout, auth=auth, transport=PublicOnlyTransport()
    )


def _to_response(response: httpx.Response) -> HttpResponse:
    try:
        json_data = response.json()
    except Exception:
        json_data = None
    return HttpResponse(
        status_code=response.status_code,
        headers=dict(response.headers),
        text=response.text,
        json_data=json_data,
    )


def make_http_request(
    *, allow_private_networks: bool = False
) -> Callable[..., Awaitable[HttpResponse]]:
    """Build an ``http_request`` tool.

    The module-level :data:`http_request` is ``make_http_request()``.

    Args:
        allow_private_networks: Let the tool reach private, loopback and
            link-local addresses, for an agent meant to call an intranet API.
            ``False`` by default, which refuses them with
            :class:`kavalai.net.UnsafeUrlError`.

    Returns:
        An ``@pythontool`` coroutine function, ready for
        ``FunctionKernel.register_python_tool``.
    """

    @pythontool
    async def http_request(
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Union[Dict[str, Any], list]] = None,
        data_body: Optional[str] = None,
        auth_user: Optional[str] = None,
        auth_password: Optional[str] = None,
        timeout: float = 30.0,
        use_proxy: bool = False,
    ) -> HttpResponse:
        """
        Perform an HTTP request (GET, POST, PUT, DELETE, etc.).

        Args:
            method: HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
            url: The URL to send the request to.
            params: Optional query parameters.
            headers: Optional HTTP headers.
            json_body: Optional JSON body for POST/PUT requests.
            data_body: Optional raw string body for POST/PUT requests.
            auth_user: Optional username for Basic Authentication.
            auth_password: Optional password for Basic Authentication.
            timeout: Request timeout in seconds (default 30.0).
            use_proxy: Whether to use the configured TOR proxy (default False).
        """
        auth = (auth_user, auth_password) if auth_user and auth_password else None
        request_url = httpx.URL(url)
        client = await _open_client(
            request_url,
            use_proxy=use_proxy,
            allow_private_networks=allow_private_networks,
            timeout=timeout,
            auth=auth,
        )
        async with client:
            response = await client.request(
                method=method.upper(),
                url=request_url,
                params=params,
                headers=headers,
                json=json_body,
                content=data_body,
            )
        return _to_response(response)

    return http_request


http_request = make_http_request()
