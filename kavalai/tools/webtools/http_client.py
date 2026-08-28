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
from typing import Optional, Dict, Any, Union

import httpx
from pydantic import BaseModel

from kavalai.functionkernel import pythontool


class HttpResponse(BaseModel):
    status_code: int
    headers: Dict[str, str]
    text: str
    json_data: Optional[Union[Dict[str, Any], list]] = None


@pythontool
def http_request(
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
    method = method.upper()
    auth = None
    if auth_user and auth_password:
        auth = (auth_user, auth_password)

    proxy = None
    if use_proxy:
        proxy_host = os.environ.get("KAVALAI_TOR_PROXY_HOST", "localhost")
        proxy_port = os.environ.get("KAVALAI_TOR_PROXY_PORT", "8118")
        proxy = f"http://{proxy_host}:{proxy_port}"

    with httpx.Client(timeout=timeout, proxy=proxy, auth=auth) as client:
        response = client.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            json=json_body,
            content=data_body,
        )

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
