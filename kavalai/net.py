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

Outbound requests to URLs a model chooses: the guard against SSRF.

A bundled web tool fetches whatever URL a model composes, and a model can be
talked into composing ``http://169.254.169.254/`` or ``http://localhost:5432/``.
This module refuses such targets.

- :func:`is_public_address` decides whether one IP address is globally
  routable unicast.
- :func:`ensure_public_url` checks a URL: scheme, authority, host name, and
  every address the host resolves to. It is a *pre-check* — by itself it is
  defeated by DNS rebinding, because the client that fetches the URL resolves
  the name a second time and may receive a different answer.
- :class:`PublicOnlyTransport` closes that gap for httpx. Its network backend
  resolves the host, checks every address and then connects to the address it
  checked, so no second lookup takes place. TLS still verifies the certificate
  for the host *name* and sends it as SNI, and the ``Host`` header is the one in
  the URL. Each redirect hop is a new request through the same transport, so it
  is checked in the same way.

The module is not imported by ``kavalai/__init__.py``: it needs httpx, and
``import kavalai`` has to work under Pyodide without it.
"""

import asyncio
import ipaddress
import socket
import typing
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import SplitResult, urlsplit

import httpcore
import httpx

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

Resolver = Callable[[str], Awaitable[list[str]]]
"""An ASCII host name to the IP addresses it resolves to, in preference order.

An empty list means the name does not resolve. :func:`resolve_host` is the
default; tests pass their own.
"""

METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(address)
    for address in (
        "169.254.169.254",
        "169.254.170.2",
        "168.63.129.16",
        "100.100.100.200",
        "fd00:ec2::254",
    )
)
"""Cloud metadata and platform endpoints.

AWS, GCP, Azure, Oracle and DigitalOcean serve instance metadata at
``169.254.169.254``, ECS task credentials at ``169.254.170.2``, Alibaba at
``100.100.100.200`` and AWS over IPv6 at ``fd00:ec2::254``. All of these fall
in ranges refused anyway; they are listed so the refusal does not depend on the
interpreter's address tables. Azure's platform endpoint ``168.63.129.16`` is a
*global* address, so without this entry it would be allowed.
"""

BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "100.64.0.0/10",
        "192.0.0.0/24",
        "198.18.0.0/15",
        "fec0::/10",
        "3fff::/20",
        "5f00::/16",
    )
)
"""Networks refused in addition to what :mod:`ipaddress` calls non-global.

``0.0.0.0/8`` (reaches the local host on Linux), carrier-grade NAT
``100.64.0.0/10``, IETF protocol assignments ``192.0.0.0/24``, benchmarking
``198.18.0.0/15``, deprecated IPv6 site-local ``fec0::/10`` (global to Python
3.12), IPv6 documentation ``3fff::/20`` (RFC 9637, global to Python 3.12) and
SRv6 segment identifiers ``5f00::/16``.
"""

NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
IPV4_COMPATIBLE_PREFIX = ipaddress.ip_network("::/96")

BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata"})
BLOCKED_SUFFIXES = (".localhost", ".internal", ".local")


class UnsafeUrlError(ValueError):
    """A URL, or an address its host resolves to, is not a public target."""


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """The IPv4 addresses an IPv6 address carries, for the forms that tunnel.

    IPv4-compatible (``::a.b.c.d``), 6to4 (``2002::/16``) and Teredo
    (``2001::/32``, server and client). IPv4-mapped and NAT64 addresses are
    handled by :func:`is_public_address` directly, since they *are* the IPv4
    host rather than a tunnel to it.
    """
    embedded = []
    if ip in IPV4_COMPATIBLE_PREFIX:
        embedded.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if ip.sixtofour is not None:
        embedded.append(ip.sixtofour)
    if ip.teredo is not None:
        embedded.extend(ip.teredo)
    return embedded


def is_public_address(address: str | IPAddress) -> bool:
    """Whether ``address`` is globally routable unicast.

    Refuses loopback, private, link-local, multicast, reserved, unspecified,
    carrier-grade NAT, benchmarking and documentation ranges, ``0.0.0.0/8`` and
    the cloud metadata endpoints in :data:`METADATA_ADDRESSES`.

    An IPv4-mapped address (``::ffff:a.b.c.d``) and a NAT64 address
    (``64:ff9b::a.b.c.d``) are judged as the IPv4 address they reach. An
    IPv6 address that tunnels to an IPv4 one — IPv4-compatible, 6to4, Teredo —
    is refused when that IPv4 address is not public, and is otherwise judged as
    an IPv6 address.

    Args:
        address: An address as text or as an :mod:`ipaddress` object. A zone
            index (``fe80::1%eth0``) is ignored.

    Raises:
        ValueError: ``address`` is not an IP address.
    """
    ip = _as_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_address(ip.ipv4_mapped)
        if ip in NAT64_PREFIX:
            return is_public_address(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if not all(is_public_address(inner) for inner in _embedded_ipv4(ip)):
            return False
        if ip.is_site_local:
            return False
    if ip in METADATA_ADDRESSES or any(ip in net for net in BLOCKED_NETWORKS):
        return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _as_address(address: str | IPAddress) -> IPAddress:
    if isinstance(address, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        address = str(address)
    return ipaddress.ip_address(address.split("%", 1)[0])


def _inet_aton_part(part: str) -> int | None:
    """One dotted part in the forms ``inet_aton`` accepts: 0x hex, 0 octal, decimal."""
    if part[:2].lower() == "0x":
        digits, base, alphabet = part[2:], 16, "0123456789abcdefABCDEF"
    elif len(part) > 1 and part[0] == "0":
        digits, base, alphabet = part[1:], 8, "01234567"
    else:
        digits, base, alphabet = part, 10, "0123456789"
    if not digits or any(c not in alphabet for c in digits):
        return None
    return int(digits, base)


def _parse_inet_aton(host: str) -> ipaddress.IPv4Address | None:
    """``host`` read the way the C library reads a numeric IPv4 host.

    ``127.0.0.1``, ``2130706433``, ``0x7f.1``, ``0177.0.0.01`` and ``127.1`` all
    name the loopback address: one to four parts, each decimal, octal or hex,
    the last part filling the remaining bytes.
    """
    parts = host.split(".")
    if len(parts) > 4:
        return None
    values = [_inet_aton_part(part) for part in parts]
    if any(value is None for value in values):
        return None
    *leading, last = values
    if any(value > 0xFF for value in leading) or last >= 256 ** (4 - len(leading)):
        return None
    number = last
    for position, value in enumerate(leading):
        number |= value << (8 * (3 - position))
    return ipaddress.IPv4Address(number)


def parse_ip_literal(host: str) -> IPAddress | None:
    """The address ``host`` spells out, or ``None`` when it is a name.

    Accepts IPv6 with or without brackets and IPv4 in every form the C
    library's ``inet_aton`` accepts: dotted, decimal, octal, hex and the short
    forms (``127.1``). One trailing dot is ignored.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif host.endswith("."):
        host = host[:-1]
    if ":" in host:
        try:
            return _as_address(host)
        except ValueError:
            return None
    return _parse_inet_aton(host) if host else None


async def resolve_host(host: str) -> list[str]:
    """``host``'s addresses from the system resolver, in its preference order.

    Returns an empty list when the name does not resolve.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return []
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _ascii_host(host: str) -> str:
    """``host`` lower-cased and, when it is not ASCII, IDNA-encoded.

    The encoding applies UTS #46 mapping, as a browser does, so a fullwidth
    ``ｌｏｃａｌｈｏｓｔ`` becomes ``localhost`` before the name checks run.
    """
    import idna

    host = host.rstrip(".")
    if host.isascii():
        return host.lower()
    try:
        return idna.encode(host, uts46=True).decode("ascii")
    except idna.IDNAError as error:
        raise UnsafeUrlError(f"{host!r} is not a valid host name: {error}") from error


def _check_host_name(host: str) -> None:
    first_label = host.split(".", 1)[0]
    if (
        host in BLOCKED_HOSTNAMES
        or first_label == "metadata"
        or host.endswith(BLOCKED_SUFFIXES)
    ):
        raise UnsafeUrlError(f"{host} is an internal host name")


def _split_url(url: str) -> SplitResult:
    """``url`` split, with everything refused that does not name a host plainly.

    ``urlsplit`` accepts an out-of-range port and raises only when ``port`` is
    read, so it is read here.
    """
    if any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        raise UnsafeUrlError("the URL contains whitespace or control characters")
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError as error:
        raise UnsafeUrlError(f"{url!r} is not a valid URL: {error}") from error
    if parts.scheme.lower() not in ("http", "https"):
        raise UnsafeUrlError(f"scheme {parts.scheme!r} is not allowed")
    if "@" in parts.netloc:
        raise UnsafeUrlError("credentials in the URL are not allowed")
    if "\\" in parts.netloc:
        raise UnsafeUrlError("a backslash in the host is not allowed")
    if not parts.hostname:
        raise UnsafeUrlError(f"{url!r} has no host")
    return parts


async def _vetted_addresses(host: str, resolver: Resolver) -> list[str]:
    """Every address ``host`` stands for, each one checked.

    Raises:
        UnsafeUrlError: An address is not public.
    """
    literal = parse_ip_literal(host)
    if literal is not None:
        address = str(literal)
        if not is_public_address(address):
            spelled = "" if host == address else f" ({host})"
            raise UnsafeUrlError(f"{address}{spelled} is a non-public address")
        return [address]
    addresses = await resolver(host)
    for address in addresses:
        if not is_public_address(address):
            raise UnsafeUrlError(f"{host} resolves to non-public address {address}")
    return addresses


async def ensure_public_url(url: str, *, resolver: Resolver | None = None) -> str:
    """Return ``url`` unchanged when it points at a public host; raise otherwise.

    Refuses a scheme other than ``http``/``https``; credentials, a backslash or
    whitespace in the URL, which different URL parsers read differently;
    ``localhost`` and ``*.localhost``; a host whose first label is
    ``metadata``; ``*.internal`` and ``*.local``; and a host with any address
    for which :func:`is_public_address` is false. IP literals are read in every
    encoding (``http://2130706433/`` is ``127.0.0.1``) and never sent to the
    resolver; non-ASCII names are IDNA-encoded first.

    This is a pre-check: the client that fetches the URL resolves the name
    again. Use :class:`PublicOnlyTransport` where the connection itself is
    under the SDK's control.

    Args:
        url: The absolute URL to check.
        resolver: Resolves a host name; :func:`resolve_host` by default.

    Raises:
        UnsafeUrlError: The URL or one of its addresses is not public, or the
            host does not resolve.
    """
    parts = _split_url(url)
    host = _ascii_host(parts.hostname)
    _check_host_name(host)
    addresses = await _vetted_addresses(host, resolver or resolve_host)
    if not addresses:
        raise UnsafeUrlError(f"{host} does not resolve")
    return url


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """An httpcore network backend that dials only addresses it has checked.

    httpcore hands :meth:`connect_tcp` the host name from the URL; TLS and the
    ``Host`` header are handled above this layer with that same name, so
    replacing the name with a vetted address here changes only where the
    socket goes.
    """

    def __init__(self, resolver: Resolver, backend: httpcore.AsyncNetworkBackend):
        self._resolver = resolver
        self._backend = backend

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[typing.Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        name = _ascii_host(host)
        _check_host_name(name)
        addresses = await _vetted_addresses(name, self._resolver)
        if not addresses:
            raise httpcore.ConnectError(f"{name} does not resolve")
        error: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as failure:
                error = failure
        raise error

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class PublicOnlyTransport(httpx.AsyncHTTPTransport):
    """An httpx transport that connects only to public addresses.

    Every connection resolves its host once, refuses it unless every address is
    public (:func:`is_public_address`), and dials the address it checked, so a
    DNS answer that changes between the check and the connection cannot redirect
    it. The certificate is still verified against the host name, which is also
    sent as SNI. Redirects are followed by httpx as new requests through this
    transport, so each hop is checked. A refusal raises :class:`UnsafeUrlError`
    from the request.

    ``verify``, ``cert``, ``trust_env``, ``http1``, ``http2``, ``limits`` and
    ``retries`` mean what they mean for :class:`httpx.AsyncHTTPTransport`.
    There is no ``proxy``: through a proxy the proxy resolves the name, so the
    guard could not apply. The constructor builds its own connection pool
    rather than calling the parent's, because that one offers no way to choose
    the network backend.

    Args:
        resolver: Resolves a host name; :func:`resolve_host` by default.
        network_backend: The httpcore backend that opens the socket to the
            vetted address; :class:`httpcore.AnyIOBackend` by default.
    """

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        network_backend: httpcore.AsyncNetworkBackend | None = None,
        verify: typing.Any = True,
        cert: typing.Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: httpx.Limits | None = None,
        retries: int = 0,
    ) -> None:
        limits = limits or httpx.Limits(
            max_connections=100, max_keepalive_connections=20
        )
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(
                verify=verify, cert=cert, trust_env=trust_env
            ),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=http1,
            http2=http2,
            retries=retries,
            network_backend=_PinnedNetworkBackend(
                resolver or resolve_host, network_backend or httpcore.AnyIOBackend()
            ),
        )
