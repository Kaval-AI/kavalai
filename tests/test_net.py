"""The SSRF guard: which addresses are public, which URLs pass, and the pinned transport.

The transport tests run against a real server on 127.0.0.1. A "public" host is
a name the fake resolver maps to a public address; the recording backend dials
the local server when asked for it, so every test observes exactly which
address the guard chose to connect to.
"""

import asyncio
import ipaddress
import socket
import ssl
import string

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kavalai import net
from kavalai.net import (
    PublicOnlyTransport,
    UnsafeUrlError,
    ensure_public_url,
    is_public_address,
    parse_ip_literal,
    resolve_host,
)
from tests.tools.local_http import (
    DEAD_PUBLIC_IP,
    OTHER_PUBLIC_IP,
    PUBLIC_IP,
    RecordingBackend,
    fake_resolver,
    local_server,
    make_certificate,
)

NON_PUBLIC_NETWORKS = [
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
        "2001:db8::/32",
        "3fff::/20",
    )
]
NAT64 = int(ipaddress.ip_address("64:ff9b::"))


async def refuse_to_resolve(host: str) -> list[str]:
    raise AssertionError(f"an IP literal was sent to the resolver: {host}")


def ipv4_addresses():
    return st.integers(0, 2**32 - 1).map(ipaddress.IPv4Address)


def addresses_in(network):
    first, last = int(network.network_address), int(network.broadcast_address)
    return st.integers(first, last).map(
        lambda number: (
            ipaddress.ip_address(number)
            if network.version == 4
            else ipaddress.IPv6Address(number)
        )
    )


def non_public_ipv4():
    return st.sampled_from([n for n in NON_PUBLIC_NETWORKS if n.version == 4]).flatmap(
        addresses_in
    )


def encode_part(value: int, form: str, upper: bool) -> str:
    if form == "hex":
        text = f"0x{value:x}"
        return text.upper() if upper else text
    if form == "oct":
        return f"0{value:o}"
    return str(value)


@st.composite
def ipv4_spellings(draw):
    """An IPv4 address and one of the ways ``inet_aton`` accepts it."""
    address = draw(ipv4_addresses())
    parts = draw(st.integers(1, 4))
    raw = address.packed
    values = list(raw[: parts - 1]) + [int.from_bytes(raw[parts - 1 :], "big")]
    forms = draw(
        st.lists(st.sampled_from(["dec", "oct", "hex"]), min_size=parts, max_size=parts)
    )
    upper = draw(st.booleans())
    spelling = ".".join(
        encode_part(v, f, upper) for v, f in zip(values, forms, strict=True)
    )
    return address, spelling


def run(coroutine):
    return asyncio.run(coroutine)


class TestIsPublicAddress:
    @pytest.mark.parametrize(
        "address,public",
        [
            ("93.184.216.34", True),
            ("8.8.8.8", True),
            ("2606:4700:4700::1111", True),
            ("10.0.0.1", False),
            ("127.0.0.1", False),
            ("169.254.169.254", False),
            ("169.254.170.2", False),
            ("168.63.129.16", False),
            ("100.100.100.200", False),
            ("fd00:ec2::254", False),
            ("192.168.1.1", False),
            ("0.0.0.0", False),
            ("0.1.2.3", False),
            ("255.255.255.255", False),
            ("224.0.0.1", False),
            ("100.64.0.1", False),
            ("198.18.0.1", False),
            ("192.0.2.1", False),
            ("::", False),
            ("::1", False),
            ("fe80::1", False),
            ("fe80::1%eth0", False),
            ("fec0::1", False),
            ("3fff::1", False),
            ("5f00::1", False),
            ("ff02::1", False),
            ("2001:db8::1", False),
            ("::ffff:10.0.0.1", False),
            ("::ffff:8.8.8.8", True),
            ("::127.0.0.1", False),
            ("::8.8.8.8", False),
            ("64:ff9b::7f00:1", False),
            ("64:ff9b::808:808", True),
            ("2002:7f00:1::1", False),
            ("2001:0:4136:e378:8000:63bf:3fff:fdd2", False),
        ],
    )
    def test_examples(self, address, public):
        assert is_public_address(address) is public

    def test_accepts_address_objects(self):
        assert is_public_address(ipaddress.ip_address("8.8.8.8")) is True
        assert is_public_address(ipaddress.ip_address("::1")) is False

    def test_refuses_what_is_not_an_address(self):
        with pytest.raises(ValueError):
            is_public_address("example.com")

    @given(st.sampled_from(NON_PUBLIC_NETWORKS).flatmap(addresses_in))
    def test_every_address_in_a_non_public_network_is_refused(self, address):
        assert not is_public_address(address)

    @given(ipv4_addresses())
    def test_mapped_and_nat64_are_judged_as_the_ipv4_address(self, address):
        mapped = ipaddress.IPv6Address((0xFFFF << 32) | int(address))
        nat64 = ipaddress.IPv6Address(NAT64 | int(address))
        assert is_public_address(mapped) == is_public_address(address)
        assert is_public_address(nat64) == is_public_address(address)

    @given(non_public_ipv4(), st.integers(0, 2**80 - 1), st.integers(0, 2**32 - 1))
    def test_tunnels_to_a_non_public_ipv4_address_are_refused(
        self, address, suffix, server
    ):
        compatible = ipaddress.IPv6Address(int(address))
        sixtofour = ipaddress.IPv6Address(
            (0x2002 << 112) | (int(address) << 80) | suffix
        )
        teredo = ipaddress.IPv6Address(
            (0x20010000 << 96) | (server << 64) | (~int(address) & 0xFFFFFFFF)
        )
        assert not is_public_address(compatible)
        assert not is_public_address(sixtofour)
        assert not is_public_address(teredo)


class TestParseIpLiteral:
    @pytest.mark.parametrize(
        "host,expected",
        [
            ("127.0.0.1", "127.0.0.1"),
            ("127.0.0.1.", "127.0.0.1"),
            ("2130706433", "127.0.0.1"),
            ("0x7f000001", "127.0.0.1"),
            ("0x7f.1", "127.0.0.1"),
            ("0177.0.0.01", "127.0.0.1"),
            ("127.1", "127.0.0.1"),
            ("1.256", "1.0.1.0"),
            ("[::1]", "::1"),
            ("::ffff:127.0.0.1", "::ffff:7f00:1"),
            ("example.com", None),
            ("08.1", None),
            ("0x", None),
            ("256.1", None),
            ("1.2.3.4.5", None),
            ("1..2", None),
            ("", None),
            ("[::1", None),
            ("::g", None),
            ("١٢٧.0.0.1", None),
        ],
    )
    def test_examples(self, host, expected):
        literal = parse_ip_literal(host)
        assert (None if literal is None else str(literal)) == expected

    @given(ipv4_spellings())
    def test_every_inet_aton_spelling_names_the_same_address(self, case):
        address, spelling = case
        assert parse_ip_literal(spelling) == address


class TestEnsurePublicUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/",
            "http://LOCALHOST./",
            "http://api.localhost/",
            "http://127.0.0.1/",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://metadata/",
            "http://metadata.google.internal/",
            "http://db.internal/",
            "http://printer.local/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://2130706433/",
            "http://0x7f.1/",
            "http://internal.test/",
            "http://half.test/",
            "http://mapped.test/",
            "http://nowhere.test/",
            "ftp://public.test/",
            "file:///etc/passwd",
            "http://10.0.0.1:8080/",
            "http://user@public.test/",
            "http://public.test@127.0.0.1/",
            "http://public.test\\@127.0.0.1/",
            "http://127.0.0.1\\.public.test/",
            "http://public .test/",
            "http://public.test\t/",
            "http:///path",
            "http://[::1",
            "http://public.test:99999/",
            "http://xn--/",
        ],
    )
    def test_refuses(self, url):
        with pytest.raises(UnsafeUrlError):
            run(ensure_public_url(url, resolver=fake_resolver))

    @pytest.mark.parametrize(
        "url",
        [
            "https://public.test/x?y=1",
            "HTTP://PUBLIC.TEST/",
            "https://93.184.216.34/",
            "http://[2606:4700:4700::1111]:8080/",
        ],
    )
    def test_allows_public_hosts_and_returns_the_url(self, url):
        assert run(ensure_public_url(url, resolver=fake_resolver)) == url

    def test_refusal_names_the_address(self):
        with pytest.raises(UnsafeUrlError, match="non-public address 127.0.0.1"):
            run(ensure_public_url("http://half.test/", resolver=fake_resolver))

    def test_unsafe_url_error_is_a_value_error(self):
        assert issubclass(UnsafeUrlError, ValueError)

    def test_international_names_are_resolved_in_punycode(self):
        asked = []

        async def resolver(host):
            asked.append(host)
            return [PUBLIC_IP]

        run(ensure_public_url("https://bücher.example/", resolver=resolver))
        assert asked == ["xn--bcher-kva.example"]

    def test_invalid_international_name_is_refused(self):
        with pytest.raises(UnsafeUrlError, match="not a valid host name"):
            run(ensure_public_url("http://a‍_b.example/", resolver=fake_resolver))

    @given(ipv4_spellings())
    def test_ip_literals_are_judged_without_the_resolver(self, case):
        address, spelling = case
        url = f"http://{spelling}/"
        if is_public_address(address):
            assert run(ensure_public_url(url, resolver=refuse_to_resolve)) == url
        else:
            with pytest.raises(UnsafeUrlError):
                run(ensure_public_url(url, resolver=refuse_to_resolve))

    @given(
        st.text(alphabet=string.printable.translate({ord(c): None for c in "/?#\\"})),
        st.none() | st.text(alphabet=string.ascii_letters + string.digits),
    )
    def test_any_userinfo_is_refused(self, user, password):
        userinfo = user if password is None else f"{user}:{password}"
        with pytest.raises(UnsafeUrlError):
            run(
                ensure_public_url(
                    f"http://{userinfo}@public.test/", resolver=fake_resolver
                )
            )

    @given(
        st.sampled_from(
            ["localhost", "metadata", "metadata.google.internal", "db.internal"]
        ),
        st.lists(
            st.sampled_from(["lower", "upper", "fullwidth"]), min_size=30, max_size=30
        ),
        st.sampled_from(["", "."]),
        st.sampled_from([".", "。", "．"]),
    )
    def test_blocked_names_are_refused_in_any_spelling(
        self, name, styles, trailing, dot
    ):
        def restyle(char, style):
            if char == "." or style == "lower":
                return dot if char == "." else char
            if style == "upper":
                return char.upper()
            return chr(ord(char.upper()) - 0x21 + 0xFF01)

        host = "".join(
            restyle(c, s) for c, s in zip(name, styles[: len(name)], strict=True)
        )
        host += trailing

        async def public(_host):
            return [PUBLIC_IP]

        with pytest.raises(UnsafeUrlError):
            run(ensure_public_url(f"http://{host}/", resolver=public))


class TestResolveHost:
    async def test_localhost_resolves_to_loopback(self):
        assert "127.0.0.1" in await resolve_host("localhost")

    async def test_a_failed_lookup_is_an_empty_list(self, monkeypatch):
        async def fail(*args, **kwargs):
            raise socket.gaierror("no such name")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fail)
        assert await resolve_host("nowhere.test") == []

    async def test_duplicates_are_dropped_in_order(self, monkeypatch):
        async def answers(*args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.2", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.2", 0)),
            ]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", answers)
        assert await resolve_host("two.test") == ["192.0.2.2", "192.0.2.1"]

    async def test_is_the_default_resolver(self, monkeypatch):
        monkeypatch.setattr(net, "resolve_host", fake_resolver)
        assert await ensure_public_url("http://public.test/") == "http://public.test/"


def guarded_client(backend, resolver=fake_resolver, **transport_options):
    transport = PublicOnlyTransport(
        resolver=resolver, network_backend=backend, **transport_options
    )
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


class TestPublicOnlyTransport:
    async def test_refuses_a_loopback_server_by_default(self):
        with local_server() as server:
            async with httpx.AsyncClient(transport=PublicOnlyTransport()) as client:
                for url in (
                    f"http://127.0.0.1:{server.port}/",
                    f"http://localhost:{server.port}/",
                    f"http://[::1]:{server.port}/",
                ):
                    with pytest.raises(UnsafeUrlError):
                        await client.get(url)
        assert server.requests == []

    async def test_dials_the_checked_address_and_keeps_the_host_header(self):
        backend = RecordingBackend()
        with local_server() as server:
            async with guarded_client(backend) as client:
                response = await client.get(f"http://public.test:{server.port}/echo")
        assert response.status_code == 200
        assert response.json()["host"] == f"public.test:{server.port}"
        assert backend.dialled == [PUBLIC_IP]

    async def test_dns_rebinding_never_reaches_the_second_answer(self):
        """The pre-check sees a public address, the connection a private one."""
        answers = iter([[PUBLIC_IP], ["127.0.0.1"], ["127.0.0.1"]])
        lookups = []

        async def rebinding(host):
            lookups.append(host)
            return next(answers)

        backend = RecordingBackend()
        with local_server() as server:
            url = f"http://public.test:{server.port}/"
            assert await ensure_public_url(url, resolver=rebinding) == url
            async with guarded_client(backend, resolver=rebinding) as client:
                with pytest.raises(UnsafeUrlError, match="127.0.0.1"):
                    await client.get(url)
        assert backend.dialled == []
        assert server.requests == []
        assert lookups == ["public.test", "public.test"]

    async def test_one_lookup_per_connection(self):
        """A connection that passed is made to the address that passed."""
        answers = iter([[PUBLIC_IP], ["127.0.0.1"]])

        async def rebinding(host):
            return next(answers)

        backend = RecordingBackend()
        with local_server() as server:
            async with guarded_client(backend, resolver=rebinding) as client:
                first = await client.get(f"http://public.test:{server.port}/")
                with pytest.raises(UnsafeUrlError):
                    await client.get(f"http://public.test:{server.port}/again")
        assert first.status_code == 200
        assert backend.dialled == [PUBLIC_IP]
        assert "127.0.0.1" not in backend.dialled

    @pytest.mark.parametrize(
        "target",
        [
            "http://internal.test:{port}/",
            "http://127.0.0.1:{port}/",
            "http://localhost:{port}/",
        ],
    )
    async def test_every_redirect_hop_is_checked(self, target):
        backend = RecordingBackend()
        with local_server() as server:
            location = target.format(port=server.port)
            async with guarded_client(backend) as client:
                with pytest.raises(UnsafeUrlError):
                    await client.get(
                        f"http://public.test:{server.port}/redirect",
                        params={"to": location},
                    )
        assert [r["path"].split("?")[0] for r in server.requests] == ["/redirect"]
        assert backend.dialled == [PUBLIC_IP]

    async def test_a_redirect_to_another_public_host_is_followed(self):
        backend = RecordingBackend()
        with local_server() as server:
            location = f"http://other-public.test:{server.port}/final"
            async with guarded_client(backend) as client:
                response = await client.get(
                    f"http://public.test:{server.port}/redirect",
                    params={"to": location},
                )
        assert response.status_code == 200
        assert response.json()["host"] == f"other-public.test:{server.port}"
        assert backend.dialled == [PUBLIC_IP, OTHER_PUBLIC_IP]

    async def test_tls_verifies_the_host_name_and_sends_it_as_sni(self, tmp_path):
        cert, key, pem = make_certificate("public.test", tmp_path)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cert, key)
        backend = RecordingBackend()
        with local_server(server_context) as server:
            async with guarded_client(
                backend, verify=ssl.create_default_context(cadata=pem)
            ) as client:
                response = await client.get(f"https://public.test:{server.port}/")
        assert response.status_code == 200
        assert response.json()["host"] == f"public.test:{server.port}"
        assert server.sni == ["public.test"]
        assert backend.dialled == [PUBLIC_IP]

    async def test_tls_refuses_a_certificate_for_another_name(self, tmp_path):
        cert, key, pem = make_certificate("other.test", tmp_path)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cert, key)
        with local_server(server_context) as server:
            async with guarded_client(
                RecordingBackend(), verify=ssl.create_default_context(cadata=pem)
            ) as client:
                with pytest.raises(
                    httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"
                ):
                    await client.get(f"https://public.test:{server.port}/")

    async def test_a_name_that_does_not_resolve_is_a_connect_error(self):
        backend = RecordingBackend()
        async with guarded_client(backend) as client:
            with pytest.raises(httpx.ConnectError, match="does not resolve"):
                await client.get("http://nowhere.test/")
        assert backend.dialled == []

    async def test_falls_back_to_the_next_checked_address(self):
        backend = RecordingBackend()
        with local_server() as server:
            async with guarded_client(backend) as client:
                response = await client.get(f"http://flaky.test:{server.port}/")
        assert response.status_code == 200
        assert backend.dialled == [DEAD_PUBLIC_IP, PUBLIC_IP]

    async def test_retries_sleep_through_the_backend(self):
        backend = RecordingBackend()
        async with guarded_client(backend, retries=1) as client:
            with pytest.raises(httpx.ConnectError, match="refused"):
                await client.get("http://dead.test/")
        assert backend.dialled == [DEAD_PUBLIC_IP, DEAD_PUBLIC_IP]
        assert len(backend.slept) == 1

    async def test_default_backend_connects_for_real(self, monkeypatch):
        """With no backend given, the pinned address is dialled by anyio."""

        async def loopback_is_fine(host):
            return ["127.0.0.1"]

        monkeypatch.setattr(net, "is_public_address", lambda address: True)
        with local_server() as server:
            transport = PublicOnlyTransport(resolver=loopback_is_fine)
            async with httpx.AsyncClient(transport=transport) as client:
                response = await client.get(f"http://public.test:{server.port}/")
        assert response.json()["host"] == f"public.test:{server.port}"
