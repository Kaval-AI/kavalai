"""A local HTTP server on 127.0.0.1, and the stand-ins the network tests share.

The guard in :mod:`kavalai.net` is tested against real sockets rather than
mocks, so what is proven is what httpx and httpcore actually do. Everything
here stays on the loopback interface: a "public" host is a name the fake
resolver maps to a public address, and the test's network backend dials the
local server whenever it is asked for that address.
"""

import datetime
import json
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpcore

PUBLIC_IP = "93.184.216.34"
OTHER_PUBLIC_IP = "93.184.216.35"
DEAD_PUBLIC_IP = "93.184.216.36"
STAND_INS = {PUBLIC_IP, OTHER_PUBLIC_IP}

ADDRESSES = {
    "public.test": [PUBLIC_IP],
    "other-public.test": [OTHER_PUBLIC_IP],
    "example.com": [PUBLIC_IP],
    "www.example.com": [PUBLIC_IP],
    "internal.test": ["10.1.2.3"],
    "half.test": [PUBLIC_IP, "127.0.0.1"],
    "mapped.test": ["::ffff:192.168.1.1"],
    "flaky.test": [DEAD_PUBLIC_IP, PUBLIC_IP],
    "dead.test": [DEAD_PUBLIC_IP],
}


async def fake_resolver(host: str) -> list[str]:
    """The addresses in :data:`ADDRESSES`; any other name does not resolve."""
    return list(ADDRESSES.get(host, []))


class _Handler(BaseHTTPRequestHandler):
    """Records each request; ``/redirect?to=<url>`` answers 302, ``/text`` plain text."""

    def do_GET(self):
        self._answer(b"")

    def do_POST(self):
        self._answer(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def _answer(self, body: bytes):
        seen = {
            "method": self.command,
            "path": self.path,
            "host": self.headers.get("Host"),
            "authorization": self.headers.get("Authorization"),
            "body": body.decode(),
        }
        self.server.requests.append(seen)
        target = urlsplit(self.path)
        if target.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", parse_qs(target.query)["to"][0])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if target.path == "/text":
            payload, kind = b"invalid json", "text/plain"
        else:
            payload, kind = json.dumps(seen).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@contextmanager
def local_server(ssl_context: ssl.SSLContext | None = None):
    """A threaded HTTP(S) server on ``127.0.0.1``; ``.port``, ``.requests``, ``.sni``."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    server.sni = []
    server.port = server.server_address[1]
    if ssl_context is not None:
        ssl_context.sni_callback = lambda _sock, name, _ctx: server.sni.append(name)
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def make_certificate(hostname: str, directory: Path) -> tuple[Path, Path, str]:
    """A self-signed certificate naming only ``hostname``: (cert, key, PEM)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM)
    cert_path = directory / f"{hostname}.crt"
    key_path = directory / f"{hostname}.key"
    cert_path.write_bytes(pem)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path, pem.decode()


class RecordingBackend(httpcore.AsyncNetworkBackend):
    """Records every address dialled; dials the local server for the stand-ins.

    :data:`DEAD_PUBLIC_IP` refuses the connection, as an unreachable host does.
    """

    def __init__(self):
        self.dialled: list[str] = []
        self.slept: list[float] = []
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        self.dialled.append(host)
        if host == DEAD_PUBLIC_IP:
            raise httpcore.ConnectError(f"{host} refused the connection")
        target = "127.0.0.1" if host in STAND_INS else host
        return await self._inner.connect_tcp(
            target,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def sleep(self, seconds):
        self.slept.append(seconds)
