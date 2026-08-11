from fastapi.testclient import TestClient
from kavalai.tools.rss import app, Feed
import kavalai.tools.rss as rss_module
from unittest.mock import MagicMock, patch
import pytest

client = TestClient(app)
auth = ("admin", "password")


@pytest.fixture(autouse=True)
def reset_auth():
    # Store original values
    orig_user = rss_module.AUTH_USER
    orig_password = rss_module.AUTH_PASSWORD
    yield
    # Restore original values after each test
    rss_module.AUTH_USER = orig_user
    rss_module.AUTH_PASSWORD = orig_password


def test_get_rss_feed_success():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [
        {"title": "Entry 1", "link": "http://example.com/1", "summary": "Summary 1"},
        {"title": "Entry 2", "link": "http://example.com/2", "summary": "Summary 2"},
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title == "Test Feed"
    assert len(result.items) == 2
    assert result.items[0].title == "Entry 1"
    assert result.items[0].summary == "Summary 1"


def test_get_rss_feed_unauthorized():
    response = client.get(
        "/get_rss_feed",
        params={"url": "http://example.com/rss"},
        auth=("wrong", "pass"),
    )
    assert response.status_code == 401


def test_get_rss_feed_empty():
    mock_feed = MagicMock()
    mock_feed.entries = []

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title is None
    assert len(result.items) == 0


def test_get_rss_feed_max_results():
    mock_feed = MagicMock()
    mock_feed.feed = {"title": "Test Feed"}
    mock_feed.entries = [{"title": f"Entry {i}"} for i in range(10)]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss", "max_results": 3},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert len(result.items) == 3


def test_get_rss_feed_missing_fields():
    mock_feed = MagicMock()
    mock_feed.feed = {}
    mock_feed.entries = [{"other": "field"}]

    with patch("feedparser.parse", return_value=mock_feed):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )

    assert response.status_code == 200
    result = Feed(**response.json())
    assert result.title is None
    assert result.items[0].title == "No Title"
    assert result.items[0].link == "No Link"
    assert result.items[0].summary == "No summary available."


def test_get_rss_feed_error():
    with patch("feedparser.parse", side_effect=Exception("Network error")):
        response = client.get(
            "/get_rss_feed",
            params={"url": "http://example.com/rss"},
            auth=auth,
        )
        assert response.status_code == 500
        assert "Error parsing feed: Network error" in response.json()["detail"]


def test_parse_args_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("RSS_AUTH_USER", raising=False)
    monkeypatch.delenv("RSS_AUTH_PASSWORD", raising=False)

    defaults = rss_module.parse_args([])
    assert (defaults.port, defaults.user, defaults.password) == (
        10000,
        "admin",
        "password",
    )

    overridden = rss_module.parse_args(
        ["--port", "1234", "--user", "u", "--password", "p"]
    )
    assert (overridden.port, overridden.user, overridden.password) == (1234, "u", "p")


def test_main_applies_credentials_and_serves(monkeypatch):
    served = {}

    def fake_run(application, host, port):
        served.update(app=application, host=host, port=port)

    monkeypatch.setattr(rss_module.uvicorn, "run", fake_run)

    rss_module.main(["--port", "1234", "--user", "alice", "--password", "secret"])

    assert served == {"app": rss_module.app, "host": "0.0.0.0", "port": 1234}
    assert rss_module.AUTH_USER == "alice"
    assert rss_module.AUTH_PASSWORD == "secret"
