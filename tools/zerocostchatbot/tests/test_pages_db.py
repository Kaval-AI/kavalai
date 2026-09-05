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

from tools.zerocostchatbot.pages_db import PagesDatabase, utcnow_iso


def make_db(tmp_path) -> PagesDatabase:
    return PagesDatabase(str(tmp_path / "pages.db"))


def test_add_urls_deduplicates(tmp_path):
    with make_db(tmp_path) as db:
        assert db.add_urls(["https://a/", "https://b/"]) == 2
        assert db.add_urls(["https://a/", "https://c/"]) == 1
        assert db.stats()["total"] == 3


def test_next_pending_order_skip_and_cap(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/"], discovered_at="2026-01-01T00:00:00")
        db.add_urls(["https://b/"], discovered_at="2026-01-02T00:00:00")
        assert db.next_pending() == "https://a/"
        assert db.next_pending(skip={"https://a/"}) == "https://b/"

        for _ in range(3):
            db.record_failure("https://a/", status_code=None, fetch_error="boom")
        # Three failed attempts park the URL.
        assert db.next_pending(max_attempts=3) == "https://b/"
        assert db.next_pending(skip={"https://b/"}, max_attempts=3) is None


def test_record_success_stores_content_and_links_atomically(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/"])
        db.record_success(
            "https://a/",
            status_code=200,
            fetch_mode="http",
            title="A",
            html="<p>hi</p>",
            markdown="hi",
            links=["https://a/next", "https://a/"],
        )
        rows = {row.url: row for row in db.iter_pages()}
        assert rows["https://a/"].markdown == "hi"
        assert rows["https://a/"].fetch_mode == "http"
        assert rows["https://a/"].attempts == 1
        assert rows["https://a/"].last_crawled_at is not None
        # The discovered link entered the frontier; the crawled page did not
        # re-enter it.
        assert rows["https://a/next"].last_crawled_at is None
        assert db.fetched_count() == 1
        assert db.next_pending() == "https://a/next"


def test_success_clears_a_previous_error(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/"])
        db.record_failure("https://a/", status_code=503, fetch_error="challenge")
        db.record_success(
            "https://a/",
            status_code=200,
            fetch_mode="browser",
            title=None,
            html="x",
            markdown="x",
        )
        (row,) = db.iter_pages()
        assert row.fetch_error is None
        assert row.attempts == 2


def test_failure_keeps_url_pending_for_the_next_run(tmp_path):
    path = tmp_path / "pages.db"
    with PagesDatabase(str(path)) as db:
        db.add_urls(["https://a/"])
        db.record_failure("https://a/", status_code=500, fetch_error="HTTP 500")
    # A new process sees the URL again — resume after a kill or a crash.
    with PagesDatabase(str(path)) as db:
        assert db.next_pending() == "https://a/"
        assert db.stats() == {"total": 1, "fetched": 0, "pending": 1, "errors": 1}


def test_record_skipped_completes_the_row(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/private"])
        db.record_skipped("https://a/private", "disallowed by robots.txt")
        assert db.next_pending() is None
        (row,) = db.iter_pages()
        assert row.fetch_error == "disallowed by robots.txt"
        assert db.fetched_count() == 0


def test_requeue_all_resets_state_but_keeps_content(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/", "https://b/"])
        db.record_success(
            "https://a/",
            status_code=200,
            fetch_mode="http",
            title="A",
            html="x",
            markdown="x",
        )
        db.record_failure("https://b/", status_code=None, fetch_error="boom")
        assert db.requeue_all() == 2
        assert db.stats()["pending"] == 2
        rows = {row.url: row for row in db.iter_pages()}
        assert rows["https://a/"].markdown == "x"
        assert rows["https://a/"].attempts == 0
        assert rows["https://b/"].fetch_error is None


def test_utcnow_iso_is_parseable():
    from datetime import datetime

    assert datetime.fromisoformat(utcnow_iso()).tzinfo is not None


def test_requeued_pages_count_toward_the_budget_again(tmp_path):
    with make_db(tmp_path) as db:
        db.add_urls(["https://a/"])
        db.record_success(
            "https://a/",
            status_code=200,
            fetch_mode="http",
            title="A",
            html="x",
            markdown="x",
        )
        assert db.fetched_count() == 1
        db.requeue_all()
        # The kept content must not eat the --max-pages budget of the re-crawl.
        assert db.fetched_count() == 0
        assert db.stats()["fetched"] == 0
