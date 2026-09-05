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

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    discovered_at TEXT NOT NULL,
    last_crawled_at TEXT,
    status_code INTEGER,
    fetch_mode TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    fetch_error TEXT,
    title TEXT,
    html_path TEXT,
    markdown TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_pending
    ON pages (last_crawled_at, attempts);
"""

FILES_SUFFIX = ".files"


def utcnow_iso() -> str:
    """The current UTC time as an ISO-8601 string, the table's timestamp format."""
    return datetime.now(timezone.utc).isoformat()


def url_slug(url: str) -> str:
    """A stable filename stem for a URL, so a re-crawl overwrites in place."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@dataclass
class PageRow:
    """One row of the ``pages`` table.

    ``html_path`` is a file name relative to the database's files directory —
    the database plus that directory move together (and can later live in a
    bucket under the same prefix).
    """

    url: str
    discovered_at: str
    last_crawled_at: Optional[str]
    status_code: Optional[int]
    fetch_mode: Optional[str]
    attempts: int
    fetch_error: Optional[str]
    title: Optional[str]
    html_path: Optional[str]
    markdown: Optional[str]


class PagesDatabase:
    """The crawl state: one SQLite table that is both frontier and content store.

    A row with ``last_crawled_at IS NULL`` is pending work; every write that
    completes a page — its content plus the links it revealed — happens in one
    transaction, so a crawler killed at any moment leaves a database a restart
    can resume from. At worst, one page whose transaction had not committed is
    fetched again.

    Bulk artefacts (raw HTML, screenshots) live as files in a sibling
    directory (``<db>.files`` next to ``<db>.db``), the table holding only
    their relative paths. A file is written *before* the row that references
    it commits, so a crash can orphan a file — which the re-crawl overwrites,
    file names being derived from the URL — but never a row.
    """

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._reject_legacy_schema()
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        stem = path[: -len(".db")] if path.endswith(".db") else path
        self.files_dir = None if path == ":memory:" else stem + FILES_SUFFIX

    def _reject_legacy_schema(self) -> None:
        """Refuse a database from before HTML moved out of the table."""
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(pages)")
        }
        if "html" in columns:
            raise ValueError(
                "This pages database stores raw HTML in the table (an old "
                "layout); delete it and re-scrape."
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PagesDatabase":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _write_file(self, name: str, data: bytes) -> str:
        """Write one artefact into the files directory, returning its name."""
        os.makedirs(self.files_dir, exist_ok=True)
        with open(os.path.join(self.files_dir, name), "wb") as handle:
            handle.write(data)
        return name

    def file_path(self, name: Optional[str]) -> Optional[str]:
        """The absolute path of a stored artefact, None when there is none."""
        if not name or not self.files_dir:
            return None
        return os.path.join(self.files_dir, name)

    def load_html(self, row: PageRow) -> Optional[str]:
        """The raw HTML stored for a page, None when there is none."""
        path = self.file_path(row.html_path)
        if path is None or not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def add_urls(self, urls: Iterable[str], discovered_at: Optional[str] = None) -> int:
        """Add URLs to the frontier, ignoring the ones already known.

        Returns:
            How many URLs were actually new.
        """
        discovered_at = discovered_at or utcnow_iso()
        before = self._conn.total_changes
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO pages (url, discovered_at) VALUES (?, ?)",
                [(url, discovered_at) for url in urls],
            )
        return self._conn.total_changes - before

    def next_pending(
        self, skip: Iterable[str] = (), max_attempts: int = 3
    ) -> Optional[str]:
        """The next URL to fetch, oldest discovery first.

        Args:
            skip: URLs to pass over — the ones already tried in this run, so a
                failure is retried on the *next* run rather than immediately.
            max_attempts: Rows at or above this many attempts are parked.
        """
        skip = set(skip)
        cursor = self._conn.execute(
            "SELECT url FROM pages"
            " WHERE last_crawled_at IS NULL AND attempts < ?"
            " ORDER BY discovered_at, rowid",
            (max_attempts,),
        )
        for (url,) in cursor:
            if url not in skip:
                return url
        return None

    def record_success(
        self,
        url: str,
        *,
        status_code: Optional[int],
        fetch_mode: str,
        title: Optional[str],
        html: Optional[str],
        markdown: Optional[str],
        links: Iterable[str] = (),
        crawled_at: Optional[str] = None,
    ) -> None:
        """Store a fetched page and enqueue its links, atomically.

        The HTML goes to the files directory first; only its name enters the
        table, in the same transaction as everything else.
        """
        crawled_at = crawled_at or utcnow_iso()
        html_path = None
        if html and self.files_dir:
            html_path = self._write_file(f"{url_slug(url)}.html", html.encode())
        with self._conn:
            self._conn.execute(
                "UPDATE pages SET last_crawled_at = ?, status_code = ?,"
                " fetch_mode = ?, attempts = attempts + 1, fetch_error = NULL,"
                " title = ?, html_path = ?, markdown = ? WHERE url = ?",
                (crawled_at, status_code, fetch_mode, title, html_path, markdown, url),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO pages (url, discovered_at) VALUES (?, ?)",
                [(link, crawled_at) for link in links],
            )

    def record_failure(
        self, url: str, *, status_code: Optional[int], fetch_error: str
    ) -> None:
        """Count a failed attempt; the row stays pending until the attempt cap."""
        with self._conn:
            self._conn.execute(
                "UPDATE pages SET status_code = ?, attempts = attempts + 1,"
                " fetch_error = ? WHERE url = ?",
                (status_code, fetch_error, url),
            )

    def record_skipped(self, url: str, reason: str) -> None:
        """Mark a URL as deliberately not fetched (e.g. robots.txt disallows it).

        The row is completed rather than failed, so no later run retries it.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE pages SET last_crawled_at = ?, fetch_error = ? WHERE url = ?",
                (utcnow_iso(), reason, url),
            )

    def save_screenshot(self, url: str, image: bytes) -> Optional[str]:
        """Store a page's screenshot beside its HTML in the files directory.

        The file name derives from the URL (``<url_slug>.png``), so it needs
        no column of its own and a re-capture overwrites in place.

        Returns:
            The absolute path of the written file, None without a files
            directory.
        """
        if not self.files_dir:
            return None
        return self.file_path(self._write_file(f"{url_slug(url)}.png", image))

    def requeue_all(self) -> int:
        """Put every URL back in the frontier, keeping the stored content.

        Returns:
            How many rows were re-queued.
        """
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pages SET last_crawled_at = NULL, attempts = 0,"
                " fetch_error = NULL"
            )
        return cursor.rowcount

    def fetched_count(self) -> int:
        """How many pages this crawl has fetched — what ``--max-pages`` caps.

        Counted by ``last_crawled_at``, not by stored content: a re-queued row
        keeps its old markdown until it is fetched again, and must still count
        as work to do.
        """
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM pages"
            " WHERE last_crawled_at IS NOT NULL AND markdown IS NOT NULL"
        ).fetchone()
        return count

    def iter_pages(self) -> Iterator[PageRow]:
        """Every row, fetched or not."""
        cursor = self._conn.execute(
            "SELECT url, discovered_at, last_crawled_at, status_code, fetch_mode,"
            " attempts, fetch_error, title, html_path, markdown"
            " FROM pages ORDER BY discovered_at, rowid"
        )
        for row in cursor:
            yield PageRow(**dict(row))

    def stats(self) -> dict:
        """Row counts by state, for progress logging and the exit report."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(last_crawled_at IS NOT NULL AND markdown IS NOT NULL) AS fetched,"
            " SUM(last_crawled_at IS NULL) AS pending,"
            " SUM(fetch_error IS NOT NULL) AS errors"
            " FROM pages"
        ).fetchone()
        return {key: row[key] or 0 for key in row.keys()}
