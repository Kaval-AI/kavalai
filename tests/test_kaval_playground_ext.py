"""Tests for ``docs/_ext/kaval_playground.py`` — the docs' in-browser playground.

The extension stages a ``kavalai-*.whl`` for Pyodide to install client-side. A
wheel left over from an earlier version installs *that* version's modules in
the browser, and the docs' own examples then fail with ``ModuleNotFoundError``
on the page — far from the cause. The extension therefore warns at build time
when the staged wheel's version does not match ``pyproject.toml``; these tests
pin that behaviour so a stale wheel cannot ship silently again.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "_ext"))

import kaval_playground as ext


def _fake_repo(tmp_path: Path, version: str) -> Path:
    """A minimal repo layout: pyproject.toml, docs/, webwidget/ (empty)."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "kavalai"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "webwidget").mkdir()
    return tmp_path


def _fake_app(repo: Path) -> SimpleNamespace:
    return SimpleNamespace(
        srcdir=str(repo / "docs"),
        config=SimpleNamespace(kaval_pyodide_url="https://example.test/pyodide.js"),
    )


def test_wheel_version_is_read_from_the_filename():
    assert ext._wheel_version("kavalai-1.0.1-py3-none-any.whl") == "1.0.1"
    assert ext._wheel_version("kavalai-2.3.4rc1-py3-none-any.whl") == "2.3.4rc1"


def test_project_version_reads_pyproject(tmp_path):
    repo = _fake_repo(tmp_path, "9.9.9")
    assert ext._project_version(repo) == "9.9.9"


def test_project_version_is_none_when_unreadable(tmp_path):
    assert ext._project_version(tmp_path) is None
    (tmp_path / "pyproject.toml").write_text("not = [toml", encoding="utf-8")
    assert ext._project_version(tmp_path) is None


def test_stale_staged_wheel_warns(tmp_path, caplog):
    repo = _fake_repo(tmp_path, "1.0.2")
    static = repo / "docs" / "_static" / "pyodide"
    static.mkdir(parents=True)
    (static / "kavalai-1.0.1-py3-none-any.whl").write_bytes(b"")

    with caplog.at_level(logging.WARNING, logger=ext.logger.name):
        ext._stage_assets(_fake_app(repo))

    messages = [r.getMessage() for r in caplog.records]
    assert any("kavalai-1.0.1-py3-none-any.whl" in m and "1.0.2" in m for m in messages)
    assert any("uv build --wheel" in m for m in messages)


def test_current_wheel_does_not_warn(tmp_path, caplog):
    repo = _fake_repo(tmp_path, "1.0.2")
    dist = repo / "dist"
    dist.mkdir()
    (dist / "kavalai-1.0.2-py3-none-any.whl").write_bytes(b"")

    with caplog.at_level(logging.WARNING, logger=ext.logger.name):
        ext._stage_assets(_fake_app(repo))

    assert not [
        r for r in caplog.records if "not the current version" in r.getMessage()
    ]
    config = (
        repo / "docs" / "_static" / "pyodide" / "playground-config.js"
    ).read_text()
    assert '"wheelName": "kavalai-1.0.2-py3-none-any.whl"' in config


def test_fresh_dist_wheel_replaces_stale_staged_one(tmp_path):
    repo = _fake_repo(tmp_path, "1.0.2")
    static = repo / "docs" / "_static" / "pyodide"
    static.mkdir(parents=True)
    (static / "kavalai-1.0.1-py3-none-any.whl").write_bytes(b"")
    dist = repo / "dist"
    dist.mkdir()
    (dist / "kavalai-1.0.2-py3-none-any.whl").write_bytes(b"")

    ext._stage_assets(_fake_app(repo))

    assert sorted(p.name for p in static.glob("*.whl")) == [
        "kavalai-1.0.2-py3-none-any.whl"
    ]


@pytest.mark.parametrize("has_pyproject", [True, False])
def test_no_wheel_anywhere_runs_plain_python(tmp_path, caplog, has_pyproject):
    repo = _fake_repo(tmp_path, "1.0.2")
    if not has_pyproject:
        (repo / "pyproject.toml").unlink()

    with caplog.at_level(logging.WARNING, logger=ext.logger.name):
        ext._stage_assets(_fake_app(repo))

    config = (
        repo / "docs" / "_static" / "pyodide" / "playground-config.js"
    ).read_text()
    assert '"wheelName": null' in config
    assert any("plain Python only" in r.getMessage() for r in caplog.records)
