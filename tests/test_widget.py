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

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from kavalai import widget
from kavalai.widget import WIDGET_FILES, asset_path, widget_dir


def test_widget_dir_holds_both_files():
    for name in WIDGET_FILES:
        assert (widget_dir() / name).is_file(), name


def test_asset_path_names_one_widget_file():
    assert asset_path("kaval-chatbot.css") == widget_dir() / "kaval-chatbot.css"
    assert "KavalChatbot" in asset_path("kaval-chatbot.js").read_text("utf-8")


@pytest.mark.parametrize("name", ["preview.html", "__init__.py", "../db.py"])
def test_asset_path_refuses_other_files(name):
    with pytest.raises(ValueError, match="not a widget file"):
        asset_path(name)


def test_widget_dir_needs_a_plain_file_install(monkeypatch):
    monkeypatch.setattr(widget.resources, "files", lambda name: object())
    with pytest.raises(RuntimeError, match="plain files"):
        widget_dir()


def test_static_files_serve_the_installed_widget():
    """The snippet in docs/reference/widget.rst."""
    app = FastAPI()
    app.mount("/widget", StaticFiles(directory=widget_dir()), name="widget")
    client = TestClient(app)

    script = client.get("/widget/kaval-chatbot.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "agentConnector" in script.text
    style = client.get("/widget/kaval-chatbot.css")
    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")
