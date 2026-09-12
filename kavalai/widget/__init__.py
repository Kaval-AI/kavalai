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

The production chat widget, shipped as package data.

The widget is two static files, ``kaval-chatbot.js`` and
``kaval-chatbot.css``, which a page loads with a ``<script>`` and a
``<link>`` tag. They ship inside the wheel so that a Python host serves the
version whose connector speaks the protocol of the installed agent server;
the functions here locate them. ``WIDGET_FILES`` names the files a host may
serve — the preview page, the README and the tests stay in the repository.
"""

from importlib import resources
from pathlib import Path

WIDGET_FILES = ("kaval-chatbot.js", "kaval-chatbot.css")


def widget_dir() -> Path:
    """The directory holding the widget files of the installed version.

    Raises ``RuntimeError`` when the package is not installed as plain files
    (a zip import), because a static file server needs a real directory.
    """
    location = resources.files(__name__)
    if not isinstance(location, Path):
        raise RuntimeError(
            "kavalai is not installed as plain files, so the widget has no"
            f" directory to serve from ({location!r})"
        )
    return location


def asset_path(name: str) -> Path:
    """The path of one widget file, ``kaval-chatbot.js`` or
    ``kaval-chatbot.css``; any other name raises ``ValueError``."""
    if name not in WIDGET_FILES:
        raise ValueError(
            f"{name!r} is not a widget file; expected one of {', '.join(WIDGET_FILES)}"
        )
    return widget_dir() / name
