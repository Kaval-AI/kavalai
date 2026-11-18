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

# Browser/IndexedDB persistence for the SQLite database backend.
#
# When Kaval.AI runs inside Pyodide there is no real filesystem: Emscripten's
# default MEMFS lives only in memory and is lost on page reload. To persist the
# SQLite database we mount Emscripten's IDBFS — a filesystem backed by the
# browser's IndexedDB — at :data:`MOUNT_DIR` and explicitly sync it:
#
#   * ``syncfs(populate=True)``  copies IndexedDB -> in-memory FS (load on start)
#   * ``syncfs(populate=False)`` copies in-memory FS -> IndexedDB (persist)
#
# Everything here is a no-op outside Pyodide, so the same code paths work on a
# normal Python interpreter (where the SQLite engine just uses an in-memory or
# on-disk database directly).

import os
import sys

# Directory mounted on IDBFS; the SQLite file lives inside it.
MOUNT_DIR = "/kavalai"

_mounted = False


def is_pyodide() -> bool:
    """Return ``True`` when running inside a Pyodide (Emscripten) runtime."""
    return sys.platform == "emscripten"


async def mount(mount_dir: str = MOUNT_DIR) -> None:
    """Mount IDBFS and load any previously persisted data from IndexedDB.

    Idempotent and a no-op when not running under Pyodide.
    """
    global _mounted
    if not is_pyodide() or _mounted:
        return

    import pyodide_js  # type: ignore[import-not-found]

    fs = pyodide_js.FS
    if not os.path.isdir(mount_dir):
        fs.mkdir(mount_dir)
    fs.mount(fs.filesystems.IDBFS, {}, mount_dir)
    # populate=True: pull the existing database out of IndexedDB into the FS.
    await _syncfs(True)
    _mounted = True


async def flush() -> None:
    """Persist the in-memory filesystem to IndexedDB.

    Call this after committing changes so they survive a page reload. No-op
    when not running under Pyodide (or before :func:`mount`).
    """
    if not is_pyodide() or not _mounted:
        return
    # populate=False: push the FS contents into IndexedDB.
    await _syncfs(False)


async def _syncfs(populate: bool) -> None:
    """Await Emscripten's callback-based ``FS.syncfs`` as a JS Promise."""
    import pyodide_js  # type: ignore[import-not-found]
    import js  # type: ignore[import-not-found]
    from pyodide.ffi import create_once_callable  # type: ignore[import-not-found]

    def executor(resolve, reject) -> None:
        def callback(err=None) -> None:
            if err:
                reject(err)
            else:
                resolve(None)

        pyodide_js.FS.syncfs(populate, create_once_callable(callback))

    await js.Promise.new(create_once_callable(executor))
