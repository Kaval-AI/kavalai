"""Packaging: the extras, the install hints that name them, and the wheel.

The base install is small on purpose, so most of Kaval.AI sits behind an extra
and a missing package surfaces as an install hint naming one. Nothing but these
tests keeps the extras, the hints and the wheel from drifting apart: the 1.0.3
wheel shipped without ``default_prompt_template.j2``, and the backoffice
imported ``sse_starlette`` without declaring it.

The wheel is built with ``uv build`` — sdist first, then the wheel from it, as
the release workflow does — which takes a few seconds with a warm cache. The
import smoke test for ``kavalai[runtime]`` installs into a fresh environment
from the package index, so it is marked ``integration`` and runs with
``pytest -m integration``.
"""

import ast
import fnmatch
import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "kavalai"
SMOKE_SCRIPT = Path(__file__).parent / "helpers" / "runtime_smoke.py"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
EXTRAS = PYPROJECT["project"]["optional-dependencies"]
PARTS_OF_COMMON = ("runtime", "webtools", "fastembed", "backoffice")

SOURCE_ONLY = (
    "kavalai/widget/README.md",
    "kavalai/widget/preview.html",
    "kavalai/widget/tests/*",
)

SELF_REFERENCE = re.compile(r"^kavalai\[([^\]]+)\]$")
HINT = re.compile(r"(?:\bkavalai|\.)\[([a-z0-9_, -]+)\]")


def requirements(extra: str) -> set[str]:
    """What an extra installs, with its references to other extras expanded."""
    resolved = set()
    for requirement in EXTRAS[extra]:
        match = SELF_REFERENCE.match(requirement)
        if match:
            for name in match.group(1).split(","):
                resolved |= requirements(name.strip())
        else:
            resolved.add(requirement)
    return resolved


def distribution_name(requirement: str) -> str:
    """``fastapi[standard]>=0.1`` → ``fastapi``, normalised as in PEP 503."""
    name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_distributions() -> set[str]:
    declared = PYPROJECT["project"]["dependencies"]
    for extra in EXTRAS:
        declared = [*declared, *requirements(extra)]
    return {distribution_name(r) for r in declared}


def imported_top_level_modules() -> set[str]:
    """Every third-party top-level module imported anywhere under ``kavalai/``."""
    modules = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules - set(sys.stdlib_module_names) - {"kavalai"}


def hint_sources() -> list[Path]:
    """Files that may tell a reader which extra to install.

    ``docs/versions.rst`` is left out: it records extras that no longer exist.
    """
    docs = [
        p
        for p in (ROOT / "docs").rglob("*")
        if p.suffix in {".rst", ".md", ".txt"}
        and "_build" not in p.parts
        and p.name != "versions.rst"
    ]
    return [
        *PACKAGE.rglob("*.py"),
        *(PACKAGE / ".agents").rglob("*.md"),
        *(ROOT / "dockerfiles").iterdir(),
        *docs,
        *(ROOT / name for name in ("README.md", "CLAUDE.md", "AGENTS.md")),
    ]


class TestExtras:
    def test_common_is_the_union_of_the_four_extras(self):
        union = set().union(*(requirements(e) for e in PARTS_OF_COMMON))
        assert requirements("common") == union

    def test_the_four_extras_do_not_overlap(self):
        owner = {}
        for extra in PARTS_OF_COMMON:
            for requirement in requirements(extra):
                assert (
                    requirement not in owner
                ), f"{requirement} is in both {owner[requirement]} and {extra}"
                owner[requirement] = extra

    def test_no_synchronous_postgres_driver_is_required(self):
        """Migrations run over asyncpg, so psycopg2 is in no install at all."""
        assert not any(d.startswith("psycopg") for d in declared_distributions())

    def test_every_imported_package_is_declared(self):
        """A package the code imports is declared, not borrowed from another.

        ``sse_starlette`` used to arrive only because ``mcp`` depends on it.
        Two packages are exempt because a declared requirement promises them:
        ``starlette`` (FastAPI's foundation) and ``uvicorn`` (``fastapi[standard]``).
        ``js``, ``pyodide`` and ``pyodide_js`` exist only inside Pyodide.
        """
        exempt = {"starlette", "uvicorn", "js", "pyodide", "pyodide_js"}
        providers = importlib.metadata.packages_distributions()
        declared = declared_distributions()
        undeclared = {
            module: providers.get(module, [])
            for module in imported_top_level_modules() - exempt
            if not {distribution_name(d) for d in providers.get(module, [])} & declared
        }
        assert undeclared == {}


class TestInstallHints:
    def test_every_hint_names_an_existing_extra(self):
        for path in hint_sources():
            for match in HINT.finditer(path.read_text()):
                for extra in match.group(1).split(","):
                    assert extra.strip() in EXTRAS, (
                        f"{path.relative_to(ROOT)} names the extra "
                        f"{extra.strip()!r}, which pyproject.toml does not define"
                    )

    @pytest.mark.parametrize(
        ("module", "missing", "extra", "importers"),
        [
            ("kavalai.server", "fastapi", "runtime", ()),
            ("kavalai.server", "uvicorn", "runtime", ()),
            ("kavalai.backoffice.server", "authlib", "backoffice", ()),
            ("kavalai.backoffice.server", "sse_starlette", "backoffice", ()),
            # Starlette imports itsdangerous in its session middleware; once an
            # earlier test has imported that, only forgetting it too makes the
            # missing package visible again.
            (
                "kavalai.backoffice.server",
                "itsdangerous",
                "backoffice",
                ("starlette.middleware.sessions",),
            ),
            ("kavalai.backoffice.embedding_projector", "sklearn", "backoffice", ()),
        ],
    )
    def test_a_missing_package_is_named_with_its_extra(
        self, monkeypatch, module, missing, extra, importers
    ):
        block(monkeypatch, missing)
        forget(monkeypatch, module)
        for importer in importers:
            forget(monkeypatch, importer)
        with pytest.raises(ImportError, match=hint_for(extra)):
            importlib.import_module(module)

    def test_a_missing_provider_sdk_is_named_with_its_extra(self, monkeypatch):
        import kavalai

        block(monkeypatch, "openai")
        forget(monkeypatch, "kavalai.llm_clients.openai_client")
        with pytest.raises(ImportError, match=hint_for("runtime")):
            kavalai.__getattr__("OpenAIClient")


def block(monkeypatch, package: str) -> None:
    """Make ``package`` and its submodules fail to import, as if uninstalled."""
    for name in list(sys.modules):
        if name.startswith(f"{package}."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, package, None)


def forget(monkeypatch, module: str) -> None:
    """Drop ``module`` from the import cache so the next import runs it again."""
    if module in sys.modules:
        monkeypatch.delitem(sys.modules, module)


def hint_for(extra: str) -> str:
    return rf'pip install "kavalai\[([a-z_]+,)*{extra}(,[a-z_]+)*\]"'


def uv() -> str:
    path = shutil.which("uv")
    if path is None:
        pytest.skip("uv is not on PATH")
    return path


def source_files() -> set[str]:
    """Every file under ``kavalai/`` that git tracks or would track."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "kavalai"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    return {
        name
        for name in listed.stdout.splitlines()
        if (ROOT / name).is_file() and "__pycache__" not in name
    }


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> set[str]:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [uv(), "build", "--quiet", "--out-dir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (path,) = out.glob("*.whl")
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


class TestWheel:
    @pytest.mark.parametrize(
        "name",
        [
            "kavalai/default_prompt_template.j2",
            "kavalai/migrations/agents/env.py",
            "kavalai/migrations/agents/script.py.mako",
            "kavalai/migrations/agents/versions/0001_initial_agents_schema.py",
            "kavalai/migrations/backoffice/env.py",
            "kavalai/.agents/skills/kavalai/SKILL.md",
            "kavalai/.agents/skills/kavalai-serving/references/config.md",
        ],
    )
    def test_the_wheel_carries(self, wheel, name):
        assert name in wheel

    def test_the_wheel_carries_every_file_of_the_package(self, wheel):
        """Data files included: the template, migrations, skills and widget.

        ``SOURCE_ONLY`` names the files that are deliberately left out — the
        widget's tests, preview page and README — so a new data file has to be
        either shipped or listed there.
        """
        shipped = {
            name
            for name in source_files()
            if not any(fnmatch.fnmatch(name, pattern) for pattern in SOURCE_ONLY)
        }
        assert shipped - wheel == set()


@pytest.mark.integration
def test_the_runtime_extra_serves_a_workflow(tmp_path):
    """``kavalai[runtime]`` alone imports the server, serves and migrates.

    A fresh environment gets only that extra, installed from a wheel built
    from this checkout; ``tests/helpers/runtime_smoke.py`` then runs there,
    outside the repository, and checks that crawl4ai, fastembed, scikit-learn,
    authlib and psycopg2 are all absent.
    """
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    subprocess.run(
        [uv(), "venv", "--quiet", "--python", sys.executable, str(venv)], check=True
    )
    subprocess.run(
        [uv(), "pip", "install", "--quiet", "--python", str(python)]
        + [f"kavalai[runtime] @ {ROOT.as_uri()}"],
        check=True,
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [str(python), str(SMOKE_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")
