"""``.env.example`` and the code that reads the environment must agree.

The committed template listed five variables nothing read (``ANTHROPIC_KEY``
when the client wants ``ANTHROPIC_API_KEY``, ``BACKOFFICE_PORT`` when the
server hard-coded 8000, and three that were simply dead) — the kind of drift
that costs someone an afternoon before they discover the name is ignored. This
checks both directions so neither side can rot silently.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
PACKAGE = ROOT / "kavalai"

ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)

# Names the code reads but that do not belong in a template: set by the
# platform, or only meaningful inside a container.
NOT_IN_TEMPLATE = {
    "PATH",
    "HOME",
    "PYODIDE",
}


def documented_names() -> set[str]:
    return set(ASSIGNMENT.findall(ENV_EXAMPLE.read_text()))


def referenced_names() -> set[str]:
    """Every environment variable named anywhere under ``kavalai/``."""
    pattern = re.compile(
        r"""(?:getenv|environ\.get|env\.str|env\.int|env\.bool|env)\(\s*["']([A-Z][A-Z0-9_]*)["']"""
        r"""|environ\[["']([A-Z][A-Z0-9_]*)["']\]"""
    )
    names = set()
    for path in PACKAGE.rglob("*.py"):
        for first, second in pattern.findall(path.read_text()):
            names.add(first or second)
    return names - NOT_IN_TEMPLATE


def test_env_example_exists():
    assert ENV_EXAMPLE.exists(), ".env.example is the onboarding template"


@pytest.mark.parametrize("name", sorted(documented_names()))
def test_documented_variables_are_read_somewhere(name):
    assert name in referenced_names(), (
        f"{name} is in .env.example but nothing under kavalai/ reads it — "
        "either wire it up or drop it from the template"
    )


@pytest.mark.parametrize("name", sorted(referenced_names()))
def test_read_variables_are_documented(name):
    assert name in documented_names(), (
        f"kavalai/ reads {name} but .env.example does not mention it — "
        "an operator has no way to discover it"
    )
