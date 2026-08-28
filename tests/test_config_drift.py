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

# Names the template lists for the operator's sake although Kaval.AI never
# reads them: the provider SDK does. Each is marked "SDK" in the template.
READ_BY_SDK = {
    "OPENAI_BASE_URL",
    "GOOGLE_API_KEY",
}

# The modules allowed to read the environment. Everything else under
# ``kavalai/`` takes its configuration as arguments, so a value can always be
# passed explicitly and a test never depends on the shell it runs in.
ENVIRONMENT_READERS = {
    # Processes.
    "server.py",
    "migrate_db.py",
    "settings.py",
    "backoffice/server.py",
    "backoffice/db.py",
    # (``eval/eval_runner.py`` reads through ``settings.py``.)
    # Provider clients falling back to their SDK's own key variable.
    "llm_clients/openai_client.py",
    "llm_clients/gemini_client.py",
    "llm_clients/anthropic_client.py",
    "llm_clients/ollama_client.py",
    "llm_clients/embeddings.py",
    # A bundled tool: its proxy is a deployment fact, not a tool argument.
    "tools/webtools/http_client.py",
    # Reads whatever *name* the workflow YAML gives (url_env, password_env…).
    "functionkernel.py",
}


def documented_names() -> set[str]:
    return set(ASSIGNMENT.findall(ENV_EXAMPLE.read_text())) - READ_BY_SDK


def referenced_names() -> set[str]:
    """Every environment variable named anywhere under ``kavalai/``."""
    pattern = re.compile(
        r"""(?:getenv|environ\.get|env\.str|env\.int|env\.bool|env|required_setting)"""
        r"""\(\s*["']([A-Z][A-Z0-9_]*)["']"""
        r"""|environ\[["']([A-Z][A-Z0-9_]*)["']\]"""
        # The ``KAVALAI_LLM_*`` table in ``kavalai/settings.py``.
        r"""|^\s*["'](KAVALAI_[A-Z0-9_]*)["']:\s*\(""",
        re.MULTILINE,
    )
    names = set()
    for path in PACKAGE.rglob("*.py"):
        for first, second, third in pattern.findall(path.read_text()):
            names.add(first or second or third)
    return names - NOT_IN_TEMPLATE


def environment_reading_modules() -> set[str]:
    """Every module under ``kavalai/`` that touches ``os.environ``."""
    pattern = re.compile(
        r"\bos\.(?:getenv|environ)\b|\benv\.(?:str|int|bool|read_env)\(|\benv\("
    )
    return {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*.py")
        if pattern.search(path.read_text())
    }


def test_env_example_exists():
    assert ENV_EXAMPLE.exists(), ".env.example is the onboarding template"


def test_sdk_variables_are_marked():
    """A variable nothing in ``kavalai/`` reads has to say who does."""
    text = ENV_EXAMPLE.read_text()
    for name in sorted(READ_BY_SDK):
        block = text[: text.index(f"\n{name}=")]
        comment = block[block.rfind("\n\n") :]
        assert "SDK" in comment, f"{name} is not marked as read by the SDK"


def test_only_entry_points_read_the_environment():
    """Library code takes arguments; only the listed modules read the shell."""
    assert environment_reading_modules() == ENVIRONMENT_READERS


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
