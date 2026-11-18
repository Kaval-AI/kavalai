"""The package is typed (PEP 561), and stays checkable from the outside.

``kavalai/py.typed`` makes a consumer's type checker read Kaval.AI's
annotations. These tests run mypy the way a consumer does — strict, and with
errors inside the library silenced, as they are for an installed package — on
two files under ``tests/typecheck/``: one that must pass and one whose every
marked line must be rejected.
"""

import ast
import re
from pathlib import Path

from mypy import api

import kavalai

ROOT = Path(__file__).resolve().parents[1]
TYPECHECK = ROOT / "tests" / "typecheck"
EXPECTED = re.compile(r"#\s*expect:\s*([a-z-]+)")
REPORTED = re.compile(r":(\d+): error: .*\[([a-z-]+)\]$")


def run_mypy(path: Path) -> tuple[str, int]:
    """Check one file as a consumer would; returns mypy's output and status."""
    output, errors, status = api.run(
        [
            "--strict",
            "--follow-imports=silent",
            "--no-incremental",
            "--config-file=",
            str(path),
        ]
    )
    return output + errors, status


def test_the_marker_is_part_of_the_package():
    assert (Path(kavalai.__file__).parent / "py.typed").is_file()


def test_a_consumer_of_the_public_surface_type_checks():
    output, status = run_mypy(TYPECHECK / "consumer.py")
    assert status == 0, output


def test_wrong_calls_into_the_package_are_rejected():
    """Each marked mistake is reported, so the surface is not ``Any``."""
    path = TYPECHECK / "mistakes.py"
    expected = {
        (number, match.group(1))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if (match := EXPECTED.search(line))
    }
    output, _ = run_mypy(path)
    reported = {
        (int(match.group(1)), match.group(2))
        for line in output.splitlines()
        if (match := REPORTED.search(line))
    }
    assert expected, "mistakes.py marks no line"
    assert reported == expected, output


def test_the_lazy_clients_are_declared_for_type_checkers():
    """The ``TYPE_CHECKING`` imports are exactly the lazily resolved names.

    A provider added to the registry becomes importable at run time through
    the module ``__getattr__``; without a static import beside it a type
    checker would report the new name as missing.
    """
    tree = ast.parse(Path(kavalai.__file__).read_text())
    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                declared.update(alias.name for alias in statement.names)
    assert declared == set(kavalai._LAZY_CLIENTS)
    for name, (module_path, _) in kavalai._LAZY_CLIENTS.items():
        assert f"from {module_path} import {name}" in Path(kavalai.__file__).read_text()
