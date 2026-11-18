"""The package is typed (PEP 561), and stays checkable from the outside.

``kavalai/py.typed`` makes a consumer's type checker read Kaval.AI's
annotations. These tests check the two things that must hold for that to
work: the marker ships with the package, and every lazily imported provider
client is also declared statically.
"""

import ast
from pathlib import Path

import kavalai


def test_the_marker_is_part_of_the_package():
    assert (Path(kavalai.__file__).parent / "py.typed").is_file()


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
