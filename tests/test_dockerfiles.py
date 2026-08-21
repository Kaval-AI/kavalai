"""The container entrypoints must name modules that actually exist.

``agent.entrypoint.sh`` invoked ``kavalai.agents.server`` for months after that
package was renamed, and nothing caught it: no test ran the entrypoint, and
``docker-compose.yml`` defines no agent-server service. This is the cheapest
possible guard against that class of rot — it does not run the containers, it
just checks that every ``python -m <module>`` in them can be imported.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ENTRYPOINTS = sorted((Path(__file__).parent.parent / "dockerfiles").glob("*.sh"))
MODULE_PATTERN = re.compile(r"python -m ([\w.]+)")


def _module_references() -> list[tuple[str, str]]:
    return [
        (script.name, module)
        for script in ENTRYPOINTS
        for module in MODULE_PATTERN.findall(script.read_text())
    ]


def test_entrypoints_are_present():
    assert ENTRYPOINTS, "no entrypoint scripts found to check"


@pytest.mark.parametrize("script,module", _module_references())
def test_entrypoint_modules_are_importable(script, module):
    assert (
        importlib.util.find_spec(module) is not None
    ), f"{script} runs 'python -m {module}', which does not exist"
