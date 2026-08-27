"""Sphinx extension: pre-render example workflow diagrams to SVG.

On every build this renders a couple of representative workflows — the home-page
chatbot and a branchy support agent — to ``_static/workflows/*.svg`` using
:func:`kavalai.workflow.render_workflow_svg`, so the docs can ``.. image::``
them. The files are regenerated on each build (not committed).
"""

from __future__ import annotations

from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _chatbot_graph():
    """The home-page chatbot: a single LLM node, start -> reply -> end."""
    from kavalai.workflow import WorkflowBuilder

    return (
        WorkflowBuilder("Chatbot", llm_model="openai/gpt-5.4-mini")
        .data_type("input", {"message": str})
        .data_type("output", {"agent_response": str})
        .start("reply")
        .llm(
            "reply",
            prompt="Reply to the user and suggest up to 3 quick-reply choices.",
            inputs={"message": "input"},
            output="output",
            next="end",
        )
        .end()
        .build()
    )


def _support_agent_graph(repo_root: Path):
    """A branchy workflow (a ``switch`` routing to several handlers)."""
    import yaml

    from kavalai.workflow import WorkflowGraph

    path = repo_root / "examples" / "support_agent" / "support_agent.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return WorkflowGraph(**data)


def _render_svgs(app: Sphinx) -> None:
    from kavalai.workflow import render_workflow_svg

    out_dir = Path(app.srcdir) / "_static" / "workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(app.srcdir).parent

    builders = {
        "chatbot": lambda: _chatbot_graph(),
        "support-agent": lambda: _support_agent_graph(repo_root),
    }
    for name, build in builders.items():
        try:
            svg = render_workflow_svg(build())
            (out_dir / f"{name}.svg").write_text(svg, encoding="utf-8")
            logger.info("[workflow-svgs] rendered %s.svg", name)
        except Exception as exc:  # pragma: no cover - best-effort during docs build
            logger.warning("[workflow-svgs] failed to render %s.svg: %s", name, exc)


def setup(app: Sphinx) -> dict:
    app.connect("builder-inited", _render_svgs)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
