"""Sphinx extension: render the architecture diagram to SVG.

The figure on :doc:`tutorials/architecture` is generated here rather than
committed as artwork, so that its content stays editable as prose rather than
as vector geometry. Colours mirror ``kavalai/workflow/render.py``, so the
architecture diagram and the rendered workflow diagrams read as one family.

All label text sits inside a filled box in white, which keeps the figure
legible on both the light and the dark documentation themes.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

WIDTH = 924
HEIGHT = 946
FONT = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, Helvetica, Arial, sans-serif"
)
ARROW = "#94a3b8"
MUTED = "#64748b"


def _build_svg() -> str:
    parts: list[str] = []

    def box(x, y, w, h, fill, stroke, title, sub=None, chips=None, title_size=15):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        cx = x + w / 2
        if sub:
            ty = y + h / 2 - 4
            parts.append(
                f'<text x="{cx}" y="{ty}" text-anchor="middle" fill="#ffffff" '
                f'font-family="{FONT}" font-size="{title_size}" '
                f'font-weight="600">{escape(title)}</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{ty + 17}" text-anchor="middle" '
                f'fill="#ffffff" fill-opacity="0.82" font-family="{FONT}" '
                f'font-size="11">{escape(sub)}</text>'
            )
        else:
            parts.append(
                f'<text x="{cx}" y="{y + h / 2 + 5}" text-anchor="middle" '
                f'fill="#ffffff" font-family="{FONT}" '
                f'font-size="{title_size}" font-weight="600">'
                f"{escape(title)}</text>"
            )

    def chip_row(y, h, labels, fill, stroke, x0=24, total=852, gap=10, size=12):
        n = len(labels)
        w = (total - gap * (n - 1)) / n
        for i, label in enumerate(labels):
            x = x0 + i * (w + gap)
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="7" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 4}" '
                f'text-anchor="middle" fill="#ffffff" font-family="{FONT}" '
                f'font-size="{size}" font-weight="600">{escape(label)}</text>'
            )

    def arrow(x, y1, y2, label=None, label_dx=10):
        parts.append(
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 2}" stroke="{ARROW}" '
            f'stroke-width="1.8" marker-end="url(#head)"/>'
        )
        if label:
            parts.append(
                f'<text x="{x + label_dx}" y="{(y1 + y2) / 2 + 4}" '
                f'fill="{MUTED}" font-family="{FONT}" font-size="11.5" '
                f'font-style="italic">{escape(label)}</text>'
            )

    # --- palette (mirrors render.py) -------------------------------------------
    SLATE = ("#475569", "#334155")
    INDIGO = ("#4f46e5", "#4338ca")
    BLUE = ("#2563eb", "#1d4ed8")
    VIOLET = ("#7c3aed", "#6d28d9")
    CYAN = ("#0891b2", "#0e7490")
    GREEN = ("#16a34a", "#15803d")
    AMBER = ("#d97706", "#b45309")
    DARK = ("#334155", "#1e293b")
    TEAL = ("#0f766e", "#115e59")

    # --- band A: how a workflow is authored ------------------------------------
    box(24, 30, 274, 54, *SLATE, "workflow.yaml", "a graph as reviewable data")
    box(314, 30, 274, 54, *SLATE, "WorkflowBuilder", "the same graph in Python")
    box(604, 30, 272, 54, *SLATE, "kavalai.server", "FastAPI · JSON and SSE")

    # --- band B: the compiled graph --------------------------------------------
    box(
        24,
        132,
        852,
        54,
        *INDIGO,
        "WorkflowGraph",
        "Pydantic model · names, edges, types and reachability checked at load",
    )
    arrow(161, 84, 132, "parsed")
    arrow(451, 84, 132, "built")
    arrow(740, 84, 132, "loads")

    # --- band C: the engine ----------------------------------------------------
    box(
        24,
        234,
        852,
        54,
        *INDIGO,
        "WorkflowEngine",
        "run_stream() is the only execution path · one engine serves many "
        "concurrent runs",
    )
    arrow(450, 186, 234, "executed by")

    chip_row(
        306,
        34,
        [
            "RunContext · per run",
            "SchemaParser · data_types → models",
            "expressions · safe evaluation",
            "TokenAccumulator · per run",
        ],
        *INDIGO,
        size=11,
    )

    # --- band D: node kinds ----------------------------------------------------
    chip_row(
        382,
        36,
        ["start", "end", "llm", "agent", "function", "if", "switch", "parallel"],
        "#6366f1",
        "#4338ca",
    )
    arrow(450, 340, 382, "dispatches to")

    # --- band E: capabilities --------------------------------------------------
    box(24, 470, 274, 56, *VIOLET, "Agent", "plan → act → observe, until answered")
    box(314, 470, 274, 56, *CYAN, "FunctionKernel", "python:// · rest:// · mcp://")
    box(604, 470, 272, 56, *GREEN, "RAG services", "index and query embeddings")
    arrow(161, 418, 470)
    arrow(451, 418, 470, "calls")
    arrow(740, 418, 470)

    # --- band F: the outside world ---------------------------------------------
    box(
        24,
        570,
        274,
        56,
        *BLUE,
        "LLM clients",
        "OpenAI · Gemini · Anthropic · Ollama · WebLLM",
    )
    box(
        314,
        570,
        274,
        56,
        *CYAN,
        "Tool servers",
        "your functions · REST APIs · MCP servers",
    )
    box(
        604,
        570,
        272,
        56,
        *BLUE,
        "Embedding clients",
        "fastembed · openai · gemini · ollama",
    )
    arrow(161, 526, 570)
    arrow(451, 526, 570)
    arrow(740, 526, 570)

    # --- band G: persistence ---------------------------------------------------
    box(
        24,
        670,
        414,
        54,
        *AMBER,
        "AgentService",
        "agents · sessions · runs · chat_messages",
    )
    box(
        462,
        670,
        414,
        54,
        *AMBER,
        "TaskLogger",
        "tasks · model_call_stats",
    )
    # The engine writes to persistence directly; draw that as a side rail.
    parts.append(
        f'<path d="M 876 261 L 898 261 L 898 648 L 669 648 L 669 666" '
        f'fill="none" stroke="{ARROW}" stroke-width="1.8" '
        f'stroke-dasharray="5 4" marker-end="url(#head)"/>'
    )
    parts.append(
        f'<text x="914" y="455" fill="{MUTED}" font-family="{FONT}" '
        f'font-size="11.5" font-style="italic" text-anchor="middle" '
        f'transform="rotate(-90 914 455)">every run recorded</text>'
    )

    # --- band H: the database --------------------------------------------------
    box(
        24,
        770,
        852,
        52,
        *DARK,
        "Your database — PostgreSQL, local SQLite, or SQLite in the browser",
        title_size=14,
    )
    arrow(231, 724, 770)
    arrow(669, 724, 770)

    # --- band I: the backoffice ------------------------------------------------
    box(
        24,
        870,
        852,
        50,
        *TEAL,
        "Backoffice UI — reads the same tables, adds nothing to them",
        title_size=14,
    )
    arrow(450, 822, 870, "queried by")

    # One element per line, indented one level: the file is committed, so it is
    # read in diffs. A single 16 kB line reports every edit as "the whole
    # picture changed"; a line per element reports the box that moved.
    body = "\n".join(f"  {part}" for part in parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        f'role="img" aria-label="Kaval.AI component architecture">\n'
        f"  <defs>\n"
        f'    <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n'
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/>\n'
        f"    </marker>\n"
        f"  </defs>\n"
        f"{body}\n"
        f"</svg>\n"
    )


def _render(app: Sphinx) -> None:
    out_dir = Path(app.srcdir) / "_static"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / "architecture.svg").write_text(_build_svg(), encoding="utf-8")
        logger.info("[architecture-svg] rendered architecture.svg")
    except Exception as exc:  # pragma: no cover - best-effort during the build
        logger.warning("[architecture-svg] failed to render: %s", exc)


def setup(app: Sphinx) -> dict:
    app.connect("builder-inited", _render)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
