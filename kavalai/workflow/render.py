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

"""Render a workflow graph to a standalone SVG diagram.

The diagram shows each node by name and connects them with arrows following the
workflow's transitions (the output of one node flows to the next). Branch nodes
mark the deciding condition on each outgoing arrow: an ``if`` labels its arrows
with the condition and ``else``; a ``switch`` labels each arrow with the case it
matches (plus ``default``). A ``parallel`` node draws one arrow per branch; the
join is where those branches' arrows converge.

``render_workflow_svg`` accepts either a :class:`~kavalai.WorkflowGraph` or the
plain ``dict`` JSON the backoffice stores for an agent, so the same renderer
serves the SDK, the backoffice UI and the documentation build.
"""

from typing import Any, Optional, Union
from xml.sax.saxutils import escape as _xml_escape

# Fill / stroke per node type. Mirrors the colours used by the backoffice graph
# so backend-rendered SVGs match the rest of the UI.
_NODE_COLORS = {
    "start": ("#16a34a", "#15803d"),
    "end": ("#dc2626", "#b91c1c"),
    "llm": ("#2563eb", "#1d4ed8"),
    "agent": ("#7c3aed", "#6d28d9"),
    "function": ("#0891b2", "#0e7490"),
    "if": ("#d97706", "#b45309"),
    "switch": ("#d97706", "#b45309"),
    "parallel": ("#4f46e5", "#4338ca"),
    "rag_query": ("#0d9488", "#0f766e"),
}
_DEFAULT_COLOR = ("#475569", "#334155")

# Layout geometry (pixels).
_BOX_W = 168
_BOX_H = 56
_GAP_X = 44
_GAP_Y = 76
_MARGIN = 28


def _esc(value: Any) -> str:
    """XML-escape a value for safe inclusion in SVG text/attributes."""
    return _xml_escape("" if value is None else str(value))


def _truncate(value: str, limit: int = 26) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _as_dict(workflow: Union[Any, dict]) -> dict:
    """Normalise a WorkflowGraph or its JSON into a plain dict."""
    if isinstance(workflow, dict):
        return workflow
    if hasattr(workflow, "model_dump"):
        return workflow.model_dump()
    raise TypeError(
        "render_workflow_svg expects a WorkflowGraph or a workflow dict, "
        f"got {type(workflow).__name__}"
    )


def _node_edges(node: dict) -> list[tuple[str, Optional[str]]]:
    """Return the outgoing ``(target, label)`` transitions of a node.

    Branch nodes label their arrows with the deciding condition; plain nodes
    have a single, unlabelled ``next`` transition.
    """
    node_type = node.get("type")
    edges: list[tuple[str, Optional[str]]] = []
    if node_type == "if":
        then = node.get("then")
        else_ = node.get("else_", node.get("else"))
        if then:
            edges.append((then, _truncate(node.get("condition") or "true")))
        if else_:
            edges.append((else_, "else"))
    elif node_type == "switch":
        for case_value, target in (node.get("cases") or {}).items():
            if target:
                edges.append((target, _truncate(str(case_value))))
        default = node.get("default")
        if default:
            edges.append((default, "default"))
    elif node_type == "parallel":
        # One arrow per branch. The join (``next``) needs no arrow of its own:
        # every branch's last node already transitions to it.
        for target in node.get("branches") or []:
            if target:
                edges.append((target, None))
    elif node_type != "end":
        nxt = node.get("next")
        if nxt:
            edges.append((nxt, None))
    return edges


def _back_edges(
    names: list[str], edges: dict[str, list[tuple[str, Optional[str]]]], start: str
) -> set[tuple[str, str]]:
    """Find loop-closing (back) edges via DFS so they don't distort the layering.

    A back-edge points to a node still on the current DFS path (an ancestor); it
    closes a cycle. Layering ignores these (otherwise a loop would push every
    node in it to the bottom), but they are still drawn.
    """
    color: dict[str, int] = {}  # 1 = on stack, 2 = done
    back: set[tuple[str, str]] = set()

    def visit(root: str) -> None:
        stack = [(root, iter(edges.get(root, [])))]
        color[root] = 1
        while stack:
            node, it = stack[-1]
            advanced = False
            for target, _label in it:
                if target not in edges:
                    continue
                state = color.get(target, 0)
                if state == 1:
                    back.add((node, target))
                elif state == 0:
                    color[target] = 1
                    stack.append((target, iter(edges.get(target, []))))
                    advanced = True
                    break
            if not advanced:
                color[node] = 2
                stack.pop()

    order = [start] + [n for n in names if n != start]
    for node in order:
        if color.get(node, 0) == 0:
            visit(node)
    return back


def _assign_depths(
    names: list[str], edges: dict[str, list[tuple[str, Optional[str]]]], start: str
) -> dict[str, int]:
    """Assign each node a row = its longest path from start over forward edges."""
    back = _back_edges(names, edges, start)
    forward = {
        n: [(t, lbl) for (t, lbl) in edges.get(n, []) if (n, t) not in back]
        for n in names
    }
    start = start if start in forward else (names[0] if names else "")
    depth: dict[str, int] = {start: 0} if start else {}
    # Longest-path over the now-acyclic forward edges (converges in <= |V| passes).
    for _ in range(len(names)):
        changed = False
        for src in list(depth):
            for target, _label in forward.get(src, []):
                nd = depth[src] + 1
                if target not in depth or depth[target] < nd:
                    depth[target] = nd
                    changed = True
        if not changed:
            break
    # Park any node unreachable from start in a row of its own at the bottom.
    bottom = max(depth.values(), default=-1) + 1
    for name in names:
        if name not in depth:
            depth[name] = bottom
            bottom += 1
    return depth


def _layout(
    names: list[str], depth: dict[str, int]
) -> tuple[dict[str, tuple[float, float]], float, float]:
    """Place nodes on a centred grid, returning positions and the canvas size."""
    rows: dict[int, list[str]] = {}
    for name in names:
        rows.setdefault(depth[name], []).append(name)

    max_in_row = max((len(r) for r in rows.values()), default=1)
    full_w = max_in_row * _BOX_W + (max_in_row - 1) * _GAP_X
    max_depth = max(depth.values(), default=0)

    pos: dict[str, tuple[float, float]] = {}
    for d, row in rows.items():
        row_w = len(row) * _BOX_W + (len(row) - 1) * _GAP_X
        x0 = _MARGIN + (full_w - row_w) / 2
        for i, name in enumerate(row):
            x = x0 + i * (_BOX_W + _GAP_X)
            y = _MARGIN + d * (_BOX_H + _GAP_Y)
            pos[name] = (x, y)

    width = _MARGIN * 2 + full_w
    height = _MARGIN * 2 + (max_depth + 1) * _BOX_H + max_depth * _GAP_Y
    return pos, width, height


def _edge_path(
    src: tuple[float, float], tgt: tuple[float, float]
) -> tuple[str, tuple[float, float], float]:
    """Return an SVG path for an edge, its label anchor, and its rightmost x."""
    sx, sy = src
    tx, ty = tgt
    s_cx = sx + _BOX_W / 2
    t_cx = tx + _BOX_W / 2
    if ty > sy:
        # Forward (downward) edge: a gentle vertical bezier, box-bottom to box-top.
        s_y, t_y = sy + _BOX_H, ty
        c = (s_y + t_y) / 2
        path = f"M {s_cx:.1f},{s_y:.1f} C {s_cx:.1f},{c:.1f} {t_cx:.1f},{c:.1f} {t_cx:.1f},{t_y:.1f}"
        return path, ((s_cx + t_cx) / 2, c), max(s_cx, t_cx)
    # Back / lateral edge: bow out to the right so it does not cross the boxes.
    s_x, s_y = sx + _BOX_W, sy + _BOX_H / 2
    t_x, t_y = tx + _BOX_W, ty + _BOX_H / 2
    bow = max(s_x, t_x) + _GAP_X + 24
    path = f"M {s_x:.1f},{s_y:.1f} C {bow:.1f},{s_y:.1f} {bow:.1f},{t_y:.1f} {t_x:.1f},{t_y:.1f}"
    return path, (bow - 6, (s_y + t_y) / 2), bow


def render_workflow_svg(workflow: Union[Any, dict]) -> str:
    """Render a workflow to a standalone SVG string.

    Args:
        workflow: a :class:`~kavalai.WorkflowGraph` or the workflow ``dict`` the
            backoffice stores for an agent.

    Returns:
        A complete ``<svg>…</svg>`` document. Nodes are drawn by name and joined
        by arrows; branch (``if`` / ``switch``) arrows are labelled with the
        condition that selects them.
    """
    data = _as_dict(workflow)
    nodes = data.get("nodes") or []
    by_name = {n.get("name"): n for n in nodes if n.get("name")}
    names = [n.get("name") for n in nodes if n.get("name")]
    edges = {name: _node_edges(by_name[name]) for name in names}
    start = data.get("start") or next(
        (n["name"] for n in nodes if n.get("type") == "start"),
        names[0] if names else "",
    )

    depth = _assign_depths(names, edges, start)
    pos, width, height = _layout(names, depth)

    # Build the edge fragments first; back-edges may bow past the grid, so widen
    # the canvas to fit them before emitting the header.
    edge_parts: list[str] = []
    for src_name in names:
        for target, label in edges[src_name]:
            if target not in pos:
                continue
            path, (lx, ly), max_x = _edge_path(pos[src_name], pos[target])
            width = max(width, max_x + _MARGIN)
            edge_parts.append(
                f'<path d="{path}" fill="none" stroke="#94a3b8" '
                'stroke-width="1.6" marker-end="url(#kv-arrow)"/>'
            )
            if label:
                w = max(len(label) * 6.6 + 10, 18)
                edge_parts.append(
                    f'<rect x="{lx - w / 2:.1f}" y="{ly - 9:.1f}" width="{w:.1f}" '
                    'height="18" rx="4" fill="#11111b" stroke="#313244"/>'
                    f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" '
                    f'font-size="11" fill="#cdd6f4">{_esc(label)}</text>'
                )

    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
            'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">'
        ),
        (
            '<defs><marker id="kv-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>'
        ),
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#1e1e2e"/>',
    ]
    parts.extend(edge_parts)

    # Node boxes (drawn last, so they sit on top of the lines).
    for name in names:
        node = by_name[name]
        node_type = node.get("type", "")
        fill, stroke = _NODE_COLORS.get(node_type, _DEFAULT_COLOR)
        x, y = pos[name]
        cx = x + _BOX_W / 2
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{_BOX_W}" height="{_BOX_H}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{cx:.1f}" y="{y + 26:.1f}" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#ffffff">'
            f"{_esc(_truncate(name, 22))}</text>"
            f'<text x="{cx:.1f}" y="{y + 43:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#ffffff" opacity="0.8">{_esc(node_type)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)
