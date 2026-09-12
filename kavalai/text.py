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

Text for a RAG index: HTML or markdown in, chunks to embed out.

:func:`parse_html` reduces a page to its title, its visible text as
:class:`Block` objects that each carry the heading path above them, its links
and its robots directives, in one pass of the standard-library HTML parser.
:func:`chunk_blocks` packs blocks into :class:`Chunk` objects of about
``target_chars`` and never more than ``max_chars``, splitting an oversized
block at line breaks, then at sentence ends, then between words, and cutting
only a single over-long word. :func:`chunk_markdown` reads markdown into the
same blocks and packs them with the same function.

The module uses the standard library only, so it runs under Pyodide, and every
step is linear in its input: no regular expression is involved, and the
parser's bookkeeping is amortised constant work per tag.

The boilerplate rules are parameters of :func:`parse_html`; the defaults are
the module constants :data:`DROP_TAGS`, :data:`CHROME_TAGS`,
:data:`DROP_ROLES`, :data:`CONTENT_TAGS` and :data:`ROBOTS_NAMES`. Fetching
a page, reading PDF and detecting the language are outside its scope.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

__all__ = [
    "CHROME_TAGS",
    "CONTENT_TAGS",
    "DROP_ROLES",
    "DROP_TAGS",
    "HEADING_SEPARATOR",
    "MAX_CHARS",
    "ROBOTS_NAMES",
    "TARGET_CHARS",
    "Block",
    "Chunk",
    "ParsedPage",
    "chunk_blocks",
    "chunk_markdown",
    "parse_html",
]

HEADING_SEPARATOR = " › "
"""Joins the headings of a heading path: ``"Products › Pricing"``."""

TARGET_CHARS = 1200
"""The chunk size :func:`chunk_blocks` packs towards."""

MAX_CHARS = 2000
"""The chunk size :func:`chunk_blocks` never exceeds. Small embedding models
truncate at 512 tokens, so a much longer chunk would be embedded only in part."""

DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "iframe",
        "canvas",
        "object",
        "nav",
        "dialog",
        "button",
        "select",
        "textarea",
    }
)
"""Elements whose text is never content: code, fallbacks, navigation and form
controls. Their links are still collected."""

CHROME_TAGS = frozenset({"header", "footer", "aside"})
"""Elements that are page chrome outside a content element and part of it
inside one: an article's own ``<header>`` holds its title and byline, the
page's holds the logo and the menu."""

DROP_ROLES = frozenset(
    {
        "navigation",
        "banner",
        "contentinfo",
        "complementary",
        "search",
        "dialog",
        "alertdialog",
        "menu",
        "menubar",
    }
)
"""ARIA landmark roles that mark boilerplate on any element."""

CONTENT_TAGS = frozenset({"main", "article"})
"""Elements that mark a page's content. An element whose ``role`` names one of
them counts as that element."""

ROBOTS_NAMES = frozenset({"robots"})
"""``<meta name>`` values whose ``content`` is read as robots directives."""

PERMALINK_GLYPHS = "¶"
"""Characters removed from all text by default: the pilcrow that Sphinx and
MkDocs attach to every heading as a permalink."""

_HEADING_LEVELS = {f"h{level}": level for level in range(1, 7)}
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "center",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "hgroup",
        "legend",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    }
)
_BLOCK_KINDS = {
    "li": "item",
    "dt": "item",
    "dd": "item",
    "tr": "row",
    "pre": "code",
    "blockquote": "quote",
}
_CELL_TAGS = frozenset({"td", "th"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_CELL_SEPARATOR = " | "

_SKIP, _CONTENT, _BLOCK, _HEADING, _TEMPLATE = 1, 2, 4, 8, 16

_SENTENCE_ENDS = ".!?…।"
_UNSPACED_SENTENCE_ENDS = "。！？"
_CLOSERS = "\"')]»”’」』"


@dataclass(frozen=True)
class Block:
    """One paragraph-sized run of visible text and the headings above it.

    Attributes:
        text: The text, whitespace collapsed; a ``<br>`` is kept as a line
            break, and a ``code`` block keeps its whitespace.
        heading: The heading path above the block, joined by
            :data:`HEADING_SEPARATOR`. For a heading block, the path ends in
            the heading itself.
        kind: ``"paragraph"``, ``"heading"``, ``"item"`` (a list item,
            definition term or description), ``"row"`` (a table row, its cells
            separated by ``" | "``), ``"code"`` or ``"quote"``.
        level: The heading level, 1 to 6, for a heading block; 0 otherwise.
    """

    text: str
    heading: str = ""
    kind: str = "paragraph"
    level: int = 0


@dataclass(frozen=True)
class Chunk:
    """One piece of a document to embed.

    Attributes:
        text: What is embedded: the prefix (title and heading path) when one
            was asked for, a blank line, then the body.
        heading: The heading path the chunk's blocks sit under.
        position: The chunk's index in the list it was returned in.
    """

    text: str
    heading: str
    position: int


@dataclass
class ParsedPage:
    """What one HTML document reduces to.

    Attributes:
        title: The ``<title>``, whitespace collapsed; the first heading when
            the page has no title.
        blocks: The visible text in document order, heading blocks included.
        links: The ``href`` of every ``<a>`` and ``<area>``, boilerplate
            included, in document order without repeats; absolute when a base
            URL was given or the page declares ``<base href>``.
        noindex: Whether a robots ``<meta>`` tag says ``noindex`` or ``none``.
        nofollow: Whether a robots ``<meta>`` tag says ``nofollow`` or
            ``none``.
    """

    title: str
    blocks: list[Block]
    links: list[str]
    noindex: bool = False
    nofollow: bool = False

    @property
    def markdown(self) -> str:
        """The blocks as markdown: ``#`` headings, ``-`` items, fenced code.

        The rendering serves RAG chunking and a readable archive, not
        fidelity: inline formatting and link targets are not reproduced.
        """
        return _render_markdown(self.blocks)


@dataclass
class _Record:
    """A block as the parser first sees it, before the content filter."""

    kind: str
    text: str
    level: int
    in_content: bool
    heading: str
    content_heading: str


class _PageParser(HTMLParser):
    """The single pass behind :func:`parse_html`.

    The open-element stack records, per element, what entering it changed
    (``_SKIP``, ``_CONTENT``, …), so closing it undoes exactly that. An end
    tag pops down to its element; every element is pushed and popped once,
    which keeps malformed nesting linear.
    """

    def __init__(
        self,
        *,
        content_tags: frozenset[str],
        drop_tags: frozenset[str],
        chrome_tags: frozenset[str],
        drop_roles: frozenset[str],
        drop_hidden: bool,
        strip_chars: str,
        robots_names: frozenset[str],
    ):
        super().__init__(convert_charrefs=True)
        self.content_tags = content_tags
        self.drop_tags = drop_tags
        self.chrome_tags = chrome_tags
        self.drop_roles = drop_roles
        self.drop_hidden = drop_hidden
        self.strip_table = str.maketrans("", "", strip_chars)
        self.robots_names = robots_names

        self.records: list[_Record] = []
        self.hrefs: list[str] = []
        self.base_href: Optional[str] = None
        self.directives: set[str] = set()
        self.title_parts: list[str] = []

        self._open: list[str] = []
        self._flags: list[int] = []
        self._open_counts: dict[str, int] = {}
        self._kinds: list[str] = []
        self._skipping = False
        self._content_depth = 0
        self._template_depth = 0
        self._in_title = False
        self._title_done = False
        self._buffer: list[str] = []
        self._line_has_text = False
        self._heading_level = 0
        self._heading_index: Optional[int] = None
        self._heading_parts: list[str] = []
        self._path: dict[int, str] = {}
        self._content_path: dict[int, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        attributes = {name: value or "" for name, value in attrs}
        if tag == "meta":
            self._read_meta(attributes)
            return
        if tag == "base":
            if self.base_href is None and attributes.get("href", "").strip():
                self.base_href = attributes["href"].strip()
            return
        if tag in ("a", "area") and not self._template_depth:
            href = attributes.get("href", "").strip()
            if href:
                self.hrefs.append(href)
        if tag == "title":
            self._in_title = not self._skipping and not self._title_done
            return
        if tag in _VOID_TAGS:
            self._void_tag(tag)
            return

        flags = _TEMPLATE if tag == "template" else 0
        if flags:
            self._template_depth += 1
        if self._skipping:
            self._push(tag, flags)
            return
        role = _role(attributes)
        if self._drops(tag, role, attributes):
            self._skipping = True
            self._push(tag, flags | _SKIP)
            return

        is_content = tag in self.content_tags or role in self.content_tags
        if tag in _HEADING_LEVELS:
            self._start_heading(_HEADING_LEVELS[tag])
            flags |= _HEADING
        elif tag in _BLOCK_TAGS or is_content:
            self._flush()
            self._kinds.append(_BLOCK_KINDS.get(tag, "paragraph"))
            flags |= _BLOCK
        elif tag in _CELL_TAGS and self._line_has_text:
            self._buffer.append(_CELL_SEPARATOR)
        if is_content:
            self._content_depth += 1
            flags |= _CONTENT
        self._push(tag, flags)

    def handle_endtag(self, tag: str):
        if tag == "title":
            if self._in_title:
                self._in_title = False
                self._title_done = True
            return
        if tag in _VOID_TAGS:
            if tag == "br":
                self._void_tag(tag)
            return
        if self._open_counts.get(tag):
            while self._pop() != tag:
                pass
        elif tag in _HEADING_LEVELS and self._heading_index is not None:
            index = self._heading_index
            while len(self._open) > index:
                self._pop()
        elif tag in _BLOCK_TAGS and not self._skipping:
            self._flush()

    def handle_data(self, data: str):
        data = data.translate(self.strip_table)
        if self._in_title:
            self.title_parts.append(data)
        elif self._skipping:
            return
        elif self._heading_index is not None:
            self._heading_parts.append(data)
        elif self._kinds and self._kinds[-1] == "code":
            self._buffer.append(data.replace("\r\n", "\n"))
            self._line_has_text = self._line_has_text or not data.isspace()
        else:
            self._buffer.append(_collapse_keeping_edges(data))
            self._line_has_text = self._line_has_text or not data.isspace()

    def close(self):
        super().close()
        if self._heading_index is not None:
            self._finish_heading()
        self._flush()

    def _read_meta(self, attributes: dict[str, str]):
        if attributes.get("name", "").strip().lower() in self.robots_names:
            for directive in attributes.get("content", "").split(","):
                self.directives.add(directive.strip().lower())

    def _void_tag(self, tag: str):
        if self._skipping:
            return
        if tag == "br":
            if self._heading_index is not None:
                self._heading_parts.append(" ")
            else:
                self._buffer.append("\n")
                self._line_has_text = False
        elif tag == "hr":
            self._flush()

    def _drops(self, tag: str, role: str, attributes: dict[str, str]) -> bool:
        return (
            tag in self.drop_tags
            or (tag in self.chrome_tags and not self._content_depth)
            or role in self.drop_roles
            or (self.drop_hidden and _is_hidden(attributes))
        )

    def _push(self, tag: str, flags: int):
        self._open.append(tag)
        self._flags.append(flags)
        self._open_counts[tag] = self._open_counts.get(tag, 0) + 1

    def _pop(self) -> str:
        """Close the innermost open element, undoing what opening it changed."""
        tag = self._open.pop()
        flags = self._flags.pop()
        self._open_counts[tag] -= 1
        if flags & _TEMPLATE:
            self._template_depth -= 1
        if flags & _SKIP:
            self._skipping = False
        if flags & _HEADING and self._heading_index == len(self._open):
            self._finish_heading()
        if flags & _BLOCK:
            self._flush()
            self._kinds.pop()
        if flags & _CONTENT:
            self._content_depth -= 1
        return tag

    def _start_heading(self, level: int):
        if self._heading_index is not None:
            self._finish_heading()
        self._flush()
        self._heading_level = level
        self._heading_index = len(self._open)
        self._heading_parts = []

    def _finish_heading(self):
        text = " ".join("".join(self._heading_parts).split())
        level = self._heading_level
        in_content = self._content_depth > 0
        self._heading_index = None
        self._heading_parts = []
        _set_heading(self._path, level, text)
        if in_content:
            _set_heading(self._content_path, level, text)
        if text:
            self.records.append(
                _Record(
                    kind="heading",
                    text=text,
                    level=level,
                    in_content=in_content,
                    heading=_join_path(self._path),
                    content_heading=_join_path(self._content_path),
                )
            )

    def _flush(self):
        if not self._buffer:
            return
        raw = "".join(self._buffer)
        self._buffer.clear()
        self._line_has_text = False
        kind = self._kinds[-1] if self._kinds else "paragraph"
        text = _tidy_code(raw) if kind == "code" else _tidy_lines(raw)
        if text:
            self.records.append(
                _Record(
                    kind=kind,
                    text=text,
                    level=0,
                    in_content=self._content_depth > 0,
                    heading=_join_path(self._path),
                    content_heading=_join_path(self._content_path),
                )
            )


def parse_html(
    html: str,
    *,
    base_url: Optional[str] = None,
    content_tags: Iterable[str] = CONTENT_TAGS,
    drop_tags: Iterable[str] = DROP_TAGS,
    chrome_tags: Iterable[str] = CHROME_TAGS,
    drop_roles: Iterable[str] = DROP_ROLES,
    drop_hidden: bool = True,
    strip_chars: str = PERMALINK_GLYPHS,
    robots_names: Iterable[str] = ROBOTS_NAMES,
) -> ParsedPage:
    """Reduce an HTML document to its title, text blocks, links and robots meta.

    When the page has text inside a content element (``content_tags``), only
    that text is kept and only the headings inside content elements form the
    heading paths; otherwise the whole visible page is kept.

    Args:
        html: The document.
        base_url: The URL the document was fetched from; relative links are
            resolved against it, or against ``<base href>`` when the page
            declares one.
        content_tags: Elements that mark the page's content.
        drop_tags: Elements whose text is dropped wherever they occur.
        chrome_tags: Elements whose text is dropped outside a content element
            and kept inside one.
        drop_roles: ``role`` values whose element's text is dropped.
        drop_hidden: Whether to drop elements that are not rendered: the
            ``hidden`` attribute (except ``until-found``),
            ``aria-hidden="true"`` and an inline ``display: none`` or
            ``visibility: hidden``.
        strip_chars: Characters removed from all text.
        robots_names: ``<meta name>`` values whose directives are read; add a
            crawler's own token (``"mybot"``) to honour directives addressed
            to it.

    Returns:
        The parsed page.
    """
    parser = _PageParser(
        content_tags=frozenset(content_tags),
        drop_tags=frozenset(drop_tags),
        chrome_tags=frozenset(chrome_tags),
        drop_roles=frozenset(drop_roles),
        drop_hidden=drop_hidden,
        strip_chars=strip_chars,
        robots_names=frozenset(name.lower() for name in robots_names),
    )
    parser.feed(html)
    parser.close()

    records = parser.records
    if any(r.in_content and r.kind != "heading" for r in records):
        blocks = [
            Block(r.text, r.content_heading, r.kind, r.level)
            for r in records
            if r.in_content
        ]
    else:
        blocks = [Block(r.text, r.heading, r.kind, r.level) for r in records]

    title = " ".join("".join(parser.title_parts).split())
    if not title:
        title = next((b.text for b in blocks if b.kind == "heading"), "")
    directives = parser.directives
    return ParsedPage(
        title=title,
        blocks=blocks,
        links=_resolve_links(parser.hrefs, base_url, parser.base_href),
        noindex="noindex" in directives or "none" in directives,
        nofollow="nofollow" in directives or "none" in directives,
    )


def chunk_blocks(
    blocks: Iterable[Block],
    *,
    title: str = "",
    target_chars: Optional[int] = TARGET_CHARS,
    max_chars: Optional[int] = MAX_CHARS,
    prefix: bool = True,
) -> list[Chunk]:
    """Pack blocks into chunks of about ``target_chars``, at most ``max_chars``.

    Consecutive blocks under the same heading path are joined by a blank line
    until the next one would pass ``target_chars``; a new heading path always
    starts a new chunk. A block longer than the space left under
    ``max_chars`` is split at line breaks, then at sentence ends, then between
    words, and a single word longer than that is cut. Heading blocks
    contribute through the heading path, not as body text; blocks with no
    text are skipped.

    With ``prefix``, every chunk begins with the title and its heading path
    (``"Acme › Products › Pricing"``) and a blank line, so a chunk retrieved
    on its own still says what it is about. A heading path that already
    begins with the title does not repeat it. ``max_chars`` counts the
    prefix, which is shortened to at most half of it.

    Args:
        blocks: The blocks, in document order.
        title: The document title for the prefix.
        target_chars: The size chunks are packed towards; ``None`` packs up
            to ``max_chars``. A target above ``max_chars`` is lowered to it.
        max_chars: The size no chunk exceeds; ``None`` never splits a block,
            so each heading section packs to ``target_chars`` or, with both
            ``None``, stays one chunk.
        prefix: Whether to begin each chunk with the title and heading path.

    Returns:
        The chunks, ``position`` numbered from 0.

    Raises:
        ValueError: A size below 1.
    """
    for name, value in (("target_chars", target_chars), ("max_chars", max_chars)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if max_chars is not None:
        target_chars = min(target_chars or max_chars, max_chars)

    chunks: list[Chunk] = []
    heading: Optional[str] = None
    head = ""
    body: list[str] = []
    size = 0
    body_target: Optional[int] = None
    body_max: Optional[int] = None

    def emit():
        nonlocal size
        if body:
            text = "\n\n".join(body)
            chunks.append(
                Chunk(
                    text=f"{head}\n\n{text}" if head else text,
                    heading=heading or "",
                    position=len(chunks),
                )
            )
            body.clear()
            size = 0

    for block in blocks:
        text = block.text.strip()
        if block.kind == "heading" or not text:
            continue
        if block.heading != heading:
            emit()
            heading = block.heading
            head = _prefix(title.strip(), heading, max_chars) if prefix else ""
            overhead = len(head) + 2 if head else 0
            body_max = None if max_chars is None else max_chars - overhead
            body_target = (
                None if target_chars is None else max(1, target_chars - overhead)
            )
        for piece in _split_text(text, body_target, body_max):
            if body and body_target is not None:
                if size + 2 + len(piece) > body_target:
                    emit()
            size = size + 2 + len(piece) if body else len(piece)
            body.append(piece)
    emit()
    return chunks


def chunk_markdown(
    markdown: str,
    *,
    title: str = "",
    target_chars: Optional[int] = TARGET_CHARS,
    max_chars: Optional[int] = MAX_CHARS,
    prefix: bool = True,
) -> list[Chunk]:
    """Chunk markdown with :func:`chunk_blocks`.

    ATX headings (``#`` to ``######``) form the heading path; a blank line
    ends a paragraph block, and a fenced code block (```````` or ``~~~``)
    is one block whose ``#`` lines are not headings. The keyword arguments
    are those of :func:`chunk_blocks`.
    """
    return chunk_blocks(
        _markdown_blocks(markdown),
        title=title,
        target_chars=target_chars,
        max_chars=max_chars,
        prefix=prefix,
    )


def _role(attributes: dict[str, str]) -> str:
    """The first token of ``role``, which is the one a browser applies."""
    tokens = attributes.get("role", "").split()
    return tokens[0].lower() if tokens else ""


def _is_hidden(attributes: dict[str, str]) -> bool:
    if "hidden" in attributes and attributes["hidden"].lower() != "until-found":
        return True
    if attributes.get("aria-hidden", "").strip().lower() == "true":
        return True
    style = "".join(attributes.get("style", "").lower().split())
    return "display:none" in style or "visibility:hidden" in style


def _collapse_keeping_edges(data: str) -> str:
    """Collapse whitespace to single spaces, keeping one at either edge.

    ``"a <b>b</b>"`` must stay two words and ``"a<b>b</b>"`` one, so the space
    at a text node's edge is significant.
    """
    collapsed = " ".join(data.split())
    if not collapsed:
        return " " if data else ""
    if data[0].isspace():
        collapsed = " " + collapsed
    if data[-1].isspace():
        collapsed += " "
    return collapsed


def _tidy_lines(raw: str) -> str:
    lines = (" ".join(line.split()) for line in raw.split("\n"))
    return "\n".join(line for line in lines if line)


def _tidy_code(raw: str) -> str:
    lines = [line.rstrip() for line in raw.split("\n")]
    return "\n".join(lines).strip("\n")


def _set_heading(path: dict[int, str], level: int, text: str):
    for deeper in [k for k in path if k >= level]:
        del path[deeper]
    if text:
        path[level] = text


def _join_path(path: dict[int, str]) -> str:
    return HEADING_SEPARATOR.join(path[k] for k in sorted(path))


def _resolve_links(
    hrefs: list[str], base_url: Optional[str], base_href: Optional[str]
) -> list[str]:
    """Absolute, de-duplicated links; an unparseable ``href`` is dropped."""
    base = base_url
    if base_href:
        base = _join_url(base_url, base_href) if base_url else base_href
    links: list[str] = []
    for href in hrefs:
        link = _join_url(base, href) if base else href
        if link:
            links.append(link)
    return list(dict.fromkeys(links))


def _join_url(base: Optional[str], href: str) -> Optional[str]:
    try:
        return urljoin(base, href)
    except ValueError:
        return None


def _prefix(title: str, heading: str, max_chars: Optional[int]) -> str:
    if (
        heading
        and title
        and (heading == title or heading.startswith(title + HEADING_SEPARATOR))
    ):
        head = heading
    else:
        head = HEADING_SEPARATOR.join(part for part in (title, heading) if part)
    if max_chars is not None:
        limit = max_chars // 2 - 2
        if limit < 1:
            return ""
        if len(head) > limit:
            head = head[: limit - 1] + "…"
    return head


def _split_text(text: str, target: Optional[int], maximum: Optional[int]) -> list[str]:
    """``text`` in pieces of at most ``maximum``, each packed towards ``target``."""
    if maximum is None or len(text) <= maximum:
        return [text]
    return _split(text, target or maximum, maximum, 0)


def _split(text: str, target: int, maximum: int, level: int) -> list[str]:
    """Split at line breaks (level 0), sentences (1), words (2), then hard (3).

    A part that fits ``maximum`` is not split further; the parts of a level
    are then packed back together up to ``target``.
    """
    if len(text) <= maximum:
        return [text]
    if level == 1:
        return [
            piece.strip()
            for piece in _pack(_sentence_units(text, target, maximum), "", target)
        ]
    if level == 0:
        parts, separator = text.split("\n"), "\n"
    elif level == 2:
        parts, separator = text.split(), " "
    else:
        return [text[i : i + target] for i in range(0, len(text), target)]
    units: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            units.extend(_split(part, target, maximum, level + 1))
    return _pack(units, separator, target)


def _sentence_units(text: str, target: int, maximum: int) -> list[str]:
    """The sentences of ``text``, each keeping one space before it if it had any.

    Sentences are packed back without a separator, so text that had no space
    between its sentences (Chinese, Japanese) gets none; the space before a
    sentence is counted against the limit only where it stays inside a piece.
    """
    units: list[str] = []
    for sentence in _sentences(text):
        pieces = _split(sentence.strip(), target, maximum, 2)
        if units and sentence[0].isspace():
            pieces[0] = " " + pieces[0]
        units.extend(pieces)
    return units


def _sentences(text: str) -> list[str]:
    """Split after sentence-ending punctuation.

    ``.``, ``!``, ``?`` and ``…`` end a sentence when whitespace follows them,
    after any closing quotes or brackets; the ideographic full stop and its
    kin end one without a space.
    """
    parts: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in _SENTENCE_ENDS or char in _UNSPACED_SENTENCE_ENDS:
            j = i + 1
            while j < n and (
                text[j] in _CLOSERS
                or text[j] in _SENTENCE_ENDS
                or text[j] in _UNSPACED_SENTENCE_ENDS
            ):
                j += 1
            if char in _UNSPACED_SENTENCE_ENDS or j == n or text[j].isspace():
                parts.append(text[start:j])
                start = j
            i = j
        else:
            i += 1
    if start < n:
        parts.append(text[start:])
    return parts


def _pack(units: list[str], separator: str, limit: int) -> list[str]:
    """Join consecutive units while the result stays within ``limit``."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(separator) + len(unit) > limit:
            pieces.append(separator.join(current))
            current = []
        size = size + len(separator) + len(unit) if current else len(unit)
        current.append(unit)
    if current:
        pieces.append(separator.join(current))
    return pieces


def _markdown_blocks(markdown: str) -> list[Block]:
    """Read markdown into heading, paragraph and code blocks."""
    blocks: list[Block] = []
    path: dict[int, str] = {}
    heading = ""
    lines: list[str] = []
    fence = ""

    def flush(kind: str = "paragraph"):
        text = "\n".join(lines).strip("\n")
        lines.clear()
        if text.strip():
            blocks.append(Block(text, heading, kind))

    for line in markdown.splitlines():
        stripped = line.strip()
        if fence:
            lines.append(line)
            if len(stripped) >= len(fence) and not stripped.strip(fence[0]):
                fence = ""
                flush("code")
            continue
        marker = _fence_marker(stripped)
        if marker:
            flush()
            fence = marker
            lines.append(line)
            continue
        level, text = _atx_heading(line)
        if level:
            flush()
            _set_heading(path, level, text)
            heading = _join_path(path)
            if text:
                blocks.append(Block(text, heading, "heading", level))
        elif not stripped:
            flush()
        else:
            lines.append(line.rstrip())
    flush("code" if fence else "paragraph")
    return blocks


def _fence_marker(stripped: str) -> str:
    """The opening fence (three or more backticks or tildes), or ``""``."""
    for char in "`~":
        if stripped.startswith(char * 3):
            return stripped[: len(stripped) - len(stripped.lstrip(char))]
    return ""


def _atx_heading(line: str) -> tuple[int, str]:
    """``(level, text)`` of an ATX heading line; level 0 for any other line."""
    body = line.lstrip(" ")
    if len(line) - len(body) > 3:
        return 0, ""
    level = len(body) - len(body.lstrip("#"))
    rest = body[level:]
    if not 1 <= level <= 6 or (rest and rest[0] not in " \t"):
        return 0, ""
    text = rest.strip()
    closed = text.rstrip("#")
    if closed != text and (not closed or closed[-1] in " \t"):
        text = closed.rstrip()
    return level, text


def _render_markdown(blocks: list[Block]) -> str:
    parts: list[str] = []
    previous = ""
    for block in blocks:
        if parts:
            tight = block.kind == previous and block.kind in ("item", "row")
            parts.append("\n" if tight else "\n\n")
        parts.append(_render_block(block))
        previous = block.kind
    return "".join(parts)


def _render_block(block: Block) -> str:
    if block.kind == "heading":
        return f"{'#' * block.level} {block.text}"
    if block.kind == "item":
        return "- " + block.text.replace("\n", "\n  ")
    if block.kind == "quote":
        return "\n".join(f"> {line}" for line in block.text.split("\n"))
    if block.kind == "code":
        fence = "`" * max(3, _longest_run(block.text, "`") + 1)
        return f"{fence}\n{block.text}\n{fence}"
    return block.text


def _longest_run(text: str, char: str) -> int:
    longest = run = 0
    for current in text:
        run = run + 1 if current == char else 0
        longest = max(longest, run)
    return longest
