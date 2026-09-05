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

Standard-library HTML → markdown-ish text extraction, for pages fetched with
a plain HTTP request. The browser path gets its markdown from crawl4ai; this
module exists so a server-side-rendered site needs no browser stack at all.
The output aims at RAG chunking, not fidelity: headings survive as ``#``
lines, lists as ``- `` lines, everything invisible (scripts, styles,
noscript fallbacks) is dropped, and links are collected separately rather
than rendered inline.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

INVISIBLE_TAGS = {"script", "style", "noscript", "template", "svg", "iframe"}

# Page chrome: its text is navigation boilerplate that would pollute every
# chunk, but its links are how a site without a sitemap is discovered — so
# the text is dropped and the hrefs are kept.
CHROME_TAGS = {"nav", "footer", "aside"}

BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "main",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "pre",
    "figure",
}
HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}

BLANK_LINES = re.compile(r"\n{3,}")
SPACES = re.compile(r"[ \t]+")


@dataclass
class ParsedHtml:
    """What one HTML document reduces to."""

    title: str = ""
    markdown: str = ""
    links: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    """Single-pass extraction of title, visible text and link targets."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._invisible_depth = 0
        self._chrome_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in INVISIBLE_TAGS:
            self._invisible_depth += 1
            return
        if self._invisible_depth:
            return
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))
            return
        if tag in CHROME_TAGS:
            self._chrome_depth += 1
            return
        if self._chrome_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in HEADING_TAGS:
            self.parts.append("\n\n" + "#" * HEADING_TAGS[tag] + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in INVISIBLE_TAGS:
            self._invisible_depth = max(0, self._invisible_depth - 1)
            return
        if self._invisible_depth:
            return
        if tag in CHROME_TAGS:
            self._chrome_depth = max(0, self._chrome_depth - 1)
            return
        if self._chrome_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._invisible_depth or self._chrome_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def parse_html(html: str, base_url: str = "") -> ParsedHtml:
    """Reduce an HTML document to a title, markdown-ish text and its links.

    Args:
        html: The document.
        base_url: Resolves relative ``href`` values; link targets are returned
            absolute.
    """
    extractor = _Extractor(base_url)
    extractor.feed(html)
    extractor.close()

    text = "".join(extractor.parts)
    # Heading-permalink glyphs (Sphinx and friends) are anchor text, not content.
    text = text.replace("\u00b6", "")
    text = SPACES.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    return ParsedHtml(
        title=" ".join("".join(extractor.title_parts).split()),
        markdown=text,
        links=extractor.links,
    )
