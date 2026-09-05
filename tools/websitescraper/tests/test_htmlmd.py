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

from tools.websitescraper.htmlmd import parse_html

PAGE = """
<html>
  <head>
    <title>  Kaval.AI —
      Docs </title>
    <style>body { color: red; }</style>
    <script>var tracking = true;</script>
  </head>
  <body>
    <noscript>Please enable JavaScript.</noscript>
    <h1>Welcome<a class="headerlink" href="#welcome">¶</a></h1>
    <p>First &amp; second.</p>
    <h2>Features</h2>
    <ul><li>alpha</li><li>beta</li></ul>
    <a href="/pricing">Pricing</a>
    <a href="https://other.example/x">Elsewhere</a>
    <a name="anchor-without-href">nothing</a>
    <svg><text>decorative</text></svg>
  </body>
</html>
"""


def test_title_is_whitespace_normalized():
    assert parse_html(PAGE).title == "Kaval.AI — Docs"


def test_markdown_keeps_structure_and_drops_invisible_content():
    markdown = parse_html(PAGE).markdown
    assert "# Welcome" in markdown
    assert "\u00b6" not in markdown
    assert "## Features" in markdown
    assert "- alpha" in markdown
    assert "First & second." in markdown
    # Scripts, styles, noscript fallbacks and SVG text are not content.
    assert "tracking" not in markdown
    assert "color: red" not in markdown
    assert "enable JavaScript" not in markdown
    assert "decorative" not in markdown


def test_links_are_absolute_and_hrefless_anchors_ignored():
    links = parse_html(PAGE, base_url="https://kaval.ai/docs/").links
    # Fragment-only anchors resolve to the page itself here; the scraper's
    # link normalization is what drops them from the frontier.
    assert links == [
        "https://kaval.ai/docs/#welcome",
        "https://kaval.ai/pricing",
        "https://other.example/x",
    ]


def test_line_breaks_and_blank_line_collapsing():
    parsed = parse_html("<p>a</p><div></div><div></div><p>b<br>c</p>")
    assert parsed.markdown == "a\n\nb\nc"


def test_empty_document():
    parsed = parse_html("")
    assert parsed.title == ""
    assert parsed.markdown == ""
    assert parsed.links == []


def test_chrome_text_is_dropped_but_its_links_survive():
    parsed = parse_html(
        '<body><nav><a href="/docs">Docs</a><span>Skip to content</span></nav>'
        "<h1>Real</h1><p>content</p>"
        '<footer><a href="/imprint">Imprint</a>© Kaval</footer></body>',
        base_url="https://kaval.ai/",
    )
    assert parsed.markdown == "# Real\n\ncontent"
    assert parsed.links == ["https://kaval.ai/docs", "https://kaval.ai/imprint"]
