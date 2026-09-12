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

import html as htmllib
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavalai.text import (
    MAX_CHARS,
    Block,
    Chunk,
    ParsedPage,
    chunk_blocks,
    chunk_markdown,
    parse_html,
)

PAGE = """
<html><head>
  <title> Acme  Shop </title>
  <script>var tracking = 1;</script><style>p { color: red }</style>
</head>
<body>
<header><a href="/">Acme</a> <nav><a href="/about">About</a></nav></header>
<div role="navigation">Home | Products</div>
<main>
  <h1>Welcome to Acme<a class="headerlink" href="#welcome">¶</a></h1>
  <p>We make <b>widgets</b> and gadgets.</p>
  <h2>Products</h2>
  <p>Our widgets come in three sizes.</p>
  <ul><li>Small</li><li>Medium</li><li>Large</li></ul>
  <h3>Pricing</h3>
  <table>
    <tr><th>Size</th><th>Price</th></tr>
    <tr><td>Small</td><td>10 EUR</td></tr>
  </table>
  <h2>Contact</h2>
  <p>Write to us.<br>We answer within a day.</p>
  <p hidden>You should not see this.</p>
</main>
<aside>Newsletter signup</aside>
<footer>© Acme 2026 <a href="/imprint">Imprint</a></footer>
</body></html>
"""


def texts(page: ParsedPage) -> list[str]:
    return [block.text for block in page.blocks if block.kind != "heading"]


def squeeze(text: str) -> str:
    return "".join(text.split())


class TestParseHtml:
    def test_boilerplate_is_dropped_and_main_kept(self):
        page = parse_html(PAGE)
        assert page.title == "Acme Shop"
        assert texts(page) == [
            "We make widgets and gadgets.",
            "Our widgets come in three sizes.",
            "Small",
            "Medium",
            "Large",
            "Size | Price",
            "Small | 10 EUR",
            "Write to us.\nWe answer within a day.",
        ]
        joined = " ".join(block.text for block in page.blocks)
        for boilerplate in ("Home", "About", "Newsletter", "©", "tracking", "¶"):
            assert boilerplate not in joined

    def test_blocks_carry_their_heading_path_and_kind(self):
        page = parse_html(PAGE)
        by_text = {block.text: block for block in page.blocks}
        assert by_text["We make widgets and gadgets."].heading == "Welcome to Acme"
        pricing = by_text["Small | 10 EUR"]
        assert pricing.heading == "Welcome to Acme › Products › Pricing"
        assert pricing.kind == "row"
        assert by_text["Medium"].kind == "item"
        # An h2 resets the h3 below the previous h2.
        assert by_text["Write to us.\nWe answer within a day."].heading == (
            "Welcome to Acme › Contact"
        )
        heading = by_text["Pricing"]
        assert (heading.kind, heading.level) == ("heading", 3)
        assert heading.heading == "Welcome to Acme › Products › Pricing"

    def test_links_include_boilerplate_and_resolve_against_the_base(self):
        page = parse_html(PAGE, base_url="https://acme.example/shop/")
        assert page.links == [
            "https://acme.example/",
            "https://acme.example/about",
            "https://acme.example/shop/#welcome",
            "https://acme.example/imprint",
        ]

    def test_links_without_a_base_are_returned_as_written(self):
        page = parse_html('<a href=" /a ">x</a><a href="/a">y</a><a name="n">z</a>')
        assert page.links == ["/a"]

    def test_base_href_overrides_the_fetch_url(self):
        markup = '<base href="/docs/"><a href="page">p</a><area href="map">'
        page = parse_html(markup, base_url="https://acme.example/x/y")
        assert page.links == [
            "https://acme.example/docs/page",
            "https://acme.example/docs/map",
        ]
        absolute = parse_html('<base href="https://cdn.example/"><a href="a">a</a>')
        assert absolute.links == ["https://cdn.example/a"]
        second = parse_html('<base href="/one/"><base href="/two/"><a href="a">a</a>')
        assert second.links == ["/one/a"]

    def test_an_unparseable_link_is_dropped(self):
        page = parse_html(
            '<a href="http://[::1">x</a><a href="/ok">ok</a>',
            base_url="https://a.example/",
        )
        assert page.links == ["https://a.example/ok"]

    def test_links_inside_a_template_are_inert(self):
        page = parse_html('<template><a href="/t">t</a></template><a href="/r">r</a>')
        assert page.links == ["/r"]

    def test_robots_meta(self):
        page = parse_html('<meta name="Robots" content="NoIndex, follow">')
        assert (page.noindex, page.nofollow) == (True, False)
        page = parse_html('<meta name="robots" content="none">')
        assert (page.noindex, page.nofollow) == (True, True)
        page = parse_html('<meta name="robots" content="nofollow">')
        assert (page.noindex, page.nofollow) == (False, True)
        page = parse_html('<meta name="description" content="noindex">')
        assert (page.noindex, page.nofollow) == (False, False)

    def test_robots_names_add_a_crawler_token(self):
        markup = '<meta name="mybot" content="noindex">'
        assert not parse_html(markup).noindex
        assert parse_html(markup, robots_names=["robots", "MyBot"]).noindex

    def test_page_without_content_element_keeps_the_body(self):
        page = parse_html(
            "<body><nav>menu</nav><h1>Only heading</h1><p>Body text.</p></body>"
        )
        assert page.title == "Only heading"
        assert texts(page) == ["Body text."]

    def test_headings_outside_the_content_do_not_enter_its_paths(self):
        page = parse_html(
            "<h1>Site</h1><div><h2>Related</h2><p>ad</p></div>"
            "<main><p>Intro.</p><h2>Part</h2><p>Body.</p></main>"
        )
        assert [(b.text, b.heading) for b in page.blocks] == [
            ("Intro.", ""),
            ("Part", "Part"),
            ("Body.", "Part"),
        ]

    def test_a_content_element_with_only_a_heading_keeps_the_whole_page(self):
        page = parse_html("<main><h1>Title</h1></main><p>Text.</p>")
        assert [b.text for b in page.blocks] == ["Title", "Text."]

    def test_role_main_counts_as_content(self):
        page = parse_html('<p>chrome</p><div role="main"><p>Real.</p></div>')
        assert texts(page) == ["Real."]

    def test_chrome_inside_an_article_belongs_to_it(self):
        page = parse_html(
            "<header>Site logo</header>"
            "<article><header><h1>Post</h1><p>By Ann</p></header>"
            "<p>Text.</p><aside>Note.</aside><footer>Tags</footer></article>"
            "<footer>Site footer</footer>"
        )
        assert texts(page) == ["By Ann", "Text.", "Note.", "Tags"]
        assert page.title == "Post"

    def test_hidden_elements(self):
        page = parse_html(
            "<p hidden>a</p><p hidden=until-found>b</p>"
            '<p aria-hidden="true">c</p><p aria-hidden="false">d</p>'
            '<p style="display: none">e</p><p style="Visibility:Hidden">f</p>'
            '<p style="color: red">g</p><div role="dialog">h</div>'
            '<div role="Banner extra">i</div><div role="">j</div>'
        )
        assert texts(page) == ["b", "d", "g", "j"]
        shown = parse_html("<p hidden>a</p>", drop_hidden=False)
        assert texts(shown) == ["a"]

    def test_rules_are_parameters(self):
        markup = "<nav>menu</nav><main><p>main</p></main><section>more</section>"
        assert texts(parse_html(markup)) == ["main"]
        assert texts(parse_html(markup, content_tags=())) == ["main", "more"]
        assert texts(parse_html(markup, content_tags=(), drop_tags=())) == [
            "menu",
            "main",
            "more",
        ]
        dropped = parse_html(markup, content_tags=(), drop_tags={"section"})
        assert texts(dropped) == ["menu", "main"]
        assert texts(parse_html("<footer>f</footer>", chrome_tags=())) == ["f"]
        assert texts(parse_html('<div role="navigation">n</div>', drop_roles=())) == [
            "n"
        ]

    def test_strip_chars(self):
        assert texts(parse_html("<p>a¶b</p>")) == ["ab"]
        assert texts(parse_html("<p>a¶b|</p>", strip_chars="|")) == ["a¶b"]

    def test_title_is_the_first_title_outside_dropped_elements(self):
        page = parse_html(
            "<svg><title>icon</title></svg><title>Real</title><title>Second</title>"
        )
        assert page.title == "Real"
        assert parse_html("<p>text</p>").title == ""

    def test_inline_whitespace(self):
        page = parse_html("<p>a<b>b</b> <i>c</i>\n  d<span> </span>e</p>")
        assert texts(page) == ["ab c d e"]

    def test_table_cells_ignore_leading_whitespace(self):
        page = parse_html("<table><tr>\n  <td>A</td>\n  <td>B</td></tr></table>")
        assert texts(page) == ["A | B"]

    def test_pre_keeps_its_whitespace(self):
        page = parse_html("<pre>\n  def f():\r\n      return 1  \n</pre>")
        (block,) = page.blocks
        assert block.kind == "code"
        assert block.text == "  def f():\n      return 1"
        assert parse_html("<pre>\n\n</pre>").blocks == []

    def test_line_breaks(self):
        page = parse_html("<p>a<br>b<br/><br>c</p><h2>x<br>y</h2><hr><p>z</p>")
        assert [b.text for b in page.blocks] == ["a\nb\nc", "x y", "z"]
        assert texts(parse_html("<p>a</br>b</p>")) == ["a\nb"]

    def test_hr_splits_blocks(self):
        assert texts(parse_html("<p>a<hr>b</p>")) == ["a", "b"]

    def test_void_tags_inside_dropped_elements(self):
        page = parse_html("<nav>a<br>b<hr>c<img src=x></nav><p>d</p>")
        assert texts(page) == ["d"]

    def test_nested_blocks(self):
        page = parse_html(
            "<ul><li>outer<ul><li>inner</li></ul>tail</li></ul>"
            "<blockquote>quoted</blockquote>"
        )
        assert [(b.text, b.kind) for b in page.blocks] == [
            ("outer", "item"),
            ("inner", "item"),
            ("tail", "item"),
            ("quoted", "quote"),
        ]

    def test_malformed_nesting(self):
        page = parse_html("<div><p>a<b>b</div>c</p><p>d")
        assert texts(page) == ["ab", "c", "d"]
        # A stray end tag of an element that is not open is ignored, except
        # that a block end still ends the text before it.
        assert texts(parse_html("a</span>b</p>c")) == ["ab", "c"]
        assert texts(parse_html("<nav>a</p>b</nav>c")) == ["c"]

    def test_heading_end_tags_of_another_level_close_the_heading(self):
        page = parse_html("<h2>Title</h3><p>text</p>")
        assert [(b.text, b.heading) for b in page.blocks] == [
            ("Title", "Title"),
            ("text", "Title"),
        ]
        wrapped = parse_html("<h2><span>Span title</h4><p>after</p>")
        assert wrapped.blocks[0].text == "Span title"
        assert wrapped.blocks[1].heading == "Span title"

    def test_a_heading_closed_by_its_parent(self):
        page = parse_html("<div><h2>Inner</div><p>after</p>")
        assert [(b.text, b.heading) for b in page.blocks] == [
            ("Inner", "Inner"),
            ("after", "Inner"),
        ]

    def test_a_new_heading_ends_an_unclosed_one(self):
        page = parse_html("<h1>One<h2>Two</h2><p>x</p>")
        assert [(b.text, b.heading) for b in page.blocks] == [
            ("One", "One"),
            ("Two", "One › Two"),
            ("x", "One › Two"),
        ]

    def test_an_unclosed_heading_ends_with_the_document(self):
        assert [b.text for b in parse_html("<p>x</p><h3>Last").blocks] == [
            "x",
            "Last",
        ]

    def test_an_empty_heading_resets_the_deeper_path(self):
        page = parse_html("<h1>A</h1><h2>B</h2><p>x</p><h2> </h2><p>y</p>")
        assert [(b.text, b.heading) for b in page.blocks if b.kind != "heading"] == [
            ("x", "A › B"),
            ("y", "A"),
        ]

    def test_empty_document(self):
        page = parse_html("")
        assert page == ParsedPage(title="", blocks=[], links=[])
        assert page.markdown == ""

    def test_markdown_rendering(self):
        page = parse_html(PAGE)
        assert page.markdown == (
            "# Welcome to Acme\n\n"
            "We make widgets and gadgets.\n\n"
            "## Products\n\n"
            "Our widgets come in three sizes.\n\n"
            "- Small\n- Medium\n- Large\n\n"
            "### Pricing\n\n"
            "Size | Price\nSmall | 10 EUR\n\n"
            "## Contact\n\n"
            "Write to us.\nWe answer within a day."
        )

    def test_markdown_of_items_quotes_and_code(self):
        page = ParsedPage(
            title="",
            blocks=[
                Block("one\ntwo", kind="item"),
                Block("said\nthis", kind="quote"),
                Block("x = 1", kind="code"),
                Block("a ```` b", kind="code"),
            ],
            links=[],
        )
        assert page.markdown == (
            "- one\n  two\n\n"
            "> said\n> this\n\n"
            "```\nx = 1\n```\n\n"
            "`````\na ```` b\n`````"
        )

    def test_the_markdown_chunks_like_the_blocks(self):
        page = parse_html(PAGE)
        from_blocks = chunk_blocks(page.blocks, title=page.title)
        from_markdown = chunk_markdown(page.markdown, title=page.title)
        assert [c.heading for c in from_markdown] == [c.heading for c in from_blocks]


class TestChunkBlocks:
    def test_blocks_are_grouped_by_heading_and_prefixed(self):
        chunks = chunk_blocks(parse_html(PAGE).blocks, title="Acme Shop")
        assert [c.heading for c in chunks] == [
            "Welcome to Acme",
            "Welcome to Acme › Products",
            "Welcome to Acme › Products › Pricing",
            "Welcome to Acme › Contact",
        ]
        assert chunks[1] == Chunk(
            text=(
                "Acme Shop › Welcome to Acme › Products\n\n"
                "Our widgets come in three sizes.\n\nSmall\n\nMedium\n\nLarge"
            ),
            heading="Welcome to Acme › Products",
            position=1,
        )
        assert [c.position for c in chunks] == [0, 1, 2, 3]

    def test_a_heading_path_that_begins_with_the_title_does_not_repeat_it(self):
        blocks = [Block("x", "Guide › Setup"), Block("y", "Guide")]
        chunks = chunk_blocks(blocks, title="Guide")
        assert [c.text for c in chunks] == ["Guide › Setup\n\nx", "Guide\n\ny"]

    def test_prefix_variants(self):
        blocks = [Block("body")]
        assert chunk_blocks(blocks)[0].text == "body"
        assert chunk_blocks(blocks, title=" T ")[0].text == "T\n\nbody"
        assert chunk_blocks([Block("b", "H")], title="T", prefix=False)[0].text == "b"

    def test_a_long_prefix_is_shortened_to_half_of_max_chars(self):
        (chunk,) = chunk_blocks([Block("body", "H" * 100)], title="T", max_chars=40)
        head, body = chunk.text.split("\n\n")
        assert len(head) == 18 and head.endswith("…")
        assert body == "body"
        # Too small a cap for any prefix at all.
        (tiny,) = chunk_blocks([Block("abc", "H")], title="T", max_chars=5)
        assert tiny.text == "abc"
        pieces = chunk_blocks([Block("abcdef", "H")], title="T", max_chars=7)
        assert [c.text for c in pieces] == ["…\n\nabcd", "…\n\nef"]

    def test_heading_blocks_and_empty_blocks_are_skipped(self):
        blocks = [Block("H", "H", "heading", 1), Block("  "), Block("x", "H")]
        assert [c.text for c in chunk_blocks(blocks)] == ["H\n\nx"]
        assert chunk_blocks([]) == []

    def test_small_blocks_pack_up_to_the_target(self):
        blocks = [Block(f"Item {i} with some words") for i in range(200)]
        chunks = chunk_blocks(blocks, target_chars=500)
        assert len(chunks) > 5
        assert all(450 < len(c.text) <= 500 for c in chunks[:-1])

    def test_a_target_above_the_cap_is_lowered_to_it(self):
        blocks = [Block("x" * 30) for _ in range(10)]
        chunks = chunk_blocks(blocks, target_chars=5000, max_chars=100)
        assert all(len(c.text) <= 100 for c in chunks)
        assert len(chunks) == 4

    def test_no_cap_keeps_whole_sections(self):
        blocks = [Block("a" * 3000, "A"), Block("b" * 3000, "A"), Block("c", "B")]
        chunks = chunk_blocks(blocks, target_chars=None, max_chars=None)
        assert [len(c.text) for c in chunks] == [3 + 6002, 4]
        packed = chunk_blocks(blocks, target_chars=1000, max_chars=None)
        assert [len(c.text) for c in packed] == [3003, 3003, 4]
        up_to_cap = chunk_blocks(blocks[:1], target_chars=None, max_chars=1000)
        assert all(len(c.text) <= 1000 for c in up_to_cap)

    def test_sizes_below_one_are_refused(self):
        with pytest.raises(ValueError, match="max_chars"):
            chunk_blocks([Block("x")], max_chars=0)
        with pytest.raises(ValueError, match="target_chars"):
            chunk_blocks([Block("x")], target_chars=0)

    def test_long_blocks_split_at_lines_first(self):
        text = "\n".join(f"line {i:02d} " + "x" * 30 for i in range(10))
        chunks = chunk_blocks([Block(text)], target_chars=100, max_chars=100)
        assert all(len(c.text) <= 100 for c in chunks)
        assert all(c.text.startswith("line ") for c in chunks)
        assert squeeze("".join(c.text for c in chunks)) == squeeze(text)

    def test_long_blocks_split_at_sentence_ends(self):
        text = "Sentence number one is here. " * 60
        chunks = chunk_blocks([Block(text)], target_chars=300, max_chars=400)
        assert all(len(c.text) <= 300 for c in chunks)
        assert all(c.text.endswith("here.") for c in chunks)

    def test_sentence_ends_after_quotes_and_in_cjk(self):
        text = 'He said "Stop." Then he left!) Why?! No… ' * 20
        chunks = chunk_blocks([Block(text)], max_chars=60)
        assert all(c.text.endswith(('."', "!)", "?!", "…")) for c in chunks)
        cjk = "これは文です。これも文です！三つ目の文？" * 30
        pieces = chunk_blocks([Block(cjk)], max_chars=100)
        assert all(len(c.text) <= 100 for c in pieces)
        assert all(c.text.endswith(("。", "！", "？")) for c in pieces)
        assert "".join(c.text for c in pieces) == cjk

    def test_a_decimal_point_is_not_a_sentence_end(self):
        text = ("Pi is 3.14 and e is 2.71 roughly " * 10).strip()
        chunks = chunk_blocks([Block(text)], max_chars=100)
        assert not any(c.text.startswith("14") for c in chunks)

    def test_sentences_longer_than_the_cap_split_between_words(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_blocks([Block(text)], target_chars=50, max_chars=60)
        assert all(len(c.text) <= 50 for c in chunks)
        assert " ".join(c.text for c in chunks) == text

    def test_a_word_longer_than_the_cap_is_cut(self):
        chunks = chunk_blocks([Block("x" * 500)], target_chars=200, max_chars=300)
        assert [c.text for c in chunks] == ["x" * 200, "x" * 200, "x" * 100]


class TestChunkMarkdown:
    MARKDOWN = (
        "Intro paragraph before any heading.\n\n"
        "# Getting started\n\n"
        "Install the package.\n\n"
        "## Details\n\n"
        "First detail paragraph.\n\n"
        "Second detail paragraph.\n\n"
        "# Reference\n\n"
        "Reference text.\n"
    )

    def test_the_heading_path_is_tracked(self):
        chunks = chunk_markdown(self.MARKDOWN, title="Kaval Docs")
        assert [c.text for c in chunks] == [
            "Kaval Docs\n\nIntro paragraph before any heading.",
            "Kaval Docs › Getting started\n\nInstall the package.",
            "Kaval Docs › Getting started › Details\n\n"
            "First detail paragraph.\n\nSecond detail paragraph.",
            "Kaval Docs › Reference\n\nReference text.",
        ]
        assert chunks[2].heading == "Getting started › Details"

    def test_without_title_or_headings(self):
        (chunk,) = chunk_markdown("Just a paragraph.")
        assert (chunk.text, chunk.heading) == ("Just a paragraph.", "")
        assert chunk_markdown("") == []
        assert chunk_markdown("# Lonely heading") == []

    def test_long_sections_are_split(self):
        body = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(10))
        chunks = chunk_markdown(f"# Big\n\n{body}", title="T", max_chars=300)
        assert len(chunks) > 1
        assert all(len(c.text) <= 300 for c in chunks)
        assert all(c.text.startswith("T › Big\n\n") for c in chunks)

    def test_heading_syntax(self):
        markdown = (
            "#tag is text\n\n"
            "####### seven is text\n\n"
            "    # indented is text\n\n"
            "   ## Closed ##\n\nunder closed\n\n"
            "## C# ##\n\nunder csharp\n\n"
            "# \n\nunder empty\n\n"
            "## Only hashes ####\n\nx\n"
        )
        chunks = chunk_markdown(markdown)
        assert [c.heading for c in chunks] == [
            "",
            "Closed",
            "C#",
            "",
            "Only hashes",
        ]
        assert chunks[0].text == (
            "#tag is text\n\n####### seven is text\n\n# indented is text"
        )

    def test_fenced_code_is_one_block(self):
        markdown = (
            "# Setup\n\n"
            "```bash\n# not a heading\n\npip install kavalai\n```\n\n"
            "~~~~\n# still code\n~~~\n~~~~\n"
            "after\n"
        )
        (chunk,) = chunk_markdown(markdown)
        assert chunk.heading == "Setup"
        assert chunk.text == (
            "Setup\n\n"
            "```bash\n# not a heading\n\npip install kavalai\n```\n\n"
            "~~~~\n# still code\n~~~\n~~~~\n\n"
            "after"
        )

    def test_an_unclosed_fence_runs_to_the_end(self):
        (chunk,) = chunk_markdown("```\n# code\n\nmore")
        assert chunk.text == "```\n# code\n\nmore"


words = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters="¶"),
    max_size=40,
)
block_texts = st.lists(words, max_size=12).map(lambda parts: " ".join(parts)) | st.text(
    max_size=400
)
blocks = st.lists(
    st.builds(Block, text=block_texts, heading=st.sampled_from(["", "A", "A › B"])),
    max_size=12,
)


class TestProperties:
    @settings(max_examples=300, deadline=None)
    @given(
        blocks=blocks,
        target=st.integers(min_value=1, max_value=500),
        maximum=st.integers(min_value=1, max_value=500),
    )
    def test_no_text_is_lost(self, blocks, target, maximum):
        chunks = chunk_blocks(
            blocks, target_chars=target, max_chars=maximum, prefix=False
        )
        assert squeeze("".join(c.text for c in chunks)) == squeeze(
            "".join(b.text for b in blocks)
        )

    @settings(max_examples=300, deadline=None)
    @given(
        blocks=blocks,
        title=st.text(max_size=80),
        target=st.integers(min_value=1, max_value=500) | st.none(),
        maximum=st.integers(min_value=1, max_value=500),
    )
    def test_every_chunk_fits_the_cap(self, blocks, title, target, maximum):
        chunks = chunk_blocks(
            blocks, title=title, target_chars=target, max_chars=maximum
        )
        assert all(len(c.text) <= maximum for c in chunks)
        assert [c.position for c in chunks] == list(range(len(chunks)))
        assert all(c.text.strip() for c in chunks)

    @settings(max_examples=100, deadline=None)
    @given(blocks=blocks, maximum=st.integers(min_value=1, max_value=300))
    def test_output_is_deterministic(self, blocks, maximum):
        first = chunk_blocks(blocks, title="T", max_chars=maximum)
        assert chunk_blocks(list(blocks), title="T", max_chars=maximum) == first

    @settings(max_examples=200, deadline=None)
    @given(paragraphs=st.lists(words.filter(lambda w: w.strip()), max_size=10))
    def test_paragraph_text_survives_parsing(self, paragraphs):
        markup = "".join(f"<p>{htmllib.escape(p)}</p>" for p in paragraphs)
        page = parse_html(markup, strip_chars="")
        assert texts(page) == [" ".join(p.split()) for p in paragraphs]

    @settings(max_examples=200, deadline=None)
    @given(markup=st.text(max_size=300))
    def test_any_input_parses(self, markup):
        page = parse_html(markup, base_url="https://a.example/")
        assert all(block.text for block in page.blocks)
        assert all(len(c.text) <= MAX_CHARS for c in chunk_blocks(page.blocks))

    @settings(max_examples=200, deadline=None)
    @given(
        paragraphs=st.lists(
            words.filter(lambda w: w.strip() and w.lstrip()[0] not in "#`~"),
            max_size=10,
        ),
        maximum=st.integers(min_value=1, max_value=200),
    )
    def test_no_markdown_text_is_lost(self, paragraphs, maximum):
        markdown = "\n\n".join(paragraphs)
        chunks = chunk_markdown(markdown, max_chars=maximum, prefix=False)
        assert all(len(c.text) <= maximum for c in chunks)
        assert squeeze("".join(c.text for c in chunks)) == squeeze(markdown)


LINEAR_BOUND_SECONDS = 5.0
"""Each input below takes well under a second when handled in linear time and
minutes when handled in quadratic time, so the bound tolerates a slow machine
without letting a quadratic regression pass."""

N = 50_000


def elapsed(function, *args, **kwargs) -> float:
    start = time.perf_counter()
    function(*args, **kwargs)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    "markup",
    [
        pytest.param("<div>" * N + "x" + "</div>" * N, id="deep-nesting"),
        pytest.param("<div>" * N + "</span>" * N, id="stray-end-tags"),
        pytest.param("<div><b>" * N + "</div>" * N, id="implied-closes"),
        pytest.param("<nav>" * N + "x" + "</nav>" * N, id="nested-drops"),
        pytest.param("<main>" * N + "<p>x" * N, id="nested-content"),
        pytest.param("<h1>x" * N + "</h2>" * N, id="unclosed-headings"),
        pytest.param("<li>" * N + "text", id="unclosed-items"),
        pytest.param('<a href="' + "x" * 50 * N + '">t</a>', id="huge-attribute"),
        pytest.param("<a " + "b=c " * N + ">", id="many-attributes"),
        pytest.param("<" * 4 * N, id="angle-brackets"),
        pytest.param("<!--" + "a" * 20 * N, id="unclosed-comment"),
        pytest.param("<pre>" + "`" * 20 * N + "</pre>", id="backtick-run"),
        pytest.param("<tr><td>x</td>" * N, id="table-cells"),
    ],
)
def test_parsing_is_linear(markup):
    def parse_and_render():
        return parse_html(markup, base_url="https://a.example/").markdown

    assert elapsed(parse_and_render) < LINEAR_BOUND_SECONDS


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("x" * 40 * N, id="one-long-word"),
        pytest.param("word " * 8 * N, id="no-sentence-end"),
        pytest.param(". " * 8 * N, id="sentence-ends-only"),
        pytest.param("." * 20 * N, id="punctuation-run"),
        pytest.param("\n" * 20 * N + "x" * 5000, id="blank-lines"),
        pytest.param("。" * 20 * N, id="ideographic-stops"),
    ],
)
def test_chunking_is_linear(text):
    assert elapsed(chunk_blocks, [Block(text, "H")], title="T") < LINEAR_BOUND_SECONDS


@pytest.mark.parametrize(
    "markdown",
    [
        pytest.param("#" * 40 * N, id="hash-run"),
        pytest.param("# h\n" * N, id="many-headings"),
        pytest.param("```\n" * N, id="many-fences"),
        pytest.param("`" * 40 * N, id="backtick-run"),
        pytest.param("line\n" * 4 * N, id="one-long-paragraph"),
    ],
)
def test_markdown_chunking_is_linear(markdown):
    assert elapsed(chunk_markdown, markdown) < LINEAR_BOUND_SECONDS
