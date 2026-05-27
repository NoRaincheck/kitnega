"""Tests for Raleigh — static site generator."""

import json
from datetime import date
from pathlib import Path

import pytest

from raleigh.__main__ import (
    Site,
    md_to_html,
    parse_date,
    parse_front_matter,
    slugify,
)

# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


class TestParseFrontMatter:
    def test_basic_front_matter(self):
        text = "---\ntitle: Hello World\ndate: 2026-05-26\n---\nBody here"
        meta, body = parse_front_matter(text)
        assert meta is not None
        assert meta["title"] == "Hello World"
        assert meta["date"] == "2026-05-26"
        assert body.strip() == "Body here"

    def test_json_tags(self):
        text = '---\ntitle: Post\ndate: 2026-01\ntags: ["python", "docs"]\n---\ncontent'
        meta, _ = parse_front_matter(text)
        assert meta["title"] == "Post"
        assert meta["date"] == "2026-01"
        assert meta["tags"] == ["python", "docs"]

    def test_no_front_matter(self):
        text = "Just plain markdown\nno front-matter here"
        meta, body = parse_front_matter(text)
        assert meta is None
        assert body.strip() == text.strip()

    def test_case_insensitive_keys(self):
        text = "---\nTITLE: Upper Case\ndate: 2026-12-31\n---\nbody"
        meta, _ = parse_front_matter(text)
        assert meta["title"] == "Upper Case"

    def test_empty_front_matter(self):
        text = "---\n---\nBody after empty fm"
        meta, body = parse_front_matter(text)
        assert meta is None
        assert body.strip() == "Body after empty fm"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_full_date(self):
        assert parse_date("2026-05-26") == date(2026, 5, 26)

    def test_month_only(self):
        assert parse_date("2026-05") == date(2026, 5, 1)

    def test_invalid(self):
        assert parse_date("not-a-date") is None

    def test_already_date(self):
        d = date(2026, 3, 15)
        assert parse_date(d) == d


# ---------------------------------------------------------------------------
# Markdown to HTML conversion
# ---------------------------------------------------------------------------


class TestMdToHtml:
    def test_headings(self):
        md = "# H1\n## H2"
        html = md_to_html(md)
        assert "<h1>H1</h1>" in html
        assert "<h2>H2</h2>" in html

    def test_bold_and_italic(self):
        md = "**bold** and *italic*"
        html = md_to_html(md)
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_links(self):
        md = "[click](http://example.com)"
        html = md_to_html(md)
        assert '<a href="http://example.com">click</a>' in html

    def test_images(self):
        md = "![alt text](img.png)"
        html = md_to_html(md)
        assert '<img src="img.png" alt="alt text">' in html

    def test_code_block(self):
        md = "```\ncode here\n```"
        html = md_to_html(md)
        assert "<pre><code>code here\n</code></pre>" in html

    def test_inline_code(self):
        md = "Use `foo` function."
        html = md_to_html(md)
        assert "<code>foo</code>" in html

    def test_unordered_list(self):
        md = "- item one\n- item two"
        html = md_to_html(md)
        assert "<ul>" in html
        assert "<li>item one</li>" in html
        assert "<li>item two</li>" in html

    def test_ordered_list(self):
        md = "1. first\n2. second"
        html = md_to_html(md)
        assert "<ol>" in html
        assert "<li>first</li>" in html

    def test_blockquote(self):
        md = "> quoted text"
        html = md_to_html(md)
        assert "<blockquote>" in html

    def test_horizontal_rule(self):
        md = "---"
        html = md_to_html(md)
        assert "<hr>" in html

    def test_paragraphs(self):
        md = "First paragraph.\n\nSecond paragraph."
        html = md_to_html(md)
        assert "<p>First paragraph.</p>" in html
        assert "<p>Second paragraph.</p>" in html


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_simple(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        # apostrophe and punctuation become dashes, collapsed
        result = slugify("It's a test!")
        assert result.startswith("it")
        assert "test" in result
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# End-to-end site build
# ---------------------------------------------------------------------------


class TestSiteBuild:
    @pytest.fixture()
    def tmp_site(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a minimal source tree and return (source_dir, output_dir)."""
        src = tmp_path / "source"
        out = tmp_path / "_site"

        # Post with front-matter
        post_md = src / "posts" / "hello-world.md"
        post_md.parent.mkdir(parents=True)
        post_md.write_text(
            '---\ntitle: Hello World\ndate: 2026-05-26\ntags: ["intro", "test"]\n---\n\nThis is my **first** post.\n'
        )

        # Another post with YYYY-MM date
        post2 = src / "posts" / "early-post.md"
        post2.write_text("---\ntitle: Early Post\ndate: 2026-01\n---\n\nAn earlier post.\n")

        # Static page (no date → treated as page, not post)
        about = src / "about" / "index.md"
        about.parent.mkdir(parents=True)
        about.write_text("---\ntitle: About\n---\n\nAbout this site.\n")

        # Homepage markdown (no date → page)
        index_md = src / "index.md"
        index_md.write_text("Welcome to the site.")

        # Config: show recent posts on homepage (tests expect post links there)
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"home": "recent"}))

        return src, out

    def test_build_creates_output(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        site = Site(source_dir=src, output_dir=out)
        count = site.build()
        assert count > 0

    def test_index_page_created(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        Site(source_dir=src, output_dir=out).build()
        index = out / "index.html"
        assert index.exists()
        content = index.read_text()
        assert "Hello World" in content
        assert "Early Post" in content

    def test_post_page_created(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        Site(source_dir=src, output_dir=out).build()
        post = out / "posts" / "hello-world.html"
        assert post.exists()
        content = post.read_text()
        assert "<h1>Hello World</h1>" in content
        assert "<strong>first</strong>" in content

    def test_static_page_created(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        Site(source_dir=src, output_dir=out).build()
        about = out / "about" / "index.html"
        assert about.exists()
        content = about.read_text()
        assert "<title>About</title>" in content
        assert "About this site." in content

    def test_tags_page_created(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        Site(source_dir=src, output_dir=out).build()
        tag_page = out / "tags" / "intro.html"
        assert tag_page.exists()
        content = tag_page.read_text()
        assert "Hello World" in content

    def test_posts_sorted_by_date(self, tmp_site: tuple[Path, Path]):
        src, out = tmp_site
        Site(source_dir=src, output_dir=out).build()
        index = out / "index.html"
        # May 2026 post should appear before Jan 2026 post
        content = index.read_text()
        hello_pos = content.index("Hello World")
        early_pos = content.index("Early Post")
        assert hello_pos < early_pos

    def test_missing_source_dir(self, tmp_path: Path):
        out = tmp_path / "_site"
        site = Site(source_dir=tmp_path / "nonexistent", output_dir=out)
        count = site.build()
        assert count == 0
