"""Raleigh — minimal static site generator from markdown with front-matter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


def parse_front_matter(text: str) -> tuple[dict | None, str]:
    """Extract front-matter and body from markdown text.

    Front-matter is delimited by ``---`` at the top of the file.  Keys are
    ``str : value`` on single lines; values must be JSON-parsable.

    Returns ``(meta_dict, body_text)`` — meta is None if no front-matter found.
    """
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return None, text

    end = stripped.index("---", 3)
    fm_block = stripped[4:end].strip()
    body = stripped[end + 3 :].lstrip("\n")

    meta: dict = {}
    for line in fm_block.splitlines():
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        key = line[:colon_idx].strip().lower()
        val_raw = line[colon_idx + 1 :].strip()
        try:
            meta[key] = json.loads(val_raw)
        except json.JSONDecodeError, ValueError:
            meta[key] = val_raw

    return meta or None, body


# ---------------------------------------------------------------------------
# Minimal markdown → HTML converter (no external deps)
# ---------------------------------------------------------------------------


def _inline(text: str) -> str:
    """Process inline markup."""
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_html(md_text: str) -> str:
    """Convert markdown text to HTML.

    Handles headings, bold/italic, code, links, images, lists, blockquotes,
    horizontal rules, and paragraphs.
    """
    lines = md_text.splitlines()
    html_parts: list[str] = []
    i = 0
    n = len(lines)

    def _flush(block_lines: list[str]) -> None:
        rendered = _render_block(block_lines)
        if rendered:
            html_parts.append(rendered)

    current_block: list[str] = []

    while i < n:
        line = lines[i]

        # Blank line ends a block
        if not line.strip():
            _flush(current_block)
            current_block = []
            i += 1
            continue

        # Code fence
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            _flush(current_block)
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                escaped = lines[i].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                code_lines.append(escaped)
                i += 1
            lang = m.group(1) or ""
            tag = "pre" if not lang else f'pre class="language-{lang}"'
            html_parts.append(f"<{tag}><code>{''.join(code_lines)}\n</code></{tag}>")
            current_block = []
            i += 1
            continue

        # Heading
        hm = re.match(r"^(#{1,6})\s+(.*)", line)
        if hm:
            _flush(current_block)
            level = len(hm.group(1))
            content = _inline(hm.group(2).strip())
            html_parts.append(f"<h{level}>{content}</h{level}>")
            current_block = []
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line):
            _flush(current_block)
            html_parts.append("<hr>")
            current_block = []
            i += 1
            continue

        # Blockquote
        if line.strip().startswith(">"):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                bq_text = re.sub(r"^\s*>\s?", "", lines[i])
                bq_lines.append(bq_text)
                i += 1
            _flush(current_block)
            inner = md_to_html("\n".join(bq_lines))
            html_parts.append(f"<blockquote>{inner}</blockquote>")
            continue

        # Unordered list
        if re.match(r"^\s*[-*+]\s+", line):
            _flush(current_block)
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                content = re.sub(r"^\s*[-*+]\s+", "", lines[i])
                items.append(f"<li>{_inline(content)}</li>")
                i += 1
            html_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        # Ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            _flush(current_block)
            items: list[str] = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                content = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                items.append(f"<li>{_inline(content)}</li>")
                i += 1
            html_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        current_block.append(line)
        i += 1

    _flush(current_block)
    return "\n".join(html_parts)


def _render_block(block: list[str]) -> str | None:
    """Render a single paragraph block."""
    text = " ".join(block)
    if not text.strip():
        return None
    return f"<p>{_inline(text)}</p>"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_date(val: str | datetime | date) -> date | None:
    """Parse YYYY-MM or YYYY-MM-DD into a date object."""
    if isinstance(val, (datetime, date)):
        return val.date() if hasattr(val, "date") else val
    if not isinstance(val, str):
        return None
    try:
        if len(val) == 7 and "-" in val:
            return datetime.strptime(val, "%Y-%m").date()
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Site generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
<header><a href="/">← {site_title}</a></header>
<main>
{content}
</main>
<footer><small>Built with Raleigh</small></footer>
</body>
</html>"""


def slugify(text: str) -> str:
    """Create a URL-friendly slug from text."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return s.strip("-").replace("--", "-") or "untitled"


class Site:
    """Minimal static site generator.

    Directory conventions (Jekyll / Hugo inspired):

        source/            root input directory
          index.md         homepage
          posts/           blog post collection
            2026-05-26-title.md
          about/
            index.md
          assets/          static files copied verbatim
    """

    def __init__(
        self,
        source_dir: str | Path = "source",
        output_dir: str | Path = "_site",
        site_title: str = "My Site",
    ) -> None:
        self.source = Path(source_dir).resolve()
        self.output = Path(output_dir).resolve()
        self.site_title = site_title

    def build(self) -> int:
        """Generate the entire site. Returns number of pages written."""
        if not self.source.is_dir():
            print(f"Error: source directory {self.source} does not exist", file=sys.stderr)
            return 0

        # Collect all markdown files with front-matter
        posts: list[tuple[dict, Path]] = []
        pages: list[tuple[dict | None, str, Path]] = []

        for md_file in sorted(self.source.rglob("*.md")):
            raw = md_file.read_text(encoding="utf-8")
            meta, body = parse_front_matter(raw)
            rel = md_file.relative_to(self.source)

            if "date" in (meta or {}):
                posts.append((meta or {}, md_file))
            else:
                pages.append((meta, body, rel))

        # Sort posts by date descending
        posts.sort(
            key=lambda p: parse_date(p[0].get("date", "")) or date.min,
            reverse=True,
        )

        self.output.mkdir(parents=True, exist_ok=True)

        # Copy static assets
        assets = self.source / "assets"
        if assets.is_dir():
            for asset in assets.rglob("*"):
                dest = self.output / "assets" / asset.relative_to(assets)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if asset.is_file():
                    dest.write_bytes(asset.read_bytes())

        count = 0

        # Generate individual pages (no date in front-matter)
        for meta, body, rel in pages:
            html_body = md_to_html(body)
            title = (meta or {}).get("title", str(rel))
            out_path = self.output / rel.with_suffix(".html")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if slugify(title):
                out_path.write_text(
                    HTML_TEMPLATE.format(
                        site_title=self.site_title,
                        title=title,
                        content=html_body,
                    ),
                    encoding="utf-8",
                )
                count += 1

        # Generate index page listing posts
        post_links = ""
        for meta, _path in posts:
            d = parse_date(meta.get("date", ""))
            date_str = d.strftime("%Y-%m") if d else ""
            title = meta.get("title", "Untitled")
            tag_list = ", ".join(str(t) for t in meta.get("tags", []))
            link_path = f"posts/{slugify(title)}.html"
            post_links += (
                f'<article><h2><a href="{link_path}">{title}</a></h2>'
                f"<p>{date_str}" + (f"  · <em>{tag_list}</em>" if tag_list else "") + "</p></article>\n"
            )

        index_html = HTML_TEMPLATE.format(
            site_title=self.site_title,
            title="Home",
            content=post_links or "<p>No posts yet.</p>",
        )
        (self.output / "index.html").write_text(index_html, encoding="utf-8")
        count += 1

        # Generate per-page HTML for each post
        for meta, md_file in posts:
            body = parse_front_matter(md_file.read_text(encoding="utf-8"))[1]
            html_body = md_to_html(body)
            title = meta.get("title", "Untitled")
            d = parse_date(meta.get("date", ""))
            date_str = d.strftime("%B %d, %Y") if d else ""
            tags = meta.get("tags", [])
            tag_list = ", ".join(f'<a href="tags/{slugify(str(t))}.html">{t}</a>' for t in tags)

            post_html = HTML_TEMPLATE.format(
                site_title=self.site_title,
                title=title,
                content=f'<h1>{title}</h1>\n<p class="date">{date_str}'
                + (f" · {tag_list}" if tag_list else "")
                + "</p>\n"
                + html_body,
            )
            out_path = self.output / "posts" / f"{slugify(title)}.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(post_html, encoding="utf-8")
            count += 1

        # Generate tag index pages
        tags: dict[str, list[tuple[dict, date]]] = {}
        for meta, _path in posts:
            for t in meta.get("tags", []):
                d = parse_date(meta.get("date", "")) or date.min
                tags.setdefault(str(t), []).append((meta, d))

        tag_dir = self.output / "tags"
        tag_dir.mkdir(parents=True, exist_ok=True)
        for t, items in sorted(tags.items()):
            items.sort(key=lambda x: x[1], reverse=True)
            link_items = "".join(
                f'<li><a href="posts/{slugify(str(m.get("title", "")))}.html">{m.get("title", "Untitled")}</a></li>'
                for m, _d in items
            )
            tag_html = HTML_TEMPLATE.format(
                site_title=self.site_title,
                title=f"Posts tagged '{t}'",
                content=f"<h1>Posts tagged '{t}'</h1>\n<ul>{link_items}</ul>",
            )
            (tag_dir / f"{slugify(t)}.html").write_text(tag_html, encoding="utf-8")

        print(f"Built {count} pages → {self.output}")
        return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raleigh",
        description="Minimal static site generator from markdown with front-matter.",
    )
    parser.add_argument("source", nargs="?", default="source", help="Source directory (default: source)")
    parser.add_argument("-o", "--output", default="_site", help="Output directory (default: _site)")
    parser.add_argument("--title", default="My Site", help="Site title")

    args = parser.parse_args(argv)
    site = Site(source_dir=args.source, output_dir=args.output, site_title=args.title)
    return 0 if site.build() > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
