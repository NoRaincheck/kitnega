---
title: Raleigh — Static Site Generator
---

# Raleigh

A minimal static site generator from markdown with front-matter. Pure Python
stdlib, zero dependencies.

## Usage

```bash
uv run raleigh                        # build source/ → _site/
uv run raleigh -o docs/_site          # custom output directory
uv run raleigh --title "My Blog"      # override site title
```

## Directory Layout

```
source/
  index.md                  Homepage
  posts/2026-05-26-post.md  Blog posts (date in front-matter)
  about/index.md            Sub-pages
  assets/                   Static files (copied verbatim)
```

## Front-Matter

Markdown files can include YAML-like front-matter delimited by `---`:

```markdown
---
title: My First Post
date: 2026-05-27
tags: [python, static-sites]
---

Post body goes here.
```

Keys are parsed as JSON values when possible; otherwise treated as strings.

## Config (`config.json`)

Place a `config.json` alongside the source directory for site-wide settings:

```json
{
    "site_title": "My Site",
    "footer": "Built with Raleigh",
    "nav": [{"name": "Home", "href": "/"}],
    "home": "recent",
    "blog_index": "blog.html",
    "date_format": "%B %Y",
    "date_format_full": "%B %d, %Y"
}
```

| Key              | Default        | Description                              |
| ---------------- | -------------- | ---------------------------------------- |
| `site_title`     | _(none)_       | Site title used in HTML head             |
| `footer`         | _(none)_       | Footer text rendered on every page       |
| `nav`            | `[]`           | Navigation bar entries (name + href)     |
| `home`           | `"recent"`     | Homepage mode: `"page"`, `"blog"`, or `"recent"` |
| `blog_index`     | `"blog.html"`  | Blog listing page filename               |
| `date_format`    | `%B %Y`        | Short date format (e.g., "May 2026")     |
| `date_format_full` | `%B %d, %Y`  | Full date format (e.g., "May 27, 2026")  |

## CLI Options

| Flag              | Default    | Description                           |
| ----------------- | ---------- | ------------------------------------- |
| `source`          | `source/`  | Input directory                       |
| `-o, --output`    | `_site/`   | Output directory                      |
| `--title`         | _(none)_   | Override site title from config.json  |
