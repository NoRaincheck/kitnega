---
title: kitnega — Agenti(k) Backwards
---

# kitnega

> *agenti(k)* backwards — a monorepo of stdlib-only Python tools.

| Package    | Role                                                                                |
| ---------- | ----------------------------------------------------------------------------------- |
| [Cody](/cody/)   | Agentic coding harness — LLM-powered coding agent                                   |
| [Duncan](/duncan/) | TTRPG procedural oracle — dice-driven random generation                             |
| [Carly](/carly/)  | Procedural map generator — diamond-square + Voronoi                                 |
| [Merlin](/merlin/) | Random-split forests (ExtraTrees, IsolationForest, Mondrian) with built-in TreeSHAP |
| [Klaus](/klaus/)  | Minimal Slack clone — Bottle + htmx + sqlite3 chat server                           |
| [Raleigh](/raleigh/) | Static site generator from markdown with front-matter                               |

## Workspace

This repository uses [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

```bash
uv sync              # install all workspace members
uv run cody <cmd>    # run the coding agent
uv run duncan <cmd>  # run the TTRPG oracle
uv run carly <args>  # run the map generator
uv run merlin <cmd>  # run ML experiments
uv run klaus <cmd>   # run the chat server (serve, init-db, create-admin)
uv run raleigh       # build static site from markdown source
uv run pytest        # run tests
uv run ruff check    # lint
```

## Configuration

All config is generated from `~/.kitnega/` — the directory is created on first use by each tool.

| Variable       | Default                              | Tool     | Purpose                  |
| -------------- | ------------------------------------ | -------- | ------------------------ |
| `KN_API`       | `http://localhost:1234/v1/responses` | Cody/lib | LLM API endpoint         |
| `KN_MODEL`     | `qwen3.6-35b-a3b`                    | Cody/lib | Model name               |
| `KN_API_KEY`   | _(empty)_                            | Cody/lib | API key for LLM          |
| `KN_STREAM`    | `1`                                  | Cody     | SSE streaming toggle     |
| `KN_MAX_STEPS` | `20`                                 | Cody     | Max tool call iterations |
| `KN_APPROVE`   | `all`                                | Cody     | Auto-approve mode        |

## Inspired By

- [pnegahdar/nano](https://github.com/pnegahdar/nano) — minimalist agentic coding workflow ideas
- [badlogic/pi-mono](https://github.com/badlogic/pi-mono) — terminal-focused agent harness design
