---
title: kitnega — Agenti(k) Backwards
---

# kitnega

> _agenti(k)_ backwards — a monorepo of stdlib-only Python tools.

| Package                      | Role                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| [Cody](/kitnega/cody/)       | Agentic coding harness — LLM-powered coding agent                                   |
| [Duncan](/kitnega/duncan/)   | TTRPG procedural oracle — dice-driven random generation                             |
| [Carly](/kitnega/carly/)     | Procedural map generator — diamond-square + Voronoi                                 |
| [Merlin](/kitnega/merlin/)   | Random-split forests (ExtraTrees, IsolationForest, Mondrian) with built-in TreeSHAP |
| [Kenneth](/kitnega/kenneth/) | Kneser-Ney language model with NLTK alignment, PyTorch/ONNX export                  |
| [Klaus](/kitnega/klaus/)     | Minimal Slack clone — Bottle + htmx + sqlite3 chat server                           |
| [Raleigh](/kitnega/raleigh/) | Static site generator from markdown with front-matter                               |

## Workspace

This repository uses
[uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

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

All config is stored in JSON files under `~/.kitnega/` — the directory is
created on first use. There are no environment variables.

### Shared settings (`config.json`)

Cody and lib read from a single shared config file:

```json
{
  "api": "http://localhost:1234/v1/responses",
  "model": "qwen3.6-35b-a3b",
  "api_key": "",
  "stream": true,
  "max_steps": 20,
  "approve_all": false,
  "read_limit": 30,
  "turn_cap": 100,
  "bash_mode": "auto",
  "bash_allow": []
}
```

### Tool-specific files

| Path                       | Tool  | Purpose              |
| -------------------------- | ----- | -------------------- |
| `~/.kitnega/sessions.json` | Cody  | Session persistence  |
| `~/.kitnega/klaus.db`      | Klaus | SQLite chat database |
| `~/.kitnega/klaus_secret`  | Klaus | HMAC signing secret  |

## Inspired By

- [pnegahdar/nano](https://github.com/pnegahdar/nano) — minimalist agentic
  coding workflow ideas
- [badlogic/pi-mono](https://github.com/badlogic/pi-mono) — terminal-focused
  agent harness design
