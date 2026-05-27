# AGENTS.md

## Project

`uv`-managed Python workspace with seven member packages: `cody` (`cody/`),
`duncan` (`duncan/`), `carly` (`carly/`), `merlin` (`merlin/`), `klaus`
(`klaus/`), `raleigh` (`raleigh/`), and `lib` (`lib/`). At their core, all
packages use only Python stdlib.

`lib` contains single-file modules that may be vendored. Avoid modifying where
possible.

## Workflow

- `uv sync` — install all workspace members
- `uv run cody <command>` — run the coding agent
- `uv run duncan <command>` — run the TTRPG oracle
- `uv run carly <args>` — run the map generator
- `uv run merlin <cmd>` — run ML experiments
- `uv run klaus <cmd>` — run the chat server (serve, init-db, create-admin)
- `uv run raleigh` — build static site from markdown source
- `uv run pytest` — run tests
- `uv run ruff check` — lint

## Change discipline

Keep changes small and focused. Make one logical change at a time. This prevents
context overload for both humans and automated tools. Do not refactor unrelated
code while working on a task — that can wait.

## Coding rules

- Keep files between 150-500 LoC, functions small and focused
- Prefer explicit over clever; use boring, consistent names
- Leave `__init__.py` empty
- Add comments only when intent isn't clear from the code
- Cache filesystem walks and external reads per-process where possible
- Use `.gitignore` awareness in file-search tools (grep, find)
- Use sqlite3 for a database if required. Remember that sqlite3 supports full
  text search (bm25)

## Testing

Focused tests for core behavior and edge cases. Run via `uv run`. Tests live in
`tests/` under the member package or workspace root.

## Configuration

All config is generated from `~/.kitnega/` — the directory is created on first
use by each tool. Files:

| Path                       | Tool  | Purpose              |
| -------------------------- | ----- | -------------------- |
| `~/.kitnega/sessions.json` | Cody  | Session persistence  |
| `~/.kitnega/klaus.db`      | Klaus | SQLite chat database |
| `~/.kitnega/klaus_secret`  | Klaus | HMAC signing secret  |

Environment variables use the `KN_` prefix:

| Variable       | Default                              | Tool     | Purpose                  |
| -------------- | ------------------------------------ | -------- | ------------------------ |
| `KN_API`       | `http://localhost:1234/v1/responses` | Cody/lib | LLM API endpoint         |
| `KN_MODEL`     | `qwen3.6-35b-a3b`                    | Cody/lib | Model name               |
| `KN_API_KEY`   | _(empty)_                            | Cody/lib | API key for LLM          |
| `KN_STREAM`    | `1`                                  | Cody     | SSE streaming toggle     |
| `KN_MAX_STEPS` | `20`                                 | Cody     | Max tool call iterations |
| `KN_APPROVE`   | `all`                                | Cody     | Auto-approve mode        |

## Documentation

Never embed file-tree listings in README.md files. Describe layout at a high
level instead.

## Decision rule

Prefer simpler concrete implementations over abstract designs.
