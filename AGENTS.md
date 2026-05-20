# AGENTS.md

## Project

`uv`-managed Python workspace with two member packages: `cody` (`cody/`) and
`duncan` (`duncan/`). At their core both should only use Python stdlib.

`lib` contains single file modules that maybe vendored. Avoid modifying where possible.

## Workflow

- `uv sync` — install all workspace members
- `uv run cody <command>` — run the coding agent
- `uv run duncan <command>` — run the TTRPG oracle
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
- Use sqlite3 for a database if required. Remember that sqlite3 supports full text search (bm25)

## Testing

Focused tests for core behavior and edge cases. Run via `uv run`. Tests live in
`tests/` under the member package or workspace root.

## Decision rule

Prefer simpler concrete implementations over abstract designs.
