# AGENTS.md

## Project

`uv`-managed Python workspace with one member package: `cody` (`cody/`).
At its core it should only use Python stdlib.

## Workflow

- `uv sync` — install all workspace members
- `uv run cody <command>` — run the coding agent
- `uv run <tool>` — run tests, linting, scripts within the workspace

## Change discipline

Keep changes small and focused. Make one logical change at a time.
This prevents context overload for both humans and automated tools.
Do not refactor unrelated code while working on a task — that can wait.

## Coding rules

- Keep files under 150 lines, functions small and focused
- Prefer explicit over clever; use boring, consistent names
- Leave `__init__.py` empty
- Add comments only when intent isn't clear from the code

## Testing

Focused tests for core behavior and edge cases. Run via `uv run`.
Tests live in `tests/` under the member package or workspace root.

## Decision rule

Prefer simpler concrete implementations over abstract designs.
