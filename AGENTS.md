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
- `python scripts/rollup.py --source-dir duncan/duncan --output out.py` —
  consolidate a module into a single pktpy-compatible script
- `python scripts/rollup.py --source-dir duncan/duncan --output out.py --strip-prefix TN. --strip-prefix TA.`
  — with namespace prefix stripping
- `uv sync --group pktpy` — install optional pktpy runtime
- `uv run pktpy out.py` — run a consolidated script via pocketpy

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

## pktpy compatibility

[pocketpy](https://github.com/pocketpy/pocketpy) (via `uv run pktpy`) is a
portable Python subset. The consolidated output of `scripts/rollup.py` targets
this runtime. Source packages stay CPython-only; all transforms happen at rollup
time.

### Unavailable stdlib modules (pktpy)

Only ~25 modules are available. Major omissions: `re`, `pathlib`, `itertools`,
`argparse`, `hashlib`, `subprocess`, `threading`, `glob`, `shutil`, `copy`,
`types`, `struct`, `ctypes`, `urllib`, `socket`, `sqlite3`.

Available: `random`, `math`, `json`, `os` (limited), `sys` (limited), `time`,
`collections`, `datetime`, `functools`, `enum`.

### Syntax limitations

- No generator expressions `(x for x in y)` — use list comprehensions
  `[x for x in y]` instead
- No `try/except else/finally` (only bare try/except)
- No `__del__`, `__slots__`, `__iadd__` / `__imul__` (in-place magic)
- No `a, *b, c = x` star unpacking (only `a, *b = x`)
- No multiple inheritance

### Rollup transforms

`scripts/rollup.py` applies these to the output:

| Source issue                                 | Transform                                            |
| -------------------------------------------- | ---------------------------------------------------- |
| `import hashlib`                             | Replaced with `_hash_str()` (simple polynomial hash) |
| `import argparse`                            | Replaced with manual `sys.argv` parsing              |
| `from lib.llm` / relative imports            | Stripped (feature not available under pktpy)         |
| Generator expressions                        | Converted to list comprehensions                     |
| `sys.stdout.write()`                         | Replace with `print()`                               |
| `def main()` and `if __name__ == "__main__"` | Stripped (avoids need for argparse conversion)       |

### Graceful fallback pattern

When code must conditionally use unavailable stdlib, prefer:

```python
try:
    from some_module import thing
except ImportError:
    thing = None
```

This is transparent under CPython (no behavior change) and degrades gracefully
under pktpy.

## Configuration

All config is stored in JSON files under `~/.kitnega/` — the directory is
created on first use. There are no environment variables.

| Path                       | Tool     | Purpose                                                  |
| -------------------------- | -------- | -------------------------------------------------------- |
| `~/.kitnega/config.json`   | Cody/lib | Shared settings (API, model, streaming, bash mode, etc.) |
| `~/.kitnega/sessions.json` | Cody     | Session persistence                                      |
| `~/.kitnega/klaus.db`      | Klaus    | SQLite chat database                                     |
| `~/.kitnega/klaus_secret`  | Klaus    | HMAC signing secret                                      |

### `config.json`

```
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

## Documentation

Never embed file-tree listings in README.md files. Describe layout at a high
level instead.

## Decision rule

Prefer simpler concrete implementations over abstract designs.
