# AGENTS.md

This file describes conventions and constraints for any AI agent working in this repository.

## Project Overview

**kitnega** is a minimal agentic coding harness written in pure Python stdlib.
It connects to OpenAI-compatible API endpoints (LM Studio, Ollama, etc.) and provides
a shell-based agent loop with human-in-the-loop approvals.

## Architecture

```
src/kitnega/
├── __init__.py    # Package metadata
├── __main__.py    # `python -m kitnega` entry point
├── agent.py       # Core agentic loop + API communication
├── cli.py         # CLI (REPL, one-shot, pipe mode, sessions)
├── config.py      # Configuration (providers, models, env vars)
└── tools.py       # Tool definitions + executors
```

### Design Principles

1. **Zero dependencies** — Python stdlib only (urllib, subprocess, json, etc.)
2. **Unix philosophy** — pipe-able, composable, low ceremony
3. **OpenAI-compatible** — works with any `/chat/completions` endpoint
4. **Human-in-the-loop** — tool approvals by default, opt-out with `--approve-all`

## Code Style

- **Formatter**: ruff (line length 120)
- **Imports**: ruff with I001 (sorted imports)
- **Type hints**: use type hints throughout; prefer `list[str]` over `List[str]`
- **Docstrings**: one-line for simple functions, multi-line for complex ones
- **No external deps** in `src/kitnega/` — stdlib only

## Running

```bash
# Interactive REPL
uv run python -m kitnega

# One-shot
uv run python -m kitnega "fix the tests"

# Pipe mode
echo "explain the codebase" | uv run python -m kitnega

# Continue last session
uv run python -m kitnega -c

# Pick a session
uv run python -m kitnega -s
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `KITNEGA_BASE_URL` | API base URL | `http://localhost:1234/v1` |
| `KITNEGA_MODEL` | Model ID | `qwen3-coder-next` |
| `KITNEGA_API_KEY` | API key (optional) | (empty) |
| `KITNEGA_MAX_STEPS` | Max tool call steps | `200` |
| `KITNEGA_TIMEOUT` | Request timeout (seconds) | `120` |
| `KITNEGA_APPROVE` | Auto-approve mode | (off) |

### Config File

Place `kitnega.json` or `.kitnega.json` in the project root, or
`~/.kitnega/models.json` globally. Format:

```json
{
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "unset",
      "models": [
        {"id": "qwen3-coder-next", "contextWindow": 100000, "maxTokens": 25000}
      ]
    }
  }
}
```

## Tools Available to the Agent

- **execute_shell** — Run shell commands (primary tool)
- **read_file** — Read file contents with optional line range
- **write_file** — Write content to a file (creates dirs)
- **list_dir** — List directory contents (optional recursive)

## Adding New Tools

1. Add the tool schema to `src/kitnega/tools.py` (as a `TOOL_*` dict)
2. Add the executor function to `src/kitnega/tools.py`
3. Register in `_TOOL_EXECUTORS` and `ALL_TOOLS`
4. Update the system prompt in `agent.py` to mention the new tool

## Testing

```bash
uv run pytest tests/ -v
```

## Dev Workflow

1. Make changes
2. Run `uv run ruff check src/` and `uv run ruff format src/`
3. Run `uv run pytest tests/`
4. Commit with descriptive message
