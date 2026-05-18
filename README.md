# kitnega

A minimal agentic coding harness using only Python stdlib.

```
while not done:
    response = ask_llm(context)
    if response.wants_to_run_tool:
        if human_approves(tool_call):
            result = execute(tool_call)
            context.append(result)
    else:
        print(response)
        done = true
```

## Quick Start

```bash
# Interactive REPL
uv run python -m kitnega

# One-shot
uv run python -m kitnega "fix the tests"

# Pipe mode
echo "explain the codebase" | uv run python -m kitnega

# Continue last session
uv run python -m kitnega -c

# Auto-approve (YOLO mode)
KITNEGA_APPROVE=all uv run python -m kitnega "run tests and fix failures"
```

## Configuration

Set via environment variables or a `models.json`-style config file:

```bash
# Environment variables
export KITNEGA_BASE_URL="http://localhost:1234/v1"
export KITNEGA_MODEL="qwen3-coder-next"
export KITNEGA_API_KEY="your-key"  # optional, for endpoints that require it
```

Or create `~/.kitnega/models.json`:

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

## Features

- **Zero dependencies** — pure Python stdlib (urllib, subprocess, json)
- **OpenAI-compatible** — works with LM Studio, Ollama, any `/chat/completions` endpoint
- **Pipe-able** — `echo "prompt" | kitnega` for scripting
- **Human-in-the-loop** — approve each tool call, or `--approve-all` / `KITNEGA_APPROVE=all`
- **Session persistence** — continue conversations with `-c` or pick with `-s`
- **Auto context discovery** — finds AGENTS.md, README.md, CLAUDE.md, skill files

## Tools

| Tool | Description |
|---|---|
| `execute_shell` | Run shell commands (primary tool) |
| `read_file` | Read file contents with optional line range |
| `write_file` | Write content to a file |
| `list_dir` | List directory contents |

## REPL Commands

| Command | Description |
|---|---|
| `:q` / `:quit` | Exit |
| `:reset` | Reset conversation |
| `:h` / `:help` | Show help |
| `:session` | List recent sessions |
| `:continue N` | Continue session by index |
