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
uv run cody

# One-shot
uv run cody "fix the tests"

# Pipe mode
echo "explain the codebase" | uv run cody

# Continue last session
uv run cody -c

# Auto-approve (YOLO mode) — also enables parallel tool execution
CODY_APPROVE=all uv run cody "run tests and fix failures"
```

## Configuration

Set via environment variables:

```bash
export CODY_API="http://localhost:1234/v1/responses"
export CODY_MODEL="qwen3-coder-next"
export CODY_API_KEY="your-key"  # optional
export CODY_STREAM="1"          # enable SSE streaming for real-time output (default: 1)
export CODY_MAX_STEPS="200"     # max tool call iterations (default: 200)
```

## Workspaces

This repository uses [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/). The main package is `cody` (`cody/`).

```bash
uv sync              # install all workspace members
```

## Features

- **Zero dependencies** — pure Python stdlib (urllib, subprocess, json)
- **OpenAI-compatible** — works with LM Studio, Ollama, any Responses API endpoint
- **SSE streaming** — real-time text output via server-sent events
- **Parallel tool execution** — runs independent tools concurrently in auto-approve mode
- **.gitignore-aware** — respects project gitignore rules when searching files
- **Cached context walks** — discovers AGENTS.md/README.md once per session, not every turn
- **In-memory session cache** — defers disk writes, avoids redundant I/O
- **Pipe-able** — `echo "prompt" | cody` for scripting
- **Human-in-the-loop** — approve each write/edit/bash call, read-only tools skip approval
- **Session persistence** — continue conversations with `-c` or pick with `-s`

## Built-in Tools

| Tool | Approval | Description |
|---|---|---|
| `read` | ❌ skip | Read file contents with optional line range |
| `write` | ✅ required | Create or overwrite a file |
| `edit` | ✅ required | Find/replace patch in a file |
| `bash` | ✅ required | Run shell commands |
| `grep` | ❌ skip | Search file contents with regex |
| `find` | ❌ skip | Find files by glob pattern |
| `ls` | ❌ skip | List directory contents |

## REPL Commands

| Command | Description |
|---|---|
| `:q` / `:quit` | Exit |
| `:reset` | Reset conversation |
| `:load` | List recent sessions |
| `:load <id>` | Continue a specific session |
