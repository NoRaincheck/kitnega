---
title: Cody — Agentic Coding Harness
---

# Cody

A minimal agentic coding harness using only Python stdlib.

```python
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

# List sessions
uv run cody -s

# Auto-approve (YOLO mode) — also enables parallel tool execution
# Set "approve_all": true in ~/.kitnega/config.json
```

## Configuration

All settings live in `~/.kitnega/config.json`:

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

## Features

- **Zero dependencies** — pure Python stdlib (urllib, subprocess, json)
- **OpenAI-compatible** — works with LM Studio, Ollama, any Responses API
  endpoint
- **SSE streaming** — real-time text output via server-sent events
- **Parallel tool execution** — runs independent tools concurrently in
  auto-approve mode
- **.gitignore-aware** — respects project gitignore rules when searching files
- **Cached context walks** — discovers AGENTS.md/README.md once per session, not
  every turn
- **In-memory session cache** — defers disk writes, avoids redundant I/O
- **Pipe-able** — `echo "prompt" | cody` for scripting
- **Human-in-the-loop** — approve each write/edit/bash call, read-only tools
  skip approval
- **Session persistence** — continue conversations with `-c` or pick with `-s`

## Built-in Tools

| Tool    | Approval    | Description                                 |
| ------- | ----------- | ------------------------------------------- |
| `read`  | ❌ skip     | Read file contents with optional line range |
| `write` | ✅ required | Create or overwrite a file                  |
| `edit`  | ✅ required | Find/replace patch in a file                |
| `bash`  | ✅ required | Run shell commands                          |
| `grep`  | ❌ skip     | Search file contents with regex             |
| `find`  | ❌ skip     | Find files by glob pattern                  |
| `ls`    | ❌ skip     | List directory contents                     |

## REPL Commands

| Command        | Description                 |
| -------------- | --------------------------- |
| `:q` / `:quit` | Exit                        |
| `:reset`       | Reset conversation          |
| `:load`        | List recent sessions        |
| `:load <id>`   | Continue a specific session |
