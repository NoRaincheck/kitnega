# AGENTS.md

This file describes conventions and constraints for any AI agent working in this repository.

## Project Overview

**kitnega / nano** is a minimal agentic coding harness — one Python stdlib script (`nano.py`, ~200 LOC). It connects to OpenAI-compatible `/responses` endpoints (LM Studio, Ollama) and provides an interactive shell-agent loop with human-in-the-loop approvals. Also serves as its own development environment via `nano -c`.

## Architecture

```
./
└── nano.py    # Everything: agent loop, tool executor, CLI, session store (~250 LOC)
```

### Design Principles

1. **Single file** — zero structure, stdlib only (urllib, subprocess, json, threading, etc.)
2. **Unix philosophy** — pipe-able, composable, low ceremony
3. **OpenAI-compatible** — works with any `/responses` endpoint
4. **Human-in-the-loop** — tool approvals by default (`[y]/[n]`, opt-out with `[a] Approve All`)

## Running

```bash
# Interactive REPL (no arguments)
python nano.py                          # or: ./nano.py

# One-shot command
python nano.py "explain the codebase"   # prints answer, exits

# Continue last session in this directory
python nano.py -c                       # resumes latest prompt

# List sessions to resume manually
python nano.py -s                       # shows recent sessions

# Pipe mode (REPL stdin)
echo "fix my bugs" | python nano.py
```

### Session Commands (inside REPL)

| Command  | Action                              |
|----------|-------------------------------------|
| `:q` / `:quit`  | Exit repl                  |
| `:reset`         | Clear conversation history   |
| `:load <id>`     | Resume a past session by ID prefix |

Sessions are stored in `~/.nano_sessions.json`, scoped to each working directory (last 50 entries).

## Environment Variables

| Variable          | Description              | Default                           |
|-------------------|--------------------------|-----------------------------------|
| `NANO_API`        | API base URL             | `http://localhost:1234/v1/responses` |
| `NANO_MODEL`      | Model ID                 | `qwen3.6-35b-a3b`                |
| `NANO_API_KEY`    | API key (optional)       | *(empty)*                         |
| `NANO_MAX_STEPS`  | Max tool call steps     | `200`                             |
| `NANO_APPROVE`    | Auto-approve (`all`)    | off                               |

### Session Store File

Sessions are persisted in `~/.nano_sessions.json`. Each entry records: response ID, first prompt label (truncated to 80 chars), working directory CWD, and timestamp. Only entries matching the current CWD are loaded; last 50 per dir are retained.

## Agent Capabilities

**System Identity**: "Nano" — general-purpose shell agent with **one tool**.

| Tool            | Description                    |
|-----------------|--------------------------------|
| `execute_shell` | Run a shell command (primary)  |

### Tool Schema (`execute_shell`)

```json
{
  "command":      {"type": "string"},        // required: the command to run
  "description":  {"type": "string"},        // required: why this is useful, 5-10 words
  "cwd":          {"type": ["string", "null"]},    // optional working directory (default cwd)
  "timeout":      {"type": "integer"},       // optional seconds (default 60)
  "env":          {"type": "object"}         // optional env var overrides
}
```

**Validation**: descriptions shorter or longer than 5–10 words are rejected with a clear error. Invalid JSON arguments produce `bad arguments` errors. The tool caps output at ~12 KB to stay within context budgets.

### Auto-Discovered Context Files (injected into system prompt)

The agent auto-discovers and injects paths for important docs:
- **Docs**: searches current dir + home for `{agents.md, readme.md}`
- **Skill files**: checks `.agents/skills/` + `~/.pi/agent/skills/` for `{skill.md}`

### Tool Execution Flow

1. Agent calls `execute_shell(command, description)` via function call
2. System validates: args parse as JSON ✓, description length 5–10 words ✓
3. If approved (user or auto-approve), runs via `subprocess.run`
4. Output piped back to API for next step; denied → red "denied by user" text

## Code Style Notes

This is a single file with pragmatic conventions:
- **Stdlib only** — no imports beyond urllib, subprocess, json, threading, time, sys, os, platform
- **Inline config constants** at top (API, MODEL, MAX_STEPS) for easy env override
- **ANSI color helper**: `_color(code, text)` guarded by TTY check (`_TTY`)
- **Spinner thread**: runs in background while API request is pending; stops on response
- **Session management**: simple list of dicts persisted to JSON (~50 entries per directory)
