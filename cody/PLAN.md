# PLAN: Small Local Model Support for Cody

Adapt small-coder's extension patterns into cody's Python architecture to help
small local models (Qwen 3.6-35B, etc.) produce reliable tool calls, avoid
common failure modes, and stay within context budgets.

## Reference: https://github.com/NoRaincheck/small-coder

Small-coder ships as a pi package with TypeScript extensions for small-model
harnessing (output parsing, quality monitoring, guards, checkpointing, skills).
We adapted the core patterns into cody's Python monolith — no separate extension
system needed, just hooks in the right places.

---

## Implementation Status: ✅ COMPLETE

### Phase 1: Core Quality & Safety ✅

#### 1. Output Parser (`cody/output_parser.py`)

After `respond()` returns, scans assistant text for embedded tool calls (fenced
`tool blocks,`json blocks, XML-style tags). Returns extracted calls for
logging/warning purposes. Small models often embed tool calls as code blocks
instead of using the native tool-call channel — this detects those patterns.

#### 2. Quality Monitor (`cody/quality_monitor.py`)

Per-turn assessment detecting:

- **Empty responses** → sends correction message back to model
- **Unknown/hallucinated tools** → sends available tools list
- **Repeated exact same call** → loop detection with auto-correction
- Corrections are injected as system messages, letting the model self-correct

#### 3. Write Guard (`cody/write_guard.py`)

In `_handle_write`: checks if file exists → refuses with Edit suggestion.
Normalizes bare paths like `/foo.md` into `<cwd>/foo.md`. Also validates edit
paths exist before processing.

#### 4. Read Guard (`cody/read_guard.py`)

In `_handle_read`: limits to first 30 lines by default (configurable via `read_limit`
in config). Appends "[TRIMMED: N more lines]" message with grep suggestion.

### Phase 2: Control & Safety ✅

#### 5. Turn Cap (`cody/turn_cap.py`)

Configurable `turn_cap` in config (default 100). In the run() loop, aborts with
clear message when limit exceeded. Prevents small models from looping endlessly.

#### 6. Permission Gate (`cody/permission_gate.py`)

Configurable whitelist of allowed bash command prefixes. Three modes:

- `auto` (default): silently block + notify
- `accept-all`: no gating
- `manual`: prompt user for each blocked command

Whitelist includes git subcommands, npm/pnpm/yarn, pip/cargo/go commands, file
operations (cp/mv/touch), search tools. Extra prefixes configurable via `bash_allow`
in config.

#### 7. Checkpoint (`cody/checkpoint.py`)

Before Write/Edit: copies file to `~/.kitnega/checkpoints/` with session-scoped
subdirectory. Best-effort — never fails the operation if checkpointing fails.

### Phase 3: Context & Guidance ✅

#### 8. Extra Tools (`cody/extra_tools.py`)

Added `glob` tool using fnmatch/glob with heavy-dir pruning (node_modules, .git,
**pycache**, etc.). More intuitive for small models than find+grep combo.

#### 9. Skills System (`cody/skills.py`)

- Loads `.kitnega_skills/` directory for markdown skill cards (YAML frontmatter)
- Selection: error-recovery > recency > intent prediction
- Supports `disable_model_invocation` flag to prevent circular tool calls
- Injected as `## Tool Usage Guidance` into system prompt

#### 10. Knowledge Injection (`cody/knowledge.py`)

- Loads `.kitnega_knowledge/` directory for algorithm cheat sheets
- Scores against user prompt via keyword + bigram matching
- Top 3 matches injected as `## Algorithm Reference` into system prompt

### Phase 4: Polish ✅

#### 11. Updated System Prompt (`cody/system_prompt.py`)

Added small-model-specific guidelines:

- "Write refuses on existing files — use Edit with exact old_string/new_string"
- "Read is trimmed to 30 lines by default — use Grep first for large files"
- "Bash commands have a 30s timeout unless overridden"

#### 12. Configuration (`lib/lib/config.py`, `tools.py`)

Shared JSON config at `~/.kitnega/config.json` with keys:

| Key           | Default | Purpose                          |
| ------------- | ------- | -------------------------------- |
| `turn_cap`    | `100`   | Max turns per run                |
| `bash_mode`   | `auto`  | Permission gate mode             |
| `bash_allow`  | `[]`    | Extra allow prefixes             |
| `read_limit`  | `30`    | Default read line limit          |

---

## Files Added/Modified

### New files in `cody/cody/`:

- `read_guard.py` — Line limiting for reads
- `write_guard.py` — Existing file refusal + path normalization
- `output_parser.py` — Embedded tool call detection
- `quality_monitor.py` — Response quality assessment + corrections
- `turn_cap.py` — Max turns enforcement
- `permission_gate.py` — Bash command whitelist
- `checkpoint.py` — Backup before modifications
- `extra_tools.py` — Glob tool implementation
- `skills.py` — Skill card loading & selection
- `knowledge.py` — Algorithm cheat sheet scoring

### Modified files:

- `tools.py` — Integrated all new modules, added glob tool, quality monitoring
  loop
- `system_prompt.py` — Added small-model-specific guidelines
- `__main__.py` — Session ID initialization for checkpoints

## Tests

- `tests/test_cody_modules.py` — 34 tests covering all new modules
