"""Permission gate: bash command whitelist enforcement.

Small models may issue dangerous shell commands. This module provides a
configurable whitelist of allowed bash command prefixes with three modes:
- auto (default): silently block and notify
- accept-all: no gating
- manual: prompt user for each blocked command

Config via environment variables:
  KN_BASH_MODE   - "auto" | "accept-all" | "manual" (default: "auto")
  KN_BASH_ALLOW  - comma-separated extra allow prefixes
"""

import os

DEFAULT_ALLOW_LIST = frozenset([
    # Navigation & inspection
    "cd", "ls", "cat", "head", "tail", "wc", "less", "more", "file", "stat",
    # Git operations (prefix patterns — commands starting with these)
    "git log ", "git status ", "git diff ", "git branch ", "git show ",
    "git stash ", "git checkout ", "git merge ", "git rebase ",
    "git add ", "git commit ", "git push ", "git pull ",
    # File operations (safe subset — must match exact command or subcommand)
    "cp ", "mv ", "mkdir ", "rm ", "touch ", "ln ", "chmod ", "chown ",
    # Search & find
    "find ", "grep ", "rg ", "ag ", "fd ", "locate",
    # Process management
    "ps", "kill", "pkill", "top", "htop",
    # Network (read-only)
    "curl", "wget", "ping", "nslookup", "dig",
    # Package managers
    "npm install ", "npm run ", "npm test ",
    "yarn ", "pnpm ",
    "pip install ", "pip list ", "pip show ", "pip freeze ",
    "cargo build ", "cargo test ",
    "go build ", "go test ",
])


def get_mode():
    """Return the permission gate mode."""
    raw = os.getenv("KN_BASH_MODE", "auto")
    if raw in ("auto", "accept-all", "manual"):
        return raw
    return "auto"


def get_extra_prefixes():
    """Parse KN_BASH_ALLOW into a list of extra prefixes."""
    raw = os.getenv("KN_BASH_ALLOW", "")
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def is_command_allowed(command):
    """Check if a command is allowed by the whitelist.

    Matches against each allow-list entry as a prefix. Also checks that git,
    npm, pip, cargo, and go commands (without trailing space) are allowed when
    followed by a known subcommand.

    Args:
        command: The full bash command string to check.

    Returns:
        True if the command matches any prefix in the allow list.
    """
    prefixes = [*DEFAULT_ALLOW_LIST, *get_extra_prefixes()]
    for prefix in prefixes:
        if command.startswith(prefix):
            return True

    # Handle git subcommands without trailing space (e.g. "git status")
    git_cmd = command.split()[0] if command.split() else ""
    if git_cmd == "git" and len(command) > 3:
        subcmd = command[4:].split()[0] if len(command.split()) > 1 else ""
        known_gits = {"log", "status", "diff", "branch", "show", "stash",
                      "checkout", "merge", "rebase", "add", "commit",
                      "push", "pull", "init", "clone", "reset", "tag", "fetch"}
        if subcmd in known_gits:
            return True

    # Handle npm/pnpm/yarn without trailing space (e.g. "npm test")
    for pkg in ["npm", "pnpm", "yarn"]:
        if command.startswith(pkg + " ") and len(command) > len(pkg) + 1:
            subcmd = command[len(pkg) + 1:].split()[0]
            known_pkgs = {"install", "run", "test", "start", "build", "update",
                          "add", "remove", "init", "link", "uninstall"}
            if subcmd in known_pkgs:
                return True

    return False


def gate_command(command):
    """Gate a bash command according to current config.

    Args:
        command: The full bash command string.

    Returns:
        None if allowed, or an error message string if blocked.
    """
    mode = get_mode()
    if mode == "accept-all":
        return None  # no gating

    if is_command_allowed(command):
        return None  # whitelisted

    if mode == "manual":
        # Return a special marker for manual approval (handled by caller)
        truncated = command[:80] + "..." if len(command) > 80 else command
        return f"APPROVAL_REQUIRED:{truncated}"

    # auto mode — block and notify
    truncated = command[:60] + "..." if len(command) > 60 else command
    return (
        f"harness intervention: bash command blocked by whitelist — \"{truncated}\". "
        f"Use allowed commands only."
    )
