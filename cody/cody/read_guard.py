"""Read guard: trim oversized reads to prevent context overflow.

Small models have tight context windows. A single large file read can blow the
entire conversation. This module enforces a per-read line limit (default 30) and
suggests grep/find for deeper inspection.
"""

from lib.config import get_config


def MAX_LINES():
    """Max lines per read (read from config)."""
    return get_config().get("read_limit", 30)


def trim_result(result, path):
    """Trim a read result to MAX_LINES if it exceeds the limit."""
    prefix = f"--- {path} ("
    if not result.startswith(prefix):
        return result

    # Split on the header line and content
    first_newline = result.index("\n")
    header = result[:first_newline]
    content = result[first_newline + 1 :]

    max_lines = MAX_LINES()
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return result

    trimmed = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return (
        f"{header} (truncated to {max_lines} of {len(lines)} lines)\n"
        f"{trimmed}\n"
        f"[TRIMMED: {remaining} more lines omitted. "
        f"Use Grep or Find to locate specific content.]"
    )
