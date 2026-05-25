"""Read guard: trim oversized reads to prevent context overflow.

Small models have tight context windows. A single large file read can blow the
entire conversation. This module enforces a per-read line limit (default 30) and
suggests grep/find for deeper inspection.
"""

import os

MAX_LINES = int(os.getenv("KN_READ_LIMIT", "30"))


def trim_result(result, path):
    """Trim a read result to MAX_LINES if it exceeds the limit."""
    prefix = f"--- {path} ("
    if not result.startswith(prefix):
        return result

    # Split on the header line and content
    first_newline = result.index("\n")
    header = result[:first_newline]
    content = result[first_newline + 1:]

    lines = content.split("\n")
    if len(lines) <= MAX_LINES:
        return result

    trimmed = "\n".join(lines[:MAX_LINES])
    remaining = len(lines) - MAX_LINES
    return (
        f"{header} (truncated to {MAX_LINES} of {len(lines)} lines)\n"
        f"{trimmed}\n"
        f"[TRIMMED: {remaining} more lines omitted. "
        f"Use Grep or Find to locate specific content.]"
    )
