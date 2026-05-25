"""Turn cap: enforce maximum turns per agent run.

Small models can loop endlessly without a turn limit. This module provides
a configurable max-turns via KN_MAX_TURNS environment variable (default 100).
When exceeded, the run() function returns with an abort message.
"""

import os


def get_turn_cap():
    """Return the configured maximum number of turns."""
    raw = os.getenv("KN_MAX_TURNS", "100")
    try:
        val = int(raw)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return 100


def check_turn_cap(turn_index, cap):
    """Check if the current turn has exceeded the limit.

    Args:
        turn_index: Current zero-based turn index within run() loop
        cap: Maximum turns (None = unlimited)

    Returns:
        True if the turn should be aborted.
    """
    if cap is None:
        return False
    # turn_index is 0-based, so we check >= cap
    return turn_index >= cap
