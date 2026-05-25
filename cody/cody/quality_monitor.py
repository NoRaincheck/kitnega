"""Quality monitor: detect empty responses, hallucinated tools, and repeated loops.

Small models commonly produce:
- Empty responses with no tool calls
- Tool calls referencing non-existent tools
- Repeated identical tool calls (infinite loops)

This module assesses each turn's quality and returns correction messages when
issues are detected. A 2-strike cap prevents endless correction loops.
"""


def assess_response(text, tool_calls, recent_tool_calls, known_tools):
    """Assess a response for common small-model failure modes.

    Args:
        text: Assistant text content (empty string if none)
        tool_calls: List of tool call dicts from this turn
        recent_tool_calls: Tool calls from the previous turn
        known_tools: Set of valid tool names

    Returns:
        Dict with 'ok' key (True/False) and optional 'reason' on failure.
    """
    # 1. Empty response with no tool calls
    if not text.strip() and len(tool_calls) == 0:
        return {"ok": False, "reason": "empty_response"}

    # 2. Hallucinated tool names (only when registry is populated)
    for tc in tool_calls:
        name = tc.get("name", "")
        if not name:
            return {"ok": False, "reason": "empty_tool_name"}
        if known_tools and name.lower() not in {t.lower() for t in known_tools}:
            return {"ok": False, "reason": f"unknown_tool:{name}"}

    # 3. Repeated tool call loop (exact match with previous turn)
    if tool_calls and recent_tool_calls:
        for tc in tool_calls:
            for prev in recent_tool_calls:
                if (tc.get("name") == prev.get("name")
                        and tc.get("arguments", "{}") == prev.get("arguments", "{}")):
                    return {"ok": False, "reason": "repeated_tool_call"}

    return {"ok": True}


def build_correction_message(reason):
    """Build a user-facing correction message for a quality issue."""
    corrections = {
        "empty_response": (
            "Your previous response was empty. Please respond with either "
            "text or a tool call to make progress on the task."
        ),
        "empty_tool_name": (
            "Your tool call had an empty name. Please specify a valid tool name. "
            "Available tools include: read, write, edit, bash, grep, find, ls."
        ),
        "repeated_tool_call": (
            "You just made the exact same tool call as your previous turn. "
            "This suggests you may be stuck in a loop. Please try a different "
            "approach or explain what you're trying to accomplish."
        ),
    }

    if reason.startswith("unknown_tool:"):
        tool_name = reason[len("unknown_tool:"):]
        return (
            f"Tool '{tool_name}' does not exist. "
            "Available tools are: read, write, edit, bash, grep, find, ls."
        )

    return corrections.get(reason, f"Issue detected: {reason}. Please try again.")


def phrase_for_user(reason):
    """Human-readable phrase describing the quality issue."""
    phrases = {
        "empty_response": "the model returned an empty response",
        "empty_tool_name": "the model emitted a tool call with no name",
        "repeated_tool_call": "the model repeated its previous tool call verbatim",
    }

    if reason.startswith("unknown_tool:"):
        return f"the model called a tool that doesn't exist ({reason.split(':', 1)[1]})"

    return phrases.get(reason, f"quality issue ({reason})")
