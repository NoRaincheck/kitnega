"""Output parser: detect embedded tool calls in assistant text.

Small models often embed tool calls as fenced ```tool blocks or XML tags instead
of using the native tool-call channel. This module detects those patterns and
returns extracted calls for logging/warning purposes.
"""

import re


def parse_text_tool_calls(text):
    """Extract tool call names from fenced code blocks in assistant text.

    Looks for:
      - ```tool ... {name, arguments} ...```  (fenced tool blocks)
      - ```json ... {name, arguments} ...```  (fenced json blocks)
      - <tool>name</tool><args>{...}</args>   (XML-style tags)

    Returns a list of dicts with 'name' and 'input'.
    """
    if not isinstance(text, str):
        return []

    calls = []

    # Strategy 1: fenced ```tool or ```json blocks
    for match in re.finditer(r"```(?:tool|json)\s*\n([\s\S]*?)```", text):
        inner = match.group(1).strip()
        obj_start = inner.find("{")
        if obj_start == -1:
            continue

        # Try to extract name and arguments from JSON-like content
        name_match = re.search(r'["\']?name["\']?\s*:\s*["\'](\w+)["\']', inner[obj_start:])
        # Try to find arguments/input as either a JSON object or string
        args_match = re.search(
            r'["\']?(?:arguments|input)["\']?\s*:\s*(\{[\s\S]*\}|["\'][^"\']*["\'])',
            inner[obj_start:],
        )

        if name_match:
            input_data = {}
            if args_match:
                try:
                    import json

                    input_data = json.loads(args_match.group(1))
                except json.JSONDecodeError, ValueError:
                    pass
            calls.append({"name": name_match.group(1), "input": input_data})

    # Strategy 2: XML-style <tool>...</tool><args>...</args> tags
    for match in re.finditer(r"<tool>\s*(\w+)\s*</tool>", text):
        name = match.group(1)
        rest = text[match.end() :]
        args_match = re.search(
            r"<(?:args|arguments)\s*>\s*(\{[\s\S]*?\})\s*</(?:args|arguments)>",
            rest,
        )

        input_data = {}
        if args_match:
            try:
                import json

                input_data = json.loads(args_match.group(1))
            except json.JSONDecodeError, ValueError:
                pass
        calls.append({"name": name, "input": input_data})

    return calls
