"""Skills system: per-turn tool skill card injection.

Small models benefit from context-specific guidance about how to use tools.
This module loads markdown skill cards (YAML frontmatter) and injects the most
relevant ones into the system prompt each turn.

Selection algorithm: error-recovery > recency > intent prediction.
"""

import os
import re


def _parse_frontmatter(content):
    """Parse YAML-like frontmatter from a markdown file.

    Expected format:
        ---
        name: Read
        description: Reading files efficiently
        priority: 10
        tags: [read, file, inspect]
        error_recovery_tags: [empty_response, missing_context]
        disable_model_invocation: false
        ---
    """
    fm = {}
    match = re.match(r"^---\s*\n([\s\S]*?)\n---", content)
    if not match:
        return fm

    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        # Parse list values like [a, b, c] or ["a", "b"]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            fm[key] = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        else:
            # Parse boolean or integer
            if value.lower() == "false":
                fm[key] = False
            elif value.lower() == "true":
                fm[key] = True
            elif re.match(r"^-?\d+$", value):
                fm[key] = int(value)
            else:
                fm[key] = value

    return fm


def _extract_body(content):
    """Extract markdown body after frontmatter."""
    match = re.match(r"^---\s*\n[\s\S]*?\n---\s*\n", content)
    if match:
        return content[match.end() :].strip()
    return content.strip()


def load_skills(skill_dirs):
    """Load all skill cards from the given directories.

    Args:
        skill_dirs: List of directory paths to search for .md skill files.

    Returns:
        List of dicts with 'path', 'name', 'description', 'tags',
        'priority', 'error_recovery_tags', 'body'.
    """
    all_skills = []
    seen = set()

    for base_dir in skill_dirs:
        if not os.path.isdir(base_dir):
            continue

        try:
            files = sorted(os.listdir(base_dir))
        except OSError:
            continue

        for filename in files:
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(base_dir, filename)
            try:
                with open(filepath, errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            fm = _parse_frontmatter(content)
            if not fm.get("name"):
                continue

            # Deduplicate by path
            abs_path = os.path.abspath(filepath)
            if abs_path in seen:
                continue
            seen.add(abs_path)

            all_skills.append(
                {
                    "path": filepath,
                    "name": fm.get("name", filename.replace(".md", "")),
                    "description": fm.get("description", ""),
                    "tags": fm.get("tags", []),
                    "priority": fm.get("priority", 0) or 0,
                    "error_recovery_tags": fm.get("error_recovery_tags", []),
                    "disable_model_invocation": fm.get("disable_model_invocation", False),
                    "body": _extract_body(content),
                }
            )

    # Sort by priority descending
    all_skills.sort(key=lambda s: s["priority"], reverse=True)
    return all_skills


def select_skills(all_skills, failed_tools, recent_tool_names, prompt):
    """Select the most relevant skill cards for this turn.

    Selection algorithm (in order of priority):
      1. Error recovery — skills matching previously failed tools
      2. Recency — skills matching recently used tool names
      3. Intent prediction — keyword match against user prompt

    Args:
        all_skills: All loaded skill entries.
        failed_tools: List of tool names that were blocked/failed last turn.
        recent_tool_names: Set of tool names used in the session.
        prompt: Current user prompt text.

    Returns:
        List of selected skill dicts (max 3).
    """
    MAX_SKILLS = 3
    selected = []
    injected = set()

    # 1. Error recovery — highest priority
    for skill in all_skills:
        if len(selected) >= MAX_SKILLS:
            break
        if skill["path"] in injected:
            continue

        for failed_tool in failed_tools:
            lower_name = failed_tool.lower()
            skill_tags = [t.lower() for t in skill.get("tags", [])]
            recovery_tags = [t.lower() for t in skill.get("error_recovery_tags", [])]

            if (
                lower_name in skill_tags
                or any(r in skill_tags for r in recovery_tags)
                or lower_name in skill["body"].lower()
            ):
                selected.append(skill)
                injected.add(skill["path"])
                break

    # 2. Recency — inject skills matching recently used tools
    if len(selected) < MAX_SKILLS:
        for skill in all_skills:
            if len(selected) >= MAX_SKILLS:
                break
            if skill["path"] in injected:
                continue

            skill_tags = [t.lower() for t in skill.get("tags", [])]
            if any(t in recent_tool_names for t in skill_tags):
                selected.append(skill)
                injected.add(skill["path"])

    # 3. Intent prediction — match skills to the current prompt
    if len(selected) < MAX_SKILLS and prompt:
        lower_prompt = prompt.lower()
        scored = []
        for skill in all_skills:
            if skill["path"] in injected:
                continue
            score = sum(1 for t in skill.get("tags", []) if t.lower() in lower_prompt)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, skill in scored[:3]:
            selected.append(skill)
            injected.add(skill["path"])

    return selected


def build_skill_card(skill):
    """Build a formatted skill card string from a skill entry."""
    title = skill.get("name", skill["path"].split("/")[-1].replace(".md", ""))
    body = skill.get("body", "").strip()

    if not body:
        return f"## Tool Usage Guidance\n- **{title}**: (empty skill file)"

    lines = ["## Tool Usage Guidance", f"### {title}", "", body]
    return "\n".join(lines)


def format_skills_block(selected_skills):
    """Format selected skills into a system prompt section."""
    if not selected_skills:
        return ""

    cards = [build_skill_card(s) for s in selected_skills]
    return "\n\n---\n## Tool Usage Guidance\n" + "\n\n".join(cards)
