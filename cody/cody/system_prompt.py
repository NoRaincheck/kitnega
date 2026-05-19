"""System prompt construction following pi coding-agent semantics."""

import os
from datetime import date

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "venv", ".ruff_cache", ".pytest_cache"}


_CONTEXT_CACHE = {}


def _find_context_files(roots, names, limit=40):
    key = (tuple(roots), tuple(sorted(names)), limit)
    if key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[key]
    home = os.path.expanduser("~")
    found = []
    for root_expanded in (os.path.expanduser(r) for r in roots):
        if not os.path.isdir(root_expanded):
            continue
        for base, dirs, files in os.walk(root_expanded):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in (f for f in files if f.lower() in names):
                path = os.path.abspath(os.path.join(base, name))
                found.append("~" + path[len(home) :] if path.startswith(home + os.sep) else os.path.relpath(path))
                if len(found) >= limit:
                    result = ", ".join(sorted(dict.fromkeys(found)))
                    _CONTEXT_CACHE[key] = result
                    return result
    result = ", ".join(sorted(dict.fromkeys(found))) or "none"
    _CONTEXT_CACHE[key] = result
    return result


def _format_skills_for_prompt(skills):
    visible = [s for s in skills if not s.get("disable_model_invocation")]
    if not visible:
        return ""

    def esc(v):
        return str(v).replace("&", "&amp;").replace("<", "&lt;")

    lines = ["", "<available_skills>"]
    for sk in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{esc(sk.get('name', ''))}</name>")
        lines.append(f"    <description>{esc(sk.get('description', ''))}</description>")
        loc = sk.get("location")
        if loc:
            lines.append(f"    <location>{esc(loc)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    lines.append("")
    return "\n".join(lines)


def build_system_prompt(
    *,
    custom_prompt=None,
    selected_tools=None,
    tool_snippets=None,
    prompt_guidelines=None,
    append_system_prompt="",
    cwd=".",
    context_files=None,
    skills=None,
):
    prompt_cwd = cwd.replace("\\", "/")
    date_str = date.today().isoformat()

    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""
    context_files = context_files or []
    skills = skills or []

    if custom_prompt:
        prompt = custom_prompt
        if append_section:
            prompt += append_section
        if context_files:
            prompt += "\n\n<project_context>\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for cf in context_files:
                prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
            prompt += "</project_context>\n"
        has_read = not selected_tools or "read" in selected_tools
        if has_read and skills:
            prompt += _format_skills_for_prompt(skills)
        prompt += f"\nCurrent date: {date_str}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    tools = selected_tools or ["read", "bash", "edit", "write"]
    tool_snippets = tool_snippets or {}
    visible_tools = [t for t in tools if t in tool_snippets]
    tools_list = ", ".join(visible_tools) if visible_tools else "(none)"

    guidelines_list = []
    guidelines_set = set()

    def add_guideline(g):
        if g not in guidelines_set:
            guidelines_set.add(g)
            guidelines_list.append(g)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        add_guideline("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    for g in prompt_guidelines or []:
        g = g.strip()
        if g:
            add_guideline(g)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    readme_path = _find_context_files([cwd], {"readme.md"})
    agents_path = _find_context_files([cwd], {"agents.md"})

    prompt = f"""You are an expert coding assistant operating inside Cody, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools: {tools_list}

Guidelines:
{guidelines}

Cody documentation (read only when the user asks about Cody itself, its configuration, or skills):
- Main documentation: {readme_path}
- Agent configuration: {agents_path}
- When working on Cody topics, read the relevant files before implementing"""

    if append_section:
        prompt += append_section

    if context_files:
        prompt += "\n\n<project_context>\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
        prompt += "</project_context>\n"

    if has_read and skills:
        prompt += _format_skills_for_prompt(skills)

    prompt += f"\nCurrent date: {date_str}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"

    return prompt
