"""Knowledge injection: algorithm cheat sheet scoring & prompt matching.

Small models lack algorithm knowledge mid-task. This module loads markdown
cheat sheets (YAML frontmatter) from a knowledge directory and scores them
against the user prompt using keyword + bigram matching. Top matches are
injected as `## Algorithm Reference` into the system prompt.
"""

import os
import re


def _parse_frontmatter(content):
    """Parse YAML-like frontmatter (same format as skills.py)."""
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


def tokenize(text):
    """Tokenize text into words and bigrams for scoring.

    Returns a dict with 'words' (set of lowercase tokens) and
    'bigrams' (set of 'word1_word2' pairs).
    """
    lower = text.lower()
    tokens = re.findall(r"[a-z]+", lower)
    words = set(tokens)

    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add(f"{tokens[i]}_{tokens[i + 1]}")

    return {"words": words, "bigrams": bigrams}


def score_entry(entry, tokens):
    """Score a knowledge entry against tokenized prompt.

    Scoring: word match = 1.0, bigram match = 2.0.
    Also checks if tag keywords appear in the entry body itself.
    """
    if not entry.get("tags"):
        return 0.0

    score = 0.0
    lower_body = entry["body"].lower()

    for tag in entry["tags"]:
        lower_tag = tag.lower()
        # Word match against prompt tokens
        if lower_tag in tokens["words"]:
            score += 1.0
            # Bonus: how many times the tag appears in the body
            count = lower_body.count(lower_tag)
            score += min(count * 0.5, 2.0)

        # Bigram match against prompt bigrams
        tag_tokens = re.findall(r"[a-z]+", lower_tag)
        for i in range(len(tag_tokens) - 1):
            bigram = f"{tag_tokens[i]}_{tag_tokens[i + 1]}"
            if bigram in tokens["bigrams"]:
                score += 2.0

    return score


def load_knowledge(knowledge_dirs):
    """Load all knowledge entries from the given directories.

    Args:
        knowledge_dirs: List of directory paths to search for .md cheat sheets.

    Returns:
        List of dicts with 'path', 'name', 'tags', and 'body'.
    """
    all_entries = []
    seen = set()

    for base_dir in knowledge_dirs:
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
            abs_path = os.path.abspath(filepath)
            if abs_path in seen:
                continue
            seen.add(abs_path)

            try:
                with open(filepath, errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            fm = _parse_frontmatter(content)
            all_entries.append(
                {
                    "path": filepath,
                    "name": fm.get("name", filename.replace(".md", "")),
                    "tags": fm.get("tags", []),
                    "body": _extract_body(content),
                }
            )

    return all_entries


def select_knowledge(all_entries, prompt):
    """Select the most relevant knowledge entries for this turn.

    Scores each entry against the user prompt via keyword/bigram matching.
    Returns top 3 non-zero scored entries.

    Args:
        all_entries: All loaded knowledge entries.
        prompt: Current user prompt text.

    Returns:
        List of selected knowledge dicts (max 3).
    """
    MAX_ENTRIES = 3
    if not prompt or not all_entries:
        return []

    tokens = tokenize(prompt)

    scored = [(score_entry(entry, tokens), entry) for entry in all_entries]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [entry for score, entry in scored if score > 0][:MAX_ENTRIES]
    return selected


def build_reference_card(entry):
    """Build a formatted algorithm reference card string."""
    title = entry.get("name", entry["path"].split("/")[-1].replace(".md", ""))
    body = entry.get("body", "").strip()

    if not body:
        return f"## Algorithm Reference\n- **{title}**: (empty)"

    lines = ["## Algorithm Reference", f"### {title}", "", body]
    return "\n".join(lines)


def format_knowledge_block(selected_entries):
    """Format selected knowledge entries into a system prompt section."""
    if not selected_entries:
        return ""

    cards = [build_reference_card(e) for e in selected_entries]
    return "\n\n---\n## Algorithm Reference\n" + "\n\n".join(cards)
