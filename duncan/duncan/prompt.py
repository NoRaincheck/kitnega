from lib.llm import prompt

_CLASSIFY_PROMPT = """You are a classifier for a TTRPG oracle tool. Given a description, return the single best-matching oracle type from this list: npc, event, location, faction, encounter, hook, oracle, weapon, rune.

Respond with ONLY the type name — no explanation, no punctuation, no formatting.

Examples:
  "a mysterious elf in the tavern" -> npc
  "what happens when the caravan arrives" -> event
  "the ruined tower in the dark forest" -> location
  "a band of mercenaries controlling the pass" -> faction
  "a pack of wolves blocks the road" -> encounter
  "the king needs someone to retrieve the crown" -> hook
  "a strange symbol glows on the wall" -> oracle
  "what weapon does the guard carry" -> weapon
  "a stone etched with ancient symbols" -> rune"""


def classify_oracle_type(description: str) -> str | None:
    try:
        text = prompt(description, system=_CLASSIFY_PROMPT, timeout=10)
    except Exception:
        return None

    candidate = text.strip().lower().splitlines()[0].split()[0] if text.strip() else ""
    ORACLE_TYPES = {"npc", "event", "location", "faction", "encounter", "hook", "oracle", "weapon", "rune"}
    return candidate if candidate in ORACLE_TYPES else None
