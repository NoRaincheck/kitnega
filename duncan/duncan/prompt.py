import json
import os
from urllib.request import Request, urlopen

API = os.getenv("CODY_API", "http://localhost:1234/v1/responses")
MODEL = os.getenv("CODY_MODEL", "qwen3.6-35b-a3b")


def _api_key():
    return os.getenv("CODY_API_KEY") or ""


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
    body = {
        "model": MODEL,
        "instructions": _CLASSIFY_PROMPT,
        "input": description,
    }
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = urlopen(
            Request(API, json.dumps(body).encode(), headers=headers),
            timeout=10,
        )
        data = json.load(resp)
    except Exception:
        return None

    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")

    candidate = text.strip().lower().splitlines()[0].split()[0] if text.strip() else ""
    ORACLE_TYPES = {"npc", "event", "location", "faction", "encounter", "hook", "oracle", "weapon", "rune"}
    return candidate if candidate in ORACLE_TYPES else None
