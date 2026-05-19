import argparse
import sys

from .oracle import Dice
from .oracles import adventure_hook, oracle, encounter, event, faction, location, npc, rune, weapon
from .prompt import classify_oracle_type

ORACLES = {
    "npc": npc,
    "event": event,
    "location": location,
    "faction": faction,
    "encounter": encounter,
    "hook": adventure_hook,
    "oracle": oracle,
    "weapon": weapon,
    "rune": rune,
}

_INTENT_KEYWORDS = {
    "npc": [
        "npc", "character", "person", "someone", "merchant",
        "guard", "priest", "thief", "blacksmith", "innkeeper",
        "wizard", "ranger", "knight", "peasant", "lord",
        "lady", "villager", "stranger", "ally", "villain",
        "elf", "dwarf", "orc", "hag", "troll", "ogre",
    ],
    "event": [
        "event", "happen", "scenario", "incident", "occurrence",
        "situation", "development", "happening",
    ],
    "location": [
        "location", "place", "tavern", "dungeon", "forest",
        "temple", "shop", "castle", "cave", "ruin", "town",
        "city", "village", "inn", "library", "market",
        "tower", "pass", "swamp", "hills",
    ],
    "faction": [
        "faction", "guild", "order", "cult", "kingdom",
        "clan", "society", "group", "organization",
        "druid", "orc clan", "tribe",
    ],
    "encounter": [
        "encounter", "monster", "creature", "beast", "ambush",
        "combat", "fight", "battle", "enemy", "enemies",
        "statue", "lizard", "demon", "skeleton", "wolf", "zombie",
    ],
    "hook": [
        "hook", "adventure", "quest", "patron", "mission",
    ],
    "oracle": [
        "oracle", "action", "theme", "spark", "inspiration",
    ],
    "weapon": [
        "weapon", "equipment", "gear", "arm", "sword", "bow",
    ],
    "rune": [
        "rune", "rune magic", "rune magic",
    ],
}


def _detect_intent(text):
    text_lower = text.lower()
    scores = {k: 0 for k in ORACLES}
    for oracle_type, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[oracle_type] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "event"


def _generate(oracle_type, seed, description):
    dice = Dice(seed=seed or description or None)
    gen = ORACLES[oracle_type]
    result = gen(dice, description or "")
    header = f"seed: {seed or 'random'}" if seed else ""
    return f"{result}\n{header}" if header else result


def main():
    parser = argparse.ArgumentParser(
        description="duncan — TTRPG procedural oracle (Mythiria setting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  npc        Generate an NPC (Mythiria rules)\n"
            "  event      Generate an event\n"
            "  location   Generate a location\n"
            "  faction    Generate a faction\n"
            "  encounter  Generate an encounter (Mythiria bestiary)\n"
            "  hook       Generate an adventure hook\n"
            "  oracle       Free-form action / theme / adjective\n"
            "  weapon     Generate a weapon + gear\n"
            "  rune       Generate a Mythiria rune\n"
            "\n"
            "Without a subcommand, the oracle type is inferred from text.\n"
            "\n"
            "Examples:\n"
            "  duncan npc\n"
            "  duncan hook --seed tavern42\n"
            "  duncan \"a ruined tower in the mountains\"\n"
            "  duncan location \"abandoned mine\" --seed 8675309\n"
        ),
    )
    parser.add_argument(
        "--seed", "-s",
        default=None,
        help="Seed for reproducible generation (string or number)",
    )
    parser.add_argument(
        "--prompt", "-p",
        action="store_true",
        help="Use LLM to infer oracle type from description instead of keyword matching",
    )
    parser.add_argument("tokens", nargs="*",
                        help="Oracle type followed by description")

    args = parser.parse_args()
    tokens = args.tokens

    oracle_type = None
    desc_parts = tokens
    if tokens and tokens[0] in ORACLES:
        oracle_type = tokens[0]
        desc_parts = tokens[1:]

    desc = " ".join(desc_parts)

    if not oracle_type:
        if desc:
            if args.prompt:
                oracle_type = classify_oracle_type(desc) or _detect_intent(desc)
            else:
                oracle_type = _detect_intent(desc)
        else:
            parser.print_help()
            return

    result = _generate(oracle_type, args.seed, desc)
    sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()
