# Duncan — TTRPG Procedural Oracle

A minimal Python agent that acts as a dungeon master oracle for tabletop RPGs.
Duncan does not tell stories or run campaigns — it _generates content_: NPCs,
events, locations, factions, and encounters via procedural (code-based) weighted
tables and seeded randomness.

This makes it **FAIR**: transparent, reproducible, and free of LLM bias.

## Design

Duncan shares the same stdlib-only, modular philosophy as `cody` but replaces
LLM orchestration with pure procedural generation. Every oracle uses a seeded
`Dice` engine and weighted lookup tables — no API calls, no prompts, no hidden
bias.

```
duncan npc                           # random NPC
duncan event --seed tavern42         # reproducible event
duncan "goblin ambush in the cave"   # intent detected from text
duncan location "abandoned mine"     # location generation
```

## Package layout

```
packages/duncan/
├── pyproject.toml      # setuptools config, entry: duncan.__main__:main
├── duncan/
│   ├── __init__.py
│   ├── __main__.py     # CLI: argparse subcommands + intent detection
│   ├── _shared.py      # TTY detection, ANSI colour helper
│   ├── oracle.py       # Seeded Dice engine, weighted tables
│   ├── oracles.py      # Generator functions (npc/event/location/faction/encounter)
│   ├── tables_npc.py   # NPC table data
│   └── tables_scenario.py  # Event, location, faction, encounter table data
└── README.md
```

## Oracles

| Command            | Generates                                                     |
| ------------------ | ------------------------------------------------------------- |
| `duncan npc`       | Name, species, appearance, traits, profession, goal, secret   |
| `duncan event`     | Type, trigger, participants, complication, atmosphere         |
| `duncan location`  | Name, type, atmosphere, features, points of interest, hazards |
| `duncan faction`   | Name, type, goal, reputation, resources, internal conflict    |
| `duncan encounter` | Creatures, behaviour, situation, treasure                     |
| `duncan oracle`    | Free-form action / theme / adjective via seeded dice          |

## Saved results (`.oracle`) files

Append `--output .oracle` or pipe to save a `.oracle` file for replay and
shareability:

```
duncan npc > session-01.oracle
cat session-01.oracle    # view saved result
echo "---"                 # append notes between rolls  
duncan event >> session-01.oracle
```

## Seeded randomness

Pass `--seed` (string or number) for deterministic output. Same seed always
produces the same result, making rolls verifiable at the table.

```
duncan npc --seed "session-3"
```

Without `--seed`, each run picks a fresh random seed.

## Intent detection

When no subcommand is given, Duncan scans the description for keywords to infer
the oracle type automatically:

```
duncan "a shady merchant in a tavern"   → npc  (merchant, tavern)
duncan "an ancient order hiding secrets" → faction (order)
duncan "what lurks in the dark forest"  → location (forest)
```

If no keywords match, it defaults to `event`. Use an explicit subcommand to
override.

## Dice expressions

The oracle engine supports standard RPG dice notation:

- `d20`, `3d6`, `2d8+4` — any `NdS±M` expression
- Weighted table rolls via `pick()` / `weighted()` / `sample()` / `shuffle()`

## Tokens

`tokens` are the positional arguments passed to Duncan. They come in two forms:

- **Explicit oracle type + description**: `duncan npc "a grizzled guard"` first
  word matches an oracle name (npc, event, location etc), rest is textual
  context fed to the generator
- **Free-form text** (no subcommand): `duncan "a shady merchant in the tavern"`
  tokens become full input; intent detection scans them for keywords and picks
  the most likely oracle type automatically Tokens can also be empty which means
  Duncan prints help

```bash
duncan npc                       # tokens = ["npc"], desc = ""
duncan location "abandoned mine" # tokens = ["location", "abandoned mine"]
duncan "a shady merchant in tavern"  # no subcommand, intent detection picks npc
duncan oracle                    # tokens = ["oracle"], freeform action/theme/adjective
```

## CLI

```
usage: duncan [-h] [--seed SEED] ...

duncan — TTRPG procedural oracle

positional arguments:
  tokens           Oracle type (npc, event, location,
                   faction, encounter, hook, oracle, weapon, rune)
                   followed by optional description text

options:
  -h, --help       show this help message and exit
  --seed, -s SEED  Seed for reproducible generation (string or number)

Examples:
  duncan npc
  duncan event --seed tavern42
  duncan "a shady merchant in a tavern"
  duncan location "abandoned mine" --seed 8675309
```
