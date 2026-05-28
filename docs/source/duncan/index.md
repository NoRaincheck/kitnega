---
title: Duncan — TTRPG Procedural Oracle
---

# Duncan

A minimal Python agent that acts as a dungeon master oracle for tabletop RPGs.
Duncan does not tell stories or run campaigns — it _generates content_: NPCs,
events, locations, factions, and encounters via procedural (code-based) weighted
tables and seeded randomness.

This makes it **FAIR**: transparent, reproducible, and free of LLM bias.

## Usage

```bash
duncan npc                           # random NPC
duncan event --seed tavern42         # reproducible event
duncan "goblin ambush in the cave"   # intent detected from text
duncan location "abandoned mine"     # location generation
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

## Saved Results (`.oracle`) Files

Append `--output .oracle` or pipe to save a `.oracle` file for replay and
shareability:

```bash
duncan npc > session-01.oracle
cat session-01.oracle    # view saved result
echo "---"                 # append notes between rolls
duncan event >> session-01.oracle
```

## Seeded Randomness

Pass `--seed` (string or number) for deterministic output. Same seed always
produces the same result, making rolls verifiable at the table.

```bash
duncan npc --seed "session-3"
```

Without `--seed`, each run picks a fresh random seed.

## Intent Detection

When no subcommand is given, Duncan scans the description for keywords to infer
the oracle type automatically:

```bash
duncan "a shady merchant in a tavern"   → npc  (merchant, tavern)
duncan "an ancient order hiding secrets" → faction (order)
duncan "what lurks in the dark forest"  → location (forest)
```

If no keywords match, it defaults to `event`. Use an explicit subcommand to
override.

## Dice Expressions

The oracle engine supports standard RPG dice notation:

- `d20`, `3d6`, `2d8+4` — any `NdS±M` expression
- Weighted table rolls via `pick()` / `weighted()` / `sample()` / `shuffle()`

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
```
