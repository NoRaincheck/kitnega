from . import tables_adventure as TA
from . import tables_npc as TN
from . import tables_scenario as TS


def _join(parts):
    return ", ".join(p for p in parts if p)


def npc(dice, description=""):
    d = dice
    species = d.weighted(TN.SPECIES)
    title = d.weighted(TN.TITLES) if d.roll("1d6") <= 2 else ""
    given = d.weighted(TN.NAMES_GIVEN)
    surname = d.weighted(TN.NAMES_SUR)
    name = f"{title} {given} {surname}" if title else f"{given} {surname}"

    role = d.weighted(TN.ROLES)
    trait_A = d.weighted(TN.TRAITS)
    trait_B = d.weighted(TN.TRAITS) if d.roll("1d6") > 1 else ""

    build = d.weighted(TN.BUILDS)
    height = d.weighted(TN.HEIGHTS)
    skin = d.weighted(TN.SKINS)
    hair_color = d.weighted(TN.HAIR_COLORS)
    hair_style = d.weighted(TN.HAIR_STYLES)
    eyes = d.weighted(TN.EYES)
    face = d.weighted(TN.FACES)
    notable = d.weighted(TN.NOTABLE_FEATURES)
    clothing = d.weighted(TN.CLOTHING)

    appearance = _join([f"{height} {build}",
                        f"{skin} skin",
                        f"{hair_color} hair, {hair_style}",
                        f"{eyes} eyes",
                        face,
                        notable,
                        clothing])

    mannerism = d.weighted(TN.MANNERISMS)
    quirk = d.weighted(TN.QUIRKS)
    demeanor = _join([mannerism, quirk])

    profession = d.weighted(TN.PROFESSIONS)
    goal = d.weighted(TN.GOALS)
    secret = d.weighted(TN.SECRETS)

    lines = [
        f"\u2550\u2550\u2550 NPC: {name} \u2550\u2550\u2550",
        f"Species: {species}  |  Role: {role}",
        f"Traits: {_join([trait_A, trait_B])}",
        f"Profession: {profession}",
        f"Appearance: {appearance}",
        f"Demeanor: {demeanor}",
        f"Goal: {goal}",
        f"Secret: {secret}",
    ]
    return "\n".join(lines)


def event(dice, description=""):
    d = dice
    type_ = d.weighted(TS.EVENT_TYPES)
    trigger = d.weighted(TS.EVENT_TRIGGERS)
    participants = d.weighted(TS.EVENT_PARTICIPANTS)
    complication = d.weighted(TS.EVENT_COMPLICATIONS)
    env = d.weighted(TS.EVENT_ENVIRONMENTS)

    lines = [
        f"\u2550\u2550\u2550 Event: {type_} \u2550\u2550\u2550",
        f"Trigger: {trigger}",
        f"Participants: {participants}",
        f"Complication: {complication}",
        f"Atmosphere: {env}",
    ]
    return "\n".join(lines)


def location(dice, description=""):
    d = dice
    type_ = d.weighted(TS.LOCATION_TYPES)
    prefix = d.weighted(TS.LOC_PREFIXES) if d.roll("1d6") <= 3 else ""
    suffix = d.weighted(TS.LOC_SUFFIXES) if d.roll("1d6") <= 3 else ""
    name = " ".join(p for p in [prefix, type_, suffix] if p)

    atmosphere = d.weighted(TS.ATMOSPHERES)
    feature = d.weighted(TS.LOC_FEATURES)
    poi = d.weighted(TS.LOC_POI)
    secret = d.weighted(TS.LOC_SECRETS)
    hazard = d.weighted(TS.LOC_HAZARDS)

    lines = [
        f"\u2550\u2550\u2550 Location: {name} \u2550\u2550\u2550",
        f"Type: {type_}",
        f"Atmosphere: {atmosphere}",
        f"Feature: {feature}",
        f"Point of Interest: {poi}",
        f"Hidden Detail: {secret}",
        f"Hazard: {hazard}",
    ]
    return "\n".join(lines)


def faction(dice, description=""):
    d = dice
    type_ = d.weighted(TA.FACTION_TYPES)
    prefix = d.weighted(TA.FACTION_PREFIXES)
    suffix = d.weighted(TA.FACTION_SUFFIXES)
    name = f"{prefix} {suffix}" if d.roll("1d6") <= 4 else f"{prefix} {suffix} ({type_})"

    goal = d.weighted(TA.FACTION_GOALS)
    reputation = d.weighted(TA.FACTION_REPUTATIONS)
    resource = d.weighted(TA.FACTION_RESOURCES)
    conflict = d.weighted(TA.FACTION_CONFLICTS)

    lines = [
        f"\u2550\u2550\u2550 Faction: {name} \u2550\u2550\u2550",
        f"Type: {type_}",
        f"Goal: {goal}",
        f"Reputation: {reputation}",
        f"Key Resource: {resource}",
        f"Internal Conflict: {conflict}",
    ]
    return "\n".join(lines)


def encounter(dice, description=""):
    d = dice
    type_ = d.weighted(TA.ENCOUNTER_TYPES)
    behavior = d.weighted(TA.ENCOUNTER_BEHAVIORS)
    situation = d.weighted(TA.ENCOUNTER_SITUATIONS)
    treasure = d.weighted(TA.ENCOUNTER_TREASURES)
    num_expr = d.weighted(TA.ENCOUNTER_NUMBERS)
    number = d.roll(num_expr)

    lines = [
        "\u2550\u2550\u2550 Encounter \u2550\u2550\u2550",
        f"Creatures: {number} × {type_}",
        f"Behavior: {behavior}",
        f"Situation: {situation}",
        f"Treasure: {treasure}",
    ]
    return "\n".join(lines)


def adventure_hook(dice, description=""):
    d = dice
    patron = d.weighted(TA.PATRONS)
    quest = d.weighted(TA.QUESTS)
    site = d.weighted(TA.ADVENTURE_SITES)
    threat = d.weighted(TA.BEWARE_THREATS)

    lines = [
        "\u2550\u2550\u2550 Adventure Hook \u2550\u2550\u2550",
        f"Patron: {patron}",
        f"Quest: {quest}",
        f"Location: {site}",
        f"Beware: {threat}",
    ]
    return "\n".join(lines)


def oracle(dice, description=""):
    d = dice
    action = d.pick(TA.D66_ACTIONS)
    theme = d.pick(TA.D66_THEMES)
    adjective = d.pick(TA.D66_ADJECTIVES)

    lines = [
        "\u2550\u2550\u2550 Oracle \u2550\u2550\u2550",
        f"Action: {action}",
        f"Theme: {theme}",
        f"Adjective: {adjective}",
    ]
    return "\n".join(lines)


def weapon(dice, description=""):
    d = dice
    wpn = d.weighted(TN.WEAPONS)
    gear = d.weighted(TN.GEAR)
    lines = [
        "\u2550\u2550\u2550 Equipment \u2550\u2550\u2550",
        f"Weapon: {wpn}",
        f"Gear: {gear}",
    ]
    return "\n".join(lines)


def rune(dice, description=""):
    d = dice
    rune_type = d.weighted(TN.RUNES)
    lines = [
        "\u2550\u2550\u2550 Rune \u2550\u2550\u2550",
        f"Rune: {rune_type}",
        "A soapstone symbol carved with ancient markings.",
    ]
    return "\n".join(lines)
