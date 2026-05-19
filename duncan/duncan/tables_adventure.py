FACTION_TYPES = {
    "Druid Circle": 1, "Orc Clan": 1, "Dwarf Trading Company": 1,
    "Mages' Circle": 1, "Noble House": 1, "Religious Order": 1,
    "Thieves' Guild": 1, "Mercenary Company": 1, "Cult": 1,
    "Military Order": 1, "Secret Society": 1, "Explorers' League": 1,
    "Warlord's Horde": 1, "Rune Smith Guild": 1,
}

FACTION_PREFIXES = {
    "The Order of the": 1, "The Children of the": 1, "The Hands of": 1,
    "The Shield of": 1, "The Voice of": 1, "The": 1,
    "The Blood of": 1, "The Circle of the": 1, "The Covenant of": 1,
    "The Followers of": 1, "The Keepers of the": 1,
}

FACTION_SUFFIXES = {
    "Fang": 1, "Dawn": 1, "Eye": 1, "Flame": 1, "Shadow": 1,
    "Crown": 1, "Scale": 1, "Wing": 1, "Star": 1, "Thorn": 1,
    "Oak": 1, "Serpent": 1, "Iron": 1, "Ashen": 1, "Silver": 1,
    "Mountain": 1, "Empire": 1,
}

FACTION_GOALS = {
    "summon something from beyond the mortal realm": 1, "unite the orc tribes under one warlord": 1,
    "control the mountain pass trade routes": 1, "uncover ancient magical artifacts": 1,
    "overthrow the current rulers": 1, "protect the natural world": 1,
    "eradicate a rival group": 1, "achieve political dominance": 1,
    "expand territory": 1, "maintain the status quo": 1,
    "bring about a prophesied event": 1, "purge heretics": 1,
    "achieve immortality": 1, "unite scattered peoples": 1,
}

FACTION_REPUTATIONS = {
    "widely respected": 1, "feared and hated": 1, "unknown to most": 1,
    "viewed with suspicion": 1, "beloved by the common folk": 1,
    "officially banned": 1, "regarded as a myth": 1,
    "tolerated but not trusted": 1, "seen as corrupt": 1,
    "considered honorable": 1,
}

FACTION_RESOURCES = {
    "ancient magical artifacts": 1, "control of key mountain passes": 1,
    "vast wealth from trade": 1, "a standing army of orcs": 1,
    "blackmail material on nobles": 1, "a network of spies": 1,
    "skilled assassins": 1, "rare knowledge": 1, "fertile lands": 1,
    "a powerful relic": 1, "political alliances": 1,
}

FACTION_CONFLICTS = {
    "a power struggle between two leaders": 1, "a mole feeding info to rivals": 1,
    "disagreement over a new recruit": 1, "funds have gone missing": 1,
    "a member betrayed the group": 1, "an external threat closes in": 1,
    "a schism over ideology": 1, "a prophecy divides the members": 1,
    "a forbidden romance between members": 1, "succession crisis": 1,
    "a dark influence corrupts the leadership": 1,
}

ENCOUNTER_TYPES = {
    "Animated Statue": 1, "Giant Lizard": 1, "Goat Demon": 1,
    "Bandit": 2, "Skeleton": 1, "Lizardman": 1,
    "Wolf": 2, "Zombie": 1, "Orc Raider": 2,
    "Giant Spider": 1, "Dark Cultist": 1, "Troll": 1,
}

ENCOUNTER_BEHAVIORS = {
    "hostile and attacking": 2, "defending territory": 2,
    "frightened and fleeing": 1, "curious but cautious": 1,
    "looking for food": 1, "patrolling": 1, "sleeping": 1,
    "feeding on a kill": 1, "playing or socializing": 1,
    "guarding something": 1, "injured and desperate": 1,
    "shambling mindlessly": 1, "lying in ambush": 1,
}

ENCOUNTER_SITUATIONS = {
    "caught in a trap": 1, "blocking the path": 1,
    "fighting something else": 1, "can be avoided with caution": 1,
    "distracted by a noise": 1, "surrounding a campsite": 1,
    "emerging from the ground": 1, "perched above the trail": 1,
    "foraging for food": 1, "migrating through the area": 1,
    "bathing in a pool": 1, "bargaining among themselves": 1,
    "guarding a crate": 1, "guarding a mountain pass": 1,
}

ENCOUNTER_TREASURES = {
    "a few coins": 2, "a piece of jewelry": 1, "a potion": 1,
    "a map fragment": 1, "an uncommon weapon": 1, "a scroll": 1,
    "a key to somewhere": 1, "a gemstone": 1, "nothing of value": 2,
    "a letter with information": 1, "a curious trinket": 1,
    "a bag of rare herbs": 1, "an unbound rune": 1,
}

ENCOUNTER_NUMBERS = {
    "1": 1, "1d4": 2, "2d4": 2, "1d6": 1, "3d6": 1, "1d8+2": 1,
}

PATRONS = {
    "Baron Fin'nae": 1, "Lord Falner": 1, "Spy Sir Caldwell": 1,
    "The Sisters Three": 1, "The Seer": 1, "Archmage Radagast": 1,
    "the Village Elders": 1, "the Merchant Guild": 1,
}

QUESTS = {
    "Find the Lost Crown": 1, "Destroy the Evil Amulet": 1,
    "Rescue the Baron's Daughter": 1, "Deliver the Pinnacle Stone": 1,
    "Defeat the Tyrant Mage": 1, "Stop the Undead Plague": 1,
    "Investigate the Ruined Tower": 1, "Clear the Mountain Pass": 1,
    "Uncover the Druids' Secret": 1, "Destroy the Plague Ship": 1,
    "Seal the Dark Portal": 1, "Recover the Stolen Runes": 1,
}

ADVENTURE_SITES = {
    "Falls of Sorrow": 1, "Broken City Ruins": 1, "Howling Hills": 1,
    "Winterwood Forest": 1, "Dragonweed Swamp": 1, "Ironfist Citadel": 1,
    "the Wizard's Tower near Rockdale": 1, "Sundered Mountains": 1,
    "Verdant Forest": 1, "the Port Town of Greenside": 1,
    "the Village of Rockdale": 1, "the Dwarf Halls of Ironforge": 1,
}

BEWARE_THREATS = {
    "The Soulless Curse": 1, "Army of the Dead": 1, "The Hell Titans": 1,
    "Barnic The Vengeful": 1, "Battlefield Ghosts": 1, "Blood Magic Cult": 1,
    "a Slumbering Ancient Evil": 1, "a Corrupted Druid Circle": 1,
    "an Orc Warlord's Horde": 1, "a Blight Spreading Inland": 1,
}

D66_ACTIONS = [
    "Scheme", "Clash", "Weaken", "Initiate", "Create", "Swear",
    "Avenge", "Guard", "Defeat", "Control", "Break", "Risk",
    "Surrender", "Inspect", "Raid", "Evade", "Assault", "Deflect",
    "Threaten", "Attack", "Leave", "Preserve", "Manipulate", "Remove",
    "Eliminate", "Withdraw", "Abandon", "Investigate", "Hold", "Focus",
    "Uncover", "Breach", "Aid", "Uphold", "Falter", "Suppress",
]

D66_THEMES = [
    "Risk", "Ability", "Price", "Ally", "Battle", "Safety",
    "Love", "Barrier", "Creation", "Decay", "Trade", "Bond",
    "Death", "Honor", "Labor", "Solution", "Tool", "Balance",
    "Vow", "Protection", "Nature", "Opinion", "Burden", "Vengeance",
    "Time", "Duty", "Secret", "Innocence", "Renown", "Direction",
    "Survival", "Weapon", "Wound", "Shelter", "Leader", "Fear",
]

D66_ADJECTIVES = [
    "Perilous", "Corrupted", "Blocked", "Exposed", "Grim", "Advanced",
    "Settled", "Low", "Shrouded", "Wild", "Stolen", "Flooded",
    "Raised", "Drafty", "Dripping", "Smooth", "Silent", "Foggy",
    "Petrified", "Strange", "Ghastly", "Oozing", "Moldy", "Shadowy",
    "Depleted", "Haunted", "Ensnaring", "Solid", "Dry", "Bright",
    "Damaged", "Decaying", "Broken", "Foreboding", "Tiny", "Exposed",
]
