"""Tests for the duncan oracle engine and generators."""


class TestDice:
    def test_roll_d20(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.roll("d20")
        assert 1 <= result <= 20

    def test_roll_3d6(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.roll("3d6")
        assert 3 <= result <= 18

    def test_roll_with_modifier(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.roll("2d4+3")
        assert 5 <= result <= 11

    def test_roll_deterministic(self):
        from duncan.oracle import Dice

        a = Dice(seed="hello").roll("3d6")
        b = Dice(seed="hello").roll("3d6")
        assert a == b

    def test_roll_different_seeds_different(self):
        from duncan.oracle import Dice

        a = Dice(seed="hello").roll("3d6")
        b = Dice(seed="world").roll("3d6")
        assert a != b

    def test_pick(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        items = ["a", "b", "c"]
        result = d.pick(items)
        assert result in items

    def test_weighted(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        table = {"a": 100, "b": 1}
        results = [d.weighted(table) for _ in range(50)]
        assert all(r in ("a", "b") for r in results)

    def test_picks(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.picks(["a", "b", "c"], 3)
        assert len(result) == 3

    def test_sample(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.sample(["a", "b", "c", "d"], 2)
        assert len(result) == 2
        assert len(set(result)) == 2

    def test_shuffle(self):
        from duncan.oracle import Dice

        d = Dice(seed=42)
        result = d.shuffle(["a", "b", "c", "d"])
        assert sorted(result) == ["a", "b", "c", "d"]


class TestNPCGeneration:
    def test_npc_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import npc

        result = npc(Dice(seed=1))
        assert isinstance(result, str)
        assert "NPC:" in result

    def test_npc_includes_name_and_species(self):
        from duncan.oracle import Dice
        from duncan.oracles import npc

        result = npc(Dice(seed=42))
        assert "Species:" in result
        assert "Role:" in result
        assert "Traits:" in result
        assert "Appearance:" in result
        assert "Demeanor:" in result
        assert "Goal:" in result
        assert "Secret:" in result

    def test_npc_uses_mythiria_names(self):
        from duncan.oracle import Dice
        from duncan.oracles import npc

        from duncan import tables_npc as TN

        result = npc(Dice(seed=1))
        name_line = [line for line in result.splitlines() if "NPC:" in line][0]
        for given, _ in TN.NAMES_GIVEN:
            if given in name_line:
                break
        else:
            for sur, _ in TN.NAMES_SUR:
                if sur in name_line:
                    break
            else:
                assert False, f"Name not from Mythiria tables: {name_line}"


class TestEventGeneration:
    def test_event_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import event

        result = event(Dice(seed=1))
        assert isinstance(result, str)
        assert "Event:" in result

    def test_event_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import event

        result = event(Dice(seed=42))
        assert "Trigger:" in result
        assert "Participants:" in result
        assert "Complication:" in result


class TestLocationGeneration:
    def test_location_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import location

        result = location(Dice(seed=1))
        assert isinstance(result, str)
        assert "Location:" in result

    def test_location_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import location

        result = location(Dice(seed=42))
        assert "Type:" in result
        assert "Atmosphere:" in result
        assert "Hazard:" in result


class TestFactionGeneration:
    def test_faction_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import faction

        result = faction(Dice(seed=1))
        assert isinstance(result, str)
        assert "Faction:" in result

    def test_faction_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import faction

        result = faction(Dice(seed=42))
        assert "Type:" in result
        assert "Goal:" in result
        assert "Internal Conflict:" in result


class TestEncounterGeneration:
    def test_encounter_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import encounter

        result = encounter(Dice(seed=1))
        assert isinstance(result, str)
        assert "Encounter" in result

    def test_encounter_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import encounter

        result = encounter(Dice(seed=42))
        assert "Creatures:" in result
        assert "Behavior:" in result
        assert "Situation:" in result


class TestAdventureHook:
    def test_hook_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import adventure_hook

        result = adventure_hook(Dice(seed=1))
        assert isinstance(result, str)
        assert "Adventure Hook" in result

    def test_hook_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import adventure_hook

        result = adventure_hook(Dice(seed=42))
        assert "Patron:" in result
        assert "Quest:" in result
        assert "Location:" in result
        assert "Beware:" in result


class TestOracle:
    def test_oracle_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import oracle as d66

        result = d66(Dice(seed=1))
        assert isinstance(result, str)
        assert "Oracle" in result

    def test_oracle_includes_sections(self):
        from duncan.oracle import Dice
        from duncan.oracles import oracle as d66

        result = d66(Dice(seed=42))
        assert "Action:" in result
        assert "Theme:" in result
        assert "Adjective:" in result


class TestWeapon:
    def test_weapon_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import weapon

        result = weapon(Dice(seed=1))
        assert isinstance(result, str)
        assert "Equipment" in result or "Weapon:" in result


class TestRune:
    def test_rune_returns_string(self):
        from duncan.oracle import Dice
        from duncan.oracles import rune

        result = rune(Dice(seed=1))
        assert isinstance(result, str)
        assert "Rune:" in result

    def test_rune_from_mythiria_types(self):
        from duncan.oracle import Dice
        from duncan.oracles import rune

        from duncan import tables_npc as TN

        result = rune(Dice(seed=1))
        rune_line = [line for line in result.splitlines() if line.startswith("Rune:")][0]
        for rune_type in TN.RUNES:
            if rune_type in rune_line:
                break
        else:
            assert False, f"Rune type not from Mythiria: {rune_line}"


class TestSeedReproducibility:
    def test_same_seed_same_npc(self):
        from duncan.oracle import Dice
        from duncan.oracles import npc

        a = npc(Dice(seed="test_repro"))
        b = npc(Dice(seed="test_repro"))
        assert a == b

    def test_different_seed_different_npc(self):
        from duncan.oracle import Dice
        from duncan.oracles import npc

        a = npc(Dice(seed="alpha"))
        b = npc(Dice(seed="beta"))
        assert a != b

    def test_same_seed_same_hook(self):
        from duncan.oracle import Dice
        from duncan.oracles import adventure_hook

        a = adventure_hook(Dice(seed="repro_hook"))
        b = adventure_hook(Dice(seed="repro_hook"))
        assert a == b
