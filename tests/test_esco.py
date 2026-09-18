"""Unit characterisation of aiisco/esco.py.

These pin the awkward corners of the real ESCO download: naming variants
between releases, chains that loop or stop early, relations that point at
nothing, and rows with blank keys.
"""

import csv
import os

import pytest

from aiisco import esco

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "esco")

GRP = "http://example.org/esco/isco/"
OCC = "http://example.org/esco/occupation/"
SKL = "http://example.org/esco/skill/"


def touch(directory, *names):
    """Create empty files so find_csv has something to match on."""
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


def isco_row(uri, code, label):
    return {"conceptUri": uri, "code": code, "preferredLabel": label}


def broader_row(child, parent):
    return {"conceptUri": child, "broaderUri": parent}


def relation_row(occ, skill, relation="essential"):
    return {"occupationUri": occ, "skillUri": skill, "relationType": relation}


def skill_row(uri, skill_type="skill/competence", reuse="transversal", label="a skill"):
    return {
        "conceptUri": uri,
        "skillType": skill_type,
        "reuseLevel": reuse,
        "preferredLabel": label,
        "description": f"description of {label}",
    }


@pytest.fixture
def index():
    """A small index: two clean ISCO branches, one cycle, one codeless group."""
    isco_rows = [
        isco_row(GRP + "C2", "2", "Professionals"),
        isco_row(GRP + "C25", "25", "ICT professionals"),
        isco_row(GRP + "C251", "251", "Developers"),
        isco_row(GRP + "C2511", "2511", "Systems analysts"),
        isco_row(GRP + "CE", "", "Unclassified helpers"),
        isco_row(GRP + "C9", "9", "Elementary occupations"),
        isco_row(GRP + "CYC1", "911", "Cleaners"),
        isco_row(GRP + "CYC2", "91", "Cleaners and helpers"),
    ]
    broader_rows = [
        broader_row(GRP + "C25", GRP + "C2"),
        broader_row(GRP + "C251", GRP + "C25"),
        broader_row(GRP + "C2511", GRP + "C251"),
        broader_row(OCC + "deep", GRP + "C2511"),
        broader_row(OCC + "shallow", GRP + "C2"),
        broader_row(OCC + "codeless", GRP + "CE"),
        broader_row(GRP + "CE", GRP + "C9"),
        broader_row(OCC + "looping", GRP + "CYC1"),
        broader_row(GRP + "CYC1", GRP + "CYC2"),
        broader_row(GRP + "CYC2", GRP + "CYC1"),
        broader_row(OCC + "orphan", GRP + "C-GONE"),
    ]
    tables = {
        "isco": esco.EscoTable("ISCO groups", "isco.csv", isco_rows),
        "broader": esco.EscoTable("Broader", "broader.csv", broader_rows),
        "skills": esco.EscoTable("Skills", "skills.csv", [skill_row(SKL + "s1")]),
        "relations": esco.EscoTable("Relations", "relations.csv", []),
    }
    return esco.build_index(tables)


# ---------------------------------------------------------------------------
# Finding and reading the CSVs
# ---------------------------------------------------------------------------

def test_list_dir_is_empty_for_a_directory_that_does_not_exist(tmp_path):
    assert esco.list_dir(str(tmp_path / "nope")) == []


def test_list_dir_returns_the_names_it_finds(tmp_path):
    touch(tmp_path, "a.csv")
    assert esco.list_dir(str(tmp_path)) == ["a.csv"]


@pytest.mark.parametrize("present, expected", [
    (["occupations_en.csv"], "occupations_en.csv"),
    (["occupations.csv"], "occupations.csv"),
    (["occupations_v1.2.1_en.csv"], "occupations_v1.2.1_en.csv"),
    (["OCCUPATIONS.CSV"], "OCCUPATIONS.CSV"),
])
def test_find_csv_tolerates_the_naming_variants(tmp_path, present, expected):
    touch(tmp_path, *present)
    hints = ("occupations_en.csv", "occupations.csv", "occupations")
    assert esco.find_csv(str(tmp_path), hints) == os.path.join(str(tmp_path), expected)


def test_find_csv_prefers_the_first_hint_that_matches(tmp_path):
    touch(tmp_path, "occupations.csv", "occupations_en.csv")
    hints = ("occupations_en.csv", "occupations.csv", "occupations")
    found = esco.find_csv(str(tmp_path), hints)
    assert os.path.basename(found) == "occupations_en.csv"


def test_find_csv_prefers_an_exact_match_over_a_longer_substring_match(tmp_path):
    touch(tmp_path, "regional_occupations.csv", "occupations.csv")
    found = esco.find_csv(str(tmp_path), ("occupations.csv",))
    assert os.path.basename(found) == "occupations.csv"


def test_find_csv_ignores_substring_matches_that_are_not_csv(tmp_path):
    touch(tmp_path, "occupations_en.csv.bak", "notes_about_occupations.txt")
    assert esco.find_csv(str(tmp_path), ("occupations",)) is None


def test_find_csv_returns_none_when_nothing_matches(tmp_path):
    touch(tmp_path, "skills_en.csv")
    assert esco.find_csv(str(tmp_path), ("occupations", "occupations.csv")) is None


def test_find_csv_returns_none_when_the_directory_is_missing(tmp_path):
    assert esco.find_csv(str(tmp_path / "gone"), ("occupations",)) is None


def test_read_csv_returns_one_dict_per_row(tmp_path):
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["conceptUri", "preferredLabel"])
        writer.writerow(["u1", "gérer un café"])
    assert esco.read_csv(str(path)) == [{"conceptUri": "u1", "preferredLabel": "gérer un café"}]


def test_load_tables_reads_every_table_from_the_fixture_directory():
    tables = esco.load_tables(FIXTURES)
    assert set(tables) == {spec.key for spec in esco.TABLE_SPECS}
    assert len(tables["occupations"].rows) == 7
    assert len(tables["skills"].rows) == 10
    assert len(tables["relations"].rows) == 14
    assert tables["isco"].label == "ISCO groups"
    assert os.path.basename(tables["broader"].path) == "broaderRelationsOccPillar_en.csv"


def test_load_tables_yields_an_empty_table_when_a_csv_is_absent(tmp_path):
    touch(tmp_path, "occupations_en.csv")
    tables = esco.load_tables(str(tmp_path))
    assert tables["skills"].path is None
    assert tables["skills"].rows == []
    assert tables["skills"].label == "Skills"


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def test_build_isco_lookup_defaults_missing_columns_to_blank():
    lookup = esco.build_isco_lookup([{"conceptUri": GRP + "C1"}])
    assert lookup[GRP + "C1"] == {"code": "", "label": ""}


def test_build_isco_lookup_keeps_the_last_row_for_a_repeated_uri():
    lookup = esco.build_isco_lookup([
        isco_row(GRP + "C1", "1", "first"),
        isco_row(GRP + "C1", "1", "second"),
    ])
    assert lookup[GRP + "C1"]["label"] == "second"


def test_build_broader_map_drops_rows_with_a_blank_side():
    parent_of = esco.build_broader_map([
        broader_row(OCC + "a", GRP + "C1"),
        broader_row("", GRP + "C1"),
        broader_row(OCC + "b", ""),
    ])
    assert parent_of == {OCC + "a": GRP + "C1"}


@pytest.mark.parametrize("relation, bucket", [
    ("essential", "essential"),
    ("Essential", "essential"),
    ("ESSENTIAL", "essential"),
    ("optional", "optional"),
])
def test_build_skill_relations_sorts_by_relation_type(relation, bucket):
    essential, optional = esco.build_skill_relations([relation_row(OCC + "a", SKL + "s1", relation)])
    found = essential if bucket == "essential" else optional
    assert found[OCC + "a"] == [SKL + "s1"]


def test_build_skill_relations_treats_an_unknown_relation_type_as_optional():
    # BUG: anything that is not "essential" lands in the optional bucket, so a
    # typo or a new ESCO relation type would silently be published as optional.
    essential, optional = esco.build_skill_relations([relation_row(OCC + "a", SKL + "s1", "nonsense")])
    assert essential == {}
    assert optional[OCC + "a"] == [SKL + "s1"]


def test_build_skill_relations_drops_rows_with_a_blank_uri():
    essential, optional = esco.build_skill_relations([
        relation_row("", SKL + "s1"),
        relation_row(OCC + "a", ""),
    ])
    assert essential == {} and optional == {}


def test_build_skill_relations_keeps_duplicate_rows():
    # BUG: the real download can repeat a relation; it is never de-duplicated,
    # so the skill is listed twice on the occupation and counted twice below.
    essential, _ = esco.build_skill_relations([
        relation_row(OCC + "a", SKL + "s1"),
        relation_row(OCC + "a", SKL + "s1"),
    ])
    assert essential[OCC + "a"] == [SKL + "s1", SKL + "s1"]


@pytest.mark.parametrize("raw_type, expected", [
    ("knowledge", "knowledge"),
    ("Knowledge", "knowledge"),
    ("some knowledge area", "knowledge"),
    ("skill/competence", "skill"),
    ("", "skill"),
])
def test_build_skill_lookup_normalises_the_skill_type(raw_type, expected):
    # BUG: a blank skillType becomes "skill" rather than staying blank, so the
    # "(empty)" label in the type breakdown can never be printed.
    lookup = esco.build_skill_lookup([skill_row(SKL + "s1", skill_type=raw_type)])
    assert lookup[SKL + "s1"]["type"] == expected


def test_build_skill_lookup_keeps_an_empty_reuse_level():
    lookup = esco.build_skill_lookup([skill_row(SKL + "s1", reuse="")])
    assert lookup[SKL + "s1"]["reuse_level"] == ""


def test_build_skill_lookup_defaults_missing_columns_to_blank():
    lookup = esco.build_skill_lookup([{"conceptUri": SKL + "s1"}])
    assert lookup[SKL + "s1"] == {
        "uri": SKL + "s1",
        "title": "",
        "description": "",
        "type": "skill",
        "reuse_level": "",
    }


def test_build_skill_lookup_keeps_the_last_row_for_a_repeated_uri():
    lookup = esco.build_skill_lookup([
        skill_row(SKL + "s1", label="first"),
        skill_row(SKL + "s1", label="second"),
    ])
    assert lookup[SKL + "s1"]["title"] == "second"


# ---------------------------------------------------------------------------
# The ISCO-08 hierarchy
# ---------------------------------------------------------------------------

def test_ancestors_walks_up_to_the_root(index):
    assert list(esco.ancestors(OCC + "deep", index.parent_of)) == [
        GRP + "C2511", GRP + "C251", GRP + "C25", GRP + "C2",
    ]


def test_ancestors_stops_on_a_cycle_instead_of_looping_forever(index):
    walked = list(esco.ancestors(OCC + "looping", index.parent_of))
    assert walked == [GRP + "CYC1", GRP + "CYC2", GRP + "CYC1"]


def test_ancestors_of_an_unconnected_uri_is_empty(index):
    assert list(esco.ancestors(OCC + "unknown", index.parent_of)) == []


def test_resolve_hierarchy_builds_the_full_chain_broadest_first(index):
    code, group, chain = esco.resolve_hierarchy(OCC + "deep", index)
    assert (code, group) == ("2511", "Systems analysts")
    assert chain == ["Professionals", "ICT professionals", "Developers", "Systems analysts"]


def test_resolve_hierarchy_accepts_a_chain_shorter_than_four_levels(index):
    code, group, chain = esco.resolve_hierarchy(OCC + "shallow", index)
    assert (code, group, chain) == ("2", "Professionals", ["Professionals"])


def test_resolve_hierarchy_skips_a_group_without_a_code(index):
    code, group, chain = esco.resolve_hierarchy(OCC + "codeless", index)
    assert (code, group) == ("9", "Elementary occupations")
    assert chain == ["Elementary occupations", "Unclassified helpers"]


def test_resolve_hierarchy_repeats_a_label_when_the_chain_cycles(index):
    # BUG: the cycle guard stops the walk but only after re-visiting the first
    # group, so a looping branch publishes that label twice in the hierarchy.
    code, group, chain = esco.resolve_hierarchy(OCC + "looping", index)
    assert (code, group) == ("911", "Cleaners")
    assert chain == ["Cleaners", "Cleaners and helpers", "Cleaners"]


def test_resolve_hierarchy_is_empty_when_the_parent_is_unknown(index):
    assert esco.resolve_hierarchy(OCC + "orphan", index) == ("", "", [])


def test_resolve_hierarchy_is_empty_when_there_is_no_parent_at_all(index):
    assert esco.resolve_hierarchy(OCC + "detached", index) == ("", "", [])


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_skill_entry_describes_a_known_skill():
    skills = esco.build_skill_lookup([skill_row(SKL + "s1", "knowledge", "cross-sector", "data law")])
    assert esco.skill_entry(SKL + "s1", skills) == {
        "uri": SKL + "s1",
        "title": "data law",
        "type": "knowledge",
        "reuse_level": "cross-sector",
    }


def test_skill_entry_is_blank_for_a_relation_pointing_at_an_unknown_uri():
    # BUG: a relation to a skill missing from skills_en.csv is published as an
    # entry with no title and no type rather than being dropped.
    assert esco.skill_entry(SKL + "ghost", {}) == {
        "uri": SKL + "ghost",
        "title": "",
        "type": "",
        "reuse_level": "",
    }


def test_build_occupation_falls_back_to_the_isco_group_column(index):
    occupation = esco.build_occupation(
        {"conceptUri": OCC + "orphan", "iscoGroup": "5223", "preferredLabel": "shop assistant"},
        index,
    )
    # BUG: the fallback fills isco_code but leaves isco_group blank, so these
    # occupations carry a code with no group label.
    assert occupation["isco_code"] == "5223"
    assert occupation["isco_group"] == ""
    assert occupation["hierarchy"] == []


def test_build_occupation_ignores_the_isco_group_column_when_the_walk_succeeds(index):
    occupation = esco.build_occupation(
        {"conceptUri": OCC + "deep", "iscoGroup": "9999", "preferredLabel": "data steward"},
        index,
    )
    assert occupation["isco_code"] == "2511"


def test_build_occupation_has_no_skills_when_no_relation_points_at_it(index):
    occupation = esco.build_occupation({"conceptUri": OCC + "deep"}, index)
    assert occupation["essential_skills"] == []
    assert occupation["optional_skills"] == []
    assert occupation["title"] == ""
    assert occupation["description"] == ""


def test_build_occupations_skips_rows_without_a_concept_uri(index):
    rows = [{"conceptUri": OCC + "deep"}, {"conceptUri": ""}, {"preferredLabel": "no uri column"}]
    assert [o["uri"] for o in esco.build_occupations(rows, index)] == [OCC + "deep"]


def test_count_usage_counts_every_listing_including_duplicates():
    occupations = [
        {"essential_skills": [{"uri": SKL + "s1"}, {"uri": SKL + "s1"}], "optional_skills": []},
        {"essential_skills": [], "optional_skills": [{"uri": SKL + "s1"}, {"uri": SKL + "s2"}]},
    ]
    essential, optional = esco.count_usage(occupations)
    assert essential[SKL + "s1"] == 2
    assert optional[SKL + "s1"] == 1
    assert optional[SKL + "s2"] == 1


def test_build_skills_keeps_csv_order_and_zeroes_unused_skills():
    tables = {
        "isco": esco.EscoTable("ISCO", None, []),
        "broader": esco.EscoTable("Broader", None, []),
        "relations": esco.EscoTable("Relations", None, []),
        "skills": esco.EscoTable("Skills", "skills.csv", [
            skill_row(SKL + "s2", label="second in the csv"),
            skill_row(SKL + "s1", label="first in the csv"),
        ]),
    }
    index = esco.build_index(tables)
    skills = esco.build_skills(index, {SKL + "s1": 3}, {})
    assert [s["uri"] for s in skills] == [SKL + "s2", SKL + "s1"]
    assert skills[0]["essential_for_count"] == 0
    assert skills[0]["optional_for_count"] == 0
    # NOTE: the counts are plain integers, not the strings a caller might expect.
    assert skills[1]["essential_for_count"] == 3
    assert isinstance(skills[1]["essential_for_count"], int)


def test_build_skills_carries_the_description_that_occupations_omit():
    tables = {
        "isco": esco.EscoTable("ISCO", None, []),
        "broader": esco.EscoTable("Broader", None, []),
        "relations": esco.EscoTable("Relations", None, []),
        "skills": esco.EscoTable("Skills", "skills.csv", [skill_row(SKL + "s1")]),
    }
    skill = esco.build_skills(esco.build_index(tables), {}, {})[0]
    assert skill["description"] == "description of a skill"
    assert list(skill) == [
        "uri", "title", "description", "type", "reuse_level",
        "essential_for_count", "optional_for_count",
    ]


def test_join_tables_counts_usage_from_the_occupations_it_kept():
    occupations, skills = esco.join_tables(esco.load_tables(FIXTURES))
    assert [o["uri"] for o in occupations] == [OCC + f"occ-0{n}" for n in range(1, 7)]
    counts = {s["uri"]: s["essential_for_count"] for s in skills}
    assert counts[SKL + "s10"] == 2
    assert counts[SKL + "s05"] == 0
    assert SKL + "unknown-999" not in counts


def test_join_tables_skips_an_occupation_whose_only_row_has_no_uri():
    tables = esco.load_tables(FIXTURES)
    assert len(tables["occupations"].rows) == 7
    occupations, _ = esco.join_tables(tables)
    assert len(occupations) == 6
