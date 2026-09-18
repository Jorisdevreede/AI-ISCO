"""Unit tests for the pure index builders."""

import json
import os

import pytest

from aiisco import site_indexes as ix

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "site_indexes")


@pytest.fixture(scope="module")
def portfolio():
    with open(os.path.join(FIXTURES, "portfolio_data.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def occupations(portfolio):
    return sorted(portfolio["occupations"], key=lambda o: o["s"])


@pytest.fixture(scope="module")
def labels():
    return {"2": "Professionals", "25": "ICT professionals", "251": "Developers",
            "2512": "Software developers", "3": "Technicians"}


# --- encoding -------------------------------------------------------------

def test_encode_compact_has_no_padding_and_keeps_utf8():
    payload = ix.encode_compact({"t": "sommelière", "n": [1, 2]})
    assert payload == '{"t":"sommelière","n":[1,2]}'.encode()


def test_gzipped_size_is_stable_across_calls():
    payload = ix.encode_compact(["a"] * 100)
    assert ix.gzipped_size(payload) == ix.gzipped_size(payload)
    assert 0 < ix.gzipped_size(payload) < len(payload)


# --- text -----------------------------------------------------------------

def test_normalise_lowercases_and_collapses_whitespace():
    assert ix.normalise("  Software   Developer\n") == "software developer"


def test_stems_drops_a_plural_s_only_on_longer_words():
    assert ix.stems("Software Developers gas") == ["software", "developer", "gas"]


@pytest.mark.parametrize("word,expected", [("soft", True), ("developer", True), ("nurse", False)])
def test_covered_matches_title_word_prefixes(word, expected):
    assert ix.covered(word, ["software", "developer"]) is expected


def test_alt_labels_dedupes_drops_the_title_and_sorts_shortest_first():
    raw = "programmer\nsoftware engineer\nprogrammer\nSoftware Developer\nsoftware developers\ncoder"
    assert ix.alt_labels("software developer", raw) == [
        "coder", "programmer", "software engineer",
    ]


def test_alt_labels_of_an_empty_cell_is_empty():
    assert ix.alt_labels("systems analyst", "") == []


def test_labels_by_slug_prefers_the_row_whose_isco_code_matches(occupations):
    esco = {
        "bookkeeper": [("4311", "accounts assistant"), ("3313", "ledger clerk")],
        "software developer": [("2512", "programmer")],
    }
    by_slug = ix.labels_by_slug(occupations, esco)
    assert by_slug["bookkeeper"] == ["ledger clerk"]
    assert by_slug["software-developer"] == ["programmer"]
    assert by_slug["lathe-operator"] == []  # no ESCO row at all


def test_labels_by_slug_falls_back_to_the_first_row_when_no_code_matches(occupations):
    esco = {"bookkeeper": [("9999", "ledger clerk")]}
    assert ix.labels_by_slug(occupations, esco)["bookkeeper"] == ["ledger clerk"]


# --- group keys -----------------------------------------------------------

def test_group_keys_splits_a_code_into_four_levels():
    assert ix.group_keys("2512") == ("major:2", "sub:25", "minor:251", "unit:2512")


@pytest.mark.parametrize("key,parent", [
    ("all", None), ("major:2", "all"), ("sub:25", "major:2"),
    ("minor:251", "sub:25"), ("unit:2512", "minor:251"),
])
def test_parent_of_walks_one_level_up(key, parent):
    assert ix.parent_of(key) == parent


def test_group_members_puts_every_occupation_in_all_and_four_groups(occupations):
    members = ix.group_members(occupations)
    assert len(members["all"]) == 6
    assert [o["s"] for o in members["unit:2512"]] == ["software-developer"]
    assert len(members["major:2"]) == 3


def test_children_of_each_level(occupations):
    keys = set(ix.group_members(occupations))
    assert ix.children_of("all", keys) == ["major:2", "major:3", "major:7"]
    assert ix.children_of("major:2", keys) == ["sub:22", "sub:25"]
    assert ix.children_of("unit:2512", keys) == []


# --- statistics -----------------------------------------------------------

@pytest.mark.parametrize("fraction,expected", [
    (0.0, 1.0), (0.5, 3.0), (1.0, 5.0), (0.10, 1.4), (0.90, 4.6),
])
def test_percentile_interpolates_between_closest_ranks(fraction, expected):
    assert ix.percentile([1.0, 2.0, 3.0, 4.0, 5.0], fraction) == pytest.approx(expected)


def test_percentile_of_an_even_list_blends_the_middle_pair():
    assert ix.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_percentile_of_one_value_is_that_value():
    assert ix.percentile([7.5], 0.9) == 7.5


def test_distribution_rounds_to_one_decimal():
    assert ix.distribution([1.0, 2.0, 3.0, 4.0, 5.0]) == {
        "mean": 3.0, "p10": 1.4, "p50": 3.0, "p90": 4.6,
    }


def test_driving_skills_counts_high_scoring_essential_skills(occupations, portfolio):
    skills = portfolio["skills"]
    auto = ix.driving_skills(occupations, skills, "a")
    assert {"id": "aa000006", "n": 3} in auto          # record patient notes, a=9
    assert all(skills[row["id"]]["a"] >= 6 for row in auto)
    assert len(auto) <= ix.TOP_N


def test_driving_skills_ignores_unknown_and_unscored_skills(occupations):
    assert ix.driving_skills(occupations, {"aa000001": {"a": None}}, "a") == []


def test_extreme_slugs_ranks_by_automation(occupations):
    most, least = ix.extreme_slugs(occupations)
    assert most[0] == "bookkeeper"
    assert least[0] == "nurse-responsible-for-general-care"
    assert len(most) == len(least) == ix.TOP_N


def test_group_label_falls_back_to_the_code(labels):
    assert ix.group_label("all", labels) == "All occupations"
    assert ix.group_label("major:2", labels) == "Professionals"
    assert ix.group_label("unit:7223", labels) == "7223"


def test_build_groups_shapes_one_entry(occupations, labels, portfolio):
    groups = ix.build_groups(occupations, labels, portfolio["skills"])
    entry = groups["major:2"]
    assert entry["label"] == "Professionals"
    assert (entry["level"], entry["code"], entry["n"]) == ("major", "2", 3)
    assert entry["q"] == {"EVOLVE": 1, "TRANSFORM": 2}
    assert entry["auto"]["p50"] == 6.0
    assert entry["children"] == ["sub:22", "sub:25"]
    assert entry["parent"] == "all"
    assert set(entry["skills"]) == {"auto", "amp"}


def test_build_groups_includes_the_whole_dataset(occupations, labels, portfolio):
    entry = ix.build_groups(occupations, labels, portfolio["skills"])["all"]
    assert (entry["level"], entry["code"], entry["n"]) == ("all", "", 6)
    assert entry["parent"] is None
    assert entry["top"][0] == "bookkeeper"


# --- search index ---------------------------------------------------------

def test_search_rows_carry_the_major_group_label(occupations, labels):
    row = next(r for r in ix.search_rows(occupations, labels) if r["s"] == "software-developer")
    assert row == {"t": "software developer", "s": "software-developer", "c": "2512",
                   "mg": "Professionals", "a": 6.9, "m": 7.0, "q": "TRANSFORM", "alt": []}


def test_alt_candidates_order_gives_every_occupation_its_best_label_first(occupations):
    by_slug = {"software-developer": ["coder", "programmer"], "bookkeeper": ["ledger clerk"]}
    ranks = [c[0] for c in ix.alt_candidates(occupations, by_slug)]
    assert ranks == [0, 0, 1]


def test_build_search_index_fills_up_to_the_budget(occupations, labels):
    rows = ix.search_rows(occupations, labels)
    by_slug = {o["s"]: ["alpha", "beta", "gamma"] for o in occupations}
    candidates = ix.alt_candidates(occupations, by_slug)
    index = ix.build_search_index(rows, candidates, 10_000)
    assert all(r["alt"] == ["beta", "alpha", "gamma"] for r in index)  # shortest first


def test_build_search_index_drops_labels_that_do_not_fit(occupations, labels):
    rows = ix.search_rows(occupations, labels)
    by_slug = {o["s"]: [f"{o['s']} viewed as a quite unusual synonym {i}"]
               for i, o in enumerate(occupations)}
    candidates = ix.alt_candidates(occupations, by_slug)
    budget = ix.gzipped_size(ix.encode_compact(rows)) + 60
    index = ix.build_search_index(rows, candidates, budget)
    assert 0 < sum(len(r["alt"]) for r in index) < len(candidates)
    assert ix.gzipped_size(ix.encode_compact(index)) <= budget


# --- skills ---------------------------------------------------------------

def test_skill_counts_counts_each_occupation_once(occupations):
    essential, optional = ix.skill_counts(occupations)
    assert essential["aa000001"] == 1          # listed twice by software-developer
    assert essential["aa000006"] == 3
    assert optional["aa00000c"] == 4


def test_build_skill_index_is_sorted_by_title(portfolio, occupations):
    rows = ix.build_skill_index(portfolio["skills"], ix.skill_counts(occupations))
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
    row = next(r for r in rows if r["id"] == "aa000001")
    assert row == {"id": "aa000001", "t": "write software", "a": 8.0, "m": 7.0,
                   "ne": 1, "no": 0}


def test_build_skill_occupations_lists_slugs_in_order(occupations):
    index = ix.build_skill_occupations(occupations)
    assert index["aa000006"]["e"] == ["bookkeeper", "nurse-assistant",
                                      "nurse-responsible-for-general-care"]
    assert index["aa000001"]["e"] == ["software-developer"]
    assert index["aa000001"]["o"] == []
    assert list(index) == sorted(index)


# --- stats ----------------------------------------------------------------

@pytest.mark.parametrize("auto,amp,near", [
    (6.0, 1.0, True), (5.5, 1.0, True), (6.5, 1.0, True), (5.4, 1.0, False),
    (1.0, 5.9, True), (1.0, 1.0, False),
])
def test_is_near_line_on_either_axis(auto, amp, near):
    assert ix.is_near_line({"ar": auto, "ap": amp}, 6) is near


def test_build_stats_counts_shares_and_the_near_line_band(occupations):
    stats = ix.build_stats(occupations, 12, "2026-09-18", 6)
    assert stats["built"] == "2026-09-18"
    assert stats["threshold"] == 6
    assert stats["occupations"] == 6
    assert stats["skills_scored"] == 12
    assert stats["quadrants"]["counts"] == {"EVOLVE": 1, "SHRINK": 2, "STABLE": 1,
                                            "TRANSFORM": 2}
    assert stats["quadrants"]["shares"]["SHRINK"] == pytest.approx(0.3333, abs=1e-4)
    # systems-analyst (auto 6.0), the nurse (amp 6.0) and nurse-assistant (both) qualify.
    assert stats["near_line"] == {"count": 3, "share": 0.5}


def test_budgets_cover_every_output():
    assert set(ix.BUDGET_GZ_KB) == {"search_index", "groups", "stats",
                                    "skill_index", "skill_occupations"}
