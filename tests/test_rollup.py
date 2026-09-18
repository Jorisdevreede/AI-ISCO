"""Tests for aiisco/rollup.py, the weighted skill-to-occupation roll-up."""

import pytest

from aiisco.rollup import (
    ESSENTIAL_WEIGHT,
    OPTIONAL_WEIGHT,
    QUADRANT_THRESHOLD,
    assign_quadrant,
    evolution_potential,
    get_major_group,
    get_sub_major_group,
    hierarchy_level,
    index_skill_scores,
    scorable,
    scored_skills,
    scorer_suffix,
    slugify,
    weighted_averages,
)


def skill(uri, title=""):
    return {"uri": uri, "title": title}


@pytest.mark.parametrize("auto, amp, expected", [
    (9.0, 9.0, "TRANSFORM"),
    (9.0, 1.0, "SHRINK"),
    (1.0, 9.0, "EVOLVE"),
    (1.0, 1.0, "STABLE"),
    (QUADRANT_THRESHOLD, QUADRANT_THRESHOLD, "TRANSFORM"),
    (QUADRANT_THRESHOLD, QUADRANT_THRESHOLD - 0.1, "SHRINK"),
    (QUADRANT_THRESHOLD - 0.1, QUADRANT_THRESHOLD, "EVOLVE"),
])
def test_assign_quadrant_splits_on_the_threshold(auto, amp, expected):
    assert assign_quadrant(auto, amp) == expected


def test_evolution_potential_is_the_product_over_ten_at_one_decimal():
    assert evolution_potential(7.5, 9.0) == 6.8
    assert evolution_potential(0.0, 9.0) == 0.0


def test_scored_skills_yields_essential_skills_first_with_their_weight():
    occupation = {"essential_skills": [skill("e1")], "optional_skills": [skill("o1")]}

    yielded = list(scored_skills(occupation, lambda s: s["uri"]))

    assert [(s["uri"], score, weight, relation) for s, score, weight, relation in yielded] == [
        ("e1", "e1", ESSENTIAL_WEIGHT, "essential"),
        ("o1", "o1", OPTIONAL_WEIGHT, "optional"),
    ]


def test_scored_skills_skips_skills_the_lookup_cannot_score():
    occupation = {"essential_skills": [skill("e1"), skill("e2")]}

    yielded = list(scored_skills(occupation, lambda s: None if s["uri"] == "e1" else 1.0))

    assert [s["uri"] for s, _score, _weight, _relation in yielded] == ["e2"]


def test_scored_skills_accepts_an_occupation_without_skill_keys():
    assert list(scored_skills({}, lambda s: 1.0)) == []


def test_weighted_averages_weighs_essential_skills_twice_as_heavily():
    contributions = [(9.0, 1.0, ESSENTIAL_WEIGHT), (3.0, 4.0, OPTIONAL_WEIGHT)]

    assert weighted_averages(contributions) == (7.0, 2.0)


def test_weighted_averages_rounds_to_one_decimal():
    assert weighted_averages([(1.0, 2.0, 1.0), (2.0, 3.0, 2.0)]) == (1.7, 2.7)


def test_weighted_averages_is_none_when_nothing_contributed():
    assert weighted_averages([]) is None


@pytest.mark.parametrize("entry, expected", [
    ({"uri": "u", "automation_risk": 1, "amplification_potential": 2}, True),
    ({"uri": "", "automation_risk": 1, "amplification_potential": 2}, False),
    ({"automation_risk": 1, "amplification_potential": 2}, False),
    ({"uri": "u", "automation_risk": None, "amplification_potential": 2}, False),
    ({"uri": "u", "automation_risk": 1, "amplification_potential": None}, False),
])
def test_scorable_needs_a_uri_and_both_axes(entry, expected):
    assert scorable(entry) is expected


def test_index_skill_scores_keeps_floats_and_drops_unusable_entries():
    entries = [{"uri": "u1", "automation_risk": 8, "amplification_potential": 2},
               {"uri": "", "automation_risk": 1, "amplification_potential": 1}]

    assert index_skill_scores(entries) == {
        "u1": {"automation_risk": 8.0, "amplification_potential": 2.0}}


def test_index_skill_scores_keeps_the_extra_keys_a_caller_asks_for():
    entries = [{"uri": "u1", "title": "t", "automation_risk": 8,
                "amplification_potential": 2}]

    scores = index_skill_scores(entries, lambda e: {"title": e.get("title", "")})

    assert scores["u1"]["title"] == "t"


@pytest.mark.parametrize("hierarchy, index, expected", [
    (["a", "b"], 0, "a"),
    (["a", "b"], 1, "b"),
    (["a", "b"], 2, ""),
    ([], 0, ""),
])
def test_hierarchy_level_falls_back_to_an_empty_label(hierarchy, index, expected):
    assert hierarchy_level(hierarchy, index) == expected


def test_group_labels_fall_back_to_the_broadest_level_available():
    assert get_major_group(["Managers", "Hotel managers"]) == "Managers"
    assert get_sub_major_group(["Managers", "Hotel managers"]) == "Hotel managers"
    assert get_sub_major_group(["Managers"]) == "Managers"
    assert get_sub_major_group([]) == ""
    assert get_sub_major_group(["Managers", ""]) == ""


@pytest.mark.parametrize("title, expected", [
    ("Data entry clerk", "data-entry-clerk"),
    ("Café manager", "caf-manager"),
    ("  Chief (acting) officer!  ", "chief-acting-officer"),
    ("air-traffic  controller", "air-traffic-controller"),
    ("-- edge --", "edge"),
])
def test_slugify_makes_url_friendly_slugs(title, expected):
    assert slugify(title) == expected


@pytest.mark.parametrize("argv, expected", [([], ""), (["--scorer", "gemini"], ""),
                                            (["--scorer", "typesafe"], "_typesafe")])
def test_scorer_suffix_only_suffixes_the_non_default_scorer(argv, expected):
    assert scorer_suffix("description", argv) == expected


def test_scorer_suffix_rejects_an_unknown_scorer(capsys):
    with pytest.raises(SystemExit):
        scorer_suffix("description", ["--scorer", "nope"])
    assert "invalid choice" in capsys.readouterr().err
