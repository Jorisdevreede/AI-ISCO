"""Tests for aiisco/portfolio.py, the compact portfolio dataset shapes."""

import hashlib

import pytest

from aiisco import portfolio
from aiisco.portfolio import (
    MAX_ADJACENT,
    MAX_GAP_SKILLS,
    PortfolioIndex,
    ScoredOccupation,
    amplification_rank,
    build_occupations,
    build_short_id_map,
    build_skills,
    collect_skill_info,
    compact_narrative,
    compute_adjacency,
    gap_skill_ids,
    invert_essential,
    jaccard,
    rounded,
    scored_occupation,
    skill_entry,
    skill_record,
    skill_short_id,
    top_neighbours,
    unpaired_candidates,
)

# Two URIs whose md5 hashes share their first eight hex characters.
COLLIDING = ("http://example.org/esco/skill/c28695",
             "http://example.org/esco/skill/c52431")


def occupation(title, essential=(), optional=(), evolution=1.0, **kwargs):
    """A ScoredOccupation shaped like the ones build_portfolio_data.py makes."""
    occ = {"title": title, "uri": f"urn:{title}",
           "essential_skills": [{"uri": uri} for uri in essential],
           "optional_skills": [{"uri": uri} for uri in optional]}
    occ.update(kwargs.pop("occ", {}))
    return ScoredOccupation(occ=occ, essential_uris=set(essential),
                            all_uris=set(essential) | set(optional),
                            auto=kwargs.get("auto", 5.0), amp=kwargs.get("amp", 5.0),
                            evolution=evolution, quadrant=kwargs.get("quadrant", "STABLE"))


def index(skill_info=None, uris=(), narratives=None):
    """A PortfolioIndex whose short IDs are just the URIs themselves."""
    return PortfolioIndex(skill_info=skill_info or {},
                          uri_to_short={uri: uri for uri in uris},
                          narratives=narratives or {})


# ---------------------------------------------------------------------------
# Short IDs
# ---------------------------------------------------------------------------

def test_skill_short_id_is_an_md5_prefix():
    uri = "http://example.org/esco/skill/a1"

    assert skill_short_id(uri) == hashlib.md5(uri.encode()).hexdigest()[:8]
    assert len(skill_short_id(uri)) == 8
    assert len(skill_short_id(uri, 12)) == 12


def test_short_id_map_uses_eight_characters_when_nothing_collides():
    uris = {"http://example.org/esco/skill/a1", "http://example.org/esco/skill/a2"}

    ids = build_short_id_map(uris)

    assert sorted(ids) == sorted(uris)
    assert all(len(short) == 8 for short in ids.values())
    assert len(set(ids.values())) == 2


def test_short_id_map_lengthens_both_sides_of_a_collision():
    first, second = COLLIDING
    assert skill_short_id(first) == skill_short_id(second)

    ids = build_short_id_map({first, second})

    assert ids[first] == skill_short_id(first, 9)
    assert ids[second] == skill_short_id(second, 9)
    assert ids[first] != ids[second]


def test_a_collision_does_not_disturb_the_other_skills():
    others = {"http://example.org/esco/skill/a1", "http://example.org/esco/skill/a2"}

    ids = build_short_id_map(set(COLLIDING) | others)

    assert len(set(ids.values())) == 4
    assert all(len(ids[uri]) == 8 for uri in others)


def test_an_unresolvable_collision_is_an_error(monkeypatch):
    monkeypatch.setattr(portfolio, "skill_short_id", lambda uri, length=8: "same")

    with pytest.raises(ValueError, match="Cannot resolve collision"):
        build_short_id_map({"a", "b"})


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

def test_scored_occupation_weighs_essential_skills_twice_as_heavily():
    scores = {"e1": {"automation_risk": 9.0, "amplification_potential": 1.0},
              "o1": {"automation_risk": 3.0, "amplification_potential": 4.0}}
    occ = {"title": "T", "essential_skills": [{"uri": "e1"}],
           "optional_skills": [{"uri": "o1"}]}

    result = scored_occupation(occ, scores)

    assert (result.auto, result.amp) == (7.0, 2.0)
    assert result.evolution == 1.4
    assert result.quadrant == "SHRINK"
    assert result.essential_uris == {"e1"}
    assert result.all_uris == {"e1", "o1"}


def test_scored_occupation_is_none_without_a_single_scored_skill():
    occ = {"title": "T", "essential_skills": [{"uri": "e1"}]}

    assert scored_occupation(occ, {}) is None


def test_skill_record_keeps_null_scores_for_unscored_skills():
    assert skill_record({"uri": "u", "title": "t"}, None) == {
        "title": "t", "automation_risk": None, "amplification_potential": None}


def test_skill_record_keeps_a_rationale_when_there_is_one():
    scores = {"automation_risk": 1.0, "amplification_potential": 2.0,
              "rationale": "because"}

    assert skill_record({"uri": "u"}, scores) == {
        "title": "", "automation_risk": 1.0, "amplification_potential": 2.0,
        "rationale": "because"}


def test_collect_skill_info_keeps_the_first_title_a_uri_appears_with():
    occupations = [{"essential_skills": [{"uri": "u", "title": "first"}]},
                   {"optional_skills": [{"uri": "u", "title": "second"}]}]

    info = collect_skill_info(occupations, {})

    assert info["u"]["title"] == "first"


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def test_jaccard_is_the_overlap_over_the_union():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == 1.0


def test_invert_essential_lists_the_occupations_needing_each_skill():
    assert invert_essential([{"a"}, {"a", "b"}]) == {"a": {0, 1}, "b": {1}}


def test_unpaired_candidates_reports_every_pair_only_once():
    by_skill = invert_essential([{"a"}, {"a"}])
    seen = set()

    first = unpaired_candidates(0, {"a"}, by_skill, seen)
    second = unpaired_candidates(1, {"a"}, by_skill, seen)

    assert first == [1]
    assert second == []


def test_compute_adjacency_keeps_pairs_above_the_overlap_threshold():
    shared = {f"s{i}" for i in range(10)}
    adjacency = compute_adjacency([shared, shared | {"x"}, {"only-mine"}])

    assert sorted(adjacency) == [0, 1]
    assert adjacency[0][0][0] == 1
    assert adjacency[0][0][1] == pytest.approx(10 / 11)
    assert adjacency[1][0][0] == 0


def test_compute_adjacency_drops_pairs_that_barely_overlap():
    left = {f"s{i}" for i in range(10)}
    right = {"s0"} | {f"other{i}" for i in range(10)}

    assert compute_adjacency([left, right]) == {}


def test_compute_adjacency_skips_occupations_without_essential_skills():
    assert compute_adjacency([set(), {"a"}]) == {}


# ---------------------------------------------------------------------------
# Compact records
# ---------------------------------------------------------------------------

def test_top_neighbours_keeps_only_the_better_ones_best_first():
    here = occupation("here", evolution=3.0)
    occ_data = [here, occupation("worse", evolution=1.0),
                occupation("better", evolution=4.0), occupation("best", evolution=9.0)]

    chosen = top_neighbours(here, [(1, 0.9), (2, 0.5), (3, 0.4)], occ_data)

    assert chosen == [(3, 0.4), (2, 0.5)]


def test_top_neighbours_is_capped(monkeypatch):
    here = occupation("here", evolution=0.0)
    occ_data = [here] + [occupation(f"n{i}", evolution=float(i + 1))
                         for i in range(MAX_ADJACENT + 3)]
    neighbours = [(i, 0.5) for i in range(1, len(occ_data))]

    assert len(top_neighbours(here, neighbours, occ_data)) == MAX_ADJACENT


def test_amplification_rank_sorts_unscored_skills_last():
    skill_info = {"scored": {"amplification_potential": 4.0}, "unscored": {}}

    assert amplification_rank("scored", skill_info) == 4.0
    assert amplification_rank("unscored", skill_info) == -1
    assert amplification_rank("unknown", skill_info) == -1


def test_gap_skill_ids_are_the_most_amplifiable_ones_first():
    uris = [f"u{i}" for i in range(MAX_GAP_SKILLS + 2)]
    skill_info = {uri: {"amplification_potential": float(i)}
                  for i, uri in enumerate(uris)}

    chosen = gap_skill_ids(set(uris), index(skill_info, uris))

    assert chosen == list(reversed(uris))[:MAX_GAP_SKILLS]


@pytest.mark.parametrize("narrative, expected", [
    (None, {}),
    ({}, {}),
    ({"evolution_story": "", "advice": ""}, {}),
    ({"time_savings_pct": 0}, {"ts": 0}),
    ({"evolution_story": "s", "advice": "a", "timeline": ["t"]},
     {"story": "s", "tl": ["t"], "adv": "a"}),
])
def test_compact_narrative_keeps_only_filled_in_fields(narrative, expected):
    compact = compact_narrative(narrative)

    assert compact == expected
    assert list(compact) == list(expected)


def test_rounded_keeps_none_for_unscored_skills():
    assert rounded(None) is None
    assert rounded(2.349) == 2.3


def test_skill_entry_only_carries_a_rationale_when_there_is_one():
    assert skill_entry({"title": "t", "automation_risk": 1.25,
                        "amplification_potential": None}) == \
        {"t": "t", "a": 1.2, "m": None}
    assert skill_entry({"title": "t", "automation_risk": 1.0,
                        "amplification_potential": 2.0, "rationale": "r"})["r"] == "r"


def test_build_skills_only_holds_referenced_skills():
    skill_info = {"used": {"title": "used", "automation_risk": 1.0,
                           "amplification_potential": 2.0},
                  "unused": {"title": "unused", "automation_risk": None,
                             "amplification_potential": None}}

    skills = build_skills([occupation("here", essential=["used"])],
                          index(skill_info, ["used", "unused"]))

    assert list(skills) == ["used"]


def test_build_occupations_adds_adjacency_and_narrative_last():
    here = occupation("here", essential=["a"], evolution=1.0)
    there = occupation("there", essential=["a", "b"], evolution=5.0)
    skill_info = {"a": {"amplification_potential": 1.0}, "b": {"amplification_potential": 9.0}}
    built = build_occupations(
        [here, there], {0: [(1, 0.5)]},
        index(skill_info, ["a", "b"], {"urn:here": {"advice": "move"}}))

    assert list(built[0]) == ["t", "s", "c", "cat", "mg", "q", "e", "ar", "ap",
                              "se", "so", "adj", "n"]
    assert built[0]["adj"][0]["t"] == "there"
    assert built[0]["adj"][0]["gap"] == ["b"]
    assert built[0]["n"] == {"adv": "move"}
    assert "n" not in built[1]
    assert built[1]["adj"] == []
