"""The compact portfolio dataset: short skill IDs, adjacency and gap skills.

The Skill Portfolio Analyzer ships one JSON file to the browser, so every skill
is referenced by a short md5 prefix instead of its ESCO URI and every record uses
one- or two-letter keys. This module holds those shapes and the pure computation
behind them; build_portfolio_data.py does the reading, writing and reporting.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from aiisco.rollup import (
    assign_quadrant,
    evolution_potential,
    get_major_group,
    get_sub_major_group,
    scored_skills,
    slugify,
    weighted_averages,
)

MIN_JACCARD_OVERLAP = 0.15
MAX_ADJACENT = 8
MAX_GAP_SKILLS = 5

SHORT_ID_LENGTH = 8
MAX_SHORT_ID_LENGTH = 32

NARRATIVE_FIELDS = (
    ("evolution_story", "story", bool),
    ("time_savings_pct", "ts", lambda value: value is not None),
    ("automated_tasks", "auto", bool),
    ("amplified_capabilities", "amp", bool),
    ("ai_tools_applicable", "tools", bool),
    ("rebalanced_week", "week", bool),
    ("timeline", "tl", bool),
    ("advice", "adv", bool),
)


@dataclass
class ScoredOccupation:
    """An occupation with at least one scored skill, plus its rolled-up scores."""

    occ: dict
    essential_uris: set
    all_uris: set
    auto: float
    amp: float
    evolution: float
    quadrant: str


@dataclass
class PortfolioIndex:
    """What the compact records are looked up in while they are built."""

    skill_info: dict
    uri_to_short: dict
    narratives: dict


# ---------------------------------------------------------------------------
# Short ID registry (handles collisions)
# ---------------------------------------------------------------------------

def skill_short_id(uri, length=SHORT_ID_LENGTH):
    """Generate a short ID from a skill URI using an md5 hash prefix."""
    return hashlib.md5(uri.encode()).hexdigest()[:length]


def unused_short_id(uri, id_to_uri):
    """The shortest md5 prefix of `uri` that no other URI has claimed."""
    for length in range(SHORT_ID_LENGTH, MAX_SHORT_ID_LENGTH + 1):
        short = skill_short_id(uri, length)
        existing = id_to_uri.get(short)
        if existing is None or existing == uri:
            return short
    raise ValueError(f"Cannot resolve collision for {uri}")


def remap_collider(uri, length, id_to_uri, uri_to_id):
    """Move the URI that forced `uri` to lengthen onto a prefix of the same length."""
    collider = id_to_uri.pop(skill_short_id(uri), None)
    if collider and collider != uri:
        longer = skill_short_id(collider, length)
        id_to_uri[longer] = collider
        uri_to_id[collider] = longer


def build_short_id_map(all_uris):
    """Build a mapping from skill URI to collision-free short ID.

    Uses the first 8 characters of the md5 hash; when two URIs share that prefix
    both of them move to a longer one.
    """
    id_to_uri = {}
    uri_to_id = {}
    for uri in sorted(all_uris):
        short = unused_short_id(uri, id_to_uri)
        if len(short) > SHORT_ID_LENGTH:
            remap_collider(uri, len(short), id_to_uri, uri_to_id)
        id_to_uri[short] = uri
        uri_to_id[uri] = short
    return uri_to_id


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def scored_occupation(occ, skill_scores):
    """Roll one occupation up, or None when not one of its skills is scored."""
    contributions = [(score["automation_risk"], score["amplification_potential"], weight)
                     for _, score, weight, _ in
                     scored_skills(occ, lambda skill: skill_scores.get(skill["uri"]))]
    averages = weighted_averages(contributions)
    if averages is None:
        return None
    auto, amp = averages
    essential = [skill["uri"] for skill in occ.get("essential_skills", [])]
    optional = [skill["uri"] for skill in occ.get("optional_skills", [])]
    return ScoredOccupation(occ=occ, essential_uris=set(essential),
                            all_uris=set(essential) | set(optional), auto=auto, amp=amp,
                            evolution=evolution_potential(auto, amp),
                            quadrant=assign_quadrant(auto, amp))


def skill_record(skill, scores):
    """One skill's title and scores, with None scores when it was never scored."""
    record = {"title": skill.get("title", ""),
              "automation_risk": scores["automation_risk"] if scores else None,
              "amplification_potential": scores["amplification_potential"] if scores else None}
    for key in ("rationale", "rationale_from"):
        if scores and scores.get(key):
            record[key] = scores[key]
    return record


def collect_skill_info(occupations, skill_scores):
    """Title and scores per skill URI, keeping the first title each URI appears with."""
    info = {}
    for occ in occupations:
        for skill in occ.get("essential_skills", []) + occ.get("optional_skills", []):
            if skill["uri"] not in info:
                info[skill["uri"]] = skill_record(skill, skill_scores.get(skill["uri"]))
    return info


# ---------------------------------------------------------------------------
# Adjacency computation
# ---------------------------------------------------------------------------

def jaccard(left, right):
    """Overlap of two skill sets, at least one of which is not empty."""
    return len(left & right) / len(left | right)


def invert_essential(essential_sets):
    """Map each essential skill URI to the indices of the occupations needing it."""
    by_skill = defaultdict(set)
    for i, skills in enumerate(essential_sets):
        for uri in skills:
            by_skill[uri].add(i)
    return by_skill


def unpaired_candidates(i, skills, by_skill, seen):
    """Occupations sharing a skill with `i` that have not been paired with it yet."""
    candidates = set()
    for uri in skills:
        candidates.update(by_skill[uri])
    candidates.discard(i)
    fresh = [j for j in candidates if (i, j) not in seen]
    seen.update((i, j) for j in fresh)
    seen.update((j, i) for j in fresh)
    return fresh


def compute_adjacency(essential_sets):
    """Jaccard similarity between occupations sharing essential skills.

    Returns a dict: occupation index -> list of (other index, jaccard score).
    Only pairs with jaccard >= MIN_JACCARD_OVERLAP are included.
    """
    by_skill = invert_essential(essential_sets)
    adjacency = defaultdict(list)
    seen = set()
    for i, skills in enumerate(essential_sets):
        if not skills:
            continue
        for j in unpaired_candidates(i, skills, by_skill, seen):
            overlap = jaccard(skills, essential_sets[j])
            if overlap >= MIN_JACCARD_OVERLAP:
                adjacency[i].append((j, overlap))
                adjacency[j].append((i, overlap))
    return adjacency


# ---------------------------------------------------------------------------
# Compact records
# ---------------------------------------------------------------------------

def top_neighbours(occupation, neighbours, occ_data):
    """The adjacent occupations with higher evolution potential, best first."""
    better = [(j, overlap) for j, overlap in neighbours
              if occ_data[j].evolution > occupation.evolution]
    better.sort(key=lambda pair: -occ_data[pair[0]].evolution)
    return better[:MAX_ADJACENT]


def amplification_rank(uri, skill_info):
    """Ranking score for a gap skill; skills that were never scored sort last."""
    amp = skill_info.get(uri, {}).get("amplification_potential")
    return amp if amp is not None else -1


def gap_skill_ids(gap_uris, index):
    """Short IDs of the skills the adjacent occupation needs and this one lacks."""
    ranked = sorted(gap_uris, key=lambda uri: -amplification_rank(uri, index.skill_info))
    return [index.uri_to_short[uri] for uri in ranked[:MAX_GAP_SKILLS]]


def adjacent_entry(occupation, other, overlap, index):
    """One adjacency row: where to move next and which skills are missing for it."""
    return {
        "s": slugify(other.occ["title"]),
        "t": other.occ["title"],
        "ov": round(overlap, 2),
        "e": other.evolution,
        "q": other.quadrant,
        "gap": gap_skill_ids(other.all_uris - occupation.all_uris, index),
    }


def compact_narrative(narrative):
    """Short-key copy of a narrative, keeping only the fields that are filled in."""
    compact = {}
    for source, short, keep in NARRATIVE_FIELDS:
        value = (narrative or {}).get(source)
        if keep(value):
            compact[short] = value
    return compact


def occupation_entry(occupation, index):
    """The compact record the site loads for one occupation."""
    occ = occupation.occ
    hierarchy = occ.get("hierarchy", [])
    return {
        "t": occ["title"],
        "s": slugify(occ["title"]),
        "c": occ.get("isco_code", ""),
        "cat": get_sub_major_group(hierarchy),
        "mg": get_major_group(hierarchy),
        "q": occupation.quadrant,
        "e": occupation.evolution,
        "ar": occupation.auto,
        "ap": occupation.amp,
        "se": [index.uri_to_short[s["uri"]] for s in occ.get("essential_skills", [])],
        "so": [index.uri_to_short[s["uri"]] for s in occ.get("optional_skills", [])],
    }


def build_occupation(occupation, neighbours, occ_data, index):
    """One occupation entry with its adjacency list and its narrative."""
    entry = occupation_entry(occupation, index)
    entry["adj"] = [adjacent_entry(occupation, occ_data[j], overlap, index)
                    for j, overlap in top_neighbours(occupation, neighbours, occ_data)]
    narrative = compact_narrative(index.narratives.get(occupation.occ.get("uri", "")))
    if narrative:
        entry["n"] = narrative
    return entry


def build_occupations(occ_data, adjacency, index):
    """The occupations array of the portfolio dataset."""
    return [build_occupation(occupation, adjacency.get(i, []), occ_data, index)
            for i, occupation in enumerate(occ_data)]


def rounded(score):
    """One-decimal score, or None when the skill was never scored."""
    return None if score is None else round(score, 1)


def rationale_source(source):
    """Compact form of who wrote a borrowed rationale and for which scores."""
    return {"s": source["scorer"],
            "a": rounded(source["automation_risk"]),
            "m": rounded(source["amplification_potential"])}


def skill_entry(info):
    """The compact record for one skill: title, both scores, rationale if any.

    "rf" is present only when the rationale was written by another scorer.
    """
    entry = {"t": info["title"],
             "a": rounded(info["automation_risk"]),
             "m": rounded(info["amplification_potential"])}
    if info.get("rationale"):
        entry["r"] = info["rationale"]
    if info.get("rationale_from"):
        entry["rf"] = rationale_source(info["rationale_from"])
    return entry


def build_skills(occ_data, index):
    """The skills map, holding every skill an included occupation references."""
    referenced = set()
    for occupation in occ_data:
        referenced.update(occupation.all_uris)
    return {index.uri_to_short[uri]: skill_entry(index.skill_info[uri])
            for uri in referenced}
