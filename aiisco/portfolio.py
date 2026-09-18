"""The compact portfolio dataset: short skill IDs, adjacency and gap skills.

The Skill Portfolio Analyzer ships one JSON file to the browser, so every skill
is referenced by a short md5 prefix instead of its ESCO URI and every record uses
one- or two-letter keys. This module holds those shapes and the pure computation
behind them; build_portfolio_data.py does the reading, writing and reporting.
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from aiisco import v2
from aiisco.rollup import (
    assign_quadrant,
    evolution_potential,
    get_major_group,
    get_sub_major_group,
    scored_skills,
    slugify,
    weighted_averages,
    weighted_mean,
)

MIN_JACCARD_OVERLAP = 0.15
MAX_ADJACENT = 8
MAX_GAP_SKILLS = 5

#: The per-skill fields a compact record carries when the scorer supplies them.
OPTIONAL_SKILL_KEYS = ("rationale", "rationale_from", "mechanical_automation",
                       "class", "sub", "comp", "mech")

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
    """An occupation with at least one scored skill, plus its rolled-up scores.

    `v2` holds the share-scheme roll-up when the scorer supplies one, and is None
    for the scorers whose classes are quadrants.
    """

    occ: dict
    essential_uris: set
    all_uris: set
    auto: float
    amp: float
    evolution: float
    quadrant: str
    v2: dict = None


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

def weighted_scores(occ, skill_scores):
    """(score, weight) for every skill of the occupation the scores can reach."""
    return [(score, weight) for _, score, weight, _ in
            scored_skills(occ, lambda skill: skill_scores.get(skill["uri"]))]


def v2_block(rows):
    """The share-scheme roll-up of an occupation, or None for an older scorer."""
    if not any("class" in score for score, _ in rows):
        return None
    block = v2.roll_up([v2.contribution(score, weight) for score, weight in rows])
    block["mechanical"] = weighted_mean(
        [(score["mechanical_automation"], weight) for score, weight in rows])
    return block


def scored_occupation(occ, skill_scores):
    """Roll one occupation up, or None when not one of its skills is scored."""
    rows = weighted_scores(occ, skill_scores)
    averages = weighted_averages(
        [(score["automation_risk"], score["amplification_potential"], weight)
         for score, weight in rows])
    if averages is None:
        return None
    return rolled_up(occ, rows, averages)


def rolled_up(occ, rows, averages):
    """The scored occupation, taking its class from the v2 block when there is one."""
    auto, amp = averages
    block = v2_block(rows)
    essential = {skill["uri"] for skill in occ.get("essential_skills", [])}
    optional = {skill["uri"] for skill in occ.get("optional_skills", [])}
    return ScoredOccupation(
        occ=occ, essential_uris=essential, all_uris=essential | optional,
        auto=auto, amp=amp, evolution=evolution_potential(auto, amp),
        quadrant=block["type"] if block else assign_quadrant(auto, amp), v2=block)


def kept_fields(scores):
    """The optional per-skill fields a record carries when the scorer filled them in."""
    if not scores:
        return {}
    return {key: scores[key] for key in OPTIONAL_SKILL_KEYS
            if scores.get(key) is not None}


def skill_record(skill, scores):
    """One skill's title and scores, with None scores when it was never scored."""
    record = {"title": skill.get("title", ""),
              "automation_risk": scores["automation_risk"] if scores else None,
              "amplification_potential": scores["amplification_potential"] if scores else None}
    record.update(kept_fields(scores))
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
    """Short IDs of the skills the adjacent occupation needs and this one lacks.

    `gap_uris` is a set, so the URI breaks ties: without it, two skills with the
    same amplification would swap places between runs over identical input.
    """
    ranked = sorted(gap_uris,
                    key=lambda uri: (-amplification_rank(uri, index.skill_info), uri))
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


def v2_occupation_fields(block):
    """The share-scheme additions to a compact occupation record."""
    if not block:
        return {}
    return {"ak": block["mechanical"], "sh": v2.share_list(block["shares"]),
            "nl": block["near_line"], "why": block["why_insulated"]}


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
        **v2_occupation_fields(occupation.v2),
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


def v2_skill_fields(info):
    """The share-scheme additions to a compact skill record: machine score, class, probabilities."""
    if "class" not in info:
        return {}
    return {"k": rounded(info["mechanical_automation"]), "c": info["class"],
            "p": [round(info[key], 2) for key in ("sub", "comp", "mech")]}


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
    entry.update(v2_skill_fields(info))
    return entry


def build_skills(occ_data, index):
    """The skills map, holding every skill an included occupation references."""
    referenced = set()
    for occupation in occ_data:
        referenced.update(occupation.all_uris)
    return {index.uri_to_short[uri]: skill_entry(index.skill_info[uri])
            for uri in referenced}


# ---------------------------------------------------------------------------
# One file per ISCO unit group
# ---------------------------------------------------------------------------
# The whole dataset is 14 MB, and a page that shows one job needs one job, its
# unit group and the jobs it is adjacent to. Slicing it by unit group turns a
# 14 MB fetch into a 10-40 KB one, and `units.json` is the small map that says
# which slice a slug lives in.

UNIT_CODE = re.compile(r"^\d{4}$")


def in_a_unit(occupation):
    """True when an occupation carries a four-digit ISCO unit group to live in.

    Every occupation of the real ESCO export does. One without a code cannot be
    reached through `jobs/<unit>` at all, so it is left out of the map rather than
    given a file whose name is empty.
    """
    return bool(UNIT_CODE.match(occupation.get("c", "")))


def build_units(occupations):
    """Occupation slug -> its four-digit ISCO unit group."""
    return {occupation["s"]: occupation["c"]
            for occupation in sorted(occupations, key=lambda o: o["s"])
            if in_a_unit(occupation)}


def occupations_by_unit(occupations):
    """The occupation records of the dataset, grouped by their unit group."""
    units = defaultdict(list)
    for occupation in occupations:
        if in_a_unit(occupation):
            units[occupation["c"]].append(occupation)
    return units


def neighbours_of(members, by_slug):
    """The full records of the adjacent occupations that live in another unit.

    ``members`` are one unit group's occupations, so the comparison is against
    that group's code rather than against the members themselves: a job adjacent
    to a job in its own unit is already in the shard.
    """
    unit = members[0]["c"]
    wanted = {adjacent["s"] for occupation in members
              for adjacent in occupation.get("adj", [])}
    return [by_slug[slug] for slug in sorted(wanted)
            if slug in by_slug and by_slug[slug]["c"] != unit]


def referenced_skills(records, skills):
    """The skills map restricted to what these occupation records reference."""
    used = set()
    for record in records:
        used.update(record.get("se", []) + record.get("so", []))
    return {sid: skills[sid] for sid in sorted(used) if sid in skills}


def unit_shard(members, neighbours, skills, meta):
    """One unit group's slice of the dataset, shaped like the dataset itself."""
    return {"skills": referenced_skills(members + neighbours, skills),
            "occupations": members, **meta, "neighbours": neighbours}


def build_unit_shards(dataset):
    """Unit group code -> the slice of the dataset a job page needs for it."""
    by_slug = {occupation["s"]: occupation for occupation in dataset["occupations"]}
    meta = {key: dataset[key] for key in ("scheme", "model") if key in dataset}
    units = occupations_by_unit(dataset["occupations"])
    return {unit: unit_shard(units[unit], neighbours_of(units[unit], by_slug),
                             dataset["skills"], meta)
            for unit in sorted(units)}
