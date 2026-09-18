"""
Build a compact JSON dataset for the Skill Portfolio Analyzer feature.

Reads per-skill scores from data/skill_scores.json and occupation-skill
mappings from data/esco_occupations.json. Computes Jaccard-based adjacency
between occupations, identifies gap skills, and outputs a deduplicated
compact JSON to site/portfolio_data.json.

With --scorer typesafe it reads data/skill_scores_typesafe.json and writes
site/portfolio_data_typesafe.json, leaving the Gemini file untouched.

Usage:
    uv run python build_portfolio_data.py
    uv run python build_portfolio_data.py --scorer typesafe
"""

import json
import os
from collections import defaultdict

from aiisco.jsonio import load_json
from aiisco.portfolio import (
    MIN_JACCARD_OVERLAP,
    PortfolioIndex,
    build_occupations,
    build_short_id_map,
    build_skills,
    collect_skill_info,
    compute_adjacency,
    scored_occupation,
)
from aiisco.rollup import QUADRANTS, index_skill_scores, scorer_suffix

DATA_DIR = "data"
SITE_DIR = "site"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def rationale_of(entry):
    """The rationale of a raw score entry, as a dict to merge into its scores."""
    return {"rationale": entry["rationale"]} if entry.get("rationale") else {}


def load_inputs(suffix):
    """Load the occupations and the per-skill scores of the chosen scorer."""
    occupations = load_json(os.path.join(DATA_DIR, "esco_occupations.json"))
    raw_scores = load_json(os.path.join(DATA_DIR, f"skill_scores{suffix}.json"))
    print(f"  Occupations loaded: {len(occupations)}")
    print(f"  Skill scores loaded: {len(raw_scores)}")
    skill_scores = index_skill_scores(raw_scores, rationale_of)
    print(f"  Skills with valid scores: {len(skill_scores)}")
    return occupations, skill_scores


def load_narratives():
    """Occupation narratives by URI; the dataset is built without them too."""
    path = os.path.join(DATA_DIR, "occupation_narratives.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping narratives")
        return {}
    narratives = {entry["uri"]: entry for entry in load_json(path) if entry.get("uri")}
    print(f"  Occupation narratives loaded: {len(narratives)}")
    return narratives


def build_index(occupations, skill_scores):
    """Collect the skills, their short IDs and the narratives to look up later."""
    narratives = load_narratives()
    skill_info = collect_skill_info(occupations, skill_scores)
    uri_to_short = build_short_id_map(set(skill_info))
    print(f"  Unique skills across occupations: {len(skill_info)}")
    print(f"  Short IDs generated: {len(uri_to_short)}")
    return PortfolioIndex(skill_info=skill_info, uri_to_short=uri_to_short,
                          narratives=narratives)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def score_occupations(occupations, skill_scores):
    """Roll every occupation up, dropping those without a single scored skill."""
    scored = [scored_occupation(occ, skill_scores) for occ in occupations]
    occ_data = [occupation for occupation in scored if occupation is not None]
    print(f"  Occupations with scored skills: {len(occ_data)}")
    return occ_data


def build_adjacency(occ_data):
    """Relate occupations by the essential skills they have in common."""
    print("\nComputing Jaccard adjacency (essential skills only)...")
    adjacency = compute_adjacency([occupation.essential_uris for occupation in occ_data])
    edges = sum(len(pairs) for pairs in adjacency.values()) // 2
    print(f"  Adjacency pairs (>= {MIN_JACCARD_OVERLAP} overlap): {edges}")
    return adjacency


def build_dataset(occ_data, adjacency, index):
    """The whole compact dataset: the skills map and the occupation entries."""
    skills = build_skills(occ_data, index)
    print(f"\nSkills in output: {len(skills)}")
    return {"skills": skills,
            "occupations": build_occupations(occ_data, adjacency, index)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_dataset(dataset, suffix):
    """Write the dataset as one compact line, the way the browser fetches it."""
    out_path = os.path.join(SITE_DIR, f"portfolio_data{suffix}.json")
    os.makedirs(SITE_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, separators=(",", ":"))
    return out_path


def report_counts(out_path, dataset, narratives):
    """Print the size of what was written and how much of it carries scores."""
    skills, occupations = dataset["skills"], dataset["occupations"]
    file_size = os.path.getsize(out_path)
    print(f"\n{'=' * 60}")
    print("Output summary")
    print(f"{'=' * 60}")
    print(f"  Output file: {out_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Skills: {len(skills)}")
    print(f"  Occupations: {len(occupations)}")
    print(f"  Skills with scores: {count(skills.values(), lambda s: s['a'] is not None)}"
          f" / {len(skills)}")
    print(f"  Skills with rationales: {count(skills.values(), lambda s: s.get('r'))}"
          f" / {len(skills)}")
    if narratives:
        print(f"  Occupations with narratives: {count(occupations, lambda o: o.get('n'))}"
              f" / {len(occupations)}")


def count(items, matches):
    """How many items the predicate holds for."""
    return sum(1 for item in items if matches(item))


def report_adjacency(occupations):
    """Print how many neighbours the occupations ended up with."""
    counts = [len(occ["adj"]) for occ in occupations]
    print(f"  Occupations with adjacency: {count(counts, bool)} / {len(occupations)}")
    if counts:
        print(f"  Adjacent occupations per entry: "
              f"avg={sum(counts)/len(counts):.1f}  "
              f"min={min(counts)}  max={max(counts)}")


def report_quadrants(occupations):
    """Print how the occupations are spread over the four quadrants."""
    counts = defaultdict(int)
    for occ in occupations:
        counts[occ["q"]] += 1
    print("\n  Quadrant distribution:")
    for quadrant in QUADRANTS:
        found = counts.get(quadrant, 0)
        pct = found / len(occupations) * 100 if occupations else 0
        print(f"    {quadrant:12s}: {found:4d} ({pct:5.1f}%)")


def report(out_path, dataset, index):
    """Print the summary the operator reads to sanity-check a run."""
    report_counts(out_path, dataset, index.narratives)
    report_adjacency(dataset["occupations"])
    report_quadrants(dataset["occupations"])
    print("\nDone.")


def main():
    """Build the portfolio dataset for the scorer named on the command line."""
    suffix = scorer_suffix(__doc__.split("\n\n")[0])
    print("Portfolio data builder")
    print("=" * 60)
    occupations, skill_scores = load_inputs(suffix)
    index = build_index(occupations, skill_scores)
    occ_data = score_occupations(occupations, skill_scores)
    adjacency = build_adjacency(occ_data)
    dataset = build_dataset(occ_data, adjacency, index)
    report(write_dataset(dataset, suffix), dataset, index)


if __name__ == "__main__":
    main()
