"""
Build a compact JSON dataset for the Skill Portfolio Analyzer feature.

Reads per-skill scores from data/skill_scores.json and occupation-skill
mappings from data/esco_occupations.json. Computes Jaccard-based adjacency
between occupations, identifies gap skills, and outputs a deduplicated
compact JSON to site/portfolio_data.json.

With --scorer typesafe it reads data/skill_scores_typesafe.json and writes
site/portfolio_data_typesafe.json, leaving the Gemini file untouched. That
scorer returns numbers only, so each skill borrows the Gemini rationale, marked
with who wrote it and for which scores ("rf"). --scorer v2 is the same in that
respect, and additionally carries the skill classes, the four shares and the
occupation types of docs/scoring-v2.md.

The dataset stays published, but no page fetches it: fourteen megabytes to show
one job is not a page. For the two scorers the site's switch offers, the run also
writes what the pages actually read - site/jobs/units.json, one
site/jobs/<unit group>.json per ISCO unit group, and the rationales sharded into
site/skill_notes/.

Usage:
    uv run python build_portfolio_data.py
    uv run python build_portfolio_data.py --scorer typesafe
    uv run python build_portfolio_data.py --scorer v2
"""

import json
import os
from collections import defaultdict

from aiisco import site_indexes as ix
from aiisco import v2
from aiisco.jsonio import load_json, write_shard_files
from aiisco.portfolio import (
    MIN_JACCARD_OVERLAP,
    PortfolioIndex,
    build_occupations,
    build_short_id_map,
    build_skills,
    build_unit_shards,
    build_units,
    collect_skill_info,
    compute_adjacency,
    scored_occupation,
)
from aiisco.rollup import (
    QUADRANTS,
    SCORER_SUFFIX,
    index_skill_scores,
    parse_scorer,
)

DATA_DIR = "data"
SITE_DIR = "site"
JOBS_DIR = "jobs"
NOTES_DIR = "skill_notes"
UNITS_FILE = "units.json"
PUBLISHED_SCORER = "gemini"  # the scorer behind the unsuffixed files; it writes rationales
V2 = "v2"
#: The scorers the site's "Scores from" switch offers, and so the only ones whose
#: per-unit job files and rationale shards are worth writing.
SHARDED_SCORERS = (PUBLISHED_SCORER, V2)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def rationale_of(entry):
    """The rationale of a raw score entry, as a dict to merge into its scores."""
    return {"rationale": entry["rationale"]} if entry.get("rationale") else {}


def published_rationales(suffix):
    """The published scorer's entries that carry a rationale, by skill URI.

    Empty for the published scorer itself, which has nothing to borrow.
    """
    path = os.path.join(DATA_DIR, "skill_scores.json")
    if not suffix or not os.path.exists(path):
        return {}
    published = index_skill_scores(load_json(path), rationale_of)
    return {uri: entry for uri, entry in published.items() if "rationale" in entry}


def written_for(entry):
    """Who wrote a borrowed rationale, and the scores they wrote it for."""
    return {"scorer": PUBLISHED_SCORER,
            "automation_risk": entry["automation_risk"],
            "amplification_potential": entry["amplification_potential"]}


def borrow_rationales(skill_scores, published):
    """Give skills without a rationale the published one, marked as borrowed.

    A scorer that returns numbers only writes no text. The page shows the
    borrowed text with a note saying which model wrote it, and for which scores.
    """
    lacking = [uri for uri, scores in skill_scores.items()
               if uri in published and "rationale" not in scores]
    for uri in lacking:
        skill_scores[uri].update(rationale=published[uri]["rationale"],
                                 rationale_from=written_for(published[uri]))
    if published:
        print(f"  Rationales borrowed from the {PUBLISHED_SCORER} scores: {len(lacking)}")


def kept_per_skill(scorer):
    """What index_skill_scores keeps per skill beyond the two display scores."""
    if scorer == V2:
        return lambda entry: {**rationale_of(entry), **v2.skill_extra(entry)}
    return rationale_of


def dataset_meta(scorer, raw_scores):
    """The scheme and the model a share-scheme dataset carries at its top level.

    build_site_indexes.py reads the portfolio file and nothing else, so the
    scheme and the model that produced it have to travel with it.
    """
    if scorer != V2:
        return {}
    models = sorted({entry["model"] for entry in raw_scores if entry.get("model")})
    return {"scheme": ix.SCHEME_SHARES, "model": ", ".join(models)}


def load_inputs(suffix, scorer):
    """Load the occupations and the per-skill scores of the chosen scorer."""
    occupations = load_json(os.path.join(DATA_DIR, "esco_occupations.json"))
    raw_scores = load_json(os.path.join(DATA_DIR, f"skill_scores{suffix}.json"))
    print(f"  Occupations loaded: {len(occupations)}")
    print(f"  Skill scores loaded: {len(raw_scores)}")
    skill_scores = index_skill_scores(raw_scores, kept_per_skill(scorer))
    print(f"  Skills with valid scores: {len(skill_scores)}")
    borrow_rationales(skill_scores, published_rationales(suffix))
    return occupations, skill_scores, dataset_meta(scorer, raw_scores)


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


def build_dataset(occ_data, adjacency, index, meta):
    """The whole compact dataset: the skills map, the occupations, and how to read them."""
    skills = build_skills(occ_data, index)
    print(f"\nSkills in output: {len(skills)}")
    return {"skills": skills,
            "occupations": build_occupations(occ_data, adjacency, index), **meta}


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


def write_units(dataset):
    """Write the slug -> unit group map every page starts from; it has no suffix."""
    path = os.path.join(SITE_DIR, JOBS_DIR, UNITS_FILE)
    payload = write_shard_files(os.path.join(SITE_DIR, JOBS_DIR),
                                {UNITS_FILE: build_units(dataset["occupations"])})[0]
    return path, len(payload), ix.gzipped_size(payload)


def write_named_shards(directory, files):
    """Write a family of shards and measure what each one costs a browser."""
    payloads = write_shard_files(os.path.join(SITE_DIR, directory), files)
    return [len(payload) for payload in payloads], \
        [ix.gzipped_size(payload) for payload in payloads]


def write_job_shards(dataset, suffix):
    """One file per ISCO unit group, each carrying its jobs and their neighbours."""
    shards = build_unit_shards(dataset)
    return write_named_shards(JOBS_DIR, {f"{unit}{suffix}.json": shard
                                         for unit, shard in shards.items()})


def write_note_shards(dataset, suffix):
    """The rationales, sharded so one skill costs one small file."""
    shards = ix.build_skill_notes(dataset["skills"])
    return write_named_shards(NOTES_DIR, {f"{name}{suffix}.json": shard
                                          for name, shard in shards.items()})


def write_page_files(dataset, scorer, suffix):
    """Write everything the pages fetch; return a line about each family written.

    Only the scorers the site's switch offers get them: nothing loads the
    `_typesafe` set, so sharding it would be 426 files nobody fetches.
    """
    if scorer not in SHARDED_SCORERS:
        return []
    path, raw, packed = write_units(dataset)
    return [f"  {path}: {raw:,} bytes, {packed:,} gzipped",
            "  " + ix.shard_report(f"jobs{suffix}", *write_job_shards(dataset, suffix)),
            "  " + ix.shard_report(f"skill_notes{suffix}",
                                   *write_note_shards(dataset, suffix))]


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


def report_classes(occupations, codes, heading, width):
    """Print how the occupations are spread over a fixed set of class codes."""
    counts = defaultdict(int)
    for occ in occupations:
        counts[occ["q"]] += 1
    print(f"\n  {heading}:")
    for code in codes:
        found = counts.get(code, 0)
        pct = found / len(occupations) * 100 if occupations else 0
        print(f"    {code:{width}s}: {found:4d} ({pct:5.1f}%)")


def report(out_path, dataset, index, pages=()):
    """Print the summary the operator reads to sanity-check a run."""
    report_counts(out_path, dataset, index.narratives)
    for line in pages:
        print(line)
    report_adjacency(dataset["occupations"])
    if dataset.get("scheme") == ix.SCHEME_SHARES:
        report_classes(dataset["occupations"], v2.TYPE_ORDER, "Type distribution", 20)
    else:
        report_classes(dataset["occupations"], QUADRANTS, "Quadrant distribution", 12)
    print("\nDone.")


def main():
    """Build the portfolio dataset for the scorer named on the command line."""
    scorer = parse_scorer(__doc__.split("\n\n")[0])
    suffix = SCORER_SUFFIX[scorer]
    print("Portfolio data builder")
    print("=" * 60)
    occupations, skill_scores, meta = load_inputs(suffix, scorer)
    index = build_index(occupations, skill_scores)
    occ_data = score_occupations(occupations, skill_scores)
    adjacency = build_adjacency(occ_data)
    dataset = build_dataset(occ_data, adjacency, index, meta)
    out_path = write_dataset(dataset, suffix)
    pages = write_page_files(dataset, scorer, suffix)
    report(out_path, dataset, index, pages)


if __name__ == "__main__":
    main()
