"""
Aggregate skill-level scores to occupation-level scores and build site data.

Reads per-skill automation_risk and amplification_potential from
data/skill_scores.json, looks up which skills belong to each occupation
via data/esco_occupations.json, and computes weighted averages
(essential skills weight 2.0, optional skills weight 1.0).

Outputs:
  - data/occupation_scores.json  (full occupation-level scores)
  - data/site_data.json          (compact format for treemap frontend)
  - site/data.json               (copy of site_data.json served by the frontend)

With --scorer typesafe it reads data/skill_scores_typesafe.json and writes the
same three files with a _typesafe suffix, leaving the Gemini files untouched.
With --scorer v2 it reads the scoring v2 answers, rolls each occupation up into
the four skill-class shares of docs/scoring-v2.md, and puts its type code in the
quadrant field.

Usage:
    uv run python aggregate_scores.py
    uv run python aggregate_scores.py --scorer typesafe
    uv run python aggregate_scores.py --scorer v2
"""

import os
import shutil
from collections import Counter, defaultdict

from aiisco import v2
from aiisco.jsonio import load_json, write_json
from aiisco.rollup import (
    ESSENTIAL_WEIGHT,  # noqa: F401  re-exported: compare_skill_scores.py imports it here
    OPTIONAL_WEIGHT,  # noqa: F401
    QUADRANT_THRESHOLD,  # noqa: F401
    QUADRANTS,
    SCORER_SUFFIX,
    assign_quadrant,
    evolution_potential,
    get_sub_major_group,
    hierarchy_level,
    index_skill_scores,
    parse_scorer,
    scored_skills,
    slugify,
    weighted_averages,
    weighted_mean,
)

DATA_DIR = "data"
SITE_DIR = "site"

V2 = "v2"

QUADRANT_LABEL_WIDTH = 12
TYPE_LABEL_WIDTH = 20

TOP_SKILLS_PER_AXIS = 5
TOP_SITE_SKILLS = 10
TOP_LIST_SIZE = 10

TOP_LISTS = (
    ("highest evolution potential", "evolution_potential",
     (("evol", "evolution_potential"), ("auto", "automation_risk"),
      ("amp", "amplification_potential"))),
    ("highest automation risk", "automation_risk",
     (("auto", "automation_risk"), ("amp", "amplification_potential"))),
    ("highest amplification potential", "amplification_potential",
     (("amp", "amplification_potential"), ("auto", "automation_risk"))),
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def kept_per_skill(scorer):
    """What index_skill_scores keeps per skill beyond the two display scores."""
    if scorer == V2:
        return lambda entry: {"title": entry.get("title", ""), **v2.skill_extra(entry)}
    return lambda entry: {"title": entry.get("title", "")}


def load_inputs(suffix, scorer):
    """Load the occupations and the per-skill scores of the chosen scorer."""
    occupations = load_json(os.path.join(DATA_DIR, "esco_occupations.json"))
    skills_meta = load_json(os.path.join(DATA_DIR, "esco_skills.json"))
    raw_scores = load_json(os.path.join(DATA_DIR, f"skill_scores{suffix}.json"))
    print(f"  Occupations: {len(occupations)}")
    print(f"  Skills (metadata): {len(skills_meta)}")
    print(f"  Skills (scored): {len(raw_scores)}")
    skill_scores = index_skill_scores(raw_scores, kept_per_skill(scorer))
    print(f"  Skills with valid scores: {len(skill_scores)}")
    return occupations, skill_scores


# ---------------------------------------------------------------------------
# Occupation-level scores
# ---------------------------------------------------------------------------

def skill_rows(occ, skill_scores):
    """One row per scored skill of the occupation, essential skills first."""
    rows = []
    for skill, score, weight, relation in scored_skills(
            occ, lambda s: skill_scores.get(s.get("uri", ""))):
        rows.append({"title": score["title"] or skill.get("title", ""),
                     "auto": score["automation_risk"],
                     "amp": score["amplification_potential"],
                     "weight": weight,
                     "relation": relation,
                     "score": score})
    return rows


def top_skills(rows, axis, limit=TOP_SKILLS_PER_AXIS):
    """The highest-scoring skills on one axis, highest first."""
    ranked = [{"title": row["title"], "score": row[axis], "relation": row["relation"]}
              for row in rows]
    ranked.sort(key=lambda entry: -entry["score"])
    return ranked[:limit]


def v2_block(rows, comp_key=v2.COMP_KEY):
    """The scoring v2 roll-up of an occupation, from the rows of its scored skills."""
    return v2.roll_up([v2.contribution(row["score"], row["weight"], comp_key)
                       for row in rows])


def v2_fields(rows):
    """The v2 additions to an occupation record; its type replaces the quadrant."""
    block = v2_block(rows)
    return {
        "mechanical_automation": weighted_mean(
            [(row["score"]["mechanical_automation"], row["weight"]) for row in rows]),
        "shares": block["shares"],
        "mu": block["mu"],
        "sigma": block["sigma"],
        "why_insulated": block["why_insulated"],
        "near_line": block["near_line"],
        "quadrant": block["type"],
    }


def occupation_record(occ, skill_scores, scorer):
    """Full occupation-level record, or None when no skill of it was scored."""
    rows = skill_rows(occ, skill_scores)
    averages = weighted_averages([(r["auto"], r["amp"], r["weight"]) for r in rows])
    if averages is None:
        return None
    auto_avg, amp_avg = averages
    record = {
        "uri": occ.get("uri", ""),
        "title": occ.get("title", ""),
        "isco_code": occ.get("isco_code", ""),
        "isco_group": occ.get("isco_group", ""),
        "hierarchy": occ.get("hierarchy", []),
        "automation_risk": auto_avg,
        "amplification_potential": amp_avg,
        "evolution_potential": evolution_potential(auto_avg, amp_avg),
        "quadrant": assign_quadrant(auto_avg, amp_avg),
        "num_skills_scored": len(rows),
        "num_skills_total": (len(occ.get("essential_skills", []))
                             + len(occ.get("optional_skills", []))),
        "top_automated_skills": top_skills(rows, "auto"),
        "top_amplified_skills": top_skills(rows, "amp"),
    }
    return record if scorer != V2 else {**record, **v2_fields(rows)}


def aggregate(occupations, skill_scores, scorer):
    """Occupation records for every occupation with at least one scored skill."""
    records = [occupation_record(occ, skill_scores, scorer) for occ in occupations]
    results = [record for record in records if record is not None]
    print(f"\nAggregated scores for {len(results)} occupations")
    return results


# ---------------------------------------------------------------------------
# Site data
# ---------------------------------------------------------------------------

def merged_top_skills(occ):
    """Both top lists merged into one, each skill carrying both of its scores."""
    auto_by_title = {s["title"]: s["score"] for s in occ["top_automated_skills"]}
    amp_by_title = {s["title"]: s["score"] for s in occ["top_amplified_skills"]}
    merged = {}
    for skill in occ["top_automated_skills"] + occ["top_amplified_skills"]:
        title = skill["title"]
        if title not in merged:
            merged[title] = {"title": title, "auto": auto_by_title.get(title),
                             "amp": amp_by_title.get(title)}
        if len(merged) >= TOP_SITE_SKILLS:
            break
    return list(merged.values())


def site_shares(occ):
    """The share-scheme fields the site data carries, empty for the older scorers."""
    if "shares" not in occ:
        return {}
    return {"mechanical_automation": occ["mechanical_automation"],
            "shares": occ["shares"]}


def site_record(occ, source):
    """The compact record the treemap frontend reads for one occupation."""
    hierarchy = source.get("hierarchy", [])
    return {**site_base(occ, source, hierarchy), **site_shares(occ)}


def site_base(occ, source, hierarchy):
    """The fields every scorer's site data carries for one occupation."""
    return {
        "title": occ["title"],
        "slug": slugify(occ["title"]),
        "category": get_sub_major_group(occ["hierarchy"]),
        "major_group": hierarchy_level(hierarchy, 0),
        "sub_major_group": hierarchy_level(hierarchy, 1),
        "minor_group": hierarchy_level(hierarchy, 2),
        "unit_group": hierarchy_level(hierarchy, 3),
        "isco_code": occ["isco_code"],
        "automation_risk": occ["automation_risk"],
        "amplification_potential": occ["amplification_potential"],
        "evolution_potential": occ["evolution_potential"],
        "quadrant": occ["quadrant"],
        "num_skills": occ["num_skills_total"],
        "num_essential_skills": len(source["essential_skills"]),
        "top_skills": merged_top_skills(occ),
    }


def build_site_data(occ_results, occupations):
    """Compact records for the frontend, one per aggregated occupation."""
    by_uri = {occ["uri"]: occ for occ in occupations}
    return [site_record(occ, by_uri[occ["uri"]]) for occ in occ_results]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(occ_results, site_data, suffix):
    """Write both data files and publish the site data where the frontend reads it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    occ_path = os.path.join(DATA_DIR, f"occupation_scores{suffix}.json")
    write_json(occ_path, occ_results)
    print(f"Wrote {len(occ_results)} occupations to {occ_path}")
    site_path = os.path.join(DATA_DIR, f"site_data{suffix}.json")
    write_json(site_path, site_data)
    print(f"Wrote {len(site_data)} occupations to {site_path}")
    publish(site_path, suffix)


def publish(site_path, suffix):
    """Copy the site data into site/, the file the live frontend fetches."""
    os.makedirs(SITE_DIR, exist_ok=True)
    frontend_path = os.path.join(SITE_DIR, f"data{suffix}.json")
    shutil.copy(site_path, frontend_path)
    print(f"Copied to {frontend_path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def distribution_rows(counts, codes, total, width):
    """One printed line per code of a distribution, with its bar."""
    for code in codes:
        found = counts.get(code, 0)
        pct = found / total * 100 if total else 0
        yield f"  {code:{width}s}: {found:4d} ({pct:5.1f}%) {'#' * int(pct / 2)}"


def print_quadrant_distribution(occ_results):
    """Print how many occupations fell into each quadrant."""
    counts = defaultdict(int)
    for occ in occ_results:
        counts[occ["quadrant"]] += 1
    print("\nDistribution by quadrant:")
    for line in distribution_rows(counts, QUADRANTS, len(occ_results),
                                  QUADRANT_LABEL_WIDTH):
        print(line)


def occupation_types(occupations, skill_scores, comp_key):
    """The type of every occupation with a scored skill, at one complementarity cut-off."""
    rows = (skill_rows(occ, skill_scores) for occ in occupations)
    return [v2_block(row, comp_key)["type"] for row in rows if row]


def print_type_distributions(occupations, skill_scores):
    """Print the seven types under both complementarity cut-offs.

    The published classes read complementarity at COMP_LEVEL; the stricter
    alternative is printed beside them so the choice can be revisited from an
    existing run rather than by scoring 13,939 skills again.
    """
    for level, comp_key in sorted(v2.COMP_KEYS.items()):
        types = occupation_types(occupations, skill_scores, comp_key)
        note = " (published)" if level == v2.COMP_LEVEL else ""
        print(f"\nDistribution by type, COMP_LEVEL = {level}{note}:")
        for line in distribution_rows(Counter(types), v2.TYPE_ORDER, len(types),
                                      TYPE_LABEL_WIDTH):
            print(line)


def average(group, key):
    """Mean of one score over a group of occupations."""
    return sum(occ[key] for occ in group) / len(group)


def major_group_row(major, group):
    """One row of the table of average scores per ISCO major group."""
    label = group[0]["hierarchy"][0] if group[0]["hierarchy"] else ""
    return (f"  {major:6s} {len(group):5d} {average(group, 'automation_risk'):6.1f} "
            f"{average(group, 'amplification_potential'):6.1f} "
            f"{average(group, 'evolution_potential'):6.1f}  {label}")


def print_major_group_averages(occ_results):
    """Print the average scores per 1-digit ISCO major group."""
    groups = defaultdict(list)
    for occ in occ_results:
        groups[occ["isco_code"][0] if occ["isco_code"] else "?"].append(occ)
    print("\nAverage scores by ISCO major group:")
    print(f"  {'Group':6s} {'Count':>5s} {'Auto':>6s} {'Amp':>6s} {'Evol':>6s}  Label")
    print(f"  {'-'*5:6s} {'-'*5:>5s} {'-'*5:>6s} {'-'*5:>6s} {'-'*5:>6s}  {'-'*20}")
    for major in sorted(groups.keys()):
        print(major_group_row(major, groups[major]))


def ranked_by(occ_results, key):
    """The occupations sorted by one score, highest first."""
    return sorted(occ_results, key=lambda occ: -occ[key])


def top_list_row(rank, occ, fields):
    """One row of a top-10 table, showing the fields that table leads with."""
    scores = "  ".join(f"{label}={occ[field]:4.1f}" for label, field in fields)
    return f"  {rank:2d}. {occ['title'][:50]:52s} {scores}  [{occ['quadrant']}]"


def print_top_lists(occ_results):
    """Print the three top-10 tables, one per axis."""
    for heading, key, fields in TOP_LISTS:
        print(f"\nTop {TOP_LIST_SIZE} {heading}:")
        for rank, occ in enumerate(ranked_by(occ_results, key)[:TOP_LIST_SIZE], 1):
            print(top_list_row(rank, occ, fields))


def print_summary(occ_results, occupations, skill_scores, scorer):
    """Print the summary the operator reads to sanity-check a run."""
    print(f"\n{'='*60}")
    print("Summary statistics")
    print(f"{'='*60}")
    if scorer == V2:
        print_type_distributions(occupations, skill_scores)
    else:
        print_quadrant_distribution(occ_results)
    print_major_group_averages(occ_results)
    print_top_lists(occ_results)
    print("\nDone.")


def main():
    """Aggregate to occupation level for the scorer named on the command line."""
    scorer = parse_scorer(__doc__.split("\n\n")[0])
    suffix = SCORER_SUFFIX[scorer]
    print("Occupation score aggregation")
    print("=" * 60)
    occupations, skill_scores = load_inputs(suffix, scorer)
    occ_results = aggregate(occupations, skill_scores, scorer)
    write_outputs(occ_results, build_site_data(occ_results, occupations), suffix)
    print_summary(occ_results, occupations, skill_scores, scorer)


if __name__ == "__main__":
    main()
