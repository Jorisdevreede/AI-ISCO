"""
Aggregate skill-level scores to occupation-level scores and build site data.

Reads per-skill automation_risk and amplification_potential from
data/skill_scores.json, looks up which skills belong to each occupation
via data/esco_occupations.json, and computes weighted averages
(essential skills weight 2.0, optional skills weight 1.0).

Outputs:
  - data/occupation_scores.json  (full occupation-level scores)
  - data/site_data.json          (compact format for treemap frontend)

Usage:
    uv run python aggregate_scores.py
"""

import json
import os
import re
import sys
from collections import defaultdict

DATA_DIR = "data"

ESSENTIAL_WEIGHT = 2.0
OPTIONAL_WEIGHT = 1.0
QUADRANT_THRESHOLD = 6


def slugify(title):
    """Convert a title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")


def load_json(filename):
    """Load a JSON file from DATA_DIR."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assign_quadrant(auto, amp):
    """Assign quadrant based on automation_risk and amplification_potential."""
    if auto >= QUADRANT_THRESHOLD and amp >= QUADRANT_THRESHOLD:
        return "TRANSFORM"
    elif auto >= QUADRANT_THRESHOLD and amp < QUADRANT_THRESHOLD:
        return "SHRINK"
    elif auto < QUADRANT_THRESHOLD and amp >= QUADRANT_THRESHOLD:
        return "EVOLVE"
    else:
        return "STABLE"


def get_sub_major_group(isco_code, hierarchy):
    """Extract the ISCO sub-major group (2-digit level) label.

    The hierarchy list goes from broadest to most specific.
    The sub-major group is the 2-digit level, which is typically
    the second element in the hierarchy (after the 1-digit major group).
    Falls back to the first hierarchy element or the isco_group.
    """
    # Try to find a 2-digit group from hierarchy
    if len(hierarchy) >= 2:
        return hierarchy[1]
    if len(hierarchy) >= 1:
        return hierarchy[0]
    return ""


def main():
    print("Occupation score aggregation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    occupations = load_json("esco_occupations.json")
    skills_meta = load_json("esco_skills.json")
    skill_scores_list = load_json("skill_scores.json")

    print(f"  Occupations: {len(occupations)}")
    print(f"  Skills (metadata): {len(skills_meta)}")
    print(f"  Skills (scored): {len(skill_scores_list)}")

    # Build skill_scores lookup: uri -> {automation_risk, amplification_potential}
    skill_scores = {}
    for s in skill_scores_list:
        uri = s.get("uri", "")
        if uri and s.get("automation_risk") is not None and s.get("amplification_potential") is not None:
            skill_scores[uri] = {
                "automation_risk": float(s["automation_risk"]),
                "amplification_potential": float(s["amplification_potential"]),
                "title": s.get("title", ""),
            }

    print(f"  Skills with valid scores: {len(skill_scores)}")

    # ------------------------------------------------------------------
    # 2. Aggregate to occupation level
    # ------------------------------------------------------------------
    occ_results = []

    for occ in occupations:
        uri = occ.get("uri", "")
        title = occ.get("title", "")
        isco_code = occ.get("isco_code", "")
        isco_group = occ.get("isco_group", "")
        hierarchy = occ.get("hierarchy", [])

        essential_skills = occ.get("essential_skills", [])
        optional_skills = occ.get("optional_skills", [])
        num_skills_total = len(essential_skills) + len(optional_skills)

        # Collect weighted scores
        auto_weighted_sum = 0.0
        amp_weighted_sum = 0.0
        total_weight = 0.0
        num_scored = 0

        # Track individual skill scores for top-N lists
        skill_auto_scores = []
        skill_amp_scores = []

        for skill in essential_skills:
            skill_uri = skill.get("uri", "")
            if skill_uri in skill_scores:
                sc = skill_scores[skill_uri]
                auto_weighted_sum += sc["automation_risk"] * ESSENTIAL_WEIGHT
                amp_weighted_sum += sc["amplification_potential"] * ESSENTIAL_WEIGHT
                total_weight += ESSENTIAL_WEIGHT
                num_scored += 1
                skill_auto_scores.append({
                    "title": sc["title"] or skill.get("title", ""),
                    "score": sc["automation_risk"],
                    "relation": "essential",
                })
                skill_amp_scores.append({
                    "title": sc["title"] or skill.get("title", ""),
                    "score": sc["amplification_potential"],
                    "relation": "essential",
                })

        for skill in optional_skills:
            skill_uri = skill.get("uri", "")
            if skill_uri in skill_scores:
                sc = skill_scores[skill_uri]
                auto_weighted_sum += sc["automation_risk"] * OPTIONAL_WEIGHT
                amp_weighted_sum += sc["amplification_potential"] * OPTIONAL_WEIGHT
                total_weight += OPTIONAL_WEIGHT
                num_scored += 1
                skill_auto_scores.append({
                    "title": sc["title"] or skill.get("title", ""),
                    "score": sc["automation_risk"],
                    "relation": "optional",
                })
                skill_amp_scores.append({
                    "title": sc["title"] or skill.get("title", ""),
                    "score": sc["amplification_potential"],
                    "relation": "optional",
                })

        if total_weight == 0:
            continue

        auto_avg = round(auto_weighted_sum / total_weight, 1)
        amp_avg = round(amp_weighted_sum / total_weight, 1)
        evolution = round((auto_avg * amp_avg) / 10, 1)
        quadrant = assign_quadrant(auto_avg, amp_avg)

        # Top 5 skills by automation risk and amplification potential
        skill_auto_scores.sort(key=lambda x: -x["score"])
        skill_amp_scores.sort(key=lambda x: -x["score"])
        top_automated = skill_auto_scores[:5]
        top_amplified = skill_amp_scores[:5]

        occ_results.append({
            "uri": uri,
            "title": title,
            "isco_code": isco_code,
            "isco_group": isco_group,
            "hierarchy": hierarchy,
            "automation_risk": auto_avg,
            "amplification_potential": amp_avg,
            "evolution_potential": evolution,
            "quadrant": quadrant,
            "num_skills_scored": num_scored,
            "num_skills_total": num_skills_total,
            "top_automated_skills": top_automated,
            "top_amplified_skills": top_amplified,
        })

    print(f"\nAggregated scores for {len(occ_results)} occupations")

    # ------------------------------------------------------------------
    # 3. Write occupation_scores.json
    # ------------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    occ_path = os.path.join(DATA_DIR, "occupation_scores.json")
    with open(occ_path, "w", encoding="utf-8") as f:
        json.dump(occ_results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(occ_results)} occupations to {occ_path}")

    # ------------------------------------------------------------------
    # 4. Build and write site_data.json
    # ------------------------------------------------------------------
    # Pre-build lookup for essential skill counts
    occ_by_uri = {o["uri"]: o for o in occupations}

    site_data = []
    for occ in occ_results:
        category = get_sub_major_group(occ["isco_code"], occ["hierarchy"])

        # Merge top skills into a combined top list
        auto_by_title = {s["title"]: s["score"] for s in occ["top_automated_skills"]}
        amp_by_title = {s["title"]: s["score"] for s in occ["top_amplified_skills"]}

        seen = set()
        top_skills = []
        for s in occ["top_automated_skills"] + occ["top_amplified_skills"]:
            if s["title"] not in seen:
                seen.add(s["title"])
                top_skills.append({
                    "title": s["title"],
                    "auto": auto_by_title.get(s["title"]),
                    "amp": amp_by_title.get(s["title"]),
                })
            if len(top_skills) >= 10:
                break

        orig = occ_by_uri.get(occ["uri"])
        num_essential = len(orig["essential_skills"]) if orig else 0

        hierarchy = orig.get("hierarchy", []) if orig else []
        site_data.append({
            "title": occ["title"],
            "slug": slugify(occ["title"]),
            "category": category,
            "major_group": hierarchy[0] if len(hierarchy) >= 1 else "",
            "sub_major_group": hierarchy[1] if len(hierarchy) >= 2 else "",
            "minor_group": hierarchy[2] if len(hierarchy) >= 3 else "",
            "unit_group": hierarchy[3] if len(hierarchy) >= 4 else "",
            "isco_code": occ["isco_code"],
            "automation_risk": occ["automation_risk"],
            "amplification_potential": occ["amplification_potential"],
            "evolution_potential": occ["evolution_potential"],
            "quadrant": occ["quadrant"],
            "num_skills": occ["num_skills_total"],
            "num_essential_skills": num_essential,
            "top_skills": top_skills,
        })

    site_path = os.path.join(DATA_DIR, "site_data.json")
    with open(site_path, "w", encoding="utf-8") as f:
        json.dump(site_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(site_data)} occupations to {site_path}")

    # Also copy to site/data.json for the frontend
    import shutil
    os.makedirs("site", exist_ok=True)
    shutil.copy(site_path, os.path.join("site", "data.json"))
    print(f"Copied to site/data.json")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary statistics")
    print(f"{'='*60}")

    # Distribution by quadrant
    quadrant_counts = defaultdict(int)
    for occ in occ_results:
        quadrant_counts[occ["quadrant"]] += 1
    print("\nDistribution by quadrant:")
    for q in ["TRANSFORM", "SHRINK", "EVOLVE", "STABLE"]:
        count = quadrant_counts.get(q, 0)
        pct = count / len(occ_results) * 100 if occ_results else 0
        bar = "#" * int(pct / 2)
        print(f"  {q:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Average scores by ISCO major group (1-digit)
    major_groups = defaultdict(list)
    for occ in occ_results:
        code = occ["isco_code"]
        if code:
            major = code[0]
        else:
            major = "?"
        major_groups[major].append(occ)

    print("\nAverage scores by ISCO major group:")
    print(f"  {'Group':6s} {'Count':>5s} {'Auto':>6s} {'Amp':>6s} {'Evol':>6s}  Label")
    print(f"  {'-'*5:6s} {'-'*5:>5s} {'-'*5:>6s} {'-'*5:>6s} {'-'*5:>6s}  {'-'*20}")
    for major in sorted(major_groups.keys()):
        group = major_groups[major]
        avg_auto = sum(o["automation_risk"] for o in group) / len(group)
        avg_amp = sum(o["amplification_potential"] for o in group) / len(group)
        avg_evol = sum(o["evolution_potential"] for o in group) / len(group)
        # Get a representative label from hierarchy
        label = ""
        if group[0]["hierarchy"]:
            label = group[0]["hierarchy"][0]
        print(f"  {major:6s} {len(group):5d} {avg_auto:6.1f} {avg_amp:6.1f} {avg_evol:6.1f}  {label}")

    # Top 10 by evolution potential
    sorted_by_evol = sorted(occ_results, key=lambda x: -x["evolution_potential"])
    print("\nTop 10 highest evolution potential:")
    for i, occ in enumerate(sorted_by_evol[:10], 1):
        print(f"  {i:2d}. {occ['title'][:50]:52s} "
              f"evol={occ['evolution_potential']:4.1f}  "
              f"auto={occ['automation_risk']:4.1f}  "
              f"amp={occ['amplification_potential']:4.1f}  "
              f"[{occ['quadrant']}]")

    # Top 10 by automation risk
    sorted_by_auto = sorted(occ_results, key=lambda x: -x["automation_risk"])
    print("\nTop 10 highest automation risk:")
    for i, occ in enumerate(sorted_by_auto[:10], 1):
        print(f"  {i:2d}. {occ['title'][:50]:52s} "
              f"auto={occ['automation_risk']:4.1f}  "
              f"amp={occ['amplification_potential']:4.1f}  "
              f"[{occ['quadrant']}]")

    # Top 10 by amplification potential
    sorted_by_amp = sorted(occ_results, key=lambda x: -x["amplification_potential"])
    print("\nTop 10 highest amplification potential:")
    for i, occ in enumerate(sorted_by_amp[:10], 1):
        print(f"  {i:2d}. {occ['title'][:50]:52s} "
              f"amp={occ['amplification_potential']:4.1f}  "
              f"auto={occ['automation_risk']:4.1f}  "
              f"[{occ['quadrant']}]")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
