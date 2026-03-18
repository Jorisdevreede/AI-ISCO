"""
Ingest ESCO v1.2 CSV data and produce enriched JSON files.

Reads CSV files from data/esco/, joins occupations with skills and
the ISCO hierarchy, and outputs:
  - data/esco_occupations.json  (occupations with attached skills)
  - data/esco_skills.json       (skills with usage counts)

Usage:
    uv run python ingest_esco.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

DATA_DIR = "data/esco"
OUT_DIR = "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_csv(name_hints):
    """Find a CSV in DATA_DIR matching any of the *name_hints* substrings.

    Handles common naming variants: occupations_en.csv, occupations.csv, etc.
    Returns the first match or None.
    """
    try:
        files = os.listdir(DATA_DIR)
    except FileNotFoundError:
        return None
    for hint in name_hints:
        # Exact match first
        for f in files:
            if f.lower() == hint.lower():
                return os.path.join(DATA_DIR, f)
        # Substring match (e.g. "occupations" matches "occupations_en.csv")
        for f in files:
            if hint.lower() in f.lower() and f.lower().endswith(".csv"):
                return os.path.join(DATA_DIR, f)
    return None


def read_csv(path):
    """Read a CSV file and return a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_read(name_hints, label):
    """Find and read a CSV, printing status. Returns rows or empty list."""
    path = find_csv(name_hints)
    if path is None:
        print(f"  WARNING: Could not find {label} CSV in {DATA_DIR}/")
        return []
    rows = read_csv(path)
    print(f"  {label}: {path} ({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_isco_lookup(isco_rows):
    """Build uri -> {code, label} lookup for ISCO groups."""
    lookup = {}
    for row in isco_rows:
        uri = row.get("conceptUri", "")
        code = row.get("code", "")
        label = row.get("preferredLabel", "")
        lookup[uri] = {"code": code, "label": label}
    return lookup


def build_broader_map(broader_rows):
    """Build child_uri -> parent_uri map from broaderRelationsOccPillar."""
    parent_of = {}
    for row in broader_rows:
        child = row.get("conceptUri", "")
        parent = row.get("broaderUri", "")
        if child and parent:
            parent_of[child] = parent
    return parent_of


def resolve_hierarchy(uri, parent_of, isco_lookup):
    """Walk up the broader-relation chain to build the ISCO hierarchy list.

    Returns (isco_code, isco_group_label, hierarchy_list) where
    hierarchy_list goes from the broadest group to the most specific.
    """
    chain = []
    current = uri
    visited = set()
    while current in parent_of and current not in visited:
        visited.add(current)
        current = parent_of[current]
        if current in isco_lookup:
            chain.append(isco_lookup[current]["label"])
    chain.reverse()

    # The most specific ISCO group is the direct parent (or grandparent)
    # that has a code in the ISCO lookup.
    isco_code = ""
    isco_group = ""
    current = uri
    visited = set()
    while current in parent_of and current not in visited:
        visited.add(current)
        current = parent_of[current]
        if current in isco_lookup:
            info = isco_lookup[current]
            if info["code"]:
                isco_code = info["code"]
                isco_group = info["label"]
                break

    return isco_code, isco_group, chain


def build_skill_relations(relation_rows):
    """Parse occupationSkillRelations into per-occupation essential/optional skill URI sets."""
    occ_essential = defaultdict(list)
    occ_optional = defaultdict(list)
    for row in relation_rows:
        occ_uri = row.get("occupationUri", "")
        skill_uri = row.get("skillUri", "")
        rel_type = row.get("relationType", "").lower()
        if not occ_uri or not skill_uri:
            continue
        if "essential" in rel_type:
            occ_essential[occ_uri].append(skill_uri)
        else:
            occ_optional[occ_uri].append(skill_uri)
    return occ_essential, occ_optional


def build_skill_lookup(skill_rows):
    """Build uri -> skill info dict."""
    lookup = {}
    for row in skill_rows:
        uri = row.get("conceptUri", "")
        skill_type_raw = row.get("skillType", "")
        # Normalise: the CSV may say "skill/competence" or "knowledge" or similar
        if "knowledge" in skill_type_raw.lower():
            skill_type = "knowledge"
        else:
            skill_type = "skill"
        lookup[uri] = {
            "uri": uri,
            "title": row.get("preferredLabel", ""),
            "description": row.get("description", ""),
            "type": skill_type,
            "reuse_level": row.get("reuseLevel", ""),
        }
    return lookup


def main():
    print("ESCO data ingestion")
    print("=" * 60)

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: {DATA_DIR}/ directory not found.")
        sys.exit(1)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_DIR}/")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files in {DATA_DIR}/")
    print(f"Loading data...\n")

    # ------------------------------------------------------------------
    # 1. Load CSVs
    # ------------------------------------------------------------------
    occ_rows = safe_read(
        ["occupations_en.csv", "occupations.csv", "occupations"],
        "Occupations",
    )
    skill_rows = safe_read(
        ["skills_en.csv", "skills.csv", "skills"],
        "Skills",
    )
    relation_rows = safe_read(
        ["occupationSkillRelations_en.csv", "occupationSkillRelations.csv",
         "occupationSkillRelations"],
        "Occupation-Skill relations",
    )
    isco_rows = safe_read(
        ["ISCOGroups_en.csv", "ISCOGroups.csv", "ISCOGroups",
         "iscoGroups_en.csv", "iscoGroups.csv"],
        "ISCO groups",
    )
    broader_rows = safe_read(
        ["broaderRelationsOccPillar_en.csv", "broaderRelationsOccPillar.csv",
         "broaderRelationsOccPillar"],
        "Broader relations (occ pillar)",
    )

    if not occ_rows:
        print("\nERROR: Cannot proceed without occupations data.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build lookups
    # ------------------------------------------------------------------
    isco_lookup = build_isco_lookup(isco_rows)
    parent_of = build_broader_map(broader_rows)
    occ_essential, occ_optional = build_skill_relations(relation_rows)
    skill_lookup = build_skill_lookup(skill_rows)

    # Track skill usage counts
    essential_count = Counter()
    optional_count = Counter()

    # ------------------------------------------------------------------
    # 3. Build occupation objects
    # ------------------------------------------------------------------
    occupations = []
    for row in occ_rows:
        uri = row.get("conceptUri", "")
        if not uri:
            continue

        isco_code, isco_group, hierarchy = resolve_hierarchy(
            uri, parent_of, isco_lookup,
        )

        # Fallback: use iscoGroup column if hierarchy walk didn't find a code
        if not isco_code and row.get("iscoGroup"):
            isco_code = row["iscoGroup"]

        def make_skill_entry(skill_uri):
            info = skill_lookup.get(skill_uri)
            if info:
                return {
                    "uri": info["uri"],
                    "title": info["title"],
                    "type": info["type"],
                    "reuse_level": info["reuse_level"],
                }
            return {"uri": skill_uri, "title": "", "type": "", "reuse_level": ""}

        ess_uris = occ_essential.get(uri, [])
        opt_uris = occ_optional.get(uri, [])

        for su in ess_uris:
            essential_count[su] += 1
        for su in opt_uris:
            optional_count[su] += 1

        occupations.append({
            "uri": uri,
            "title": row.get("preferredLabel", ""),
            "description": row.get("description", ""),
            "isco_code": isco_code,
            "isco_group": isco_group,
            "hierarchy": hierarchy,
            "essential_skills": [make_skill_entry(su) for su in ess_uris],
            "optional_skills": [make_skill_entry(su) for su in opt_uris],
        })

    # ------------------------------------------------------------------
    # 4. Build skill objects
    # ------------------------------------------------------------------
    skills = []
    for uri, info in skill_lookup.items():
        skills.append({
            "uri": info["uri"],
            "title": info["title"],
            "description": info["description"],
            "type": info["type"],
            "reuse_level": info["reuse_level"],
            "essential_for_count": essential_count.get(uri, 0),
            "optional_for_count": optional_count.get(uri, 0),
        })

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    occ_path = os.path.join(OUT_DIR, "esco_occupations.json")
    with open(occ_path, "w", encoding="utf-8") as f:
        json.dump(occupations, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(occupations)} occupations to {occ_path}")

    skill_path = os.path.join(OUT_DIR, "esco_skills.json")
    with open(skill_path, "w", encoding="utf-8") as f:
        json.dump(skills, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(skills)} skills to {skill_path}")

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary statistics")
    print(f"{'='*60}")
    print(f"Total occupations: {len(occupations)}")
    print(f"Total skills:      {len(skills)}")

    if occupations:
        skill_counts = [
            len(o["essential_skills"]) + len(o["optional_skills"])
            for o in occupations
        ]
        ess_counts = [len(o["essential_skills"]) for o in occupations]
        opt_counts = [len(o["optional_skills"]) for o in occupations]

        print(f"\nSkills per occupation (essential + optional):")
        print(f"  avg: {sum(skill_counts)/len(skill_counts):.1f}  "
              f"min: {min(skill_counts)}  max: {max(skill_counts)}")
        print(f"Essential per occupation:")
        print(f"  avg: {sum(ess_counts)/len(ess_counts):.1f}  "
              f"min: {min(ess_counts)}  max: {max(ess_counts)}")
        print(f"Optional per occupation:")
        print(f"  avg: {sum(opt_counts)/len(opt_counts):.1f}  "
              f"min: {min(opt_counts)}  max: {max(opt_counts)}")

    if skills:
        type_counts = Counter(s["type"] for s in skills)
        reuse_counts = Counter(s["reuse_level"] for s in skills)
        print(f"\nSkill type breakdown:")
        for t, c in sorted(type_counts.items()):
            print(f"  {t or '(empty)'}: {c}")
        print(f"\nReuse level breakdown:")
        for r, c in sorted(reuse_counts.items()):
            print(f"  {r or '(empty)'}: {c}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
