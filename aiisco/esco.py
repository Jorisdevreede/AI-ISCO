"""Reading the ESCO v1.2 CSV download and joining it into occupations and skills.

The download is a directory of CSVs whose names differ between releases
(occupations_en.csv, occupations.csv, ...), so each table is located by name
hints rather than by a fixed path. Nothing here prints or writes: callers do
the reporting, which keeps the join testable against a handful of rows.
"""

import csv
import os
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass

TableSpec = namedtuple("TableSpec", "key label hints")

TABLE_SPECS = (
    TableSpec("occupations", "Occupations",
              ("occupations_en.csv", "occupations.csv", "occupations")),
    TableSpec("skills", "Skills",
              ("skills_en.csv", "skills.csv", "skills")),
    TableSpec("relations", "Occupation-Skill relations",
              ("occupationSkillRelations_en.csv", "occupationSkillRelations.csv",
               "occupationSkillRelations")),
    TableSpec("isco", "ISCO groups",
              ("ISCOGroups_en.csv", "ISCOGroups.csv", "ISCOGroups",
               "iscoGroups_en.csv", "iscoGroups.csv")),
    TableSpec("broader", "Broader relations (occ pillar)",
              ("broaderRelationsOccPillar_en.csv", "broaderRelationsOccPillar.csv",
               "broaderRelationsOccPillar")),
)


@dataclass
class EscoTable:
    """One ESCO CSV: the label used when reporting it, where it was found, its rows."""

    label: str
    path: str | None
    rows: list


@dataclass
class EscoIndex:
    """The lookups the join needs, all keyed by ESCO concept URI."""

    isco_groups: dict
    parent_of: dict
    skills: dict
    essential: dict
    optional: dict


# ---------------------------------------------------------------------------
# Finding and reading the CSVs
# ---------------------------------------------------------------------------

def list_dir(data_dir):
    """List the names in *data_dir*, empty when the directory does not exist."""
    try:
        return os.listdir(data_dir)
    except FileNotFoundError:
        return []


def exact_match(names, hint):
    """Return the name equal to *hint*, ignoring case."""
    for name in names:
        if name.lower() == hint.lower():
            return name
    return None


def csv_containing(names, hint):
    """Return the first CSV whose name contains *hint*, ignoring case."""
    for name in names:
        if hint.lower() in name.lower() and name.lower().endswith(".csv"):
            return name
    return None


def match_name(names, hint):
    """Return the name equal to *hint*, else the first CSV containing it."""
    return exact_match(names, hint) or csv_containing(names, hint)


def find_csv(data_dir, name_hints):
    """Find a CSV in *data_dir* matching any of the *name_hints*, else None.

    Hints are tried in order, so a caller can prefer occupations_en.csv over
    the bare occupations.csv of an older download.
    """
    names = list_dir(data_dir)
    for hint in name_hints:
        match = match_name(names, hint)
        if match is not None:
            return os.path.join(data_dir, match)
    return None


def read_csv(path):
    """Read a CSV file and return a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_table(data_dir, spec):
    """Locate and read one table, returning an empty table when it is absent."""
    path = find_csv(data_dir, spec.hints)
    if path is None:
        return EscoTable(spec.label, None, [])
    return EscoTable(spec.label, path, read_csv(path))


def load_tables(data_dir):
    """Load every ESCO table from *data_dir*, keyed by TABLE_SPECS key."""
    return {spec.key: load_table(data_dir, spec) for spec in TABLE_SPECS}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def build_isco_lookup(isco_rows):
    """Build uri -> {code, label} lookup for ISCO groups."""
    lookup = {}
    for row in isco_rows:
        lookup[row.get("conceptUri", "")] = {
            "code": row.get("code", ""),
            "label": row.get("preferredLabel", ""),
        }
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


def build_skill_relations(relation_rows):
    """Split occupationSkillRelations into per-occupation essential/optional URIs.

    Anything that is not flagged essential counts as optional, and duplicate
    rows are kept, because the usage counts are taken from these lists.
    """
    essential = defaultdict(list)
    optional = defaultdict(list)
    for row in relation_rows:
        occ_uri = row.get("occupationUri", "")
        skill_uri = row.get("skillUri", "")
        if not occ_uri or not skill_uri:
            continue
        bucket = essential if "essential" in row.get("relationType", "").lower() else optional
        bucket[occ_uri].append(skill_uri)
    return essential, optional


def build_skill_lookup(skill_rows):
    """Build uri -> skill info dict, normalising the skill type."""
    lookup = {}
    for row in skill_rows:
        uri = row.get("conceptUri", "")
        raw_type = row.get("skillType", "")
        lookup[uri] = {
            "uri": uri,
            "title": row.get("preferredLabel", ""),
            "description": row.get("description", ""),
            "type": "knowledge" if "knowledge" in raw_type.lower() else "skill",
            "reuse_level": row.get("reuseLevel", ""),
        }
    return lookup


def build_index(tables):
    """Turn the loaded tables into the lookups the join walks."""
    essential, optional = build_skill_relations(tables["relations"].rows)
    return EscoIndex(
        isco_groups=build_isco_lookup(tables["isco"].rows),
        parent_of=build_broader_map(tables["broader"].rows),
        skills=build_skill_lookup(tables["skills"].rows),
        essential=essential,
        optional=optional,
    )


# ---------------------------------------------------------------------------
# The ISCO-08 hierarchy
# ---------------------------------------------------------------------------

def ancestors(uri, parent_of):
    """Yield the broader concepts of *uri*, stopping when a chain loops back."""
    current = uri
    visited = set()
    while current in parent_of and current not in visited:
        visited.add(current)
        current = parent_of[current]
        yield current


def hierarchy_labels(uri, index):
    """List the ISCO group labels above *uri*, broadest group first."""
    labels = [
        index.isco_groups[parent]["label"]
        for parent in ancestors(uri, index.parent_of)
        if parent in index.isco_groups
    ]
    labels.reverse()
    return labels


def nearest_isco_group(uri, index):
    """Return (code, label) of the closest broader ISCO group that carries a code."""
    for parent in ancestors(uri, index.parent_of):
        info = index.isco_groups.get(parent)
        if info and info["code"]:
            return info["code"], info["label"]
    return "", ""


def resolve_hierarchy(uri, index):
    """Walk up the broader-relation chain to place *uri* in ISCO-08.

    Returns (isco_code, isco_group_label, hierarchy_list) with hierarchy_list
    running from the broadest group to the most specific.
    """
    isco_code, isco_group = nearest_isco_group(uri, index)
    return isco_code, isco_group, hierarchy_labels(uri, index)


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def skill_entry(skill_uri, skills):
    """Describe one skill of an occupation, blank when the URI is unknown."""
    info = skills.get(skill_uri)
    if info is None:
        return {"uri": skill_uri, "title": "", "type": "", "reuse_level": ""}
    return {
        "uri": info["uri"],
        "title": info["title"],
        "type": info["type"],
        "reuse_level": info["reuse_level"],
    }


def build_occupation(row, index):
    """Build one occupation record with its ISCO placing and its skills."""
    uri = row.get("conceptUri", "")
    isco_code, isco_group, hierarchy = resolve_hierarchy(uri, index)
    if not isco_code and row.get("iscoGroup"):
        isco_code = row["iscoGroup"]
    return {
        "uri": uri,
        "title": row.get("preferredLabel", ""),
        "description": row.get("description", ""),
        "isco_code": isco_code,
        "isco_group": isco_group,
        "hierarchy": hierarchy,
        "essential_skills": [skill_entry(s, index.skills) for s in index.essential.get(uri, [])],
        "optional_skills": [skill_entry(s, index.skills) for s in index.optional.get(uri, [])],
    }


def build_occupations(occ_rows, index):
    """Build the occupation records, skipping rows without a concept URI."""
    return [build_occupation(row, index) for row in occ_rows if row.get("conceptUri", "")]


def count_usage(occupations):
    """Count how many occupations need each skill, essentially and optionally."""
    essential = Counter()
    optional = Counter()
    for occupation in occupations:
        essential.update(entry["uri"] for entry in occupation["essential_skills"])
        optional.update(entry["uri"] for entry in occupation["optional_skills"])
    return essential, optional


def build_skills(index, essential_count, optional_count):
    """Build the skill records with their usage counts, in CSV order."""
    return [
        {
            "uri": info["uri"],
            "title": info["title"],
            "description": info["description"],
            "type": info["type"],
            "reuse_level": info["reuse_level"],
            "essential_for_count": essential_count.get(uri, 0),
            "optional_for_count": optional_count.get(uri, 0),
        }
        for uri, info in index.skills.items()
    ]


def join_tables(tables):
    """Join the loaded tables into the occupation and skill records to write."""
    index = build_index(tables)
    occupations = build_occupations(tables["occupations"].rows, index)
    essential_count, optional_count = count_usage(occupations)
    return occupations, build_skills(index, essential_count, optional_count)
