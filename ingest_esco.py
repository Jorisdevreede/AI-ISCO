"""
Ingest ESCO v1.2 CSV data and produce enriched JSON files.

Reads CSV files from data/esco/, joins occupations with skills and
the ISCO hierarchy, and outputs:
  - data/esco_occupations.json  (occupations with attached skills)
  - data/esco_skills.json       (skills with usage counts)

Usage:
    uv run python ingest_esco.py
"""

import os
import sys
from collections import Counter
from typing import NoReturn

from aiisco import esco
from aiisco.jsonio import write_json

DATA_DIR = "data/esco"
OUT_DIR = "data"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def fail(message) -> NoReturn:
    """Print an error and stop: none of the later steps work without the CSVs."""
    print(message)
    sys.exit(1)


def require_csv_files():
    """Return the CSV names in DATA_DIR, failing when there are none."""
    if not os.path.isdir(DATA_DIR):
        fail(f"ERROR: {DATA_DIR}/ directory not found.")
    names = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not names:
        fail(f"ERROR: No CSV files found in {DATA_DIR}/")
    return names


def report_table(table):
    """Print where one ESCO table was found, or that it is missing."""
    if table.path is None:
        print(f"  WARNING: Could not find {table.label} CSV in {DATA_DIR}/")
    else:
        print(f"  {table.label}: {table.path} ({len(table.rows)} rows)")


def load_tables():
    """Load every ESCO table from DATA_DIR, reporting each one."""
    tables = esco.load_tables(DATA_DIR)
    for table in tables.values():
        report_table(table)
    if not tables["occupations"].rows:
        fail("\nERROR: Cannot proceed without occupations data.")
    return tables


def join(tables):
    """Join the loaded tables, stopping when the download contradicts itself."""
    try:
        return esco.join_tables(tables)
    except ValueError as clash:
        fail(f"\nERROR: {clash}")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_records(name, records):
    """Write *records* into OUT_DIR as JSON and return the path written."""
    path = os.path.join(OUT_DIR, name)
    write_json(path, records)
    return path


def write_outputs(occupations, skills):
    """Write both JSON files, reporting how much went into each."""
    os.makedirs(OUT_DIR, exist_ok=True)
    occ_path = write_records("esco_occupations.json", occupations)
    print(f"\nWrote {len(occupations)} occupations to {occ_path}")
    skill_path = write_records("esco_skills.json", skills)
    print(f"Wrote {len(skills)} skills to {skill_path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_spread(counts):
    """Print the avg/min/max line used for every per-occupation count."""
    print(f"  avg: {sum(counts)/len(counts):.1f}  min: {min(counts)}  max: {max(counts)}")


def print_occupation_stats(occupations):
    """Print how many skills an occupation carries, in total and per relation."""
    essential = [len(o["essential_skills"]) for o in occupations]
    optional = [len(o["optional_skills"]) for o in occupations]
    print("\nSkills per occupation (essential + optional):")
    print_spread([e + o for e, o in zip(essential, optional)])
    print("Essential per occupation:")
    print_spread(essential)
    print("Optional per occupation:")
    print_spread(optional)


def print_breakdown(title, counts):
    """Print a sorted '<value>: <count>' block, naming blanks explicitly."""
    print(f"\n{title} breakdown:")
    for value, count in sorted(counts.items()):
        print(f"  {value or '(empty)'}: {count}")


def print_skill_stats(skills):
    """Print the skill type and reuse level breakdowns."""
    print_breakdown("Skill type", Counter(s["type"] for s in skills))
    print_breakdown("Reuse level", Counter(s["reuse_level"] for s in skills))


def print_summary(occupations, skills):
    """Print the closing statistics block."""
    print(f"\n{'='*60}")
    print("Summary statistics")
    print(f"{'='*60}")
    print(f"Total occupations: {len(occupations)}")
    print(f"Total skills:      {len(skills)}")
    if occupations:
        print_occupation_stats(occupations)
    if skills:
        print_skill_stats(skills)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Read the ESCO CSVs, join them and write the two enriched JSON files."""
    print("ESCO data ingestion")
    print("=" * 60)
    csv_files = require_csv_files()
    print(f"\nFound {len(csv_files)} CSV files in {DATA_DIR}/")
    print("Loading data...\n")

    occupations, skills = join(load_tables())

    write_outputs(occupations, skills)
    print_summary(occupations, skills)
    print("\nDone.")


if __name__ == "__main__":
    main()
