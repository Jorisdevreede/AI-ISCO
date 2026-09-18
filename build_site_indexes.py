#!/usr/bin/env python3
"""Build the five index files the redesigned site loads.

Usage:
    uv run python build_site_indexes.py
    uv run python build_site_indexes.py --scorer typesafe

Inputs are site/portfolio_data.json (canonical for scores, quadrants and the
ISCO code), data/esco/occupations_en.csv (alternative labels for search) and
data/esco/ISCOGroups_en.csv (labels for all four ISCO levels). site/data.json is
deliberately not read: it is a stale copy of an older scoring run and none of the
five outputs needs a field that only it carries.

--scorer typesafe reads site/portfolio_data_typesafe.json and writes every output
with a _typesafe suffix, which is what the site's "Scores from" switch loads.

Exits non-zero when an output misses its gzipped budget.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

from aiisco import site_indexes as ix
from aiisco.jsonio import load_json, write_json

SCORER_SUFFIX = {"gemini": "", "typesafe": "_typesafe"}


def read_alt_labels(path):
    """Lower-cased ESCO title -> [(iscoGroup, altLabels)]; a few titles repeat."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[ix.normalise(row["preferredLabel"])].append(
                (row["iscoGroup"], row["altLabels"])
            )
    return rows


def read_isco_labels(path):
    """ISCO code -> group label, for all four levels."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["code"]: row["preferredLabel"] for row in csv.DictReader(handle)}


def input_paths(site_dir, esco_dir, suffix):
    """The three files a build reads, newest-first order irrelevant."""
    return (
        os.path.join(site_dir, f"portfolio_data{suffix}.json"),
        os.path.join(esco_dir, "occupations_en.csv"),
        os.path.join(esco_dir, "ISCOGroups_en.csv"),
    )


def build_date(given, paths):
    """The date stamped into stats.json: the one given, else the newest input's
    modification date. Never "now", so a rebuild of unchanged inputs is
    byte-identical."""
    if given:
        return given
    newest = max(os.path.getmtime(path) for path in paths)
    stamp = datetime.datetime.fromtimestamp(newest, tz=datetime.timezone.utc)
    return stamp.date().isoformat()


def build_documents(portfolio, esco, labels, budget_bytes):
    """The four large documents, keyed by output name."""
    occupations = sorted(portfolio["occupations"], key=lambda o: o["s"])
    skills = portfolio["skills"]
    rows = ix.search_rows(occupations, labels)
    candidates = ix.alt_candidates(occupations, ix.labels_by_slug(occupations, esco))
    return {
        "search_index": ix.build_search_index(rows, candidates, budget_bytes),
        "groups": ix.build_groups(occupations, labels, skills),
        "skill_index": ix.build_skill_index(skills, ix.skill_counts(occupations)),
        "skill_occupations": ix.build_skill_occupations(occupations),
    }


def write_compact(path, data):
    """Write one of the large files and return its raw bytes."""
    payload = ix.encode_compact(data)
    with open(path, "wb") as handle:
        handle.write(payload)
    return payload


def write_documents(documents, site_dir, suffix):
    """Write every large document; return (name, raw size, gzipped size) rows."""
    sizes = []
    for name, data in sorted(documents.items()):
        payload = write_compact(os.path.join(site_dir, f"{name}{suffix}.json"), data)
        sizes.append((name, len(payload), ix.gzipped_size(payload)))
    return sizes


def write_stats(stats, site_dir, suffix):
    """stats.json stays indented; it is tiny and people read it."""
    write_json(os.path.join(site_dir, f"stats{suffix}.json"), stats)
    payload = json.dumps(stats, indent=2, ensure_ascii=False).encode("utf-8")
    return ("stats", len(payload), ix.gzipped_size(payload))


def report(sizes):
    """Print every size against its budget; True when all of them fit."""
    fits_all = True
    for name, raw, packed in sorted(sizes):
        budget = ix.BUDGET_GZ_KB.get(name)
        fits = budget is None or packed <= budget * 1024
        fits_all = fits_all and fits
        limit = "lazy, no budget" if budget is None else f"budget {budget} KB gz"
        print(
            f"{name:<18} {raw / 1024:9.1f} KB raw {packed / 1024:8.1f} KB gz  "
            f"{limit:<18} {'ok' if fits else 'OVER BUDGET'}"
        )
    return fits_all


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scorer", choices=sorted(SCORER_SUFFIX), default="gemini")
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--esco-dir", default=os.path.join("data", "esco"))
    parser.add_argument("--date", default=None, help="ISO build date for stats.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    suffix = SCORER_SUFFIX[args.scorer]
    paths = input_paths(args.site_dir, args.esco_dir, suffix)
    portfolio = load_json(paths[0])
    documents = build_documents(
        portfolio, read_alt_labels(paths[1]), read_isco_labels(paths[2]),
        ix.BUDGET_GZ_KB["search_index"] * 1024,
    )
    stats = ix.build_stats(
        portfolio["occupations"], len(portfolio["skills"]),
        build_date(args.date, paths), ix.QUADRANT_THRESHOLD,
    )
    sizes = write_documents(documents, args.site_dir, suffix)
    sizes.append(write_stats(stats, args.site_dir, suffix))
    return 0 if report(sizes) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
