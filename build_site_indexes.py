#!/usr/bin/env python3
"""Build the five index files the redesigned site loads.

Usage:
    uv run python build_site_indexes.py
    uv run python build_site_indexes.py --scorer typesafe
    uv run python build_site_indexes.py --scorer v2

Inputs are site/portfolio_data.json (canonical for scores, quadrants and the
ISCO code), data/esco/occupations_en.csv (alternative labels for search) and
data/esco/ISCOGroups_en.csv (labels for all four ISCO levels). site/data.json is
deliberately not read: it is a stale copy of an older scoring run and none of the
five outputs needs a field that only it carries.

--scorer typesafe reads site/portfolio_data_typesafe.json and writes every output
with a _typesafe suffix, which is what the site's "Scores from" switch loads;
--scorer v2 does the same with _v2. How the outputs classify occupations is not
read off the flag but off the portfolio file's own "scheme", so a dataset and the
indexes built from it can never disagree.

Exits non-zero when an output misses its gzipped budget.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict, namedtuple

from aiisco import site_indexes as ix
from aiisco.jsonio import load_json, write_compact, write_json, write_shard_files
from aiisco.portfolio import build_short_id_map
from aiisco.rollup import SCORER_SUFFIX, add_scorer_argument

Labels = namedtuple("Labels", "alt isco")


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


def build_documents(portfolio, labels, answers, budget_bytes):
    """The four large documents, keyed by output name."""
    occupations = sorted(portfolio["occupations"], key=lambda o: o["s"])
    skills = portfolio["skills"]
    scheme = ix.scheme_of(portfolio)
    rows = ix.search_rows(occupations, labels.isco, scheme)
    candidates = ix.alt_candidates(occupations,
                                   ix.labels_by_slug(occupations, labels.alt))
    return {
        "search_index": ix.build_search_index(rows, candidates, budget_bytes),
        "groups": ix.build_groups(occupations, labels.isco, skills, scheme),
        "skill_index": ix.build_skill_index(skills, ix.skill_counts(occupations),
                                            scheme, answers),
        "skill_occupations": ix.build_skill_occupations(occupations),
    }


def load_scores(data_dir, suffix):
    """The per-skill scores the share scheme publishes the stored answers from."""
    path = os.path.join(data_dir, f"skill_scores{suffix}.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. The share scheme publishes every "
                 f"skill's stored answers, so this build needs the scores file "
                 f"the scorer wrote.")
    return load_json(path)


def answers_by_id(entries, skills):
    """Short id -> scored entry, for the indexed skills that were scored.

    The short ids are the portfolio's own, rebuilt from the URIs. A skill the
    index carries without a score has no answers to publish, which is ordinary; a
    scored one whose answers cannot be found means the two files came from
    different runs, and is fatal rather than a silently missing shard.
    """
    short = build_short_id_map({entry["uri"] for entry in entries})
    by_id = {short[entry["uri"]]: entry for entry in entries}
    scored = {sid for sid, skill in skills.items() if skill.get("a") is not None}
    missing = sorted(scored - set(by_id))
    if missing:
        sys.exit(f"ERROR: {len(missing)} scored skills of the portfolio dataset "
                 f"have no stored answers (first: {missing[0]}). Rebuild "
                 f"build_portfolio_data.py and this step from the same scores.")
    return {sid: by_id[sid] for sid in sorted(scored)}


def write_documents(documents, site_dir, suffix):
    """Write every large document; return (name, raw size, gzipped size) rows."""
    sizes = []
    for name, data in sorted(documents.items()):
        payload = write_compact(os.path.join(site_dir, f"{name}{suffix}.json"), data)
        sizes.append((name, len(payload), ix.gzipped_size(payload)))
    return sizes


def write_indented(name, data, site_dir, suffix):
    """Write one of the small files people read, and measure what was written."""
    write_json(os.path.join(site_dir, f"{name}{suffix}.json"), data)
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    return (name, len(payload), ix.gzipped_size(payload))


def write_shards(shards, site_dir, suffix):
    """Write one answers file per shard; return its raw and gzipped sizes."""
    payloads = write_shard_files(
        os.path.join(site_dir, f"skill_answers{suffix}"),
        {f"{name}.json": data for name, data in shards.items()})
    return ([len(payload) for payload in payloads],
            [ix.gzipped_size(payload) for payload in payloads])


def report(sizes, budgets):
    """Print every size against its budget; True when all of them fit."""
    fits_all = True
    for name, raw, packed in sorted(sizes):
        budget = budgets.get(name)
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
    add_scorer_argument(parser)
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--esco-dir", default=os.path.join("data", "esco"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--date", default=None, help="ISO build date for stats.json")
    return parser.parse_args(argv)


def stored_answers(portfolio, args, suffix):
    """Every indexed skill's answers, or None for a scheme that publishes none."""
    if ix.scheme_of(portfolio) != ix.SCHEME_SHARES:
        return None
    return answers_by_id(load_scores(args.data_dir, suffix), portfolio["skills"])


def write_scoring(portfolio, answers, args, suffix):
    """Write the rubric and the answers shards; return the rubric's sizes."""
    raw, packed = write_shards(ix.build_skill_answers(answers), args.site_dir, suffix)
    print(ix.shard_report(f"skill_answers{suffix}", raw, packed))
    return write_indented("rubric", ix.build_rubric(portfolio.get("model", "")),
                          args.site_dir, suffix)


def main(argv=None):
    args = parse_args(argv)
    suffix = SCORER_SUFFIX[args.scorer]
    paths = input_paths(args.site_dir, args.esco_dir, suffix)
    portfolio = load_json(paths[0])
    answers = stored_answers(portfolio, args, suffix)
    budgets = ix.budgets_for(ix.scheme_of(portfolio))
    documents = build_documents(
        portfolio,
        Labels(alt=read_alt_labels(paths[1]), isco=read_isco_labels(paths[2])),
        answers, budgets["search_index"] * 1024,
    )
    stats = ix.build_stats(
        portfolio["occupations"], portfolio["skills"],
        ix.stats_context(portfolio, build_date(args.date, paths)),
    )
    sizes = write_documents(documents, args.site_dir, suffix)
    sizes.append(write_indented("stats", stats, args.site_dir, suffix))
    if answers is not None:
        sizes.append(write_scoring(portfolio, answers, args, suffix))
    return 0 if report(sizes, budgets) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
