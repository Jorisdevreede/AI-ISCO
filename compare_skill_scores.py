"""
Compare the TypeSafe skill scores with the published Gemini scores.

The Gemini per-skill scores are not in the repo (data/skill_scores.json is a
gitignored build product), but the published site embeds them: each skill in
site/portfolio_data.json carries a (automation), m (amplification), r (rationale),
keyed by an md5 prefix of the ESCO URI.

Reports, per axis: rank correlation, agreement on the rubric band, mean absolute
gap on the 1-10 scale, how agreement moves with TypeSafe's confidence, and the
largest disagreements. With --occupations it also rolls both score sets up to
occupations (same weights as aggregate_scores.py) and compares the quadrants.

For your own evaluation only: TypeSafe's customer agreement (section 2.3(f))
prohibits publishing benchmarks or performance information about their
service, so do not publish this script's output without their written consent.

Usage:
    uv run python compare_skill_scores.py
    uv run python compare_skill_scores.py --occupations
    uv run python compare_skill_scores.py --top 25 --out data/skill_score_comparison.json
"""

import argparse
from collections import Counter
from dataclasses import dataclass

from aiisco.jsonio import load_json, write_json
from aiisco.portfolio import skill_short_id
from aiisco.rollup import (
    ESSENTIAL_WEIGHT,
    OPTIONAL_WEIGHT,
    QUADRANT_THRESHOLD,
    QUADRANTS,
    assign_quadrant,
    scored_skills,
    weighted_averages,
)
from aiisco.stats import band, mean, share, spearman, top_band

TYPESAFE_FILE = "data/skill_scores_typesafe.json"
PUBLISHED_FILE = "site/portfolio_data.json"
SKILLS_FILE = "data/esco_skills.json"
OCCUPATIONS_FILE = "data/esco_occupations.json"

AXES = [
    ("automation risk", "a", "automation_risk", "automation_probs",
     "automation_confidence"),
    ("amplification potential", "m", "amplification_potential",
     "amplification_probs", "amplification_confidence"),
]

CONFIDENCE_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01))

JOINED_FIELDS = (
    ("uri", "typesafe", "uri"),
    ("title", "typesafe", "title"),
    ("gemini_automation", "gemini", "a"),
    ("typesafe_automation", "typesafe", "automation_risk"),
    ("automation_confidence", "typesafe", "automation_confidence"),
    ("gemini_amplification", "gemini", "m"),
    ("typesafe_amplification", "typesafe", "amplification_potential"),
    ("amplification_confidence", "typesafe", "amplification_confidence"),
)


@dataclass
class AxisScores:
    """Both scorers' numbers for one axis, aligned with the joined rows."""

    conf_key: str
    gemini: list
    typesafe: list
    gemini_band: list
    typesafe_band: list


# ---------------------------------------------------------------------------
# Joining the two score sets
# ---------------------------------------------------------------------------

def published_match(skill, published, by_title):
    """The published entry for a TypeSafe skill: by md5 short ID, else by title."""
    pub = published.get(skill_short_id(skill["uri"]))
    if pub is None or pub["t"] != skill["title"]:
        return by_title.get(skill["title"])
    return pub


def has_both_scores(pub):
    """True when a published entry exists and carries both axes."""
    return pub is not None and pub.get("a") is not None and pub.get("m") is not None


def join_rows(typesafe, published, kinds):
    """One row per TypeSafe skill that has a published Gemini score."""
    by_title = {entry["t"]: entry for entry in published.values()}
    rows = []
    for skill in typesafe:
        pub = published_match(skill, published, by_title)
        if has_both_scores(pub):
            rows.append({"typesafe": skill, "gemini": pub,
                         "type": kinds.get(skill["uri"], "")})
    return rows


def load_rows():
    """Join the private TypeSafe scores to the scores the site publishes."""
    typesafe = load_json(TYPESAFE_FILE)
    published = load_json(PUBLISHED_FILE)["skills"]
    kinds = {skill["uri"]: skill.get("type", "") for skill in load_json(SKILLS_FILE)}
    rows = join_rows(typesafe, published, kinds)
    print(f"TypeSafe skills: {len(typesafe)} | matched to published Gemini "
          f"scores: {len(rows)}")
    return rows


def axis_scores(rows, axis):
    """Line up both scorers' numbers and rubric bands for one axis."""
    _label, pub_key, ts_key, probs_key, conf_key = axis
    gemini = [r["gemini"][pub_key] for r in rows]
    return AxisScores(conf_key=conf_key,
                      gemini=gemini,
                      typesafe=[r["typesafe"][ts_key] for r in rows],
                      gemini_band=[band(score) for score in gemini],
                      typesafe_band=[top_band(r["typesafe"][probs_key]) for r in rows])


# ---------------------------------------------------------------------------
# Per-axis report
# ---------------------------------------------------------------------------

def print_agreement(scores):
    """Print how close the two scorers are on one axis."""
    same = [a == b for a, b in zip(scores.typesafe_band, scores.gemini_band)]
    near = [abs(a - b) <= 1 for a, b in zip(scores.typesafe_band, scores.gemini_band)]
    gaps = [abs(a - b) for a, b in zip(scores.gemini, scores.typesafe)]
    print(f"  Spearman rho:            {spearman(scores.gemini, scores.typesafe):.3f}")
    print(f"  same rubric band:        {share(same):.1%}")
    print(f"  within one band:         {share(near):.1%}")
    print(f"  mean |gap| on 1-10:      {mean(gaps):.2f}")
    print(f"  mean  gemini {mean(scores.gemini):.2f} | typesafe "
          f"{mean(scores.typesafe):.2f}")


def subset(values, idx):
    """The values at the given positions, in that order."""
    return [values[i] for i in idx]


def print_by_type(rows, scores):
    """Print the breakdown by ESCO skill type."""
    for kind in sorted({r["type"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["type"] == kind]
        gem, jev = subset(scores.gemini, idx), subset(scores.typesafe, idx)
        print(f"  {kind or 'untyped':>10} items: {len(idx):>6}, rho {spearman(gem, jev):.3f}, "
              f"mean gemini {mean(gem):.2f} | typesafe {mean(jev):.2f}")


def print_by_confidence(rows, scores):
    """Print how band agreement moves with TypeSafe's own confidence."""
    print("  by TypeSafe confidence:")
    for lo, hi in CONFIDENCE_BUCKETS:
        idx = [i for i, r in enumerate(rows)
               if lo <= r["typesafe"][scores.conf_key] < hi]
        if not idx:
            continue
        agree = share([scores.typesafe_band[i] == scores.gemini_band[i] for i in idx])
        print(f"    {lo:.2f}-{min(hi, 1.0):.2f}: {len(idx):>6} skills "
              f"({len(idx) / len(rows):.0%}), same band {agree:.1%}")


def print_disagreements(rows, scores, top):
    """Print the skills the two scorers disagree about the most."""
    worst = sorted(range(len(rows)),
                   key=lambda i: -abs(scores.gemini[i] - scores.typesafe[i]))
    print("  largest disagreements (gemini | typesafe, confidence):")
    for i in worst[:top]:
        print(f"    {rows[i]['typesafe']['title']!r}: {scores.gemini[i]:.0f} | "
              f"{scores.typesafe[i]:.1f} ({rows[i]['typesafe'][scores.conf_key]:.2f})")


def report_axis(rows, axis, top):
    """Print the whole report for one axis."""
    scores = axis_scores(rows, axis)
    print(f"\n== {axis[0]}")
    print_agreement(scores)
    print_by_type(rows, scores)
    print_by_confidence(rows, scores)
    print_disagreements(rows, scores, top)


# ---------------------------------------------------------------------------
# Occupation roll-up
# ---------------------------------------------------------------------------

def roll_up(occupation, scores):
    """Weighted occupation score, same weights as aggregate_scores.py."""
    return weighted_averages(
        [(score[0], score[1], weight) for _, score, weight, _ in
         scored_skills(occupation, lambda skill: scores.get(skill["uri"]))])


def score_map(rows, side, keys):
    """Skill URI -> the (automation, amplification) pair one scorer gave it."""
    auto_key, amp_key = keys
    return {r["typesafe"]["uri"]: (r[side][auto_key], r[side][amp_key]) for r in rows}


def rolled_up_pairs(rows):
    """Both scorers' occupation scores, for the occupations both can score."""
    gemini = score_map(rows, "gemini", ("a", "m"))
    typesafe = score_map(rows, "typesafe",
                         ("automation_risk", "amplification_potential"))
    scored = [(occ["title"], roll_up(occ, gemini), roll_up(occ, typesafe))
              for occ in load_json(OCCUPATIONS_FILE)]
    return [pair for pair in scored if pair[1] and pair[2]]


def print_occupation_axes(pairs):
    """Print rank correlation and gap per axis for the occupation roll-up."""
    for axis, label in ((0, "automation risk"), (1, "amplification potential")):
        gem = [pair[1][axis] for pair in pairs]
        jev = [pair[2][axis] for pair in pairs]
        gaps = [abs(a - b) for a, b in zip(gem, jev)]
        print(f"  {label}: rho {spearman(gem, jev):.3f}, mean |gap| {mean(gaps):.2f}, "
              f"mean gemini {mean(gem):.2f} | typesafe {mean(jev):.2f}")


def print_quadrant_confusion(pairs):
    """Print how often both scorers put an occupation in the same quadrant."""
    confusion = Counter((assign_quadrant(*g), assign_quadrant(*t)) for _, g, t in pairs)
    same = sum(n for (gem, jev), n in confusion.items() if gem == jev)
    print(f"  same quadrant (threshold {QUADRANT_THRESHOLD}): "
          f"{same / len(pairs):.1%}")
    print("  gemini row -> typesafe column")
    print("  " + " " * 10 + "".join(f"{name:>10}" for name in QUADRANTS))
    for gem in QUADRANTS:
        print(confusion_row(gem, confusion))


def confusion_row(gem, confusion):
    """One row of the quadrant confusion table."""
    return f"  {gem:>10}" + "".join(f"{confusion[(gem, jev)]:>10}" for jev in QUADRANTS)


def compare_occupations(rows):
    """Roll both score sets up to occupations and compare what the site shows."""
    pairs = rolled_up_pairs(rows)
    print(f"\n== occupation roll-up ({len(pairs)} occupations, essential x"
          f"{ESSENTIAL_WEIGHT:g}, optional x{OPTIONAL_WEIGHT:g})")
    print_occupation_axes(pairs)
    print_quadrant_confusion(pairs)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def joined_row(row):
    """One flat record per matched skill, for further analysis elsewhere."""
    joined = {name: row[side][key] for name, side, key in JOINED_FIELDS}
    joined["gemini_rationale"] = row["gemini"].get("r", "")
    return joined


def write_joined(path, rows):
    """Write the joined rows so the comparison can be re-read without re-running."""
    joined = [joined_row(row) for row in rows]
    write_json(path, joined, indent=1)
    print(f"\nWrote {len(joined)} joined rows to {path}")


def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--top", type=int, default=12,
                        help="How many of the largest disagreements to list")
    parser.add_argument("--occupations", action="store_true",
                        help="Also compare the occupation-level roll-up")
    parser.add_argument("--out", default=None,
                        help="Write the joined rows to this JSON file")
    return parser.parse_args(argv)


def main():
    """Report how far the private TypeSafe scores sit from the published ones."""
    args = parse_args()
    rows = load_rows()
    if not rows:
        return
    for axis in AXES:
        report_axis(rows, axis, args.top)
    if args.occupations:
        compare_occupations(rows)
    if args.out:
        write_joined(args.out, rows)


if __name__ == "__main__":
    main()
