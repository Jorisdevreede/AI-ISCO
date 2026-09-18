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

Usage:
    uv run python compare_skill_scores.py
    uv run python compare_skill_scores.py --occupations
    uv run python compare_skill_scores.py --top 25 --out data/skill_score_comparison.json
"""

import argparse
import hashlib
import json

from aggregate_scores import (
    ESSENTIAL_WEIGHT,
    OPTIONAL_WEIGHT,
    QUADRANT_THRESHOLD,
    assign_quadrant,
)

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


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def band(score_1_to_10):
    """Rubric band 0-4 for a 1-10 score (1-2, 3-4, 5-6, 7-8, 9-10)."""
    return min(4, max(0, int((score_1_to_10 - 1) // 2)))


def roll_up(occupation, scores):
    """Weighted occupation score, same weights as aggregate_scores.py."""
    auto = amp = weight = 0.0
    for key, w in (("essential_skills", ESSENTIAL_WEIGHT),
                   ("optional_skills", OPTIONAL_WEIGHT)):
        for skill in occupation.get(key, []):
            sc = scores.get(skill["uri"])
            if sc is None:
                continue
            auto += sc[0] * w
            amp += sc[1] * w
            weight += w
    if weight == 0:
        return None
    return round(auto / weight, 1), round(amp / weight, 1)


def compare_occupations(rows):
    """Roll both score sets up to occupations and compare what the site shows."""
    with open(OCCUPATIONS_FILE) as f:
        occupations = json.load(f)
    gemini = {r["typesafe"]["uri"]: (r["gemini"]["a"], r["gemini"]["m"])
              for r in rows}
    typesafe = {r["typesafe"]["uri"]: (r["typesafe"]["automation_risk"],
                                       r["typesafe"]["amplification_potential"])
                for r in rows}

    pairs = []
    for occ in occupations:
        g, t = roll_up(occ, gemini), roll_up(occ, typesafe)
        if g and t:
            pairs.append((occ["title"], g, t))

    print(f"\n== occupation roll-up ({len(pairs)} occupations, essential x"
          f"{ESSENTIAL_WEIGHT:g}, optional x{OPTIONAL_WEIGHT:g})")
    for axis, label in ((0, "automation risk"), (1, "amplification potential")):
        g = [p[1][axis] for p in pairs]
        t = [p[2][axis] for p in pairs]
        gap = sum(abs(a - b) for a, b in zip(g, t)) / len(pairs)
        print(f"  {label}: rho {spearman(g, t):.3f}, mean |gap| {gap:.2f}, "
              f"mean gemini {sum(g) / len(g):.2f} | typesafe {sum(t) / len(t):.2f}")

    confusion = {}
    for _, g, t in pairs:
        key = (assign_quadrant(*g), assign_quadrant(*t))
        confusion[key] = confusion.get(key, 0) + 1
    same = sum(n for (a, b), n in confusion.items() if a == b)
    print(f"  same quadrant (threshold {QUADRANT_THRESHOLD}): "
          f"{same / len(pairs):.1%}")
    names = ["TRANSFORM", "SHRINK", "EVOLVE", "STABLE"]
    print("  gemini row -> typesafe column")
    print("  " + " " * 10 + "".join(f"{n:>10}" for n in names))
    for a in names:
        print(f"  {a:>10}" + "".join(f"{confusion.get((a, b), 0):>10}"
                                     for b in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--top", type=int, default=12,
                        help="How many of the largest disagreements to list")
    parser.add_argument("--occupations", action="store_true",
                        help="Also compare the occupation-level roll-up")
    parser.add_argument("--out", default=None,
                        help="Write the joined rows to this JSON file")
    args = parser.parse_args()

    with open(TYPESAFE_FILE) as f:
        typesafe = json.load(f)
    with open(PUBLISHED_FILE) as f:
        published = json.load(f)["skills"]
    by_title = {v["t"]: v for v in published.values()}
    with open(SKILLS_FILE) as f:
        kinds = {s["uri"]: s.get("type", "") for s in json.load(f)}

    rows = []
    for s in typesafe:
        pub = published.get(hashlib.md5(s["uri"].encode()).hexdigest()[:8])
        if pub is None or pub["t"] != s["title"]:
            pub = by_title.get(s["title"])
        if pub is None or pub.get("a") is None or pub.get("m") is None:
            continue
        rows.append({"typesafe": s, "gemini": pub,
                     "type": kinds.get(s["uri"], "")})

    print(f"TypeSafe skills: {len(typesafe)} | matched to published Gemini "
          f"scores: {len(rows)}")
    if not rows:
        return

    for label, pub_key, ts_key, probs_key, conf_key in AXES:
        gem = [r["gemini"][pub_key] for r in rows]
        jev = [r["typesafe"][ts_key] for r in rows]
        top_band = [max(range(5), key=lambda i: r["typesafe"][probs_key][i])
                    for r in rows]
        gem_band = [band(g) for g in gem]
        same = sum(a == b for a, b in zip(top_band, gem_band))
        near = sum(abs(a - b) <= 1 for a, b in zip(top_band, gem_band))
        gap = sum(abs(a - b) for a, b in zip(gem, jev)) / len(rows)

        print(f"\n== {label}")
        print(f"  Spearman rho:            {spearman(gem, jev):.3f}")
        print(f"  same rubric band:        {same / len(rows):.1%}")
        print(f"  within one band:         {near / len(rows):.1%}")
        print(f"  mean |gap| on 1-10:      {gap:.2f}")
        print(f"  mean  gemini {sum(gem) / len(gem):.2f} | typesafe "
              f"{sum(jev) / len(jev):.2f}")

        for kind in sorted({r["type"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["type"] == kind]
            rho = spearman([gem[i] for i in idx], [jev[i] for i in idx])
            print(f"  {kind or 'untyped':>10} items: {len(idx):>6}, rho {rho:.3f}, "
                  f"mean gemini {sum(gem[i] for i in idx) / len(idx):.2f} | "
                  f"typesafe {sum(jev[i] for i in idx) / len(idx):.2f}")

        print("  by TypeSafe confidence:")
        for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
            idx = [i for i, r in enumerate(rows)
                   if lo <= r["typesafe"][conf_key] < hi]
            if not idx:
                continue
            agree = sum(top_band[i] == gem_band[i] for i in idx) / len(idx)
            print(f"    {lo:.2f}-{min(hi, 1.0):.2f}: {len(idx):>6} skills "
                  f"({len(idx) / len(rows):.0%}), same band {agree:.1%}")

        worst = sorted(range(len(rows)), key=lambda i: -abs(gem[i] - jev[i]))
        print("  largest disagreements (gemini | typesafe, confidence):")
        for i in worst[:args.top]:
            r = rows[i]
            print(f"    {r['typesafe']['title']!r}: {gem[i]:.0f} | {jev[i]:.1f} "
                  f"({r['typesafe'][conf_key]:.2f})")

    if args.occupations:
        compare_occupations(rows)

    if args.out:
        joined = [
            {
                "uri": r["typesafe"]["uri"],
                "title": r["typesafe"]["title"],
                "gemini_automation": r["gemini"]["a"],
                "typesafe_automation": r["typesafe"]["automation_risk"],
                "automation_confidence": r["typesafe"]["automation_confidence"],
                "gemini_amplification": r["gemini"]["m"],
                "typesafe_amplification": r["typesafe"]["amplification_potential"],
                "amplification_confidence": r["typesafe"]["amplification_confidence"],
                "gemini_rationale": r["gemini"].get("r", ""),
            }
            for r in rows
        ]
        with open(args.out, "w") as f:
            json.dump(joined, f, indent=1, ensure_ascii=False)
        print(f"\nWrote {len(joined)} joined rows to {args.out}")


if __name__ == "__main__":
    main()
