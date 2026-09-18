"""Pure builders for the five index files the redesigned site loads.

Everything here takes plain data and returns plain data, so it can be tested
without touching the filesystem. Reading and writing lives in
``build_site_indexes.py``.

Scores and quadrants come from ``portfolio_data.json`` only. ``data.json`` is a
stale copy of an older scoring run and is deliberately not an input: an
occupation's four ISCO groups are derived from its own 4-digit code instead.
"""

import gzip
import json
import re
from collections import Counter, defaultdict

try:  # pragma: no cover - exercised by the real build, not by the fixtures
    from aggregate_scores import QUADRANT_THRESHOLD
except ImportError:  # pragma: no cover
    QUADRANT_THRESHOLD = 6

# Group key levels: name, and how many leading digits of the ISCO code it uses.
LEVELS = (("major", 1), ("sub", 2), ("minor", 3), ("unit", 4))
LEVEL_NAMES = [name for name, _ in LEVELS]

ALL_KEY = "all"
ALL_LABEL = "All occupations"

NEAR_LINE = 0.5
HIGH_SCORE = 6
TOP_N = 5

#: Gzipped budgets in KiB, from the brief's "Data files" table. ``skill_index``
#: is the one departure: 13,475 opaque 8-character ids cost 60.5 KB gzipped on
#: their own and the skill titles a further 106 KB, so the brief's 120 KB is
#: below the floor for any format that carries both. See site/js/README.md.
BUDGET_GZ_KB = {
    "search_index": 80,
    "groups": 150,
    "stats": 20,
    "skill_index": 300,
    "skill_occupations": None,
}

WORD = re.compile(r"[a-z0-9]+")


# --- encoding -------------------------------------------------------------

def encode_compact(data):
    """JSON bytes for the large files: no spaces, real UTF-8, stable order."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def gzipped_size(payload):
    """Size of ``payload`` after gzip, as a browser would receive it."""
    return len(gzip.compress(payload, 9, mtime=0))


# --- text -----------------------------------------------------------------

def normalise(text):
    """Lower-case and collapse whitespace, so labels compare predictably."""
    return " ".join(text.lower().split())


def stems(text):
    """Words of ``text``, with a trailing plural ``s`` dropped."""
    words = WORD.findall(text.lower())
    return [word[:-1] if word.endswith("s") and len(word) > 3 else word for word in words]


def covered(word, title_stems):
    """True when a title word already starts with ``word``, so search finds it."""
    return any(stem == word or stem.startswith(word) for stem in title_stems)


def label_order(label):
    """Shortest first, then alphabetical: the most distinct labels lead."""
    return (len(label), label)


def worth_indexing(label, title, title_stems):
    """True when ``label`` adds a word the title does not already offer search."""
    if not label or label == title:
        return False
    return not all(covered(word, title_stems) for word in stems(label))


def alt_labels(title, raw):
    """Indexable ESCO alternative labels for one occupation.

    ``raw`` is the newline-separated ``altLabels`` cell. Labels are lower-cased
    and de-duplicated; the title itself and labels whose every word is already a
    prefix of a title word (``software developers`` beside ``software
    developer``) add nothing to search and are dropped.
    """
    clean = normalise(title)
    title_stems = stems(clean)
    seen = dict.fromkeys(normalise(line) for line in raw.split("\n"))
    kept = [label for label in seen if worth_indexing(label, clean, title_stems)]
    return sorted(kept, key=label_order)


def labels_by_slug(occupations, esco):
    """Slug -> alternative labels. ``esco`` maps a lower-cased ESCO title to
    ``[(iscoGroup, altLabels)]``; a handful of titles repeat, so prefer the row
    whose ISCO group matches the occupation."""
    out = {}
    for occupation in occupations:
        rows = esco.get(normalise(occupation["t"]), [])
        chosen = next((r for r in rows if r[0] == occupation["c"]), None)
        chosen = chosen or (rows[0] if rows else ("", ""))
        out[occupation["s"]] = alt_labels(occupation["t"], chosen[1])
    return out


# --- groups ---------------------------------------------------------------

def group_keys(code):
    """The four group keys an ISCO code belongs to, widest first."""
    return tuple(f"{name}:{code[:width]}" for name, width in LEVELS)


def parent_of(key):
    """The key one level up; majors hang off ``all``, which has no parent."""
    if key == ALL_KEY:
        return None
    name, code = key.split(":", 1)
    index = LEVEL_NAMES.index(name)
    if index == 0:
        return ALL_KEY
    above, width = LEVELS[index - 1]
    return f"{above}:{code[:width]}"


def group_members(occupations):
    """Group key -> the occupations in it, including the ``all`` bucket."""
    members = defaultdict(list)
    for occupation in occupations:
        members[ALL_KEY].append(occupation)
        for key in group_keys(occupation["c"]):
            members[key].append(occupation)
    return members


def children_of(key, keys):
    """Keys one level down that sit under ``key``, sorted."""
    if key == ALL_KEY:
        return sorted(k for k in keys if k.startswith("major:"))
    name, code = key.split(":", 1)
    index = LEVEL_NAMES.index(name)
    if index + 1 == len(LEVELS):
        return []
    below = LEVEL_NAMES[index + 1]
    return sorted(k for k in keys if k.startswith(f"{below}:{code}"))


def percentile(ordered, fraction):
    """Linear interpolation between closest ranks (NumPy's default method).

    ``ordered`` must be sorted ascending. The position is ``(n - 1) * fraction``;
    a fractional position blends the two neighbouring values. p50 of an
    even-length list is therefore the mean of the middle pair.
    """
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def distribution(values):
    """Mean and the 10th/50th/90th percentiles, each to one decimal."""
    ordered = sorted(values)
    summary = {
        "mean": sum(ordered) / len(ordered),
        "p10": percentile(ordered, 0.10),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
    }
    return {name: round(value, 1) for name, value in summary.items()}


def scores_high(scores, skill, field):
    """True when a scored skill reaches 6 on ``field``; unscored never does."""
    return ((scores.get(skill) or {}).get(field) or 0) >= HIGH_SCORE


def driving_skills(occupations, scores, field):
    """The five skills most common among these occupations' essential skills
    that score at least 6 on ``field`` (``a`` automation, ``m`` amplification)."""
    counts = Counter()
    for occupation in occupations:
        essential = dict.fromkeys(occupation.get("se", []))
        counts.update(s for s in essential if scores_high(scores, s, field))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"id": skill, "n": total} for skill, total in ranked[:TOP_N]]


def extreme_slugs(occupations):
    """The five most and five least automation-exposed slugs."""
    by_exposure = sorted(occupations, key=lambda o: (-o["ar"], o["s"]))
    most = [o["s"] for o in by_exposure[:TOP_N]]
    least = [o["s"] for o in reversed(by_exposure[-TOP_N:])]
    return most, least


def group_label(key, labels):
    """Human label for a group key, falling back to the key's own code."""
    if key == ALL_KEY:
        return ALL_LABEL
    code = key.split(":", 1)[1]
    return labels.get(code, code)


def group_entry(key, occupations, labels, scores):
    """One value of groups.json."""
    most, least = extreme_slugs(occupations)
    return {
        "label": group_label(key, labels),
        "level": key.split(":", 1)[0] if key != ALL_KEY else ALL_KEY,
        "code": key.split(":", 1)[1] if key != ALL_KEY else "",
        "n": len(occupations),
        "q": dict(sorted(Counter(o["q"] for o in occupations).items())),
        "auto": distribution([o["ar"] for o in occupations]),
        "amp": distribution([o["ap"] for o in occupations]),
        "top": most,
        "bottom": least,
        "skills": {
            "auto": driving_skills(occupations, scores, "a"),
            "amp": driving_skills(occupations, scores, "m"),
        },
        "parent": parent_of(key),
    }


def build_groups(occupations, labels, scores):
    """groups.json: every ISCO level plus ``all``, keyed ``"<level>:<code>"``."""
    members = group_members(occupations)
    groups = {}
    for key in sorted(members):
        entry = group_entry(key, members[key], labels, scores)
        entry["children"] = children_of(key, members)
        groups[key] = entry
    return groups


# --- search index ---------------------------------------------------------

def search_rows(occupations, labels):
    """One search_index.json row per occupation, before alternative labels."""
    return [
        {
            "t": o["t"], "s": o["s"], "c": o["c"],
            "mg": group_label(f"major:{o['c'][:1]}", labels),
            "a": o["ar"], "m": o["ap"], "q": o["q"], "alt": [],
        }
        for o in occupations
    ]


def alt_candidates(occupations, by_slug):
    """Every (rank, length, label, slug, row) candidate, best first.

    Rank leads, so every occupation gets its single most distinct label before
    any occupation gets a second one.
    """
    out = []
    for row, occupation in enumerate(occupations):
        for rank, label in enumerate(by_slug.get(occupation["s"], [])):
            out.append((rank, len(label), label, occupation["s"], row))
    return sorted(out)


def with_alt_labels(rows, candidates, keep):
    """``rows`` with the first ``keep`` candidates attached."""
    chosen = defaultdict(list)
    for candidate in candidates[:keep]:
        chosen[candidate[4]].append(candidate[2])
    return [dict(row, alt=sorted(chosen[i], key=label_order)) for i, row in enumerate(rows)]


def build_search_index(rows, candidates, budget_bytes):
    """search_index.json, filled with as many alternative labels as fit.

    The brief allows capping the labels to meet the size budget; this binary
    searches the number kept so the file lands just under ``budget_bytes``
    gzipped, which is both deterministic and the most synonyms we can afford.
    """
    low, high = 0, len(candidates)
    while low < high:
        middle = (low + high + 1) // 2
        payload = encode_compact(with_alt_labels(rows, candidates, middle))
        if gzipped_size(payload) <= budget_bytes:
            low = middle
        else:
            high = middle - 1
    return with_alt_labels(rows, candidates, low)


# --- skills ---------------------------------------------------------------

def skill_counts(occupations):
    """Skill id -> how many occupations need it, essential and optional."""
    essential, optional = Counter(), Counter()
    for occupation in occupations:
        essential.update(list(dict.fromkeys(occupation.get("se", []))))
        optional.update(list(dict.fromkeys(occupation.get("so", []))))
    return essential, optional


def build_skill_index(skills, counts):
    """skill_index.json: one row per scored skill, sorted by title."""
    essential, optional = counts
    rows = [
        {"id": sid, "t": skill["t"], "a": skill.get("a"), "m": skill.get("m"),
         "ne": essential.get(sid, 0), "no": optional.get(sid, 0)}
        for sid, skill in skills.items()
    ]
    return sorted(rows, key=lambda row: (row["t"], row["id"]))


def build_skill_occupations(occupations):
    """skill_occupations.json: skill id -> the slugs that need it."""
    index = defaultdict(lambda: {"e": [], "o": []})
    for occupation in sorted(occupations, key=lambda o: o["s"]):
        for skill in dict.fromkeys(occupation.get("se", [])):
            index[skill]["e"].append(occupation["s"])
        for skill in dict.fromkeys(occupation.get("so", [])):
            index[skill]["o"].append(occupation["s"])
    return {sid: index[sid] for sid in sorted(index)}


# --- stats ----------------------------------------------------------------

def is_near_line(occupation, threshold):
    """Within 0.5 of the quadrant cut-off on either axis."""
    gaps = (abs(occupation["ar"] - threshold), abs(occupation["ap"] - threshold))
    return min(gaps) <= NEAR_LINE


def build_stats(occupations, skills_scored, built, threshold):
    """stats.json: every number the pages quote. ``occupations`` must be non-empty."""
    total = len(occupations)
    counts = dict(sorted(Counter(o["q"] for o in occupations).items()))
    near = sum(1 for o in occupations if is_near_line(o, threshold))
    return {
        "built": built,
        "threshold": threshold,
        "occupations": total,
        "skills_scored": skills_scored,
        "quadrants": {
            "counts": counts,
            "shares": {name: round(n / total, 4) for name, n in counts.items()},
        },
        "near_line": {"count": near, "share": round(near / total, 4)},
    }
