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
from dataclasses import dataclass

from aiisco import rubric_v2, v2

try:  # pragma: no cover - exercised by the real build, not by the fixtures
    from aggregate_scores import QUADRANT_THRESHOLD
except ImportError:  # pragma: no cover
    QUADRANT_THRESHOLD = 6

#: How a set of scores classifies its occupations. The older sets cut two averaged
#: scores into quadrants; scoring v2 gives each occupation four skill-class shares
#: and a type. Pages read this out of stats.json and lay themselves out by it.
SCHEME_QUADRANTS = "quadrants"
SCHEME_SHARES = "shares"

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
#: ``search_index`` is the second: at 80 KB the fill kept 1.18 alternative labels
#: per occupation and "software engineer" found nothing, so the budget buys the
#: synonyms people actually type instead of the smallest file.
BUDGET_GZ_KB = {
    "search_index": 260,
    "groups": 150,
    "stats": 20,
    "skill_index": 300,
    "skill_occupations": None,
}

#: The share scheme puts more fields on every row of the skill index, and none of
#: them is derivable from the rest: the class, three probabilities, the machine
#: score, what the item is and how it is exercised. Its budget is the smallest
#: round number they fit in. The search budget is shared, so both schemes carry
#: the same alternative labels.
SHARES_BUDGET_GZ_KB = {**BUDGET_GZ_KB, "skill_index": 380}

#: How many ESCO alternative labels one occupation may contribute to the search
#: index. The fill is round-robin by rank, so every occupation gets its most
#: distinct label before any gets a second; the cap stops one many-synonymed
#: occupation from spending the whole budget.
MAX_ALT_LABELS = 12


def budgets_for(scheme):
    """The gzipped budgets in force for one scheme."""
    return SHARES_BUDGET_GZ_KB if scheme == SCHEME_SHARES else BUDGET_GZ_KB


WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class StatsContext:
    """What stats.json needs beyond the occupations and skills themselves."""

    built: str
    threshold: float = QUADRANT_THRESHOLD
    scheme: str = SCHEME_QUADRANTS
    model: str = ""


@dataclass(frozen=True)
class GroupInputs:
    """The lookups and the scheme every entry of groups.json is built with."""

    labels: dict
    scores: dict
    scheme: str = SCHEME_QUADRANTS


def scheme_of(portfolio):
    """How a portfolio dataset classifies its occupations."""
    return portfolio.get("scheme", SCHEME_QUADRANTS)


def stats_context(portfolio, built):
    """The stats context a portfolio dataset implies."""
    if scheme_of(portfolio) != SCHEME_SHARES:
        return StatsContext(built=built)
    return StatsContext(built=built, threshold=v2.CLASS_THRESHOLD,
                        scheme=SCHEME_SHARES, model=portfolio.get("model", ""))


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


def mean_shares(occupations):
    """The mean of each of the four shares over a group's occupations."""
    return [round(sum(o["sh"][i] for o in occupations) / len(occupations), 2)
            for i in range(len(v2.SHARE_NAMES))]


def group_shares(occupations, scheme):
    """The share-scheme additions to one group entry."""
    if scheme != SCHEME_SHARES:
        return {}
    return {"mech": distribution([o["ak"] for o in occupations]),
            "sh": mean_shares(occupations)}


def group_entry(key, occupations, inputs):
    """One value of groups.json."""
    most, least = extreme_slugs(occupations)
    return {
        "label": group_label(key, inputs.labels),
        "level": key.split(":", 1)[0] if key != ALL_KEY else ALL_KEY,
        "code": key.split(":", 1)[1] if key != ALL_KEY else "",
        "n": len(occupations),
        "near": near_the_line(occupations, inputs.scheme),
        "q": dict(sorted(Counter(o["q"] for o in occupations).items())),
        "auto": distribution([o["ar"] for o in occupations]),
        "amp": distribution([o["ap"] for o in occupations]),
        "top": most,
        "bottom": least,
        "skills": {
            "auto": driving_skills(occupations, inputs.scores, "a"),
            "amp": driving_skills(occupations, inputs.scores, "m"),
        },
        "parent": parent_of(key),
        **group_shares(occupations, inputs.scheme),
    }


def build_groups(occupations, labels, scores, scheme=SCHEME_QUADRANTS):
    """groups.json: every ISCO level plus ``all``, keyed ``"<level>:<code>"``."""
    inputs = GroupInputs(labels=labels, scores=scores, scheme=scheme)
    members = group_members(occupations)
    groups = {}
    for key in sorted(members):
        entry = group_entry(key, members[key], inputs)
        entry["children"] = children_of(key, members)
        groups[key] = entry
    return groups


# --- search index ---------------------------------------------------------

def row_shares(occupation, scheme):
    """The share-scheme additions to a search row: machine score, shares, near-line."""
    if scheme != SCHEME_SHARES:
        return {}
    return {"k": occupation["ak"], "sh": occupation["sh"], "nl": occupation["nl"]}


def search_rows(occupations, labels, scheme=SCHEME_QUADRANTS):
    """One search_index.json row per occupation, before alternative labels."""
    return [
        {
            "t": o["t"], "s": o["s"], "c": o["c"],
            "mg": group_label(f"major:{o['c'][:1]}", labels),
            "a": o["ar"], "m": o["ap"], "q": o["q"], "alt": [],
            **row_shares(o, scheme),
        }
        for o in occupations
    ]


def alt_candidates(occupations, by_slug, cap=MAX_ALT_LABELS):
    """Every (rank, length, label, slug, row) candidate, best first.

    Rank leads, so the fill is round-robin: every occupation gets its single most
    distinct label before any occupation gets a second one. ``cap`` is how many
    labels one occupation may offer at all.
    """
    out = []
    for row, occupation in enumerate(occupations):
        for rank, label in enumerate(by_slug.get(occupation["s"], [])[:cap]):
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


#: How a skill's kind and its mode of exercise are spelled in a compact row.
ITEM_KIND = {"knowledge": "k"}
SKILL_KIND = "s"
MODE_LETTER = {"on_things": "t", "with_people": "p", "directing_others": "d",
               "through_software": "s", "on_paper_in_place": "a"}


@dataclass(frozen=True)
class SkillInputs:
    """What every row of skill_index.json is built from besides the skill itself."""

    counts: tuple
    scheme: str = SCHEME_QUADRANTS
    answers: dict | None = None


def skill_shares(skill, scheme):
    """The share-scheme additions to a skill row: machine score, class, probabilities."""
    if scheme != SCHEME_SHARES or "c" not in skill:
        return {}
    return {"k": skill.get("k"), "c": skill["c"], "p": skill["p"]}


def skill_facets(entry):
    """What a skill is and how it is exercised, so the table can filter on them."""
    if not entry:
        return {}
    return {"ty": ITEM_KIND.get(entry["type"], SKILL_KIND),
            "mo": MODE_LETTER[entry["answers"]["mode"]["choice"]]}


def skill_index_row(sid, skill, inputs):
    """One row of skill_index.json: the skill, its scores and how many jobs need it."""
    essential, optional = inputs.counts
    return {"id": sid, "t": skill["t"], "a": skill.get("a"), "m": skill.get("m"),
            "ne": essential.get(sid, 0), "no": optional.get(sid, 0),
            **skill_shares(skill, inputs.scheme),
            **skill_facets((inputs.answers or {}).get(sid))}


def build_skill_index(skills, counts, scheme=SCHEME_QUADRANTS, answers=None):
    """skill_index.json: one row per scored skill, sorted by title."""
    inputs = SkillInputs(counts=counts, scheme=scheme, answers=answers)
    rows = [skill_index_row(sid, skill, inputs) for sid, skill in skills.items()]
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


# --- the rubric, as the pages read it --------------------------------------

def published_levels(question):
    """A graded question's five level descriptions, in level order."""
    return list(question.criteria)


def published_options(question):
    """A chosen question's options, in the order they are offered."""
    return [{"name": name, "text": text} for name, text in question.criteria.items()]


def published_answers(question):
    """Whichever of `levels` or `options` a question's kind calls for."""
    if question.kind == rubric_v2.SCORE:
        return {"levels": published_levels(question)}
    return {"options": published_options(question)}


def published_knowledge(name):
    """The knowledge-item variant of one question, or None when it has none."""
    variant = rubric_v2.KNOWLEDGE_VARIANTS.get(name)
    if variant is None:
        return None
    return {"instructions": variant.instructions,
            "levels": published_levels(variant)}


def published_question(name, question):
    """One question of site/rubric_v2.json."""
    return {"id": name,
            "kind": rubric_v2.PUBLISHED_KIND[question.kind],
            "label": question.label,
            "instructions": question.instructions,
            **published_answers(question),
            "knowledge": published_knowledge(name)}


def build_rubric(model):
    """rubric_v2.json: the questions exactly as asked, and how answers become a class.

    Generated from aiisco/rubric_v2.py rather than written out, so the page and
    the scorer cannot describe two different rubrics.
    """
    return {
        "model": model,
        "preamble": rubric_v2.PREAMBLE,
        "questions": [published_question(name, rubric_v2.QUESTIONS[name])
                      for name in rubric_v2.QUESTION_IDS],
        "classes": [{"code": code, "name": name, "rule": rule}
                    for code, name, rule in v2.CLASS_RULES],
        "display": v2.DISPLAY_FORMULA,
    }


# --- the stored answers, per skill -----------------------------------------

ANSWER_DECIMALS = 2

#: Question id -> the short key it is published under in an answers shard.
ANSWER_KEYS = {"ai_substitution": "s", "mechanical": "k", "complementarity": "c",
               "mode": "mo", "deployment": "dp"}
SHARD_PREFIX = 2


def rounded_list(probs):
    """A level distribution at the precision an answers shard publishes."""
    return [round(value, ANSWER_DECIMALS) for value in probs]


def rounded_map(probs):
    """An option distribution at the precision an answers shard publishes."""
    return {name: round(value, ANSWER_DECIMALS) for name, value in probs.items()}


def answer_record(entry):
    """One skill's stored answers, in the compact shape a detail view fetches."""
    answers = entry["answers"]
    graded = {ANSWER_KEYS[name]: rounded_list(answers[name]["probs"])
              for name in rubric_v2.GRADED_IDS}
    chosen = {ANSWER_KEYS[name]: rounded_map(answers[name]["probs"])
              for name in rubric_v2.CHOICE_IDS}
    confidence = {ANSWER_KEYS[name]: round(answers[name]["confidence"],
                                           ANSWER_DECIMALS)
                  for name in rubric_v2.GRADED_IDS + rubric_v2.CHOICE_IDS}
    return {"ty": ITEM_KIND.get(entry["type"], SKILL_KIND),
            "d": round(answers["digital_output"]["yes"], ANSWER_DECIMALS),
            **graded, **chosen, "cf": confidence}


def build_skill_answers(answers):
    """skill_answers/<xx>.json: every skill's answers, sharded by its id's prefix."""
    shards = defaultdict(dict)
    for sid in sorted(answers):
        shards[sid[:SHARD_PREFIX]][sid] = answer_record(answers[sid])
    return {name: shards[name] for name in sorted(shards)}


def skill_note(skill):
    """The rationale a compact skill record carries, and who wrote it."""
    return {key: skill[key] for key in ("r", "rf") if skill.get(key)}


def build_skill_notes(skills):
    """skill_notes/<xx>.json: the rationales, sharded like the answers.

    A page that shows one skill should not fetch fourteen megabytes of dataset to
    read one paragraph, so the paragraphs travel on their own.
    """
    shards = defaultdict(dict)
    for sid in sorted(skills):
        note = skill_note(skills[sid])
        if note:
            shards[sid[:SHARD_PREFIX]][sid] = note
    return {name: shards[name] for name in sorted(shards)}


# --- shard sizes ----------------------------------------------------------

def size_summary(sizes):
    """Median, 90th percentile, largest and total of some byte sizes, in KiB.

    ``sizes`` must not be empty; ``shard_report`` is what handles a family that
    turned out to have no files at all.
    """
    ordered = sorted(size / 1024 for size in sizes)
    return {"median": percentile(ordered, 0.5), "p90": percentile(ordered, 0.9),
            "max": ordered[-1], "total": sum(ordered)}


def shard_report(label, raw, packed):
    """One line describing a family of shards, which have no individual budget."""
    if not raw:
        return f"{label:<18}    0 files"
    plain, gz = size_summary(raw), size_summary(packed)
    return (
        f"{label:<18} {len(raw):4d} files  "
        f"raw med/p90/max {plain['median']:.1f}/{plain['p90']:.1f}/{plain['max']:.1f} KB  "
        f"gz {gz['median']:.1f}/{gz['p90']:.1f}/{gz['max']:.1f} KB  "
        f"total {plain['total']:.0f} KB raw, {gz['total']:.0f} KB gz"
    )


# --- stats ----------------------------------------------------------------

def is_near_line(occupation, scheme, threshold=QUADRANT_THRESHOLD):
    """Whether an occupation sits near its scheme's cut-off.

    Under quadrants that is within 0.5 of the threshold on either axis; under
    shares the roll-up already worked it out and published it as ``nl``.
    """
    if scheme == SCHEME_SHARES:
        return bool(occupation["nl"])
    gaps = (abs(occupation["ar"] - threshold), abs(occupation["ap"] - threshold))
    return min(gaps) <= NEAR_LINE


def near_the_line(occupations, scheme):
    """How many of these occupations sit near their scheme's cut-off."""
    return sum(1 for occupation in occupations if is_near_line(occupation, scheme))


def tally(codes, found):
    """Counts over a fixed set of codes, with each as a share of the total."""
    counts = {code: found.get(code, 0) for code in codes}
    total = sum(counts.values()) or 1
    return {"counts": counts,
            "shares": {code: round(n / total, 4) for code, n in counts.items()}}


def quadrant_stats(occupations, _skills, _context):
    """The quadrant counts and shares, over the quadrants that occur."""
    counts = dict(sorted(Counter(o["q"] for o in occupations).items()))
    total = len(occupations)
    return {"quadrants": {
        "counts": counts,
        "shares": {name: round(n / total, 4) for name, n in counts.items()},
    }}


def share_stats(occupations, skills, context):
    """The scheme, the model, the seven types and the four skill classes."""
    types = tally(v2.TYPE_ORDER, Counter(o["q"] for o in occupations))
    classes = tally(v2.SKILL_CLASSES,
                    Counter(s["c"] for s in skills.values() if "c" in s))
    return {"scheme": SCHEME_SHARES, "model": context.model,
            "types": {"order": list(v2.TYPE_ORDER), **types},
            "skill_classes": classes}


def build_stats(occupations, skills, context):
    """stats.json: every number the pages quote. ``occupations`` must be non-empty."""
    total = len(occupations)
    near = near_the_line(occupations, context.scheme)
    scheme_stats = (share_stats if context.scheme == SCHEME_SHARES
                    else quadrant_stats)
    return {
        "built": context.built,
        "threshold": context.threshold,
        "occupations": total,
        "skills_scored": len(skills),
        **scheme_stats(occupations, skills, context),
        "near_line": {"count": near, "share": round(near / total, 4)},
    }
