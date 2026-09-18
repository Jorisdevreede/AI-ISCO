"""The weighted skill-to-occupation roll-up shared by the scoring scripts.

An occupation scores as the weighted average of its skills, essential ones
counting twice as heavily as optional ones, and lands in a quadrant by comparing
both averages with the same threshold. Occupation scores, site data and
portfolio data only agree with each other because they all roll up through here.
"""

import argparse
import re

ESSENTIAL_WEIGHT = 2.0
OPTIONAL_WEIGHT = 1.0
QUADRANT_THRESHOLD = 6

QUADRANTS = ("TRANSFORM", "SHRINK", "EVOLVE", "STABLE")

RELATIONS = (("essential_skills", "essential", ESSENTIAL_WEIGHT),
             ("optional_skills", "optional", OPTIONAL_WEIGHT))


def assign_quadrant(auto, amp):
    """Assign quadrant based on automation_risk and amplification_potential."""
    if auto >= QUADRANT_THRESHOLD:
        return "TRANSFORM" if amp >= QUADRANT_THRESHOLD else "SHRINK"
    return "EVOLVE" if amp >= QUADRANT_THRESHOLD else "STABLE"


def evolution_potential(auto_avg, amp_avg):
    """Combined score the site ranks on, kept on the same 1-10 scale."""
    return round((auto_avg * amp_avg) / 10, 1)


def scored_skills(occupation, lookup):
    """Yield (skill, score, weight, relation) for the skills `lookup` can score.

    `lookup` maps one skill reference to its score, or to None when the skill has
    no usable score; the caller decides what a score looks like. Essential skills
    come first, which is the order the top-N lists are built in.
    """
    for key, relation, weight in RELATIONS:
        for skill in occupation.get(key, []):
            score = lookup(skill)
            if score is not None:
                yield skill, score, weight, relation


def weighted_mean(contributions):
    """The weighted average of (value, weight) pairs, on the same 1-10 scale."""
    total = sum(weight for _, weight in contributions)
    return round(sum(value * weight for value, weight in contributions) / total, 1)


def weighted_averages(contributions):
    """Round the weighted averages of (automation, amplification, weight) triples.

    Returns None when nothing contributed, which is how an occupation without a
    single scored skill is dropped from every output.
    """
    total = sum(weight for _, _, weight in contributions)
    if total == 0:
        return None
    auto = sum(auto * weight for auto, _, weight in contributions) / total
    amp = sum(amp * weight for _, amp, weight in contributions) / total
    return round(auto, 1), round(amp, 1)


def scorable(entry):
    """True when a raw score entry has a URI and both axes filled in."""
    return bool(entry.get("uri")) and entry.get("automation_risk") is not None \
        and entry.get("amplification_potential") is not None


def index_skill_scores(entries, extra=None):
    """Map skill URI to its two scores, dropping entries missing either axis.

    `extra(entry)` may return further keys to keep per skill, which is how one
    caller keeps the skill title and another keeps the rationale.
    """
    scores = {}
    for entry in entries:
        if not scorable(entry):
            continue
        scored = {"automation_risk": float(entry["automation_risk"]),
                  "amplification_potential": float(entry["amplification_potential"])}
        if extra:
            scored.update(extra(entry))
        scores[entry["uri"]] = scored
    return scores


def hierarchy_level(hierarchy, index):
    """The ISCO label at `index`, or "" when the hierarchy does not go that deep."""
    return hierarchy[index] if len(hierarchy) > index else ""


def get_major_group(hierarchy):
    """The ISCO major group (1-digit level) label."""
    return hierarchy_level(hierarchy, 0)


def get_sub_major_group(hierarchy):
    """The ISCO sub-major group (2-digit level) label.

    The hierarchy list goes from broadest to most specific, so the sub-major
    group is its second element; it falls back to the major group.
    """
    if len(hierarchy) >= 2:
        return hierarchy[1]
    return hierarchy_level(hierarchy, 0)


def slugify(title):
    """Convert a title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")


#: Every scorer the pipeline can build from, and the file suffix each one writes.
#: Declared once here: three steps take --scorer and all three must agree on it.
SCORER_SUFFIX = {"gemini": "", "typesafe": "_typesafe", "v2": "_v2"}

SCORER_HELP = (
    "Which skill scores to build from (default: gemini). Every other scorer reads "
    "skill_scores_<scorer>.json and writes its outputs with a matching suffix, "
    "leaving the Gemini files untouched."
)


def add_scorer_argument(parser):
    """Add the --scorer flag the three pipeline steps share."""
    parser.add_argument("--scorer", choices=list(SCORER_SUFFIX), default="gemini",
                        help=SCORER_HELP)


def parse_scorer(description, argv=None):
    """The scorer named on the command line, for a step that takes only that flag."""
    parser = argparse.ArgumentParser(description=description)
    add_scorer_argument(parser)
    return parser.parse_args(argv).scorer
