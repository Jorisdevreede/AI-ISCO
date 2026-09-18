"""Scoring v2: what a skill's answers mean, and what an occupation's skills add up to.

Pure arithmetic over the answers written to data/skill_scores_v2.json, specified
in docs/scoring-v2.md. Nothing here reads a file or calls a model, so every rule
can be checked one threshold at a time.

Two design rules carry most of the file. Derived quantities are threshold
probabilities, never averages of ordinal levels, because averaging ordinal ratings
can invert the true ordering. And every class is decided by a first-match-wins
list of rules over round numbers fixed before anyone looked at the output, so the
boundary cannot be tuned into agreement with a result.
"""

import math
from dataclasses import dataclass

from aiisco.systemone import to_ten_scale

#: Probabilities and shares are published to this many decimals.
DECIMALS = 4

# --------------------------------------------------------------------------
# Per skill
# --------------------------------------------------------------------------

SUB_LEVEL = 3        # "nearly all of it, under supervision" and above
COMP_LEVEL = 3       # "a clear gain across most of the work" and above
COMP_PART_LEVEL = 2  # "a clear gain on part of the work": reported, never published
MECH_LEVEL = 3       # "nearly all of it" and above

#: The one cut every class is decided by. MECH is cut here like the rest even
#: though the tuned machinery wording reads conservatively (robot welding lands
#: near 0.33): the cut was left where the others are rather than tuned to taste.
CLASS_THRESHOLD = 0.5

SUBSTITUTED = "S"
ASSISTED = "A"
MECHANISED = "M"
INSULATED = "I"
SKILL_CLASSES = (SUBSTITUTED, ASSISTED, MECHANISED, INSULATED)

COMP_KEY = "comp"
COMP_PART_KEY = "comp_part"
#: The field a roll-up reads complementarity from, by the level it is cut at.
#: Level 2 turned out degenerate on the full run - 62% of skills assisted, 86% of
#: occupations augmented - so it is kept as the reported alternative only.
COMP_KEYS = {COMP_LEVEL: COMP_KEY, COMP_PART_LEVEL: COMP_PART_KEY}

#: The derived fields of a scored skill that the roll-up and the site files read.
SCORE_FIELDS = ("sub", "comp", "comp_part", "mech", "class",
                "mechanical_automation")


def p_at_least(probs, level):
    """The probability of an answer at or above `level` in a 0-4 distribution."""
    return sum(probs[level:])


def skill_class(sub, comp, mech):
    """The one class a skill gets, first match wins."""
    if sub >= CLASS_THRESHOLD:
        return SUBSTITUTED
    if comp >= CLASS_THRESHOLD:
        return ASSISTED
    if mech >= CLASS_THRESHOLD:
        return MECHANISED
    return INSULATED


def display_scores(answers):
    """The three 1-10 numbers shown beside the older score sets.

    Display only: no class depends on them, which is why they may be the
    probability-weighted level while everything that classifies is a threshold.
    """
    return {
        "automation_risk": to_ten_scale(answers["ai_substitution"]["position"]),
        "amplification_potential": to_ten_scale(answers["complementarity"]["position"]),
        "mechanical_automation": to_ten_scale(answers["mechanical"]["position"]),
    }


def derive_skill(answers):
    """Every derived field of one scored skill, from its six answers."""
    complementarity = answers["complementarity"]["probs"]
    sub = round(p_at_least(answers["ai_substitution"]["probs"], SUB_LEVEL)
                * answers["digital_output"]["yes"], DECIMALS)
    comp = round(p_at_least(complementarity, COMP_LEVEL), DECIMALS)
    mech = round(p_at_least(answers["mechanical"]["probs"], MECH_LEVEL), DECIMALS)
    return {"sub": sub, "comp": comp,
            "comp_part": round(p_at_least(complementarity, COMP_PART_LEVEL), DECIMALS),
            "mech": mech, "class": skill_class(sub, comp, mech),
            **display_scores(answers)}


def class_of(score, comp_key=COMP_KEY):
    """The class of an indexed skill score, reading complementarity at one level."""
    return skill_class(score["sub"], score[comp_key], score["mech"])


def skill_extra(entry):
    """The v2 fields index_skill_scores keeps beside the two display scores.

    The mode distribution travels with the scores because the roll-up weighs the
    reasons an occupation is insulated by it; the rest of the answers do not.
    """
    kept = {name: entry[name] for name in SCORE_FIELDS if name in entry}
    kept["modes"] = entry.get("answers", {}).get("mode", {}).get("probs", {})
    return kept


# --------------------------------------------------------------------------
# Per occupation
# --------------------------------------------------------------------------

SHARE_NAMES = ("substituted", "assisted", "mechanised", "insulated")
SHARE_OF_CLASS = dict(zip(SKILL_CLASSES, SHARE_NAMES))

PHYSICAL = "physical"
PEOPLE = "people"
OTHER = "other"
WHY_REASONS = (PHYSICAL, PEOPLE, OTHER)
#: Why an insulated skill stays human, by the way its work is carried out. Every
#: mode is mapped, because the reasons are weighed by probability mass and an
#: unmapped mode would silently drop its share of the answer.
WHY_BY_MODE = {"on_things": PHYSICAL,
               "with_people": PEOPLE, "directing_others": PEOPLE,
               "through_software": OTHER, "on_paper_in_place": OTHER}

AUTOMATION_HEAVY = "AUTOMATION_HEAVY"
TRANSFORMING = "TRANSFORMING"
AUGMENTED = "AUGMENTED"
MECHANISABLE = "MECHANISABLE"
INSULATED_PHYSICAL = "INSULATED_PHYSICAL"
INSULATED_PEOPLE = "INSULATED_PEOPLE"
MIXED = "MIXED"

HEAVY_SUBSTITUTED = 0.50
TRANSFORMING_SUBSTITUTED = 0.30
TRANSFORMING_ASSISTED = 0.20
AUGMENTED_ASSISTED = 0.30
MECHANISABLE_MECHANISED = 0.30
LOW_SUBSTITUTED = 0.30
INSULATED_SHARE = 0.40

#: How far one share is moved to ask whether an occupation sits near a cut-off.
NEAR_LINE_STEP = 0.05

# The seven types, in the order they are tried. Mixed is the residual: its skills
# point in different directions, and a single label would mislead.
#
# Automation-heavy once also required sigma <= 0.20. That condition was dropped:
# SUB is a threshold probability, so it sits near 0 or near 1 by construction and
# its standard deviation is large for any occupation whose substituted share is
# between a half and four fifths - exactly the jobs the rule is about. Translator,
# accountant and data entry clerk all fell through to Mixed because of it.
TYPE_RULES = (
    (AUTOMATION_HEAVY,
     lambda s, why: s["substituted"] >= HEAVY_SUBSTITUTED),
    (TRANSFORMING,
     lambda s, why: (s["substituted"] >= TRANSFORMING_SUBSTITUTED
                     and s["assisted"] >= TRANSFORMING_ASSISTED)),
    (AUGMENTED,
     lambda s, why: (s["assisted"] >= AUGMENTED_ASSISTED
                     and s["substituted"] < LOW_SUBSTITUTED)),
    (MECHANISABLE,
     lambda s, why: (s["mechanised"] >= MECHANISABLE_MECHANISED
                     and s["substituted"] < LOW_SUBSTITUTED)),
    (INSULATED_PHYSICAL,
     lambda s, why: s["insulated"] >= INSULATED_SHARE and why == PHYSICAL),
    (INSULATED_PEOPLE,
     lambda s, why: s["insulated"] >= INSULATED_SHARE and why == PEOPLE),
)

TYPE_ORDER = tuple(code for code, _rule in TYPE_RULES) + (MIXED,)


@dataclass(frozen=True)
class Contribution:
    """One scored skill of an occupation, as the roll-up sees it.

    `modes` is the whole distribution over the five ways of working, not the one
    the model picked: the roll-up weighs reasons by probability mass.
    """

    klass: str
    modes: dict
    sub: float
    weight: float


def contribution(score, weight, comp_key=COMP_KEY):
    """One roll-up row from an indexed skill score and its link weight."""
    return Contribution(klass=class_of(score, comp_key), modes=score.get("modes", {}),
                        sub=score["sub"], weight=weight)


def balanced(rounded):
    """Give the largest share whatever the other three leave it.

    Rounding four shares independently can leave them summing to 0.9999 or
    1.0001, and the site draws a bar out of them. The largest share absorbs it,
    so the four published numbers sum to 1 at the precision they are published to.
    """
    biggest = max(SHARE_NAMES, key=lambda name: (rounded[name], name))
    others = sum(value for name, value in rounded.items() if name != biggest)
    return {**rounded, biggest: round(1.0 - others, DECIMALS)}


def shares(rows):
    """The weighted share of an occupation in each skill class; the four sum to 1."""
    total = sum(row.weight for row in rows)
    raw = dict.fromkeys(SHARE_NAMES, 0.0)
    for row in rows:
        raw[SHARE_OF_CLASS[row.klass]] += row.weight / total
    return balanced({name: round(value, DECIMALS) for name, value in raw.items()})


def spread(rows):
    """The weighted mean and standard deviation of SUB over an occupation's skills."""
    total = sum(row.weight for row in rows)
    mu = sum(row.sub * row.weight for row in rows) / total
    variance = sum(row.weight * (row.sub - mu) ** 2 for row in rows) / total
    return round(mu, DECIMALS), round(math.sqrt(variance), DECIMALS)


def reason_mass(rows):
    """Weighted probability mass behind each reason, over the insulated skills."""
    mass = dict.fromkeys(WHY_REASONS, 0.0)
    for row in rows:
        if row.klass != INSULATED:
            continue
        for mode, probability in row.modes.items():
            mass[WHY_BY_MODE.get(mode, OTHER)] += row.weight * probability
    return mass


def why_insulated(rows):
    """Why the insulated skills stay human: the reason carrying the most mass.

    Read off the whole distribution rather than each skill's modal answer. The
    argmax threw most of the answer away and let the integer link weights tie, and
    a tie broken by anything other than the evidence decides a published type.

    An exact tie, and an occupation with nothing insulated, give "other"; no type
    rule can reach either, because both insulated types need a share of at least
    INSULATED_SHARE first.
    """
    mass = reason_mass(rows)
    leaders = [reason for reason, value in mass.items() if value == max(mass.values())]
    return leaders[0] if len(leaders) == 1 else OTHER


def occupation_type(share, why):
    """The one type an occupation gets, first match wins.

    Only the four shares and the reason its insulated skills are insulated decide
    it. `mu` and `sigma` are still computed and published; no rule reads them.
    """
    for code, matches in TYPE_RULES:
        if matches(share, why):
            return code
    return MIXED


def nudged(share, name, step):
    """The four shares with one of them moved, kept inside [0, 1]."""
    return {**share, name: min(1.0, max(0.0, round(share[name] + step, DECIMALS)))}


def near_line(share, why):
    """True when moving any one share by NEAR_LINE_STEP would change the type."""
    settled = occupation_type(share, why)
    return any(occupation_type(nudged(share, name, step), why) != settled
               for name in SHARE_NAMES
               for step in (NEAR_LINE_STEP, -NEAR_LINE_STEP))


def roll_up(rows):
    """The v2 block of an occupation record, from the rows of its scored skills."""
    share = shares(rows)
    mu, sigma = spread(rows)
    why = why_insulated(rows)
    return {"shares": share, "mu": mu, "sigma": sigma, "why_insulated": why,
            "type": occupation_type(share, why),
            "near_line": near_line(share, why)}


def share_list(share):
    """The four shares as the two-decimal list the compact site files carry."""
    return [round(share[name], 2) for name in SHARE_NAMES]


# --------------------------------------------------------------------------
# What the pages are told about the rules
# --------------------------------------------------------------------------

#: The four classes as site/rubric_v2.json publishes them, in the order they are
#: tried. The rule is the arithmetic, spelled out for a reader.
CLASS_RULES = (
    (SUBSTITUTED, "Substituted", "SUB >= 0.5"),
    (ASSISTED, "Assisted", "COMP >= 0.5, and not already substituted"),
    (MECHANISED, "Mechanised", "MECH >= 0.5, and neither substituted nor assisted"),
    (INSULATED, "Insulated", "none of the above"),
)

DISPLAY_FORMULA = "1.5 + 2.0 * position"
