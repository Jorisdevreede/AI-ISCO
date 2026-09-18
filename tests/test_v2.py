"""Every rule of scoring v2, one threshold at a time.

Pure arithmetic, so every number here is chosen to sit on, just above or just
below a boundary. The thresholds themselves are read from the module rather than
typed out again: a test that restates a constant cannot notice it moving.
"""

from decimal import Decimal

import pytest

from aiisco import v2

# The worked example of docs/scoring-v2.md, as the scores file stores it.
ANSWERS = {
    "digital_output": {"yes": 0.93},
    "ai_substitution": {"probs": [0.0, 0.1, 0.3, 0.5, 0.1], "position": 2.6,
                        "confidence": 0.61},
    "mechanical": {"probs": [0.85, 0.1, 0.05, 0.0, 0.0], "position": 0.2,
                   "confidence": 0.8},
    "complementarity": {"probs": [0.0, 0.03, 0.3, 0.4, 0.27], "position": 2.9,
                        "confidence": 0.66},
    "mode": {"choice": "through_software",
             "probs": {"through_software": 0.8, "with_people": 0.2},
             "confidence": 0.8},
    "deployment": {"choice": "routine", "probs": {}, "confidence": 0.6},
}


def shares(substituted=0.0, assisted=0.0, mechanised=0.0, insulated=0.0):
    """One occupation's four shares, for the type rules to be tried against."""
    return {"substituted": substituted, "assisted": assisted,
            "mechanised": mechanised, "insulated": insulated}


def rows(*specs):
    """Roll-up rows from (class, mode, sub, weight) tuples.

    The mode is given as a name for readability and becomes a certain answer; the
    tests that care about an uncertain one pass a distribution instead.
    """
    return [v2.Contribution(klass=klass, modes=modes if isinstance(modes, dict)
                            else {modes: 1.0}, sub=sub, weight=weight)
            for klass, modes, sub, weight in specs]


# --- threshold probabilities ------------------------------------------------

@pytest.mark.parametrize(("level", "expected"), [
    (0, 1.0), (1, 0.9), (2, 0.7), (3, 0.4), (4, 0.1),
])
def test_p_at_least_sums_the_tail_of_the_distribution(level, expected):
    assert v2.p_at_least([0.1, 0.2, 0.3, 0.3, 0.1], level) == pytest.approx(expected)


def test_substitution_is_gated_by_the_digital_output_question():
    assert v2.derive_skill(ANSWERS)["sub"] == pytest.approx(0.6 * 0.93)


def test_a_skill_whose_result_is_never_digital_can_never_be_substituted():
    physical = {**ANSWERS, "digital_output": {"yes": 0.0}}
    assert v2.derive_skill(physical)["sub"] == 0.0


def test_complementarity_is_read_at_both_cut_offs():
    """`comp` is the class basis at level 3; `comp_part` is the reported alternative."""
    derived = v2.derive_skill(ANSWERS)
    assert derived["comp"] == pytest.approx(0.67)
    assert derived["comp_part"] == pytest.approx(0.97)


def test_the_machinery_axis_reads_its_own_distribution():
    assert v2.derive_skill(ANSWERS)["mech"] == 0.0


def test_the_cut_offs_are_the_ones_the_specification_names():
    assert (v2.SUB_LEVEL, v2.COMP_LEVEL, v2.COMP_PART_LEVEL, v2.MECH_LEVEL) == \
        (3, 3, 2, 3)
    assert v2.CLASS_THRESHOLD == 0.5


# --- the skill classes ------------------------------------------------------

@pytest.mark.parametrize(("sub", "comp", "mech", "expected"), [
    (0.9, 0.0, 0.0, "S"),
    (0.0, 0.9, 0.0, "A"),
    (0.0, 0.0, 0.9, "M"),
    (0.0, 0.0, 0.0, "I"),
    (0.49, 0.49, 0.49, "I"),
])
def test_each_class_is_reached_by_its_own_rule(sub, comp, mech, expected):
    assert v2.skill_class(sub, comp, mech) == expected


@pytest.mark.parametrize(("sub", "comp", "mech"), [
    (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
])
def test_a_probability_exactly_on_the_threshold_counts(sub, comp, mech):
    assert v2.skill_class(sub, comp, mech) != "I"


@pytest.mark.parametrize(("sub", "comp", "mech", "expected"), [
    (0.9, 0.9, 0.9, "S"),   # substitution wins over everything
    (0.1, 0.9, 0.9, "A"),   # assistance wins over machinery
    (0.1, 0.1, 0.9, "M"),
])
def test_the_first_matching_class_wins(sub, comp, mech, expected):
    assert v2.skill_class(sub, comp, mech) == expected


def test_the_class_can_be_recomputed_at_the_looser_complementarity_cut_off():
    score = {"sub": 0.1, "comp": 0.2, "comp_part": 0.8, "mech": 0.1}
    assert v2.class_of(score) == "I"
    assert v2.class_of(score, v2.COMP_PART_KEY) == "A"


def test_the_two_complementarity_cut_offs_name_their_own_field():
    assert v2.COMP_KEYS == {3: "comp", 2: "comp_part"}


# --- the display scores -----------------------------------------------------

def test_the_three_display_scores_are_the_weighted_level_on_a_one_to_ten_scale():
    assert v2.display_scores(ANSWERS) == {
        "automation_risk": 6.7,
        "amplification_potential": 7.3,
        "mechanical_automation": 1.9,
    }


def test_derive_skill_returns_every_published_field():
    assert set(v2.derive_skill(ANSWERS)) == {
        "sub", "comp", "comp_part", "mech", "class", "automation_risk",
        "amplification_potential", "mechanical_automation"}


# --- what the roll-up reads off a scored skill ------------------------------

def test_skill_extra_keeps_the_derived_fields_and_the_mode():
    entry = {"uri": "u", "sub": 0.1, "comp": 0.05, "comp_part": 0.2, "mech": 0.3,
             "class": "I", "mechanical_automation": 2.0, "answers": ANSWERS}
    assert v2.skill_extra(entry) == {
        "sub": 0.1, "comp": 0.05, "comp_part": 0.2, "mech": 0.3, "class": "I",
        "mechanical_automation": 2.0,
        "modes": {"through_software": 0.8, "with_people": 0.2}}


def test_skill_extra_tolerates_an_entry_without_answers():
    assert v2.skill_extra({"uri": "u", "class": "S"}) == {"class": "S", "modes": {}}


def test_a_contribution_takes_its_class_from_the_cut_off_it_is_given():
    score = {"sub": 0.1, "comp": 0.2, "comp_part": 0.8, "mech": 0.1,
             "modes": {"on_things": 1.0}}
    assert v2.contribution(score, 2.0).klass == "I"
    assert v2.contribution(score, 2.0, v2.COMP_PART_KEY).klass == "A"
    assert v2.contribution(score, 2.0).modes == {"on_things": 1.0}


# --- the four shares --------------------------------------------------------

def test_the_shares_weigh_essential_skills_twice_as_heavily():
    share = v2.shares(rows(("S", "through_software", 0.9, 2.0),
                           ("I", "on_things", 0.0, 1.0)))
    assert share == {"substituted": pytest.approx(2 / 3, abs=1e-4), "assisted": 0.0,
                     "mechanised": 0.0, "insulated": pytest.approx(1 / 3, abs=1e-4)}


def summed(share):
    """The four published shares added up at the precision they are published to."""
    return sum(Decimal(str(value)) for value in share.values())


@pytest.mark.parametrize("weights", [(2.0, 2.0, 2.0), (2.0, 1.0, 1.0),
                                     (1.0, 1.0, 2.0), (2.0, 1.0, 2.0),
                                     (1.0, 2.0, 2.0)])
def test_the_four_shares_always_sum_to_one(weights):
    share = v2.shares(rows(("S", "through_software", 0.9, weights[0]),
                           ("A", "through_software", 0.2, weights[1]),
                           ("I", "with_people", 0.0, weights[2])))
    assert summed(share) == 1


def test_the_largest_share_absorbs_the_rounding_residue():
    """Three equal thirds round to 0.3333 each, which is 0.0001 short of the whole."""
    share = v2.shares(rows(("S", "through_software", 0.9, 1.0),
                           ("A", "through_software", 0.2, 1.0),
                           ("I", "with_people", 0.0, 1.0)))
    assert summed(share) == 1
    assert sorted(share.values()) == [0.0, 0.3333, 0.3333, 0.3334]


def test_the_shares_become_the_two_decimal_list_the_site_files_carry():
    assert v2.share_list(shares(0.4242, 0.3131, 0.1616, 0.1011)) == \
        [0.42, 0.31, 0.16, 0.1]


# --- the spread of substitution ---------------------------------------------

def test_the_mean_and_standard_deviation_are_weighted():
    assert v2.spread(rows(("S", "x", 1.0, 3.0), ("I", "x", 0.0, 1.0))) == (0.75, 0.433)


def test_an_occupation_whose_skills_all_agree_has_no_spread():
    assert v2.spread(rows(("S", "x", 0.8, 2.0), ("S", "x", 0.8, 1.0))) == (0.8, 0.0)


# --- why the insulated skills are insulated ---------------------------------

@pytest.mark.parametrize(("mode", "expected"), [
    ("on_things", "physical"),
    ("with_people", "people"),
    ("directing_others", "people"),
    ("through_software", "other"),
    ("on_paper_in_place", "other"),
    ("", "other"),
])
def test_each_mode_maps_to_the_reason_the_site_shows(mode, expected):
    assert v2.why_insulated(rows(("I", mode, 0.0, 1.0))) == expected


def test_only_insulated_skills_are_asked_why():
    mixed = rows(("S", "on_things", 0.9, 5.0), ("I", "with_people", 0.0, 1.0))
    assert v2.why_insulated(mixed) == "people"


def test_the_reason_carrying_the_most_mass_decides():
    mostly_physical = rows(("I", "on_things", 0.0, 3.0), ("I", "with_people", 0.0, 1.0),
                           ("I", "directing_others", 0.0, 1.0))
    assert v2.why_insulated(mostly_physical) == "physical"
    assert v2.why_insulated(mostly_physical
                            + rows(("I", "with_people", 0.0, 2.0))) == "people"


def test_the_whole_distribution_is_weighed_not_the_answer_the_model_picked():
    """Two skills whose modal answer is physical, but whose mass is not."""
    hesitant = rows(("I", {"on_things": 0.4, "with_people": 0.35,
                           "directing_others": 0.25}, 0.0, 2.0))
    assert v2.why_insulated(hesitant) == "people"


def test_an_exact_tie_between_two_reasons_gives_neither():
    tied = rows(("I", "on_things", 0.0, 1.0), ("I", "with_people", 0.0, 1.0))
    assert v2.why_insulated(tied) == "other"


def test_the_mass_behind_every_reason_is_weighed_by_the_link_weight():
    assert v2.reason_mass(rows(("I", {"on_things": 0.5, "with_people": 0.5},
                                0.0, 2.0))) == {"physical": 1.0, "people": 1.0,
                                                "other": 0.0}


def test_an_unknown_mode_falls_to_other_rather_than_vanishing():
    assert v2.reason_mass(rows(("I", {"telepathy": 1.0}, 0.0, 1.0)))["other"] == 1.0


def test_an_occupation_with_nothing_insulated_gives_no_reason():
    assert v2.why_insulated(rows(("S", "through_software", 0.9, 2.0))) == "other"


# --- the seven types --------------------------------------------------------

def test_an_occupation_whose_skills_are_mostly_substituted_is_automation_heavy():
    assert v2.occupation_type(shares(substituted=0.5, insulated=0.5),
                              "other") == "AUTOMATION_HEAVY"


def test_how_far_apart_the_substituted_skills_are_no_longer_decides_the_type():
    """SUB is a threshold probability, so its spread says little; sigma gates nothing.

    A translator sits at 0.71 substituted with a large sigma, and used to fall
    through to Mixed for it.
    """
    spread_out = v2.roll_up(rows(("S", "through_software", 0.99, 5.0),
                                 ("I", "with_people", 0.0, 2.0)))
    assert spread_out["sigma"] > 0.2
    assert spread_out["type"] == "AUTOMATION_HEAVY"


def test_substitution_beside_assistance_is_transforming():
    assert v2.occupation_type(shares(substituted=0.3, assisted=0.2, insulated=0.5),
                              "other") == "TRANSFORMING"


def test_assistance_without_much_substitution_is_augmented():
    assert v2.occupation_type(shares(substituted=0.29, assisted=0.3, insulated=0.41),
                              "other") == "AUGMENTED"


def test_machinery_without_much_substitution_is_mechanisable():
    assert v2.occupation_type(shares(substituted=0.29, mechanised=0.3, insulated=0.41),
                              "other") == "MECHANISABLE"


def test_insulated_physical_work_has_its_own_type():
    assert v2.occupation_type(shares(substituted=0.2, insulated=0.8),
                              "physical") == "INSULATED_PHYSICAL"


def test_insulated_work_with_people_has_its_own_type():
    assert v2.occupation_type(shares(substituted=0.2, insulated=0.8),
                              "people") == "INSULATED_PEOPLE"


def test_an_insulated_occupation_for_neither_reason_falls_through_to_mixed():
    assert v2.occupation_type(shares(substituted=0.2, insulated=0.8),
                              "other") == "MIXED"


def test_an_occupation_that_matches_no_rule_is_mixed():
    assert v2.occupation_type(shares(0.29, 0.19, 0.29, 0.23), "physical") == "MIXED"


@pytest.mark.parametrize(("share", "why", "expected"), [
    # Automation-heavy is tried before transforming, which is tried before augmented.
    (shares(0.5, 0.3), "other", "AUTOMATION_HEAVY"),
    (shares(0.3, 0.3, 0.4), "other", "TRANSFORMING"),
    (shares(0.2, 0.4, 0.4), "other", "AUGMENTED"),
    # Insulated physical is tried before insulated people.
    (shares(0.1, 0.1, 0.1, 0.7), "physical", "INSULATED_PHYSICAL"),
])
def test_the_first_matching_type_rule_wins(share, why, expected):
    assert v2.occupation_type(share, why) == expected


def test_the_seven_types_are_tried_in_the_order_the_specification_lists():
    assert v2.TYPE_ORDER == ("AUTOMATION_HEAVY", "TRANSFORMING", "AUGMENTED",
                             "MECHANISABLE", "INSULATED_PHYSICAL",
                             "INSULATED_PEOPLE", "MIXED")


# --- near a cut-off ---------------------------------------------------------

def test_an_occupation_a_nudge_below_a_cut_off_is_near_the_line():
    assert v2.near_line(shares(substituted=0.26, assisted=0.3, insulated=0.44),
                        "other") is True


def test_an_occupation_far_from_every_cut_off_is_not_near_the_line():
    assert v2.near_line(shares(insulated=1.0), "people") is False


@pytest.mark.parametrize(("assisted", "near"), [(0.26, True), (0.2, False)])
def test_the_near_line_flag_turns_over_one_nudge_from_the_cut_off(assisted, near):
    """0.26 assisted is one nudge from the 0.30 that would make it augmented; 0.20 is two."""
    share = shares(substituted=0.1, assisted=assisted,
                   insulated=round(0.9 - assisted, 4))
    assert v2.near_line(share, "other") is near


def test_a_nudge_is_the_step_the_specification_names():
    assert v2.NEAR_LINE_STEP == 0.05


def test_a_nudge_cannot_push_a_share_outside_its_range():
    assert v2.nudged(shares(substituted=0.98), "substituted", 0.05)["substituted"] == 1.0
    assert v2.nudged(shares(substituted=0.02), "substituted", -0.05)["substituted"] == 0.0


# --- the whole roll-up ------------------------------------------------------

def test_roll_up_returns_every_field_an_occupation_record_needs():
    block = v2.roll_up(rows(("S", "through_software", 0.9, 2.0),
                            ("I", "with_people", 0.05, 2.0)))
    assert block == {
        "shares": {"substituted": 0.5, "assisted": 0.0, "mechanised": 0.0,
                   "insulated": 0.5},
        "mu": 0.475,
        "sigma": 0.425,
        "why_insulated": "people",
        "type": "AUTOMATION_HEAVY",
        "near_line": True,  # a nudge off 0.50 substituted leaves it insulated by people
    }


def test_the_spread_of_substitution_is_still_published():
    block = v2.roll_up(rows(("S", "through_software", 0.9, 2.0),
                            ("S", "through_software", 0.8, 2.0)))
    assert (block["mu"], block["sigma"]) == (0.85, 0.05)


def test_roll_up_can_be_asked_for_the_looser_complementarity_cut_off():
    score = {"sub": 0.1, "comp": 0.2, "comp_part": 0.8, "mech": 0.1,
             "modes": {"on_things": 1.0}}
    published = v2.roll_up([v2.contribution(score, 1.0)])
    alternative = v2.roll_up([v2.contribution(score, 1.0, v2.COMP_PART_KEY)])
    assert published["type"] == "INSULATED_PHYSICAL"
    assert alternative["type"] == "AUGMENTED"


def test_a_skill_class_maps_to_exactly_one_share():
    assert set(v2.SHARE_OF_CLASS) == set(v2.SKILL_CLASSES)
    assert tuple(v2.SHARE_OF_CLASS.values()) == v2.SHARE_NAMES
