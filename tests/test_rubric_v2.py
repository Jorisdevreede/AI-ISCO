"""The scoring v2 question set: which questions are asked, and in which words.

The digests below are the contract with the pilot that measured this rubric. A
question whose wording changes is a question whose retranslation and
self-consistency evidence no longer applies to it, so it has to fail here first.
"""

import hashlib

import pytest
from typesafe_sdk import Choice, Noul, Score

from aiisco import rubric_v2 as rubric

#: Questions the pilot measured and dropped. Asking any of them again would cost
#: requests on an answer that was near-constant, near-redundant, or guessed.
DROPPED = ("halves_the_time", "needs_software", "raises_novices", "tacit_context")


def criteria_text(criteria):
    """The answer descriptions of one question, as one string."""
    if isinstance(criteria, dict):
        return "".join(f"{name}\n{text}\n" for name, text in criteria.items())
    return "".join(f"{text}\n" for text in criteria)


def digest(question):
    """A stable hash over one question's instructions and its answer descriptions."""
    blob = question.instructions + "\n" + criteria_text(question.criteria)
    return hashlib.sha256(blob.encode()).hexdigest()


def asked(skill_type, name):
    """One question of the rubric for a skill type."""
    return rubric.rubric_for(skill_type)[name]


# --- which questions are asked ---------------------------------------------

def test_a_skill_is_asked_the_six_questions_of_the_specification():
    assert tuple(rubric.rubric_for("skill")) == (
        "digital_output", "ai_substitution", "mechanical", "complementarity",
        "mode", "deployment")


def test_a_knowledge_item_is_asked_the_same_six_ids():
    assert tuple(rubric.rubric_for("knowledge")) == rubric.QUESTION_IDS


def test_the_answers_file_is_written_in_the_order_the_questions_are_declared():
    assert rubric.QUESTION_IDS == tuple(rubric.QUESTIONS)


@pytest.mark.parametrize("name", DROPPED)
@pytest.mark.parametrize("skill_type", ["skill", "knowledge"])
def test_a_dropped_question_is_never_asked_again(name, skill_type):
    assert name not in rubric.rubric_for(skill_type)


def test_only_substitution_and_mechanical_have_a_knowledge_variant():
    assert tuple(rubric.KNOWLEDGE_VARIANTS) == ("ai_substitution", "mechanical")


def test_a_knowledge_item_keeps_the_skill_wording_everywhere_else():
    skill = rubric.rubric_for("skill")
    knowledge = rubric.rubric_for("knowledge")
    shared = [name for name in rubric.QUESTION_IDS
              if name not in rubric.KNOWLEDGE_VARIANTS]
    assert all(knowledge[name] is skill[name] for name in shared)


def test_a_knowledge_variant_replaces_the_skill_wording():
    for name in rubric.KNOWLEDGE_VARIANTS:
        assert asked("knowledge", name) is not asked("skill", name)


# --- the wording ------------------------------------------------------------

@pytest.mark.parametrize(("skill_type", "name", "expected"), [
    ("skill", "digital_output",
     "5076d5f0b2dc4132f4e0ebc9396ea779ec2d81e19bca824119e03abf4f1e4f65"),
    ("skill", "ai_substitution",
     "8b1d7225013b195f826bf6782473de81bf5909a279052edbda2f08d6cf0860d8"),
    ("skill", "mechanical",
     "39805c19908b65309233f6ec4ff940c6547fb5dda2ec8d3d2e7d113e4627fb82"),
    ("skill", "complementarity",
     "1875a5e5387a02655cb15497239de3444582d7ef95cd918175baa0e84445dece"),
    ("skill", "mode",
     "1f6ec4720700c88e753be7b70ced58904689d97c0730b9638bb080b6cd45378e"),
    ("skill", "deployment",
     "89c473d24b91113adaed590e143d79d6d07b71a69b4e1ab95ed50d67b8d8aa87"),
    ("knowledge", "ai_substitution",
     "06e393c55262c6249e60342dfa1f84fc36105b49e75b133232a9d942a57e69b4"),
    ("knowledge", "mechanical",
     "112916302f8e6f0c730e0635061c905224ebb18ef25dddb45eef9657da66156c"),
])
def test_the_measured_wording_is_unchanged(skill_type, name, expected):
    assert digest(asked(skill_type, name)) == expected


def test_every_question_that_mentions_ai_shares_the_capability_preamble():
    for name in ("ai_substitution", "complementarity"):
        assert asked("skill", name).instructions.startswith(rubric.PREAMBLE)
    assert asked("knowledge", "ai_substitution").instructions.startswith(rubric.PREAMBLE)


def test_the_machinery_question_sets_ai_aside_and_excludes_software():
    for skill_type in ("skill", "knowledge"):
        instructions = asked(skill_type, "mechanical").instructions
        assert "Set aside artificial intelligence entirely." in instructions
        assert "Software, apps, online services, navigation systems and artificial "\
               "intelligence are not machinery here" in instructions
        assert rubric.PREAMBLE not in instructions


def test_every_graded_question_offers_five_standalone_levels():
    for skill_type in ("skill", "knowledge"):
        for name in rubric.GRADED_IDS:
            assert len(asked(skill_type, name).criteria) == 5


# --- turning the data into SDK primitives -----------------------------------

def test_the_graded_and_chosen_questions_are_the_ones_the_rules_read():
    assert rubric.GRADED_IDS == ("ai_substitution", "mechanical", "complementarity")
    assert rubric.CHOICE_IDS == ("mode", "deployment")


@pytest.mark.parametrize(("name", "kind"), [
    ("digital_output", Noul), ("ai_substitution", Score), ("mechanical", Score),
    ("complementarity", Score), ("mode", Choice), ("deployment", Choice),
])
def test_each_question_becomes_its_matching_primitive(name, kind):
    assert isinstance(rubric.questions_for("skill")[name], kind)


def test_a_primitive_carries_the_wording_it_was_built_from():
    built = rubric.questions_for("knowledge")["mechanical"]
    assert built.instructions == asked("knowledge", "mechanical").instructions
    assert list(built.criteria) == list(rubric.MECHANICAL_KNOWLEDGE_LEVELS)


def test_the_primitives_are_built_once_per_skill_type():
    assert rubric.questions_for("skill") is rubric.questions_for("skill")
    assert rubric.questions_for("skill") is not rubric.questions_for("knowledge")


def test_an_unknown_skill_type_is_asked_the_plain_skill_wording():
    assert rubric.questions_for("") == rubric.questions_for("skill")


def test_options_for_lists_the_choices_in_the_order_they_are_offered():
    assert rubric.options_for("mode") == (
        "through_software", "on_things", "with_people", "on_paper_in_place",
        "directing_others")
    assert rubric.options_for("deployment") == (
        "routine", "available", "demonstrated", "not_shown")


def test_the_mode_options_cover_every_reason_the_roll_up_maps():
    assert set(rubric.options_for("mode")) == {
        "on_things", "with_people", "directing_others", "through_software",
        "on_paper_in_place"}


# --- the state a request judges ---------------------------------------------

def test_skill_state_trims_the_description():
    state = rubric.skill_state({"uri": "u", "title": "brew coffee",
                                "description": "  Prepare.  ", "type": "skill",
                                "reuse_level": "cross-sector"})
    assert state == {"skill": {"title": "brew coffee", "description": "Prepare.",
                               "type": "skill", "reuse_level": "cross-sector"}}


def test_skill_state_tolerates_a_skill_without_description_type_or_reuse_level():
    assert rubric.skill_state({"uri": "u", "title": "x"})["skill"] == {
        "title": "x", "description": "", "type": "", "reuse_level": ""}


def test_skill_state_tolerates_a_null_description():
    assert rubric.skill_state({"title": "x", "description": None})["skill"][
        "description"] == ""
