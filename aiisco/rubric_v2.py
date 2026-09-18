"""The scoring v2 question set, as data.

Six questions per ESCO skill, specified in docs/scoring-v2.md and worded by the
pilot that measured them (retranslation gate, three-run self-consistency, a
labelled probe set for `mechanical`). The wording lives here as plain strings and
lists, and nothing else in the module knows what any question says, so changing a
question is a data edit: rewrite the constant, leave the code alone.

Two questions have a knowledge variant. ESCO mixes activities ("manage musical
staff") with knowledge areas ("thermodynamics"), and asking whether AI or a
machine can do "thermodynamics" is ill-posed: the answer describes what machines
embody, not what work people stop doing. The variants ask about the work the
knowledge is applied in instead.

`deployment` is descriptive colour only. It asks about the state of the world at a
point in time, answered from a training set with a cutoff, so no class boundary
may depend on it.
"""

from dataclasses import dataclass
from functools import cache

from typesafe_sdk import Choice, Noul, Score

KNOWLEDGE_TYPE = "knowledge"

# --------------------------------------------------------------------------
# The shared capability preamble
# --------------------------------------------------------------------------
# Adapted from the Eloundou et al. exposure rubric preamble (arXiv:2303.10130,
# Appendix A.1), whose load-bearing sentence is "You do not have access to any
# other physical tools or materials". This one paragraph is what stops "sift
# powder" scoring at the top of an AI axis.

PREAMBLE = (
    "Consider a current general-purpose AI system: a large language model that reads and writes "
    "text, code, tables and documents, interprets images, searches and retrieves information, and "
    "calls other software. It runs on ordinary computers. It has no body, no hands, no eyes in a "
    "room and no physical presence. It can control other software; it cannot control machinery, "
    "vehicles, tools or materials unless that machinery is already computer-controlled and exposed "
    "through software. Assume a competent worker in the role has this system available, along with "
    "the software and computer hardware they already use, and no other new physical equipment. "
)

# --------------------------------------------------------------------------
# digital_output: the gate on AI substitution
# --------------------------------------------------------------------------

DIGITAL_OUTPUT_INSTRUCTIONS = (
    "Is the thing this skill produces an artefact that exists as text, code, numbers, images, "
    "audio, or a document or record - rather than a change in the physical world, a change in "
    "another person, or a physical act performed in a place?"
)

DIGITAL_OUTPUT_CRITERIA = {
    "true": (
        "What the worker hands over is information: a written document, a decision recorded in "
        "a system, a drawing or model, a calculation, a piece of code, an image, a plan, a "
        "diagnosis, an answer."
    ),
    "false": (
        "What the worker hands over is a physical change - an object made, moved, repaired, "
        "cut, cleaned or installed; a body treated; a vehicle or machine operated - or an "
        "effect on another person that happens in their presence."
    ),
}

# --------------------------------------------------------------------------
# ai_substitution: how much of the work the system carries out itself
# --------------------------------------------------------------------------
# Five levels, every level a standalone description on the O*NET behavioural
# anchor pattern. A System One model judges each level in isolation, so "more than
# the previous level" would be useless.

SUBSTITUTION_LEVELS = [
    ("None of it. The system cannot produce any part of the result, because the work happens in the "
     "physical world, in someone's presence, or depends on information the system cannot be given."),
    ("A minor part. The system can draft, look up or summarise something the worker then uses, but "
     "the worker still does the work and produces the result themselves."),
    ("About half. The system produces a usable first version of the result, and the worker completes "
     "it, corrects it and takes it the rest of the way."),
    ("Nearly all of it, under supervision. The system produces the finished result, and a qualified "
     "person reads it, checks it and signs it off before it is used."),
    ("All of it, unsupervised. The system produces the finished result and it is used as it stands, "
     "without a person reading it first."),
]

SUBSTITUTION_INSTRUCTIONS = (
    PREAMBLE + "How much of the work of this skill can that system carry out itself, from the "
    "inputs a worker would give it to the finished result?"
)

KNOWLEDGE_SUBSTITUTION_LEVELS = [
    ("None of it. This knowledge is applied by acting physically on things or people, and the system "
     "cannot act."),
    ("A minor part. The system can supply the relevant facts, and a person still does the work of "
     "applying them."),
    ("About half. The system applies this knowledge to produce a first answer, and a person judges "
     "whether it fits the case in front of them."),
    ("Nearly all of it, under supervision. The system applies this knowledge to real cases and "
     "produces the decision or document, which a qualified person approves."),
    ("All of it, unsupervised. The system applies this knowledge to real cases and the result is "
     "acted on without a person reviewing it."),
]

KNOWLEDGE_SUBSTITUTION_INSTRUCTIONS = (
    PREAMBLE + "The item below is a body of knowledge that workers apply, not an activity. Do "
    "not judge whether the system can recite these facts - assume it can. Judge the work that "
    "people use this knowledge to do. How much of that work can the system carry out itself?"
)

# --------------------------------------------------------------------------
# mechanical: the machinery axis, with AI set aside
# --------------------------------------------------------------------------
# The separation that fixes the v1 defect: v1 put plant and machine operators
# second-highest on "AI automation risk", where every published measure puts them
# near the bottom. The instructions are assembled from four paragraphs because the
# exclusions have to be repeated: the pilot's false positives were all cases where
# a software capability (satnav, speech synthesis) was read as machinery, or where
# a human faculty was called mechanised because a device resembles it.

MECHANICAL_FRAME = (
    "Set aside artificial intelligence entirely. Think only about machines: production "
    "equipment, robots, conveyors, automated inspection rigs, vehicles that drive themselves, "
    "plant that runs a process on its own. "
)

MECHANICAL_EXCLUSION = (
    "Count only physical equipment that moves, shapes, joins, carries, sorts, fills, processes or "
    "inspects material things: production lines, CNC machines, robots, conveyors, automated "
    "storage, process plant, farm machinery, self-driving vehicles, automated checkouts. Software, "
    "apps, online services, navigation systems and artificial intelligence are not machinery here "
    "and belong to a different question. Work whose job is to tend, feed or monitor such equipment "
    "counts as mechanised work - that is what level 3 describes. "
)

MECHANICAL_HUMAN_FACULTY = (
    "A human ability or performance - perceiving, singing, judging, caring, persuading - is not "
    "mechanised merely because some device can do something similar; answer for how the work this "
    "skill belongs to is actually carried out in equipped workplaces. "
)

MECHANICAL_ASK_SKILL = (
    "How much of the work of this skill can machinery like that do, in workplaces that can afford "
    "it?"
)

MECHANICAL_ASK_KNOWLEDGE = (
    "How much of the work that people apply this knowledge to can machinery like that do, in "
    "workplaces that can afford it?"
)

MECHANICAL_KNOWLEDGE_FRAME = (
    "The item below is a body of knowledge that workers apply, not an activity. Do not judge "
    "whether machines embody this knowledge or could recite it - judge the work that people use it "
    "to do. "
)

MECHANICAL_LEVELS = [
    ("None of it. There is no machine that does this work; it is carried out by a person deciding or "
     "interacting, not by equipment running."),
    ("A small part. Machinery helps with one step - lifting, moving, measuring - and the worker does "
     "the rest by hand."),
    ("About half. Machinery performs the repetitive core of the work while a person sets it up, feeds "
     "it, watches it and handles what it cannot."),
    ("Nearly all of it. A machine performs the work while a person monitors it, intervenes when it "
     "stops and deals with exceptions."),
    ("All of it. Equipment performs this work continuously without a person present, and workplaces "
     "that have installed it no longer employ anyone to do it by hand."),
]

MECHANICAL_KNOWLEDGE_LEVELS = [
    ("None of it. There is no machine that does the work this knowledge is applied to; it is "
     "carried out by a person deciding or interacting, not by equipment running."),
    ("A small part. Machinery helps with one step of that work - lifting, moving, measuring - and "
     "the worker does the rest by hand."),
    ("About half. Machinery performs the repetitive core of that work while a person sets it up, "
     "feeds it, watches it and handles what it cannot."),
    ("Nearly all of it. A machine performs that work while a person monitors it, intervenes when it "
     "stops and deals with exceptions."),
    ("All of it. Equipment performs that work continuously without a person present, and workplaces "
     "that have installed it no longer employ anyone to do it by hand."),
]

MECHANICAL_INSTRUCTIONS = (
    MECHANICAL_FRAME + MECHANICAL_EXCLUSION + MECHANICAL_HUMAN_FACULTY + MECHANICAL_ASK_SKILL
)

MECHANICAL_KNOWLEDGE_INSTRUCTIONS = (
    MECHANICAL_KNOWLEDGE_FRAME + MECHANICAL_FRAME + MECHANICAL_EXCLUSION
    + MECHANICAL_HUMAN_FACULTY + MECHANICAL_ASK_KNOWLEDGE
)

# --------------------------------------------------------------------------
# complementarity: how much better the work gets when the person keeps it
# --------------------------------------------------------------------------
# The levels grade HOW MUCH OF THE WORK is affected, not how impressive the help
# sounds: the pilot's first wording saturated, 62% of answers landing on one
# level. v1's amplification axis had the same top-end defect, so "read the level
# that sounds most impressive" is the failure mode this wording designs against.
# The multiples v1 quoted ("two to five times", "ten or more") were fabricated
# precision; the RCT evidence finds effects in the tens of percent.

COMPLEMENTARITY_LEVELS = [
    ("No difference. The person works the same way at the same pace. There is nothing in this work "
     "the system can take part in."),
    ("A little, on a minor part. The system saves some time on something small - a lookup, some "
     "typing, a bit of formatting or tidying - and the work as a whole takes about as long as it "
     "did."),
    ("A clear gain on part of the work. Drafting, searching or checking is now noticeably faster or "
     "noticeably better because the system does it, but the core of this work still takes the "
     "person as long as it always did."),
    ("A clear gain across most of the work. Most of what this work consists of, not one part of it, "
     "goes noticeably faster or comes out noticeably better with the system available."),
    ("A change in what the person can take on. They get through several times as much of this work "
     "as before, or they now do work that was previously too slow or too expensive to attempt at "
     "all."),
]

COMPLEMENTARITY_INSTRUCTIONS = (
    PREAMBLE + "Now assume the work stays with the person and the system only assists. Judge "
    "how much of this work is affected, not how impressive the assistance sounds: a large "
    "improvement to one small part of the work is a lower level than a clear improvement "
    "across most of it. How much better does the work get?"
)

# --------------------------------------------------------------------------
# mode: why a skill is the way it is, in five kinds of work
# --------------------------------------------------------------------------
# The pilot asked this per occupation-skill link, where it was the most confident
# question in the set; the full run is skill-level only, so it is asked without
# the occupation. The roll-up reads it for one thing: a context-free reason why an
# occupation's insulated skills are insulated.

MODE_INSTRUCTIONS = (
    "How is the work of this skill mainly carried out? Choose the way it is mostly done."
)

MODE_OPTIONS = {
    "through_software": (
        "The worker does this at a computer, in a system, a document or an application. The "
        "work is reading and writing information."
    ),
    "on_things": (
        "The worker does this with their hands, with tools, machines, materials or vehicles, in "
        "a physical place."
    ),
    "with_people": (
        "The worker does this in the presence of other people, by talking with them, examining "
        "them, caring for them, teaching them or persuading them."
    ),
    "on_paper_in_place": (
        "The worker does this by hand on paper or on a form, at a site or on the move, away "
        "from a computer."
    ),
    "directing_others": (
        "The worker does this by deciding and instructing, and other people carry it out."
    ),
}

# --------------------------------------------------------------------------
# deployment: descriptive only, never an input to a class
# --------------------------------------------------------------------------

DEPLOYMENT_INSTRUCTIONS = (
    "Where does software that does this work stand in the real world today - not what is "
    "possible in principle, but what organisations are actually using?"
)

DEPLOYMENT_OPTIONS = {
    "routine": (
        "Organisations already run software that does this work, and it is ordinary rather than "
        "remarkable to find it in use."
    ),
    "available": (
        "Products that do this work can be bought today, and some organisations use them, but "
        "most doing this work still do it without such software."
    ),
    "demonstrated": (
        "It has been shown to work in research, in demonstrations or in pilots, but there is no "
        "product an ordinary organisation can buy and run."
    ),
    "not_shown": (
        "Nobody has shown software doing this work to a standard that would be accepted in a "
        "workplace."
    ),
}


# --------------------------------------------------------------------------
# The question set
# --------------------------------------------------------------------------

NOUL = "noul"
SCORE = "score"
CHOICE = "choice"


#: How each kind of question is named to the pages, which do not know the SDK.
PUBLISHED_KIND = {NOUL: "yesno", SCORE: "levels", CHOICE: "choice"}


@dataclass(frozen=True)
class Question:
    """One question of the rubric: how it is answered, what is asked, what the answers mean.

    `criteria` is a list of level descriptions for a score, and a mapping of
    outcome name to description for a noul or a choice. `label` is the short plain
    name a page prints above the question; the instructions are what was asked.
    """

    kind: str
    label: str
    instructions: str
    criteria: object


QUESTIONS = {
    "digital_output": Question(NOUL, "Digital output", DIGITAL_OUTPUT_INSTRUCTIONS,
                               DIGITAL_OUTPUT_CRITERIA),
    "ai_substitution": Question(SCORE, "AI substitution", SUBSTITUTION_INSTRUCTIONS,
                                SUBSTITUTION_LEVELS),
    "mechanical": Question(SCORE, "Machine automation", MECHANICAL_INSTRUCTIONS,
                           MECHANICAL_LEVELS),
    "complementarity": Question(SCORE, "AI assistance", COMPLEMENTARITY_INSTRUCTIONS,
                                COMPLEMENTARITY_LEVELS),
    "mode": Question(CHOICE, "How it is exercised", MODE_INSTRUCTIONS, MODE_OPTIONS),
    "deployment": Question(CHOICE, "Deployment", DEPLOYMENT_INSTRUCTIONS,
                           DEPLOYMENT_OPTIONS),
}

KNOWLEDGE_VARIANTS = {
    "ai_substitution": Question(SCORE, "AI substitution",
                                KNOWLEDGE_SUBSTITUTION_INSTRUCTIONS,
                                KNOWLEDGE_SUBSTITUTION_LEVELS),
    "mechanical": Question(SCORE, "Machine automation",
                           MECHANICAL_KNOWLEDGE_INSTRUCTIONS,
                           MECHANICAL_KNOWLEDGE_LEVELS),
}

#: Every question id, in the order answers are written to the scores file.
QUESTION_IDS = tuple(QUESTIONS)

#: The graded questions, the only ones a derived threshold probability reads.
GRADED_IDS = tuple(name for name, q in QUESTIONS.items() if q.kind == SCORE)

#: The questions answered by picking a named option.
CHOICE_IDS = tuple(name for name, q in QUESTIONS.items() if q.kind == CHOICE)

PRIMITIVES = {NOUL: Noul, SCORE: Score, CHOICE: Choice}


def primitive(question):
    """The SDK primitive that asks one question of the rubric."""
    return PRIMITIVES[question.kind](instructions=question.instructions,
                                     criteria=question.criteria)


def rubric_for(skill_type):
    """The six questions for a skill type, with the knowledge variants where they apply."""
    if skill_type == KNOWLEDGE_TYPE:
        return {**QUESTIONS, **KNOWLEDGE_VARIANTS}
    return dict(QUESTIONS)


@cache
def questions_for(skill_type):
    """The rubric as SDK primitives, built once per skill type and shared after that."""
    return {name: primitive(question) for name, question in rubric_for(skill_type).items()}


def options_for(name):
    """The answer labels of a choice question, in the order they are offered."""
    return tuple(QUESTIONS[name].criteria)


def skill_state(skill):
    """The state a System One request judges: one skill, trimmed."""
    return {
        "skill": {
            "title": skill["title"],
            "description": (skill.get("description") or "").strip(),
            "type": skill.get("type", ""),
            "reuse_level": skill.get("reuse_level", ""),
        }
    }
