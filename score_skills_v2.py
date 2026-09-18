"""
Score each ESCO skill against the scoring v2 rubric with TypeSafe's System One.

One request per skill asks all six questions of docs/scoring-v2.md over the same
state, and the answers arrive typed: a probability per level, not prose to parse.
The questions themselves live in aiisco/rubric_v2.py and the arithmetic over the
answers in aiisco/v2.py, so this script only moves data between them.

What it writes, per skill, to data/skill_scores_v2.json: the whole distribution of
every answer, the three threshold probabilities derived from them (SUB, COMP,
MECH), the skill's class, and the three 1-10 display scores. No rationale, because
System One models return judgments rather than text, and no prices: the run
reports requests and input tokens and leaves money to whoever pays the bill.

The model is pinned rather than flagged: the version is an index parameter, and
the jev-latest alias moves. The API key is read from TYPESAFE_API_KEY (.env
works), falling back to the macOS login keychain item "typesafe-api-key"; it never
reaches a command line, a log or the output. The run checkpoints every 250 skills
and on the way out, resumes from what it wrote rather than re-asking, and stops
itself after twenty consecutive failures.

This project is not affiliated with or endorsed by TypeSafe AI, Inc.

A rule change costs nothing: --rederive recomputes every derived field from the
answers already on disk, so a threshold moves without a single request.

Usage:
    uv run python score_skills_v2.py
    uv run python score_skills_v2.py --sample 300
    uv run python score_skills_v2.py --start 0 --end 50
    uv run python score_skills_v2.py --workers 6 --rps 10
    uv run python score_skills_v2.py --rederive
"""

import argparse
import time
from collections import Counter
from functools import partial

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from aiisco import rubric_v2, v2
from aiisco.checkpoint import (
    index_by_uri,
    load_checkpoint,
    pending,
    save_checkpoint_atomically,
)
from aiisco.jsonio import load_json
from aiisco.systemone import (
    PoolConfig,
    add_run_arguments,
    load_api_key,
    run_pool,
    select_items,
)

load_dotenv()

MODEL = "jev-1.13.0"  # pinned: the jev-latest alias moves
INPUT_FILE = "data/esco_skills.json"
OUTPUT_FILE = "data/skill_scores_v2.json"
CHECKPOINT_EVERY = 250

PROB_DECIMALS = 4
CONFIDENCE_DECIMALS = 3


# ---------------------------------------------------------------------------
# Answers in
# ---------------------------------------------------------------------------

def noul_answer(answer):
    """A yes/no answer: the probability of yes, and nothing else.

    A NoulAnswer carries no confidence and no value field; its distance from 0.5
    is the only certainty signal there is.
    """
    return {"yes": round(answer.noul, PROB_DECIMALS)}


def score_answer(answer):
    """A graded answer: the whole distribution over the five levels, in level order."""
    return {
        "probs": [round(answer.probabilities[level], PROB_DECIMALS)
                  for level in sorted(answer.probabilities)],
        "position": round(answer.score, PROB_DECIMALS),
        "confidence": round(answer.confidence, CONFIDENCE_DECIMALS),
    }


def choice_answer(answer, options):
    """A picked option, with every option's probability in the order they are offered."""
    return {
        "choice": answer.choice,
        "probs": {name: round(answer.probabilities.get(name, 0.0), PROB_DECIMALS)
                  for name in options},
        "confidence": round(answer.confidence, CONFIDENCE_DECIMALS),
    }


def serialise(name, answer):
    """One answer in the shape the scores file publishes."""
    if answer.type == rubric_v2.NOUL:
        return noul_answer(answer)
    if answer.type == rubric_v2.CHOICE:
        return choice_answer(answer, rubric_v2.options_for(name))
    return score_answer(answer)


def answers_of(response):
    """Every answer of one response, always in the rubric's own order."""
    return {name: serialise(name, response.answers[name])
            for name in rubric_v2.QUESTION_IDS}


def score_skill(client, limiter, skill):
    """Ask all six questions about one skill in a single request."""
    limiter.wait()
    response = client.system_one(
        state=rubric_v2.skill_state(skill),
        questions=rubric_v2.questions_for(skill.get("type", "")),
        model=MODEL,
    )
    answers = answers_of(response)
    return {"uri": skill["uri"], "title": skill["title"],
            "type": skill.get("type", ""), "answers": answers,
            **v2.derive_skill(answers),
            "model": response.model, "input_tokens": response.usage.input_tokens}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save(scored):
    """Checkpoint the scores collected so far, through a temporary file."""
    save_checkpoint_atomically(OUTPUT_FILE, scored, indent=1)


def rederived(entry):
    """One entry with its derived fields recomputed from its stored answers.

    Rebuilt rather than updated, so a field a rule change has retired leaves the
    file instead of lingering beside its replacement.
    """
    return {"uri": entry["uri"], "title": entry["title"], "type": entry["type"],
            "answers": entry["answers"], **v2.derive_skill(entry["answers"]),
            "model": entry["model"], "input_tokens": entry["input_tokens"]}


def rederive():
    """Recompute every derived field from the answers already on disk.

    A change to a threshold or a class rule is arithmetic over answers that were
    paid for once. This re-reads them, rewrites the file in the order it already
    has, and asks the model nothing: no requests, no key, no cost.
    """
    scored = index_by_uri([rederived(entry) for entry in load_json(OUTPUT_FILE)])
    save(scored)
    print(f"Rederived {len(scored)} skills in {OUTPUT_FILE}")
    print_class_distribution(scored)


def print_class_distribution(scored):
    """Print how the skills scored so far fell into the four classes."""
    if not scored:
        print("\nNothing was scored, so there is no distribution to show.")
        return
    counts = Counter(entry["class"] for entry in scored.values())
    print(f"\nSkill classes across {len(scored)} skills:")
    for code in v2.SKILL_CLASSES:
        found = counts.get(code, 0)
        print(f"  {code}: {found:6d} ({found / len(scored) * 100:5.1f}%)")


def print_summary(run):
    """Print the closing totals: what was scored, what failed, what it cost."""
    elapsed = time.monotonic() - run.started
    print(f"\nDone in {elapsed:.0f}s. Skills scored: {len(run.scored)}, "
          f"errors: {len(run.errors)}.")
    if run.stopped:
        print(f"Stopped early: {run.stopped}. Re-run to resume.")
    print(f"Input tokens this run: {run.tokens:,}")
    for uri in run.errors[:10]:
        print(f"  failed: {uri}")
    print_class_distribution(run.scored)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    """Parse the command line. The model is pinned, so it is not a flag."""
    parser = argparse.ArgumentParser(
        description="Score ESCO skills against the scoring v2 rubric"
    )
    parser.add_argument("--rederive", action="store_true",
                        help="Recompute the derived fields of the existing scores "
                             "from their stored answers and exit; no requests, no key")
    add_run_arguments(parser, workers=6, rps=10.0)
    return parser.parse_args()


def pool_config(client, args):
    """How the pool runs one scoring pass for this command line."""
    return PoolConfig(score_one=partial(score_skill, client), save=save,
                      workers=args.workers, rps=args.rps,
                      checkpoint_every=CHECKPOINT_EVERY)


def main():
    """Score the skills named on the command line, resuming from the checkpoint."""
    args = parse_args()
    if args.rederive:
        return rederive()
    subset = select_items(load_json(INPUT_FILE), args)
    scored = load_checkpoint(OUTPUT_FILE, args.force)
    to_score = pending(subset, scored)

    print(f"Scoring {len(subset)} skills with {MODEL}")
    print(f"Already scored: {len(scored)}")
    print(f"Remaining to score: {len(to_score)}")
    if not to_score:
        print("Nothing to score. Use --force to re-score all.")
        return

    client = TypeSafeClient(api_key=load_api_key())
    print_summary(run_pool(to_score, scored, pool_config(client, args)))


if __name__ == "__main__":
    main()
