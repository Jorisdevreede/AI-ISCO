"""
Score each ESCO skill on dual axes (automation risk + AI amplification) using
TypeSafe's System One API instead of a generative LLM.

Same rubric and same output fields as score_skills.py, so aggregate_scores.py
can consume the result. The differences:

- One request per skill, two Score questions over the same state. No batching,
  no JSON parsing, no code fences: the answer arrives typed.
- Each axis comes back as a probability distribution over the five rubric bands.
  The 1-10 value is the probability-weighted band centre (1.5, 3.5 ... 9.5), so
  it is continuous rather than an integer.
- There is no rationale: System One models return judgments, not text.

Reads data/esco_skills.json, writes data/skill_scores_typesafe.json, and
checkpoints so it can be resumed.

The API key is read from TYPESAFE_API_KEY (.env works), falling back to the
macOS login keychain item "typesafe-api-key".

The output of the full run is committed as data/skill_scores_typesafe.json, and
the README's Step 2b compares it with the Gemini scores. This project is not
affiliated with or endorsed by TypeSafe AI, Inc.

Usage:
    uv run python score_skills_typesafe.py
    uv run python score_skills_typesafe.py --sample 300
    uv run python score_skills_typesafe.py --start 0 --end 50
    uv run python score_skills_typesafe.py --workers 6 --rps 15
"""

import argparse
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv
from typesafe_sdk import Score, TypeSafeClient, TypeSafeError

from aiisco.checkpoint import load_checkpoint, pending, save_checkpoint_atomically
from aiisco.jsonio import load_json

load_dotenv()

DEFAULT_MODEL = "jev-1.13.0"  # pinned: the jev-latest alias moves
INPUT_FILE = "data/esco_skills.json"
OUTPUT_FILE = "data/skill_scores_typesafe.json"
CHECKPOINT_EVERY = 250

# The five bands of the score_skills.py rubric. Each level has to stand on its
# own: the model judges levels in isolation, not relative to their neighbours.
AUTOMATION_LEVELS = [
    ("Cannot be automated: doing this requires physical presence, empathy, or "
     "human judgment in unpredictable situations"),
    ("Very difficult to automate: a complex embodied skill or nuanced "
     "interpersonal interaction that AI cannot carry out"),
    ("Partially automatable: AI can assist with this but human oversight "
     "remains essential"),
    ("Highly automatable: AI can perform this with minimal supervision in most "
     "cases"),
    "Fully automatable: AI already does this better than most humans",
]

AMPLIFICATION_LEVELS = [
    ("Minimal amplification: the skill is physical or binary and AI tools "
     "barely help the person doing it"),
    ("Some amplification: AI provides minor assistance and a marginal "
     "productivity gain"),
    ("Moderate amplification: AI tools meaningfully improve the quality or "
     "speed of the person's output"),
    ("High amplification: AI gives the person significant leverage, making "
     "them two to five times more productive"),
    ("Transformative amplification: AI enables fundamentally new capabilities, "
     "making the person ten or more times more productive"),
]

# ESCO mixes skills ("manage musical staff") with knowledge areas ("types of
# sugars"). "Will AI automate this knowledge area" can be read as "can AI recall
# it", which is not what the occupation roll-up needs, so knowledge items are
# asked whether AI takes over the work the knowledge is applied in.
KNOWLEDGE_AUTOMATION_LEVELS = [
    ("Cannot be automated: this knowledge is applied through physical presence, "
     "empathy, or human judgment in unpredictable situations"),
    ("Very difficult to automate: this knowledge is applied through embodied "
     "skill or nuanced interpersonal interaction that AI cannot carry out"),
    ("Partially automatable: AI can retrieve and reason with this knowledge, "
     "but a person still has to apply it and oversee the result"),
    ("Highly automatable: AI can apply this knowledge to the task with minimal "
     "supervision in most cases"),
    ("Fully automatable: AI already applies this knowledge to real tasks better "
     "than most humans"),
]

KNOWLEDGE_AUTOMATION = Score(
    instructions=(
        "The item in `skill` is a knowledge area that workers apply in their "
        "job. How likely is it that, within the next 5 to 10 years, AI will "
        "carry out the work in which this knowledge is applied, so that a "
        "person no longer applies it themselves? Judge the work the knowledge "
        "is used for, not whether AI can recall the facts."
    ),
    criteria=KNOWLEDGE_AUTOMATION_LEVELS,
)

QUESTIONS = {
    "automation": Score(
        instructions=(
            "How likely is it that AI will automate the skill described in "
            "`skill` within the next 5 to 10 years?"
        ),
        criteria=AUTOMATION_LEVELS,
    ),
    "amplification": Score(
        instructions=(
            "How much can AI tools amplify the productivity of a person "
            "applying the skill or knowledge area described in `skill`?"
        ),
        criteria=AMPLIFICATION_LEVELS,
    ),
}


def load_api_key():
    """Return the TypeSafe key from the environment or the login keychain."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["security", "find-generic-password", "-a", "typesafe",
         "-s", "typesafe-api-key", "-w"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item "
            "'typesafe-api-key'."
        )
    return result.stdout.strip()


def to_ten_scale(level_score):
    """Map a 0-4 band position onto the rubric's 1-10 scale (band centres)."""
    return round(1.5 + 2.0 * level_score, 2)


class RateLimiter:
    """Space request starts evenly so the run stays under the per-minute cap."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        if start > now:
            time.sleep(start - now)


def skill_state(skill):
    """The state a System One request judges: one skill, trimmed."""
    return {
        "skill": {
            "title": skill["title"],
            "description": skill.get("description", "").strip(),
            "type": skill.get("type", ""),
        }
    }


def questions_for(skill):
    """The two questions to ask, with the knowledge wording where it applies."""
    if skill.get("type") == "knowledge":
        return {**QUESTIONS, "automation": KNOWLEDGE_AUTOMATION}
    return QUESTIONS


def score_skill(client, limiter, skill, model):
    """Score one skill on both axes in a single request."""
    limiter.wait()
    response = client.system_one(state=skill_state(skill),
                                 questions=questions_for(skill), model=model)
    auto = response.answers["automation"]
    amp = response.answers["amplification"]
    return {
        "uri": skill["uri"],
        "title": skill["title"],
        "automation_risk": to_ten_scale(auto.score),
        "amplification_potential": to_ten_scale(amp.score),
        "automation_probs": [round(p, 4) for p in auto.probabilities.values()],
        "amplification_probs": [round(p, 4) for p in amp.probabilities.values()],
        "automation_confidence": round(auto.confidence, 3),
        "amplification_confidence": round(amp.confidence, 3),
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
    }


def save(scored):
    """Checkpoint the scores collected so far, through a temporary file."""
    save_checkpoint_atomically(OUTPUT_FILE, scored, indent=1)


@dataclass
class PoolRun:
    """What the worker pool writes to, and what the progress lines report."""

    scored: dict
    total: int
    started: float
    errors: list = field(default_factory=list)
    done: int = 0
    tokens: int = 0


def collect(run, future, skill):
    """Fold one finished request into the run, recording a refused skill."""
    try:
        entry = future.result()
    except TypeSafeError as e:
        run.errors.append(skill["uri"])
        print(f"\n  ERROR {skill['title']!r}: {e}")
        return
    run.scored[entry["uri"]] = entry
    run.tokens += entry["input_tokens"]


def print_progress(run):
    """Print the throughput line written at every checkpoint."""
    rate = run.done / (time.monotonic() - run.started)
    print(
        f"  {run.done}/{run.total} "
        f"({rate:.1f}/s, {run.tokens:,} tokens, "
        f"errors {len(run.errors)})",
        flush=True,
    )


def score_all(to_score, scored, client, args):
    """Run every remaining skill through the pool, checkpointing as it goes."""
    limiter = RateLimiter(args.rps)
    run = PoolRun(scored=scored, total=len(to_score), started=time.monotonic())
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {
                pool.submit(score_skill, client, limiter, s, args.model): s
                for s in to_score
            }
            for future in as_completed(futures):
                collect(run, future, futures[future])
                run.done += 1
                if run.done % CHECKPOINT_EVERY == 0:
                    save(run.scored)
                    print_progress(run)
    finally:
        save(run.scored)
    return run


def select_skills(all_skills, args):
    """Take the requested slice, then the fixed random sample if asked for."""
    subset = all_skills[args.start:args.end]
    if args.sample:
        random.seed(args.seed)
        subset = random.sample(subset, min(args.sample, len(subset)))
    return subset


def print_summary(run):
    """Print the closing totals, the refused skills and the two averages."""
    elapsed = time.monotonic() - run.started
    print(f"\nDone in {elapsed:.0f}s. Total scored: {len(run.scored)}, "
          f"errors: {len(run.errors)}.")
    print(f"Input tokens this run: {run.tokens:,}")
    for uri in run.errors[:10]:
        print(f"  failed: {uri}")

    vals = list(run.scored.values())
    avg_risk = sum(s["automation_risk"] for s in vals) / len(vals)
    avg_amp = sum(s["amplification_potential"] for s in vals) / len(vals)
    print(f"\nSummary across {len(vals)} skills:")
    print(f"  Average automation risk:         {avg_risk:.2f}")
    print(f"  Average amplification potential: {avg_amp:.2f}")


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Score ESCO skills with TypeSafe System One"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Score a fixed random sample of this size")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for --sample")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rps", type=float, default=15.0,
                        help="Request starts per second; keep under TypeSafe's "
                             "documented rate limit")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if already cached")
    return parser.parse_args()


def main():
    args = parse_args()
    subset = select_skills(load_json(INPUT_FILE), args)
    scored = load_checkpoint(OUTPUT_FILE, args.force)
    to_score = pending(subset, scored)

    print(f"Scoring {len(subset)} skills with {args.model}")
    print(f"Already scored: {len(scored)}")
    print(f"Remaining to score: {len(to_score)}")
    if not to_score:
        print("Nothing to score. Use --force to re-score all.")
        return

    client = TypeSafeClient(api_key=load_api_key())
    print_summary(score_all(to_score, scored, client, args))


if __name__ == "__main__":
    main()
