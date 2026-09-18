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

TypeSafe's customer agreement (section 2.3(f)) prohibits publishing benchmarks
or performance information about their service. Treat this script's output,
timings and any comparison built on it as private unless TypeSafe has agreed
in writing; data/skill_scores_typesafe.json is gitignored for that reason.

Usage:
    uv run python score_skills_typesafe.py
    uv run python score_skills_typesafe.py --sample 300
    uv run python score_skills_typesafe.py --start 0 --end 50
    uv run python score_skills_typesafe.py --workers 6 --rps 15
"""

import argparse
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from typesafe_sdk import Score, TypeSafeClient, TypeSafeError

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


def score_skill(client, limiter, skill, model):
    """Score one skill on both axes in a single request."""
    limiter.wait()
    state = {
        "skill": {
            "title": skill["title"],
            "description": skill.get("description", "").strip(),
            "type": skill.get("type", ""),
        }
    }
    questions = QUESTIONS
    if skill.get("type") == "knowledge":
        questions = {**QUESTIONS, "automation": KNOWLEDGE_AUTOMATION}
    response = client.system_one(state=state, questions=questions, model=model)
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
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(list(scored.values()), f, indent=1)
    os.replace(tmp, OUTPUT_FILE)


def main():
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
    args = parser.parse_args()

    with open(INPUT_FILE) as f:
        all_skills = json.load(f)

    subset = all_skills[args.start:args.end]
    if args.sample:
        random.seed(args.seed)
        subset = random.sample(subset, min(args.sample, len(subset)))

    scored = {}
    if os.path.exists(OUTPUT_FILE) and not args.force:
        with open(OUTPUT_FILE) as f:
            for entry in json.load(f):
                scored[entry["uri"]] = entry

    to_score = [s for s in subset if s["uri"] not in scored]
    print(f"Scoring {len(subset)} skills with {args.model}")
    print(f"Already scored: {len(scored)}")
    print(f"Remaining to score: {len(to_score)}")
    if not to_score:
        print("Nothing to score. Use --force to re-score all.")
        return

    client = TypeSafeClient(api_key=load_api_key())
    limiter = RateLimiter(args.rps)
    errors = []
    done = 0
    tokens = 0
    started = time.monotonic()

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {
                pool.submit(score_skill, client, limiter, s, args.model): s
                for s in to_score
            }
            for future in as_completed(futures):
                skill = futures[future]
                try:
                    entry = future.result()
                    scored[entry["uri"]] = entry
                    tokens += entry["input_tokens"]
                except TypeSafeError as e:
                    errors.append(skill["uri"])
                    print(f"\n  ERROR {skill['title']!r}: {e}")
                done += 1
                if done % CHECKPOINT_EVERY == 0:
                    save(scored)
                    rate = done / (time.monotonic() - started)
                    print(
                        f"  {done}/{len(to_score)} "
                        f"({rate:.1f}/s, {tokens:,} tokens, "
                        f"errors {len(errors)})",
                        flush=True,
                    )
    finally:
        save(scored)

    elapsed = time.monotonic() - started
    print(f"\nDone in {elapsed:.0f}s. Total scored: {len(scored)}, "
          f"errors: {len(errors)}.")
    print(f"Input tokens this run: {tokens:,}")
    for uri in errors[:10]:
        print(f"  failed: {uri}")

    vals = list(scored.values())
    avg_risk = sum(s["automation_risk"] for s in vals) / len(vals)
    avg_amp = sum(s["amplification_potential"] for s in vals) / len(vals)
    print(f"\nSummary across {len(vals)} skills:")
    print(f"  Average automation risk:         {avg_risk:.2f}")
    print(f"  Average amplification potential: {avg_amp:.2f}")


if __name__ == "__main__":
    main()
