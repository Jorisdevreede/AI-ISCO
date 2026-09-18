"""
Score each ESCO skill on dual axes (automation risk + AI amplification) using
an LLM via OpenRouter.

Reads skill definitions from data/esco_skills.json, sends batches to an LLM
with a scoring rubric, and collects structured scores. Results are cached
incrementally to data/skill_scores.json so the script can be resumed if
interrupted.

Usage:
    uv run python score_skills.py
    uv run python score_skills.py --model google/gemini-3-flash-preview
    uv run python score_skills.py --start 0 --end 50
    uv run python score_skills.py --batch-size 15 --delay 2.0
"""

import argparse
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

from aiisco.checkpoint import load_checkpoint, pending, save_checkpoint
from aiisco.jsonio import load_json
from aiisco.openrouter import (
    BatchLoop,
    ChatRequest,
    call_model,
    collect_results,
    iter_batches,
    print_failures,
    print_histogram,
    run_batches,
    tally,
)

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
INPUT_FILE = "data/esco_skills.json"
OUTPUT_FILE = "data/skill_scores.json"

SYSTEM_PROMPT = """\
You are an expert analyst evaluating how AI will affect individual skills and \
knowledge areas. You will be given a batch of skills from the ESCO \
(European Skills, Competences, Qualifications and Occupations) taxonomy.

For each skill, provide TWO scores:

1. **Automation Risk (1-10)**: How likely is this skill/knowledge to be \
automated by AI in the next 5-10 years?
   - 1-2: Cannot be automated (requires physical presence, empathy, human \
judgment in unpredictable situations)
   - 3-4: Very difficult to automate (complex embodied skills, nuanced \
interpersonal interaction)
   - 5-6: Partially automatable (AI can assist but human oversight is essential)
   - 7-8: Highly automatable (AI can perform this with minimal supervision \
in most cases)
   - 9-10: Fully automatable (AI already does this better than most humans)

2. **AI Amplification Potential (1-10)**: How much can AI tools amplify \
human productivity for this skill?
   - 1-2: Minimal amplification (skill is binary/physical, AI doesn't help much)
   - 3-4: Some amplification (AI provides minor assistance, marginal \
productivity gain)
   - 5-6: Moderate amplification (AI tools meaningfully enhance output \
quality/speed)
   - 7-8: High amplification (AI creates significant leverage, 2-5x \
productivity possible)
   - 9-10: Transformative amplification (AI enables fundamentally new \
capabilities, 10x+ potential)

Respond with ONLY a JSON array of objects, one per skill, in this exact \
format (no other text):
[
  {
    "title": "<skill title exactly as given>",
    "automation_risk": <1-10>,
    "amplification_potential": <1-10>,
    "rationale": "<1-2 sentences explaining key factors for both scores>"
  }
]

Return the skills in the SAME ORDER as provided.\
"""


@dataclass
class ScoreState:
    """The scores collected so far and the averages printed after each batch."""

    scored: dict = field(default_factory=dict)
    count: int = 0
    automation: float = 0.0
    amplification: float = 0.0

    def add(self, skill, result):
        """Record one scored skill and fold it into the running averages."""
        entry = score_entry(skill, result)
        self.scored[entry["uri"]] = entry
        self.count += 1
        self.automation += entry["automation_risk"]
        self.amplification += entry["amplification_potential"]

    def progress(self):
        """The line printed after a batch the model answered."""
        avg_auto = self.automation / self.count if self.count else 0
        avg_amp = self.amplification / self.count if self.count else 0
        return (f"OK ({self.count} scored, "
                f"avg risk={avg_auto:.1f}, avg amp={avg_amp:.1f})")


def build_batch_prompt(skills_batch):
    """Build the user prompt for a batch of skills."""
    lines = ["Score each of the following skills/knowledge areas:\n"]
    for idx, skill in enumerate(skills_batch, 1):
        description = skill.get("description", "").strip()
        if description:
            lines.append(f'{idx}. "{skill["title"]}" - {description}')
        else:
            lines.append(f'{idx}. "{skill["title"]}"')
    lines.append(
        "\nFor each skill, respond with a JSON array of objects with fields: "
        "title, automation_risk, amplification_potential, rationale."
    )
    return "\n".join(lines)


def validate_scores(item):
    """Reject a scored skill whose two axes are not numbers."""
    # ValueError, not TypeError: call_model only retries on ValueError.
    if not isinstance(item.get("automation_risk"), (int, float)):
        raise ValueError(  # noqa: TRY004
            f"Invalid automation_risk for '{item.get('title')}'"
        )
    if not isinstance(item.get("amplification_potential"), (int, float)):
        raise ValueError(  # noqa: TRY004
            f"Invalid amplification_potential for '{item.get('title')}'"
        )


def score_batch(client, skills_batch, model):
    """Send one batch of skills to the LLM and parse the structured response."""
    request = ChatRequest(system_prompt=SYSTEM_PROMPT, model=model, timeout=120)
    return call_model(client, request, build_batch_prompt(skills_batch),
                      validate_scores)


def score_entry(skill, result):
    """Build the record stored for one scored skill."""
    return {
        "uri": skill["uri"],
        "title": skill["title"],
        "automation_risk": int(result["automation_risk"]),
        "amplification_potential": int(result["amplification_potential"]),
        "rationale": result.get("rationale", ""),
    }


def score_all(to_score, state, args):
    """Score every remaining skill, batch by batch, checkpointing as it goes."""
    client = httpx.Client()

    def handle(batch, errors):
        collect_results(state, batch, score_batch(client, batch, args.model),
                        errors)
        print(state.progress())

    errors = run_batches(BatchLoop(
        batches=iter_batches(to_score, args.batch_size),
        noun="skills",
        handle=handle,
        checkpoint=lambda: save_checkpoint(OUTPUT_FILE, state.scored),
        delay=args.delay,
    ))
    client.close()
    return errors


def score_label(score):
    """Right-align a 1-10 score so the distribution bars line up."""
    return f"{score:>2}"


def print_distributions(vals):
    """Print how the scored skills spread over both 1-10 axes."""
    print_histogram("Automation risk distribution:",
                    tally(s["automation_risk"] for s in vals), score_label)
    print_histogram("Amplification potential distribution:",
                    tally(s["amplification_potential"] for s in vals),
                    score_label)


def print_summary(scored):
    """Print the averages and both distributions over everything scored."""
    vals = [s for s in scored.values() if "automation_risk" in s]
    if not vals:
        return
    avg_risk = sum(s["automation_risk"] for s in vals) / len(vals)
    avg_amp = sum(s["amplification_potential"] for s in vals) / len(vals)
    print(f"\nSummary across {len(vals)} skills:")
    print(f"  Average automation risk:        {avg_risk:.2f}")
    print(f"  Average amplification potential: {avg_amp:.2f}")
    print_distributions(vals)


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Score ESCO skills on automation risk and AI amplification"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between batches")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of skills per LLM call")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if already cached")
    return parser.parse_args()


def main():
    args = parse_args()
    subset = load_json(INPUT_FILE)[args.start:args.end]
    state = ScoreState(scored=load_checkpoint(OUTPUT_FILE, args.force))

    print(f"Scoring {len(subset)} skills with {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Already scored: {len(state.scored)}")
    to_score = pending(subset, state.scored)
    print(f"Remaining to score: {len(to_score)}")

    if not to_score:
        print("Nothing to score. Use --force to re-score all.")
        return

    errors = score_all(to_score, state, args)
    print(f"\nDone. Total scored: {len(state.scored)}, errors: {len(errors)}.")
    print_failures(errors)
    print_summary(state.scored)


if __name__ == "__main__":
    main()
