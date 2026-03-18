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
import json
import os
import re
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
INPUT_FILE = "data/esco_skills.json"
OUTPUT_FILE = "data/skill_scores.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

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

# Maximum retries for transient / rate-limit errors
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


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


def strip_code_fences(content):
    """Strip markdown code fences from LLM response."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]  # remove first line
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return content


def fix_json(text):
    """Fix common LLM JSON issues: trailing commas, comments."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def score_batch(client, skills_batch, model):
    """Send one batch of skills to the LLM and parse the structured response."""
    user_prompt = build_batch_prompt(skills_batch)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            content = strip_code_fences(content)
            content = fix_json(content)
            results = json.loads(content)

            # Validate the response structure
            if not isinstance(results, list):
                raise ValueError(
                    f"Expected a JSON array, got {type(results).__name__}"
                )
            for item in results:
                if not isinstance(item.get("automation_risk"), (int, float)):
                    raise ValueError(
                        f"Invalid automation_risk for '{item.get('title')}'"
                    )
                if not isinstance(item.get("amplification_potential"), (int, float)):
                    raise ValueError(
                        f"Invalid amplification_potential for "
                        f"'{item.get('title')}'"
                    )

            return results

        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            # Retry on rate limit (429) or server errors (5xx)
            if status == 429 or status >= 500:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(
                    f"\n    Rate limited/server error ({status}), "
                    f"retrying in {backoff:.0f}s (attempt {attempt + 1}/"
                    f"{MAX_RETRIES})...",
                    flush=True,
                )
                time.sleep(backoff)
                continue
            raise
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(
                    f"\n    Parse error: {e}, retrying in {backoff:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                    flush=True,
                )
                time.sleep(backoff)
                continue
            raise

    raise last_error


def main():
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
    args = parser.parse_args()

    # Load skills from ingested ESCO data
    with open(INPUT_FILE) as f:
        all_skills = json.load(f)

    subset = all_skills[args.start:args.end]

    # Load existing scores for resume support
    scored = {}
    if os.path.exists(OUTPUT_FILE) and not args.force:
        with open(OUTPUT_FILE) as f:
            for entry in json.load(f):
                scored[entry["uri"]] = entry

    print(f"Scoring {len(subset)} skills with {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Already scored: {len(scored)}")

    # Filter out already-scored skills (preserve order)
    to_score = [s for s in subset if s["uri"] not in scored]
    print(f"Remaining to score: {len(to_score)}")

    if not to_score:
        print("Nothing to score. Use --force to re-score all.")
        return

    # Split into batches
    batches = []
    for i in range(0, len(to_score), args.batch_size):
        batches.append(to_score[i : i + args.batch_size])

    errors = []
    client = httpx.Client()
    total_scored = 0
    sum_automation = 0.0
    sum_amplification = 0.0

    for batch_idx, batch in enumerate(batches):
        titles = [s["title"] for s in batch]
        print(
            f"\n  Batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} skills): {titles[0]!r} ... {titles[-1]!r}",
            end=" ",
            flush=True,
        )

        try:
            results = score_batch(client, batch, args.model)

            # Match results back to skills by position (fallback to title)
            result_by_title = {r["title"].lower(): r for r in results}

            for skill in batch:
                # Try positional match first, then title match
                idx_in_batch = batch.index(skill)
                if idx_in_batch < len(results):
                    result = results[idx_in_batch]
                else:
                    result = result_by_title.get(skill["title"].lower())

                if result is None:
                    print(f"\n    WARNING: No result for '{skill['title']}'")
                    errors.append(skill["uri"])
                    continue

                scored[skill["uri"]] = {
                    "uri": skill["uri"],
                    "title": skill["title"],
                    "automation_risk": int(result["automation_risk"]),
                    "amplification_potential": int(
                        result["amplification_potential"]
                    ),
                    "rationale": result.get("rationale", ""),
                }
                total_scored += 1
                sum_automation += int(result["automation_risk"])
                sum_amplification += int(result["amplification_potential"])

            # Progress
            avg_auto = sum_automation / total_scored if total_scored else 0
            avg_amp = sum_amplification / total_scored if total_scored else 0
            print(
                f"OK ({total_scored} scored, "
                f"avg risk={avg_auto:.1f}, avg amp={avg_amp:.1f})"
            )

        except Exception as e:
            print(f"ERROR: {e}")
            for skill in batch:
                errors.append(skill["uri"])

        # Incremental checkpoint after each batch
        with open(OUTPUT_FILE, "w") as f:
            json.dump(list(scored.values()), f, indent=2)

        # Delay between batches (skip after last)
        if batch_idx < len(batches) - 1:
            time.sleep(args.delay)

    client.close()

    print(f"\nDone. Total scored: {len(scored)}, errors: {len(errors)}.")
    if errors:
        print(f"Failed URIs ({len(errors)}):")
        for uri in errors[:10]:
            print(f"  {uri}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Summary statistics
    vals = [s for s in scored.values() if "automation_risk" in s]
    if vals:
        avg_risk = sum(s["automation_risk"] for s in vals) / len(vals)
        avg_amp = sum(s["amplification_potential"] for s in vals) / len(vals)
        print(f"\nSummary across {len(vals)} skills:")
        print(f"  Average automation risk:        {avg_risk:.2f}")
        print(f"  Average amplification potential: {avg_amp:.2f}")

        print("\nAutomation risk distribution:")
        risk_dist = {}
        for s in vals:
            bucket = s["automation_risk"]
            risk_dist[bucket] = risk_dist.get(bucket, 0) + 1
        for k in sorted(risk_dist):
            print(f"  {k:>2}: {'█' * risk_dist[k]} ({risk_dist[k]})")

        print("\nAmplification potential distribution:")
        amp_dist = {}
        for s in vals:
            bucket = s["amplification_potential"]
            amp_dist[bucket] = amp_dist.get(bucket, 0) + 1
        for k in sorted(amp_dist):
            print(f"  {k:>2}: {'█' * amp_dist[k]} ({amp_dist[k]})")


if __name__ == "__main__":
    main()
