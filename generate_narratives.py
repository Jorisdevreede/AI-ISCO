"""
Generate AI evolution narratives for ESCO occupations using an LLM via
OpenRouter.

Reads occupation definitions from data/esco_occupations.json and per-skill
scores from data/skill_scores.json. For each occupation, builds a rich context
(quadrant, aggregated scores, scored essential skills with rationales, top
automated/amplified skills) and asks the LLM to produce a structured evolution
narrative. Results are cached incrementally to data/occupation_narratives.json
so the script can be resumed if interrupted.

Usage:
    uv run python generate_narratives.py
    uv run python generate_narratives.py --model google/gemini-3-flash-preview
    uv run python generate_narratives.py --start 0 --end 20
    uv run python generate_narratives.py --batch-size 3 --delay 3.0
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
OCCUPATIONS_FILE = "data/esco_occupations.json"
SKILL_SCORES_FILE = "data/skill_scores.json"
OUTPUT_FILE = "data/occupation_narratives.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

ESSENTIAL_WEIGHT = 2.0
OPTIONAL_WEIGHT = 1.0
QUADRANT_THRESHOLD = 6

SYSTEM_PROMPT = """\
You are an expert analyst specializing in how artificial intelligence will \
transform specific occupations over the next 5-10 years.

For each occupation below, you receive:
- The job title and ISCO code
- Its quadrant classification (TRANSFORM / SHRINK / EVOLVE / STABLE) based \
on aggregated automation risk and AI amplification potential scores
- Overall weighted average scores for automation_risk and \
amplification_potential (1-10 scale)
- Detailed essential skills with their individual automation_risk and \
amplification_potential scores plus rationales explaining those scores
- The top 5 most automatable and top 5 most AI-amplifiable skills

Generate a compelling evolution narrative for each occupation that considers \
how skills interact. For example, if both "budget management" and "market \
research" automate, the role shifts from data-gathering to insight-synthesis. \
Think holistically about the occupation, not just individual skills.

Respond with ONLY a JSON array of objects, one per occupation, in this exact \
format (no other text):
[
  {
    "title": "<occupation title exactly as given>",
    "evolution_story": "<2-3 paragraphs written in second person ('your \
role...'), vivid and specific, describing how AI transforms this occupation. \
Consider skill interactions, workflow changes, and emerging responsibilities.>",
    "time_savings_pct": <integer 0-80, estimated percentage of the work week \
that AI could automate>,
    "automated_tasks": ["<3-6 specific tasks this role currently does that AI \
will handle>"],
    "amplified_capabilities": ["<3-6 specific capabilities where AI makes the \
human much more effective>"],
    "ai_tools_applicable": ["<3-5 specific AI tool categories applicable to \
this role>"],
    "rebalanced_week": {
      "before": {"<category_name>": <pct>, ...},
      "after": {"<category_name>": <pct>, ..., "new_ai_augmented": <pct>}
    },
    "timeline": "<estimated time horizon for significant AI impact, e.g. \
'3-5 years'>",
    "advice": "<1-2 sentences of concrete, actionable career advice>"
  }
]

Guidelines for each field:
- evolution_story: 2-3 paragraphs, second person ("your role..."), vivid and \
specific. Show how automating certain skills frees time for amplified ones.
- time_savings_pct: realistic estimate (0-80 range). Higher for SHRINK/\
TRANSFORM quadrants, lower for STABLE/EVOLVE.
- automated_tasks: 3-6 specific current tasks AI will handle or largely \
eliminate.
- amplified_capabilities: 3-6 capabilities where AI augments human \
effectiveness significantly.
- ai_tools_applicable: 3-5 specific AI tool categories (not brand names, \
but categories like "AI-powered market research platforms").
- rebalanced_week: before/after percentage breakdowns of a typical work week. \
Use descriptive category names. Both before and after should sum to \
approximately 100. The "after" should include a "new_ai_augmented" category.
- timeline: e.g. "2-3 years", "3-5 years", "5-10 years".
- advice: 1-2 sentences of concrete, actionable career advice for someone in \
this role today.

Return the occupations in the SAME ORDER as provided.\
"""

# Maximum retries for transient / rate-limit errors
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_quadrant(auto, amp):
    """Assign quadrant based on automation_risk and amplification_potential."""
    if auto >= QUADRANT_THRESHOLD and amp >= QUADRANT_THRESHOLD:
        return "TRANSFORM"
    elif auto >= QUADRANT_THRESHOLD and amp < QUADRANT_THRESHOLD:
        return "SHRINK"
    elif auto < QUADRANT_THRESHOLD and amp >= QUADRANT_THRESHOLD:
        return "EVOLVE"
    else:
        return "STABLE"


def aggregate_occupation_scores(occ, skill_scores):
    """Compute weighted average automation_risk and amplification_potential.

    Returns (auto_avg, amp_avg, scored_essential, scored_optional) or None
    if no scored skills exist.
    """
    auto_sum = 0.0
    amp_sum = 0.0
    total_weight = 0.0
    scored_essential = []
    scored_optional = []

    for skill in occ.get("essential_skills", []):
        sc = skill_scores.get(skill["uri"])
        if sc:
            auto_sum += sc["automation_risk"] * ESSENTIAL_WEIGHT
            amp_sum += sc["amplification_potential"] * ESSENTIAL_WEIGHT
            total_weight += ESSENTIAL_WEIGHT
            scored_essential.append({
                "title": skill.get("title", ""),
                "automation_risk": sc["automation_risk"],
                "amplification_potential": sc["amplification_potential"],
                "rationale": sc.get("rationale", ""),
            })

    for skill in occ.get("optional_skills", []):
        sc = skill_scores.get(skill["uri"])
        if sc:
            auto_sum += sc["automation_risk"] * OPTIONAL_WEIGHT
            amp_sum += sc["amplification_potential"] * OPTIONAL_WEIGHT
            total_weight += OPTIONAL_WEIGHT
            scored_optional.append({
                "title": skill.get("title", ""),
                "automation_risk": sc["automation_risk"],
                "amplification_potential": sc["amplification_potential"],
                "rationale": sc.get("rationale", ""),
            })

    if total_weight == 0:
        return None

    auto_avg = round(auto_sum / total_weight, 1)
    amp_avg = round(amp_sum / total_weight, 1)
    return auto_avg, amp_avg, scored_essential, scored_optional


def build_batch_prompt(occupation_contexts):
    """Build the user prompt for a batch of occupations."""
    lines = [
        "Generate an AI evolution narrative for each of the following "
        "occupations:\n"
    ]
    for idx, ctx in enumerate(occupation_contexts, 1):
        lines.append(f"--- Occupation {idx} ---")
        lines.append(f"Title: {ctx['title']}")
        lines.append(f"ISCO Code: {ctx['isco_code']}")
        lines.append(f"Quadrant: {ctx['quadrant']}")
        lines.append(
            f"Aggregated Scores: automation_risk={ctx['auto_avg']}, "
            f"amplification_potential={ctx['amp_avg']}"
        )

        # Essential skills with scores and rationales
        lines.append(f"\nScored Essential Skills ({len(ctx['essential'])} total):")
        for sk in ctx["essential"]:
            lines.append(
                f"  - \"{sk['title']}\" "
                f"(auto={sk['automation_risk']}, amp={sk['amplification_potential']})"
                f" — {sk['rationale']}"
            )

        # Top 5 most automated
        top_auto = sorted(
            ctx["essential"], key=lambda s: s["automation_risk"], reverse=True
        )[:5]
        lines.append("\nTop 5 Most Automatable Essential Skills:")
        for sk in top_auto:
            lines.append(
                f"  - \"{sk['title']}\" (auto={sk['automation_risk']})"
            )

        # Top 5 most amplified
        top_amp = sorted(
            ctx["essential"],
            key=lambda s: s["amplification_potential"],
            reverse=True,
        )[:5]
        lines.append("\nTop 5 Most AI-Amplifiable Essential Skills:")
        for sk in top_amp:
            lines.append(
                f"  - \"{sk['title']}\" (amp={sk['amplification_potential']})"
            )

        lines.append("")  # blank line separator

    lines.append(
        "For each occupation, respond with a JSON array of objects with "
        "fields: title, evolution_story, time_savings_pct, automated_tasks, "
        "amplified_capabilities, ai_tools_applicable, rebalanced_week, "
        "timeline, advice."
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


def validate_result(item):
    """Validate a single narrative result. Returns list of error messages."""
    errors = []

    if not isinstance(item.get("title"), str) or not item["title"]:
        errors.append("missing or empty 'title'")

    if not isinstance(item.get("evolution_story"), str) or not item["evolution_story"]:
        errors.append("missing or empty 'evolution_story'")

    tsp = item.get("time_savings_pct")
    if not isinstance(tsp, (int, float)) or tsp < 0 or tsp > 100:
        errors.append(
            f"'time_savings_pct' must be a number 0-100, got {tsp!r}"
        )

    if not isinstance(item.get("automated_tasks"), list):
        errors.append("'automated_tasks' must be an array")

    if not isinstance(item.get("amplified_capabilities"), list):
        errors.append("'amplified_capabilities' must be an array")

    if not isinstance(item.get("ai_tools_applicable"), list):
        errors.append("'ai_tools_applicable' must be an array")

    rw = item.get("rebalanced_week")
    if not isinstance(rw, dict):
        errors.append("'rebalanced_week' must be an object")
    else:
        if "before" not in rw:
            errors.append("'rebalanced_week' missing 'before'")
        if "after" not in rw:
            errors.append("'rebalanced_week' missing 'after'")

    return errors


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def generate_batch(client, occupation_contexts, model):
    """Send one batch of occupations to the LLM and parse the response."""
    user_prompt = build_batch_prompt(occupation_contexts)

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
                timeout=180,
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
                validation_errors = validate_result(item)
                if validation_errors:
                    raise ValueError(
                        f"Validation failed for '{item.get('title', '?')}': "
                        f"{'; '.join(validation_errors)}"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate AI evolution narratives for ESCO occupations"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay in seconds between batches")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of occupations per LLM call")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if already cached")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    with open(OCCUPATIONS_FILE) as f:
        all_occupations = json.load(f)

    with open(SKILL_SCORES_FILE) as f:
        skill_scores_list = json.load(f)

    # Build skill_scores lookup: uri -> {automation_risk, amplification_potential, rationale}
    skill_scores = {}
    for s in skill_scores_list:
        uri = s.get("uri", "")
        if uri and s.get("automation_risk") is not None:
            skill_scores[uri] = {
                "automation_risk": float(s["automation_risk"]),
                "amplification_potential": float(s["amplification_potential"]),
                "rationale": s.get("rationale", ""),
            }

    subset = all_occupations[args.start:args.end]

    # ------------------------------------------------------------------
    # 2. Pre-compute occupation contexts (scores, quadrants, skills)
    # ------------------------------------------------------------------
    occupation_contexts = []
    for occ in subset:
        result = aggregate_occupation_scores(occ, skill_scores)
        if result is None:
            continue
        auto_avg, amp_avg, scored_essential, scored_optional = result
        quadrant = assign_quadrant(auto_avg, amp_avg)
        occupation_contexts.append({
            "uri": occ["uri"],
            "title": occ["title"],
            "isco_code": occ.get("isco_code", ""),
            "quadrant": quadrant,
            "auto_avg": auto_avg,
            "amp_avg": amp_avg,
            "essential": scored_essential,
            "optional": scored_optional,
        })

    # ------------------------------------------------------------------
    # 3. Load existing narratives for resume support
    # ------------------------------------------------------------------
    narrated = {}
    if os.path.exists(OUTPUT_FILE) and not args.force:
        with open(OUTPUT_FILE) as f:
            for entry in json.load(f):
                narrated[entry["uri"]] = entry

    print(f"Generating narratives for {len(occupation_contexts)} occupations "
          f"with {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Already narrated: {len(narrated)}")

    # Filter out already-narrated occupations (preserve order)
    to_narrate = [c for c in occupation_contexts if c["uri"] not in narrated]
    print(f"Remaining to narrate: {len(to_narrate)}")

    if not to_narrate:
        print("Nothing to narrate. Use --force to re-generate all.")
        return

    # ------------------------------------------------------------------
    # 4. Split into batches and process
    # ------------------------------------------------------------------
    batches = []
    for i in range(0, len(to_narrate), args.batch_size):
        batches.append(to_narrate[i : i + args.batch_size])

    errors = []
    client = httpx.Client()
    total_narrated = 0
    sum_time_savings = 0.0

    for batch_idx, batch in enumerate(batches):
        titles = [c["title"] for c in batch]
        print(
            f"\n  Batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} occupations): {titles[0]!r} ... {titles[-1]!r}",
            end=" ",
            flush=True,
        )

        try:
            results = generate_batch(client, batch, args.model)

            # Match results back to occupations by position (fallback to title)
            result_by_title = {r["title"].lower(): r for r in results}

            for ctx in batch:
                # Try positional match first, then title match
                idx_in_batch = batch.index(ctx)
                if idx_in_batch < len(results):
                    result = results[idx_in_batch]
                else:
                    result = result_by_title.get(ctx["title"].lower())

                if result is None:
                    print(f"\n    WARNING: No result for '{ctx['title']}'")
                    errors.append(ctx["uri"])
                    continue

                narrated[ctx["uri"]] = {
                    "uri": ctx["uri"],
                    "title": ctx["title"],
                    "evolution_story": result["evolution_story"],
                    "time_savings_pct": int(result["time_savings_pct"]),
                    "automated_tasks": result["automated_tasks"],
                    "amplified_capabilities": result["amplified_capabilities"],
                    "ai_tools_applicable": result["ai_tools_applicable"],
                    "rebalanced_week": result["rebalanced_week"],
                    "timeline": result.get("timeline", ""),
                    "advice": result.get("advice", ""),
                }
                total_narrated += 1
                sum_time_savings += int(result["time_savings_pct"])

            # Progress
            avg_savings = (
                sum_time_savings / total_narrated if total_narrated else 0
            )
            print(
                f"OK ({total_narrated} narrated, "
                f"avg time_savings={avg_savings:.1f}%)"
            )

        except Exception as e:
            print(f"ERROR: {e}")
            for ctx in batch:
                errors.append(ctx["uri"])

        # Incremental checkpoint after each batch
        with open(OUTPUT_FILE, "w") as f:
            json.dump(list(narrated.values()), f, indent=2)

        # Delay between batches (skip after last)
        if batch_idx < len(batches) - 1:
            time.sleep(args.delay)

    client.close()

    print(f"\nDone. Total narrated: {len(narrated)}, errors: {len(errors)}.")
    if errors:
        print(f"Failed URIs ({len(errors)}):")
        for uri in errors[:10]:
            print(f"  {uri}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    vals = [n for n in narrated.values() if "time_savings_pct" in n]
    if vals:
        avg_savings = sum(n["time_savings_pct"] for n in vals) / len(vals)
        print(f"\nSummary across {len(vals)} occupations:")
        print(f"  Average time_savings_pct: {avg_savings:.1f}%")

        # time_savings_pct distribution (bucketed by 10s)
        print("\nTime savings distribution:")
        savings_dist = {}
        for n in vals:
            bucket = (n["time_savings_pct"] // 10) * 10
            label = f"{bucket}-{bucket + 9}%"
            savings_dist[bucket] = savings_dist.get(bucket, 0) + 1
        for k in sorted(savings_dist):
            label = f"{k}-{k + 9}%"
            count = savings_dist[k]
            bar = "\u2588" * count
            print(f"  {label:>8}: {bar} ({count})")

        # Quadrant breakdown
        print("\nQuadrant breakdown:")
        quad_counts = {}
        for ctx in occupation_contexts:
            q = ctx["quadrant"]
            quad_counts[q] = quad_counts.get(q, 0) + 1
        total_occ = len(occupation_contexts)
        for q in ["TRANSFORM", "SHRINK", "EVOLVE", "STABLE"]:
            count = quad_counts.get(q, 0)
            pct = count / total_occ * 100 if total_occ else 0
            print(f"  {q:12s}: {count:4d} ({pct:5.1f}%)")

        # Average time_savings by quadrant
        print("\nAverage time savings by quadrant:")
        quad_savings = {}
        quad_savings_count = {}
        for n in vals:
            # Find this occupation's context to get the quadrant
            ctx = next(
                (c for c in occupation_contexts if c["uri"] == n["uri"]),
                None,
            )
            if ctx:
                q = ctx["quadrant"]
                quad_savings[q] = quad_savings.get(q, 0) + n["time_savings_pct"]
                quad_savings_count[q] = quad_savings_count.get(q, 0) + 1
        for q in ["TRANSFORM", "SHRINK", "EVOLVE", "STABLE"]:
            count = quad_savings_count.get(q, 0)
            if count:
                avg = quad_savings[q] / count
                print(f"  {q:12s}: {avg:5.1f}% (n={count})")
            else:
                print(f"  {q:12s}: n/a")


if __name__ == "__main__":
    main()
