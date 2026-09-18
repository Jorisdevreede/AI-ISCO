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
from aiisco.rollup import (
    ESSENTIAL_WEIGHT,
    OPTIONAL_WEIGHT,
    QUADRANTS,
    assign_quadrant,
)

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
OCCUPATIONS_FILE = "data/esco_occupations.json"
SKILL_SCORES_FILE = "data/skill_scores.json"
OUTPUT_FILE = "data/occupation_narratives.json"


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

REQUIRED_ARRAYS = ("automated_tasks", "amplified_capabilities",
                   "ai_tools_applicable")


# ---------------------------------------------------------------------------
# Occupation-level aggregation
# ---------------------------------------------------------------------------

def scored_skills(skills, skill_scores):
    """The skills that carry a score, in the shape the prompt shows them."""
    entries = []
    for skill in skills:
        sc = skill_scores.get(skill["uri"])
        if sc:
            entries.append({
                "title": skill.get("title", ""),
                "automation_risk": sc["automation_risk"],
                "amplification_potential": sc["amplification_potential"],
                "rationale": sc.get("rationale", ""),
            })
    return entries


def weighted_total(groups, key):
    """Sum one axis across (skills, weight) groups."""
    return sum(entry[key] * weight
               for entries, weight in groups for entry in entries)


def aggregate_occupation_scores(occ, skill_scores):
    """Compute weighted average automation_risk and amplification_potential.

    Returns (auto_avg, amp_avg, scored_essential, scored_optional) or None
    if no scored skills exist.
    """
    essential = scored_skills(occ.get("essential_skills", []), skill_scores)
    optional = scored_skills(occ.get("optional_skills", []), skill_scores)
    groups = [(essential, ESSENTIAL_WEIGHT), (optional, OPTIONAL_WEIGHT)]
    total_weight = sum(len(entries) * weight for entries, weight in groups)
    if total_weight == 0:
        return None
    auto_avg = round(weighted_total(groups, "automation_risk") / total_weight, 1)
    amp_avg = round(
        weighted_total(groups, "amplification_potential") / total_weight, 1)
    return auto_avg, amp_avg, essential, optional


def index_skill_scores(entries):
    """Index per-skill scores by uri, skipping entries that carry no score."""
    skill_scores = {}
    for s in entries:
        uri = s.get("uri", "")
        if uri and s.get("automation_risk") is not None:
            skill_scores[uri] = {
                "automation_risk": float(s["automation_risk"]),
                "amplification_potential": float(s["amplification_potential"]),
                "rationale": s.get("rationale", ""),
            }
    return skill_scores


def build_occupation_contexts(occupations, skill_scores):
    """Pre-compute the scores, quadrant and skill detail the prompt needs."""
    contexts = []
    for occ in occupations:
        result = aggregate_occupation_scores(occ, skill_scores)
        if result is None:
            continue
        auto_avg, amp_avg, essential, optional = result
        contexts.append({
            "uri": occ["uri"],
            "title": occ["title"],
            "isco_code": occ.get("isco_code", ""),
            "quadrant": assign_quadrant(auto_avg, amp_avg),
            "auto_avg": auto_avg,
            "amp_avg": amp_avg,
            "essential": essential,
            "optional": optional,
        })
    return contexts


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def skill_line(sk):
    """One scored essential skill, both axes and its rationale."""
    return (f"  - \"{sk['title']}\" "
            f"(auto={sk['automation_risk']}, amp={sk['amplification_potential']})"
            f" — {sk['rationale']}")


def top_skill_lines(essential, key, heading, abbreviation):
    """The 'Top 5' block ranking the essential skills on one axis."""
    ranked = sorted(essential, key=lambda s: s[key], reverse=True)[:5]
    return [f"\nTop 5 Most {heading} Essential Skills:"] + [
        f"  - \"{sk['title']}\" ({abbreviation}={sk[key]})" for sk in ranked
    ]


def occupation_block(index, ctx):
    """The prompt lines describing one occupation."""
    lines = [
        f"--- Occupation {index} ---",
        f"Title: {ctx['title']}",
        f"ISCO Code: {ctx['isco_code']}",
        f"Quadrant: {ctx['quadrant']}",
        (f"Aggregated Scores: automation_risk={ctx['auto_avg']}, "
         f"amplification_potential={ctx['amp_avg']}"),
        f"\nScored Essential Skills ({len(ctx['essential'])} total):",
    ]
    lines.extend(skill_line(sk) for sk in ctx["essential"])
    lines.extend(top_skill_lines(ctx["essential"], "automation_risk",
                                 "Automatable", "auto"))
    lines.extend(top_skill_lines(ctx["essential"], "amplification_potential",
                                 "AI-Amplifiable", "amp"))
    lines.append("")  # blank line separator
    return lines


def build_batch_prompt(occupation_contexts):
    """Build the user prompt for a batch of occupations."""
    lines = [
        ("Generate an AI evolution narrative for each of the following "
         "occupations:\n")
    ]
    for idx, ctx in enumerate(occupation_contexts, 1):
        lines.extend(occupation_block(idx, ctx))
    lines.append(
        "For each occupation, respond with a JSON array of objects with "
        "fields: title, evolution_story, time_savings_pct, automated_tasks, "
        "amplified_capabilities, ai_tools_applicable, rebalanced_week, "
        "timeline, advice."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_text(item, key):
    """Report a required string field that is missing or empty."""
    if not isinstance(item.get(key), str) or not item[key]:
        return [f"missing or empty '{key}'"]
    return []


def check_time_savings(item):
    """Report a time saving that is not a percentage."""
    tsp = item.get("time_savings_pct")
    if not isinstance(tsp, (int, float)) or tsp < 0 or tsp > 100:
        return [f"'time_savings_pct' must be a number 0-100, got {tsp!r}"]
    return []


def check_array(item, key):
    """Report a required list field that is not a list."""
    if not isinstance(item.get(key), list):
        return [f"'{key}' must be an array"]
    return []


def check_rebalanced_week(item):
    """Report a missing or incomplete before/after breakdown of the week."""
    rw = item.get("rebalanced_week")
    if not isinstance(rw, dict):
        return ["'rebalanced_week' must be an object"]
    return [f"'rebalanced_week' missing '{part}'"
            for part in ("before", "after") if part not in rw]


def validate_result(item):
    """Validate a single narrative result. Returns list of error messages."""
    errors = check_text(item, "title") + check_text(item, "evolution_story")
    errors += check_time_savings(item)
    for key in REQUIRED_ARRAYS:
        errors += check_array(item, key)
    return errors + check_rebalanced_week(item)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def validate_narrative(item):
    """Reject a narrative the model returned incomplete."""
    validation_errors = validate_result(item)
    if validation_errors:
        raise ValueError(
            f"Validation failed for '{item.get('title', '?')}': "
            f"{'; '.join(validation_errors)}"
        )


def generate_batch(client, occupation_contexts, model):
    """Send one batch of occupations to the LLM and parse the response."""
    request = ChatRequest(system_prompt=SYSTEM_PROMPT, model=model,
                          timeout=180, retry_statuses=(402, 429),
                          max_tokens=4096)
    return call_model(client, request,
                      build_batch_prompt(occupation_contexts),
                      validate_narrative)


def narrative_entry(ctx, result):
    """Build the record stored for one narrated occupation."""
    return {
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


@dataclass
class NarrativeState:
    """The narratives collected so far and the average printed per batch."""

    narrated: dict = field(default_factory=dict)
    count: int = 0
    savings: float = 0.0

    def add(self, ctx, result):
        """Record one narrated occupation and fold in its time saving."""
        entry = narrative_entry(ctx, result)
        self.narrated[entry["uri"]] = entry
        self.count += 1
        self.savings += entry["time_savings_pct"]

    def progress(self):
        """The line printed after a batch the model answered."""
        avg_savings = self.savings / self.count if self.count else 0
        return (f"OK ({self.count} narrated, "
                f"avg time_savings={avg_savings:.1f}%)")


def narrate_all(to_narrate, state, output, args):
    """Narrate every remaining occupation, checkpointing after each batch."""
    client = httpx.Client()

    def handle(batch, errors):
        collect_results(state, batch,
                        generate_batch(client, batch, args.model), errors)
        print(state.progress())

    errors = run_batches(BatchLoop(
        batches=iter_batches(to_narrate, args.batch_size),
        noun="occupations",
        handle=handle,
        checkpoint=lambda: save_checkpoint(output, state.narrated),
        delay=args.delay,
    ))
    client.close()
    return errors


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def savings_label(bucket):
    """Format one ten-point time-savings bucket for the distribution."""
    return f"{bucket}-{bucket + 9}%".rjust(8)


def print_quadrant_breakdown(contexts):
    """Print how the occupations divide over the four quadrants."""
    counts = tally(ctx["quadrant"] for ctx in contexts)
    print("\nQuadrant breakdown:")
    for quadrant in QUADRANTS:
        count = counts.get(quadrant, 0)
        pct = count / len(contexts) * 100 if contexts else 0
        print(f"  {quadrant:12s}: {count:4d} ({pct:5.1f}%)")


def print_savings_by_quadrant(vals, contexts):
    """Print the average time saving within each quadrant."""
    quadrant_of = {ctx["uri"]: ctx["quadrant"] for ctx in contexts}
    totals = {}
    counts = {}
    for narrative in vals:
        quadrant = quadrant_of.get(narrative["uri"])
        if quadrant:
            totals[quadrant] = (totals.get(quadrant, 0)
                                + narrative["time_savings_pct"])
            counts[quadrant] = counts.get(quadrant, 0) + 1
    print("\nAverage time savings by quadrant:")
    for quadrant in QUADRANTS:
        print(f"  {quadrant:12s}: " + (
            f"{totals[quadrant] / counts[quadrant]:5.1f}% (n={counts[quadrant]})"
            if counts.get(quadrant) else "n/a"))


def print_summary(narrated, contexts):
    """Print the closing statistics over everything narrated so far."""
    vals = [n for n in narrated.values() if "time_savings_pct" in n]
    if not vals:
        return
    avg_savings = sum(n["time_savings_pct"] for n in vals) / len(vals)
    print(f"\nSummary across {len(vals)} occupations:")
    print(f"  Average time_savings_pct: {avg_savings:.1f}%")
    print_histogram("Time savings distribution:",
                    tally((n["time_savings_pct"] // 10) * 10 for n in vals),
                    savings_label)
    print_quadrant_breakdown(contexts)
    print_savings_by_quadrant(vals, contexts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse the command line."""
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
    parser.add_argument("--output", default=None,
                        help="Output file path (default: data/occupation_narratives.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    # Allow per-shard output files for parallel execution
    output = args.output or OUTPUT_FILE
    contexts = build_occupation_contexts(
        load_json(OCCUPATIONS_FILE)[args.start:args.end],
        index_skill_scores(load_json(SKILL_SCORES_FILE)),
    )
    state = NarrativeState(narrated=load_checkpoint(output, args.force))

    print(f"Generating narratives for {len(contexts)} occupations "
          f"with {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Already narrated: {len(state.narrated)}")
    to_narrate = pending(contexts, state.narrated)
    print(f"Remaining to narrate: {len(to_narrate)}")

    if not to_narrate:
        print("Nothing to narrate. Use --force to re-generate all.")
        return

    errors = narrate_all(to_narrate, state, output, args)
    print(f"\nDone. Total narrated: {len(state.narrated)}, "
          f"errors: {len(errors)}.")
    print_failures(errors)
    print_summary(state.narrated, contexts)


if __name__ == "__main__":
    main()
