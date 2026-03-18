"""
Build a compact JSON dataset for the Skill Portfolio Analyzer feature.

Reads per-skill scores from data/skill_scores.json and occupation-skill
mappings from data/esco_occupations.json. Computes Jaccard-based adjacency
between occupations, identifies gap skills, and outputs a deduplicated
compact JSON to site/portfolio_data.json.

Usage:
    uv run python build_portfolio_data.py
"""

import hashlib
import json
import os
import re
import sys
from collections import defaultdict

DATA_DIR = "data"
OUT_PATH = os.path.join("site", "portfolio_data.json")

ESSENTIAL_WEIGHT = 2.0
OPTIONAL_WEIGHT = 1.0
QUADRANT_THRESHOLD = 6

MIN_JACCARD_OVERLAP = 0.15
MAX_ADJACENT = 8
MAX_GAP_SKILLS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(filename):
    """Load a JSON file from DATA_DIR."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def slugify(title):
    """Convert a title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")


def skill_short_id(uri, length=8):
    """Generate a short ID from a skill URI using an md5 hash prefix."""
    return hashlib.md5(uri.encode()).hexdigest()[:length]


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


def get_sub_major_group(hierarchy):
    """Extract the ISCO sub-major group (2-digit level) label."""
    if len(hierarchy) >= 2:
        return hierarchy[1]
    if len(hierarchy) >= 1:
        return hierarchy[0]
    return ""


def get_major_group(hierarchy):
    """Extract the ISCO major group (1-digit level) label."""
    if len(hierarchy) >= 1:
        return hierarchy[0]
    return ""


# ---------------------------------------------------------------------------
# Short ID registry (handles collisions)
# ---------------------------------------------------------------------------

def build_short_id_map(all_uris):
    """Build a mapping from skill URI to collision-free short ID.

    Uses the first 8 characters of the md5 hash. If a collision is
    detected, extends the hash length until unique.
    """
    id_to_uri = {}
    uri_to_id = {}

    for uri in sorted(all_uris):
        length = 8
        while True:
            short = hashlib.md5(uri.encode()).hexdigest()[:length]
            existing = id_to_uri.get(short)
            if existing is None or existing == uri:
                break
            # Collision -- also extend the other entry
            length += 1
            if length > 32:
                raise ValueError(f"Cannot resolve collision for {uri}")

        # If we extended beyond 8, also re-map the colliding URI
        if length > 8:
            old_short = hashlib.md5(uri.encode()).hexdigest()[:8]
            collider = id_to_uri.pop(old_short, None)
            if collider and collider != uri:
                new_collider_short = hashlib.md5(collider.encode()).hexdigest()[:length]
                id_to_uri[new_collider_short] = collider
                uri_to_id[collider] = new_collider_short

        id_to_uri[short] = uri
        uri_to_id[uri] = short

    return uri_to_id


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def aggregate_occupation_scores(occ, skill_scores):
    """Compute weighted average automation_risk and amplification_potential.

    Returns (auto_avg, amp_avg, evolution, num_scored) or None if no
    scored skills exist.
    """
    auto_sum = 0.0
    amp_sum = 0.0
    total_weight = 0.0
    num_scored = 0

    for skill in occ.get("essential_skills", []):
        sc = skill_scores.get(skill["uri"])
        if sc:
            auto_sum += sc["automation_risk"] * ESSENTIAL_WEIGHT
            amp_sum += sc["amplification_potential"] * ESSENTIAL_WEIGHT
            total_weight += ESSENTIAL_WEIGHT
            num_scored += 1

    for skill in occ.get("optional_skills", []):
        sc = skill_scores.get(skill["uri"])
        if sc:
            auto_sum += sc["automation_risk"] * OPTIONAL_WEIGHT
            amp_sum += sc["amplification_potential"] * OPTIONAL_WEIGHT
            total_weight += OPTIONAL_WEIGHT
            num_scored += 1

    if total_weight == 0:
        return None

    auto_avg = round(auto_sum / total_weight, 1)
    amp_avg = round(amp_sum / total_weight, 1)
    evolution = round((auto_avg * amp_avg) / 10, 1)
    return auto_avg, amp_avg, evolution, num_scored


# ---------------------------------------------------------------------------
# Adjacency computation
# ---------------------------------------------------------------------------

def compute_adjacency(occ_data_list):
    """Compute Jaccard similarity between all occupation pairs using essential skills.

    Returns a dict: occ_index -> list of (other_index, jaccard_score).
    Only pairs with jaccard >= MIN_JACCARD_OVERLAP are included.
    """
    n = len(occ_data_list)

    # Pre-build essential skill URI sets per occupation
    ess_sets = []
    for occ in occ_data_list:
        ess_sets.append(occ["_ess_uris"])

    # Build inverted index: skill_uri -> set of occ indices
    skill_to_occs = defaultdict(set)
    for i, s in enumerate(ess_sets):
        for uri in s:
            skill_to_occs[uri].add(i)

    # For each occupation, find candidate neighbours via inverted index
    adjacency = defaultdict(list)
    seen_pairs = set()

    for i in range(n):
        if not ess_sets[i]:
            continue

        # Collect all occupations that share at least one essential skill
        candidates = set()
        for uri in ess_sets[i]:
            candidates.update(skill_to_occs[uri])
        candidates.discard(i)

        for j in candidates:
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            seen_pairs.add((j, i))

            intersection = len(ess_sets[i] & ess_sets[j])
            union = len(ess_sets[i] | ess_sets[j])
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= MIN_JACCARD_OVERLAP:
                adjacency[i].append((j, jaccard))
                adjacency[j].append((i, jaccard))

    return adjacency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Portfolio data builder")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    occupations = load_json("esco_occupations.json")
    skill_scores_list = load_json("skill_scores.json")

    print(f"  Occupations loaded: {len(occupations)}")
    print(f"  Skill scores loaded: {len(skill_scores_list)}")

    # Build skill_scores lookup: uri -> {automation_risk, amplification_potential, rationale}
    skill_scores = {}
    for s in skill_scores_list:
        uri = s.get("uri", "")
        if uri and s.get("automation_risk") is not None and s.get("amplification_potential") is not None:
            entry = {
                "automation_risk": float(s["automation_risk"]),
                "amplification_potential": float(s["amplification_potential"]),
            }
            if s.get("rationale"):
                entry["rationale"] = s["rationale"]
            skill_scores[uri] = entry

    print(f"  Skills with valid scores: {len(skill_scores)}")

    # Load occupation narratives (optional — continue without if missing)
    narratives_path = os.path.join(DATA_DIR, "occupation_narratives.json")
    narratives_by_uri = {}
    if os.path.exists(narratives_path):
        with open(narratives_path, encoding="utf-8") as f:
            narratives_list = json.load(f)
        for narr in narratives_list:
            uri = narr.get("uri", "")
            if uri:
                narratives_by_uri[uri] = narr
        print(f"  Occupation narratives loaded: {len(narratives_by_uri)}")
    else:
        print(f"  WARNING: {narratives_path} not found — skipping narratives")

    # ------------------------------------------------------------------
    # 2. Collect all skill URIs and titles, build short ID map
    # ------------------------------------------------------------------
    skill_info = {}  # uri -> {title, automation_risk, amplification_potential}
    for occ in occupations:
        for skill in occ.get("essential_skills", []) + occ.get("optional_skills", []):
            uri = skill["uri"]
            if uri not in skill_info:
                sc = skill_scores.get(uri)
                info_entry = {
                    "title": skill.get("title", ""),
                    "automation_risk": sc["automation_risk"] if sc else None,
                    "amplification_potential": sc["amplification_potential"] if sc else None,
                }
                if sc and sc.get("rationale"):
                    info_entry["rationale"] = sc["rationale"]
                skill_info[uri] = info_entry

    all_skill_uris = set(skill_info.keys())
    uri_to_short = build_short_id_map(all_skill_uris)
    print(f"  Unique skills across occupations: {len(all_skill_uris)}")
    print(f"  Short IDs generated: {len(uri_to_short)}")

    # ------------------------------------------------------------------
    # 3. Aggregate scores and filter occupations
    # ------------------------------------------------------------------
    occ_data = []
    for occ in occupations:
        result = aggregate_occupation_scores(occ, skill_scores)
        if result is None:
            continue

        auto_avg, amp_avg, evolution, num_scored = result
        ess_uris = set(s["uri"] for s in occ.get("essential_skills", []))
        all_uris = ess_uris | set(s["uri"] for s in occ.get("optional_skills", []))

        occ_data.append({
            "_ess_uris": ess_uris,
            "_all_uris": all_uris,
            "_occ": occ,
            "auto": auto_avg,
            "amp": amp_avg,
            "evolution": evolution,
            "quadrant": assign_quadrant(auto_avg, amp_avg),
            "num_scored": num_scored,
        })

    print(f"  Occupations with scored skills: {len(occ_data)}")

    # ------------------------------------------------------------------
    # 4. Compute adjacency
    # ------------------------------------------------------------------
    print("\nComputing Jaccard adjacency (essential skills only)...")
    adjacency = compute_adjacency(occ_data)

    total_edges = sum(len(v) for v in adjacency.values()) // 2
    print(f"  Adjacency pairs (>= {MIN_JACCARD_OVERLAP} overlap): {total_edges}")

    # ------------------------------------------------------------------
    # 5. Build skills dict
    # ------------------------------------------------------------------
    # Only include skills that are referenced by at least one included occupation
    referenced_uris = set()
    for od in occ_data:
        referenced_uris.update(od["_all_uris"])

    skills_dict = {}
    for uri in referenced_uris:
        short_id = uri_to_short[uri]
        info = skill_info[uri]
        entry = {"t": info["title"]}
        if info["automation_risk"] is not None:
            entry["a"] = round(info["automation_risk"], 1)
        else:
            entry["a"] = None
        if info["amplification_potential"] is not None:
            entry["m"] = round(info["amplification_potential"], 1)
        else:
            entry["m"] = None
        if info.get("rationale"):
            entry["r"] = info["rationale"]
        skills_dict[short_id] = entry

    print(f"\nSkills in output: {len(skills_dict)}")

    # ------------------------------------------------------------------
    # 6. Build occupation entries with adjacency
    # ------------------------------------------------------------------
    occ_output = []
    for i, od in enumerate(occ_data):
        occ = od["_occ"]
        hierarchy = occ.get("hierarchy", [])

        # Essential and optional skill short IDs
        se = [uri_to_short[s["uri"]] for s in occ.get("essential_skills", [])]
        so = [uri_to_short[s["uri"]] for s in occ.get("optional_skills", [])]

        # Build adjacency list: top 8 neighbours with HIGHER evolution potential
        adj_raw = adjacency.get(i, [])
        adj_candidates = []
        for j, jaccard in adj_raw:
            other = occ_data[j]
            if other["evolution"] <= od["evolution"]:
                continue
            adj_candidates.append((j, jaccard, other))

        # Sort by evolution_potential descending
        adj_candidates.sort(key=lambda x: -x[2]["evolution"])
        adj_candidates = adj_candidates[:MAX_ADJACENT]

        adj_list = []
        for j, jaccard, other in adj_candidates:
            other_occ = other["_occ"]

            # Gap skills: in adjacent but not in current (all skills)
            gap_uris = other["_all_uris"] - od["_all_uris"]

            # Sort gap skills by amplification_potential (descending), nulls last
            gap_with_scores = []
            for uri in gap_uris:
                info = skill_info.get(uri, {})
                amp = info.get("amplification_potential")
                gap_with_scores.append((uri, amp if amp is not None else -1))
            gap_with_scores.sort(key=lambda x: -x[1])
            gap_ids = [uri_to_short[uri] for uri, _ in gap_with_scores[:MAX_GAP_SKILLS]]

            adj_list.append({
                "s": slugify(other_occ["title"]),
                "t": other_occ["title"],
                "ov": round(jaccard, 2),
                "e": other["evolution"],
                "q": other["quadrant"],
                "gap": gap_ids,
            })

        entry = {
            "t": occ["title"],
            "s": slugify(occ["title"]),
            "c": occ.get("isco_code", ""),
            "cat": get_sub_major_group(hierarchy),
            "mg": get_major_group(hierarchy),
            "q": od["quadrant"],
            "e": od["evolution"],
            "ar": od["auto"],
            "ap": od["amp"],
            "se": se,
            "so": so,
            "adj": adj_list,
        }

        # Add narrative if available
        narr = narratives_by_uri.get(occ.get("uri", ""))
        if narr:
            compact_narr = {}
            if narr.get("evolution_story"):
                compact_narr["story"] = narr["evolution_story"]
            if narr.get("time_savings_pct") is not None:
                compact_narr["ts"] = narr["time_savings_pct"]
            if narr.get("automated_tasks"):
                compact_narr["auto"] = narr["automated_tasks"]
            if narr.get("amplified_capabilities"):
                compact_narr["amp"] = narr["amplified_capabilities"]
            if narr.get("ai_tools_applicable"):
                compact_narr["tools"] = narr["ai_tools_applicable"]
            if narr.get("rebalanced_week"):
                compact_narr["week"] = narr["rebalanced_week"]
            if narr.get("timeline"):
                compact_narr["tl"] = narr["timeline"]
            if narr.get("advice"):
                compact_narr["adv"] = narr["advice"]
            if compact_narr:
                entry["n"] = compact_narr

        occ_output.append(entry)

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    output = {
        "skills": skills_dict,
        "occupations": occ_output,
    }

    os.makedirs("site", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    file_size = os.path.getsize(OUT_PATH)

    # ------------------------------------------------------------------
    # 8. Stats
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Output summary")
    print(f"{'=' * 60}")
    print(f"  Output file: {OUT_PATH}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Skills: {len(skills_dict)}")
    print(f"  Occupations: {len(occ_output)}")

    scored_skills = sum(1 for s in skills_dict.values() if s["a"] is not None)
    print(f"  Skills with scores: {scored_skills} / {len(skills_dict)}")

    skills_with_rationales = sum(1 for s in skills_dict.values() if s.get("r"))
    print(f"  Skills with rationales: {skills_with_rationales} / {len(skills_dict)}")

    if narratives_by_uri:
        occs_with_narratives = sum(1 for o in occ_output if o.get("n"))
        print(f"  Occupations with narratives: {occs_with_narratives} / {len(occ_output)}")

    adj_counts = [len(o["adj"]) for o in occ_output]
    with_adj = sum(1 for c in adj_counts if c > 0)
    print(f"  Occupations with adjacency: {with_adj} / {len(occ_output)}")
    if adj_counts:
        print(f"  Adjacent occupations per entry: "
              f"avg={sum(adj_counts)/len(adj_counts):.1f}  "
              f"min={min(adj_counts)}  max={max(adj_counts)}")

    # Quadrant distribution
    quad_counts = defaultdict(int)
    for o in occ_output:
        quad_counts[o["q"]] += 1
    print(f"\n  Quadrant distribution:")
    for q in ["TRANSFORM", "SHRINK", "EVOLVE", "STABLE"]:
        count = quad_counts.get(q, 0)
        pct = count / len(occ_output) * 100 if occ_output else 0
        print(f"    {q:12s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
