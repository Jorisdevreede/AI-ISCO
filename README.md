# AI-ISCO: Job Evolution Explorer

[![Live site](https://img.shields.io/badge/live-jorisdevreede.github.io%2FAI--ISCO-2ea44f)](https://jorisdevreede.github.io/AI-ISCO/)
[![Deploy to GitHub Pages](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml/badge.svg)](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![ESCO v1.2.1](https://img.shields.io/badge/data-ESCO%20v1.2.1-003399)

**Live:** [jorisdevreede.github.io/AI-ISCO](https://jorisdevreede.github.io/AI-ISCO/)

A deep analysis of how AI reshapes 3,000+ European occupations — not by guessing at the job level, but by scoring every individual skill on two dimensions and generating rich AI evolution narratives for each occupation.

```bash
git clone https://github.com/Jorisdevreede/AI-ISCO.git && cd AI-ISCO
python3 -m http.server 8000 --directory site    # no install, no API key → http://localhost:8000
```

## TL;DR

**The problem.** Most "AI and jobs" analyses give a whole occupation one exposure number. That hides what actually changes inside the job: which parts AI takes over, and which parts it makes more valuable.

**The solution.** AI-ISCO scores every one of the 13,939 ESCO skills on two independent axes, then rolls them up to 3,043 occupations:

- **Automation Risk (1-10):** How likely is AI to replace this skill entirely?
- **Amplification Potential (1-10):** How much can AI supercharge a human doing this skill?

Jobs where both scores are high don't just disappear — they **transform** into something better. That's the core insight.

| Why AI-ISCO | What you get |
|---|---|
| Skill-level scoring | 13,939 skills scored, aggregated with essential skills weighted 2x, so you can see *which* skills drive an occupation's score |
| Two axes, four quadrants | TRANSFORM, SHRINK, EVOLVE, STABLE instead of a single "exposure" number |
| A story per occupation | 3,039 evolution narratives with time savings, a rebalanced work week, timeline and career advice |
| Career moves | Adjacent occupations by skill overlap, plus the gap skills to learn |
| Two independent scorers | The published Gemini scores, checked against a second model that returns typed judgments ([Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy)) |
| No build step | Static pages in vanilla JS, served from `site/` |

Inspired by [karpathy/jobs](https://github.com/karpathy/jobs). Built on [ESCO](https://esco.ec.europa.eu/) by the European Commission.

## Quick start

Browse the published analysis locally. The scored data is committed, so this needs no install and no API key:

```bash
git clone https://github.com/Jorisdevreede/AI-ISCO.git
cd AI-ISCO
python3 -m http.server 8000 --directory site
# open http://localhost:8000
```

Re-run part of the analysis yourself. This scores a 300-skill sample with the TypeSafe scorer (about 20 seconds, under $0.01) and compares it with the published scores:

```bash
uv sync
uv run python ingest_esco.py                           # ESCO CSVs → data/esco_skills.json
echo "TYPESAFE_API_KEY=your_key_here" >> .env
uv run python score_skills_typesafe.py --sample 300
uv run python compare_skill_scores.py
```

The full pipeline, including the Gemini scoring and the narratives, is under [Setup](#setup).

## Data foundation

Built on [ESCO v1.2.1](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations), which maps **13,939 skills** across **3,043 occupations** in the full [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco08/) hierarchy.

| Dataset | Records | Source |
|---------|---------|--------|
| Occupations | 3,043 | ESCO v1.2.1 |
| Skills & competences | 13,939 | ESCO v1.2.1 (skill, knowledge, transversal, language) |
| Skill-occupation links | ~50,000+ | ESCO occupationSkillRelations (essential + optional) |
| AI evolution narratives | 3,039 | LLM-generated (Gemini Flash via OpenRouter) |
| ISCO hierarchy levels | 4 | Major (10) → Sub-major (43) → Minor (130) → Unit (436) |

### Comparison to Karpathy's approach

| | karpathy/jobs | AI-ISCO |
|---|---|---|
| Taxonomy | US BLS (342 occupations) | ESCO (3,043 occupations) |
| Scoring level | Occupation-level | Skill-level (13,939 skills aggregated) |
| Dimensions | Single AI exposure axis | Dual-axis (automation risk + amplification potential) |
| Narratives | None | Full evolution stories, time savings, career advice |
| Adjacency | None | Jaccard similarity between occupations with gap skills |

## The pipeline

```
ESCO CSVs ─→ ingest_esco.py ─→ esco_occupations.json + esco_skills.json
                                         │
                                         ├──────────────────────────────┐
                                         │                              │
                               score_skills.py              score_skills_typesafe.py
                               (Gemini scores each          (experiment: TypeSafe, same
                                skill on 2 axes)             rubric as typed judgments)
                                         │                              │
                               skill_scores.json            skill_scores_typesafe.json
                               (13,939 skills scored)                   │
                                         │                  compare_skill_scores.py
                                         │                  (vs. the published scores;
                                         │                   not part of the site build)
                                         │
                     aggregate_scores.py (weighted avg per occupation)
                                         │
                     occupation_scores.json + site/data.json
                                         │
                ┌────────────────────────┴────────────────────────┐
                │                                                 │
  generate_narratives.py                           build_portfolio_data.py
  (LLM evolution stories)                          (Jaccard adjacency, gap skills)
                │                                                 │
  occupation_narratives.json                       site/portfolio_data.json
                │                                                 │
                └─────────────────┬──────────────────────────────┘
                                  │
                       Static frontend (no build step)
                       Auto-deployed via GitHub Pages
```

### Step 1: Ingest ESCO taxonomy (`ingest_esco.py`)

Parses ESCO v1.2.1 CSV files and builds structured JSON. Joins skills to occupations via URI-based relations, resolves the full ISCO-08 hierarchy by walking the `broaderRelationsOccPillar` chain.

- **Input:** `data/esco/*.csv` (occupations, skills, occupationSkillRelations, ISCOGroups, broaderRelationsOccPillar)
- **Output:** `data/esco_occupations.json` (3,043 occupations with essential/optional skill lists) + `data/esco_skills.json` (13,939 skills with metadata)

### Step 2: Score every skill (`score_skills.py`)

Each of the 13,939 skills is scored by an LLM on two independent axes with a detailed rubric:

**Automation Risk (1-10):**
- 1-2: Fundamentally human — requires physical presence, deep empathy, or creative judgment
- 3-4: Hard to automate — complex reasoning, nuanced human interaction
- 5-6: Partially automatable — routine aspects can be handled by AI
- 7-8: Largely automatable — AI can handle most cases with human oversight
- 9-10: AI already outperforms humans — pattern matching, data processing, repetitive tasks

**Amplification Potential (1-10):**
- 1-2: Minimal AI leverage — physical/manual skills, simple procedures
- 3-4: Some enhancement — AI provides minor efficiency gains
- 5-6: Meaningful augmentation — AI tools significantly speed up work
- 7-8: Major productivity multiplier — AI enables 3-5x output
- 9-10: Transformative leverage — AI enables 10x+ output or entirely new capabilities

Each skill also receives a **rationale** explaining both scores in the context of current AI capabilities.

- **Model:** Gemini Flash via OpenRouter (configurable with `--model`)
- **Batching:** 10 skills per LLM call (configurable with `--batch-size`)
- **Resume:** Automatically skips already-scored skills; incremental checkpointing after each batch
- **Retry:** Exponential backoff on 402/429/5xx errors

### Step 2b (experiment): the same rubric as typed judgments (`score_skills_typesafe.py`)

A second, independent scorer that asks [TypeSafe](https://typesafe.ai)'s System One model instead of a generative LLM. It exists to answer one question: if a different kind of model applies the same rubric, do the scores hold up? The site still runs on the Gemini scores from step 2.

Each skill is one request carrying two `Score` questions, one per axis, whose levels are the five rubric bands above. The answer arrives typed, as a probability distribution over the bands:

```python
QUESTIONS = {
    "automation": Score(
        instructions="How likely is it that AI will automate the skill described in `skill` within the next 5 to 10 years?",
        criteria=AUTOMATION_LEVELS,       # the five rubric bands, each a standalone description
    ),
    "amplification": Score(instructions=..., criteria=AMPLIFICATION_LEVELS),
}
response = client.system_one(state={"skill": {...}}, questions=QUESTIONS, model="jev-1.13.0")
response.answers["automation"].probabilities   # e.g. 0.01, 0.05, 0.68, 0.25, 0.00 for "receive goods"
```

The 1-10 value is the probability-weighted band centre (1.5, 3.5 … 9.5), so "receive goods" above scores 5.86. ESCO knowledge items ("types of sugars") get their own automation question, asking whether AI takes over the work the knowledge is applied in rather than whether AI can recall it.

**The two scorers side by side:**

| | `score_skills.py` | `score_skills_typesafe.py` |
|---|---|---|
| Model | Gemini Flash via OpenRouter (generative) | TypeSafe `jev-1.13.0` (judgments only, no text) |
| Request shape | 10 skills per call, JSON array back | 1 skill per call, two typed answers back |
| Output handling | Strip code fences, fix trailing commas, `json.loads`, retry on malformed output | None, the SDK returns typed objects |
| Score | Integer 1-10 | Continuous 1.5-9.5, plus per-band probabilities and a confidence per axis |
| Rationale text | Yes, 1-2 sentences per skill | No |
| Full run, 13,939 skills | Hours | 935 s at 15 requests/s, 0 errors, 8.9M input tokens, $0.38 (2026-09-18) |
| Feeds the site | Yes | No |

**How the scores compare.** `compare_skill_scores.py --occupations` lines the TypeSafe scores up against the published Gemini scores (recovered from `site/portfolio_data.json`, since `skill_scores.json` is not committed). Gemini is a second opinion here, not ground truth:

| | Automation risk | Amplification potential |
|---|---|---|
| Skill level, Spearman rho (13,475 matched skills) | 0.82 | 0.86 |
| Skill level, within one rubric band | 91.6% | 97.5% |
| Skill level, same rubric band | 42.3% | 53.6% |
| Mean score, Gemini / TypeSafe | 4.97 / 5.80 | 6.17 / 5.49 |
| Occupation level, Spearman rho (3,043 occupations) | 0.93 | 0.94 |

What that shows:

- **Same ranking, different level.** The two models order skills and occupations alike, but TypeSafe scores automation about 0.8 higher and amplification about 0.7 lower.
- **The offset moves quadrants.** Against the fixed threshold of 6, only 57.8% of occupations land in the same quadrant, and 671 move from EVOLVE to TRANSFORM. With thresholds matched to Gemini's shares (6.5 for automation, 5.5 for amplification), agreement is 81.8%. Occupations near the threshold are less settled than the quadrant labels suggest.
- **Knowledge items agree least.** Rho for automation is 0.75 on knowledge items against 0.84 on skills. The knowledge-specific question helps only a little: on the same 59 items the mean gap fell from 1.31 to 1.13.
- **Confidence is a weak flag.** Same-band agreement runs 33%, 43%, 48%, 39% from the lowest to the highest confidence bucket, so it does not pick out the contested skills.
- **The rubric is ambiguous about machines.** Gemini scored "sift powder" 9 and "polish stone by hand" 2. Rewording the levels to mean AI software only raised band agreement but dropped rho from 0.84 to 0.76, so the original wording stayed.

**Commands:**

```bash
uv run python score_skills_typesafe.py                      # all skills, resumes from the checkpoint
uv run python score_skills_typesafe.py --sample 300         # fixed random sample (--seed to vary it)
uv run python score_skills_typesafe.py --start 0 --end 50   # a slice
uv run python score_skills_typesafe.py --workers 6 --rps 15 # concurrency and request starts per second
uv run python score_skills_typesafe.py --force              # ignore the checkpoint and re-score

uv run python compare_skill_scores.py                       # skill-level comparison
uv run python compare_skill_scores.py --occupations         # plus occupation roll-up and quadrant table
uv run python compare_skill_scores.py --top 25 --out data/skill_score_comparison.json
```

- **Output:** `data/skill_scores_typesafe.json`, same fields as `skill_scores.json` plus `automation_probs`, `amplification_probs`, a confidence per axis, the model version and the input tokens per request
- **Resume:** checkpoints every 250 skills and on exit; re-running skips what is already scored
- **Rate limit:** `--rps 15` is 900 requests a minute, under the API's 1,200; the SDK retries with backoff on 429
- **Key:** `TYPESAFE_API_KEY` in `.env`, or the macOS keychain item `typesafe-api-key`

### Step 3: Aggregate to occupations (`aggregate_scores.py`)

Computes weighted averages from skill-level to occupation-level scores:

- **Essential skills** weighted **2.0x** (core to the role)
- **Optional skills** weighted **1.0x** (supplementary)
- **Evolution Potential** = `(automation_risk × amplification_potential) / 10`

Assigns each occupation to a quadrant and outputs compact site data with the top 5 most-automated and top 5 most-amplified skills per occupation.

### Step 4: Generate AI evolution narratives (`generate_narratives.py`)

For each occupation, the LLM receives the full context — quadrant, aggregated scores, every scored essential skill with rationales, and the top automated/amplified skills — and generates a structured narrative:

| Field | Description |
|-------|-------------|
| `evolution_story` | 2-3 paragraphs in second person ("your role..."), describing how AI transforms the occupation. Considers skill interactions and workflow shifts. |
| `time_savings_pct` | Estimated percentage of work week AI could automate (0-80 range) |
| `automated_tasks` | 3-6 specific current tasks AI will handle or eliminate |
| `amplified_capabilities` | 3-6 capabilities where AI makes the human much more effective |
| `ai_tools_applicable` | 3-5 AI tool categories (e.g. "AI-powered diagnostic platforms") |
| `rebalanced_week` | Before/after percentage breakdown of a typical work week, including a `new_ai_augmented` category |
| `timeline` | Estimated horizon for significant AI impact (e.g. "3-5 years") |
| `advice` | 1-2 sentences of concrete, actionable career advice |

**Parallel execution:** Supports sharding via `--start`, `--end`, and `--output` flags. Multiple agents can process different occupation ranges simultaneously, writing to separate shard files. `merge_narrative_shards.py` consolidates all shards into the final file.

The narratives for all 3,039 occupations were generated using 3 parallel agents (2 Claude Code + 1 Codex) coordinated via [NTM](https://github.com/Dicklesworthstone/agentic_coding_flywheel_setup) with [Beads](https://github.com/Dicklesworthstone/agentic_coding_flywheel_setup) for task tracking.

### Step 5: Build portfolio data (`build_portfolio_data.py`)

Creates the compressed dataset for the Skill Portfolio Analyzer:

**Occupation adjacency** (Jaccard similarity):
- Computes pairwise Jaccard similarity on essential skills using an inverted index for efficiency
- Minimum overlap threshold: 15% (`MIN_JACCARD_OVERLAP = 0.15`)
- Only keeps adjacent occupations with higher evolution potential
- Maximum 8 adjacent occupations per entry

**Gap skills:**
- For each adjacency, identifies skills in the target occupation but not the source
- Sorted by amplification potential (most valuable skills first)
- Maximum 5 gap skills per adjacency

**Compression:** Skills use 8-character MD5 hash IDs with collision resolution. Occupation keys are heavily abbreviated (e.g. `t`=title, `ar`=automation_risk, `se`=essential_skills). Final output: ~9.6 MB.

## The quadrant model

Every occupation lands in one of four quadrants based on its aggregate scores (threshold = 6):

| Quadrant | Auto Risk | Amp Potential | What happens | Avg time savings |
|---|---|---|---|---|
| **TRANSFORM** | High (≥6) | High (≥6) | Job evolves into something new and better | ~61% |
| **SHRINK** | High (≥6) | Low (<6) | Job contracts — automation without upside | ~58% |
| **EVOLVE** | Low (<6) | High (≥6) | Job grows — AI augments without replacing | ~43% |
| **STABLE** | Low (<6) | Low (<6) | Job stays roughly the same | ~28% |

Distribution across 3,043 occupations:
- **EVOLVE:** 51.3% (1,561 occupations) — the largest group
- **STABLE:** 26.0% (790)
- **TRANSFORM:** 16.7% (508)
- **SHRINK:** 6.0% (184)

## The frontend

Four static pages — pure vanilla JS, no framework, no build step.

### Job Explorer ([index.html](https://jorisdevreede.github.io/AI-ISCO/))

Drill-down canvas treemap of all 3,000+ occupations through the ISCO-08 hierarchy (10 major groups → 43 sub-major → 130 minor → 436 unit groups → individual jobs).

- **Color modes:** Evolution Potential, Automation Risk, Amplification Potential, or Quadrant
- **Interaction:** Click to drill down, breadcrumb navigation to go back
- **Tooltips:** Job title, scores, number of skills

### Skill Portfolio Analyzer ([portfolio.html](https://jorisdevreede.github.io/AI-ISCO/portfolio.html))

Treats your career like an investment portfolio. Search any occupation to see:

- **2D skill scatter plot** — every skill plotted on automation risk vs amplification potential
- **Portfolio health score** — strong, mixed, or at-risk based on skill distribution
- **Depreciating skills** — high automation risk, losing value
- **Appreciating skills** — high amplification potential, gaining value
- **AI evolution narrative** — full story of how the occupation transforms, with timeline and career advice
- **Rebalanced work week** — before/after breakdown of how time allocation shifts
- **Evolution paths** — adjacent occupations with higher evolution potential and shared skill overlap
- **Rebalancing recommendations** — gap skills from adjacent TRANSFORM/EVOLVE careers you should learn

### ISCO Explorer ([explorer.html](https://jorisdevreede.github.io/AI-ISCO/explorer.html))

List-based browse and search interface with expandable detail rows:

- **Search** by occupation title, ISCO code, or category
- **Filter** by quadrant or evolution potential range
- **Sort** by any column
- **Detail panel** with full narrative, automated/amplified tasks, career advice, rebalanced week, and applicable AI tools

### Insights ([insights.html](https://jorisdevreede.github.io/AI-ISCO/insights.html))

Aggregated findings across all occupations, written up as a newspaper-style article.

## Setup

```bash
uv sync
```

Browsing the site needs no key. The pipeline steps that call a model read their key from `.env`:

```
OPENROUTER_API_KEY=your_key_here     # score_skills.py, generate_narratives.py
TYPESAFE_API_KEY=your_key_here       # score_skills_typesafe.py (experiment only)
```

| Way in | Command | When |
|---|---|---|
| Just look | `python3 -m http.server 8000 --directory site` | No install, no key |
| uv (recommended) | `uv sync` | Running any pipeline step |
| pip | `pip install httpx python-dotenv beautifulsoup4 typesafe-sdk` | No uv available; then run the scripts with `python` instead of `uv run python` |

### Full pipeline (from scratch)

```bash
uv run python ingest_esco.py              # Parse ESCO CSVs → JSON (~seconds)
uv run python score_skills.py             # LLM-score all 13,939 skills (~hours, resumable)
uv run python score_skills_typesafe.py    # Optional second opinion (~16 min, ~$0.40, resumable)
uv run python compare_skill_scores.py --occupations   # Optional: compare the two (~seconds)
uv run python aggregate_scores.py         # Aggregate to occupation level (~seconds)
uv run python generate_narratives.py      # Generate evolution narratives (~hours, resumable)
uv run python build_portfolio_data.py     # Build portfolio adjacency data (~minutes)
```

### Resume after interruption

`score_skills.py`, `score_skills_typesafe.py` and `generate_narratives.py` auto-resume from checkpoints — just re-run them. Use `--force` to regenerate already-processed items.

### Parallel narrative generation

```bash
# Run in separate terminals or tmux panes:
uv run python generate_narratives.py --start 0 --end 1000 --output data/shard_1.json
uv run python generate_narratives.py --start 1000 --end 2000 --output data/shard_2.json
uv run python generate_narratives.py --start 2000 --end 3043 --output data/shard_3.json

# Merge when all complete:
uv run python merge_narrative_shards.py
```

### Serve locally

```bash
cd site && python -m http.server 8000
```

### Deploy

Push to master — GitHub Actions auto-deploys to GitHub Pages:

```bash
git push origin master
```

## Key files

| File | Purpose |
|------|---------|
| `ingest_esco.py` | Parse ESCO v1.2.1 CSVs into structured JSON |
| `score_skills.py` | Dual-axis LLM scoring of all 13,939 skills |
| `score_skills_typesafe.py` | Experiment: the same rubric scored as typed judgments with TypeSafe |
| `compare_skill_scores.py` | Compare the TypeSafe scores with the published scores, per skill and per occupation |
| `aggregate_scores.py` | Weighted skill→occupation aggregation + quadrant assignment |
| `generate_narratives.py` | LLM-generated AI evolution narratives for each occupation |
| `merge_narrative_shards.py` | Consolidate parallel narrative shards |
| `build_portfolio_data.py` | Jaccard adjacency, gap skills, compressed portfolio data |
| `site/index.html` | Drill-down treemap explorer |
| `site/portfolio.html` | Skill Portfolio Analyzer with narratives |
| `site/explorer.html` | ISCO occupation explorer with search/filter |
| `data/esco/` | Raw ESCO v1.2.1 CSV files |
| `data/skill_scores.json` | All 13,939 skills scored on both axes (generated, not committed) |
| `data/skill_scores_typesafe.json` | The TypeSafe scores with per-band probabilities (generated, not committed) |
| `data/occupation_narratives.json` | 3,039 occupation evolution narratives |

## Stack

- **Frontend:** Pure vanilla HTML/CSS/JS — no framework, no build step
- **Visualization:** Canvas-based treemap and scatter plots
- **Backend:** Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- **LLM API:** OpenRouter (Gemini Flash via `google/gemini-3-flash-preview`)
- **Judgment API (experiment):** TypeSafe System One (`jev-1.13.0`, pinned)
- **Hosting:** GitHub Pages via Actions
- **Dependencies:** httpx, python-dotenv, beautifulsoup4, typesafe-sdk

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: data/skill_scores.json not found.` from `aggregate_scores.py` or `build_portfolio_data.py` | The per-skill scores are a generated file and are not committed | Run `score_skills.py` first |
| `FileNotFoundError: … data/esco_skills.json` | The ingested ESCO JSON is generated, not committed | Run `uv run python ingest_esco.py` |
| Every batch prints `Parse error: 'OPENROUTER_API_KEY', retrying…` | `.env` is missing the key; the lookup error is caught by the parse-retry handler, so it looks like a bad model response | Add `OPENROUTER_API_KEY` to `.env` |
| `No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item 'typesafe-api-key'.` | `score_skills_typesafe.py` found neither | Add `TYPESAFE_API_KEY` to `.env` |
| `site/data.json` shows up as modified in `git status` after a local experiment | `aggregate_scores.py` copies its result straight into `site/data.json`, the file the live site serves | Check `git diff --stat site/` before committing; only commit it when you mean to publish new scores |
| Rate-limit errors from TypeSafe | More than 1,200 requests a minute | Lower `--rps` (default 15); the SDK already retries with backoff |
| `compare_skill_scores.py` matches fewer skills than were scored (13,475 of 13,939) | It can only compare skills that appear in `site/portfolio_data.json` | Expected; the rest are skipped |

## Limitations

- **The scores are model judgments, not measurements.** There is no human-labelled ground truth in this repo. Two different models agreeing (rho 0.82 and 0.86 per skill) shows the ranking is stable, not that it is right.
- **The rubric is ambiguous about machines.** "Automation" can mean AI software or AI plus industrial machinery, and the published scores are not consistent about it ("sift powder" 9, "polish stone by hand" 2).
- **Quadrants are a hard cut at 6.** An occupation at 5.9 and one at 6.1 get different labels. The second scorer puts only 57.8% of occupations in the same quadrant at that threshold, 81.8% once its level offset is removed.
- **Narratives are LLM-generated** for 3,039 of the 3,043 occupations. Time savings, timelines and the rebalanced week are estimates, not forecasts.
- **English labels only.** The pipeline reads the `_en` ESCO files.
- **The per-skill Gemini scores are not committed.** They survive only in compressed form inside `site/portfolio_data.json`, so re-aggregating from scratch means re-scoring.
- **The TypeSafe scorer cannot write rationales,** its confidence does not pick out the contested skills, and it was run against a single model version (`jev-1.13.0`).

## FAQ

**Do I need an API key?**
Not to browse. The scored data is committed, so `python3 -m http.server 8000 --directory site` is enough. Keys are only for re-running the model steps.

**Why score skills instead of occupations?**
An occupation-level number cannot tell you which part of the job changes. Scoring the 13,939 skills and rolling them up (essential skills weighted 2x) shows which skills pull an occupation towards automation and which towards amplification, and it is what makes the Skill Portfolio Analyzer possible.

**Which scorer produced the numbers on the site?**
Gemini Flash, via `score_skills.py`. The TypeSafe scorer is an experiment and feeds nothing on the site.

**Can I build the site from the TypeSafe scores instead?**
The file format is compatible: `cp data/skill_scores_typesafe.json data/skill_scores.json` and `aggregate_scores.py` runs unchanged. Three caveats: that script overwrites `site/data.json`, the quadrant thresholds would need recalibrating because TypeSafe sits at a different level, and the narratives prompt is built from per-skill rationales, which TypeSafe does not produce.

**Why do the two scorers disagree on quadrants when they agree on ranking?**
A constant offset. TypeSafe scores automation about 0.8 higher and amplification about 0.7 lower, which pushes many occupations across the fixed threshold of 6. See [Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy).

**What does a full scoring run cost?**
The TypeSafe run was measured: 8.9M input tokens, $0.38, 16 minutes. The cost of the original Gemini run was not recorded.

**How is Evolution Potential calculated?**
`(automation_risk × amplification_potential) / 10`, on the occupation-level weighted averages.
