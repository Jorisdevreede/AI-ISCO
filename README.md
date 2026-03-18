# AI-ISCO: Job Evolution Explorer

**Live:** [jorisdevreede.github.io/AI-ISCO](https://jorisdevreede.github.io/AI-ISCO/)

A deep analysis of how AI reshapes 3,000+ European occupations — not by guessing at the job level, but by scoring every individual skill on two dimensions and generating rich AI evolution narratives for each occupation.

- **Automation Risk (1-10):** How likely is AI to replace this skill entirely?
- **Amplification Potential (1-10):** How much can AI supercharge a human doing this skill?

Jobs where both scores are high don't just disappear — they **transform** into something better. That's the core insight.

Inspired by [karpathy/jobs](https://github.com/karpathy/jobs). Built on [ESCO](https://esco.ec.europa.eu/) by the European Commission.

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
                               score_skills.py (LLM scores each skill on 2 axes)
                                         │
                               skill_scores.json (13,939 skills scored)
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

Three single-page applications — pure vanilla JS, no framework, no build step.

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

## Setup

```bash
uv sync
```

Requires an OpenRouter API key in `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

### Full pipeline (from scratch)

```bash
uv run python ingest_esco.py              # Parse ESCO CSVs → JSON (~seconds)
uv run python score_skills.py             # LLM-score all 13,939 skills (~hours, resumable)
uv run python aggregate_scores.py         # Aggregate to occupation level (~seconds)
uv run python generate_narratives.py      # Generate evolution narratives (~hours, resumable)
uv run python build_portfolio_data.py     # Build portfolio adjacency data (~minutes)
```

### Resume after interruption

Both `score_skills.py` and `generate_narratives.py` auto-resume from checkpoints — just re-run them. Use `--force` to regenerate already-processed items.

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
| `aggregate_scores.py` | Weighted skill→occupation aggregation + quadrant assignment |
| `generate_narratives.py` | LLM-generated AI evolution narratives for each occupation |
| `merge_narrative_shards.py` | Consolidate parallel narrative shards |
| `build_portfolio_data.py` | Jaccard adjacency, gap skills, compressed portfolio data |
| `site/index.html` | Drill-down treemap explorer |
| `site/portfolio.html` | Skill Portfolio Analyzer with narratives |
| `site/explorer.html` | ISCO occupation explorer with search/filter |
| `data/esco/` | Raw ESCO v1.2.1 CSV files |
| `data/skill_scores.json` | All 13,939 skills scored on both axes |
| `data/occupation_narratives.json` | 3,039 occupation evolution narratives |

## Stack

- **Frontend:** Pure vanilla HTML/CSS/JS — no framework, no build step
- **Visualization:** Canvas-based treemap and scatter plots
- **Backend:** Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- **LLM API:** OpenRouter (Gemini Flash via `google/gemini-3-flash-preview`)
- **Hosting:** GitHub Pages via Actions
- **Dependencies:** httpx, python-dotenv, beautifulsoup4
