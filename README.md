# AI-ISCO: Job Evolution Explorer

[![Live site](https://img.shields.io/badge/live-jorisdevreede.github.io%2FAI--ISCO-2ea44f)](https://jorisdevreede.github.io/AI-ISCO/)
[![Deploy to GitHub Pages](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml/badge.svg)](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![ESCO v1.2.1](https://img.shields.io/badge/data-ESCO%20v1.2.1-003399)
[![Licence: MIT code, CC BY 4.0 data](https://img.shields.io/badge/licence-MIT%20code%20%C2%B7%20CC%20BY%204.0%20data-lightgrey)](THIRD-PARTY-NOTICES.md)

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
| Pages built around questions | "How will my job be affected?", "which sectors, and how differently?", "how will this skill evolve?" and "how sure is this?" each have a page ([The frontend](#the-frontend)) |
| A second opinion on every score | The same rubric through a different kind of model, with both score sets and their comparison published ([Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy)) |
| No build step | Static pages in vanilla JS, served from `site/` |

Started from [karpathy/jobs](https://github.com/karpathy/jobs), whose BLS pipeline is still in this repository (see [Licensing and attribution](#licensing-and-attribution)). The ESCO skill-level pipeline, the narratives and every page of the site were written for this project. This publication uses the [ESCO](https://esco.ec.europa.eu/) classification of the European Commission.

## Quick start

Browse the published analysis locally. The scored data is committed, so this needs no install and no API key:

```bash
git clone https://github.com/Jorisdevreede/AI-ISCO.git
cd AI-ISCO
python3 -m http.server 8000 --directory site
# open http://localhost:8000
```

Re-run part of the analysis yourself. This scores a 300-skill sample with the optional TypeSafe scorer and compares it with the published Gemini scores (results of the full run are in [Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy)):

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
                               (Gemini scores each          (optional: TypeSafe, same
                                skill on 2 axes)             rubric as typed judgments)
                                         │                              │
                               skill_scores.json            skill_scores_typesafe.json
                               (13,939 skills scored)       (committed; every later step
                                         │                   takes --scorer typesafe and
                                         │                   writes *_typesafe.json files)
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

An optional second scorer that asks [TypeSafe](https://typesafe.ai)'s System One API instead of a generative LLM. It applies the same rubric: each skill is one request carrying two `Score` questions, one per axis, whose levels are the five rubric bands above. The answer arrives typed, as a probability distribution over the bands, and the 1-10 value is the probability-weighted band centre (1.5, 3.5 … 9.5). ESCO knowledge items ("types of sugars") get their own automation question, asking whether AI takes over the work the knowledge is applied in rather than whether AI can recall it.

```python
QUESTIONS = {
    "automation": Score(
        instructions="How likely is it that AI will automate the skill described in `skill` within the next 5 to 10 years?",
        criteria=AUTOMATION_LEVELS,       # the five rubric bands, each a standalone description
    ),
    "amplification": Score(instructions=..., criteria=AMPLIFICATION_LEVELS),
}
response = client.system_one(state={"skill": {...}}, questions=QUESTIONS, model=DEFAULT_MODEL)
```

**One request, as it went out and came back.** The state is the ESCO skill; the answer for each question is a probability per rubric band, a score and a confidence:

```python
state = {"skill": {"title": "operate cash register",
                   "description": "Register and handle cash transactions by using point of sale register.",
                   "type": "skill"}}

# answer, as stored in data/skill_scores_typesafe.json (model jev-1.13.0)
{"automation_probs":    [0.00, 0.00, 0.08, 0.55, 0.37],   # cannot … fully automatable
 "amplification_probs": [0.26, 0.51, 0.19, 0.04, 0.00],   # minimal … transformative
 "automation_risk": 8.04,          "automation_confidence": 0.62,
 "amplification_potential": 3.54,  "amplification_confidence": 0.56}
```

Gemini gave the same skill 9 and 3, with the sentence "Self-checkout and automated payment systems have already largely automated this role." jev returns no text, so on the site its scores borrow Gemini's rationale, marked with an "i" that says who wrote it and for which scores.

**What came out.** All 13,939 skills were scored in one run of about 16 minutes with no failed requests. The full output is in `data/skill_scores_typesafe.json`. After `uv run python ingest_esco.py`, `uv run python compare_skill_scores.py --occupations` reproduces every number below from it and the published Gemini scores (13,475 skills appear in both sets).

| | Automation risk | Amplification |
|---|---|---|
| Rank correlation with Gemini, per skill (Spearman) | 0.82 | 0.86 |
| Same rubric band | 42% | 54% |
| Within one band | 92% | 98% |
| Mean gap on the 1-10 scale | 1.26 | 1.03 |
| Mean score, Gemini → jev | 4.97 → 5.80 | 6.17 → 5.49 |
| Rank correlation per occupation, after the roll-up | 0.93 | 0.94 |

The two models rank skills and occupations much alike, but they use the scale differently: jev scores automation about 0.8 higher and amplification about 0.7 lower. Because the quadrants are a hard cut at 6, that shift moves a lot of occupations across a line even though their ranking barely changes:

| Gemini ↓ / jev → | TRANSFORM | SHRINK | EVOLVE | STABLE |
|---|---|---|---|---|
| **TRANSFORM** | 432 | 75 | 0 | 1 |
| **SHRINK** | 0 | 182 | 0 | 2 |
| **EVOLVE** | 671 | 102 | 538 | 250 |
| **STABLE** | 1 | 170 | 11 | 608 |

58% of the 3,043 occupations land in the same quadrant under both models. The largest single move is 671 occupations from EVOLVE to TRANSFORM: same amplification story, but jev sees more of the work as automatable. This is the most useful uncertainty estimate the project has, and the [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) page computes the same comparison live (over the 3,039 occupations in the site's index, so its counts differ by a few). A quadrant label depends on which model you ask and on a threshold, so read it as a region, not a verdict. Neither set is ground truth: nobody measured a real job.

The largest per-skill disagreements are instructive. jev rates physical and sensory skills as far more automatable than Gemini does ("display spirits" 2 vs 8.7, "sing" 2 vs 8.3, "use hand pliers" 2 vs 7.1), which suggests it reads the rubric's "automation" as including machinery, where Gemini reads it as AI software. Agreement is also not highest where jev is most confident: in its top confidence band the same-band rate drops again.

This project is not affiliated with or endorsed by TypeSafe AI, Inc.

How the two scorers differ in shape:

| | `score_skills.py` | `score_skills_typesafe.py` |
|---|---|---|
| Model | Gemini Flash via OpenRouter (generative) | TypeSafe System One (judgments only, no text) |
| Request shape | 10 skills per call, JSON array back | 1 skill per call, two typed answers back |
| Output handling | Strip code fences, fix trailing commas, `json.loads`, retry on malformed output | None, the SDK returns typed objects |
| Score | Integer 1-10 | Continuous, plus per-band probabilities and a confidence per axis |
| Rationale text | Yes, 1-2 sentences per skill | No; the site borrows Gemini's, marked as such |
| On the published site | Yes, the default | Yes, through the "Scores from" switch |

**Commands:**

```bash
uv run python score_skills_typesafe.py                      # all skills, resumes from the checkpoint
uv run python score_skills_typesafe.py --sample 300         # fixed random sample (--seed to vary it)
uv run python score_skills_typesafe.py --start 0 --end 50   # a slice
uv run python score_skills_typesafe.py --workers 6 --rps 15 # concurrency and request starts per second
uv run python score_skills_typesafe.py --force              # ignore the checkpoint and re-score

uv run python compare_skill_scores.py                       # comparison with the Gemini scores
uv run python compare_skill_scores.py --occupations         # plus occupation roll-up and quadrant table

uv run python aggregate_scores.py --scorer typesafe         # site/data_typesafe.json
uv run python build_portfolio_data.py --scorer typesafe     # site/portfolio_data_typesafe.json, with borrowed rationales
uv run python build_site_indexes.py --scorer typesafe       # the *_typesafe.json index files
```

- **Output:** `data/skill_scores_typesafe.json` (committed), same fields as `skill_scores.json` plus `automation_probs`, `amplification_probs`, a confidence per axis and the model version
- **Resume:** checkpoints every 250 skills and on exit; re-running skips what is already scored
- **Rate limit:** keep `--rps` under the limit in TypeSafe's own documentation; the SDK retries with backoff on 429
- **Key:** `TYPESAFE_API_KEY` in `.env`, or the macOS keychain item `typesafe-api-key`. Never commit it
- **Seeing it in the site:** the "Scores from" switch in the navigation bar, or `?scorer=typesafe` on any page (see [The frontend](#the-frontend))

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

### Step 6: Build the site indexes (`build_site_indexes.py`)

The pages should not download 13 MB to draw a search box. This step reads `site/data.json` and `site/portfolio_data.json` and writes the small files each page actually needs:

| File | What is in it | Loaded by |
|---|---|---|
| `site/search_index.json` | Title, slug, ISCO code, both scores, quadrant and ESCO's alternative labels per occupation (80 KB gzipped) | Find a job, Browse sectors, What we found |
| `site/groups.json` | Every ISCO group at every level: job count, quadrant counts, score spread, the most and least exposed jobs, children and parent | Browse sectors, What we found |
| `site/stats.json` | Every number the pages quote: totals, quadrant shares, the share of occupations within 0.5 of a cut-off, the build date | all pages |
| `site/skill_index.json` | Title, both scores and occupation counts per skill | Look up a skill, Browse sectors |
| `site/skill_occupations.json` | Which occupations need each skill, essential and optional | Look up a skill |

Alternative labels are why a search for "programmer" finds Software developer. No number on the site is typed into the prose: the pages compute them from these files as they load.

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

Static pages in vanilla JS: ES modules, no framework, no build step. The site is organised around the questions a visitor arrives with, not around the taxonomy:

| The visitor asks | Page |
|---|---|
| "How will my profession be affected?" | [Find a job](https://jorisdevreede.github.io/AI-ISCO/) → the job page |
| "How will someone else's profession be affected?" | The same job page in neutral wording, with a link you can share |
| "Which groups of professions are affected, and how differently?" | [Browse sectors](https://jorisdevreede.github.io/AI-ISCO/groups.html) |
| "How will a skill evolve?" | [Look up a skill](https://jorisdevreede.github.io/AI-ISCO/skill.html) |
| "Where could I move next, and what should I learn?" | The job page: nearby jobs and skills to learn |
| "How sure is this?" | On every quadrant badge, and on [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) |

The decisions behind this layout, and the audit findings that led to them, are in [docs/frontend-redesign-brief.md](docs/frontend-redesign-brief.md).

Deep links are hash-based, so GitHub Pages serves them unchanged:

```
job.html#software-developer                   my-job wording
job.html#software-developer&for=other         neutral wording, for someone else's job
job.html#optical-engineer&from=unit:2149      adds a way back to that group
groups.html#g=major:2&view=ranked|scatter|treemap|table
skill.html#59b27e7b
```

`site/scorer.js` lets every page show a second set of scores. Next to `data.json`, `portfolio_data.json` and the index files sit `_typesafe` copies built with `--scorer typesafe`. A **Scores from** switch in the navigation bar chooses between them, remembers the choice across pages, and can be set with `?scorer=typesafe`. With the second set on screen a banner says that the narratives and skill rationales were written by Gemini for its own scores, and each borrowed rationale carries an "i" with the scores it was written for. A checkout without the `_typesafe` files shows no switch.

### Find a job ([index.html](https://jorisdevreede.github.io/AI-ISCO/))

The first page is a search box. It matches titles and ESCO's alternative labels, so "programmer" finds Software developer. Below it: a few example jobs, the jobs you viewed recently (kept in your browser only), and a way into the sectors. It loads the 80 KB search index and nothing else.

### The job page ([job.html#software-developer](https://jorisdevreede.github.io/AI-ISCO/job.html#software-developer))

Everything about one occupation, and the one page every other surface links to:

- **Both scores** with what each one means, and a quadrant badge that explains why the job landed there and whether it sits within 0.5 of a cut-off
- **What to do next** before any detail: a job with high automation risk never gets a verdict without a next step
- **Where the skills sit** — every skill plotted on the two axes, cut-off lines at 6, with a table alternative
- **Why each essential skill scores as it does** — the model's rationale per skill
- **How the work could change** — the narrative, what AI takes on and what it amplifies, the rebalanced week
- **Where you could move next** and **skills you could learn** — nearby jobs by skill overlap, each labelled for what it is (a step up, a sideways move, or more exposed)
- `&for=other` switches the wording to neutral for someone else's job and keeps it out of your recently viewed list

`portfolio.html#<slug>` links from before the redesign redirect here.

### Browse sectors ([groups.html](https://jorisdevreede.github.io/AI-ISCO/groups.html))

Any ISCO-08 group at any level (10 major groups → 43 sub-major → 130 minor → 436 unit groups): how its jobs split over the four quadrants, then four views of the same jobs — a ranked dot plot, a scatter, a drill-down treemap and a sortable table — plus the most and least exposed jobs, the skills that drive the group, and a side-by-side comparison of two groups. The treemap is keyboard-navigable and every chart has a text summary and a table alternative. `explorer.html` redirects here.

### Look up a skill ([skill.html](https://jorisdevreede.github.io/AI-ISCO/skill.html))

Both scores for one skill, the model's rationale, the occupations that need it (essential and optional) and the skills it most often appears with.

### What we found ([insights.html](https://jorisdevreede.github.io/AI-ISCO/insights.html))

Aggregated findings across all occupations, written up as a newspaper-style article. Every number in it is computed from the published data as the page loads.

### How sure is this? ([method.html](https://jorisdevreede.github.io/AI-ISCO/method.html))

What the scores are (model estimates, nobody measured a real job), how the quadrants are cut, how many occupations sit close enough to a cut-off that a small change in the scores would move them, and how far a second model agrees: the share of occupations in the same quadrant, the rank correlations and the quadrant-by-quadrant table, computed in the browser from the two published score files.

### How the front end is put together

| Where | What |
|---|---|
| `site/js/*.js` | Shared modules: search ranking, quadrant and near-the-line rules, URL state, group statistics, formatting, data loading, the navigation and attribution footer, an accessible combobox, the quadrant badge, the treemap layout. [site/js/README.md](site/js/README.md) documents each one |
| `site/js/pages/*.js` | One thin DOM module per page, plus a pure `*-model.js` beside it that holds the logic and imports nothing from the DOM |
| `site/css/app.css` | Shared tokens and components; each page adds its own small stylesheet |

Accessibility is part of done: real links and buttons, an ARIA 1.2 combobox with a live region, visible focus, text contrast of at least 4.5:1, touch targets of at least 24 px, no horizontal scroll at 390 px, and a loading and an error state for every fetch.

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
uv run python score_skills_typesafe.py    # Optional second scorer (resumable); see Step 2b
uv run python aggregate_scores.py         # Aggregate to occupation level (~seconds)
uv run python generate_narratives.py      # Generate evolution narratives (~hours, resumable)
uv run python build_portfolio_data.py     # Build portfolio adjacency data (~minutes)
uv run python build_site_indexes.py       # Small per-page index files under site/ (~seconds)
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

## Tests and code quality

```bash
uv sync --dev
uv run pytest --ignore=tests/e2e --cov --cov-branch    # Python: unit and characterisation tests
npm test                                               # JavaScript: node --test over the pure modules
uv run playwright install chromium                     # once, for the end-to-end tests
uv run pytest tests/e2e -q                             # the site's flows in a real browser (about a minute)
uvx ruff check aiisco tests build_*.py aggregate_scores.py ingest_esco.py score_skills*.py
uv run radon cc -s -n B aiisco                         # anything more complex than grade A
```

| What | How it is kept |
|---|---|
| Python coverage | Line and branch coverage over the pipeline scripts and the `aiisco/` package; CI fails below 95% |
| Behaviour | Golden-output tests were written before any refactoring, and the real pipeline output was compared byte for byte before and after |
| Unit size and complexity | Following the SIG maintainability guidelines: short functions, low cyclomatic complexity, at most four parameters, no duplicated blocks |
| Front-end logic | Pure modules (search ranking, quadrant rules, URL state, page models, treemap layout) run under `node --test`; no browser needed |
| The flows visitors take | `tests/e2e/` serves a copy of `site/` and drives it with Playwright: find a job by a synonym, every chip opens the job it names, group → job → back, neutral wording, the four group views and their table alternative, skill lookup, a keyboard-only path, the second score set and its borrowed rationales, no horizontal scroll at 390 px, no console errors. Set `AIISCO_E2E_CHANNEL=chrome` to use an installed Chrome |
| No network, no keys | Every model call is faked in the tests. The fixtures under `tests/fixtures/` are synthetic ([why](tests/fixtures/README.md)) |
| Known bugs | Behaviour that looks wrong but is published is pinned by a test marked `# BUG:` rather than silently changed, because fixing it changes the published numbers |

`.github/workflows/tests.yml` runs the Python tests with the coverage gate, ruff, the JavaScript tests and the end-to-end suite on every push and pull request. The files copied from karpathy/jobs are outside the scope of the tests.

The shared Python code lives in the `aiisco/` package: `esco.py` (reading ESCO), `rollup.py` (weights, the threshold, quadrants), `portfolio.py` (adjacency and gap skills), `site_indexes.py`, `stats.py`, `openrouter.py` and `checkpoint.py` (model calls, retry, resume) and `jsonio.py`. The scripts at the top level are thin entry points over it.

## Key files

| File | Purpose |
|------|---------|
| `ingest_esco.py` | Parse ESCO v1.2.1 CSVs into structured JSON |
| `score_skills.py` | Dual-axis LLM scoring of all 13,939 skills |
| `score_skills_typesafe.py` | Experiment: the same rubric scored as typed judgments with TypeSafe |
| `compare_skill_scores.py` | Compares the TypeSafe scores with the published Gemini scores: per skill, per occupation and per quadrant |
| `aggregate_scores.py` | Weighted skill→occupation aggregation + quadrant assignment |
| `generate_narratives.py` | LLM-generated AI evolution narratives for each occupation |
| `merge_narrative_shards.py` | Consolidate parallel narrative shards |
| `build_portfolio_data.py` | Jaccard adjacency, gap skills, compressed portfolio data |
| `site/scorer.js` | Picks which score files the pages load; shows a "Scores from" switch only when a local second set exists |
| `build_site_indexes.py` | The small index files each page loads (search, groups, stats, skills) |
| `aiisco/` | Shared, tested Python: ESCO reading, roll-up and quadrant rules, adjacency, model calls, checkpoints |
| `site/index.html` | Find a job: the search-first landing page |
| `site/job.html` | The one job page (`portfolio.html` redirects to it) |
| `site/groups.html` | Browse sectors: quadrant split, ranked, scatter, treemap, table (`explorer.html` redirects to it) |
| `site/skill.html`, `site/method.html`, `site/insights.html` | Skill lookup, the method page, the findings article |
| `site/js/`, `site/css/` | Shared ES modules and stylesheet, one page module per page |
| `tests/` | pytest for the Python, `node --test` for the JavaScript, synthetic fixtures |
| `docs/frontend-redesign-brief.md` | The questions the site answers and the decisions behind its layout |
| `data/esco/` | Raw ESCO v1.2.1 CSV files |
| `data/skill_scores.json` | All 13,939 skills scored on both axes (generated, not committed) |
| `data/occupation_narratives.json` | 3,039 occupation evolution narratives |

## Stack

- **Frontend:** Pure vanilla HTML/CSS/JS as ES modules — no framework, no build step
- **Visualization:** Canvas treemap, SVG scatter and dot plots, each with a table alternative
- **Tests:** pytest with coverage, `node --test`, ruff, radon; GitHub Actions
- **Backend:** Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- **LLM API:** OpenRouter (Gemini Flash via `google/gemini-3-flash-preview`)
- **Judgment API (optional experiment):** TypeSafe System One, model version pinned in the script
- **Hosting:** GitHub Pages via Actions
- **Dependencies:** httpx, python-dotenv, beautifulsoup4, typesafe-sdk

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: data/skill_scores.json not found.` from `aggregate_scores.py` or `build_portfolio_data.py` | The per-skill scores are a generated file and are not committed | Run `score_skills.py` first |
| `FileNotFoundError: … data/esco_skills.json` | The ingested ESCO JSON is generated, not committed | Run `uv run python ingest_esco.py` |
| Every batch prints `Parse error: 'OPENROUTER_API_KEY', retrying…` | `.env` is missing the key; the lookup error is caught by the parse-retry handler, so it looks like a bad model response | Add `OPENROUTER_API_KEY` to `.env` |
| `No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item 'typesafe-api-key'.` | `score_skills_typesafe.py` found neither | Add `TYPESAFE_API_KEY` to `.env` |
| `site/data.json` shows up as modified in `git status` after a local experiment | `aggregate_scores.py` copies its result straight into `site/data.json`, the file the live site serves | Check `git diff --stat site/` before committing; only commit it when you mean to publish new scores. `--scorer typesafe` writes to separate `_typesafe` files and leaves the published ones alone |
| Rate-limit errors from TypeSafe | Too many requests a minute | Lower `--rps`; the SDK already retries with backoff |
| `compare_skill_scores.py` matches fewer skills than were scored | It can only compare skills that appear in `site/portfolio_data.json` | Expected; the rest are skipped |

## Limitations

- **The scores are model judgments, not measurements.** There is no human-labelled ground truth in this repo.
- **The rubric is ambiguous about machines.** "Automation" can mean AI software or AI plus industrial machinery, and the published scores are not consistent about it ("sift powder" 9, "polish stone by hand" 2).
- **Quadrants are a hard cut at 6.** An occupation at 5.9 and one at 6.1 get different labels, so occupations near the threshold are less settled than the label suggests.
- **Narratives are LLM-generated** for 3,039 of the 3,043 occupations. Time savings, timelines and the rebalanced week are estimates, not forecasts.
- **English labels only.** The pipeline reads the `_en` ESCO files.
- **The job page downloads one large file.** `site/portfolio_data.json` holds every occupation's skills, so the first job page you open loads about 13 MB (less over the wire, and cached afterwards). It shows a loading state while it does. One file per job is a planned increment.
- **The per-skill Gemini scores are not committed.** They survive only in compressed form inside `site/portfolio_data.json`, so re-aggregating from scratch means re-scoring.
- **The optional TypeSafe scorer returns scores only,** no rationale text. With its scores on screen the site shows Gemini's rationales and narratives, marked as written for different numbers.
- **Two models, two answers.** The second scorer puts 58% of occupations in the same quadrant as the first (see Step 2b). Treat a quadrant as a region, not a verdict.

## FAQ

**Do I need an API key?**
Not to browse. The scored data is committed, so `python3 -m http.server 8000 --directory site` is enough. Keys are only for re-running the model steps.

**Why score skills instead of occupations?**
An occupation-level number cannot tell you which part of the job changes. Scoring the 13,939 skills and rolling them up (essential skills weighted 2x) shows which skills pull an occupation towards automation and which towards amplification, and it is what makes the Skill Portfolio Analyzer possible.

**Which scorer produced the numbers on the site?**
Gemini Flash, via `score_skills.py`, by default. The "Scores from" switch shows the same rubric scored by TypeSafe's jev model instead.

**How much do the two scorers agree?**
They rank skills and occupations much alike (rank correlation 0.82 to 0.86 per skill, 0.93 to 0.94 per occupation) but use the scale differently, so only 58% of occupations land in the same quadrant. The table is in [Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy).

**How is Evolution Potential calculated?**
`(automation_risk × amplification_potential) / 10`, on the occupation-level weighted averages.

## Licensing and attribution

Different parts of this repository carry different terms. [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) has the full list; in short:

| What | Terms |
|---|---|
| Code written for this project (the ESCO pipeline scripts, the `aiisco/` package, every page and module under `site/`, the tests) | [MIT](LICENSE) |
| The scores, quadrants, rationales and narratives this project generated | [CC BY 4.0](LICENSE-DATA) |
| ESCO classification (`data/esco/`, titles and descriptions in `site/`) | European Commission reuse terms, not covered by the two licences above |
| ISCO-08 group structure, titles and definitions | © 2012 International Labour Organization, not covered |
| BLS Occupational Outlook Handbook pages and the ISCO-SOC crosswalk | Public domain, source: U.S. Bureau of Labor Statistics |
| Files copied from karpathy/jobs (`score.py` and seven other scripts, `prompt.md`, `scores.json`, the BLS scrape) | No licence published upstream, so none granted here |

This publication uses the ESCO classification of the European Commission. The data in `site/` is a modified and adapted version of ESCO v1.2.1: the scores, quadrants, rationales and narratives are AI-generated additions (Google Gemini Flash via OpenRouter) and are not part of ESCO. The European Commission, the International Labour Organization, the U.S. Bureau of Labor Statistics, Google and TypeSafe do not endorse this project.
