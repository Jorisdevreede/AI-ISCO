# AI-ISCO: Job Evolution Explorer

[![Live site](https://img.shields.io/badge/live-jorisdevreede.github.io%2FAI--ISCO-2ea44f)](https://jorisdevreede.github.io/AI-ISCO/)
[![Deploy to GitHub Pages](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml/badge.svg)](https://github.com/Jorisdevreede/AI-ISCO/actions/workflows/pages.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![ESCO v1.2.1](https://img.shields.io/badge/data-ESCO%20v1.2.1-003399)
[![Licence: MIT code, CC BY 4.0 data](https://img.shields.io/badge/licence-MIT%20code%20%C2%B7%20CC%20BY%204.0%20data-lightgrey)](THIRD-PARTY-NOTICES.md)

**Live:** [jorisdevreede.github.io/AI-ISCO](https://jorisdevreede.github.io/AI-ISCO/)

A deep analysis of how AI reshapes 3,000+ European occupations — not by guessing at the job level, but by scoring every one of the 13,939 individual ESCO skills and rolling the answers up to each job.

```bash
git clone https://github.com/Jorisdevreede/AI-ISCO.git && cd AI-ISCO
python3 -m http.server 8000 --directory site    # no install, no API key → http://localhost:8000
```

## TL;DR

**The problem.** Most "AI and jobs" analyses give a whole occupation one exposure number. That hides what actually changes inside the job: which parts AI takes over, which parts it only assists with, and which parts are not about AI at all.

**What you see on the site today.** Every one of the 3,039 occupations is split into four shares of its skill weight, and lands in one of seven types:

> **Secondary school teacher** — AI can take over 21% · AI assists 40% · Machines can do 0% · Stays human 39% → **Augmented**

The shares come from scoring each of the 13,939 ESCO skills with [TypeSafe](https://typesafe.ai)'s jev model against the [scoring v2 rubric](docs/scoring-v2.md): six questions per skill, answered as probability distributions rather than prose. That is the default score set ([Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py)).

The original view — Gemini rating every skill on two 1-10 axes, with each job in one of four quadrants — is still published and is one switch away in the navigation bar.

| Why AI-ISCO | What you get |
|---|---|
| Skill-level scoring | 13,939 skills scored, rolled up with essential skills weighted 2x, so you can see *which* skills put a job where it is |
| Four shares, seven types | What AI can take over, what it assists with, what machinery does and what stays human — as shares of the job, not one exposure number |
| A second rubric, not a second opinion | Two score sets side by side, built on deliberately different models of what "automation" means ([Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy), [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py)) |
| The scoring itself, published | The six questions as they were asked, and the model's full answer distribution per skill, on the Skill scores page |
| A story per occupation | 3,039 evolution narratives with time savings, a rebalanced work week, timeline and career advice |
| Career moves | Adjacent occupations by skill overlap, plus the gap skills to learn |
| Pages built around questions | "How will my job be affected?", "which sectors, and how differently?", "how is a skill scored?" and "how sure is this?" each have a page ([The frontend](#the-frontend)) |
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

Re-run part of the analysis yourself. This scores a 300-skill sample against the rubric behind the default score set and prints how the sample fell into the four skill classes (results of the full run are in [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py)):

```bash
uv sync
uv run python ingest_esco.py                           # ESCO CSVs → data/esco_skills.json
echo "TYPESAFE_API_KEY=your_key_here" >> .env
uv run python score_skills_v2.py --sample 300
```

The full pipeline, including the Gemini scoring and the narratives, is under [Setup](#setup).

## Data foundation

Built on [ESCO v1.2.1](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations), which maps **13,939 skills** across **3,039 occupations** in the full [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco08/) hierarchy.

| Dataset | Records | Source |
|---------|---------|--------|
| Occupations | 3,039 | ESCO v1.2.1, after the ingest drops four duplicated rows |
| Skills & competences | 13,939 | ESCO v1.2.1 (skill, knowledge, transversal, language) |
| Skill-occupation links | ~50,000+ | ESCO occupationSkillRelations (essential + optional) |
| Skill scores, v2 rubric (the default) | 13,939 | TypeSafe System One, model `jev-1.13.0` — `data/skill_scores_v2.json` |
| Skill scores, original rubric | 13,939 | Gemini Flash via OpenRouter, and the same rubric re-scored by `jev-1.13.0` |
| AI evolution narratives | 3,039 | LLM-generated (Gemini Flash via OpenRouter) |
| ISCO hierarchy levels | 4 | Major (10) → Sub-major (43) → Minor (130) → Unit (436) |

### Comparison to Karpathy's approach

| | karpathy/jobs | AI-ISCO |
|---|---|---|
| Taxonomy | US BLS (342 occupations) | ESCO (3,039 occupations) |
| Scoring level | Occupation-level | Skill-level (13,939 skills aggregated) |
| Dimensions | Single AI exposure axis | Three separated axes (AI substitution, AI assistance, machine automation), four shares per job |
| Narratives | None | Full evolution stories, time savings, career advice |
| Adjacency | None | Jaccard similarity between occupations with gap skills |

## The pipeline

```
ESCO CSVs ─→ ingest_esco.py ─→ esco_occupations.json + esco_skills.json
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
     score_skills.py         score_skills_typesafe.py      score_skills_v2.py
     (Gemini, 2 axes,        (the same rubric, scored      (six questions, scored
      the original rubric)    as typed judgments)           as typed judgments)
              │                          │                          │
   skill_scores.json      skill_scores_typesafe.json   skill_scores_v2.json
   (--scorer gemini,      (--scorer typesafe,          (--scorer v2, the default
    no file suffix)        *_typesafe.json files)       on the site, *_v2.json)
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                     aggregate_scores.py (weighted roll-up per occupation)
                                         │
                     occupation_scores.json + site/data.json
                                         │
                ┌────────────────────────┴────────────────────────┐
                │                                                 │
  generate_narratives.py                           build_portfolio_data.py
  (LLM evolution stories, Gemini only)             (Jaccard adjacency, gap skills,
                │                                   per-unit job files)
  occupation_narratives.json                       site/portfolio_data.json
                │                                   + site/jobs/
                │                                                 │
                └─────────────────┬──────────────────────────────┘
                                  │
                       build_site_indexes.py
                       (the small per-page files under site/)
                                  │
                       Static frontend (no build step)
                       Auto-deployed via GitHub Pages
```

### Step 1: Ingest ESCO taxonomy (`ingest_esco.py`)

Parses ESCO v1.2.1 CSV files and builds structured JSON. Joins skills to occupations via URI-based relations, resolves the full ISCO-08 hierarchy by walking the `broaderRelationsOccPillar` chain.

- **Input:** `data/esco/*.csv` (occupations, skills, occupationSkillRelations, ISCOGroups, broaderRelationsOccPillar)
- **Output:** `data/esco_occupations.json` (3,039 occupations with essential/optional skill lists) + `data/esco_skills.json` (13,939 skills with metadata)

ESCO's own occupations CSV lists four occupations twice, the two rows differing only in `modifiedDate` — `early years teaching assistant`, `legal policy officer`, `mining, construction and civil engineering machinery distribution manager` and `food service worker` — and the ingest drops them, which is why the count is 3,039 and not 3,043.

### Step 2: Score every skill (`score_skills.py`)

The first rubric, and the one the site's alternative view still uses. Each of the 13,939 skills is scored by an LLM on two axes:

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

A second scorer that asks [TypeSafe](https://typesafe.ai)'s System One API instead of a generative LLM, holding the rubric fixed so that only the model changes. It is what showed that the rubric itself was the problem, and it is the experiment [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py) came out of. Each skill is one request carrying two `Score` questions, one per axis, whose levels are the five rubric bands above. The answer arrives typed, as a probability distribution over the bands, and the 1-10 value is the probability-weighted band centre (1.5, 3.5 … 9.5). ESCO knowledge items ("types of sugars") get their own automation question, asking whether AI takes over the work the knowledge is applied in rather than whether AI can recall it.

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
| **TRANSFORM** | 431 | 75 | 0 | 1 |
| **SHRINK** | 0 | 182 | 0 | 2 |
| **EVOLVE** | 671 | 102 | 537 | 249 |
| **STABLE** | 1 | 170 | 11 | 607 |

58% of the 3,039 occupations land in the same quadrant under both models. The largest single move is 671 occupations from EVOLVE to TRANSFORM: same amplification story, but jev sees more of the work as automatable. This is the most useful uncertainty estimate the project has, and the [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) page computes the same comparison live, from the two published search indexes rather than from anything typed here. A quadrant label depends on which model you ask and on a threshold, so read it as a region, not a verdict. Neither set is ground truth: nobody measured a real job.

The largest per-skill disagreements are instructive. jev rates physical and sensory skills as far more automatable than Gemini does ("display spirits" 2 vs 8.7, "sing" 2 vs 8.3, "use hand pliers" 2 vs 7.1), which suggests it reads the rubric's "automation" as including machinery, where Gemini reads it as AI software. Agreement is also not highest where jev is most confident: in its top confidence band the same-band rate drops again.

That is the finding that ended the experiment. Two models reading one word two ways is not a disagreement you can average out; it means the question was ambiguous. [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py) replaces the rubric rather than the model.

This project is not affiliated with or endorsed by TypeSafe AI, Inc.

How the two scorers differ in shape:

| | `score_skills.py` | `score_skills_typesafe.py` |
|---|---|---|
| Model | Gemini Flash via OpenRouter (generative) | TypeSafe System One (judgments only, no text) |
| Request shape | 10 skills per call, JSON array back | 1 skill per call, two typed answers back |
| Output handling | Strip code fences, fix trailing commas, `json.loads`, retry on malformed output | None, the SDK returns typed objects |
| Score | Integer 1-10 | Continuous, plus per-band probabilities and a confidence per axis |
| Rationale text | Yes, 1-2 sentences per skill | No; the site borrows Gemini's, marked as such |
| On the published site | Yes, as the alternative in the "Scores from" switch | Not in the switch; it feeds the same-rubric comparison on the [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) page |

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
- **Seeing it in the site:** not in the "Scores from" switch. `site/scorer.js` keeps this set for one job only — the same-rubric comparison the [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) page computes between Gemini and jev. A link shared before scoring v2 existed, carrying `?scorer=typesafe`, now lands on the default set instead

### Step 2c: a new rubric, and the default score set (`score_skills_v2.py`)

Step 2b held the rubric fixed and changed the model. What it found was not a disagreement between two models but an ambiguous question. Scoring v2 is the replacement: six questions per skill instead of two, answered by the same judgment model, and classified by threshold probabilities instead of averaged ratings. It is what the site shows by default. The specification is [docs/scoring-v2.md](docs/scoring-v2.md); this section is the summary and the results.

**Why a new rubric.** Four defects, each measured on output the first rubric had already produced:

1. **"Automation" meant two things at once.** The word covered AI software and industrial machinery, and a score reflected whichever the model had in mind. On the pilot sample, the first rubric's Gemini automation score lines up *better* with v2's machine-automation axis than with v2's AI-substitution axis: Spearman **+0.56** against **+0.42**, over the 742 skills both runs scored. The first rubric's "automation risk" was substantially an index of machine automation wearing an AI label. `operate cash register` scored 9 out of 10 on it; under v2 the same skill reads AI substitution 0.04 and machine automation 0.31. `drive vehicles` scored 7; under v2, 0.00 and 0.43.
2. **The two v2 axes are close to independent**, which is the check that the conflation is gone rather than moved. Spearman(AI substitution, machine automation) is **−0.04** on the pilot's own wording of the machine question and **−0.31** on the wording that shipped; over the full run of all 13,939 skills it is **−0.31**.
3. **Knowledge items were scored like activities.** ESCO mixes "manage musical staff" with "thermodynamics". Asking how automatable thermodynamics is describes what machines embody, not what work people stop doing.
4. **Averaging flattened the middle, and the cut-off then did the classifying.** Two ordinal ratings averaged over a job's skills leave almost every occupation near the centre of both scales, so a hard threshold at 6 decides the label. Under the first rubric 1,507 of the 3,039 occupations — 50% — sit within 0.5 of a cut-off.

#### The six questions

One request per skill to `jev-1.13.0`, carrying all six. Every question that mentions AI shares a preamble fixing what "the system" is: software that reads and writes text, code, tables, documents and images and can call other software, with "no body, no hands, no eyes in a room and no physical presence". The wording lives in [`aiisco/rubric_v2.py`](aiisco/rubric_v2.py) and is published to the site as `site/rubric_v2.json`, generated from that module so the page and the scorer cannot drift apart.

| id | primitive | What it asks | What it feeds |
|---|---|---|---|
| `digital_output` | `Noul` (yes/no) | Is what the worker hands over information — a document, a record, a drawing, an answer — rather than a physical change or an effect on a person in the room? | the gate on `SUB` |
| `ai_substitution` | `Score` (5 levels) | How much of the work can that system carry out itself, from "none of it" to "all of it, unsupervised"? | `SUB`, and the **AI substitution** display score |
| `mechanical` | `Score` (5 levels) | AI set aside: how much of the work does physical equipment do — production lines, CNC machines, robots, conveyors, process plant? | `MECH`, and the **Machine automation** display score |
| `complementarity` | `Score` (5 levels) | If the person keeps the work, how much better does it get, from "no difference" to "several times as much work"? | `COMP`, and the **AI assistance** display score |
| `mode` | `Choice` (5 options) | How the work is mainly carried out: on things, with people, directing others, through software, on paper in place | `why_insulated`: why an occupation's untouched skills stay human |
| `deployment` | `Choice` (4 options) | Is software that does this work in routine use, available to buy, demonstrated only, or not shown at all? | nothing. Descriptive only; no class may depend on it |

`ai_substitution` and `mechanical` have a knowledge variant, used for ESCO knowledge items, which asks about the work the knowledge is applied in rather than whether the model can recall the facts.

#### A worked example: `write meeting reports`

One real skill, straight out of `data/skill_scores_v2.json`. This is what the model returned:

| Question | The model's answer |
|---|---|
| `digital_output` | yes **0.98** |
| `ai_substitution` | none 0.00 · a minor part 0.00 · about half 0.11 · nearly all, supervised **0.83** · all, unsupervised 0.06 (confidence 0.85) |
| `mechanical` | none **0.99** · a small part 0.01 · about half 0.00 · nearly all 0.00 · all 0.00 (confidence 0.99) |
| `complementarity` | no difference 0.00 · a little 0.00 · part of the work 0.06 · most of the work **0.89** · a change in what they can take on 0.05 (confidence 0.91) |
| `mode` | through software **1.00** (confidence 1.00) |
| `deployment` | routine **0.53** · available 0.47 (confidence 0.38) |

The three derived quantities are threshold probabilities, never averages of the levels:

```
SUB  = P(ai_substitution >= 3) * P(digital_output = yes)
     = (0.83 + 0.06) * 0.98  = 0.89 * 0.98 = 0.8722     AI can take the work over
COMP = P(complementarity >= 3)  = 0.89 + 0.05 = 0.94    AI gives a clear gain across most of the work
MECH = P(mechanical >= 3)       = 0.00 + 0.00 = 0.00    machinery does the work
```

`SUB` is 0.87, which is at or above 0.5, so the first rule matches and the skill's class is **substituted** — "AI can take over" on the site. The three 1-10 display scores are the probability-weighted level put through `1.5 + 2.0 * position`:

```
AI substitution    position = 2(0.11) + 3(0.83) + 4(0.06) = 2.95  →  1.5 + 2.0(2.95) = 7.40
AI assistance      position = 2(0.06) + 3(0.89) + 4(0.05) = 2.99  →  1.5 + 2.0(2.99) = 7.48
Machine automation position = 1(0.01)                     = 0.01  →  1.5 + 2.0(0.01) = 1.52
```

Those three numbers exist so the set can be read next to the older ones. No class depends on them.

#### The four skill classes

Every skill gets exactly one class, first match wins:

| Class | Code | Rule | The site says |
|---|---|---|---|
| substituted | `S` | `SUB >= 0.5` | AI can take over |
| assisted | `A` | `COMP >= 0.5`, and not already substituted | AI assists |
| mechanised | `M` | `MECH >= 0.5`, and neither substituted nor assisted | Machines can do |
| insulated | `I` | none of the above | Stays human |

#### From skills to occupations

Link weights are the existing ones — essential 2, optional 1 — normalised within the occupation, with unscored skills left out. The occupation's skill weight then splits four ways, one share per class, and the four sum to 1. Two more quantities come out of the roll-up: `mu` and `sigma`, the weighted mean and standard deviation of `SUB`, which are published but which no rule reads; and `why_insulated`, the reason the insulated skills stay human, taken as the largest probability mass of `mode` over those skills (physical, people, or other). It reads the whole distribution rather than each skill's single most likely option, because with weights of 1 and 2 counting modal choices ties for about one occupation in twenty.

An occupation gets one type, first match wins. Every threshold is a round number fixed before anyone looked at the output:

| # | Type | Rule | Occupations | Share |
|---|---|---|---|---|
| 1 | **Automation-heavy** | `share_substituted >= 0.50` | 145 | 4.8% |
| 2 | **Transforming** | `share_substituted >= 0.30` and `share_assisted >= 0.20` | 201 | 6.6% |
| 3 | **Augmented** | `share_assisted >= 0.30` and `share_substituted < 0.30` | 944 | 31.1% |
| 4 | **Mechanisable** | `share_mechanised >= 0.30` and `share_substituted < 0.30` | 273 | 9.0% |
| 5 | **Insulated by physical work** | `share_insulated >= 0.40` and the reason is physical | 1,010 | 33.2% |
| 6 | **Insulated by work with people** | `share_insulated >= 0.40` and the reason is people | 383 | 12.6% |
| 7 | **Mixed** | everything else | 83 | 2.7% |

Counts and shares come from `site/stats_v2.json`, which `build_site_indexes.py --scorer v2` writes, and they move whenever the data is rebuilt — that file, not this table, is the source the pages read. Mixed is a residual, not a finding that AI will leave a job alone: its skills point in different directions and a single label would mislead.

What that looks like on real occupations, from `site/data_v2.json`:

| Occupation | AI can take over | AI assists | Machines can do | Stays human | Type |
|---|---|---|---|---|---|
| Accountant | 75% | 13% | 0% | 12% | Automation-heavy |
| Translator | 71% | 14% | 0% | 14% | Automation-heavy |
| Secondary school teacher | 21% | 40% | 0% | 39% | Augmented |
| Welder | 1% | 6% | 30% | 63% | Mechanisable |
| Hairdresser | 15% | 20% | 0% | 65% | Insulated by physical work |
| Specialist nurse | 4% | 28% | 0% | 68% | Insulated by work with people |

Shares are rounded to whole percent here and a row may not add to 100; the published values carry four decimals and sum to 1. Welder is the useful one to look at: its mechanised share is 0.3033, which clears rule 4 by three thousandths, and it is duly flagged as sitting near a cut-off. The site rounds with the largest-remainder method so a printed set of shares always adds to 100.

#### What came out

All 13,939 skills were scored, one request each, every answer from `jev-1.13.0`. Across the 13,475 skills that are linked to at least one occupation, and which are therefore the ones the site's numbers are computed over:

| Skill class | Skills | Share |
|---|---|---|
| substituted — AI can take over | 1,526 | 11.3% |
| assisted — AI assists | 2,595 | 19.3% |
| mechanised — Machines can do | 1,332 | 9.9% |
| insulated — Stays human | 8,022 | 59.5% |

The other 464 scored skills are in ESCO but linked to no occupation. Over all 13,939, the counts are 1,748 substituted, 2,626 assisted, 1,334 mechanised and 8,231 insulated.

**How many occupations sit near a line.** An occupation is near a cut-off when moving any one of its four shares by 0.05 would change its type. 909 of 3,039 occupations, **30%**, are. Under the first rubric the equivalent question — is the job within 0.5 of a threshold on either axis — catches 1,507, **50%**. The two are not the same measurement, because the two models cut on different quantities; what the comparison says is that replacing one averaged score and one hard threshold with four shares and a first-match-wins list leaves fewer jobs balanced on a boundary, not that any individual job is now settled.

#### Two choices made after the full run

The thresholds were fixed before the run, with two exceptions that the run's own distribution forced. Neither was tuned to make a particular job land anywhere.

1. **Assistance is read at level 3, not level 2.** `COMP` was first defined as `P(complementarity >= 2)`, "a clear gain on part of the work". On all 13,939 skills that makes 62% of skills assisted and 86% of occupations Augmented — a classification that says the same thing about almost everything. Read at level 3, "a clear gain across most of the work", the types spread. The price is that `COMP` at level 3 correlates strongly with `SUB`: **+0.91** over the full run. That matters less than it sounds, because the class rules only consult `COMP` for skills AI cannot take over. `P(complementarity >= 2)` is still stored beside it as `comp_part`, so the choice can be revisited with `--rederive` and no new requests.
2. **The first type rule lost a condition.** It read `share_substituted >= 0.50 and sigma <= 0.20`, borrowed from the ILO's mean-and-spread rule. But `SUB` is a threshold probability, close to 0 or close to 1 by construction, so its spread is largest exactly when the substituted share is between a half and four fifths. Translators (71% of skill weight substituted), accountants (75%) and data entry clerks (70%) all fell through to Mixed. `mu` and `sigma` are still published; they no longer gate a type.

A third decision was left alone rather than changed: the cut for "Machines can do" stayed at 0.5 like every other class, although the shipped wording of the machine question is conservative — robot welding reads about 0.33. Read that share as a floor.

#### How well the questions behave

Every wording was measured before the full run, on a pilot of 800 stratified ESCO skills plus named probes. The pilot ran in a separate repository and is not published here; its findings are summarised below and the questions it settled on are in `aiisco/rubric_v2.py`.

| Check | Result |
|---|---|
| Retranslation gate | Each level description fed back as the item to be scored must return as itself. 5/5 for every graded question, and 5/5 for both knowledge variants |
| Self-consistency, 3 identical runs of 60 skills | Ordinal Krippendorff alpha: `ai_substitution` 0.964, `mode` 0.964, `digital_output` 0.952, `complementarity` 0.940, `deployment` 0.936, `mechanical` 0.909. Index reliability is the minimum, not the mean: **0.909**, against Krippendorff's 0.800 bar |
| Independent transcription | A second agent wrote its own wording of the same questions and scored its own sample; on the 218 overlapping skills, modal agreement 94-99% |
| Questions dropped | Nine became six. `needs_software` (85% "yes" at confidence 0.27), `raises_novices` (+0.88 with `SUB`, so it restates it), `tacit_context` (confidence 0.24), `halves_the_time` (92% "no"; its rewrite reached 70% "yes" but at confidence 0.22, with 96% of answers under 0.5) |

The machine question was the one that had to be rewritten, and it was chosen by a bake-off rather than by taste. 25 high and 25 low real ESCO titles were labelled *before* any wording was scored; the rule was fixed in advance as highest AUC, gap as the tiebreak, no further wordings after seeing results:

| Wording | What it is | AUC | High-label mean | Low-label mean |
|---|---|---|---|---|
| `w0` | the pilot's original question | 0.9752 | 0.806 | 0.224 |
| `w1` | a rewrite that asked whether equipment does the work instead of the worker | 0.9448 | 0.226 | 0.025 |
| `w2` | `w0` plus a paragraph excluding software, apps and navigation | 0.9800 | 0.706 | 0.062 |
| **`w3`** | **`w2` plus a sentence that a human faculty is not mechanised because a device resembles it — shipped** | **0.9824** | 0.634 | 0.052 |

`w1` was rejected: it removed the false positives by removing machine work itself (arc welding 0.79 → 0.03, CNC tending 0.88 → 0.24, sorting waste 0.95 → 0.37). Tending a machine is what is left after mechanisation, and answering "no" for it defeats the question. `w3` beats `w2` by 0.0024, which a paired bootstrap of 2,000 draws puts at 95% CI [−0.0016, +0.0128] — a tie; `w3` was taken for lowering the worst surviving false positive (`have spatial awareness` 0.68 → 0.59), at the cost of a lower mean on the high-labelled probes. The shipped wording is more conservative than the pilot's across the board: on the 800-skill sample the mean machine score falls from 0.243 to 0.120 and the share above 0.5 from 17.4% to 8.2%. Two false positives survive it, `have spatial awareness` at 0.62 and `thermodynamics` at 0.63.

#### Why the link level was not scored

The original design asked the questions once per occupation-skill link, not once per skill, so that "perform risk analysis" could score differently for a surgeon and for an insurance clerk. That was piloted on 1,287 links and dropped, for two measured reasons:

- **ESCO's own flag already carries the weight.** Asking how central a skill is to an occupation separates essential from optional links at Cohen's d = **+1.31**. The free binary flag in the data gives most of that signal.
- **Context moves the answer, but not much.** On a panel built to make context matter — 45 high-reuse skills across up to 8 ISCO major groups each, 360 links — the median within-skill standard deviation of the context factor is **0.145**, against a run-to-run noise floor of about 0.02. Almost all of it is carried by one question, `mode`, which is also the most confident question in the set (mean confidence 0.84 at skill level).

So `mode` is asked once per skill and the link level, about 126,000 further requests, was never run. The cost of that choice is stated plainly: a skill carries the same three probabilities into every occupation that needs it, and the roll-up distinguishes occupations only by which skills they need and how they are weighted.

#### Reproducing it

```bash
uv sync
uv run python ingest_esco.py                             # ESCO CSVs → data/esco_skills.json
echo "TYPESAFE_API_KEY=your_key_here" >> .env            # or the macOS keychain item 'typesafe-api-key'

uv run python score_skills_v2.py                         # 13,939 requests, one per skill; resumes from the checkpoint
uv run python score_skills_v2.py --rederive              # recompute SUB, COMP, MECH and the classes from the stored
                                                         # answers: no requests, no key, no model

uv run python aggregate_scores.py --scorer v2            # site/data_v2.json
uv run python build_portfolio_data.py --scorer v2        # site/portfolio_data_v2.json and the per-unit files under site/jobs/
uv run python build_site_indexes.py --scorer v2          # the *_v2 index files, site/rubric_v2.json, site/skill_answers_v2/
                                                         # and the rationale shards under site/skill_notes/
```

Other flags, all shared with `score_skills_typesafe.py`: `--sample N` (with `--seed`), `--start`/`--end` for a slice, `--workers` and `--rps` for the pool, `--force` to ignore the checkpoint. The model is pinned in the script rather than exposed as a flag, because the `jev-latest` alias moves.

- **Output:** `data/skill_scores_v2.json` (committed), one entry per skill: every answer with its full probability distribution and the model's confidence, the three derived probabilities, the class, the three display scores and the model version
- **Resume:** checkpoints every 250 skills and on the way out; a re-run skips what is already scored
- **Failure guard:** the pool stops itself after twenty consecutive failures, while the checkpoint is still worth resuming from
- **Key:** `TYPESAFE_API_KEY` from the environment or `.env`, falling back to the macOS login keychain item `typesafe-api-key`. It never reaches a command line, a log or the output

### Step 3: Aggregate to occupations (`aggregate_scores.py`)

Computes weighted averages from skill-level to occupation-level scores:

- **Essential skills** weighted **2.0x** (core to the role)
- **Optional skills** weighted **1.0x** (supplementary)
- **Evolution Potential** = `(automation_risk × amplification_potential) / 10`

Assigns each occupation to a quadrant and outputs compact site data with the top 5 most-automated and top 5 most-amplified skills per occupation.

`--scorer v2` uses the same link weights and the same script, but the roll-up is the one in [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py): four shares of the skill weight, `mu` and `sigma`, `why_insulated`, one of seven types, and whether the occupation sits near a cut-off. It writes `site/data_v2.json` and leaves the published Gemini files alone. `--scorer typesafe` does the same with the second Step 2b score set.

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

Creates the compressed dataset the job page is served from:

**Occupation adjacency** (Jaccard similarity):
- Computes pairwise Jaccard similarity on essential skills using an inverted index for efficiency
- Minimum overlap threshold: 15% (`MIN_JACCARD_OVERLAP = 0.15`)
- Only keeps adjacent occupations with higher evolution potential
- Maximum 8 adjacent occupations per entry

**Gap skills:**
- For each adjacency, identifies skills in the target occupation but not the source
- Sorted by amplification potential (most valuable skills first)
- Maximum 5 gap skills per adjacency

**Compression:** Skills use 8-character MD5 hash IDs with collision resolution. Occupation keys are heavily abbreviated (e.g. `t`=title, `ar`=automation_risk, `se`=essential_skills).

**Sharding.** The whole dataset stays published as `site/portfolio_data<suffix>.json`, but no page downloads it. The same step also writes `site/jobs/units.json` (occupation slug → its 4-digit ISCO unit group, the same file whichever scorer built it) and one `site/jobs/<unit><suffix>.json` per unit group: the same record shapes as the full file, restricted to that group's occupations, plus every occupation they link to that lives in another group, plus the skills of both. A job page therefore fetches two small files instead of one large one.

### Step 6: Build the site indexes (`build_site_indexes.py`)

The pages should not download 13 MB to draw a search box. This step reads `site/data<suffix>.json` and `site/portfolio_data<suffix>.json` and writes the small files each page actually needs. `<suffix>` is empty for Gemini, `_v2` for the default set and `_typesafe` for the Step 2b set:

| File | What is in it | Loaded by |
|---|---|---|
| `search_index<suffix>.json` | Per occupation: title, slug, ISCO code, major group, the display scores, its quadrant or its type, ESCO's alternative labels, and under the shares scheme also the four shares and whether it is near a cut-off | Find a job, Browse sectors, Job tree, What we found |
| `groups<suffix>.json` | Every ISCO group at every level: job count, counts per quadrant or per type, score spread, how many of its jobs sit near a cut-off, the most and least exposed jobs, the skills that drive it, children and parent | Browse sectors, What we found |
| `stats<suffix>.json` | Every number the pages quote: totals, the scheme the file follows, quadrant or type counts and shares, skill-class counts, the near-a-cut-off share, the model, the build date | all pages |
| `skill_index<suffix>.json` | Per skill: title, the display scores, occupation counts, and under the shares scheme its class, the three probabilities, its mode and whether it is a skill or a knowledge item | Skill scores, Browse sectors |
| `skill_occupations<suffix>.json` | Which occupations need each skill, essential and optional | Skill scores, only once a skill is open |
| `skill_notes/<xx><suffix>.json` | The rationale per skill, in 256 shards keyed by the first two characters of the skill id | Skill scores, one shard per opened skill |
| `rubric_v2.json` | The six questions exactly as they were asked, the class rules and the display formula, generated from `aiisco/rubric_v2.py` | Skill scores |
| `skill_answers_v2/<xx>.json` | The model's answer distributions and confidences per skill, in 256 shards | Skill scores, one shard per opened skill |

The last three are written for the v2 scorer; the rubric and the answer shards have no counterpart under the older rubric, because a two-score run has no per-question answers to show. Alternative labels are why a search for "programmer" finds Software developer. No number on the site is typed into the prose: the pages compute them from these files as they load.

## The quadrant model

This is the first model, and what the site shows when the switch is set to Gemini. The default view uses the four shares and seven types of [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py) instead.

Every occupation lands in one of four quadrants based on its aggregate scores (threshold = 6):

| Quadrant | Auto Risk | Amp Potential | What happens | Avg time savings |
|---|---|---|---|---|
| **TRANSFORM** | High (≥6) | High (≥6) | Job evolves into something new and better | ~61% |
| **SHRINK** | High (≥6) | Low (<6) | Job contracts — automation without upside | ~58% |
| **EVOLVE** | Low (<6) | High (≥6) | Job grows — AI augments without replacing | ~43% |
| **STABLE** | Low (<6) | Low (<6) | Job stays roughly the same | ~28% |

Distribution across 3,039 occupations:
- **EVOLVE:** 51.3% (1,559 occupations) — the largest group
- **STABLE:** 26.0% (789)
- **TRANSFORM:** 16.7% (507)
- **SHRINK:** 6.1% (184)

## The frontend

Static pages in vanilla JS: ES modules, no framework, no build step. The site is organised around the questions a visitor arrives with, not around the taxonomy:

| The visitor asks | Page |
|---|---|
| "How will my profession be affected?" | [Find a job](https://jorisdevreede.github.io/AI-ISCO/) → the job page |
| "How will someone else's profession be affected?" | The same job page in neutral wording, with a link you can share |
| "Which groups of professions are affected, and how differently?" | [Browse sectors](https://jorisdevreede.github.io/AI-ISCO/groups.html), or the whole hierarchy at once in the [Job tree](https://jorisdevreede.github.io/AI-ISCO/tree.html) |
| "How will a skill evolve, and how was it scored?" | [Skill scores](https://jorisdevreede.github.io/AI-ISCO/skill.html) |
| "Where could I move next, and what should I learn?" | The job page: nearby jobs and skills to learn |
| "How sure is this?" | On every badge, and on [How sure is this?](https://jorisdevreede.github.io/AI-ISCO/method.html) |

The decisions behind this layout, and the audit findings that led to them, are in [docs/frontend-redesign-brief.md](docs/frontend-redesign-brief.md).

Deep links are hash-based, so GitHub Pages serves them unchanged:

```
job.html#software-developer                   my-job wording
job.html#software-developer&for=other         neutral wording, for someone else's job
job.html#optical-engineer&from=unit:2149      adds a way back to that group
groups.html#g=major:2&view=ranked|scatter|treemap|table
tree.html#job=software-developer              the tree opened at one job
skill.html#59b27e7b
```

#### Two score sets, two ways of describing a job

Three sets of files are published side by side, and [`site/scorer.js`](site/scorer.js) decides which one a page loads. Pages call `scorerFetch('data')` rather than `fetch('data.json')`, and the suffix is appended for them:

| Set | Files | In the switch |
|---|---|---|
| **TypeSafe** — jev on the v2 rubric, four shares | `*_v2.json` | Yes, and it is the default wherever those files are deployed |
| **Gemini** — the original rubric, two scores | no suffix | Yes, as the alternative |
| jev on the original rubric | `*_typesafe.json` | No. It exists only for the same-rubric comparison on the method page |

A **Scores from** switch in the navigation bar offers the first two, labelled with what changes — "TypeSafe · four shares" and "Gemini · two scores" — and remembers the choice across pages. Whichever set is on, a note under the navigation bar says what that means, including that the stories and skill explanations are the ones Gemini wrote for its own older scores.

The URL parameter behaves as follows:

| Parameter | What happens |
|---|---|
| `?scorer=v2` | the default set. Selecting the default removes the parameter from the URL again |
| `?scorer=gemini` | the older two-score set, carried across in-page links while it is set |
| `?scorer=typesafe` | an alias for the default. Links shared before scoring v2 existed meant "the jev scores", and that is where they land |
| anything else | ignored; the stored choice or the default is used |

`scorer.js` makes one `HEAD` request per set to see what is actually deployed, so a checkout with only the Gemini files behaves as a plain single-scorer site with no switch and no dead options.

Every page reads `stats.scheme` from the set it loaded — `shares` or `quadrants`, absent meaning quadrants — through `site/js/scheme.js`, and renders accordingly. Under `shares`, a badge, a mix bar or a filter that showed four quadrants shows seven types; a pair of scores becomes the four-share bar with the three display scores as secondary detail; and skill rows carry the skill's class. Under `quadrants` the pages are as they were.

### Find a job ([index.html](https://jorisdevreede.github.io/AI-ISCO/))

The first page is a search box. It matches titles and ESCO's alternative labels, so "programmer" finds Software developer. Below it: a few example jobs, the jobs you viewed recently (kept in your browser only), and a way into the sectors. It loads the 80 KB search index and nothing else.

### The job page ([job.html#software-developer](https://jorisdevreede.github.io/AI-ISCO/job.html#software-developer))

Everything about one occupation, and the one page every other surface links to:

- **How the job splits** — under the default set, the four shares as a bar, then the three display scores as detail; under Gemini, the two scores. Either way a badge explains which rule the job met and whether it sits near a cut-off
- **What to do next** before any detail: a job with a large substituted share never gets a verdict without a next step
- **Where the skills sit** — every skill plotted on two axes, with a table alternative. Under the quadrant scheme the cut-off lines at 6 are drawn; under the shares scheme they are not, because there they are not class boundaries
- **Why each essential skill scores as it does** — the model's rationale per skill
- **How the work could change** — the narrative, what AI takes on and what it amplifies, the rebalanced week
- **Where you could move next** and **skills you could learn** — nearby jobs by skill overlap, each labelled for what it is (a step up, a sideways move, or more exposed)
- `&for=other` switches the wording to neutral for someone else's job and keeps it out of your recently viewed list

The page loads `jobs/units.json` to find the job's ISCO unit group, then that group's `jobs/<unit><suffix>.json`, which carries the occupation, its skills and the neighbouring occupations it links to. It never downloads the full portfolio file. `portfolio.html#<slug>` links from before the redesign redirect here.

### Browse sectors ([groups.html](https://jorisdevreede.github.io/AI-ISCO/groups.html))

Any ISCO-08 group at any level (10 major groups → 43 sub-major → 130 minor → 436 unit groups): how its jobs split over the four quadrants or the seven types, then four views of the same jobs — a ranked dot plot, a scatter, a drill-down treemap and a sortable table — plus the most and least exposed jobs, the skills that drive the group, how many of its jobs sit near a cut-off, and a side-by-side comparison of two groups. Under the shares scheme the scatter plots the two shares the type rules actually cut on — AI can take over across, AI assists up — with the rule lines drawn lightly, rather than two correlated 1-10 scores. The treemap is keyboard-navigable and every chart has a text summary and a table alternative. `explorer.html` redirects here.

### Job tree ([tree.html](https://jorisdevreede.github.io/AI-ISCO/tree.html))

The whole ISCO-08 hierarchy as an expandable tree on the left, down to the 3,000+ individual jobs, and a detail pane on the right. Every group row shows its job count and its mix of quadrants or types; every job row shows where it landed and its scores. Pick a group to see how it splits; pick a job to see its badge, its scores or shares, how its skills split over the four quadrants or the four classes, and a sortable table of every skill (essential or optional, its scores, its own quadrant or class) with links to the skill pages and the full job page. A filter box narrows the tree by title or alternative label, and shows the matching jobs as a flat list above the hierarchy. The tree follows the WAI-ARIA tree pattern (arrow keys, Home/End, type-ahead), renders only the rows that are open, and fetches a selected job's unit-group file only once that job is picked. `tree.html#job=<slug>` and `tree.html#g=<level>:<code>` are deep links.

### Skill scores ([skill.html](https://jorisdevreede.github.io/AI-ISCO/skill.html))

The home of the scoring itself, renamed from "Look up a skill" and widened with scoring v2. Without a skill selected it offers the lookup, a filterable and sortable table of every scored skill — its class, the three display scores, how the work is carried out, skill or knowledge item, paged so the DOM never holds thousands of rows — and **How a skill is scored**: the six questions in their exact wording, read from `rubric_v2.json`, and how the answers become a class, in words rather than in rubric identifiers.

With a skill selected it adds **How the model answered**: per question, every level or option with the probability the model gave it, the chosen one marked, and the model's confidence, read from that skill's shard of `skill_answers_v2/`. Below that, the model's rationale (Gemini's, marked as such), the occupations that need the skill, essential and optional, and the skills it most often appears with. Under the quadrant scheme the table shows the two scores and there is no answers section, because that run has no per-question answers to show.

### What we found ([insights.html](https://jorisdevreede.github.io/AI-ISCO/insights.html))

Aggregated findings across all occupations, written up as a newspaper-style article. Every number in it is computed from the published data as the page loads.

### How sure is this? ([method.html](https://jorisdevreede.github.io/AI-ISCO/method.html))

What the scores are (model estimates, nobody measured a real job), how the active scheme draws its boundaries, how many occupations sit close enough to one that a small change would move them, and how far a second model agrees on the older rubric: the share of occupations in the same quadrant, the rank correlations and the quadrant-by-quadrant table, computed in the browser from the two published score files rather than typed into the prose.

### How the front end is put together

| Where | What |
|---|---|
| `site/js/*.js` | Shared modules: search ranking, quadrant and near-the-line rules, the scheme switch (`scheme.js`), share arithmetic and rounding (`shares.js`, `shares-bar.js`), URL state, group statistics, formatting, borrowed-rationale marking (`rationale.js`), the agreement computation the method page uses (`agreement.js`), data loading, the navigation and attribution footer, an accessible combobox, the badge, the treemap layout. [site/js/README.md](site/js/README.md) documents each one |
| `site/js/pages/*.js` | One thin DOM module per page, plus a pure `*-model.js` beside it that holds the logic and imports nothing from the DOM |
| `site/scorer.js` | Which of the published score sets a page loads, and the "Scores from" switch |
| `site/css/app.css` | Shared tokens and components; each page adds its own small stylesheet |

Accessibility is part of done: real links and buttons, an ARIA 1.2 combobox with a live region, visible focus, text contrast of at least 4.5:1, touch targets of at least 24 px, no horizontal scroll at 390 px, and a loading and an error state for every fetch. Every page that is not a redirect carries Open Graph and Twitter card tags and a meta description, sharing one card image at `site/card.png`.

## Setup

```bash
uv sync
```

Browsing the site needs no key. The pipeline steps that call a model read their key from `.env`:

```
OPENROUTER_API_KEY=your_key_here     # score_skills.py, generate_narratives.py
TYPESAFE_API_KEY=your_key_here       # score_skills_v2.py, score_skills_typesafe.py
```

Both TypeSafe scorers fall back to the macOS login keychain when the variable is unset, reading the generic password `typesafe-api-key` for the account `typesafe`:

```bash
security add-generic-password -a typesafe -s typesafe-api-key -w
```

The key is read into the process and never appears on a command line, in a log or in any output file.

| Way in | Command | When |
|---|---|---|
| Just look | `python3 -m http.server 8000 --directory site` | No install, no key |
| uv (recommended) | `uv sync` | Running any pipeline step |
| pip | `pip install httpx python-dotenv beautifulsoup4 typesafe-sdk` | No uv available; then run the scripts with `python` instead of `uv run python` |

### Full pipeline (from scratch)

```bash
uv run python ingest_esco.py              # Parse ESCO CSVs → JSON (~seconds)
uv run python score_skills.py             # LLM-score all 13,939 skills (~hours, resumable)
uv run python score_skills_v2.py          # The default score set: 13,939 typed requests; see Step 2c
uv run python score_skills_typesafe.py    # The Step 2b second scorer (resumable)
uv run python generate_narratives.py      # Generate evolution narratives (~hours, resumable)

# Then once per score set. Without --scorer the Gemini files are built.
uv run python aggregate_scores.py     --scorer v2   # Roll up to occupation level (~seconds)
uv run python build_portfolio_data.py --scorer v2   # Adjacency, gap skills, per-unit job files (~minutes)
uv run python build_site_indexes.py   --scorer v2   # Small per-page files under site/ (~seconds)
```

### Resume after interruption

`score_skills.py`, `score_skills_v2.py`, `score_skills_typesafe.py` and `generate_narratives.py` auto-resume from checkpoints — just re-run them. Use `--force` to regenerate already-processed items. `score_skills_v2.py --rederive` is the cheap path for a rule change: it recomputes every derived field from the answers already on disk without asking the model anything.

### Parallel narrative generation

```bash
# Run in separate terminals or tmux panes:
uv run python generate_narratives.py --start 0 --end 1000 --output data/shard_1.json
uv run python generate_narratives.py --start 1000 --end 2000 --output data/shard_2.json
uv run python generate_narratives.py --start 2000 --end 3039 --output data/shard_3.json

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
uv run pytest tests/e2e -q                             # the site's flows in a real browser (218 tests, two to three minutes)
uv run radon cc -s -n B aiisco                         # anything more complex than grade A

uvx ruff check aiisco tests aggregate_scores.py build_portfolio_data.py compare_skill_scores.py \
  generate_narratives.py ingest_esco.py merge_narrative_shards.py score_skills.py \
  score_skills_typesafe.py score_skills_v2.py build_site_indexes.py
```

| What | How it is kept |
|---|---|
| Python coverage | 732 tests, 100% line and branch coverage over the pipeline scripts and the `aiisco/` package; CI fails below 95% |
| Behaviour | Golden-output tests were written before any refactoring, and the real pipeline output was compared byte for byte before and after |
| Unit size and complexity | Following the SIG maintainability guidelines: short functions, low cyclomatic complexity, at most four parameters, no duplicated blocks |
| Scoring v2 arithmetic | `tests/test_v2.py`, `tests/test_rubric_v2.py`, `tests/test_score_skills_v2.py` and `tests/test_pipeline_v2.py` cover the thresholds, the class and type rules one boundary at a time, the answer serialisation and `--rederive` |
| The published files themselves | `tests/test_search_synonyms.py` asserts against the built indexes of both schemes that the labels people actually type reach the right job ("software engineer" → Software developer, "lorry driver" → a truck driver). `tests/test_site_head.py` asserts that every page that is not a redirect has a real `<title>`, a meta description and the Open Graph block |
| Front-end logic | Pure modules (search ranking, quadrant rules, the scheme switch, share arithmetic, URL state, page models, treemap layout, data loading) run under `node --test`: 462 tests, every line of those modules covered; no browser needed |
| The flows visitors take | `tests/e2e/` serves a copy of `site/` and drives it with Playwright: find a job by a synonym, every chip opens the job it names, group → job → back, neutral wording, the four group views and their table alternative, the skill pages, a keyboard-only path, the second score set and its borrowed rationales, the per-page load budget, no horizontal scroll at 390 px, no console errors. Set `AIISCO_E2E_CHANNEL=chrome` to use an installed Chrome |
| No network, no keys | Every model call is faked in the tests. The fixtures under `tests/fixtures/` are synthetic ([why](tests/fixtures/README.md)) |
| Static analysis | `sonar-project.properties` scopes a SonarQube scan to the project's own code (no data, no BLS pages, no karpathy/jobs files). The last local scan had no open issues. The browser-only modules are left out of Sonar's coverage figure because `tests/e2e` covers them in a real browser, which Sonar cannot see; four rule-and-file suppressions are each explained in that file |
| Known bugs | Behaviour that looks wrong but is published is pinned by a test marked `# BUG:` rather than silently changed, because fixing it changes the published numbers |

`.github/workflows/tests.yml` runs three jobs on every push and pull request:

| Job | What it runs |
|---|---|
| `python` | `uv run pytest --ignore=tests/e2e --cov --cov-branch --cov-fail-under=95`, then `ruff` over the pipeline scripts and `aiisco/` |
| `javascript` | `npm test` on Node 22 |
| `e2e` | Chromium via `playwright install --with-deps`, then `uv run pytest tests/e2e -q` against a served copy of `site/` |

The files copied from karpathy/jobs are outside the scope of the tests.

The shared Python code lives in the `aiisco/` package: `esco.py` (reading ESCO), `rollup.py` (weights, the threshold, quadrants), `rubric_v2.py` (the six v2 questions as data) and `v2.py` (the v2 arithmetic: threshold probabilities, skill classes, the roll-up and the seven type rules), `systemone.py` (key loading, rate limiting and the worker pool both TypeSafe scorers share), `portfolio.py` (adjacency and gap skills), `site_indexes.py`, `stats.py`, `openrouter.py` and `checkpoint.py` (model calls, retry, resume) and `jsonio.py`. The scripts at the top level are thin entry points over it.

## Key files

| File | Purpose |
|------|---------|
| `ingest_esco.py` | Parse ESCO v1.2.1 CSVs into structured JSON |
| `score_skills.py` | Dual-axis LLM scoring of all 13,939 skills, the original rubric |
| `score_skills_typesafe.py` | Step 2b: the same rubric scored as typed judgments with TypeSafe |
| `score_skills_v2.py` | Step 2c: the six-question v2 rubric, the default score set. `--rederive` recomputes the derived fields with no requests |
| `compare_skill_scores.py` | Compares the Step 2b scores with the published Gemini scores: per skill, per occupation and per quadrant |
| `aggregate_scores.py` | Weighted skill→occupation roll-up: quadrants, or the four shares and seven types under `--scorer v2` |
| `generate_narratives.py` | LLM-generated AI evolution narratives for each occupation |
| `merge_narrative_shards.py` | Consolidate parallel narrative shards |
| `build_portfolio_data.py` | Jaccard adjacency, gap skills, the compressed portfolio dataset and the per-unit files under `site/jobs/` |
| `build_site_indexes.py` | The small per-page files: search, groups, stats, skills, the rationale shards, and for v2 the rubric and the answer shards |
| `aiisco/rubric_v2.py` | The six v2 questions as data: wording, levels, options, knowledge variants |
| `aiisco/v2.py` | The v2 arithmetic: SUB, COMP, MECH, the four skill classes, the roll-up, the seven type rules, near-a-cut-off |
| `aiisco/systemone.py` | Key loading, rate limiting, the worker pool and the failure guard both TypeSafe scorers use |
| `aiisco/` | Shared, tested Python: ESCO reading, roll-up rules, adjacency, index building, model calls, checkpoints |
| `site/scorer.js` | Picks which score files the pages load; renders the "Scores from" switch when more than one set is deployed |
| `site/js/scheme.js`, `site/js/shares.js` | The `quadrants`/`shares` split, and the share arithmetic and rounding the pages draw from |
| `site/index.html` | Find a job: the search-first landing page |
| `site/job.html` | The one job page (`portfolio.html` redirects to it) |
| `site/groups.html` | Browse sectors: the quadrant or type split, ranked, scatter, treemap, table (`explorer.html` redirects to it) |
| `site/tree.html` | Job tree: the ISCO hierarchy down to each job, with a skills pane |
| `site/skill.html`, `site/method.html`, `site/insights.html` | Skill scores, the method page, the findings article |
| `site/js/`, `site/css/` | Shared ES modules and stylesheet, one page module per page |
| `tests/` | pytest for the Python, `node --test` for the JavaScript, Playwright for the flows, synthetic fixtures |
| `docs/scoring-v2.md` | The v2 specification: the questions, the derived quantities, the rules, the data contract |
| `docs/frontend-redesign-brief.md` | The questions the site answers and the decisions behind its layout |
| `data/esco/` | Raw ESCO v1.2.1 CSV files |
| `data/skill_scores.json` | All 13,939 skills on both original axes (generated, not committed) |
| `data/skill_scores_v2.json` | All 13,939 skills under the v2 rubric: every answer distribution, the derived probabilities, the class (committed) |
| `data/skill_scores_typesafe.json` | All 13,939 skills, the original rubric scored by jev (committed) |
| `data/occupation_narratives.json` | 3,039 occupation evolution narratives |
| `site/rubric_v2.json`, `site/skill_answers_v2/` | The questions as asked, and the model's answers per skill, both read by the Skill scores page |
| `site/jobs/`, `site/skill_notes/` | Per-unit-group job files and per-shard rationales, so no page downloads the whole dataset |

## Stack

- **Frontend:** Pure vanilla HTML/CSS/JS as ES modules — no framework, no build step
- **Visualization:** Canvas treemap, SVG scatter and dot plots, each with a table alternative
- **Tests:** pytest with coverage, `node --test`, ruff, radon; GitHub Actions
- **Backend:** Python 3.10+ with [uv](https://github.com/astral-sh/uv)
- **LLM API:** OpenRouter (Gemini Flash via `google/gemini-3-flash-preview`) for the original scores and the narratives
- **Judgment API:** TypeSafe System One, model `jev-1.13.0`, pinned in the script because the `jev-latest` alias moves. It produces the default score set
- **Hosting:** GitHub Pages via Actions
- **Dependencies:** httpx, python-dotenv, beautifulsoup4, typesafe-sdk

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: data/skill_scores.json not found.` from `aggregate_scores.py` or `build_portfolio_data.py` | The per-skill scores are a generated file and are not committed | Run `score_skills.py` first |
| `FileNotFoundError: … data/esco_skills.json` | The ingested ESCO JSON is generated, not committed | Run `uv run python ingest_esco.py` |
| Every batch prints `Parse error: 'OPENROUTER_API_KEY', retrying…` | `.env` is missing the key; the lookup error is caught by the parse-retry handler, so it looks like a bad model response | Add `OPENROUTER_API_KEY` to `.env` |
| `No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item 'typesafe-api-key'.` | `score_skills_v2.py` or `score_skills_typesafe.py` found neither | Add `TYPESAFE_API_KEY` to `.env`, or add the keychain item |
| `site/data.json` shows up as modified in `git status` after a local experiment | `aggregate_scores.py` copies its result straight into `site/data.json`, the file the live site serves | Check `git diff --stat site/` before committing; only commit it when you mean to publish new scores. `--scorer v2` and `--scorer typesafe` write to separate suffixed files and leave the Gemini ones alone |
| Rate-limit errors from TypeSafe | Too many requests a minute | Lower `--rps`; the SDK already retries with backoff |
| `Stopped early: 20 requests in a row failed. Re-run to resume.` | The pool's failure guard fired, rather than burning through the remaining skills against a broken endpoint | Fix the cause, then re-run; the checkpoint is intact and the run resumes |
| The site shows two 1-10 scores and four quadrants, with no "Scores from" switch | `scorer.js` found no `data_v2.json`, so it fell back to the Gemini set | Build the v2 set, or pull the committed `site/*_v2.json` files |
| `compare_skill_scores.py` matches fewer skills than were scored | It can only compare skills that appear in `site/portfolio_data.json` | Expected; the rest are skipped |

## Limitations

- **The scores are model judgments, not measurements.** Nobody measured a real job. There is no human-labelled ground truth in this repository, so nothing here can be scored for accuracy — only for consistency, which is what [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py) reports.
- **The words on a job page were written for different numbers.** The jev model returns judgments, not text, so every narrative and every per-skill rationale you read under the default score set is the one Gemini wrote for its own older, two-axis scores. They are borrowed and marked with an "i" saying who wrote them and for which scores, and they can contradict the class beside them: a skill Gemini called low-risk may sit under "AI can take over". Third-person and re-generated narratives are a later increment.
- **30% of occupations sit near a cut-off.** 909 of 3,039 would change type if any one of their four shares moved by 0.05. Under the older rubric, 1,507 sit within 0.5 of a score threshold. A type is a region, not a verdict.
- **No context was scored.** The questions were asked once per skill, not once per occupation-skill link, so "perform risk analysis" carries the same probabilities into a surgeon's job and an insurance clerk's. The pilot measured that context is real but small (median within-skill standard deviation 0.145 on the panel built to make it matter) and the link level was not run.
- **"AI assists" and "AI can take over" are not independent.** Read at level 3, `COMP` correlates with `SUB` at +0.91 over the full run. The class rules only consult `COMP` for skills AI cannot take over, which is what keeps the two classes distinguishable, but the two underlying probabilities move together.
- **"Machines can do" is a floor.** The shipped wording of the machine question is conservative by choice — robot welding reads about 0.33, below the 0.5 class cut — so the mechanised share understates how much of a job equipment already does.
- **The older rubric is ambiguous about machines.** Under the Gemini set, "automation" can mean AI software or AI plus industrial machinery, and the published scores are not consistent about it ("sift powder" 9, "polish stone by hand" 2). That is the defect scoring v2 exists to fix; it is not fixed in the Gemini files, which are published as they were.
- **Quadrants are a hard cut at 6.** In the Gemini view an occupation at 5.9 and one at 6.1 get different labels.
- **Narratives are LLM-generated**, one for each of the 3,039 occupations. Time savings, timelines and the rebalanced week are estimates, not forecasts.
- **ESCO's occupations CSV has four duplicate rows.** The ingest drops them; anything comparing this repository's counts with an older build of it, or with a raw ESCO row count, will be four apart.
- **English labels only.** The pipeline reads the `_en` ESCO files.
- **The per-skill Gemini scores are not committed.** They survive only in compressed form inside `site/portfolio_data.json`, so re-aggregating from scratch means re-scoring. The two jev score files are committed in full.
- **The jev output is not under this repository's CC BY grant.** `data/skill_scores_v2.json`, `data/skill_scores_typesafe.json`, the `site/*_v2.json` and `site/*_typesafe.json` files and `site/skill_answers_v2/` are published so the figures can be checked; reuse them only as far as TypeSafe's own terms allow. See [Licensing and attribution](#licensing-and-attribution).

## FAQ

**Do I need an API key?**
Not to browse. The scored data is committed, so `python3 -m http.server 8000 --directory site` is enough. Keys are only for re-running the model steps.

**Why score skills instead of occupations?**
An occupation-level number cannot tell you which part of the job changes. Scoring the 13,939 skills and rolling them up (essential skills weighted 2x) is what lets a job be split into four shares rather than given one exposure number, and it is what makes the nearby-jobs and gap-skills features possible.

**Which scorer produced the numbers on the site?**
TypeSafe's jev model on the v2 rubric, via `score_skills_v2.py`, by default ([Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py)). The "Scores from" switch shows the original Gemini two-axis scores instead.

**What does "Mixed" mean?**
That the job's skills point in different directions and no single label fits: it met none of the six rules above it. It is a residual, not a finding that AI will leave the job alone. 83 of 3,039 occupations are Mixed.

**Why do the two score sets disagree about my job?**
Because they answer different questions. The Gemini set asks how automatable a skill is on one axis that covers AI and machinery together; the v2 set separates those into two questions and adds a digital-output gate. A job can be TRANSFORM under one and Insulated by physical work under the other without either being a mistake — see the defects listed in [Step 2c](#step-2c-a-new-rubric-and-the-default-score-set-score_skills_v2py).

**How much do the two models agree on the *same* rubric?**
That is the Step 2b measurement, and it is the cleaner uncertainty signal because only the model changes. They rank skills and occupations much alike (rank correlation 0.82 to 0.86 per skill, 0.93 to 0.94 per occupation) but use the scale differently, so only 58% of occupations land in the same quadrant. The table is in [Step 2b](#step-2b-experiment-the-same-rubric-as-typed-judgments-score_skills_typesafepy).

**How is Evolution Potential calculated?**
`(automation_risk × amplification_potential) / 10`, on the occupation-level weighted averages. It belongs to the quadrant scheme; the shares scheme does not use it.

## Licensing and attribution

Different parts of this repository carry different terms. [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) has the full list; in short:

| What | Terms |
|---|---|
| Code written for this project (the ESCO pipeline scripts, the `aiisco/` package including the v2 question wording, every page and module under `site/`, the tests) | [MIT](LICENSE) |
| What Gemini generated for this project: the automation-risk, amplification-potential and evolution-potential scores, the quadrant assignments, the per-skill rationales and the occupation narratives | [CC BY 4.0](LICENSE-DATA) |
| What TypeSafe's model returned: `data/skill_scores_v2.json`, `data/skill_scores_typesafe.json`, the `site/*_v2.json` and `site/*_typesafe.json` files and `site/skill_answers_v2/` | Published here for reading and for checking the figures, but **not** under the CC BY grant — TypeSafe's own terms apply. The question wording in `site/rubric_v2.json` and the borrowed Gemini rationales inside those files are this project's own and stay MIT and CC BY respectively |
| ESCO classification (`data/esco/`, titles and descriptions in `site/`) | European Commission reuse terms, not covered by the licences above |
| ISCO-08 group structure, titles and definitions | © 2012 International Labour Organization, not covered |
| BLS Occupational Outlook Handbook pages and the ISCO-SOC crosswalk | Public domain, source: U.S. Bureau of Labor Statistics |
| Files copied from karpathy/jobs (`score.py` and seven other scripts, `prompt.md`, `scores.json`, the BLS scrape) | No licence published upstream, so none granted here |

This publication uses the ESCO classification of the European Commission. The data in `site/` and `data/occupation_narratives.json` is a modified and adapted version of ESCO v1.2.1: the scores, skill classes, quadrants, occupation types, rationales and narratives are AI-generated additions made by this project and are not part of ESCO. This project is not affiliated with or endorsed by TypeSafe AI, Inc.; "TypeSafe" and "jev" are their names. The European Commission, the International Labour Organization, the U.S. Bureau of Labor Statistics, Google and TypeSafe do not endorse this project.
