# AI-ISCO: Job Evolution Explorer

**Live:** [jorisdevreede.github.io/AI-ISCO](https://jorisdevreede.github.io/AI-ISCO/)

A deep analysis of how AI reshapes 3,000+ occupations — not by guessing at the job level, but by scoring every individual skill on two dimensions:

- **Automation Risk (1–10):** How likely is AI to replace this skill entirely?
- **Amplification Potential (1–10):** How much can AI supercharge a human doing this skill?

Jobs where both scores are high don't just disappear — they **transform** into something better, more demanded, and higher paid. That's the core insight.

Inspired by [karpathy/jobs](https://github.com/karpathy/jobs). Built on [ESCO](https://esco.ec.europa.eu/) by the European Commission.

## How it works

Built on [ESCO v1.2.1](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations), which maps **13,939 skills** to **3,043 occupations** across the full [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/isco08/) hierarchy.

### The pipeline

```
ESCO CSVs → ingest_esco.py → esco_occupations.json + esco_skills.json
                                        ↓
                              score_skills.py (LLM scores each skill on 2 axes)
                                        ↓
                              skill_scores.json
                                        ↓
                    aggregate_scores.py (weighted avg per occupation)
                                        ↓
                    occupation_scores.json + site/data.json
                                        ↓
                    build_portfolio_data.py (adjacency, gap skills)
                                        ↓
                    site/portfolio_data.json
```

Each skill is scored by an LLM (Gemini Flash via OpenRouter) in batches of 15, with incremental checkpointing and resume support. Essential skills are weighted 2x, optional skills 1x when aggregating to occupation level.

### The quadrant model

Every occupation lands in one of four quadrants based on its aggregate scores:

| Quadrant | Auto Risk | Amp Potential | What happens |
|---|---|---|---|
| **TRANSFORM** | High (≥6) | High (≥6) | Job evolves into something new and better |
| **SHRINK** | High (≥6) | Low (<6) | Job contracts — automation without upside |
| **EVOLVE** | Low (<6) | High (≥6) | Job grows — AI augments without replacing |
| **STABLE** | Low (<6) | Low (<6) | Job stays roughly the same |

**Evolution Potential** = (automation_risk × amplification_potential) / 10 — rewards jobs high on *both* dimensions.

## The frontend

### Job Explorer ([index.html](https://jorisdevreede.github.io/AI-ISCO/))
Drill-down treemap of all 3,000+ occupations through the ISCO hierarchy (10 major groups → 43 sub-major → 130 minor → 436 unit groups → individual jobs). Color by evolution potential, automation risk, amplification potential, or quadrant.

### Skill Portfolio Analyzer ([portfolio.html](https://jorisdevreede.github.io/AI-ISCO/portfolio.html))
Treats your career like an investment portfolio. Search for any occupation to see:
- **2D skill scatter plot** — every skill plotted on automation risk vs amplification potential
- **Portfolio health score** — strong, mixed, or at risk
- **Depreciating skills** — high automation risk, losing value
- **Appreciating skills** — high amplification potential, gaining value
- **Rebalancing recommendations** — skills from adjacent TRANSFORM/EVOLVE careers you should learn
- **Evolution paths** — adjacent occupations with higher evolution potential and shared skill overlap

## Resuming tomorrow

The skill scoring runs in a tmux session and takes a while (~14K skills). Here's how to pick up:

### 1. Check if scoring is still running

```bash
# Attach to see live progress (Ctrl+B, D to detach without stopping)
/usr/bin/tmux attach -t scoring

# Or just check the count
cat data/skill_scores.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)}/13939 skills scored')"
```

### 2. Resume scoring (if incomplete or session died)

```bash
# It auto-skips already-scored skills, so just re-run
/usr/bin/tmux new-session -d -s scoring \
  'cd /data/projects/AI-ISCO && .venv/bin/python score_skills.py 2>&1 | tee -a scoring.log'
```

### 3. Rebuild everything after scoring completes

```bash
uv run python aggregate_scores.py        # occupation-level scores → site/data.json
uv run python build_portfolio_data.py     # portfolio data → site/portfolio_data.json

git add site/data.json site/portfolio_data.json
git commit -m "Update site data with full skill scores"
git push origin master                    # auto-deploys via GitHub Actions
```

## Setup (from scratch)

```bash
uv sync
```

Requires an OpenRouter API key in `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

Full pipeline:
```bash
uv run python ingest_esco.py          # parse ESCO CSVs → JSON
uv run python score_skills.py         # LLM-score all skills (takes hours)
uv run python aggregate_scores.py     # aggregate to occupation level
uv run python build_portfolio_data.py # build portfolio adjacency data
cd site && python -m http.server 8000 # serve locally
```

## Key files

| File | Description |
|------|-------------|
| `ingest_esco.py` | Parses ESCO v1.2.1 CSVs into structured JSON |
| `score_skills.py` | Dual-axis LLM scoring of all 13,939 skills |
| `aggregate_scores.py` | Weighted skill→occupation aggregation + site data |
| `build_portfolio_data.py` | Occupation adjacency via Jaccard similarity on shared skills |
| `site/index.html` | Drill-down treemap explorer |
| `site/portfolio.html` | Skill Portfolio Analyzer |
| `data/esco/` | Raw ESCO v1.2.1 CSV files |

## Stack

- Pure vanilla HTML/CSS/JS — no framework, no build step
- Canvas-based treemap and scatter plot visualizations
- Python pipeline with incremental checkpointing
- OpenRouter API (Gemini Flash) for skill scoring
- GitHub Pages via Actions
