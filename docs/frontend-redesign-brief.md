# Frontend redesign brief

Decisions for rebuilding the AI-ISCO site around the questions visitors arrive with. The evidence behind them is a UX audit (Nielsen heuristics, WCAG 2.2 AA, Growth.Design principles) run against the live pages on 2026-09-18.

## The questions the app answers

| # | In the visitor's words | Who asks | Where it is answered |
|---|---|---|---|
| 1 | "How will my profession be affected?" | Anyone, after a headline | Landing search → job page |
| 2 | "How will someone else's profession be affected?" | Manager, HR, recruiter, counsellor, parent, journalist | Same search → job page in neutral wording, shareable link |
| 3 | "Which groups of professions are affected, and how differently?" | Manager, works council, policy, staffing | Group overview |
| 4 | "How will a skill evolve?" | Anyone with a specialism, L&D | Skill lookup |
| 5 | "Where could I move next, and what should I learn?" | Mid-career, after question 1 | Job page: paths and gap skills |
| 6 | "How do two professions compare?" | Someone choosing between paths | Compare (later increment) |
| 7 | "How sure is this?" | Anyone the result worries | In context on every badge, plus a method page |

## What was wrong

- The first page was the treemap. It answers a taxonomy question, has no search box, and a job tile was a dead end.
- Five of the eight "popular" chips on the job page opened a different job than the one named, because a chip ran a prefix search and took the first hit.
- Search matched titles only, so "programmer" never reached Software Developer. ESCO's alternative labels were dropped at ingest.
- `data.json` and `portfolio_data.json` were built from different scoring runs: the treemap and the job page disagreed on the quadrant of 788 occupations.
- The Insights prose hard-codes percentages that the live data no longer matches.
- "Evolution Potential" is automation × amplification, so it rewards automation risk. A nurse the app labels EVOLVE ("enhanced") also got a red "At Risk" banner.
- The scatter plot drew its quadrant lines at 5.5 while the model cuts at 6.
- Nothing in the data was reachable by keyboard, canvases had no text alternative, and every page overflowed at 390 px.
- The job page downloads and parses 13 MB before showing a word, with no loading or error state.

## Information architecture

| Page | Purpose | Replaces |
|---|---|---|
| `index.html` — **Find a job** | Search-first landing: one combobox, verified chips, recently viewed, "browse a sector". Loads only `search_index.json`. | the treemap as first page |
| `job.html#<slug>` — the one canonical job page | Everything about one occupation. Every surface links here with a real `<a href>`. | `portfolio.html` (kept as a redirect for old links) |
| `groups.html#g=<level>:<key>` — **Browse sectors** | Group overview at any ISCO level: quadrant mix, ranked list, scatter, treemap, table. | `index.html` treemap and `explorer.html` |
| `skill.html#<skill_id>` — **Look up a skill** | Both scores, the rationale, which occupations need it. | new |
| `insights.html` — **What we found** | The article, with every number computed from `stats.json`. | same page, no prose constants |
| `method.html` — **How sure is this?** | Plain-language method and limits. | new |

Navigation, same on every page: `Find a job` · `Browse sectors` · `Look up a skill` · `What we found` · `How sure is this?`

Deep links are hash-based so GitHub Pages serves them unchanged:

```
job.html#software-developer                   my-job wording
job.html#software-developer&for=other         neutral wording, for someone else's job
job.html#optical-engineer&from=unit:2149      adds "← Back to <group> (<n> jobs)"
groups.html#g=major:2&view=ranked|scatter|treemap|table
skill.html#59b27e7b
```

`?scorer=typesafe` keeps working everywhere and is carried across links when present (see `site/scorer.js`).

In-page navigation uses `history.pushState`, never `replaceState`, so the browser's Back button walks the real trail. A job page opened cold shows `← All jobs in <unit group>` derived from the occupation itself.

## Decisions on the audit's open questions

1. **Canonical scores: the Gemini run.** It is the default everywhere. A second score set (same rubric, TypeSafe's jev model) sits behind the "Scores from" switch; it was local-only until its publication was cleared on 2026-09-18.
2. **Keep the four quadrant names, hedge them.** A badge is a button that explains why the job landed there and how close it sits to a cut-off. A job within 0.5 of a cut-off on either axis is marked "near the line".
3. **Demote "Evolution Potential".** It stays in the data. The UI calls it "AI exposure", never colours a first view by it, and never derives a health verdict from it. The "At Risk" banner goes; a job page never delivers a bad verdict without a next step.
4. **"Someone else's profession" without new model calls.** `&for=other` switches headings, labels and the advice heading to neutral wording, stops writing to recently viewed, and adds a line saying the story below is addressed to the job holder. Third-person narratives need a model pass and are a later increment.
5. **One source of truth.** `data.json` is regenerated from the same per-skill scores as `portfolio_data.json`. All new index files are built in one run from those two files.

## Uncertainty copy

Everything said about certainty must be computable from the published data, never typed into the prose. When this brief was written the second scoring run was private, so the pages quote only facts about the default score set. Now that both sets are published, the agreement between them is the better uncertainty evidence: the method page computes it from the two search indexes as it loads (`site/js/agreement.js`), and the README's Step 2b has the per-skill detail.

What can be said, because it is true of the published data alone:
- the scores are model estimates; nobody measured a real job; there is no ground truth here
- the quadrants are a hard cut at 6, so a job at 5.9 and one at 6.1 get different labels
- `stats.json` carries the share of occupations within 0.5 of a cut-off, and the pages quote it

Tone: never "at risk" as a standing label for someone's job; say "more exposed to automation". No urgency, countdowns or scarcity. High-automation jobs always show "What to do next".

## Data files (built by `build_site_indexes.py`, all under `site/`)

| File | Shape | Budget |
|---|---|---|
| `search_index.json` | `[{"t","s","c","mg","a","m","q","alt":[...]}]` — title, slug, ISCO code, major group, scores, quadrant, ESCO alternative labels | ≤ 80 KB gzipped |
| `groups.json` | keyed `"<level>:<code>"` → label, level, n, quadrant counts, automation and amplification mean/p10/p50/p90, top and bottom slugs, driving skills, children, parent | ≤ 150 KB gzipped |
| `stats.json` | every number the pages quote: totals, quadrant counts and shares, near-the-line share, build date | tiny |
| `skill_index.json` | `[{"id","t","a","m","ne","no"}]` — essential and optional occupation counts | ≤ 120 KB gzipped |
| `skill_occupations.json` | `{"<skill_id>": {"e": [slug...], "o": [slug...]}}`, loaded only by `skill.html` | lazy |

Sharding `portfolio_data.json` into one file per job is a later increment; until then `job.html` shows a loading state and an error state.

## Front-end code

- No framework and no build step. ES modules under `site/js/`, one shared stylesheet `site/css/app.css`, pages stay plain HTML.
- Logic lives in pure modules (search ranking, quadrant and near-the-line rules, URL state, group statistics, formatting) that import nothing from the DOM, so `node --test` can run them. DOM wiring stays thin.
- Shared chrome (navigation, attribution footer, the scorer switch hook) is rendered by one module, not pasted into each page.
- Accessibility is part of done: real links and buttons, a combobox with `role`/`aria-expanded`/`aria-activedescendant` and a polite live region, a table alternative beside every canvas, visible focus, text contrast ≥ 4.5:1, no horizontal scroll at 390 px, touch targets ≥ 24 px.
- Every fetch has a loading state (after 200 ms) and an error state with a way forward.

## Tests

- `npm test` (`node --test tests/js/*.test.js`) for the pure modules.
- `pytest tests/e2e/` with Playwright for the flows: find a job by a synonym and land on the right page; every chip opens the job it names; treemap → group → job → back to the group; neutral-wording link round-trips; group overview views and the table alternative; skill lookup → an occupation that needs it; keyboard-only path from landing to a job page; no horizontal scroll at 390 px; no console errors.
- `pytest` for `build_site_indexes.py` against small synthetic fixtures.

## Build order

1. Index build script and the shared JS/CSS foundation.
2. In parallel: landing, job page, group overview, skill lookup with method page and the Insights number fix.
3. End-to-end suite, CI workflow, README.
4. Later: per-job data shards, compare page, third-person narratives.
