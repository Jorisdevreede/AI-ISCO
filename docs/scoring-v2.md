# Scoring v2: the model and its data contract

The first model asked one question per axis ("how likely is AI to automate this skill", "how much can AI amplify it") and cut two averaged scores at 6. Measured on its own output, that design had four defects: the automation axis mixed AI software with industrial machinery, knowledge items were scored like activities, averaging a job's skills flattened almost everything into the middle, and the cut-off then did most of the classifying. The evidence and the literature behind the redesign are in the research notes; this file is the specification the code follows.

## What is asked, per skill

One request per ESCO skill to a judgment model that returns a probability per answer level (TypeSafe System One, model version pinned in the script). Every question that mentions AI shares one preamble that fixes what "the system" is: software that reads and writes text, code, tables, documents and images and can call other software, with "no body, no hands, no eyes in a room and no physical presence".

| id | kind | asks |
|---|---|---|
| `digital_output` | yes/no | Is the result of this work something that exists as data or a document? |
| `ai_substitution` | 5 levels | How much of the work can that system carry out itself, from "none of it" to "all of it, unsupervised"? Knowledge items get a variant that asks about the work the knowledge is applied in. |
| `mechanical` | 5 levels | AI set aside: how much of the work does physical equipment do (production lines, CNC machines, robots, conveyors, process plant)? Knowledge items get the same kind of variant. |
| `complementarity` | 5 levels | If the person keeps the work, how much better does it get with that system, from "no difference" to "several times as much work"? |
| `mode` | choice | How the skill is mainly exercised: on things, with people, directing others, through software, on paper in place. |
| `deployment` | choice | Whether such systems are in routine use, available, demonstrated or not shown. Descriptive only; never an input to a class. |

The exact wording lives in `aiisco/rubric_v2.py`. Each graded question passed a retranslation check (the model recovers the intended order of its own level descriptions) and a three-run self-consistency check before the full run.

## Derived per skill

Threshold probabilities, not averages of ordinal levels:

```
SUB   = P(ai_substitution >= 3) * P(digital_output = yes)     AI can take the work over
COMP  = P(complementarity >= 3)                                 AI gives a clear gain across most of the work
MECH  = P(mechanical >= 3)                                     machinery does the work
```

The looser reading of assistance, `COMP_PART = P(complementarity >= 2)` ("a clear gain on part of the work"), is stored next to it and reported by the pipeline, so the choice can be revisited without re-scoring: `uv run python score_skills_v2.py --rederive` recomputes every derived field from the stored answers.

A skill gets exactly one class, first match wins:

| class | code | rule | the site says |
|---|---|---|---|
| substituted | `S` | `SUB >= 0.5` | AI can take over |
| assisted | `A` | `COMP >= 0.5` | AI assists |
| mechanised | `M` | `MECH >= 0.5` | Machines can do |
| insulated | `I` | none of the above | Stays human |

For display next to the older score sets, each graded answer is also put on the 1-10 scale as `1.5 + 2.0 * position`, where `position` is the probability-weighted level (0-4): `automation_risk` from `ai_substitution`, `amplification_potential` from `complementarity`, `mechanical_automation` from `mechanical`. These are display numbers; no class depends on them.

## Rolled up per occupation

Link weights `w` are the existing ones (essential 2, optional 1), normalised within the occupation; skills without scores are left out.

```
share_substituted, share_assisted, share_mechanised, share_insulated   sum of w per skill class; they sum to 1
mu, sigma                                                              weighted mean and standard deviation of SUB
why_insulated                                                          among insulated skills, the largest probability mass of `mode`:
                                                                       physical = sum of w * P(on_things)
                                                                       people   = sum of w * (P(with_people) + P(directing_others))
                                                                       other    = sum of w * (P(through_software) + P(on_paper_in_place))
```

`why_insulated` uses the model's whole answer rather than its single most likely option: with link weights of 1 and 2, counting modal choices ties for about one occupation in twenty.

An occupation gets one type, first match wins. Every threshold is a round number fixed before looking at the output:

| # | code | name | short | rule |
|---|---|---|---|---|
| 1 | `AUTOMATION_HEAVY` | Automation-heavy | Automation-heavy | `share_substituted >= 0.50` |
| 2 | `TRANSFORMING` | Transforming | Transforming | `share_substituted >= 0.30` and `share_assisted >= 0.20` |
| 3 | `AUGMENTED` | Augmented | Augmented | `share_assisted >= 0.30` and `share_substituted < 0.30` |
| 4 | `MECHANISABLE` | Mechanisable | Mechanisable | `share_mechanised >= 0.30` and `share_substituted < 0.30` |
| 5 | `INSULATED_PHYSICAL` | Insulated by physical work | Physical work | `share_insulated >= 0.40` and `why_insulated == "physical"` |
| 6 | `INSULATED_PEOPLE` | Insulated by work with people | People work | `share_insulated >= 0.40` and `why_insulated == "people"` |
| 7 | `MIXED` | Mixed | Mixed | everything else |

Mixed is a residual class, not a finding that AI will leave the job alone: its skills point in different directions and a single label would mislead.

An occupation is **near a cut-off** (`nl`) when moving any one share by 0.05 would change its type.

### Two choices made after the full run, stated openly

The thresholds above were fixed before the full run, with two exceptions that the full run's own distribution forced. Neither was tuned to make any particular job land anywhere; both are structural.

1. **Assistance is read at level 3, not level 2.** The design first read `COMP` as `P(complementarity >= 2)`. On all 13,939 skills that makes 62% of skills "AI assists" and 86% of occupations "Augmented": a classification that says the same thing about almost everything. Read at level 3 ("a clear gain across most of the work") the types spread, and jobs that can be judged by eye land where they should. The price is that `COMP` at level 3 correlates strongly with `SUB`; that matters less than it sounds, because the class rules only consult `COMP` for skills AI cannot take over.
2. **The first rule lost a condition.** It used to read `share_substituted >= 0.50 and sigma <= 0.20`, borrowed from the ILO's mean-and-spread rule. But `SUB` is a threshold probability, close to 0 or close to 1 by construction, so its spread is large exactly when the substituted share is between a half and four fifths. Translators (71% of skill weight substituted), accountants (75%) and data entry clerks (70%) fell through to "Mixed". `mu` and `sigma` are still published; they no longer gate a type.

The cut for "Machines can do" was deliberately left at 0.5 like every other class, although the final wording of the mechanical question is conservative (robot welding reads about 0.33). Read that share as a floor.

## Files

Same pipeline, one more scorer: `--scorer v2` reads `data/skill_scores_v2.json` and writes every output with a `_v2` suffix. Shapes are the existing ones plus the fields below; nothing existing changes meaning, so a page that ignores the new fields still works.

`data/skill_scores_v2.json` (committed), one entry per skill:

```json
{"uri": "...", "title": "...", "type": "skill|knowledge",
 "answers": {"digital_output": {"yes": 0.93},
             "ai_substitution": {"probs": [0.0, 0.1, 0.3, 0.5, 0.1], "position": 2.6, "confidence": 0.61},
             "mechanical": {"probs": [...], "position": 0.2, "confidence": 0.8},
             "complementarity": {"probs": [...], "position": 2.9, "confidence": 0.66},
             "mode": {"choice": "through_software", "probs": {"through_software": 0.8, "...": 0.2}, "confidence": 0.8},
             "deployment": {"choice": "routine", "probs": {"...": 0.0}, "confidence": 0.6}},
 "sub": 0.56, "comp": 0.61, "comp_part": 0.97, "mech": 0.00, "class": "S",
 "automation_risk": 6.7, "amplification_potential": 7.3, "mechanical_automation": 1.9,
 "model": "jev-1.13.0", "input_tokens": 2383}
```

Site files (`site/*_v2.json`):

| file | added |
|---|---|
| `stats_v2.json` | `"scheme": "shares"` (the older sets say `"quadrants"` or omit it), `"model"`, `"types": {"order": [...], "counts": {...}, "shares": {...}}`, `"skill_classes": {"counts": {...}, "shares": {...}}`, `"near_line"` as before. `quadrants` is absent |
| `search_index_v2.json` | per row `"k"` (mechanical, 1-10), `"sh": [substituted, assisted, mechanised, insulated]` (two decimals), `"nl"` (boolean), and `"q"` holds the type code |
| `groups_v2.json` | `"q"` counts by type code, `"mech": {mean, p10, p50, p90}`, `"sh"`: the mean of each share over the group's jobs |
| `skill_index_v2.json` | per row `"k"`, `"c"` (skill class code), `"p": [sub, comp, mech]` |
| `portfolio_data_v2.json` | skills: `"k"`, `"c"`, `"p"`, and the rationale borrowed from the Gemini scores with `"rf"` as for any numbers-only scorer; occupations: `"ak"` (mechanical mean), `"sh"`, `"nl"`, `"why"` (`why_insulated`), `"q"` = type code |
| `data_v2.json` | `mechanical_automation`, `shares`, and `quadrant` holds the type code |

### The scoring itself, published for the pages

Two more outputs of `build_site_indexes.py --scorer v2`, so the site can show how each skill was scored:

- `site/rubric_v2.json` — the question set exactly as asked, generated from `aiisco/rubric_v2.py` so the page and the scorer cannot drift apart: `{"model", "preamble", "questions": [{"id", "kind": "yesno|levels|choice", "label", "instructions", "levels": [...] or "options": [{"name", "text"}], "knowledge": {"instructions", "levels"} or null}], "classes": [{"code", "name", "rule"}], "display": "1.5 + 2.0 * position"}`. `label` is the short plain name a page shows ("AI substitution", "AI assistance", "Machine automation", "Digital output", "How it is exercised", "Deployment").
- `site/skill_answers_v2/<xx>.json` — the stored answers per skill, sharded by the first two hex characters of the skill's short id (256 files, a few kilobytes each, fetched only by a skill's detail view): `{"<id>": {"ty": "s|k", "d": 0.93, "s": [p0..p4], "k": [p0..p4], "c": [p0..p4], "mo": {"through_software": 0.8, ...}, "dp": {"routine": 0.6, ...}, "cf": {"s": 0.61, "k": 0.80, "c": 0.66, "mo": 0.80, "dp": 0.60}}}` — `d` digital output (probability of yes), `s` AI substitution, `k` mechanical, `c` complementarity, `mo` mode, `dp` deployment, `cf` the model's confidence per question; probabilities to two decimals.
- `site/skill_index_v2.json` rows also carry `"ty"` (`"s"` skill or `"k"` knowledge item) and `"mo"` (the chosen mode as one letter: `t` on things, `p` with people, `d` directing others, `s` through software, `a` on paper in place), so the table of all skills can filter without loading anything else.

## The site

`site/scorer.js` offers the v2 set as **TypeSafe** and makes it the default wherever the `_v2` files exist; **Gemini** (the original rubric, files without a suffix) is the alternative. The `_typesafe` files (the original rubric scored by the same judgment model) stay published for the method page's same-rubric comparison and are not in the switch.

Pages read `scheme` from the stats file of the active set:

- `quadrants`: everything as before.
- `shares`: wherever a quadrant badge, a quadrant mix bar or a quadrant filter appears, the seven types take its place (same components, more segments, type names from one module). Wherever "automation risk" and "amplification" appear as a pair of numbers, the page leads with the four-share bar ("AI can take over 42% · AI assists 31% · machines 5% · stays human 22%") and shows the display scores, now three of them, as secondary detail named "AI substitution", "AI assistance" and "Machine automation". Skill rows show the skill's class. The scatter keeps its two axes (AI substitution across, AI assistance up) without the cut-off lines, because under this scheme they are not class boundaries. The "near the line" marker reads `nl`. The badge explains a type from the job's own shares and the rule it met.

`skill.html` is the home of the scoring itself ("Skill scores" in the navigation). Without a skill selected it offers the lookup, a filterable and sortable table of every scored skill (class, the three display scores, mode, skill or knowledge item; paged, never thousands of rows in the DOM), and "How a skill is scored": the six questions in their exact wording from `rubric_v2.json`, how the answers become a class, and a link to the method page. With a skill selected it adds "How the model answered": per question the levels or options with the probability the model gave each, the chosen one marked, and the model's confidence, read from the skill's answers shard. Under the quadrant scheme the table shows the two scores and the detail view has no answers section.

Tone rules are unchanged: never "at risk" as a label for someone's job; a job with a large substituted share always shows a next step; "Mixed" is explained, not apologised for.
