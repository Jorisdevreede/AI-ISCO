# Shared front-end modules

Plain ES modules, no framework and no build step. Import them with a relative
path from a page in `site/`:

```js
import { rankOccupations } from './js/search.js';
```

Seven of them are **pure** — they import nothing from the DOM and are unit-tested
with `npm test` (`node --test tests/js/*.test.js`). Five are **thin DOM wrappers**: put logic in the
pure ones so it stays testable.

## Two schemes

The site publishes score sets of two shapes, and a page learns which one it has
from the `scheme` field of the stats file it loaded:

| `scheme` | sets | a job's class comes from | `job.q` holds |
|---|---|---|---|
| `quadrants` | Gemini (no suffix), `_typesafe` | two 1-10 scores cut at 6 | `TRANSFORM` `SHRINK` `EVOLVE` `STABLE` |
| `shares` | `_v2` (TypeSafe) | four shares over the job's skills | one of seven type codes |

`quadrant.js` owns the four boxes and is unchanged. `scheme.js` owns everything
that has to work under both, and `shares.js` / `shares-bar.js` own the four-share
figure. **Nothing outside `quadrant.js` should read `QUADRANT_NAMES` directly any
more**: `typeLabel(code)` names a code of either scheme, and the two code sets are
disjoint, so it almost never needs a `scheme` argument.

A working example of everything below is [`_selftest.html`](./_selftest.html),
served at `http://localhost:8001/js/_selftest.html`.

---

## Page skeleton

Copy this. The order of the last two `<script>` tags matters.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Find a job — AI-ISCO</title>
<link rel="stylesheet" href="css/app.css">
<link rel="preload" href="search_index.json" as="fetch" crossorigin>
</head>
<body>

<!-- Must be static markup: scorer.js looks this element up by id. -->
<nav id="site-nav"></nav>

<main id="main" class="page">
  <h1>Will AI change your job?</h1>
</main>

<script src="scorer.js" defer></script>
<script type="module" src="js/page-find-a-job.js"></script>
</body>
</html>
```

### Script loading order

1. `css/app.css` in `<head>`.
2. `<nav id="site-nav"></nav>` as the first element of `<body>`, **in the HTML**.
   `site/scorer.js` calls `document.getElementById('site-nav')` at execution time
   and appends its score-set switch to whatever it finds. If the element is not
   in the markup, the switch silently disappears.
3. `scorer.js` with `defer`. It must run before your module so `window.scorerFetch`
   exists by the time `loadJSON` is called.
4. Your page module. Deferred scripts and module scripts run in document order, so
   this is guaranteed.

`renderChrome` only ever **appends** to `#site-nav` (its own nodes carry
`data-chrome`), so the scorer switch, which arrives later after a network probe,
is never wiped out.

---

## Pure modules

### `search.js`

```js
rankOccupations(index, query, limit = 12)
  → Array<{ row: Object, tier: number, alt: string|null }>
```
Ranks `search_index.json` rows. `row` is the index row; `alt` is the alternative
label that matched, so the UI can print `also matches "programmer"`, and is
`null` when the title itself matched. An empty or whitespace query returns `[]`.
Pass `Infinity` as the limit to get every match (for "+ 41 more").

Order, best first: exact title → title prefix (whole-word before mid-word, so
`nurse assistant` beats `nursery school head teacher`) → a word inside the title
→ alternative label → substring. Ties break by title length, then alphabetically,
so the same input always gives the same order. Matching folds case and diacritics.

```js
TIERS                                  // { EXACT: 0, TITLE_PREFIX: 1, WORD_PREFIX: 2, ALT: 3, SUBSTRING: 4 }
fold(text)            → string         // lower-cased, diacritics stripped, whitespace collapsed
editDistance(a, b)    → number         // Levenshtein, on already-folded strings
nearestTitles(index, query, n = 3)
  → Array<Object>                      // index ROWS (not strings), closest first, for the no-match state
```

### `quadrant.js`

```js
THRESHOLD             // 6, the same cut-off as aiisco/rollup.py
NEAR_LINE             // 0.5
QUADRANT_NAMES        // { TRANSFORM: 'Transform', SHRINK: 'Shrink', EVOLVE: 'Evolve', STABLE: 'Stable' }

quadrantOf(a, m)          → 'TRANSFORM'|'SHRINK'|'EVOLVE'|'STABLE'|null
distanceToLine(a, m)      → number|null    // to the closer axis, rounded to 2 decimals
isNearLine(a, m)          → boolean        // distance <= 0.5
exposureWord(score)       → 'low'|'medium'|'high'|null   // <4, <7, >=7
explainQuadrant(job)      → { heading: string, sentences: string[], nearLine: boolean }
```

`job` is `{ t?, a, m, q? }`. `explainQuadrant` is the **only** place the
uncertainty copy lives. It says: which side of the cut-off each score falls, how
far this job sits from the line, that the scores are model estimates with no
ground truth, and that the boxes are a hard cut at 6. Every number in it is
computed from the job's own scores; it quotes nothing about the second scoring
run, and a test keeps typed-in figures out of it.

Unchanged by scoring v2, and still four boxes only. For a page that must work
under either scheme, use `scheme.js` below.

### `scheme.js`

```js
QUADRANTS, SHARES                      // 'quadrants', 'shares'
TYPE_ORDER                             // the seven type codes, rule order
TYPE_NAMES, TYPE_SHORT                 // code -> 'Insulated by physical work' / 'Physical work'
TYPE_COLOR_VARS                        // code -> '--type-insulated-physical'
TYPE_DESCRIPTIONS                      // code -> one sentence about the job
TYPE_RULES                             // code -> the rule it matched, in plain words
SKILL_CLASS_ORDER                      // ['S', 'A', 'M', 'I']
SKILL_CLASS_NAMES                      // S -> 'AI can take over' … I -> 'Stays human'
SKILL_CLASS_COLOR_VARS                 // S -> '--class-substituted'

schemeOf(stats)          → 'quadrants'|'shares'   // missing field = quadrants
schemeOfCode(code)       → 'quadrants'|'shares'|null
orderOf(scheme)          → string[]               // a fresh array, in display order
orderForCounts(counts)   → string[]               // the order a counts object is keyed by
splitOf(stats)           → { scheme, order, counts, shares }   // stats.types OR stats.quadrants
thresholdOf(stats)       → number|null            // null where the scheme has no cut-off

typeLabel(code, scheme?)      → string    // 'Transform' | 'Transforming' | 'Not scored'
typeShortLabel(code, scheme?) → string
colorVarOf(code, scheme?)     → string|null
typeDescription(code)         → string
typeRule(code)                → string
skillClassName(code)          → string
skillClassColorVar(code)      → string|null
skillClassOf(row, scheme?)    → 'S'|'A'|'M'|'I'|null   // null under quadrants

explainType(job)         → { heading, sentences[], nearLine }
explain(job, scheme?)    → { heading, sentences[], nearLine }   // either scheme
isNear(job, scheme?)     → boolean        // quadrants: the 0.5 rule; shares: job.nl
nearLineCaveat(stats)    → string         // the one permanent caveat beside a split
```

`explainType` mirrors `explainQuadrant`: it names the rule the job met in plain
words, gives the four shares as one sentence, says whether the job sits near a
cut-off, and says these are model estimates with no ground truth. For `MIXED` it
adds that Mixed is a residual class, **not** a finding that AI will leave the job
alone. Every figure comes from the job's own `sh`, `q` and `nl`; a test asserts
that no other number can appear.

`typeLabel(code)` with no scheme resolves the code itself, so the common page
edit is a one-for-one swap of `QUADRANT_NAMES[x] || 'Not scored'`.

### `shares.js`

```js
SHARE_ORDER                       // ['S', 'A', 'M', 'I'] — the order `sh` stores
SHARE_LABELS                      // S -> 'AI can take over' … I -> 'stays human'

isShares(sh)          → boolean   // four finite numbers adding up to ~1
sharePercents(sh)     → Array<{ code, label, share, percent }>
shareSentence(sh)     → string    // 'AI can take over 42% · AI assists 31% · machines 5% · stays human 22%'
shareAriaLabel(sh, name?) → string
shareSegments(sh)     → Array<{ code, label, percent, width }>   // drops the 0% parts
largestShare(sh)      → { code, label, percent }|null
```

The percentages **always add up to 100**: rounding each on its own gives 99 or
101 often enough to notice, so the remainder goes to the largest fractions
(largest-remainder method) and a test pins it. Anything that is not a usable
shares array degrades to `[]` / `''`, never to `NaN`.

### `urlstate.js`

```js
parseHash(hash)           → { id: string|null, params: Object<string,string> }
buildHash(id, params)     → string       // '#...' or '' — the inverse of parseHash
jobHref(slug, opts)       → string       // opts: { for, from }
groupHref(key, view)      → string
skillHref(id)             → string
withScorer(href, search)  → string       // carries ?scorer= when the page has one
```

A hash is an optional bare id followed by `&key=value` pairs. Everything is
percent-encoded except `:`, so `#g=major:2&view=ranked` stays readable and still
round-trips. Empty values are dropped rather than written as bare keys.

```
job.html#software-developer
job.html#software-developer&for=other
job.html#optical-engineer&from=unit:2149
groups.html#g=major:2&view=ranked
skill.html#59b27e7b
```

Every in-site link goes through `withScorer(href, location.search)`, or a shared
`?scorer=` link loses its score set when the visitor follows it.

### `groupstats.js`

```js
QUADRANT_ORDER        // ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']
TABLE_COLUMNS         // [{ key, label, numeric }] — Job, ISCO, Automation, Amplification, Quadrant
SORTS                 // { automation, amplification, title } -> { key, label, descending }

groupPrefix(key)                              → string    // 'major:2' -> '2', 'all' -> ''
occupationsInGroup(rows, key)                 → Array<Object>
sortOccupations(rows, sortKey, descending?)   → Array<Object>   // new array, never mutates
typeMix(rows, scheme?)                        → { total, counts, shares, order }
quadrantMix(rows, scheme?)                    → the same function under its old name
tableColumns(scheme?)                         → TABLE_COLUMNS, last one named 'Type' under shares
tableRows(rows, scheme?)
  → Array<{ slug, title, cells: Array<{ key, text, numeric }> }>
```

With no `scheme`, `typeMix` reads the codes the rows carry: a shares set counts
and orders the seven types, a quadrant set counts the four boxes exactly as
before. `tableRows` names the last cell with `typeLabel`, so it prints a type
name rather than "Not scored" under a shares set.

All of it works on `search_index.json` rows, so the group page never has to load
the 13 MB portfolio file.

### `format.js`

```js
NOT_SCORED                        // 'Not scored'
formatScore(value)   → string     // one decimal, or NOT_SCORED — never a bare '?'
isScored(value)      → boolean
formatCount(value)   → string     // 3043 -> '3,043'
formatPercent(share, digits = 0)  → string   // 0.513 -> '51%'
formatShare(part, whole, noun)    → string   // '1,509 of 3,043 jobs (50%)'
```

### `rationale.js`

```js
scorerName(key)                   → string   // 'gemini' -> 'Gemini'
rationaleWriterLine(skills)       → string   // the sentence above a list of rationales
borrowedRationaleNote(skill.rf)   → { writer, label, text } | null
```

A scorer that returns numbers only writes no text, so the build lends its skills
the rationale another model wrote and marks each with
`rf: { s: scorer, a: its automation score, m: its amplification score }`.
`borrowedRationaleNote` turns that mark into the note behind the "i"; it returns
`null` for a skill without `rf`, whose rationale was written for the scores on
screen. Pair it with `renderInfoNote` below.

---

## DOM modules

### `data.js`

```js
loadJSON(name)                   → Promise<any>
whenSlow(promise, ms, onSlow)    → Promise<any>
clearCache()                     → void
LoadError                        // Error subclass; .message is showable as-is, .file is the name
```

`name` has no `.json` (`loadJSON('search_index')`). Goes through
`window.scorerFetch` when `scorer.js` has run, otherwise plain `fetch`. Cached per
page load; a rejected load is evicted so a retry really retries.

```js
whenSlow(loadJSON('search_index'), 200, () => { skeleton.hidden = false; })
  .then(render)
  .catch(showError);
```

### `chrome.js`

```js
NAV_ITEMS                         // [{ key, href, label }], one per page
NOTICES_URL
renderChrome({ active, search })  → { nav: HTMLElement, footer: HTMLElement }
```

Injects the skip link, the brand, the nav links (`aria-current="page"` on
the one whose `key` or `href` matches `active`) and the attribution footer.
`active` is one of `index`, `groups`, `tree`, `skill`, `insights`, `method`. `search`
defaults to `location.search` and its `?scorer=` is carried onto every link.

The footer wording is copied verbatim from the original `site/index.html`: ESCO
and the ILO require those acknowledgements. Do not reword it.

### `combobox.js`

```js
createCombobox({ input, listbox, status, getResults, renderOption, onSelect })
  → { open, close, refresh, destroy }
```

The ARIA 1.2 combobox pattern: `role="combobox"`, `aria-expanded`,
`aria-controls`, `aria-autocomplete="list"`, `aria-activedescendant`, and
`role="option"` children. Arrow Up/Down wrap, Home/End jump, Enter selects,
Escape closes; pointer and touch work through `pointerdown`, which keeps focus in
the input. `status` should be a `role="status" aria-live="polite"` element; it
receives the result count. `getResults(query)` may return an array or a promise —
stale responses are discarded.

### `badge.js`

```js
METHOD_URL                                      // 'method.html'
renderQuadrantBadge(job, { search })            → HTMLElement
renderTypeBadge(job, { scheme, search })        → HTMLElement
```

A real `<button>` carrying the class name, a "near the line" marker when
`isNear`, and a popover holding `explain(job)` plus a link to the method page.
Escape closes it and returns focus to the button; a click outside closes it too.
Returns a wrapper element — append it wherever the badge belongs.

`renderQuadrantBadge` is the four-box badge and is unchanged: a job from a shares
set renders as "Not scored" there. **`renderTypeBadge` is what a page should call
now** — it follows the active scheme, and with no `scheme` option it resolves one
from `job.q`. Pass the whole row (`{ t, q, a, m, sh, nl }`) so the popover has
the shares it explains from. The button carries `data-quadrant` under quadrants
and `data-type` under shares.

### `shares-bar.js`

```js
renderSharesBar(sh, { name, compact })   → HTMLElement
```

The four-share stacked bar: one `role="img"` whose `aria-label` is the whole
sentence, plus a legend repeating every part in words. `name` is the job title
and goes into the accessible name. `compact` swaps the four-row legend for the
one-line sentence, for a list row. Shares that are missing or malformed render as
a "Shares not scored" note, never as an empty bar.

Colour is never the only signal: each segment carries a stripe angle, each legend
swatch a different shape and its class letter, and the numbers are in the text
beside them.

### `info-note.js`

```js
renderInfoNote({ label, text })   → { button: HTMLButtonElement, note: HTMLElement }
```

An "i" button that shows and hides a short note. A disclosure rather than a
tooltip, so it works by tap and by keyboard and reports `aria-expanded`. Put the
button after the text it is about and the note where it may take a full line.

---

## Styles

`css/app.css` holds the tokens and the shared components: nav, footer, buttons,
chips, cards, combobox, badge and popover, tabs, tables, skeletons and error
blocks. Useful class names: `.page`, `.card`, `.card-grid`, `.chip`, `.chip-row`,
`.button`, `.button--quiet`, `.tabs`/`.tab`, `.table-scroll`, `.skeleton`,
`.error-block`, `.loading`, `.muted`, `.small`, `.numeric`, `.visually-hidden`.

Rules to keep:

- **Never let the page scroll sideways.** Put wide things in `.table-scroll`,
  which scrolls itself. The 390 px check is
  `document.documentElement.scrollWidth <= window.innerWidth`.
- **Contrast.** `--fg-muted` (`#9a9aa8`) is the muted text token; it reaches
  7.1:1 on the page background and 6.0:1 on a tinted chip. The old `#888894`
  dropped to 4.9:1 on a chip, so it was raised. `--fg2` is kept as an alias
  because `site/scorer.js` reads that name.
- **Touch targets** are at least `--tap` (24 px) in both directions.
- Nothing sets `outline: none`; `:focus-visible` always draws a gold ring.
- Motion is disabled under `prefers-reduced-motion`.

---

## Adapting a page to the shares scheme

A recipe, in the order that costs least.

1. **Load the scheme once.** `const scheme = schemeOf(await loadJSON('stats'))`.
   Pass it down; never sniff the URL or the file names for it.
2. **Names.** Replace every `QUADRANT_NAMES[x] || 'Not scored'` with
   `typeLabel(x)`. It resolves either scheme from the code, so this is a
   one-for-one swap with no new argument.
3. **Orders.** Replace `QUADRANT_ORDER` with `orderOf(scheme)` where you know the
   scheme, or `orderForCounts(counts)` where you are iterating a counts object —
   a four-box loop over a seven-type tally silently totals zero, which renders as
   a confident and completely invented split.
4. **The badge.** `renderQuadrantBadge(job, …)` → `renderTypeBadge(job, …)`, and
   pass the whole row so the popover has `sh` and `nl`.
5. **The split.** `stats.quadrants.counts` → `splitOf(stats)`, which reads
   `stats.types` under shares and `stats.quadrants` otherwise, and never returns
   `undefined`.
6. **The pair of numbers.** Lead with `renderSharesBar(row.sh, { name: row.t })`
   and demote the display scores to secondary detail, renamed "AI substitution"
   (`a`), "AI assistance" (`m`) and "Machine automation" (`k` / `ak`).
7. **Skill rows.** `skillClassOf(row, scheme)` gives `S`/`A`/`M`/`I`;
   `skillClassName(code)` gives the words. Never print the letter.
8. **Copy.** Anything that says "cut-off", "6", "box" or "either side of the
   line" is quadrant copy. `nearLineCaveat(stats)` and `explain(job, scheme)`
   already say the right thing for both; the rest is yours.
9. **Colour.** `colorVarOf(code, scheme)` and `skillClassColorVar(code)` give the
   custom-property names. `app.css` defines all eleven. Pair every one with a
   name or a shape — a test and the audit both require it.

Two traps worth naming. `c` means the ISCO code on an occupation row and the
skill class on a skill row, so never put skill rows through `groupstats.js`.
And `isNear(job, scheme)` reads `job.nl` under shares — the distance rule in
`isNearLine(a, m)` has no meaning there, because the two scores are not class
boundaries any more.

---

## Data files

Built by `build_site_indexes.py` (see `aiisco/site_indexes.py`).

| File | Shape | Gzipped |
|---|---|---|
| `search_index.json` | `[{t, s, c, mg, a, m, q, alt[]}]` sorted by slug | 80 KB |
| `groups.json` | `{"<level>:<code>": {label, level, code, n, q, auto, amp, top, bottom, skills, parent, children}}` | 91 KB |
| `stats.json` | `{built, threshold, occupations, skills_scored, quadrants:{counts,shares}, near_line:{count,share}}` | 0.2 KB |
| `skill_index.json` | `[{id, t, a, m, ne, no}]` sorted by title | 259 KB |
| `skill_occupations.json` | `{"<skill_id>": {e: [slug], o: [slug]}}` — load only on `skill.html` | 737 KB |

The `_v2` set carries the same files with a `_v2` suffix and the extra fields
`docs/scoring-v2.md` specifies: `sh` and `nl` on an occupation, `k` everywhere,
`c` and `p` on a skill, `types` and `skill_classes` in `stats_v2.json`, and `q`
holding a type code. `stats_v2.json` has **no** `quadrants` and no `threshold`.

Group keys are `all`, `major:2`, `sub:25`, `minor:251`, `unit:2512`. `parent`
walks up (`unit → minor → sub → major → all`), and `all.parent` is `null`.

`stats.json` is the only place a page may take a number from. `near_line` says
how many occupations sit within 0.5 of the cut-off — quote it rather than
inventing a hedge.

Rebuild with:

```
uv run python build_site_indexes.py
uv run python build_site_indexes.py --scorer typesafe   # the second score set
```

The build fails if a file misses its gzipped budget. One budget was raised from
the brief's figure: `skill_index.json` is capped at 300 KB rather than 120 KB.
13,475 opaque 8-character ids cost 60.5 KB gzipped on their own and the skill
titles another 106 KB, so no format carrying both fits in 120 KB. The
`search_index.json` budget is met exactly, by filling it with as many ESCO
alternative labels as 80 KB allows (3,600 labels; every occupation that has one
gets its most distinct).
