# Shared front-end modules

Plain ES modules, no framework and no build step. Import them with a relative
path from a page in `site/`:

```js
import { rankOccupations } from './js/search.js';
```

Five of them are **pure** — they import nothing from the DOM and are unit-tested
with `node --test tests/js/`. Four are **thin DOM wrappers**: put logic in the
pure ones so it stays testable.

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
ground truth, and that the boxes are a hard cut at 6. It must never state or
imply anything about a second scoring run — that data is private. There is a
test asserting exactly that; keep it.

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

Every in-site link goes through `withScorer(href, location.search)`, or the local
second score set resets when the visitor follows it.

### `groupstats.js`

```js
QUADRANT_ORDER        // ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']
TABLE_COLUMNS         // [{ key, label, numeric }] — Job, ISCO, Automation, Amplification, Quadrant
SORTS                 // { automation, amplification, title } -> { key, label, descending }

groupPrefix(key)                              → string    // 'major:2' -> '2', 'all' -> ''
occupationsInGroup(rows, key)                 → Array<Object>
sortOccupations(rows, sortKey, descending?)   → Array<Object>   // new array, never mutates
quadrantMix(rows)                             → { total, counts, shares, order }
tableRows(rows)
  → Array<{ slug, title, cells: Array<{ key, text, numeric }> }>
```

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
NAV_ITEMS                         // [{ key, href, label }] x5
NOTICES_URL
renderChrome({ active, search })  → { nav: HTMLElement, footer: HTMLElement }
```

Injects the skip link, the brand, the five nav links (`aria-current="page"` on
the one whose `key` or `href` matches `active`) and the attribution footer.
`active` is one of `index`, `groups`, `skill`, `insights`, `method`. `search`
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
METHOD_URL                              // 'method.html'
renderQuadrantBadge(job, { search })    → HTMLElement
```

A real `<button>` carrying the quadrant name, a "near the line" marker when
`isNearLine`, and a popover holding `explainQuadrant(job)` plus a link to the
method page. Escape closes it and returns focus to the button; a click outside
closes it too. Returns a wrapper element — append it wherever the badge belongs.

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

## Data files

Built by `build_site_indexes.py` (see `aiisco/site_indexes.py`).

| File | Shape | Gzipped |
|---|---|---|
| `search_index.json` | `[{t, s, c, mg, a, m, q, alt[]}]` sorted by slug | 80 KB |
| `groups.json` | `{"<level>:<code>": {label, level, code, n, q, auto, amp, top, bottom, skills, parent, children}}` | 91 KB |
| `stats.json` | `{built, threshold, occupations, skills_scored, quadrants:{counts,shares}, near_line:{count,share}}` | 0.2 KB |
| `skill_index.json` | `[{id, t, a, m, ne, no}]` sorted by title | 259 KB |
| `skill_occupations.json` | `{"<skill_id>": {e: [slug], o: [slug]}}` — load only on `skill.html` | 737 KB |

Group keys are `all`, `major:2`, `sub:25`, `minor:251`, `unit:2512`. `parent`
walks up (`unit → minor → sub → major → all`), and `all.parent` is `null`.

`stats.json` is the only place a page may take a number from. `near_line` says
how many occupations sit within 0.5 of the cut-off — quote it rather than
inventing a hedge.

Rebuild with:

```
uv run python build_site_indexes.py
uv run python build_site_indexes.py --scorer typesafe   # local only, gitignored
```

The build fails if a file misses its gzipped budget. One budget was raised from
the brief's figure: `skill_index.json` is capped at 300 KB rather than 120 KB.
13,475 opaque 8-character ids cost 60.5 KB gzipped on their own and the skill
titles another 106 KB, so no format carrying both fits in 120 KB. The
`search_index.json` budget is met exactly, by filling it with as many ESCO
alternative labels as 80 KB allows (3,600 labels; every occupation that has one
gets its most distinct).
