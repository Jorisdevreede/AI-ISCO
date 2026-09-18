// Ranking occupations for the search box. Pure: no DOM, no globals.
//
// Fixes the failures the audits measured: "programmer" never reached Software
// Developer (ESCO alternative labels were dropped), "nurse" put Nursery School
// Head Teacher above jobs actually called nurse, an exact synonym ranked below
// any mid-title word, and the "no results" suggestions were unrelated short
// job names because absolute edit distance always prefers short titles.

/**
 * Match tiers, best first. Exposed so the UI can label why a row matched.
 *
 * `ALT_EXACT` sits above `WORD_PREFIX`: someone who types a job's exact ESCO
 * synonym ("programmer") means that job, not every title with the word
 * somewhere inside it. It stays below `TITLE_PREFIX`, so a job actually named
 * after the query still comes first.
 */
export const TIERS = {
  EXACT: 0,
  TITLE_PREFIX: 1,
  ALT_EXACT: 2,
  WORD_PREFIX: 3,
  ALT: 4,
  SUBSTRING: 5,
};

/** How far a typed word may sit from a real one and still be it: 35% of it. */
export const TYPO_RATIO = 0.35;

/** Words shorter than this are not worth a suggestion search of their own. */
const MIN_WORD = 3;

const WORD_BREAK = /[^a-z0-9]/;

/**
 * Lower-case, strip diacritics, collapse whitespace.
 * @param {string} text
 * @returns {string}
 */
export function fold(text) {
  return String(text ?? '')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ');
}

function words(folded) {
  return folded.split(WORD_BREAK).filter(Boolean);
}

/** True when `query` ends on a word boundary inside `haystack`. */
function endsOnBoundary(haystack, query) {
  const next = haystack.charAt(query.length);
  return next === '' || WORD_BREAK.test(next);
}

function titleTier(title, query) {
  if (title === query) return { tier: TIERS.EXACT, sub: 0 };
  if (title.startsWith(query)) {
    // "nurse assistant" must beat "nursery school head teacher" for "nurse".
    return { tier: TIERS.TITLE_PREFIX, sub: endsOnBoundary(title, query) ? 0 : 1 };
  }
  if (words(title).some((word) => word.startsWith(query))) {
    return { tier: TIERS.WORD_PREFIX, sub: 0 };
  }
  if (title.includes(query)) return { tier: TIERS.SUBSTRING, sub: 0 };
  return null;
}

function altRank(label, query) {
  if (label === query) return 0;
  if (label.startsWith(query)) return 1;
  if (words(label).some((word) => word.startsWith(query))) return 2;
  if (label.includes(query)) return 3;
  return null;
}

/** The best-matching alternative label of one row, or null. */
function bestAlt(row, query) {
  let best = null;
  for (const label of row.alt || []) {
    const rank = altRank(fold(label), query);
    if (rank !== null && (best === null || rank < best.sub)) best = { label, sub: rank };
  }
  return best;
}

function matchRow(row, query) {
  const title = fold(row.t);
  const direct = titleTier(title, query);
  const alt = bestAlt(row, query);
  if (alt && alt.sub === 0 && (!direct || direct.tier > TIERS.TITLE_PREFIX)) {
    return { row, tier: TIERS.ALT_EXACT, sub: 0, alt: alt.label, title };
  }
  if (direct && (!alt || direct.tier < TIERS.ALT)) {
    return { row, tier: direct.tier, sub: direct.sub, alt: null, title };
  }
  if (alt) return { row, tier: TIERS.ALT, sub: alt.sub, alt: alt.label, title };
  return null;
}

/**
 * The ISCO major group, 10 for a row without one. Lower is higher up the
 * classification, which is how the profession itself ("nurse responsible for
 * general care", 2221) is kept above the roles that assist it ("nurse
 * assistant", 3221) when both match the query equally well.
 */
function majorGroup(match) {
  const code = String((match.row && match.row.c) ?? '');
  return /^\d/.test(code) ? Number(code[0]) : 10;
}

function compareMatches(a, b) {
  return a.tier - b.tier
    || a.sub - b.sub
    || majorGroup(a) - majorGroup(b)
    || a.title.length - b.title.length
    || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);
}

/**
 * Rank occupations for a query.
 *
 * Order: exact title, title prefix (whole-word before mid-word), an exact
 * alternative label, a word prefix in the title, any other alternative-label
 * match, then substring. Ties break by ISCO major group, then title length,
 * then alphabetically, so the order is stable for equal input.
 *
 * @param {Array<{t: string, s: string, alt?: string[]}>} index search_index.json rows
 * @param {string} query raw user input; empty or whitespace returns []
 * @param {number} [limit=12] pass Infinity for every match
 * @returns {Array<{row: Object, tier: number, alt: string|null}>}
 *   `alt` is the alternative label that matched, so the UI can show
 *   `also matches "programmer"`; it is null when the title itself matched.
 */
export function rankOccupations(index, query, limit = 12) {
  const folded = fold(query);
  if (!folded) return [];
  const matches = [];
  for (const row of index || []) {
    const match = matchRow(row, folded);
    if (match) matches.push(match);
  }
  matches.sort(compareMatches);
  return matches
    .slice(0, limit)
    .map(({ row, tier, alt }) => ({ row, tier, alt }));
}

/**
 * Levenshtein distance between two already-folded strings.
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
export function editDistance(a, b) {
  if (a === b) return 0;
  if (!a.length || !b.length) return a.length || b.length;
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const current = [i];
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost);
    }
    previous = current;
  }
  return previous[b.length];
}

/** The distinct words of a query, longest first, so "ICU nurse" tries "nurse". */
function queryWords(folded) {
  return [...new Set(words(folded))].sort((a, b) => b.length - a.length);
}

/** The first word of the query that is a real search of its own. */
function wordHits(index, list, n) {
  for (const word of list) {
    if (word.length < MIN_WORD) continue;
    const hits = rankOccupations(index, word, n);
    if (hits.length) return { kind: 'word', word, rows: hits.map((hit) => hit.row) };
  }
  return null;
}

/**
 * How far this title sits from the query, measured word against word.
 *
 * Absolute distance against the whole title is what made "nrse" suggest Chef:
 * for a short query the shortest titles in the corpus always win. Comparing
 * each typed word with each title word, and only accepting a distance that is
 * small *relative* to the longer of the two, means a suggestion has to look
 * like something the visitor typed.
 *
 * @returns {number|null} null when no pair of words is close enough
 */
function closer(found, best) {
  if (best === null) return true;
  return found.distance - Number(found.same) < best.distance - Number(best.same);
}

function bestWordDistance(title, list) {
  let best = null;
  for (const word of words(title)) {
    if (word.length < MIN_WORD) continue;
    for (const typed of list) {
      const distance = editDistance(typed, word);
      const room = Math.ceil(TYPO_RATIO * Math.max(typed.length, word.length));
      const found = { distance, same: typed[0] === word[0] };
      if (distance <= room && closer(found, best)) best = found;
    }
  }
  return best;
}

function compareTypos(a, b) {
  return a.distance - b.distance
    || Number(b.same) - Number(a.same)
    || a.title.length - b.title.length
    || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);
}

function typoHits(index, list, n) {
  const typed = list.filter((word) => word.length >= MIN_WORD);
  if (!typed.length) return [];
  const scored = [];
  for (const row of index || []) {
    const title = fold(row.t);
    const best = bestWordDistance(title, typed);
    if (best) scored.push({ row, title, ...best });
  }
  scored.sort(compareTypos);
  return scored.slice(0, n).map((item) => item.row);
}

/**
 * What to offer when a query matched nothing.
 *
 * Two steps, and either may come up empty. First every word of the query is
 * tried as a search of its own, longest first: "ICU nurse" has no match but
 * "nurse" has eleven, and those are real answers rather than guesses. Only if
 * no word matches does it fall back to typo distance, and only for candidates
 * that are close to a word the visitor actually typed. When nothing clears
 * that bar it says so — an empty list is kinder than three unrelated jobs.
 *
 * @param {Array<{t: string}>} index
 * @param {string} query
 * @param {number} [n=3]
 * @returns {{kind: 'word'|'near'|'none', word: string|null, rows: Array<Object>}}
 */
export function suggestions(index, query, n = 3) {
  const list = queryWords(fold(query));
  const empty = { kind: 'none', word: null, rows: [] };
  if (!list.length) return empty;
  const byWord = wordHits(index, list, n);
  if (byWord) return byWord;
  const rows = typoHits(index, list, n);
  return rows.length ? { kind: 'near', word: null, rows } : empty;
}

/**
 * The rows `suggestions` would offer, for a caller that only needs the links.
 *
 * @param {Array<{t: string}>} index
 * @param {string} query
 * @param {number} [n=3]
 * @returns {Array<Object>} index rows, best first; empty when nothing fits
 */
export function nearestTitles(index, query, n = 3) {
  return suggestions(index, query, n).rows;
}
