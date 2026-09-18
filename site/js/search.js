// Ranking occupations for the search box. Pure: no DOM, no globals.
//
// Fixes the two failures the audit measured: "programmer" never reached
// Software Developer (ESCO alternative labels were dropped), and "nurse" put
// Nursery School Head Teacher above jobs actually called nurse.

/** Match tiers, best first. Exposed so the UI can label why a row matched. */
export const TIERS = {
  EXACT: 0,
  TITLE_PREFIX: 1,
  WORD_PREFIX: 2,
  ALT: 3,
  SUBSTRING: 4,
};

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
  if (direct && (!alt || direct.tier < TIERS.ALT)) {
    return { row, tier: direct.tier, sub: direct.sub, alt: null, title };
  }
  if (alt) return { row, tier: TIERS.ALT, sub: alt.sub, alt: alt.label, title };
  return null;
}

function compareMatches(a, b) {
  return a.tier - b.tier
    || a.sub - b.sub
    || a.title.length - b.title.length
    || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);
}

/**
 * Rank occupations for a query.
 *
 * Order: exact title, title prefix (whole-word before mid-word), word prefix in
 * the title, alternative-label match, then substring. Ties break by title
 * length, then alphabetically, so the order is stable for equal input.
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

/**
 * The closest titles by edit distance, for the "no job called X" state.
 * Returns index rows (not strings) so the UI can link to each slug.
 *
 * @param {Array<{t: string}>} index
 * @param {string} query
 * @param {number} [n=3]
 * @returns {Array<Object>} index rows, closest first
 */
export function nearestTitles(index, query, n = 3) {
  const folded = fold(query);
  if (!folded) return [];
  return (index || [])
    .map((row) => ({ row, title: fold(row.t) }))
    .map((item) => ({ ...item, distance: editDistance(folded, item.title) }))
    .sort((a, b) => a.distance - b.distance
      || a.title.length - b.title.length
      || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0))
    .slice(0, n)
    .map((item) => item.row);
}
