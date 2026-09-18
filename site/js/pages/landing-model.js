// The landing page's decisions, as pure functions. No DOM, no globals, so
// `node --test tests/js/landing-model.test.js` can run every rule here.
//
// RECENTLY VIEWED — the contract every page shares
// ------------------------------------------------
// Key:    localStorage['ai-isco-recent']  (RECENT_KEY)
// Value:  a JSON array of occupation slugs, most recent first, at most 5
//         (RECENT_MAX), no duplicates, e.g. ["lawyer","chef","accountant"].
//         Slugs only: the title, code and quadrant are looked up in
//         search_index.json, so a rebuilt index never leaves stale copies.
//
// The job page writes an entry when it opens an occupation in the visitor's own
// framing (the brief: `&for=other` must NOT write). Do that with
//
//     import { addRecent } from './pages/landing-model.js';
//     addRecent(window.localStorage, slug);
//
// The landing page reads with `readRecent` and drops entries with `removeRecent`.
// Every accessor is wrapped in try/catch and tolerates a null store: Safari in
// private mode throws on access, and a hand-edited or corrupted value must never
// break a page — anything that is not an array of strings reads as an empty list.

import { nearestTitles, rankOccupations } from '../search.js';
import { formatCount, NOT_SCORED } from '../format.js';
import { QUADRANT_NAMES } from '../quadrant.js';

/** localStorage key holding the recently viewed slugs. */
export const RECENT_KEY = 'ai-isco-recent';

/** How many recently viewed jobs are kept and shown. */
export const RECENT_MAX = 5;

/** How many search results the list shows before "+ N more — keep typing". */
export const RESULT_LIMIT = 12;

/** How many near-misses the no-match state offers. */
export const NEAREST_COUNT = 3;

/**
 * The eight chips under the search box.
 *
 * Every slug is written out and verified against the real search_index.json by
 * tests/js/landing-model.test.js. The old page ran a prefix search per chip and
 * five of eight opened a different job than the one named (audit B2), so a chip
 * never guesses: `label` is the occupation's own ESCO title, so the chip cannot
 * promise a job it does not open. Chosen for spread across ISCO major groups.
 */
export const POPULAR_CHIPS = [
  { slug: 'software-developer', label: 'software developer' },
  { slug: 'nurse-responsible-for-general-care', label: 'nurse responsible for general care' },
  { slug: 'primary-school-teacher', label: 'primary school teacher' },
  { slug: 'accountant', label: 'accountant' },
  { slug: 'lawyer', label: 'lawyer' },
  { slug: 'chef', label: 'chef' },
  { slug: 'electrician', label: 'electrician' },
  { slug: 'bus-driver', label: 'bus driver' },
];

/** Upper-case the first letter only: ESCO titles are lower case, "ICT" is not. */
export function sentenceCase(text) {
  const value = String(text ?? '');
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/**
 * One index row by slug.
 * @param {Array<Object>} index search_index.json rows
 * @param {string} slug
 * @returns {Object|null}
 */
export function rowBySlug(index, slug) {
  return (index || []).find((row) => row && row.s === slug) || null;
}

/**
 * The secondary line of a result row: major group and quadrant word.
 * @param {{mg?: string, q?: string}} row
 * @returns {string}
 */
export function optionMeta(row) {
  const group = (row && row.mg) || 'Unclassified';
  return `${group} · ${QUADRANT_NAMES[row && row.q] || NOT_SCORED}`;
}

function outcome(state, query, extra) {
  return {
    state, query, results: [], total: 0, hidden: 0, nearest: [], ...extra,
  };
}

/**
 * What the search area should show for one query.
 *
 * States match the audit's D1: `empty` (nothing typed), `results` (everything
 * fits), `many` (more matches than fit — show `results`, offer the rest), and
 * `no-match` (nothing matched — offer `nearest`).
 *
 * @param {Array<Object>} index search_index.json rows
 * @param {string} query raw user input
 * @param {number} [limit=RESULT_LIMIT]
 * @returns {{state: string, query: string, results: Array<Object>,
 *            total: number, hidden: number, nearest: Array<Object>}}
 */
export function searchOutcome(index, query, limit = RESULT_LIMIT) {
  const text = String(query ?? '').trim();
  if (!text) return outcome('empty', '');
  const all = rankOccupations(index, text, Infinity);
  if (!all.length) {
    return outcome('no-match', text, { nearest: nearestTitles(index, text, NEAREST_COUNT) });
  }
  const results = all.slice(0, limit);
  return outcome(all.length > results.length ? 'many' : 'results', text, {
    results, total: all.length, hidden: all.length - results.length,
  });
}

/** The line under the heading. The count comes from stats.json, never a constant. */
export function subtitleText(stats) {
  const count = stats && Number.isFinite(stats.occupations) ? `${formatCount(stats.occupations)} ` : '';
  return `Search ${count}European occupations, scored for how much of the work AI `
    + 'could automate and how much it could amplify. These are model estimates, not forecasts.';
}

/** Shown only once a load has taken longer than 200 ms. */
export function loadingText(stats) {
  const count = stats && Number.isFinite(stats.occupations) ? `${formatCount(stats.occupations)} ` : '';
  return `Loading ${count}job titles…`;
}

/** The "many matches" line: show some, invite a narrower word. */
export function overflowText(hidden, query) {
  return `+ ${formatCount(hidden)} more matching “${query}” — keep typing to narrow it down, or`;
}

/** The "no match" line. Never a bare "0 results". */
export function noMatchText(query) {
  return `No job called “${query}”. Try a broader word — “nurse” rather than “ICU night nurse”.`;
}

function readStore(store, key) {
  try {
    return store ? store.getItem(key) : null;
  } catch {
    return null; // private mode, or storage disabled
  }
}

function writeStore(store, key, value) {
  try {
    if (store) store.setItem(key, value);
  } catch {
    /* private mode: the list simply does not persist */
  }
}

function parseList(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

/** Keep strings only, drop duplicates and blanks, cap at RECENT_MAX. */
function normalise(list) {
  if (!Array.isArray(list)) return [];
  const kept = [];
  for (const item of list) {
    if (typeof item === 'string' && item.trim() && !kept.includes(item)) kept.push(item);
    if (kept.length === RECENT_MAX) break;
  }
  return kept;
}

/**
 * The recently viewed slugs, most recent first.
 * @param {Storage|null} store window.localStorage, or null
 * @returns {string[]}
 */
export function readRecent(store) {
  const raw = readStore(store, RECENT_KEY);
  return normalise(raw === null ? [] : parseList(raw));
}

/**
 * Replace the list. Returns what was actually stored.
 * @param {Storage|null} store
 * @param {string[]} slugs
 * @returns {string[]}
 */
export function writeRecent(store, slugs) {
  const kept = normalise(slugs);
  writeStore(store, RECENT_KEY, JSON.stringify(kept));
  return kept;
}

/**
 * Put one slug at the front. The job page calls this.
 * @param {Storage|null} store
 * @param {string} slug
 * @returns {string[]} the new list
 */
export function addRecent(store, slug) {
  if (typeof slug !== 'string' || !slug.trim()) return readRecent(store);
  return writeRecent(store, [slug, ...readRecent(store).filter((item) => item !== slug)]);
}

/**
 * Drop one slug.
 * @param {Storage|null} store
 * @param {string} slug
 * @returns {string[]} the new list
 */
export function removeRecent(store, slug) {
  return writeRecent(store, readRecent(store).filter((item) => item !== slug));
}

/**
 * Resolve stored slugs against the index, dropping anything the index no longer
 * has, so a rebuilt index cannot leave a link to a job that is gone.
 * @param {string[]} slugs
 * @param {Array<Object>} index
 * @returns {Array<Object>} index rows, most recent first
 */
export function recentRows(slugs, index) {
  return (slugs || []).map((slug) => rowBySlug(index, slug)).filter(Boolean);
}
