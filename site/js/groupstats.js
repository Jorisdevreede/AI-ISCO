// Client-side helpers for the group overview. Pure: no DOM, no globals.
//
// Works on search_index.json rows, so the group page can slice, sort and
// tabulate without loading the 13 MB portfolio file.

import { formatScore, NOT_SCORED } from './format.js';
import { QUADRANT_NAMES } from './quadrant.js';

/** The quadrant codes, in the order the mix bars show them. */
export const QUADRANT_ORDER = ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK'];

/** Columns of the accessible table alternative, in order. */
export const TABLE_COLUMNS = [
  { key: 't', label: 'Job', numeric: false },
  { key: 'c', label: 'ISCO', numeric: false },
  { key: 'a', label: 'Automation', numeric: true },
  { key: 'm', label: 'Amplification', numeric: true },
  { key: 'q', label: 'Quadrant', numeric: false },
];

/** Sort keys the ranked view offers. */
export const SORTS = {
  automation: { key: 'a', label: 'Most automation-exposed', descending: true },
  amplification: { key: 'm', label: 'Most amplified', descending: true },
  title: { key: 't', label: 'A to Z', descending: false },
};

/**
 * The ISCO code prefix a group key selects. "all" selects everything.
 * @param {string} key such as "major:2", "unit:2512" or "all"
 * @returns {string} "" for "all"
 */
export function groupPrefix(key) {
  const text = String(key ?? '');
  if (!text || text === 'all') return '';
  const split = text.indexOf(':');
  return split === -1 ? text : text.slice(split + 1);
}

/**
 * The occupations of one group.
 * @param {Array<{c: string}>} rows search_index.json rows
 * @param {string} key group key
 * @returns {Array<Object>} the rows in the group, input order preserved
 */
export function occupationsInGroup(rows, key) {
  const prefix = groupPrefix(key);
  return (rows || []).filter((row) => String(row.c ?? '').startsWith(prefix));
}

function compareValues(a, b, key) {
  if (typeof a[key] === 'number' && typeof b[key] === 'number') return a[key] - b[key];
  return String(a[key] ?? '').localeCompare(String(b[key] ?? ''));
}

/**
 * Sort for the ranked view. Never mutates the input; ties break by title so
 * the order is stable.
 * @param {Array<Object>} rows
 * @param {string} [sortKey='automation'] a key of SORTS
 * @param {boolean} [descending] overrides the sort's own direction
 * @returns {Array<Object>} a new array
 */
export function sortOccupations(rows, sortKey = 'automation', descending) {
  const sort = SORTS[sortKey] || SORTS.automation;
  const down = descending === undefined ? sort.descending : descending;
  return [...(rows || [])].sort((a, b) => {
    const order = compareValues(a, b, sort.key) * (down ? -1 : 1);
    return order || String(a.t ?? '').localeCompare(String(b.t ?? ''));
  });
}

/**
 * Quadrant counts and shares for a set of occupations.
 * @param {Array<{q: string}>} rows
 * @returns {{total: number, counts: Object, shares: Object, order: string[]}}
 */
export function quadrantMix(rows) {
  const list = rows || [];
  const counts = {};
  for (const code of QUADRANT_ORDER) counts[code] = 0;
  for (const row of list) {
    if (counts[row.q] === undefined) counts[row.q] = 0;
    counts[row.q] += 1;
  }
  const shares = {};
  for (const [code, n] of Object.entries(counts)) {
    shares[code] = list.length ? n / list.length : 0;
  }
  return { total: list.length, counts, shares, order: QUADRANT_ORDER };
}

/**
 * Rows for the accessible table alternative beside every canvas.
 * @param {Array<Object>} rows
 * @returns {Array<{slug: string, title: string, cells: Array<{key: string, text: string, numeric: boolean}>}>}
 */
export function tableRows(rows) {
  const text = {
    t: (row) => String(row.t ?? ''),
    c: (row) => String(row.c ?? ''),
    a: (row) => formatScore(row.a),
    m: (row) => formatScore(row.m),
    q: (row) => QUADRANT_NAMES[row.q] || NOT_SCORED,
  };
  return (rows || []).map((row) => ({
    slug: row.s,
    title: String(row.t ?? ''),
    cells: TABLE_COLUMNS.map((column) => ({
      key: column.key,
      text: text[column.key](row),
      numeric: column.numeric,
    })),
  }));
}
