// How far two score sets agree about the same occupations. Pure: no DOM.
//
// Both inputs are search_index rows ({ s: slug, a, m, q }) of the ORIGINAL
// rubric: two 1-10 scores cut at 6, four quadrants. Only occupations that carry
// both scores in both sets are compared. Nothing here says which set is right:
// there is no ground truth, so agreement is the only thing to measure.
//
// A shares set has no boxes to compare, and its `a` and `m` are three different
// display scores rather than the two this rubric cut. Feeding one in would
// produce a confident matrix of nonsense, so compareScoreSets refuses it.

import { QUADRANT_ORDER } from './groupstats.js';
import { isScored } from './format.js';
import { SHARES, schemeOfCode } from './scheme.js';

function scoredBySlug(rows) {
  const map = new Map();
  for (const row of rows || []) {
    if (row && row.s && isScored(row.a) && isScored(row.m)) map.set(row.s, row);
  }
  return map;
}

/** Pairs of rows, one per occupation that both sets scored. */
export function pairRows(first, second) {
  const other = scoredBySlug(second);
  return [...scoredBySlug(first).values()]
    .filter((row) => other.has(row.s))
    .map((row) => [row, other.get(row.s)]);
}

/** Ranks from 1, with tied values sharing the mean of the ranks they span. */
export function averageRanks(values) {
  const order = values.map((value, index) => [value, index]).sort((x, y) => x[0] - y[0]);
  const ranks = new Array(values.length);
  let start = 0;
  while (start < order.length) {
    let end = start;
    while (end + 1 < order.length && order[end + 1][0] === order[start][0]) end += 1;
    for (let k = start; k <= end; k += 1) ranks[order[k][1]] = (start + end) / 2 + 1;
    start = end + 1;
  }
  return ranks;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function pearson(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let top = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    top += (xs[i] - mx) * (ys[i] - my);
    sx += (xs[i] - mx) ** 2;
    sy += (ys[i] - my) ** 2;
  }
  return sx && sy ? top / Math.sqrt(sx * sy) : null;
}

/** Spearman rank correlation, or null when it is undefined (under 2 pairs, or no spread). */
export function spearman(xs, ys) {
  if (xs.length < 2 || xs.length !== ys.length) return null;
  return pearson(averageRanks(xs), averageRanks(ys));
}

function emptyMatrix() {
  const matrix = {};
  for (const from of QUADRANT_ORDER) {
    matrix[from] = {};
    for (const to of QUADRANT_ORDER) matrix[from][to] = 0;
  }
  return matrix;
}

/** Counts of occupations by their quadrant in the first set (row) and the second (column). */
export function quadrantMatrix(pairs) {
  const matrix = emptyMatrix();
  for (const [first, second] of pairs) {
    if (matrix[first.q] && second.q in matrix[first.q]) matrix[first.q][second.q] += 1;
  }
  return matrix;
}

/** The biggest group that changes quadrant, or null when none does. */
export function largestMove(matrix) {
  let best = null;
  for (const from of QUADRANT_ORDER) {
    for (const to of QUADRANT_ORDER) {
      const count = matrix[from][to];
      if (from !== to && count > 0 && (!best || count > best.count)) best = { from, to, count };
    }
  }
  return best;
}

function axisAgreement(pairs, key) {
  const first = pairs.map(([row]) => row[key]);
  const second = pairs.map(([, row]) => row[key]);
  return {
    rho: spearman(first, second),
    meanFirst: mean(first),
    meanSecond: mean(second),
    meanGap: mean(first.map((value, i) => Math.abs(value - second[i]))),
  };
}

/**
 * True for rows of the quadrant scheme: no shares array, and no `q` holding one
 * of the seven type codes. An empty or missing list is not a shares set, so it
 * falls through to the "nothing in common" answer below.
 * @param {Array<Object>} rows
 * @returns {boolean}
 */
export function isQuadrantRows(rows) {
  if (!Array.isArray(rows)) return false;
  return !rows.some((row) => row
    && (Array.isArray(row.sh) || schemeOfCode(row.q) === SHARES));
}

/**
 * Every occupation either file knows about, scored or not: the population the
 * comparison is a subset of. Quoting the subset alone invites the reader to
 * wonder where the missing rows went, so the page states both.
 * @param {Array<Object>} first
 * @param {Array<Object>} second
 * @returns {number}
 */
export function slugUniverse(first, second) {
  const slugs = new Set();
  for (const rows of [first, second]) {
    for (const row of rows || []) if (row && row.s) slugs.add(row.s);
  }
  return slugs.size;
}

/**
 * Everything the method page says about two score sets of the ORIGINAL rubric.
 * @param {Array<Object>} first search_index rows of one quadrant-scheme set
 * @param {Array<Object>} second search_index rows of the other
 * @returns {null | {n, total, same, share, matrix, move, automation, amplification}}
 *   `n` is how many occupations both runs scored, `total` how many exist at all;
 *   null when the sets share no scored occupation, and null when either set is
 *   not a quadrant-scheme set — there is nothing here for a shares set to mean
 */
export function compareScoreSets(first, second) {
  if (!isQuadrantRows(first) || !isQuadrantRows(second)) return null;
  const pairs = pairRows(first, second);
  if (!pairs.length) return null;
  const matrix = quadrantMatrix(pairs);
  const same = QUADRANT_ORDER.reduce((sum, quadrant) => sum + matrix[quadrant][quadrant], 0);
  return {
    n: pairs.length,
    total: slugUniverse(first, second),
    same,
    share: same / pairs.length,
    matrix,
    move: largestMove(matrix),
    automation: axisAgreement(pairs, 'a'),
    amplification: axisAgreement(pairs, 'm'),
  };
}
