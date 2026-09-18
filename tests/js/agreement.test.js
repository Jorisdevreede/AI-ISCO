import test from 'node:test';
import assert from 'node:assert/strict';

import {
  averageRanks, compareScoreSets, isQuadrantRows, largestMove, pairRows, quadrantMatrix,
  slugUniverse, spearman,
} from '../../site/js/agreement.js';

const row = (s, a, m, q) => ({ s, a, m, q });

const FIRST = [
  row('nurse', 3, 7, 'EVOLVE'),
  row('clerk', 8, 7, 'TRANSFORM'),
  row('mason', 2, 3, 'STABLE'),
  row('packer', 7, 3, 'SHRINK'),
  row('unscored', null, 5, null),
  row('only-in-first', 5, 5, 'STABLE'),
];
const SECOND = [
  row('nurse', 6.5, 6.4, 'TRANSFORM'),
  row('clerk', 8.5, 6.2, 'TRANSFORM'),
  row('mason', 3, 2.5, 'STABLE'),
  row('packer', 7.5, 2, 'SHRINK'),
  row('unscored', 4, 4, 'STABLE'),
  row('only-in-second', 5, 5, 'STABLE'),
];

test('only occupations scored on both axes in both sets are paired', () => {
  const pairs = pairRows(FIRST, SECOND);
  assert.deepEqual(pairs.map(([first]) => first.s), ['nurse', 'clerk', 'mason', 'packer']);
  assert.deepEqual(pairRows(FIRST, null), []);
  assert.deepEqual(pairRows(undefined, SECOND), []);
});

test('tied values share the mean of the ranks they span', () => {
  assert.deepEqual(averageRanks([10, 20, 20, 30]), [1, 2.5, 2.5, 4]);
  assert.deepEqual(averageRanks([5, 5, 5]), [2, 2, 2]);
  assert.deepEqual(averageRanks([]), []);
});

test('rank correlation is 1 for the same order, -1 for the reverse, null without spread', () => {
  assert.equal(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  assert.equal(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
  assert.equal(spearman([1, 1, 1], [1, 2, 3]), null);
  assert.equal(spearman([1], [1]), null);
  assert.equal(spearman([1, 2], [1]), null);
});

test('rank correlation handles ties the textbook way', () => {
  // ranks [1, 2.5, 2.5, 4] against [1, 2, 3, 4]
  const rho = spearman([1, 2, 2, 3], [1, 2, 3, 4]);
  assert.ok(Math.abs(rho - 0.9486832980505138) < 1e-12, String(rho));
});

test('the matrix counts first-set quadrant by second-set quadrant', () => {
  const matrix = quadrantMatrix(pairRows(FIRST, SECOND));
  assert.equal(matrix.EVOLVE.TRANSFORM, 1);
  assert.equal(matrix.TRANSFORM.TRANSFORM, 1);
  assert.equal(matrix.STABLE.STABLE, 1);
  assert.equal(matrix.SHRINK.SHRINK, 1);
  assert.equal(matrix.EVOLVE.EVOLVE, 0);
});

test('a row with a quadrant the model does not know is left out of the matrix', () => {
  const matrix = quadrantMatrix([[row('x', 1, 1, 'OTHER'), row('x', 1, 1, 'STABLE')]]);
  assert.equal(Object.values(matrix).flatMap((cells) => Object.values(cells))
    .reduce((sum, count) => sum + count, 0), 0);
});

test('the largest move ignores the diagonal, and is null when nothing moves', () => {
  const matrix = quadrantMatrix(pairRows(FIRST, SECOND));
  assert.deepEqual(largestMove(matrix), { from: 'EVOLVE', to: 'TRANSFORM', count: 1 });
  assert.equal(largestMove(quadrantMatrix(pairRows(FIRST, FIRST))), null);
});

test('the comparison reports agreement, the shift in level and the gap per axis', () => {
  const result = compareScoreSets(FIRST, SECOND);
  assert.equal(result.n, 4);
  assert.equal(result.same, 3);
  assert.equal(result.share, 0.75);
  assert.equal(result.automation.meanFirst, 5);
  assert.equal(result.automation.meanSecond, 6.375);
  assert.equal(result.automation.meanGap, 1.375);
  assert.equal(result.automation.rho, 1);
  // amplification ranks [3.5, 3.5, 1.5, 1.5] against [4, 3, 2, 1]: 4 / sqrt(4 * 5)
  assert.ok(Math.abs(result.amplification.rho - 4 / Math.sqrt(20)) < 1e-12);
});

test('two sets that share nothing give null, not a division by zero', () => {
  assert.equal(compareScoreSets([row('a', 1, 1, 'STABLE')], [row('b', 1, 1, 'STABLE')]), null);
  assert.equal(compareScoreSets([], []), null);
});

/* --- this comparison is about the original rubric, and only that ----------- */

// Rows of a shares set: `q` holds a type code and `sh` the four shares.
const SHARES_ROWS = [
  { s: 'nurse', t: 'nurse', a: 4.2, m: 6.1, k: 1.2, q: 'AUGMENTED', sh: [0.1, 0.4, 0.0, 0.5] },
  { s: 'clerk', t: 'clerk', a: 7.7, m: 6.0, k: 1.1, q: 'TRANSFORMING', sh: [0.4, 0.3, 0.0, 0.3] },
  { s: 'mason', t: 'mason', a: 2.1, m: 3.3, k: 2.0, q: 'INSULATED_PHYSICAL', sh: [0, 0.1, 0.1, 0.8] },
  { s: 'packer', t: 'packer', a: 5.0, m: 3.0, k: 7.4, q: 'MECHANISABLE', sh: [0.1, 0.1, 0.5, 0.3] },
];

test('quadrant rows are recognised, and a type code or a shares array is not', () => {
  assert.equal(isQuadrantRows(FIRST), true);
  assert.equal(isQuadrantRows([]), true);
  assert.equal(isQuadrantRows(SHARES_ROWS), false);
  assert.equal(isQuadrantRows([row('a', 1, 1, 'STABLE'), SHARES_ROWS[0]]), false);
  assert.equal(isQuadrantRows([{ s: 'x', a: 1, m: 1, q: 'STABLE', sh: [0.25, 0.25, 0.25, 0.25] }]),
    false);
  assert.equal(isQuadrantRows(null), false);
});

test('the comparison refuses a shares set rather than inventing a matrix', () => {
  assert.equal(compareScoreSets(SHARES_ROWS, SECOND), null);
  assert.equal(compareScoreSets(FIRST, SHARES_ROWS), null);
  assert.equal(compareScoreSets(SHARES_ROWS, SHARES_ROWS), null);
  assert.equal(compareScoreSets(null, SECOND), null);
  // The two quadrant sets still compare exactly as they did.
  assert.equal(compareScoreSets(FIRST, SECOND).n, 4);
});

/* --- the comparison states its own population (A8) ------------------------ */

test('the universe is every occupation either file knows, scored or not', () => {
  // FIRST and SECOND share four scored slugs; each also has one the other lacks,
  // and "unscored" exists in both but carries no score in FIRST.
  assert.equal(slugUniverse(FIRST, SECOND), 7);
  assert.equal(slugUniverse(FIRST, FIRST), 6);
  assert.equal(slugUniverse([], []), 0);
  assert.equal(slugUniverse(null, undefined), 0);
  assert.equal(slugUniverse([{ a: 1 }, { s: 'x' }], null), 1);
});

test('the comparison reports the subset AND the whole, so neither stands alone', () => {
  const result = compareScoreSets(FIRST, SECOND);
  assert.equal(result.n, 4);
  assert.equal(result.total, 7);
  assert.ok(result.n <= result.total);
});
