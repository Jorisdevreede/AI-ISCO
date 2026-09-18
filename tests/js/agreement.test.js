import test from 'node:test';
import assert from 'node:assert/strict';

import {
  averageRanks, compareScoreSets, largestMove, pairRows, quadrantMatrix, spearman,
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
