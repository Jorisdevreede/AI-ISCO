import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SHARE_ORDER, isShares, largestShare, shareAriaLabel, sharePercents, shareSegments,
  shareSentence,
} from '../../site/js/shares.js';

const percents = (shares) => sharePercents(shares).map((part) => part.percent);

test('the four parts come back in class order', () => {
  const parts = sharePercents([0.42, 0.31, 0.05, 0.22]);
  assert.deepEqual(parts.map((part) => part.code), SHARE_ORDER);
  assert.deepEqual(parts.map((part) => part.percent), [42, 31, 5, 22]);
});

test('rounded percentages always add up to 100', () => {
  const cases = [
    [0.42, 0.31, 0.05, 0.22],
    [1 / 3, 1 / 3, 1 / 3, 0],
    [0.005, 0.005, 0.495, 0.495],
    [0.166, 0.167, 0.333, 0.334],
    [0.125, 0.125, 0.125, 0.625],
    [1, 0, 0, 0],
  ];
  for (const shares of cases) {
    assert.equal(percents(shares).reduce((sum, n) => sum + n, 0), 100,
      `${shares.join('/')} did not add up`);
  }
});

test('the remainder goes to the largest fractions, not to the first part', () => {
  // 33.3 / 33.3 / 33.3 / 0 leaves one point over; the three tie, so it goes to
  // the earliest of them and the order of the four parts never changes.
  assert.deepEqual(percents([1 / 3, 1 / 3, 1 / 3, 0]), [34, 33, 33, 0]);
  assert.deepEqual(percents([0.126, 0.249, 0.375, 0.25]), [13, 25, 37, 25]);
});

test('the sentence reads as the spec writes it', () => {
  assert.equal(shareSentence([0.42, 0.31, 0.05, 0.22]),
    'AI can take over 42% · AI assists 31% · machines 5% · stays human 22%');
});

test('the accessible name names the job and ends in a full stop', () => {
  assert.equal(shareAriaLabel([0.5, 0.25, 0.1, 0.15], 'bookkeeper'),
    'What bookkeeper is made of: AI can take over 50% · AI assists 25% · '
    + 'machines 10% · stays human 15%.');
  assert.match(shareAriaLabel([0.5, 0.25, 0.1, 0.15]), /^What this job is made of/);
});

test('segments skip the parts that would be invisible', () => {
  const segments = shareSegments([0.6, 0.4, 0, 0]);
  assert.deepEqual(segments.map((segment) => segment.code), ['S', 'A']);
  assert.deepEqual(segments.map((segment) => segment.width), ['60%', '40%']);
});

test('the largest part is reported with its label', () => {
  assert.equal(largestShare([0.1, 0.2, 0.05, 0.65]).code, 'I');
  assert.equal(largestShare([0.1, 0.2, 0.05, 0.65]).label, 'stays human');
  assert.equal(largestShare(null), null);
});

test('a broken shares array degrades to nothing, never to NaN', () => {
  for (const bad of [null, undefined, [], [0.5, 0.5], [0.9, 0.9, 0.9, 0.9],
    [0.5, 0.5, null, 0], 'x']) {
    assert.equal(isShares(bad), false);
    assert.deepEqual(sharePercents(bad), []);
    assert.equal(shareSentence(bad), '');
    assert.equal(shareAriaLabel(bad), 'Shares not scored');
  }
});

test('a rounding drift of a point or two is still a valid shares array', () => {
  assert.equal(isShares([0.42, 0.31, 0.05, 0.23]), true);
  assert.equal(isShares([0.4, 0.3, 0.05, 0.2]), false);
});
