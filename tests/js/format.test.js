import test from 'node:test';
import assert from 'node:assert/strict';

import {
  NOT_SCORED, formatCount, formatPercent, formatScore, formatShare, isScored,
} from '../../site/js/format.js';

test('a score always shows one decimal', () => {
  assert.equal(formatScore(6), '6.0');
  assert.equal(formatScore(6.94), '6.9');
  assert.equal(formatScore(8.65), '8.7');
  assert.equal(formatScore(0), '0.0');
});

test('a missing score reads "Not scored", never a bare question mark', () => {
  for (const value of [null, undefined, NaN, '7', {}]) {
    assert.equal(formatScore(value), NOT_SCORED);
  }
  assert.equal(NOT_SCORED, 'Not scored');
});

test('isScored tells a page whether to style a value as absent', () => {
  assert.equal(isScored(0), true);
  assert.equal(isScored(null), false);
  assert.equal(isScored(Infinity), false);
});

test('counts get thousands separators', () => {
  assert.equal(formatCount(3039), '3,039');
  assert.equal(formatCount(13475), '13,475');
  assert.equal(formatCount(7), '7');
  assert.equal(formatCount(null), NOT_SCORED);
});

test('shares become percentage strings', () => {
  assert.equal(formatPercent(0.513), '51%');
  assert.equal(formatPercent(0.4959, 1), '49.6%');
  assert.equal(formatPercent(0), '0%');
  assert.equal(formatPercent(1), '100%');
  assert.equal(formatPercent(undefined), NOT_SCORED);
});

test('formatShare reads as a caption', () => {
  assert.equal(formatShare(1509, 3039), '1,509 of 3,039 jobs (50%)');
  assert.equal(formatShare(0, 0, 'skills'), '0 of 0 skills (0%)');
});
