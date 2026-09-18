import test from 'node:test';
import assert from 'node:assert/strict';

import {
  NEAR_LINE, THRESHOLD, distanceToLine, explainQuadrant, exposureWord, isNearLine,
  quadrantOf,
} from '../../site/js/quadrant.js';

test('the cut-off is a hard 6 on both axes', () => {
  assert.equal(THRESHOLD, 6);
  assert.equal(NEAR_LINE, 0.5);
});

test('the quadrant flips exactly at the cut-off', () => {
  assert.equal(quadrantOf(5.9, 7.0), 'EVOLVE');
  assert.equal(quadrantOf(6.0, 7.0), 'TRANSFORM');
  assert.equal(quadrantOf(6.1, 7.0), 'TRANSFORM');
  assert.equal(quadrantOf(6.0, 5.9), 'SHRINK');
  assert.equal(quadrantOf(6.0, 6.0), 'TRANSFORM');
  assert.equal(quadrantOf(5.9, 5.9), 'STABLE');
});

test('an unscored pair has no quadrant', () => {
  assert.equal(quadrantOf(null, 7), null);
  assert.equal(quadrantOf(7, undefined), null);
  assert.equal(quadrantOf(NaN, 7), null);
});

test('distanceToLine measures the closer axis and is never negative', () => {
  assert.equal(distanceToLine(6.9, 7.0), 0.9);
  assert.equal(distanceToLine(2.0, 6.2), 0.2);
  assert.equal(distanceToLine(6.0, 6.0), 0);
  assert.equal(distanceToLine(null, 6), null);
});

test('near the line is 0.5 on either axis, inclusive', () => {
  assert.equal(isNearLine(5.5, 1.0), true);
  assert.equal(isNearLine(6.5, 1.0), true);
  assert.equal(isNearLine(5.4, 1.0), false);
  assert.equal(isNearLine(6.6, 1.0), false);
  assert.equal(isNearLine(1.0, 5.5), true);
  assert.equal(isNearLine(1.0, 6.5), true);
  assert.equal(isNearLine(1.0, 1.0), false);
  assert.equal(isNearLine(null, 6.0), false);
});

test('exposure bands are low under 4, medium under 7, high from 7', () => {
  assert.equal(exposureWord(3.9), 'low');
  assert.equal(exposureWord(4.0), 'medium');
  assert.equal(exposureWord(6.9), 'medium');
  assert.equal(exposureWord(7.0), 'high');
  assert.equal(exposureWord(10), 'high');
  assert.equal(exposureWord(null), null);
});

test('explainQuadrant names the box, the cut-off and the distance', () => {
  const explanation = explainQuadrant({
    t: 'software developer', a: 6.9, m: 7.0, q: 'TRANSFORM',
  });
  assert.equal(explanation.heading, 'Why "Transform"?');
  assert.equal(explanation.nearLine, false);
  const text = explanation.sentences.join(' ');
  assert.match(text, /6\.9/);
  assert.match(text, /cut-off of 6/);
  assert.match(text, /0\.9 from the nearest cut-off/);
  assert.match(text, /model estimates/);
});

test('explainQuadrant flags a job sitting near the line', () => {
  const explanation = explainQuadrant({ t: 'nurse assistant', a: 5.5, m: 5.9, q: 'STABLE' });
  assert.equal(explanation.nearLine, true);
  assert.match(explanation.sentences.join(' '), /near the line/);
});

test('explainQuadrant never mentions a second scoring run', () => {
  const text = explainQuadrant({ t: 'bookkeeper', a: 8.6, m: 5.2, q: 'SHRINK' })
    .sentences.join(' ').toLowerCase();
  for (const forbidden of ['scoring run', 'second run', 'both times', 'agreement',
    'typesafe', 'gemini', '57.8', '4 in 10']) {
    assert.ok(!text.includes(forbidden), `must not mention "${forbidden}"`);
  }
});

test('an unscored job gets an honest popover rather than a box', () => {
  const explanation = explainQuadrant({ t: 'mystery job', a: null, m: null });
  assert.equal(explanation.heading, 'Not scored');
  assert.equal(explanation.nearLine, false);
});
