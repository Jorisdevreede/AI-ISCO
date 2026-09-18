import test from 'node:test';
import assert from 'node:assert/strict';

import {
  TIERS, editDistance, fold, nearestTitles, rankOccupations,
} from '../../site/js/search.js';
import { INDEX } from './fixture.js';

const titles = (results) => results.map((r) => r.row.t);
const find = (results, slug) => results.find((r) => r.row.s === slug);

test('an empty or whitespace query returns nothing', () => {
  for (const query of ['', '   ', '\t\n', null, undefined]) {
    assert.deepEqual(rankOccupations(INDEX, query), []);
  }
});

test('"programmer" reaches software developer through an alternative label', () => {
  const results = rankOccupations(INDEX, 'programmer');
  const hit = find(results, 'software-developer');
  assert.ok(hit, 'software developer must be in the results');
  assert.equal(hit.alt, 'programmer', 'the UI needs the label that matched');
  assert.equal(hit.tier, TIERS.ALT);
});

test('a title match still outranks an alternative-label match', () => {
  const results = rankOccupations(INDEX, 'programmer');
  assert.equal(results[0].row.s, 'venue-programmer');
  assert.equal(results[0].alt, null);
});

test('"nurse" ranks a job called nurse above nursery school head teacher', () => {
  const found = titles(rankOccupations(INDEX, 'nurse'));
  assert.equal(found[0], 'nurse assistant');
  const nursery = found.indexOf('nursery school head teacher');
  const nurse = found.indexOf('nurse responsible for general care');
  assert.ok(nurse < nursery, `whole-word nurse (${nurse}) must beat nursery (${nursery})`);
});

test('an exact title beats a longer title with the same prefix', () => {
  const results = rankOccupations(INDEX, 'nurse assistant');
  assert.equal(results[0].row.s, 'nurse-assistant');
  assert.equal(results[0].tier, TIERS.EXACT);
  assert.equal(results[1].row.s, 'nurse-assistant-supervisor');
});

test('matching is case insensitive', () => {
  assert.deepEqual(
    titles(rankOccupations(INDEX, 'NURSE Assistant')),
    titles(rankOccupations(INDEX, 'nurse assistant')),
  );
});

test('matching is diacritic insensitive, both ways round', () => {
  assert.equal(rankOccupations(INDEX, 'sommeliere')[0].row.s, 'sommeliere');
  assert.equal(rankOccupations(INDEX, 'sommelière')[0].row.s, 'sommeliere');
  assert.equal(fold('Sommelière'), 'sommeliere');
});

test('a word inside the title matches, below a title prefix', () => {
  const results = rankOccupations(INDEX, 'software');
  assert.equal(results[0].row.s, 'software-developer');
  assert.ok(find(results, 'embedded-systems-software-developer'));
  assert.ok(results[0].tier < find(results, 'embedded-systems-software-developer').tier);
});

test('ties break by title length and then alphabetically', () => {
  const index = [
    { t: 'delta engineer', s: 'd', alt: [] },
    { t: 'alpha engineer', s: 'a', alt: [] },
    { t: 'engineer', s: 'e', alt: [] },
    { t: 'charlie engineering lead', s: 'c', alt: [] },
  ];
  assert.deepEqual(titles(rankOccupations(index, 'engineer')), [
    'engineer', 'alpha engineer', 'delta engineer', 'charlie engineering lead',
  ]);
});

test('the same input always gives the same order', () => {
  const once = titles(rankOccupations(INDEX, 'nurse', Infinity));
  const twice = titles(rankOccupations([...INDEX].reverse(), 'nurse', Infinity));
  assert.deepEqual(once, twice);
});

test('the limit caps the results and defaults to 12', () => {
  assert.equal(rankOccupations(INDEX, 'e', 3).length, 3);
  assert.ok(rankOccupations(INDEX, 'e').length <= 12);
  assert.ok(rankOccupations(INDEX, 'e', Infinity).length >= 4);
});

test('the best alternative label wins when several match', () => {
  const index = [{ t: 'x', s: 'x', alt: ['deep sea coder', 'coder'] }];
  assert.equal(rankOccupations(index, 'coder')[0].alt, 'coder');
});

test('a missing alt array is fine', () => {
  assert.deepEqual(rankOccupations([{ t: 'plumber', s: 'p' }], 'plumb').length, 1);
  assert.deepEqual(rankOccupations(null, 'plumb'), []);
});

test('editDistance is symmetric and counts single edits', () => {
  assert.equal(editDistance('nurse', 'nurse'), 0);
  assert.equal(editDistance('nurse', 'nurses'), 1);
  assert.equal(editDistance('nurse', 'purse'), 1);
  assert.equal(editDistance('', 'nurse'), 5);
  assert.equal(editDistance('nurse', ''), 5);
  assert.equal(editDistance('kitten', 'sitting'), editDistance('sitting', 'kitten'));
});

test('nearestTitles offers the closest jobs when nothing matched', () => {
  const nearest = nearestTitles(INDEX, 'bookeeper', 3);
  assert.equal(nearest.length, 3);
  assert.equal(nearest[0].s, 'bookkeeper');
  assert.deepEqual(nearestTitles(INDEX, '   ', 3), []);
});
