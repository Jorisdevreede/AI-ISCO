import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  TIERS, editDistance, fold, nearestTitles, rankOccupations, suggestions,
} from '../../site/js/search.js';
import { INDEX } from './fixture.js';

const titles = (results) => results.map((r) => r.row.t);
const find = (results, slug) => results.find((r) => r.row.s === slug);

/** The published index, so the audit's own queries are checked against it. */
const REAL = JSON.parse(
  readFileSync(new URL('../../site/search_index.json', import.meta.url), 'utf8'),
);

test('an empty or whitespace query returns nothing', () => {
  for (const query of ['', '   ', '\t\n', null, undefined]) {
    assert.deepEqual(rankOccupations(INDEX, query), []);
  }
});

// A4: the landing page's own example query, and its featured nurse chip.

test('"programmer" reaches software developer through an alternative label', () => {
  const results = rankOccupations(INDEX, 'programmer');
  const hit = find(results, 'software-developer');
  assert.ok(hit, 'software developer must be in the results');
  assert.equal(hit.alt, 'programmer', 'the UI needs the label that matched');
  assert.equal(hit.tier, TIERS.ALT_EXACT);
});

test('an exact alternative label outranks a mid-title word match', () => {
  for (const index of [INDEX, REAL]) {
    const results = rankOccupations(index, 'programmer');
    assert.equal(results[0].row.s, 'software-developer',
      'the page says "for example programmer"; it must open the job that means');
    assert.equal(results[0].alt, 'programmer');
    assert.ok(titles(results).includes('venue programmer'), 'the word matches stay, below it');
  }
});

test('a job named after the query still beats an exact synonym', () => {
  const index = [
    { t: 'coder', s: 'coder', c: '2512', alt: [] },
    { t: 'x', s: 'x', c: '2513', alt: ['coder'] },
  ];
  assert.equal(rankOccupations(index, 'coder')[0].row.s, 'coder');
  assert.equal(rankOccupations(index, 'coder')[0].tier, TIERS.EXACT);
});

test('"nurse" ranks the job named nurse first, not the one assisting it', () => {
  for (const index of [INDEX, REAL]) {
    const found = titles(rankOccupations(index, 'nurse'));
    assert.equal(found[0], 'nurse responsible for general care',
      'the landing chip names this the nurse job; the ranking must agree');
    const nursery = found.indexOf('nursery school head teacher');
    const nurse = found.indexOf('nurse responsible for general care');
    assert.ok(nursery === -1 || nurse < nursery, 'whole-word nurse beats nursery');
  }
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

test('a tie breaks by ISCO major group before title length', () => {
  const index = [
    { t: 'nurse assistant', s: 'a', c: '3221', alt: [] },
    { t: 'nurse responsible for general care', s: 'n', c: '2221', alt: [] },
  ];
  assert.deepEqual(titles(rankOccupations(index, 'nurse')),
    ['nurse responsible for general care', 'nurse assistant']);
  // A row with no code sorts after one that has a code, never crashes.
  assert.equal(rankOccupations([{ t: 'nurse x', s: 'x' }, ...index], 'nurse').length, 3);
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
  assert.equal(nearest[0].s, 'bookkeeper');
  assert.deepEqual(nearestTitles(INDEX, '   ', 3), []);
});

// A5: the four queries the audit typed, against the published index.

test('a query with a word the site knows offers that word\'s jobs', () => {
  const found = suggestions(REAL, 'ICU nurse', 3);
  assert.equal(found.kind, 'word');
  assert.equal(found.word, 'nurse');
  assert.ok(found.rows.length > 0);
  const offered = found.rows.map((row) => row.t).join(' | ');
  assert.ok(/nurse/.test(offered), `offered ${offered}`);
  assert.ok(!/ICT buyer|^Judge|Tutor/.test(offered), `offered ${offered}`);
});

test('a typo is repaired against the word it was a typo of', () => {
  const found = suggestions(REAL, 'nrse', 3);
  assert.equal(found.kind, 'near');
  const offered = found.rows.map((row) => row.t);
  assert.ok(offered.every((title) => /nurse/.test(title)), `offered ${offered}`);
  assert.ok(!offered.some((title) => /chef|cook|baker/i.test(title)));
  // A dropped letter, on an index that has no label spelling it that way.
  assert.equal(suggestions(INDEX, 'bookeeper', 1).rows[0].s, 'bookkeeper');
});

test('a word the site does not know offers nothing rather than a guess', () => {
  const found = suggestions(REAL, 'verpleegkundige', 3);
  assert.notEqual(found.kind, 'near', 'edit distance must not reach across languages');
  assert.deepEqual(found.rows, []);
  // Two letters cannot be a typo of anything: no suggestions, not "make-up artist".
  assert.deepEqual(suggestions(REAL, 'GP', 3).rows, []);
});

test('"software engineer" reaches software developer, by match or by suggestion', () => {
  const direct = rankOccupations(REAL, 'software engineer', 12);
  if (direct.length) {
    assert.equal(direct[0].row.s, 'software-developer',
      'the index carries the label, so it must be the first result');
    assert.equal(direct[0].alt, 'software engineer');
    return;
  }
  // Until the index carries that label, the words of the query still lead there.
  const offered = suggestions(REAL, 'software engineer', 8).rows;
  assert.ok(offered.length, 'the query must lead somewhere');
  assert.ok(offered.every((row) => /software|engineer/i.test(row.t)),
    `offered ${offered.map((row) => row.t)}`);
});

test('suggestions never run away with an empty or unusable query', () => {
  for (const query of ['', '   ', null, undefined, '!!']) {
    assert.deepEqual(suggestions(INDEX, query, 3), { kind: 'none', word: null, rows: [] });
  }
  assert.deepEqual(suggestions(null, 'nurse', 3).rows, []);
});
