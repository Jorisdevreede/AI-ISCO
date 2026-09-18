import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  NEAREST_COUNT, POPULAR_CHIPS, RECENT_KEY, RECENT_MAX, RESULT_LIMIT, addRecent,
  loadingText, noMatchText, optionMeta, overflowText, readRecent, recentRows,
  removeRecent, rowBySlug, searchOutcome, sentenceCase, subtitleText, writeRecent,
} from '../../site/js/pages/landing-model.js';
import { fold } from '../../site/js/search.js';
import { INDEX } from './fixture.js';

const url = (name) => new URL(`../../site/${name}`, import.meta.url);
const REAL_INDEX = JSON.parse(readFileSync(url('search_index.json'), 'utf8'));
const REAL_STATS = JSON.parse(readFileSync(url('stats.json'), 'utf8'));

/** A stand-in for localStorage, and for one that refuses to work. */
function fakeStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => data.set(key, String(value)),
    dump: () => Object.fromEntries(data),
  };
}

const brokenStorage = {
  getItem() { throw new Error('access denied'); },
  setItem() { throw new Error('quota exceeded'); },
};

// --- the chips (audit B2) --------------------------------------------------

test('every popular chip names a job that really is in the index', () => {
  assert.equal(POPULAR_CHIPS.length, 8);
  for (const chip of POPULAR_CHIPS) {
    const row = rowBySlug(REAL_INDEX, chip.slug);
    assert.ok(row, `chip slug "${chip.slug}" is not in search_index.json`);
    assert.equal(
      fold(chip.label), fold(row.t),
      `chip "${chip.label}" opens "${row.t}" — a chip must never rename the job it opens`,
    );
  }
});

test('the chips are spread across ISCO major groups, and each slug appears once', () => {
  const slugs = POPULAR_CHIPS.map((chip) => chip.slug);
  assert.equal(new Set(slugs).size, slugs.length);
  const groups = POPULAR_CHIPS.map((chip) => rowBySlug(REAL_INDEX, chip.slug).mg);
  assert.ok(new Set(groups).size >= 4, `only ${new Set(groups).size} major groups among the chips`);
});

test('a chip slug is a slug, never a search term', () => {
  for (const chip of POPULAR_CHIPS) {
    assert.match(chip.slug, /^[a-z0-9-]+$/);
  }
});

// --- search states (audit D1) ----------------------------------------------

test('nothing typed is the empty state', () => {
  for (const query of ['', '   ', null, undefined]) {
    const outcome = searchOutcome(INDEX, query);
    assert.equal(outcome.state, 'empty');
    assert.deepEqual(outcome.results, []);
    assert.equal(outcome.total, 0);
  }
});

test('a handful of matches is the plain results state', () => {
  const outcome = searchOutcome(INDEX, 'nurse assistant');
  assert.equal(outcome.state, 'results');
  assert.equal(outcome.hidden, 0);
  assert.equal(outcome.results[0].row.s, 'nurse-assistant');
});

test('more matches than fit become the "many" state, counted honestly', () => {
  const outcome = searchOutcome(INDEX, 'nurse', 2);
  assert.equal(outcome.state, 'many');
  assert.equal(outcome.results.length, 2);
  assert.equal(outcome.total, 4);
  assert.equal(outcome.hidden, 2);
});

test('an unknown word is the no-match state with the nearest titles', () => {
  const outcome = searchOutcome(INDEX, 'bookeeper');
  assert.equal(outcome.state, 'no-match');
  assert.deepEqual(outcome.results, []);
  assert.equal(outcome.nearest.length, NEAREST_COUNT);
  assert.equal(outcome.nearest[0].s, 'bookkeeper');
});

test('the query is trimmed once, so the message quotes what the visitor typed', () => {
  assert.equal(searchOutcome(INDEX, '  zzzz  ').query, 'zzzz');
  assert.ok(noMatchText('zzzz').includes('“zzzz”'));
  assert.ok(!noMatchText('zzzz').toLowerCase().includes('0 results'));
});

test('the default limit is 12 and the real index honours it', () => {
  assert.equal(RESULT_LIMIT, 12);
  const outcome = searchOutcome(REAL_INDEX, 'engineer');
  assert.equal(outcome.results.length, 12);
  assert.equal(outcome.state, 'many');
  assert.equal(outcome.hidden, outcome.total - 12);
  assert.ok(overflowText(outcome.hidden, 'engineer').includes('keep typing'));
});

test('"programmer" reaches software developer, with the label that matched', () => {
  const outcome = searchOutcome(REAL_INDEX, 'programmer');
  const hit = outcome.results.find((result) => result.row.s === 'software-developer');
  assert.ok(hit, 'software developer must be in the results for "programmer"');
  assert.equal(hit.alt, 'programmer');
});

test('"nurse" does not put nursery school head teacher first', () => {
  const slugs = searchOutcome(REAL_INDEX, 'nurse').results.map((result) => result.row.s);
  assert.notEqual(slugs[0], 'nursery-school-head-teacher');
  const nursery = slugs.indexOf('nursery-school-head-teacher');
  const nurse = slugs.indexOf('nurse-responsible-for-general-care');
  assert.ok(nurse !== -1 && (nursery === -1 || nurse < nursery));
});

// --- copy ------------------------------------------------------------------

test('the subtitle takes its count from stats.json and never hard-codes one', () => {
  assert.ok(subtitleText(REAL_STATS).includes('3,043'));
  assert.ok(subtitleText({ occupations: 12 }).includes('12 European occupations'));
  assert.ok(!subtitleText({ occupations: 12 }).includes('3,043'));
  assert.ok(subtitleText(null).includes('European occupations'));
  assert.ok(/model estimates, not forecasts/.test(subtitleText(null)));
});

test('no page copy claims certainty the published data cannot support', () => {
  const copy = [subtitleText(REAL_STATS), loadingText(REAL_STATS), noMatchText('x'),
    overflowText(3, 'x')].join(' ');
  assert.ok(!/at risk/i.test(copy));
  assert.ok(!/57\.8|81\.8|0\.9[34]|\b671\b/.test(copy), 'private second-run figures must never appear');
});

test('the loading line counts the titles when stats are in, and still reads without them', () => {
  assert.equal(loadingText(REAL_STATS), 'Loading 3,043 job titles…');
  assert.equal(loadingText(null), 'Loading job titles…');
});

test('a result line names the group and the quadrant in words', () => {
  assert.equal(optionMeta({ mg: 'Professionals', q: 'TRANSFORM' }), 'Professionals · Transform');
  assert.equal(optionMeta({ mg: 'Professionals' }), 'Professionals · Not scored');
  assert.ok(!optionMeta({ mg: 'Professionals' }).includes('?'));
});

test('sentence case leaves an acronym alone', () => {
  assert.equal(sentenceCase('software developer'), 'Software developer');
  assert.equal(sentenceCase('ICT teacher secondary school'), 'ICT teacher secondary school');
  assert.equal(sentenceCase(''), '');
});

// --- recently viewed -------------------------------------------------------

test('the stored shape is a JSON array of slugs, most recent first', () => {
  const storage = fakeStorage();
  addRecent(storage, 'chef');
  addRecent(storage, 'lawyer');
  assert.equal(storage.dump()[RECENT_KEY], '["lawyer","chef"]');
  assert.deepEqual(readRecent(storage), ['lawyer', 'chef']);
});

test('viewing a job again moves it to the front instead of duplicating it', () => {
  const storage = fakeStorage();
  for (const slug of ['chef', 'lawyer', 'chef']) addRecent(storage, slug);
  assert.deepEqual(readRecent(storage), ['chef', 'lawyer']);
});

test('the list stops at five', () => {
  const storage = fakeStorage();
  for (const slug of ['a', 'b', 'c', 'd', 'e', 'f', 'g']) addRecent(storage, slug);
  assert.equal(RECENT_MAX, 5);
  assert.deepEqual(readRecent(storage), ['g', 'f', 'e', 'd', 'c']);
});

test('removing a slug leaves the rest in order', () => {
  const storage = fakeStorage({ [RECENT_KEY]: '["a","b","c"]' });
  assert.deepEqual(removeRecent(storage, 'b'), ['a', 'c']);
  assert.deepEqual(readRecent(storage), ['a', 'c']);
  assert.deepEqual(removeRecent(storage, 'missing'), ['a', 'c']);
});

test('a corrupted, foreign or over-long value reads as an empty or cleaned list', () => {
  assert.deepEqual(readRecent(fakeStorage({ [RECENT_KEY]: 'not json' })), []);
  assert.deepEqual(readRecent(fakeStorage({ [RECENT_KEY]: '{"a":1}' })), []);
  assert.deepEqual(readRecent(fakeStorage({ [RECENT_KEY]: '["a",3,"",null,"a","b"]' })), ['a', 'b']);
  assert.deepEqual(readRecent(fakeStorage()), []);
});

test('no storage at all is not an error: nothing persists, nothing throws', () => {
  for (const storage of [null, undefined, brokenStorage]) {
    assert.deepEqual(readRecent(storage), []);
    assert.deepEqual(addRecent(storage, 'chef'), ['chef']); // the list this page would show
    assert.deepEqual(readRecent(storage), [], 'but it did not persist');
    assert.deepEqual(removeRecent(storage, 'chef'), []);
    assert.deepEqual(writeRecent(storage, ['chef']), ['chef']);
  }
});

test('an empty slug is never stored', () => {
  const storage = fakeStorage();
  assert.deepEqual(addRecent(storage, '  '), []);
  assert.deepEqual(addRecent(storage, null), []);
});

test('stored slugs resolve to index rows, and unknown slugs are dropped', () => {
  const rows = recentRows(['chef', 'gone-from-the-index', 'lawyer'], REAL_INDEX);
  assert.deepEqual(rows.map((row) => row.s), ['chef', 'lawyer']);
  assert.deepEqual(recentRows(null, REAL_INDEX), []);
  assert.deepEqual(recentRows(['chef'], null), []);
});
