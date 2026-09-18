import test from 'node:test';
import assert from 'node:assert/strict';

import {
  NO_MATCH_HINT, SKILL_TIERS, coOccurringSkills, neededBy, occupationSlugs, rankSkills,
  resolveOccupations, usedIn,
} from '../../site/js/pages/skill-model.js';
import { INDEX } from './fixture.js';

// A stand-in for skill_index.json, shaped exactly like the real rows.
const SKILLS = [
  { id: 'aaaa1111', t: 'manage budgets', a: 5.0, m: 8.0, ne: 2, no: 1 },
  { id: 'bbbb2222', t: 'manage staff', a: 3.0, m: 7.0, ne: 2, no: 0 },
  { id: 'cccc3333', t: 'budget management principles', a: 4.0, m: 6.0, ne: 1, no: 1 },
  { id: 'dddd4444', t: 'prepare the annual budget', a: 6.0, m: 7.0, ne: 1, no: 0 },
  { id: 'eeee5555', t: 'troubleshoot', a: 6.0, m: 8.0, ne: 1, no: 2 },
];

// A stand-in for skill_occupations.json over the fixture's slugs.
const OCCUPATIONS = {
  aaaa1111: { e: ['software-developer', 'bookkeeper'], o: ['venue-programmer'] },
  bbbb2222: { e: ['software-developer', 'bookkeeper'], o: [] },
  cccc3333: { e: ['bookkeeper'], o: ['sommeliere'] },
  dddd4444: { e: ['bookkeeper'], o: [] },
  eeee5555: { e: ['venue-programmer'], o: ['software-developer', 'nurse-assistant'] },
};

const bySlug = new Map(INDEX.map((row) => [row.s, row]));
const titles = (results) => results.map((result) => result.row.t);

test('an exact title beats a prefix, and a prefix beats a word inside', () => {
  const results = rankSkills(SKILLS, 'manage budgets');
  assert.equal(results[0].row.id, 'aaaa1111');
  assert.equal(results[0].tier, SKILL_TIERS.EXACT);
});

test('the tiers rank in order for one query', () => {
  const results = rankSkills(SKILLS, 'budget');
  assert.deepEqual(titles(results), [
    'budget management principles', // title prefix
    'manage budgets', // word prefix, shorter title wins the tie
    'prepare the annual budget', // word prefix
  ]);
  assert.equal(results[0].tier, SKILL_TIERS.PREFIX);
  assert.equal(results[1].tier, SKILL_TIERS.WORD_PREFIX);
  assert.equal(results[2].tier, SKILL_TIERS.WORD_PREFIX);
});

test('a substring match still matches, last', () => {
  const results = rankSkills(SKILLS, 'udget');
  assert.equal(results.every((result) => result.tier === SKILL_TIERS.SUBSTRING), true);
  assert.equal(results.length, 3);
});

test('shorter titles win a tie, then alphabet, so the order is stable', () => {
  const results = rankSkills(SKILLS, 'manage');
  assert.deepEqual(
    titles(results),
    ['manage staff', 'manage budgets', 'budget management principles'],
  );
  assert.deepEqual(titles(rankSkills(SKILLS, 'MANAGE')), titles(results));
});

test('an empty or unmatched query returns nothing, and the hint teaches ESCO wording', () => {
  assert.deepEqual(rankSkills(SKILLS, '   '), []);
  assert.deepEqual(rankSkills(SKILLS, 'budgeting'), []);
  assert.match(NO_MATCH_HINT, /ESCO wording/);
  assert.match(NO_MATCH_HINT, /manage budgets/);
});

test('the limit caps the list', () => {
  assert.equal(rankSkills(SKILLS, 'a', 2).length, 2);
  assert.equal(rankSkills(SKILLS, 'a', Infinity).length, 4); // "troubleshoot" has no "a"
});

test('used-in counts both relations and survives missing fields', () => {
  assert.equal(usedIn(SKILLS[0]), 3);
  assert.equal(usedIn({ ne: 4 }), 4);
  assert.equal(usedIn(undefined), 0);
});

test('occupation slugs keep essential first and de-duplicate', () => {
  assert.deepEqual(occupationSlugs({ e: ['a', 'b'], o: ['b', 'c'] }), ['a', 'b', 'c']);
  assert.deepEqual(occupationSlugs(undefined), []);
});

test('co-occurring skills rank by shared occupations and exclude the skill itself', () => {
  const related = coOccurringSkills(OCCUPATIONS, 'aaaa1111', { limit: 3 });
  assert.deepEqual(related, [
    { id: 'bbbb2222', n: 2 }, // software-developer + bookkeeper
    { id: 'eeee5555', n: 2 }, // venue-programmer + software-developer
    { id: 'cccc3333', n: 1 },
  ]);
  assert.equal(related.some((item) => item.id === 'aaaa1111'), false);
});

test('capping the occupations caps the work without changing the shape', () => {
  const capped = coOccurringSkills(OCCUPATIONS, 'aaaa1111', { limit: 8, maxOccupations: 1 });
  assert.deepEqual(capped, [{ id: 'bbbb2222', n: 1 }, { id: 'eeee5555', n: 1 }]);
});

test('an unknown skill has no neighbours rather than throwing', () => {
  assert.deepEqual(coOccurringSkills(OCCUPATIONS, 'no-such-id'), []);
  assert.deepEqual(coOccurringSkills({}, 'aaaa1111'), []);
});

test('slugs resolve to index rows, and an unknown slug still yields a row', () => {
  const rows = resolveOccupations(['bookkeeper', 'ghost-slug'], bySlug);
  assert.equal(rows[0].t, 'bookkeeper');
  assert.equal(rows[0].a, 8.6);
  assert.deepEqual(
    { t: rows[1].t, s: rows[1].s, a: rows[1].a },
    { t: 'ghost-slug', s: 'ghost-slug', a: null },
  );
});

test('needed-by splits essential from optional', () => {
  const lists = neededBy(OCCUPATIONS.aaaa1111, bySlug);
  assert.deepEqual(lists.essential.map((row) => row.s), ['software-developer', 'bookkeeper']);
  assert.deepEqual(lists.optional.map((row) => row.s), ['venue-programmer']);
  assert.deepEqual(neededBy(undefined, bySlug), { essential: [], optional: [] });
});
