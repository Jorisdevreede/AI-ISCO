import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ITEM_NAMES, MODE_NAMES, NO_MATCH_HINT, SKILL_TIERS, answerSections, askedAsKnowledge,
  classFilterOptions, classMeaning, coOccurringSkills, confidenceWord, countNoun,
  defaultSort, filterOptions, filterSkills, listColumns, listParams, lookupIntro, neededBy,
  neededNote, occupationScores, occupationSlugs, optionMeta, pageOf, pageSummary,
  parseListState, probabilityParts, probabilitySentence, rankSkills, resolveOccupations,
  rubricClasses, rubricIntro, rubricQuestions, scoreAxes, scoreBand, schemeOfSkillSet,
  scoresHeading, scoresNote, shardOf, sortListRows, sortOptions, sortSkillOccupations,
  usedIn,
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

test('two skills with the same title keep the order the index gave them', () => {
  // ESCO has near-duplicate wordings, so two rows can tie on tier and on length
  // and then on the title itself. With nothing left to decide it, the sort must
  // leave the pair where it found it rather than shuffle it.
  const duplicates = [
    { id: 'aaaa0001', t: 'manage budgets', a: 5.0, m: 8.0, ne: 1, no: 0 },
    { id: 'bbbb0002', t: 'manage staff', a: 3.0, m: 6.0, ne: 1, no: 0 },
    { id: 'cccc0003', t: 'manage budgets', a: 7.0, m: 4.0, ne: 1, no: 0 },
  ];
  const ids = (rows) => rankSkills(rows, 'manage').map((result) => result.row.id);
  assert.deepEqual(ids(duplicates), ['bbbb0002', 'aaaa0001', 'cccc0003']);
  assert.deepEqual(ids([...duplicates].reverse()), ['bbbb0002', 'cccc0003', 'aaaa0001']);
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

// --- the shares scheme -------------------------------------------------------
//
// Stand-ins for a `_v2` set: a skill row carries a class `c`, a third score `k`
// and the probabilities `p`; an occupation row carries `sh`, `nl` and a type in
// `q`. Every number below is invented for this test.
const SHARES = 'shares';

const SHARES_SKILLS = [
  { id: 'aaaa1111', t: 'manage budgets', a: 5.0, m: 8.0, k: 1.0, ne: 2, no: 1, c: 'A', p: [0.21, 0.74, 0.02] },
  { id: 'bbbb2222', t: 'manage staff', a: 3.0, m: 7.0, k: 1.0, ne: 2, no: 0, c: 'I', p: [0.05, 0.31, 0.01] },
];

const SHARES_INDEX = [
  {
    t: 'copy editor', s: 'copy-editor', c: '2642', a: 8.1, m: 7.0, k: 1.0,
    q: 'AUTOMATION_HEAVY', sh: [0.62, 0.18, 0.00, 0.20], nl: true,
  },
  {
    t: 'content manager', s: 'content-manager', c: '2431', a: 4.4, m: 8.2, k: 1.1,
    q: 'AUGMENTED', sh: [0.15, 0.50, 0.02, 0.33], nl: false,
  },
  {
    t: 'welder', s: 'welder', c: '7212', a: 2.0, m: 3.0, k: 6.0,
    q: 'MECHANISABLE', sh: [0.01, 0.06, 0.30, 0.63], nl: false,
  },
  {
    t: 'archivist', s: 'archivist', c: '2621', a: 5.0, m: 5.0, k: 2.0,
    q: 'MIXED', sh: null, nl: false,
  },
];

const sortedTitles = (key) => sortSkillOccupations(SHARES_INDEX, key, SHARES)
  .map((row) => row.t);

test('the scheme comes from the stats file, and from the rows when that is missing', () => {
  assert.equal(schemeOfSkillSet({ scheme: 'shares' }, SKILLS), 'shares');
  assert.equal(schemeOfSkillSet({ threshold: 6 }, SHARES_SKILLS), 'quadrants');
  assert.equal(schemeOfSkillSet(null, SHARES_SKILLS), 'shares');
  assert.equal(schemeOfSkillSet(null, SKILLS), 'quadrants');
  assert.equal(schemeOfSkillSet(null, null), 'quadrants');
});

test('a shares set has three measures, each named for what it measures', () => {
  assert.deepEqual(scoreAxes(SHARES).map((axis) => [axis.key, axis.label]),
    [['a', 'AI substitution'], ['m', 'AI assistance'], ['k', 'Machine automation']]);
  assert.deepEqual(scoreAxes().map((axis) => axis.label),
    ['Automation risk', 'Amplification']);
  assert.equal(scoresHeading(SHARES), 'What AI can do with this skill');
  assert.equal(scoresHeading(), 'Both scores');
});

test('the note under the bars explains three measures and disowns the class', () => {
  const note = scoresNote(SHARES);
  for (const name of ['AI substitution', 'AI assistance', 'Machine automation']) {
    assert.ok(note.includes(name), name);
  }
  assert.match(note, /none of them decides the class above/);
  assert.doesNotMatch(note, /quadrant|cut-off|\bbox(es)?\b/i);
  assert.match(scoresNote(), /Automation risk is how much of this skill/);
});

test('"Stays human" is never read as "AI is of no help here"', () => {
  assert.match(classMeaning('I'), /AI can still help with parts of it/);
  // C14: said the way round a reader can act on, with no double negative.
  assert.doesNotMatch(classMeaning('I'), /not the same as|no help/);
  assert.match(classMeaning('A'), /clear gain across most of it/);
  assert.match(classMeaning('S'), /nearly all of this work by itself/);
  assert.match(classMeaning('M'), /physical equipment/);
  assert.equal(classMeaning('X'), '');
  assert.equal(classMeaning(null), '');
});

test('the probabilities behind a class are quoted as the model’s own estimates', () => {
  const parts = probabilityParts([0.21, 0.74, 0.02]);
  assert.deepEqual(parts.map((part) => part.percent), ['21%', '74%', '2%']);
  const sentence = probabilitySentence([0.21, 0.74, 0.02]);
  assert.match(sentence, /the model puts 21% on an AI system being able to do nearly all/);
  assert.match(sentence, /74% on the person keeping it and getting a clear gain/);
  assert.match(sentence, /2% on physical equipment doing nearly all of it/);
  assert.match(sentence, /more likely than not/);
  assert.match(sentence, /not\s+measurements of any workplace/);
  assert.doesNotMatch(sentence, /\b6\b|cut-off/);
  assert.deepEqual(probabilityParts(undefined), []);
  assert.equal(probabilitySentence([0.2, 0.3]), '');
});

test('the sort control gains the three shares sorts, and keeps the old ones', () => {
  assert.deepEqual(sortOptions(SHARES).map((sort) => sort.label),
    ['Most can be taken over', 'Most assisted', 'Most mechanical', 'A to Z']);
  assert.deepEqual(sortOptions().map((sort) => sort.label),
    ['Most automation-exposed', 'Most amplified', 'A to Z']);
  assert.equal(defaultSort(SHARES), 'substituted');
  assert.equal(defaultSort(), 'automation');
});

test('each shares sort orders by its own share, unscored rows last', () => {
  assert.deepEqual(sortedTitles('substituted'),
    ['copy editor', 'content manager', 'welder', 'archivist']);
  assert.deepEqual(sortedTitles('assisted'),
    ['content manager', 'copy editor', 'welder', 'archivist']);
  assert.deepEqual(sortedTitles('mechanical'),
    ['welder', 'archivist', 'content manager', 'copy editor']);
  assert.deepEqual(sortedTitles('title'),
    ['archivist', 'content manager', 'copy editor', 'welder']);
  // An unknown key falls back rather than throwing, and nothing is mutated.
  assert.deepEqual(sortedTitles('nonsense'), sortedTitles('substituted'));
  assert.equal(SHARES_INDEX[0].t, 'copy editor');
});

test('the quadrant sort is still the shared one', () => {
  assert.deepEqual(sortSkillOccupations(INDEX, 'automation').map((row) => row.t)[0],
    'bookkeeper');
});

test('an occupation row shows its type and its two telling shares', () => {
  const { separator, parts } = occupationScores(SHARES_INDEX[0], SHARES);
  assert.equal(separator, ' · ');
  assert.deepEqual(parts.map((part) => part.text), ['Automation-heavy', '62%', '18%']);
  assert.deepEqual(parts.map((part) => part.label),
    ['type ', 'AI can take over ', 'AI assists ']);
  // A row with no shares says so rather than printing NaN.
  const unscored = occupationScores(SHARES_INDEX[3], SHARES);
  assert.deepEqual(unscored.parts.map((part) => part.text), ['Mixed', 'Not scored']);
});

test('the quadrant row is the pair of scores it always was', () => {
  const { separator, parts } = occupationScores(INDEX[0]);
  assert.equal(separator, ' / ');
  assert.deepEqual(parts.map((part) => part.text), ['6.4', '8.9']);
  assert.deepEqual(parts.map((part) => part.label), ['automation ', 'amplification ']);
});

test('the search box shows a class under shares and two scores under quadrants', () => {
  assert.equal(optionMeta(SHARES_SKILLS[0], SHARES), 'AI assists');
  assert.equal(optionMeta(SHARES_SKILLS[1], SHARES), 'Stays human');
  assert.equal(optionMeta({ c: 'zzz' }, SHARES), 'Not scored');
  assert.equal(optionMeta(SKILLS[0]), '5.0 / 8.0');
});

test('the empty state and the needed-by note say what the active set measures', () => {
  assert.match(lookupIntro(SHARES), /how much of it an AI system could do by itself/);
  assert.match(lookupIntro(), /scored on two\s+axes/);
  assert.match(neededNote(SHARES), /its type, then how much of it AI can take over/);
  assert.match(neededNote(), /automation \/ amplification/);
  for (const text of [lookupIntro(SHARES), neededNote(SHARES)]) {
    assert.doesNotMatch(text, /quadrant|cut-off|\bbox(es)?\b/i);
  }
});

// --- the table of every scored skill -----------------------------------------
//
// Stand-ins for the `_v2` skill index with the fields the table filters on.
// Every number, class, mode and kind below is invented for this test.
const LIST_ROWS = [
  { id: 'aaaa1111', t: 'manage budgets', a: 5.0, m: 8.0, k: 1.0, ne: 2, no: 1, c: 'A', ty: 's', mo: 's' },
  { id: 'bbbb2222', t: 'manage staff', a: 3.0, m: 7.0, k: 1.2, ne: 2, no: 0, c: 'I', ty: 's', mo: 'p' },
  { id: 'cccc3333', t: 'budget management principles', a: 4.0, m: 6.0, k: 2.0, ne: 1, no: 1, c: 'A', ty: 'k', mo: 's' },
  { id: 'dddd4444', t: 'weld steel', a: 2.0, m: 3.0, k: 7.5, ne: 1, no: 0, c: 'M', ty: 's', mo: 't' },
  { id: 'eeee5555', t: 'write copy', a: 9.0, m: 8.0, k: 1.0, ne: 1, no: 2, c: 'S', ty: 's', mo: 's' },
];

const shareColumns = () => listColumns(SHARES, LIST_ROWS);
const keysOf = (columns) => columns.map((column) => column.key);
const state = (params) => parseListState(params, shareColumns());
const titlesOf = (rows) => rows.map((row) => row.t);

test('the table has the columns its set can fill, and no more', () => {
  assert.deepEqual(keysOf(shareColumns()), ['t', 'c', 'a', 'm', 'k', 'mo', 'ty', 'n']);
  assert.deepEqual(shareColumns().map((column) => column.label), [
    'Skill', 'What AI can do with it', 'AI substitution', 'AI assistance',
    'Machine automation', 'How it is exercised', 'Skill or knowledge', 'Jobs that need it',
  ]);
  // A shares set built before `mo` and `ty` existed drops those two columns.
  const bare = LIST_ROWS.map(({ mo, ty, ...rest }) => rest);
  assert.deepEqual(keysOf(listColumns(SHARES, bare)), ['t', 'c', 'a', 'm', 'k', 'n']);
  assert.deepEqual(keysOf(listColumns(undefined, LIST_ROWS)), ['t', 'a', 'm', 'n']);
  assert.deepEqual(listColumns().map((column) => column.label),
    ['Skill', 'Automation risk', 'Amplification', 'Jobs that need it']);
});

test('a cell prints a name, never a code', () => {
  const by = Object.fromEntries(shareColumns().map((column) => [column.key, column]));
  const row = LIST_ROWS[2];
  assert.equal(by.c.text(row), 'AI assists');
  assert.equal(by.mo.text(row), 'Through software');
  assert.equal(by.ty.text(row), 'Knowledge item');
  assert.equal(by.n.text(row), '2');
  assert.equal(by.a.text(row), '4.0');
  assert.equal(by.mo.text({ mo: 'zz' }), 'Not scored');
});

test('the list state is read off the hash and checked against the data', () => {
  assert.deepEqual(state({}), {
    query: '', cls: '', item: '', mode: '', sort: 't', dir: 'asc', page: 1,
  });
  assert.deepEqual(state({ class: 'S', sort: 'a', page: '3' }), {
    query: '', cls: 'S', item: '', mode: '', sort: 'a', dir: 'desc', page: 3,
  });
  // Anything the data cannot mean is dropped rather than shown as an empty table.
  assert.deepEqual(state({ class: 'Z', item: 'x', mode: 'q', sort: 'zz', page: '-2' }), {
    query: '', cls: '', item: '', mode: '', sort: 't', dir: 'asc', page: 1,
  });
  assert.equal(state({ sort: 'a', dir: 'asc' }).dir, 'asc');
});

test('a view round-trips through the hash, and a plain one carries nothing', () => {
  const columns = shareColumns();
  assert.deepEqual(listParams(state({}), columns), {});
  const shared = { q: 'weld', class: 'M', item: 's', mode: 't', sort: 'k', page: '2' };
  assert.deepEqual(listParams(state(shared), columns), shared);
  // The natural direction of a column is not worth a parameter.
  assert.deepEqual(listParams(state({ sort: 'a' }), columns), { sort: 'a' });
  assert.deepEqual(listParams(state({ sort: 'a', dir: 'asc' }), columns),
    { sort: 'a', dir: 'asc' });
});

test('every filter narrows the rows, and they combine', () => {
  assert.deepEqual(titlesOf(filterSkills(LIST_ROWS, state({}))), titlesOf(LIST_ROWS));
  assert.deepEqual(titlesOf(filterSkills(LIST_ROWS, state({ class: 'A' }))),
    ['manage budgets', 'budget management principles']);
  assert.deepEqual(titlesOf(filterSkills(LIST_ROWS, state({ item: 'k' }))),
    ['budget management principles']);
  assert.equal(filterSkills(LIST_ROWS, state({ mode: 's' })).length, 3);
  assert.deepEqual(titlesOf(filterSkills(LIST_ROWS, { ...state({}), query: 'BUDGET' })),
    ['manage budgets', 'budget management principles']);
  assert.deepEqual(
    filterSkills(LIST_ROWS, { ...state({ class: 'A' }), query: 'principles' }).length, 1,
  );
});

test('any column sorts, either way, and ties break by title', () => {
  const columns = shareColumns();
  assert.deepEqual(titlesOf(sortListRows(LIST_ROWS, columns, state({ sort: 'k' })))[0],
    'weld steel');
  assert.deepEqual(titlesOf(sortListRows(LIST_ROWS, columns, state({ sort: 't' }))), [
    'budget management principles', 'manage budgets', 'manage staff', 'weld steel',
    'write copy',
  ]);
  const down = sortListRows(LIST_ROWS, columns, state({ sort: 't', dir: 'desc' }));
  assert.deepEqual(titlesOf(down)[0], 'write copy');
  // Two rows share a mechanical score of 1.0; the title settles it, both ways.
  const tied = sortListRows(LIST_ROWS, columns, state({ sort: 'k', dir: 'asc' }));
  assert.deepEqual(titlesOf(tied).slice(0, 2), ['manage budgets', 'write copy']);
  assert.equal(LIST_ROWS[0].t, 'manage budgets', 'the caller’s array is untouched');
});

test('a page is one page, and the line above it counts the whole list', () => {
  const view = pageOf(LIST_ROWS, 1, 2);
  assert.deepEqual(titlesOf(view.rows), ['manage budgets', 'manage staff']);
  assert.deepEqual([view.page, view.pages, view.from, view.to, view.total], [1, 3, 1, 2, 5]);
  assert.equal(pageSummary(view), 'Showing 1–2 of 5 skills');
  // A page past the end lands on the last one rather than on an empty table.
  assert.equal(pageOf(LIST_ROWS, 99, 2).page, 3);
  assert.deepEqual(pageOf(LIST_ROWS, 3, 2).rows.length, 1);
  assert.equal(pageSummary(pageOf([], 1, 2)), 'No skill matches these filters.');
});

test('a filter offers only values the rows carry, each with its count', () => {
  assert.deepEqual(classFilterOptions(LIST_ROWS).map((option) => option.label), [
    'All (5)', 'AI can take over (1)', 'AI assists (2)', 'Machines can do (1)',
    'Stays human (1)',
  ]);
  assert.deepEqual(filterOptions(LIST_ROWS, 'ty', ITEM_NAMES).map((option) => option.label),
    ['All (5)', 'Skill (4)', 'Knowledge item (1)']);
  const modes = filterOptions(LIST_ROWS, 'mo', MODE_NAMES);
  assert.deepEqual(modes.map((option) => option.value), ['', 't', 'p', 's']);
  // Nothing carries the field: only the "all" option is left, so the page can
  // leave the filter out rather than offer one that empties the table.
  assert.equal(filterOptions(LIST_ROWS, 'nope', MODE_NAMES).length, 1);
});

// --- how a skill is scored ---------------------------------------------------
//
// A stand-in for rubric_v2.json, shaped exactly like the real file. Every word
// below is made up; the page must print the file's words, never these.
const RUBRIC = {
  model: 'test-model-1',
  preamble: 'Consider a made-up system with no body.',
  questions: [
    {
      id: 'digital_output', kind: 'yesno', label: 'Digital output',
      instructions: 'Is the result data?',
      options: [{ name: 'true', text: 'Information.' }, { name: 'false', text: 'A physical change.' }],
    },
    {
      id: 'ai_substitution', kind: 'levels', label: 'AI substitution',
      instructions: 'How much can it do?',
      levels: ['None.', 'A minor part.', 'About half.', 'Nearly all.', 'All of it.'],
      knowledge: {
        instructions: 'For a body of knowledge:',
        levels: ['K none.', 'K minor.', 'K half.', 'K nearly.', 'K all.'],
      },
    },
    {
      id: 'mechanical', kind: 'levels', label: 'Machine automation',
      instructions: 'What do machines do?',
      levels: ['M none.', 'M small.', 'M half.', 'M nearly.', 'M all.'], knowledge: null,
    },
    {
      id: 'complementarity', kind: 'levels', label: 'AI assistance',
      instructions: 'How much better?',
      levels: ['C none.', 'C little.', 'C part.', 'C most.', 'C several times.'],
    },
    {
      id: 'mode', kind: 'choice', label: 'How it is exercised', instructions: 'How is it done?',
      options: [{ name: 'through_software', text: 'At a computer.' },
        { name: 'on_things', text: 'With hands.' }],
    },
    {
      id: 'deployment', kind: 'choice', label: 'Deployment', instructions: 'Where does it stand?',
      options: [{ name: 'routine', text: 'Already run.' }, { name: 'not_shown', text: 'Not shown.' }],
    },
  ],
  classes: [
    { code: 'S', name: 'Substituted', rule: 'SUB >= 0.5' },
    { code: 'A', name: 'Assisted', rule: 'COMP >= 0.5, and not already substituted' },
    { code: 'M', name: 'Mechanised', rule: 'MECH >= 0.5, and neither substituted nor assisted' },
    { code: 'I', name: 'Insulated', rule: 'none of the above' },
  ],
  display: '1.5 + 2.0 * position',
};

// One entry of an answers shard, for a knowledge item.
const ENTRY = {
  ty: 'k',
  d: 0.41,
  s: [0.03, 0.47, 0.37, 0.13, 0.0],
  k: [0.11, 0.44, 0.39, 0.06, 0.0],
  c: [0.0, 0.06, 0.74, 0.2, 0.0],
  mo: { through_software: 0.01, on_things: 0.97 },
  dp: { routine: 0.45, not_shown: 0.02 },
  cf: { s: 0.44, k: 0.48, c: 0.77, mo: 0.96, dp: 0.37 },
};

const questionById = (id) => rubricQuestions(RUBRIC).find((item) => item.id === id);
const sectionById = (id) => answerSections(RUBRIC, ENTRY).find((item) => item.id === id);

test('the six questions come out of the rubric in one shape', () => {
  const questions = rubricQuestions(RUBRIC);
  assert.deepEqual(questions.map((question) => question.label), [
    'Digital output', 'AI substitution', 'Machine automation', 'AI assistance',
    'How it is exercised', 'Deployment',
  ]);
  const levels = questionById('ai_substitution');
  assert.equal(levels.choices.length, 5);
  assert.deepEqual(levels.choices[1], { name: null, text: 'A minor part.' });
  assert.deepEqual(levels.knowledge.choices[1], { name: null, text: 'K minor.' });
  assert.equal(levels.knowledge.instructions, 'For a body of knowledge:');
  const choice = questionById('mode');
  assert.deepEqual(choice.choices[0], { name: 'through_software', text: 'At a computer.' });
  assert.equal(questionById('complementarity').knowledge, null);
  assert.deepEqual(rubricQuestions(null), []);
});

test('the class rules read as words, never as the rubric’s identifiers (R1)', () => {
  const classes = rubricClasses(RUBRIC);
  assert.deepEqual(classes.map((entry) => entry.siteName),
    ['AI can take over', 'AI assists', 'Machines can do', 'Stays human']);
  // The codes and their order come from the file; the wording is the site's.
  assert.match(classes[0].rule, /chance that an AI system could do nearly all/);
  assert.match(classes[0].rule, /above one in two/);
  assert.match(classes[3].rule, /none of the three chances reaches one in two/);
  for (const entry of classes) {
    assert.doesNotMatch(entry.rule, /SUB|COMP|MECH|>=|0\.5|Substituted|Mechanised/, entry.code);
    assert.equal('name' in entry, false, 'the rubric’s internal name never reaches a page');
  }
  assert.match(classes[3].meaning, /AI can still help with parts of it/);
  assert.deepEqual(rubricClasses(undefined), []);
});

test('the lead quotes the model and the file, and counts from the stats', () => {
  const intro = rubricIntro(RUBRIC, 13475);
  assert.match(intro.lead, /13,475 skills/);
  assert.match(intro.lead, /test-model-1/);
  assert.equal(intro.preamble, RUBRIC.preamble);
  assert.match(intro.display, /1\.5 \+ 2\.0 \* position/);
  assert.match(intro.display, /No class depends on them/);
  assert.match(rubricIntro(RUBRIC, null).lead, /every skill/);
});

// --- how the model answered --------------------------------------------------

test('a skill’s answers and its rationale live in the shard its id starts with', () => {
  assert.equal(shardOf('e130344f'), 'e1');
  assert.equal(shardOf('0005B5EC'), '00');
  assert.equal(shardOf('a'), null);
  assert.equal(shardOf(null), null);
});

test('a count agrees with its noun (A13)', () => {
  assert.equal(countNoun(1, 'occupation', 'occupations'), '1 occupation');
  assert.equal(countNoun(0, 'occupation', 'occupations'), '0 occupations');
  assert.equal(countNoun(319, 'occupation', 'occupations'), '319 occupations');
  assert.equal(countNoun(13475, 'skill', 'skills'), '13,475 skills');
});

test('a score of exactly the cut-off is not called "medium" (A20)', () => {
  // Under quadrants the site cuts at 6, so the band has to name the cut.
  assert.equal(scoreBand(6.0), 'on the cut-off');
  assert.equal(scoreBand(5.96), 'on the cut-off');
  assert.equal(scoreBand(5.9), 'below the cut-off');
  assert.equal(scoreBand(6.1), 'above the cut-off');
  assert.equal(scoreBand(1.0), 'below the cut-off');
  // Under shares these three scores cut nothing, so a plain band is honest.
  assert.equal(scoreBand(6.0, SHARES), 'medium');
  assert.equal(scoreBand(7.0, SHARES), 'high');
  assert.equal(scoreBand(3.9, SHARES), 'low');
  assert.equal(scoreBand(null), '');
  assert.equal(scoreBand(undefined, SHARES), '');
});

test('confidence reads as a word as well as a number', () => {
  assert.equal(confidenceWord(0.96), 'high');
  assert.equal(confidenceWord(0.8), 'high');
  assert.equal(confidenceWord(0.77), 'moderate');
  assert.equal(confidenceWord(0.44), 'low');
  assert.equal(confidenceWord(0), 'low');
  assert.equal(confidenceWord(undefined), '');
});

test('every question gets its levels back with the probability the model gave', () => {
  const sections = answerSections(RUBRIC, ENTRY);
  assert.deepEqual(sections.map((item) => item.id), [
    'digital_output', 'ai_substitution', 'mechanical', 'complementarity', 'mode',
    'deployment',
  ]);
  const substitution = sectionById('ai_substitution');
  assert.deepEqual(substitution.bars.map((bar) => bar.percent),
    ['3%', '47%', '37%', '13%', '0%']);
  assert.deepEqual(substitution.confidence, { share: 0.44, percent: '44%', word: 'low' });
  // Digital output was not asked for a confidence, so none is invented.
  assert.equal(sectionById('digital_output').confidence, null);
  assert.deepEqual(answerSections(RUBRIC, null), []);
});

test('a knowledge item is shown the wording it was actually asked', () => {
  assert.equal(askedAsKnowledge(ENTRY), true);
  assert.equal(askedAsKnowledge({ ty: 's' }), false);
  const substitution = sectionById('ai_substitution');
  assert.equal(substitution.knowledge, true);
  assert.deepEqual(substitution.bars.map((bar) => bar.label),
    ['K none.', 'K minor.', 'K half.', 'K nearly.', 'K all.']);
  // Machine automation has no knowledge variant, so the ordinary wording stands.
  const mechanical = sectionById('mechanical');
  assert.equal(mechanical.knowledge, false);
  assert.equal(mechanical.bars[0].label, 'M none.');
  const activity = answerSections(RUBRIC, { ...ENTRY, ty: 's' });
  assert.equal(activity[1].bars[0].label, 'None.');
});

test('the answer the model settled on is marked, and marked only once', () => {
  for (const [id, wanted] of [['ai_substitution', 1], ['mechanical', 1],
    ['complementarity', 2], ['mode', 1], ['deployment', 0]]) {
    const bars = sectionById(id).bars;
    assert.deepEqual(bars.map((bar) => bar.modal).filter(Boolean).length, 1, id);
    assert.equal(bars[wanted].modal, true, id);
  }
  // A yes/no question becomes two bars that add up, named by the rubric.
  const digital = sectionById('digital_output');
  assert.deepEqual(digital.bars.map((bar) => [bar.label, bar.percent]),
    [['Information.', '41%'], ['A physical change.', '59%']]);
  assert.equal(digital.bars[1].modal, true);
});

test('a missing or broken answer is left out rather than drawn as zero bars', () => {
  const half = answerSections(RUBRIC, { ty: 's', s: [0.1, 0.2, 0.3, 0.4, 0.0], cf: {} });
  assert.deepEqual(half.map((item) => item.id), ['ai_substitution']);
  assert.equal(half[0].confidence, null);
  assert.deepEqual(answerSections(RUBRIC, {}), []);
  assert.deepEqual(answerSections(null, ENTRY), []);
});
