import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FILTER_LIMIT,
  SHARES_SKILL_COLUMNS,
  SKILL_COLUMNS,
  SKILL_FILTERS,
  ancestorKeys,
  buildModel,
  childrenOf,
  classSegments,
  filterMatches,
  filterSkillRows,
  findOccupation,
  groupSubtitle,
  horizontalMove,
  jobSummary,
  leafFigure,
  leafParts,
  matchMessage,
  mixSegments,
  mixText,
  moveIndex,
  nearLineCaveat,
  readState,
  rowIndexOf,
  selectionId,
  selectionOf,
  skillColumns,
  skillMix,
  skillRows,
  skillSummary,
  skillTableRows,
  sortSkillRows,
  treeHash,
  treeHref,
  typeAheadIndex,
  unitChain,
  visibleRows,
  whyLine,
  alsoMatches,
  plural,
  resultRows,
  scoreKeyLine,
  SKILL_NEAR_TEXT,
} from '../../site/js/pages/tree-model.js';
import { INDEX } from './fixture.js';

// A stand-in for groups.json, shaped exactly like the real file and covering
// every occupation in fixture.js. Synthetic numbers.
const GROUPS = {
  all: {
    label: 'All occupations', level: 'all', code: '', n: 9,
    q: { TRANSFORM: 2, STABLE: 2, EVOLVE: 4, SHRINK: 1 },
    parent: null, children: ['major:1', 'major:2', 'major:3', 'major:5'],
  },
  'major:1': {
    label: 'Managers', level: 'major', code: '1', n: 1,
    q: { EVOLVE: 1 }, parent: 'all', children: ['sub:13'],
  },
  'sub:13': {
    label: 'Production managers', level: 'sub', code: '13', n: 1,
    q: { EVOLVE: 1 }, parent: 'major:1', children: ['minor:134'],
  },
  'minor:134': {
    label: 'Services managers', level: 'minor', code: '134', n: 1,
    q: { EVOLVE: 1 }, parent: 'sub:13', children: ['unit:1345'],
  },
  'unit:1345': {
    label: 'Education managers', level: 'unit', code: '1345', n: 1,
    q: { EVOLVE: 1 }, parent: 'minor:134', children: [],
  },
  'major:2': {
    label: 'Professionals', level: 'major', code: '2', n: 4,
    q: { TRANSFORM: 2, EVOLVE: 2 }, parent: 'all', children: ['sub:22', 'sub:25', 'sub:26'],
  },
  'sub:22': {
    label: 'Health professionals', level: 'sub', code: '22', n: 1,
    q: { EVOLVE: 1 }, parent: 'major:2', children: ['minor:222'],
  },
  'minor:222': {
    label: 'Nursing professionals', level: 'minor', code: '222', n: 1,
    q: { EVOLVE: 1 }, parent: 'sub:22', children: ['unit:2221'],
  },
  'unit:2221': {
    label: 'Nursing professionals', level: 'unit', code: '2221', n: 1,
    q: { EVOLVE: 1 }, parent: 'minor:222', children: [],
  },
  'sub:25': {
    label: 'ICT professionals', level: 'sub', code: '25', n: 2,
    q: { TRANSFORM: 2 }, parent: 'major:2', children: ['minor:251'],
  },
  'minor:251': {
    label: 'Software developers and analysts', level: 'minor', code: '251', n: 2,
    q: { TRANSFORM: 2 }, parent: 'sub:25', children: ['unit:2512'],
  },
  'unit:2512': {
    label: 'Software developers', level: 'unit', code: '2512', n: 2,
    q: { TRANSFORM: 2 }, parent: 'minor:251', children: [],
  },
  'sub:26': {
    label: 'Creative professionals', level: 'sub', code: '26', n: 1,
    q: { EVOLVE: 1 }, parent: 'major:2', children: ['minor:265'],
  },
  'minor:265': {
    label: 'Creative and performing artists', level: 'minor', code: '265', n: 1,
    q: { EVOLVE: 1 }, parent: 'sub:26', children: ['unit:2654'],
  },
  'unit:2654': {
    label: 'Film and stage directors', level: 'unit', code: '2654', n: 1,
    q: { EVOLVE: 1 }, parent: 'minor:265', children: [],
  },
  'major:3': {
    label: 'Technicians', level: 'major', code: '3', n: 3,
    q: { STABLE: 1, EVOLVE: 1, SHRINK: 1 }, parent: 'all', children: ['sub:32', 'sub:33'],
  },
  'sub:32': {
    label: 'Health associates', level: 'sub', code: '32', n: 2,
    q: { STABLE: 1, EVOLVE: 1 }, parent: 'major:3', children: ['minor:322'],
  },
  'minor:322': {
    label: 'Nursing associates', level: 'minor', code: '322', n: 2,
    q: { STABLE: 1, EVOLVE: 1 }, parent: 'sub:32', children: ['unit:3221'],
  },
  'unit:3221': {
    label: 'Nursing associate professionals', level: 'unit', code: '3221', n: 2,
    q: { STABLE: 1, EVOLVE: 1 }, parent: 'minor:322', children: [],
  },
  'sub:33': {
    label: 'Business associates', level: 'sub', code: '33', n: 1,
    q: { SHRINK: 1 }, parent: 'major:3', children: ['minor:331'],
  },
  'minor:331': {
    label: 'Financial associates', level: 'minor', code: '331', n: 1,
    q: { SHRINK: 1 }, parent: 'sub:33', children: ['unit:3313'],
  },
  'unit:3313': {
    label: 'Accounting associates', level: 'unit', code: '3313', n: 1,
    q: { SHRINK: 1 }, parent: 'minor:331', children: [],
  },
  'major:5': {
    label: 'Service and sales workers', level: 'major', code: '5', n: 1,
    q: { STABLE: 1 }, parent: 'all', children: ['sub:51'],
  },
  'sub:51': {
    label: 'Personal service workers', level: 'sub', code: '51', n: 1,
    q: { STABLE: 1 }, parent: 'major:5', children: ['minor:513'],
  },
  'minor:513': {
    label: 'Waiters and bartenders', level: 'minor', code: '513', n: 1,
    q: { STABLE: 1 }, parent: 'sub:51', children: ['unit:5132'],
  },
  'unit:5132': {
    label: 'Bartenders', level: 'unit', code: '5132', n: 1,
    q: { STABLE: 1 }, parent: 'minor:513', children: [],
  },
};

const STATS = {
  occupations: 9,
  quadrants: { counts: { TRANSFORM: 2, STABLE: 2, EVOLVE: 4, SHRINK: 1 } },
  near_line: { count: 3, share: 0.3333 },
};

const PORTFOLIO = {
  skills: {
    aaaa1111: { t: 'writing code', a: 7.2, m: 9.1, r: 'why' },
    bbbb2222: { t: 'talking to people', a: 2.4, m: 6.5, r: 'why' },
    cccc3333: { t: 'filing forms', a: 8.1, m: 3.2, r: 'why' },
    dddd4444: { t: 'unscored craft', a: null, m: null, r: null },
  },
  occupations: [
    { s: 'software-developer', se: ['aaaa1111', 'missing0'], so: ['bbbb2222', 'cccc3333', 'dddd4444'] },
  ],
};

const model = buildModel(GROUPS, INDEX);

function view(extra = {}) {
  return { expanded: new Set(), filter: null, selectedId: null, ...extra };
}

/* --- the model ------------------------------------------------------------ */

test('buildModel buckets jobs under their unit group and indexes slugs', () => {
  assert.equal(model.jobsByUnit.get('unit:2512').length, 2);
  assert.equal(model.jobsByUnit.get('unit:3221').length, 2);
  assert.equal(model.bySlug.get('bookkeeper').t, 'bookkeeper');
});

test('jobs inside a unit group come out alphabetically', () => {
  const titles = model.jobsByUnit.get('unit:2512').map((row) => row.t);
  assert.deepEqual(titles, ['embedded systems software developer', 'software developer']);
});

test('unitChain walks the four ISCO levels, narrowest first', () => {
  assert.deepEqual(unitChain('2512'), ['unit:2512', 'minor:251', 'sub:25', 'major:2']);
  assert.deepEqual(unitChain(''), []);
  assert.deepEqual(unitChain('25'), ['sub:25', 'major:2']);
});

test('childrenOf gives child groups, and jobs only at a unit group', () => {
  assert.deepEqual(childrenOf(model, 'all').map((node) => node.key),
    ['major:1', 'major:2', 'major:3', 'major:5']);
  const jobs = childrenOf(model, 'unit:2512');
  assert.deepEqual(jobs.map((node) => node.kind), ['job', 'job']);
  assert.equal(jobs[1].slug, 'software-developer');
  assert.deepEqual(childrenOf(model, 'nope'), []);
});

/* --- what is on screen ---------------------------------------------------- */

test('a closed tree shows only the major groups', () => {
  const rows = visibleRows(model, view());
  assert.equal(rows.length, 4);
  assert.deepEqual(rows.map((row) => row.level), [1, 1, 1, 1]);
  assert.deepEqual(rows.map((row) => row.posinset), [1, 2, 3, 4]);
  assert.equal(rows[0].setsize, 4);
  assert.equal(rows.every((row) => row.expanded === false), true);
});

test('opening a branch adds that branch and nothing else', () => {
  const rows = visibleRows(model, view({ expanded: new Set(['major:2']) }));
  assert.deepEqual(rows.map((row) => row.id),
    ['major:1', 'major:2', 'sub:22', 'sub:25', 'sub:26', 'major:3', 'major:5']);
  assert.equal(rows[1].expanded, true);
  assert.equal(rows[2].level, 2);
});

test('an open unit group shows its jobs as leaves', () => {
  const open = new Set(['major:2', 'sub:25', 'minor:251', 'unit:2512']);
  const rows = visibleRows(model, view({ expanded: open }));
  const leaves = rows.filter((row) => row.kind === 'job');
  assert.deepEqual(leaves.map((row) => row.slug),
    ['embedded-systems-software-developer', 'software-developer']);
  assert.equal(leaves[0].level, 5);
});

test('a group with no visible children is not marked expanded', () => {
  const rows = visibleRows(model, view({ expanded: new Set(['major:1']) }));
  assert.equal(rows[0].expanded, true);
  const empty = buildModel({ all: { label: 'a', level: 'all', n: 0, parent: null, children: ['unit:9999'] }, 'unit:9999': { label: 'b', level: 'unit', code: '9999', n: 0, q: {}, parent: 'all', children: [] } }, []);
  const none = visibleRows(empty, view({ expanded: new Set(['unit:9999']) }));
  assert.equal(none[0].expanded, false);
});

test('the selected row is flagged, and rowIndexOf finds it', () => {
  const rows = visibleRows(model, view({ selectedId: 'major:3' }));
  assert.equal(rows[2].selected, true);
  assert.equal(rows[0].selected, false);
  assert.equal(rowIndexOf(rows, 'major:3'), 2);
  assert.equal(rowIndexOf(rows, 'nope'), -1);
});

/* --- filtering ------------------------------------------------------------ */

test('an empty filter is no filter at all', () => {
  assert.equal(filterMatches(model, ''), null);
  assert.equal(filterMatches(model, '   '), null);
});

test('the filter keeps matching jobs and every group above them', () => {
  const result = filterMatches(model, 'programmer');
  assert.equal(result.slugs.has('software-developer'), true);
  assert.equal(result.slugs.has('venue-programmer'), true);
  assert.equal(result.keys.has('major:2'), true);
  assert.equal(result.keys.has('unit:2512'), true);
  assert.equal(result.keys.has('major:3'), false);
});

test('a filtered tree is open down to the matches and hides the rest', () => {
  const rows = visibleRows(model, view({ filter: filterMatches(model, 'programmer') }));
  const ids = rows.map((row) => row.id);
  assert.deepEqual(ids.filter((id) => id.startsWith('major:')), ['major:2']);
  assert.equal(ids.includes('job:software-developer'), true);
  assert.equal(ids.includes('job:bookkeeper'), false);
});

test('the filter is capped, and says so', () => {
  const result = filterMatches(model, 'nurse', 1);
  assert.equal(result.shown, 1);
  assert.equal(result.capped, true);
  assert.equal(matchMessage(result), '4 jobs match. Showing the first 1 — type more to narrow it.');
});

test('the count message counts every match, not the drawn ones', () => {
  assert.equal(matchMessage(filterMatches(model, 'programmer')), '2 jobs match.');
  assert.equal(matchMessage(filterMatches(model, 'sommelière')), '1 job matches.');
  assert.equal(matchMessage(filterMatches(model, 'zzzz')),
    'No jobs match. Clear the filter to see the whole tree.');
  assert.equal(matchMessage(null), '');
});

test('the cap is a real number', () => {
  assert.equal(Number.isInteger(FILTER_LIMIT) && FILTER_LIMIT > 0, true);
});

/* --- URL state ------------------------------------------------------------ */

test('readState reads a job, a group, a filter, or nothing', () => {
  assert.deepEqual(readState('#job=software-developer'),
    { selection: { kind: 'job', id: 'software-developer' }, query: '' });
  assert.deepEqual(readState('#g=unit:2512'),
    { selection: { kind: 'group', id: 'unit:2512' }, query: '' });
  assert.deepEqual(readState('#g=major:2&q=nurse'),
    { selection: { kind: 'group', id: 'major:2' }, query: 'nurse' });
  assert.deepEqual(readState(''), { selection: null, query: '' });
});

test('treeHash is the inverse of readState', () => {
  const cases = [
    { selection: { kind: 'job', id: 'software-developer' }, query: '' },
    { selection: { kind: 'group', id: 'unit:2512' }, query: 'nurse' },
    { selection: null, query: '' },
  ];
  for (const state of cases) {
    assert.deepEqual(readState(treeHash(state.selection, state.query)), state);
  }
});

test('treeHref keeps the group key readable', () => {
  assert.equal(treeHref({ kind: 'group', id: 'major:2' }, ''), 'tree.html#g=major:2');
  assert.equal(treeHref(null, ''), 'tree.html');
  assert.equal(treeHref({ kind: 'job', id: 'bookkeeper' }, 'book'),
    'tree.html#job=bookkeeper&q=book');
});

test('selectionId and selectionOf round-trip', () => {
  assert.equal(selectionId({ kind: 'job', id: 'bookkeeper' }), 'job:bookkeeper');
  assert.equal(selectionId({ kind: 'group', id: 'major:2' }), 'major:2');
  assert.equal(selectionId(null), null);
  assert.deepEqual(selectionOf('job:bookkeeper'), { kind: 'job', id: 'bookkeeper' });
  assert.deepEqual(selectionOf('major:2'), { kind: 'group', id: 'major:2' });
  assert.equal(selectionOf(null), null);
});

test('a deep link expands every group above the job', () => {
  assert.deepEqual(ancestorKeys(model, { kind: 'job', id: 'software-developer' }),
    ['unit:2512', 'minor:251', 'sub:25', 'major:2', 'all']);
  assert.deepEqual(ancestorKeys(model, { kind: 'group', id: 'sub:25' }),
    ['sub:25', 'major:2', 'all']);
  assert.deepEqual(ancestorKeys(model, { kind: 'job', id: 'nope' }), []);
  assert.deepEqual(ancestorKeys(model, null), []);
});

/* --- keyboard ------------------------------------------------------------- */

test('Up, Down, Home and End move along the visible rows', () => {
  const rows = visibleRows(model, view());
  assert.equal(moveIndex(rows, 0, 'ArrowDown'), 1);
  assert.equal(moveIndex(rows, 3, 'ArrowDown'), 3);
  assert.equal(moveIndex(rows, 0, 'ArrowUp'), 0);
  assert.equal(moveIndex(rows, 2, 'ArrowUp'), 1);
  assert.equal(moveIndex(rows, 2, 'Home'), 0);
  assert.equal(moveIndex(rows, 0, 'End'), 3);
  assert.equal(moveIndex(rows, 0, 'Enter'), -1);
});

test('Right opens a closed group, then steps into it', () => {
  const closed = visibleRows(model, view());
  assert.deepEqual(horizontalMove(closed, 1, 'ArrowRight'), { action: 'expand', key: 'major:2' });
  const open = visibleRows(model, view({ expanded: new Set(['major:2']) }));
  assert.deepEqual(horizontalMove(open, 1, 'ArrowRight'), { action: 'move', index: 2 });
});

test('Left closes an open group, then steps out to the parent', () => {
  const open = visibleRows(model, view({ expanded: new Set(['major:2']) }));
  assert.deepEqual(horizontalMove(open, 1, 'ArrowLeft'), { action: 'collapse', key: 'major:2' });
  assert.deepEqual(horizontalMove(open, 3, 'ArrowLeft'), { action: 'move', index: 1 });
  assert.equal(horizontalMove(open, 0, 'ArrowLeft'), null);
  assert.equal(horizontalMove(open, 0, 'ArrowDown'), null);
  assert.equal(horizontalMove(open, 99, 'ArrowLeft'), null);
});

test('type-ahead steps on for one letter and stays put for a word', () => {
  const rows = visibleRows(model, view());
  assert.deepEqual(rows.map((row) => row.label),
    ['Managers', 'Professionals', 'Technicians', 'Service and sales workers']);
  assert.equal(typeAheadIndex(rows, 0, 'p'), 1);
  assert.equal(typeAheadIndex(rows, 1, 's'), 3);
  assert.equal(typeAheadIndex(rows, 0, 'm'), 0);
  assert.equal(typeAheadIndex(rows, 2, 'tech'), 2);
  assert.equal(typeAheadIndex(rows, 0, 'zz'), -1);
  assert.equal(typeAheadIndex(rows, 0, ''), -1);
  assert.equal(typeAheadIndex([], 0, 'a'), -1);
});

/* --- what the panes say --------------------------------------------------- */

test('the quadrant mix names every box, with counts and percentages', () => {
  const segments = mixSegments(GROUPS['major:3'].q);
  assert.deepEqual(segments.map((segment) => segment.label),
    ['Transform', 'Stable', 'Evolve', 'Shrink']);
  assert.equal(segments[1].count, 1);
  assert.equal(segments[1].percent, '34%'); // 1/3 each: the remainder lands here
  assert.equal(segments[1].text, '1 of 3 jobs');
  assert.equal(segments[0].count, 0);
});

test('the compact mix reads largest first and skips empty boxes', () => {
  assert.equal(mixText(mixSegments(GROUPS.all.q)),
    '45% Evolve, 22% Transform, 22% Stable, 11% Shrink');
  assert.equal(mixText(mixSegments({})), 'no scored jobs');
});

test('the group subtitle names the level, the code and the size', () => {
  assert.equal(groupSubtitle(GROUPS['unit:2512']), 'Unit group 2512 · ISCO-08 · 2 jobs');
  assert.equal(groupSubtitle(GROUPS.all), 'All occupations · 9 jobs');
});

test('the caveat quotes stats.json and never a second scoring run', () => {
  const text = nearLineCaveat(STATS);
  assert.match(text, /3 of 9 jobs \(33%\) sit within 0\.5 of a cut-off/);
  assert.match(text, /a band, not a count/);
  assert.doesNotMatch(text, /agree|second|another run|run two/i);
  assert.match(nearLineCaveat(null), /read this split as a band/);
});

test('a leaf says its quadrant in words, never in colour alone', () => {
  assert.equal(jobSummary({ q: 'TRANSFORM', a: 6.4, m: 8.9 }),
    'Transform, automation 6.4, amplification 8.9');
  assert.equal(jobSummary({ q: null, a: null, m: null }),
    'Not scored, automation Not scored, amplification Not scored');
});

/* --- the skills of one job ------------------------------------------------ */

test('findOccupation resolves a slug inside portfolio_data', () => {
  assert.equal(findOccupation(PORTFOLIO, 'software-developer').s, 'software-developer');
  assert.equal(findOccupation(PORTFOLIO, 'bookkeeper'), null);
  assert.equal(findOccupation(null, 'bookkeeper'), null);
});

test('skillRows resolves ids, essential first, dropping unknown ids', () => {
  const rows = skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer'));
  assert.equal(rows.length, 4);
  assert.equal(rows[0].title, 'writing code');
  assert.equal(rows[0].essential, true);
  assert.equal(rows[1].essential, false);
  assert.equal(rows[0].quadrant, 'TRANSFORM');
  assert.equal(rows[3].quadrant, null);
  assert.equal(rows[3].auto, null);
});

test('the skills split on the same cut-off the jobs use', () => {
  const mix = skillMix(skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer')));
  assert.equal(mix.total, 4);
  assert.equal(mix.essential, 1);
  assert.equal(mix.scored, 3);
  assert.equal(mix.unscored, 1);
  const named = Object.fromEntries(mix.bars.map((bar) => [bar.label, bar.count]));
  assert.deepEqual(named, { Transform: 1, Stable: 0, Evolve: 1, Shrink: 1 });
  assert.equal(mix.bars[0].text, '1 of 3 skills');
});

test('the skills summary counts what is there and what is missing', () => {
  const mix = skillMix(skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer')));
  assert.equal(skillSummary(mix), '4 skills, 1 of them essential. 1 have no scores yet.');
  assert.equal(skillSummary({ total: 1, essential: 1, unscored: 0 }), '1 skill, 1 of them essential.');
});

/* --- the skills table ----------------------------------------------------- */

const SKILLS = skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer'));

test('the table offers the five columns and the three filters', () => {
  assert.deepEqual(SKILL_COLUMNS.map((column) => column.key),
    ['title', 'essential', 'auto', 'amp', 'quadrant']);
  assert.deepEqual(SKILL_FILTERS.map((option) => option.key), ['essential', 'optional', 'all']);
});

test('sorting never mutates and always puts missing scores last', () => {
  const before = SKILLS.map((row) => row.id);
  const sorted = sortSkillRows(SKILLS, 'auto', true);
  assert.deepEqual(SKILLS.map((row) => row.id), before);
  assert.deepEqual(sorted.map((row) => row.title),
    ['filing forms', 'writing code', 'talking to people', 'unscored craft']);
  assert.equal(sortSkillRows(SKILLS, 'auto', false)[3].title, 'unscored craft');
});

test('essential first is a sort like any other, and ties break by name', () => {
  assert.equal(sortSkillRows(SKILLS, 'essential', true)[0].title, 'writing code');
  assert.deepEqual(sortSkillRows(SKILLS, 'essential', true).slice(1).map((row) => row.title),
    ['filing forms', 'talking to people', 'unscored craft']);
  assert.equal(sortSkillRows(SKILLS, 'nonsense', false)[0].title, 'filing forms');
});

test('the filter toggle keeps essential, optional or everything', () => {
  assert.equal(filterSkillRows(SKILLS, 'essential').length, 1);
  assert.equal(filterSkillRows(SKILLS, 'optional').length, 3);
  assert.equal(filterSkillRows(SKILLS, 'all').length, 4);
  assert.notEqual(filterSkillRows(SKILLS, 'all'), SKILLS);
});

test('a missing score reads "Not scored", never a bare question mark', () => {
  const rows = skillTableRows(SKILLS);
  const unscored = rows.find((row) => row.title === 'unscored craft');
  assert.deepEqual(unscored.cells.map((cell) => cell.text),
    ['unscored craft', 'Optional', 'Not scored', 'Not scored', 'Not scored']);
  assert.equal(rows[0].quadrant, 'TRANSFORM');
  assert.deepEqual(rows[0].cells.map((cell) => cell.text),
    ['writing code', 'Essential', '7.2', '9.1', 'Transform']);
});

/* --- the shares scheme ---------------------------------------------------- */

// Synthetic stands-in for a `_v2` set: seven type codes in `q`, four shares in
// `sh`, a class on every skill and a third score. Numbers chosen so no rounding
// is accidental.
const SHARES_STATS = {
  scheme: 'shares',
  occupations: 8,
  skills_scored: 40,
  types: {
    order: ['AUTOMATION_HEAVY', 'TRANSFORMING', 'AUGMENTED', 'MECHANISABLE',
      'INSULATED_PHYSICAL', 'INSULATED_PEOPLE', 'MIXED'],
    counts: {
      AUTOMATION_HEAVY: 1,
      TRANSFORMING: 1,
      AUGMENTED: 2,
      MECHANISABLE: 0,
      INSULATED_PHYSICAL: 3,
      INSULATED_PEOPLE: 1,
      MIXED: 0,
    },
  },
  skill_classes: { counts: { S: 5, A: 8, M: 3, I: 24 } },
  near_line: { count: 2, share: 0.25 },
};

const SHARES_GROUP_COUNTS = {
  AUTOMATION_HEAVY: 1,
  TRANSFORMING: 0,
  AUGMENTED: 2,
  MECHANISABLE: 0,
  INSULATED_PHYSICAL: 1,
  INSULATED_PEOPLE: 0,
  MIXED: 0,
};

const SHARES_PORTFOLIO = {
  skills: {
    aaaa1111: { t: 'writing code', a: 7.2, m: 9.1, k: 1.4, c: 'S', r: 'why' },
    bbbb2222: { t: 'talking to people', a: 2.4, m: 6.5, k: 1.1, c: 'A', r: 'why' },
    cccc3333: { t: 'tending a press', a: 5.1, m: 3.2, k: 8.3, c: 'M', r: 'why' },
    dddd4444: { t: 'calming a patient', a: 1.9, m: 3.0, k: 1.0, c: 'I', r: 'why' },
    eeee5555: { t: 'unscored craft', a: null, m: null, k: null, c: null, r: null },
  },
  occupations: [{
    s: 'shift-supervisor',
    why: 'people',
    se: ['aaaa1111', 'bbbb2222'],
    so: ['cccc3333', 'dddd4444', 'eeee5555'],
  }],
};

const SHARES_SKILLS = skillRows(SHARES_PORTFOLIO,
  findOccupation(SHARES_PORTFOLIO, 'shift-supervisor'));

test('the mix follows the seven types, and marks itself as a type bar', () => {
  const segments = mixSegments(SHARES_GROUP_COUNTS);
  assert.equal(segments.length, 7);
  assert.deepEqual(segments.map((segment) => segment.label), [
    'Automation-heavy', 'Transforming', 'Augmented', 'Mechanisable',
    'Insulated by physical work', 'Insulated by work with people', 'Mixed',
  ]);
  assert.equal(segments.every((segment) => segment.attribute === 'data-type'), true);
  assert.equal(segments[2].count, 2);
  assert.equal(segments[2].percent, '50%');
  assert.equal(segments[2].text, '2 of 4 jobs');
});

test('the spoken mix names the types and says they are types', () => {
  const text = mixText(mixSegments(SHARES_GROUP_COUNTS));
  assert.equal(text, 'job types: 50% Augmented, 25% Automation-heavy, '
    + '25% Insulated by physical work');
  assert.doesNotMatch(text, /quadrant|box/i);
});

test('a quadrant mix keeps the wording it always had', () => {
  const segments = mixSegments(GROUPS.all.q);
  assert.equal(segments.every((segment) => segment.attribute === 'data-quadrant'), true);
  assert.equal(mixText(segments), '45% Evolve, 22% Transform, 22% Stable, 11% Shrink');
});

test('the skill-class split comes from stats, and is empty without one', () => {
  const segments = classSegments(SHARES_STATS);
  assert.deepEqual(segments.map((segment) => segment.code), ['S', 'A', 'M', 'I']);
  assert.deepEqual(segments.map((segment) => segment.label),
    ['AI can take over', 'AI assists', 'Machines can do', 'Stays human']);
  assert.equal(segments.every((segment) => segment.attribute === 'data-class'), true);
  assert.equal(segments[3].count, 24);
  assert.equal(segments[3].percent, '60%');
  assert.equal(segments[3].text, '24 of 40 skills');
  assert.deepEqual(classSegments(STATS), []);
  assert.deepEqual(classSegments(null), []);
});

test('a leaf paints data-type and leads with the substituted share', () => {
  const node = { q: 'TRANSFORMING', a: 6.9, m: 7.2, k: 2.2, sh: [0.54, 0.31, 0, 0.15] };
  assert.deepEqual(leafParts(node), {
    attribute: 'data-type',
    code: 'TRANSFORMING',
    name: 'Transforming',
    figure: '54% AI can take over',
  });
  assert.equal(leafParts({ q: 'INSULATED_PHYSICAL', sh: [0.1, 0.2, 0.2, 0.5] }).name,
    'Physical work');
});

test('a quadrant leaf is painted and worded exactly as before', () => {
  assert.deepEqual(leafParts({ q: 'TRANSFORM', a: 6.4, m: 8.9 }), {
    attribute: 'data-quadrant', code: 'TRANSFORM', name: 'Transform', figure: '6.4 / 8.9',
  });
  assert.deepEqual(leafParts({ q: null, a: null, m: null }), {
    attribute: 'data-quadrant', code: '', name: 'Not scored', figure: 'Not scored / Not scored',
  });
});

test('a leaf with a type but no usable shares says so rather than NaN', () => {
  assert.equal(leafFigure({ q: 'MIXED', sh: null }), 'Not scored');
  assert.equal(leafFigure({ q: 'MIXED', sh: [0.5, 0.5] }), 'Not scored');
});

test('a shares leaf is said out loud as its type and its four shares', () => {
  const spoken = jobSummary({ q: 'AUGMENTED', a: 5, m: 7, sh: [0.12, 0.44, 0.04, 0.4] });
  assert.equal(spoken, 'Augmented. AI can take over 12% · AI assists 44% · machines 4% '
    + '· stays human 40%.');
  assert.doesNotMatch(spoken, /automation|amplification|quadrant/i);
  // A type code without usable shares says so; it never falls back to the two
  // quadrant scores, which mean something else under this scheme.
  assert.equal(jobSummary({ q: 'MIXED', sh: null, a: 4, m: 4 }), 'Mixed. Shares not scored.');
  assert.equal(jobSummary({ q: 'TRANSFORM', a: 4, m: 4 }),
    'Transform, automation 4.0, amplification 4.0');
});

test('the caveat under a shares set names the rule that scheme uses', () => {
  const text = nearLineCaveat(SHARES_STATS);
  assert.match(text, /2 of 8 jobs \(25%\) sit near a cut-off/);
  assert.match(text, /moving any one of their four shares by 5/);
  assert.doesNotMatch(text, /within 0\.5|box/i);
});

test('the reason a job stays human is worded, never printed as a code', () => {
  const some = [0.2, 0.2, 0.2, 0.4];
  assert.match(whyLine({ why: 'people', sh: some }), /with and for other people/);
  assert.match(whyLine({ why: 'physical', sh: some }), /on things, in a place/);
  assert.match(whyLine({ why: 'other', sh: some }), /desk work/);
  assert.equal(whyLine({ why: 'nonsense', sh: some }), '');
  assert.equal(whyLine(null), '');
});

test('a job with nothing staying human is not told what stays human', () => {
  assert.equal(whyLine({ why: 'other', sh: [0.85, 0.15, 0, 0] }), '');
  assert.equal(whyLine({ why: 'people', sh: null }), '');
});

test('a skill carries its class and its third score', () => {
  assert.equal(SHARES_SKILLS.length, 5);
  assert.equal(SHARES_SKILLS[0].cls, 'S');
  assert.equal(SHARES_SKILLS[0].mech, 1.4);
  assert.equal(SHARES_SKILLS[4].cls, null);
  assert.equal(SHARES_SKILLS[4].mech, null);
  assert.equal(skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer'))[0].cls, null);
});

test('under shares the skills split by class, not by box', () => {
  const mix = skillMix(SHARES_SKILLS, 'shares');
  assert.equal(mix.total, 5);
  assert.equal(mix.essential, 2);
  assert.equal(mix.scored, 4);
  assert.equal(mix.unscored, 1);
  assert.deepEqual(mix.bars.map((bar) => [bar.label, bar.count]), [
    ['AI can take over', 1], ['AI assists', 1], ['Machines can do', 1], ['Stays human', 1],
  ]);
  assert.equal(mix.bars[0].text, '1 of 4 skills');
  assert.equal(mix.bars[0].attribute, 'data-class');
});

test('without a scheme the skills split exactly as they always have', () => {
  const mix = skillMix(SKILLS);
  assert.deepEqual(Object.fromEntries(mix.bars.map((bar) => [bar.label, bar.count])),
    { Transform: 1, Stable: 0, Evolve: 1, Shrink: 1 });
});

test('the shares table names the class and the three scores', () => {
  assert.deepEqual(skillColumns('shares').map((column) => column.label),
    ['Skill', 'In this job', 'Class', 'AI substitution', 'AI assistance', 'Machine automation']);
  assert.deepEqual(skillColumns(), SKILL_COLUMNS);
  assert.deepEqual(skillColumns('quadrants'), SKILL_COLUMNS);
  assert.equal(SHARES_SKILL_COLUMNS.some((column) => /quadrant|risk|amplification/i
    .test(column.label)), false);
});

test('a shares row prints the class in words and Not scored for a missing one', () => {
  const rows = skillTableRows(SHARES_SKILLS, 'shares');
  assert.deepEqual(rows[0].cells.map((cell) => cell.text),
    ['writing code', 'Essential', 'AI can take over', '7.2', '9.1', '1.4']);
  assert.deepEqual(rows[4].cells.map((cell) => cell.text),
    ['unscored craft', 'Optional', 'Not scored', 'Not scored', 'Not scored', 'Not scored']);
  assert.equal(rows[0].cls, 'S');
  assert.equal(rows[4].cls, null);
});

test('the class column sorts by the name shown, and missing classes go last', () => {
  // Sorted on what the cell says: "AI assists", "AI can take over", "Machines
  // can do", "Stays human" — never on the letter, which is never printed.
  const sorted = sortSkillRows(SHARES_SKILLS, 'cls', false);
  assert.deepEqual(sorted.map((row) => row.cls), ['A', 'S', 'M', 'I', null]);
  assert.equal(sortSkillRows(SHARES_SKILLS, 'cls', true)[4].cls, null);
  assert.equal(sortSkillRows(SHARES_SKILLS, 'mech', true)[0].title, 'tending a press');
});

/* --- the audit pass: plurals, a flat result list, a visible key ------------ */

test('a count of one takes the singular noun (A13)', () => {
  assert.equal(plural(1, 'job'), '1 job');
  assert.equal(plural(0, 'job'), '0 jobs');
  assert.equal(plural(2, 'job'), '2 jobs');
  assert.equal(plural(3039, 'skill'), '3,039 skills');
});

test('no bar, subtitle or leaf can print "1 jobs" any more (A13)', () => {
  assert.equal(groupSubtitle({ level: 'unit', code: '5132', n: 1 }),
    'Unit group 5132 · ISCO-08 · 1 job');
  assert.equal(mixSegments({ EVOLVE: 1 })[2].text, '1 of 1 job');
  assert.equal(mixSegments(GROUPS['major:3'].q)[1].text, '1 of 3 jobs');
});

// The order is search.js's to decide and its own tests to pin; what belongs
// here is that every hit becomes a row, in the order it arrived, carrying the
// group that holds it.
test('the filter answers with the matching jobs, each under its group (A16)', () => {
  const filter = filterMatches(model, 'programmer');
  const rows = resultRows(model, filter);
  assert.deepEqual(rows.map((row) => row.slug), filter.hits.map((hit) => hit.row.s));
  assert.deepEqual([...rows].map((row) => row.slug).sort(),
    ['software-developer', 'venue-programmer']);
  const bySlug = Object.fromEntries(rows.map((row) => [row.slug, row]));
  assert.equal(bySlug['software-developer'].group, 'Software developers');
  assert.equal(bySlug['venue-programmer'].group, 'Film and stage directors');
  assert.deepEqual(resultRows(model, null), []);
});

test('a row that matched on a synonym says which one (A16)', () => {
  const rows = resultRows(model, filterMatches(model, 'programmer'));
  const bySlug = Object.fromEntries(rows.map((row) => [row.slug, row]));
  // "programmer" is an alternative label of software developer, not its title.
  assert.equal(alsoMatches(bySlug['software-developer']),
    'also matches “programmer”');
  assert.equal(alsoMatches(bySlug['venue-programmer']), '');
  assert.equal(alsoMatches(null), '');
});

test('a job whose group is missing still gets a row, with no subtitle', () => {
  const orphan = buildModel({}, INDEX);
  const rows = resultRows(orphan, filterMatches(orphan, 'bookkeeper'));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].group, '');
  assert.equal(rows[0].title, 'bookkeeper');
});

test('the result list is capped with the tree, and never outruns it', () => {
  const filter = filterMatches(model, 'nurse', 2);
  assert.equal(resultRows(model, filter).length, 2);
  assert.equal(filter.total, 4);
  assert.match(matchMessage(filter), /^4 jobs match\. Showing the first 2/);
});

test('the figures every row prints get a key, in the words of the scheme (A19)', () => {
  assert.equal(scoreKeyLine('quadrants'),
    'Each job shows automation risk / amplification, both out of 10.');
  assert.equal(scoreKeyLine(),
    'Each job shows automation risk / amplification, both out of 10.');
  assert.match(scoreKeyLine('shares'), /type and how much of its work AI can take over/);
  assert.doesNotMatch(scoreKeyLine('shares'), /automation risk|amplification/i);
});

test('the count says what is on screen, twinned slugs and all (A16)', () => {
  // Two occupations may share a slug, so the set of slugs is smaller than the
  // list of rows. The message must describe the rows, which is what is drawn.
  const twins = [
    ...INDEX,
    { t: 'nurse assistant', s: 'nurse-assistant', c: '3221', mg: 'Technicians', a: 5.5, m: 5.9, q: 'STABLE', alt: [] },
  ];
  const twinModel = buildModel(GROUPS, twins);
  const filter = filterMatches(twinModel, 'nurse assistant');
  assert.equal(filter.slugs.size < filter.hits.length, true);
  assert.equal(filter.shown, filter.hits.length);
  assert.equal(filter.capped, false);
  assert.equal(matchMessage(filter), `${filter.total} jobs match.`);
  assert.equal(resultRows(twinModel, filter).length, filter.hits.length);
});

/* --- C12: a skill can sit near the line too -------------------------------- */

// `p` is [substitution, assistance, machinery]; the class in `c` is what the
// pipeline settled on, and is never recomputed here.
const NEAR_PORTFOLIO = {
  skills: {
    n1: { t: 'technical drawings', a: 6.2, m: 5.0, k: 1.0, c: 'I', p: [0.48, 0.30, 0.10] },
    n2: { t: 'writing code', a: 7.2, m: 9.1, k: 1.4, c: 'S', p: [0.91, 0.80, 0.02] },
    n3: { t: 'tending a press', a: 5.1, m: 3.2, k: 8.3, c: 'M', p: [0.10, 0.20, 0.52] },
    n4: { t: 'unscored craft', a: null, m: null, k: null, c: null, p: null },
  },
  occupations: [{ s: 'draughtsman', se: ['n1', 'n2'], so: ['n3', 'n4'] }],
};

const NEAR_SKILLS = skillRows(NEAR_PORTFOLIO,
  findOccupation(NEAR_PORTFOLIO, 'draughtsman'), 0.5);

test('a skill whose deciding chance is within five points is flagged (C12)', () => {
  const byTitle = Object.fromEntries(NEAR_SKILLS.map((row) => [row.title, row]));
  // 0.48 decided "not substituted", and it is two points from the cut.
  assert.equal(byTitle['technical drawings'].near, true);
  // 0.91 is nowhere near the cut.
  assert.equal(byTitle['writing code'].near, false);
  // 0.52 settled "machines can do", two points the other side.
  assert.equal(byTitle['tending a press'].near, true);
  assert.equal(byTitle['unscored craft'].near, false);
});

test('the flag travels to the table row, and the chip has words (C12)', () => {
  const rows = skillTableRows(NEAR_SKILLS, 'shares');
  const byTitle = Object.fromEntries(rows.map((row) => [row.title, row]));
  assert.equal(byTitle['technical drawings'].near, true);
  assert.equal(byTitle['writing code'].near, false);
  assert.match(SKILL_NEAR_TEXT, /within 5 points/);
  assert.doesNotMatch(SKILL_NEAR_TEXT, /at risk/i);
});

test('a quadrant set has no classes, so no skill is ever near the line (C12)', () => {
  const rows = skillRows(PORTFOLIO, findOccupation(PORTFOLIO, 'software-developer'));
  assert.equal(rows.every((row) => row.near === false), true);
  assert.equal(skillTableRows(rows).every((row) => row.near === false), true);
});

/* --- C7: the tree's own bars add up to 100 --------------------------------- */

test('a mix bar and a class bar each sum to 100 (C7)', () => {
  const sum = (segments) => segments
    .reduce((total, s) => total + Number(s.percent.replace('%', '')), 0);
  assert.equal(sum(mixSegments(GROUPS.all.q)), 100);
  assert.equal(sum(mixSegments(GROUPS['major:3'].q)), 100);
  assert.equal(sum(classSegments(SHARES_STATS)), 100);
  assert.equal(sum(skillMix(SHARES_SKILLS, 'shares').bars), 100);
});

test('a class holding one job out of many never reads "0%" (C7)', () => {
  const lopsided = mixSegments({ EVOLVE: 999, TRANSFORM: 1 });
  const byLabel = Object.fromEntries(lopsided.map((s) => [s.label, s.percent]));
  assert.equal(byLabel.Transform, '<1%');
  assert.equal(byLabel.Evolve, '100%');
  assert.equal(byLabel.Shrink, '0%');
});
