import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FILTER_LIMIT,
  SKILL_COLUMNS,
  SKILL_FILTERS,
  ancestorKeys,
  buildModel,
  childrenOf,
  filterMatches,
  filterSkillRows,
  findOccupation,
  groupSubtitle,
  horizontalMove,
  jobSummary,
  matchMessage,
  mixSegments,
  mixText,
  moveIndex,
  nearLineCaveat,
  readState,
  rowIndexOf,
  selectionId,
  selectionOf,
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
  assert.equal(segments[1].percent, '33%');
  assert.equal(segments[1].text, '1 of 3 jobs');
  assert.equal(segments[0].count, 0);
});

test('the compact mix reads largest first and skips empty boxes', () => {
  assert.equal(mixText(mixSegments(GROUPS.all.q)), '44% Evolve, 22% Transform, 22% Stable, 11% Shrink');
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
