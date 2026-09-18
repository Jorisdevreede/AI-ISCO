import test from 'node:test';
import assert from 'node:assert/strict';

import {
  NARROW_WIDTH,
  axisNames,
  axisTicks,
  breadcrumb,
  childGroups,
  colourModes,
  columnsFor,
  compareGroups,
  compareHref,
  contrastRatio,
  defaultColour,
  defaultSort,
  defaultView,
  describeTile,
  drivingSkills,
  exposureHeadings,
  exposureLists,
  groupSubtitle,
  indexBySlug,
  markerShapes,
  matchGroups,
  meanShareLabel,
  meanShareSentence,
  medianSentence,
  mixBars,
  mixSentence,
  namedRegions,
  nearLineCaveat,
  groupNearSentence,
  largestRemainder,
  plotPoint,
  plotShare,
  rangeSentence,
  rankedCountText,
  rankedMoreButtons,
  rankedPage,
  rankedRowSummary,
  rankedRows,
  rankedSorts,
  readState,
  regionOf,
  REGION_MARGIN,
  ruleLines,
  resolveGroup,
  scatterCaption,
  scatterPoints,
  shareAxisSentence,
  shareClassLegend,
  skillIdsOf,
  skillsNote,
  sortByColumn,
  sortKeyOf,
  sortWordOf,
  tableBody,
  tableSortOf,
  textColourFor,
  thresholdPoint,
  tileColour,
  treemapTiles,
  undecidedArea,
  groupViewHref,
  typeLegend,
  unknownGroupMessage,
  visualSummary,
} from '../../site/js/pages/groups-model.js';
import { TABLE_COLUMNS, tableRows } from '../../site/js/groupstats.js';
import { TYPE_ORDER } from '../../site/js/scheme.js';
import { INDEX } from './fixture.js';

// A stand-in for groups.json, shaped exactly like the real file. Synthetic
// numbers, made to match the occupations in fixture.js.
const GROUPS = {
  all: {
    label: 'All occupations', level: 'all', code: '', n: 9,
    q: { TRANSFORM: 2, STABLE: 2, EVOLVE: 4, SHRINK: 1 },
    auto: { mean: 5.3, p10: 2.8, p50: 5.2, p90: 8.6 },
    amp: { mean: 6.9, p10: 4.1, p50: 6.4, p90: 8.9 },
    top: ['bookkeeper', 'software-developer', 'missing-slug'],
    bottom: ['sommeliere', 'nursery-school-head-teacher'],
    skills: { auto: [{ id: 'aaaa1111', n: 12 }], amp: [{ id: 'bbbb2222', n: 9 }] },
    parent: null,
    children: ['major:1', 'major:2', 'major:3', 'major:5'],
  },
  'major:2': {
    label: 'Professionals', level: 'major', code: '2', n: 4,
    q: { TRANSFORM: 2, STABLE: 0, EVOLVE: 2, SHRINK: 0 },
    auto: { mean: 5.6, p10: 4.5, p50: 5.7, p90: 6.4 },
    amp: { mean: 7.6, p10: 6.0, p50: 7.4, p90: 8.9 },
    top: ['software-developer'],
    bottom: ['nurse-responsible-for-general-care'],
    skills: { auto: [{ id: 'aaaa1111', n: 3 }], amp: [] },
    parent: 'all',
    children: ['minor:251'],
  },
  'minor:251': {
    label: 'Software and applications developers', level: 'minor', code: '251', n: 2,
    q: { TRANSFORM: 2, STABLE: 0, EVOLVE: 0, SHRINK: 0 },
    auto: { mean: 6.3, p10: 6.1, p50: 6.3, p90: 6.4 },
    amp: { mean: 8.7, p10: 8.4, p50: 8.7, p90: 8.9 },
    top: [], bottom: [], skills: {}, parent: 'major:2', children: ['unit:2512'],
  },
  'unit:2512': {
    label: 'Software developers', level: 'unit', code: '2512', n: 2,
    q: { TRANSFORM: 2, STABLE: 0, EVOLVE: 0, SHRINK: 0 },
    auto: { mean: 6.3, p10: 6.1, p50: 6.3, p90: 6.4 },
    amp: { mean: 8.7, p10: 8.4, p50: 8.7, p90: 8.9 },
    top: [], bottom: [], skills: {}, parent: 'minor:251', children: [],
  },
  'major:1': {
    label: 'Managers', level: 'major', code: '1', n: 1,
    q: { TRANSFORM: 0, STABLE: 0, EVOLVE: 1, SHRINK: 0 },
    auto: { mean: 3.9, p10: 3.9, p50: 3.9, p90: 3.9 },
    amp: { mean: 6.6, p10: 6.6, p50: 6.6, p90: 6.6 },
    top: [], bottom: [], skills: {}, parent: 'all', children: [],
  },
};

const STATS = {
  occupations: 3039,
  near_line: { count: 1509, share: 0.4959 },
};

// --- the same two files under the shares scheme ----------------------------
//
// Synthetic, one occupation per type, every `sh` summing to 1, so a test can
// name a share without borrowing a number from the published run.

const SHARE_INDEX = [
  {
    t: 'subtitler', s: 'subtitler', c: '2643', mg: 'Professionals',
    a: 8.1, m: 6.2, k: 1.5, q: 'AUTOMATION_HEAVY', sh: [0.62, 0.24, 0.02, 0.12], nl: false,
    alt: [],
  },
  {
    t: 'data analyst', s: 'data-analyst', c: '2511', mg: 'Professionals',
    a: 6.8, m: 7.9, k: 2.0, q: 'TRANSFORMING', sh: [0.38, 0.34, 0.03, 0.25], nl: true, alt: [],
  },
  {
    t: 'translator', s: 'translator', c: '2643', mg: 'Professionals',
    a: 6.0, m: 7.2, k: 1.8, q: 'AUGMENTED', sh: [0.22, 0.46, 0.02, 0.3], nl: false, alt: [],
  },
  {
    t: 'welder', s: 'welder', c: '7212', mg: 'Craft and related trades workers',
    a: 3.2, m: 4.0, k: 7.5, q: 'MECHANISABLE', sh: [0.1, 0.18, 0.42, 0.3], nl: false, alt: [],
  },
  {
    t: 'rescue diver', s: 'rescue-diver', c: '5419', mg: 'Service and sales workers',
    a: 2.1, m: 3.4, k: 2.6, q: 'INSULATED_PHYSICAL', sh: [0.05, 0.12, 0.09, 0.74], nl: false,
    alt: [],
  },
  {
    t: 'child care worker', s: 'child-care-worker', c: '5311', mg: 'Service and sales workers',
    a: 2.6, m: 4.4, k: 1.9, q: 'INSULATED_PEOPLE', sh: [0.08, 0.2, 0.06, 0.66], nl: true, alt: [],
  },
  {
    t: 'cider fermentation operator', s: 'cider-fermentation-operator', c: '8160',
    mg: 'Plant and machine operators', a: 6.1, m: 5.1, k: 4.2, q: 'MIXED',
    sh: [0.16, 0.21, 0.28, 0.35], nl: false, alt: [],
  },
];

const SHARE_GROUPS = {
  all: {
    label: 'All occupations', level: 'all', code: '', n: 7,
    q: {
      AUTOMATION_HEAVY: 1, TRANSFORMING: 1, AUGMENTED: 1, MECHANISABLE: 1,
      INSULATED_PHYSICAL: 1, INSULATED_PEOPLE: 1, MIXED: 1,
    },
    auto: { mean: 5.0, p10: 2.2, p50: 6.0, p90: 7.9 },
    amp: { mean: 5.5, p10: 3.5, p50: 5.1, p90: 7.8 },
    mech: { mean: 3.1, p10: 1.6, p50: 2.0, p90: 6.8 },
    sh: [0.23, 0.25, 0.13, 0.39],
    top: ['subtitler', 'data-analyst'],
    bottom: ['rescue-diver', 'child-care-worker'],
    skills: {
      auto: [{ id: 'aaaa1111', n: 5 }, { id: 'cccc3333', n: 2 }],
      amp: [{ id: 'bbbb2222', n: 4 }, { id: 'aaaa1111', n: 5 }, { id: 'dddd4444', n: 1 }],
    },
    parent: null,
    children: ['major:2', 'major:5'],
  },
  'major:2': {
    label: 'Professionals', level: 'major', code: '2', n: 3,
    q: { AUTOMATION_HEAVY: 1, TRANSFORMING: 1, AUGMENTED: 1 },
    auto: { mean: 7.0, p10: 6.1, p50: 6.8, p90: 8.0 },
    amp: { mean: 7.1, p10: 6.3, p50: 7.2, p90: 7.8 },
    mech: { mean: 1.8, p10: 1.5, p50: 1.8, p90: 2.0 },
    sh: [0.41, 0.35, 0.02, 0.22],
    top: ['subtitler'], bottom: ['translator'],
    skills: { auto: [{ id: 'aaaa1111', n: 3 }], amp: [{ id: 'bbbb2222', n: 2 }] },
    parent: 'all', children: [],
  },
  'major:5': {
    label: 'Service and sales workers', level: 'major', code: '5', n: 2,
    q: { INSULATED_PHYSICAL: 1, INSULATED_PEOPLE: 1 },
    auto: { mean: 2.4, p10: 2.2, p50: 2.4, p90: 2.5 },
    amp: { mean: 3.9, p10: 3.5, p50: 3.9, p90: 4.3 },
    mech: { mean: 2.3, p10: 2.0, p50: 2.3, p90: 2.5 },
    sh: [0.07, 0.16, 0.07, 0.7],
    top: [], bottom: [], skills: {}, parent: 'all', children: [],
  },
};

const SHARE_STATS = {
  scheme: 'shares', occupations: 3039, near_line: { count: 995, share: 0.327 },
};

/** skill_index rows by id, as the page holds them once that file has arrived. */
const SKILLS = new Map([
  ['aaaa1111', { id: 'aaaa1111', t: 'record test data', c: 'S' }],
  ['bbbb2222', { id: 'bbbb2222', t: 'liaise with colleagues', c: 'A' }],
  ['cccc3333', { id: 'cccc3333', t: 'operate a press', c: 'M' }],
  ['dddd4444', { id: 'dddd4444', t: 'comfort a patient', c: 'I' }],
]);

const SHARES = 'shares';

const rowsOf = (code) => INDEX.filter((row) => row.c.startsWith(code));
const sharesIn = (code) => SHARE_INDEX.filter((row) => row.c.startsWith(code));

/** Nothing written for the four boxes may reach a page showing seven types. */
const QUADRANT_WORDS = /quadrant|\bbox\b|cut-off of 6|at risk/i;

/* --- URL state ----------------------------------------------------------- */

test('an empty hash is the "all" group', () => {
  const state = readState('', 1440);
  assert.equal(state.key, 'all');
  assert.equal(state.compare, null);
});

test('the hash names the group and the view', () => {
  const state = readState('#g=major:2&view=treemap', 1440);
  assert.deepEqual({ key: state.key, view: state.view }, { key: 'major:2', view: 'treemap' });
});

test('the default view is ranked on a phone and scatter on a desktop', () => {
  assert.equal(defaultView(390), 'ranked');
  assert.equal(defaultView(NARROW_WIDTH - 1), 'ranked');
  assert.equal(defaultView(NARROW_WIDTH), 'scatter');
  assert.equal(readState('#g=major:2', 390).view, 'ranked');
  assert.equal(readState('#g=major:2', 1440).view, 'scatter');
});

test('a view the page does not have falls back to the default', () => {
  assert.equal(readState('#g=all&view=sunburst', 1440).view, 'scatter');
});

// A14: the table's sort belongs in the URL the Copy link button copies.

test('the table sort and its direction survive a shared link', () => {
  const here = readState('#g=major:2&view=table&sort=m&dir=asc', 1440);
  assert.equal(here.view, 'table');
  assert.deepEqual({ sort: here.sort, descending: here.descending },
    { sort: 'm', descending: false });
  assert.deepEqual(tableSortOf(here), { key: 'm', descending: false });
  assert.deepEqual(tableSortOf(readState('#g=major:2&view=table&sort=a', 1440)),
    { key: 'a', descending: true });
  // No sort in the hash is each scheme's own default.
  const plain = readState('#g=major:2&view=table', 1440);
  assert.deepEqual({ sort: plain.sort, descending: plain.descending },
    { sort: null, descending: null });
  assert.deepEqual(tableSortOf(plain), { key: 'a', descending: true });
  assert.deepEqual(tableSortOf(plain, SHARES), { key: 's0', descending: true });
  // A column this scheme does not have falls back rather than sorting by junk.
  assert.deepEqual(tableSortOf(readState('#sort=s3', 1440)), { key: 'a', descending: true });
  assert.deepEqual(tableSortOf(readState('#sort=s3&dir=asc', 1440), SHARES),
    { key: 's3', descending: false });
});

test('a view link carries a sort only when there is one to carry', () => {
  assert.equal(groupViewHref('major:2', 'table'), 'groups.html#g=major:2&view=table');
  assert.equal(groupViewHref('major:2', 'table', { key: 'm', descending: false }),
    'groups.html#g=major:2&view=table&sort=amplification&dir=asc');
  assert.equal(groupViewHref('major:2', 'table', { key: 'k', descending: true }, SHARES),
    'groups.html#g=major:2&view=table&sort=mechanisation&dir=desc');
  assert.equal(groupViewHref('all'), 'groups.html#g=all');
});

// C13: a shared sort URL has to be readable by the person who receives it.

test('the sort in a link is a word, and the old keys still resolve', () => {
  assert.equal(sortWordOf('s1', SHARES), 'assists');
  assert.equal(sortWordOf('a', SHARES), 'substitution');
  assert.equal(sortWordOf('a'), 'automation');
  assert.equal(sortWordOf('q', SHARES), 'type');
  assert.equal(sortWordOf('q'), 'quadrant');
  assert.equal(sortKeyOf('assists', SHARES), 's1');
  assert.equal(sortKeyOf('s1', SHARES), 's1', 'a link copied before this change still works');
  assert.equal(sortKeyOf('amplification'), 'm');
  assert.equal(sortKeyOf('m'), 'm');
  assert.equal(sortKeyOf('assists'), null, 'a column this scheme does not have names nothing');
  assert.equal(sortKeyOf(null, SHARES), null);
  // Every column of both schemes has a word, and no two share one.
  for (const scheme of [undefined, SHARES]) {
    const words = columnsFor(scheme).map((column) => sortWordOf(column.key, scheme));
    assert.equal(new Set(words).size, words.length, `${words}`);
    for (const word of words) assert.match(word, /^[a-z-]+$/);
  }
  assert.deepEqual(tableSortOf(readState('#view=table&sort=assists&dir=asc', 1440), SHARES),
    { key: 's1', descending: false });
  assert.deepEqual(tableSortOf(readState('#view=table&sort=s1&dir=asc', 1440), SHARES),
    { key: 's1', descending: false });
});

test('two group keys mean a comparison', () => {
  assert.deepEqual(readState('#a=major:2&b=major:1', 1440).compare, { a: 'major:2', b: 'major:1' });
  assert.equal(readState('#a=major:2', 1440).compare, null);
  assert.equal(compareHref('major:2', 'major:1'), 'groups.html#a=major:2&b=major:1');
});

/* --- the group itself ----------------------------------------------------- */

test('a group key resolves, and an unknown one says so', () => {
  assert.equal(resolveGroup(GROUPS, 'major:2').label, 'Professionals');
  assert.equal(resolveGroup(GROUPS, 'major:9'), null);
  assert.match(unknownGroupMessage('major:9'), /don't have a group called/);
});

test('the subtitle carries the level, the ISCO code and the size', () => {
  assert.equal(groupSubtitle(GROUPS['major:2']), 'Major group 2 · ISCO-08 · 4 jobs');
  assert.equal(groupSubtitle(GROUPS['unit:2512']), 'Unit group 2512 · ISCO-08 · 2 jobs');
  assert.equal(groupSubtitle(GROUPS.all), 'All occupations · 9 jobs');
});

test('the breadcrumb walks up to "all"', () => {
  assert.deepEqual(breadcrumb(GROUPS, 'unit:2512').map((step) => step.key),
    ['all', 'major:2', 'minor:251', 'unit:2512']);
  assert.deepEqual(breadcrumb(GROUPS, 'all').map((step) => step.label), ['All occupations']);
  assert.deepEqual(breadcrumb(GROUPS, 'major:9'), []);
});

test('children are listed with their labels and sizes', () => {
  assert.deepEqual(childGroups(GROUPS, 'major:2'),
    [{ key: 'minor:251', label: 'Software and applications developers', n: 2 }]);
  assert.deepEqual(childGroups(GROUPS, 'unit:2512'), []);
  // "major:3" and "major:5" are in children but not in the file: skipped.
  assert.deepEqual(childGroups(GROUPS, 'all').map((child) => child.key), ['major:1', 'major:2']);
});

test('the picker matches group labels, exact first', () => {
  const hits = matchGroups(GROUPS, 'software');
  assert.deepEqual(hits.map((hit) => hit.key), ['minor:251', 'unit:2512']);
  assert.equal(matchGroups(GROUPS, 'Professionals')[0].key, 'major:2');
  assert.deepEqual(matchGroups(GROUPS, ''), []);
  assert.deepEqual(matchGroups(GROUPS, 'accountancy'), []);
  assert.equal(matchGroups(GROUPS, 'o', 2).length, 2);
});

/* --- how this group splits ------------------------------------------------ */

test('the quadrant bars keep a fixed order and carry counts and percentages', () => {
  const bars = mixBars(GROUPS['major:2']);
  assert.deepEqual(bars.map((bar) => bar.code), ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
  assert.deepEqual(bars.map((bar) => bar.label), ['Transform', 'Stable', 'Evolve', 'Shrink']);
  assert.deepEqual(bars.map((bar) => bar.count), [2, 0, 2, 0]);
  assert.deepEqual(bars.map((bar) => bar.percent), ['50%', '0%', '50%', '0%']);
  assert.equal(bars[0].text, '2 of 4 jobs');
});

test('an empty group has zero shares rather than NaN', () => {
  const bars = mixBars({ q: {} });
  assert.deepEqual(bars.map((bar) => bar.percent), ['0%', '0%', '0%', '0%']);
  assert.equal(mixSentence({ q: {} }), '0% Transform, 0% Stable, 0% Evolve, 0% Shrink.');
});

test('a shares set splits into the seven types, in the rule order, each named', () => {
  const bars = mixBars(SHARE_GROUPS.all, SHARES);
  assert.deepEqual(bars.map((bar) => bar.code), TYPE_ORDER);
  assert.equal(bars.length, 7);
  assert.equal(bars[4].label, 'Insulated by physical work');
  assert.deepEqual(bars.map((bar) => bar.count), [1, 1, 1, 1, 1, 1, 1]);
  assert.equal(bars[0].text, '1 of 7 jobs');
  // A type the group has none of still gets a named row at 0%.
  const professionals = mixBars(SHARE_GROUPS['major:2'], SHARES);
  assert.equal(professionals.length, 7);
  assert.equal(professionals[6].percent, '0%');
  assert.doesNotMatch(mixSentence(SHARE_GROUPS.all, SHARES), QUADRANT_WORDS);
});

test('the group\'s mean shares read as one sentence with its own jobs counted', () => {
  const sentence = meanShareSentence(SHARE_GROUPS['major:2']);
  assert.match(sentence, /^Across the 3 jobs in this group, on average /);
  assert.match(sentence, /AI can take over 41% · AI assists 35% · machines 2% · stays human 22%/);
  assert.match(meanShareLabel(SHARE_GROUPS['major:2']), /^What the jobs in Professionals are made of/);
  // A quadrant group has no shares, so the figure is not offered at all.
  assert.equal(meanShareSentence(GROUPS['major:2']), '');
  assert.equal(meanShareLabel(GROUPS['major:2']), '');
});

// A10: 35 of 604 groups printed shares that summed to 99 or 101, and one row
// read "1 of 869 jobs · 0%".

test('the printed shares always add up to 100', () => {
  assert.deepEqual(largestRemainder([1, 1, 1]), [34, 33, 33]);
  assert.deepEqual(largestRemainder([151, 79, 638, 1]).reduce((a, b) => a + b, 0), 100);
  assert.deepEqual(largestRemainder([]), []);
  assert.deepEqual(largestRemainder([0, 0]), [0, 0]);
  assert.deepEqual(largestRemainder([5, null, -2, NaN]), [100, 0, 0, 0]);
  for (const group of [GROUPS.all, GROUPS['major:2'], SHARE_GROUPS.all, SHARE_GROUPS['major:2']]) {
    const printed = mixBars(group).map((bar) => Number(bar.percent.replace(/[<%]/g, '')));
    assert.equal(printed.reduce((a, b) => a + b, 0), 100, JSON.stringify(group.q));
  }
});

test('a class with one job in it never reads "0%"', () => {
  const lopsided = { n: 869, q: { TRANSFORM: 151, STABLE: 79, EVOLVE: 638, SHRINK: 1 } };
  const bars = mixBars(lopsided);
  // 17.4 + 9.1 + 73.4 + 0.1 rounded on its own gave 99; the remainder goes to
  // the largest fraction, so Evolve reads 74% and the four add up.
  assert.deepEqual(bars.map((bar) => bar.percent), ['17%', '9%', '74%', '<1%']);
  assert.equal(bars[3].text, '1 of 869 jobs');
  assert.match(mixSentence(lopsided), /<1% Shrink/);
  // A class with nothing in it still reads 0%, because that is true.
  assert.equal(mixBars({ q: { TRANSFORM: 1, STABLE: 0, EVOLVE: 0, SHRINK: 0 } })[1].percent, '0%');
});

// A11: the site-wide near-the-line figure inside a group panel reads as a fact
// about that group, and is usually the wrong one.

test('the near-the-line sentence in a group panel counts that group', () => {
  const sentence = groupNearSentence({ n: 869, near: 252 }, undefined);
  assert.match(sentence, /252 of 869 jobs \(29%\) in this group/);
  assert.match(sentence, /within 0.5 of a cut-off/);
  assert.match(sentence, /band, not a count/);
  const shares = groupNearSentence({ n: 869, near: 252 }, SHARES);
  assert.match(shares, /252 of 869 jobs \(29%\) in this group/);
  assert.match(shares, /moving any one of their four shares by 5 points/);
  assert.doesNotMatch(shares, QUADRANT_WORDS);
  // A build without the count says nothing rather than borrowing the site's.
  assert.equal(groupNearSentence({ n: 869 }, SHARES), '');
  assert.equal(groupNearSentence({ n: 0, near: 0 }, SHARES), '');
  assert.equal(groupNearSentence(null, SHARES), '');
});

test('the caveat quotes the near-the-line share and claims nothing else', () => {
  const caveat = nearLineCaveat(STATS);
  assert.match(caveat, /1,509 of 3,039 jobs \(50%\)/);
  assert.match(caveat, /band, not a count/);
  assert.doesNotMatch(caveat, /second|run|agree|4 in 10/i);
  assert.match(nearLineCaveat(null), /band, not a count/);
  assert.match(nearLineCaveat(SHARE_STATS), /995 of 3,039 jobs \(33%\)/);
  assert.doesNotMatch(nearLineCaveat(SHARE_STATS), QUADRANT_WORDS);
});

/* --- the four views ------------------------------------------------------- */

test('ranked rows carry both dots on one 1-10 axis', () => {
  const rows = rankedRows(rowsOf('2512'), 'automation');
  assert.deepEqual(rows.map((row) => row.slug),
    ['software-developer', 'embedded-systems-software-developer']);
  assert.equal(rows[0].automation.text, '6.4');
  assert.equal(rows[0].amplification.text, '8.9');
  assert.equal(rows[0].quadrantLabel, 'Transform');
  assert.ok(Math.abs(rows[0].automation.percent - ((6.4 - 1) / 9) * 100) < 1e-9);
  assert.ok(rows[0].amplification.percent > rows[0].automation.percent);
  // A18: the summary follows the title and the ISCO code inside one link.
  assert.equal(rankedRowSummary(rows[0]), ', automation 6.4, amplification 8.9, Transform');
});

test('ranked rows sort by amplification and by title too', () => {
  assert.equal(rankedRows(rowsOf(''), 'amplification')[0].slug, 'software-developer');
  assert.equal(rankedRows(rowsOf(''), 'title')[0].title, 'bookkeeper');
});

test('an unscored job still gets a row, with no dot position', () => {
  const [row] = rankedRows([{ t: 'mystery', s: 'm', c: '0000', a: null, m: null, q: null }], 'title');
  assert.equal(row.automation.text, 'Not scored');
  assert.equal(row.automation.percent, null);
  assert.equal(row.quadrantLabel, 'Not scored');
});

test('each scheme offers its own sorts, its own default first', () => {
  assert.deepEqual(rankedSorts().map((sort) => sort.key),
    ['automation', 'amplification', 'title']);
  assert.deepEqual(rankedSorts(SHARES).map((sort) => sort.key),
    ['substituted', 'assisted', 'mechanised', 'insulated', 'title']);
  assert.deepEqual(rankedSorts(SHARES).map((sort) => sort.label), [
    'Most can be taken over', 'Most assisted', 'Most mechanical', 'Most stays human', 'A to Z',
  ]);
  assert.equal(defaultSort(), 'automation');
  assert.equal(defaultSort(SHARES), 'substituted');
  for (const sort of rankedSorts(SHARES)) assert.doesNotMatch(sort.label, QUADRANT_WORDS);
});

test('a shares row leads with its four shares and names its type', () => {
  const rows = rankedRows(SHARE_INDEX, 'substituted', SHARES);
  assert.deepEqual(rows.slice(0, 3).map((row) => row.slug),
    ['subtitler', 'data-analyst', 'translator']);
  assert.deepEqual(rows[0].shares, [0.62, 0.24, 0.02, 0.12]);
  assert.equal(rows[0].typeShort, 'Automation-heavy');
  assert.equal(rows[0].quadrantLabel, 'Automation-heavy');
  assert.match(rows[0].shareText, /^AI can take over 62%/);
  assert.equal(rankedRowSummary(rows[0], SHARES),
    `, ${rows[0].shareText}, Automation-heavy`);
  assert.doesNotMatch(rankedRowSummary(rows[0], SHARES), QUADRANT_WORDS);
  for (const scheme of [undefined, SHARES]) {
    assert.match(rankedRowSummary(rows[0], scheme), /^, /, 'names must not run together');
  }
});

test('every share sorts, and A to Z still sorts by title', () => {
  const first = (key) => rankedRows(SHARE_INDEX, key, SHARES)[0].slug;
  assert.equal(first('substituted'), 'subtitler');
  assert.equal(first('assisted'), 'translator');
  assert.equal(first('mechanised'), 'welder');
  assert.equal(first('insulated'), 'rescue-diver');
  assert.equal(first('title'), 'child care worker' && 'child-care-worker');
  // A row with no shares sorts last rather than ahead of everything.
  const withGap = [...SHARE_INDEX, { t: 'mystery', s: 'mystery', c: '0000', q: null }];
  assert.equal(rankedRows(withGap, 'substituted', SHARES).at(-1).slug, 'mystery');
});

// Review note 2: 869 rows is 114,000 px of phone.

test('the ranked list starts at one page and can be opened up', () => {
  const rows = Array.from({ length: 120 }, (_, i) => ({ t: `job ${i}` }));
  const first = rankedPage(rows, undefined);
  assert.equal(first.rows.length, 50);
  assert.deepEqual({ visible: first.visible, total: first.total, more: first.more, next: first.next },
    { visible: 50, total: 120, more: 70, next: 50 });
  assert.equal(rankedCountText(first), 'Showing 50 of 120 jobs.');
  assert.deepEqual(rankedMoreButtons(first).map((button) => button.label),
    ['Show 50 more', 'Show all 120']);
  const second = rankedPage(rows, 100);
  assert.equal(second.rows.length, 100);
  assert.equal(second.next, 20, 'the last step never promises more rows than there are');
  assert.deepEqual(rankedMoreButtons(second).map((button) => button.label), ['Show 20 more']);
  const whole = rankedPage(rows, 500);
  assert.deepEqual({ visible: whole.visible, more: whole.more }, { visible: 120, more: 0 });
  assert.equal(rankedCountText(whole), 'Showing all 120 jobs.');
  assert.deepEqual(rankedMoreButtons(whole), []);
});

test('a list that already fits offers no buttons at all', () => {
  const page = rankedPage([{ t: 'one' }], undefined);
  assert.deepEqual(page.rows.length, 1);
  assert.equal(rankedCountText(page), 'Showing all 1 job.');
  assert.deepEqual(rankedMoreButtons(page), []);
  assert.deepEqual(rankedPage([], 50).rows, []);
});

test('scatter points put automation across and amplification up', () => {
  const box = { width: 400, height: 400, pad: 40 };
  const points = scatterPoints(rowsOf('2512'), box);
  assert.equal(points.length, 2);
  const [developer] = points;
  assert.equal(developer.slug, 'software-developer');
  assert.ok(Math.abs(developer.x - (40 + ((6.4 - 1) / 9) * 320)) < 1e-9);
  assert.ok(Math.abs(developer.y - (360 - ((8.9 - 1) / 9) * 320)) < 1e-9);
  assert.match(developer.label, /software developer, automation 6.4, amplification 8.9, Transform/);
});

test('the cut-off lines cross where a score of 6 lands', () => {
  const box = { width: 400, height: 400, pad: 40 };
  assert.deepEqual(thresholdPoint(box), plotPoint(6, 6, box));
});

test('unscored jobs are left off the scatter', () => {
  const rows = [{ t: 'mystery', s: 'm', c: '0000', a: null, m: 7 }];
  assert.deepEqual(scatterPoints(rows, { width: 400, height: 400, pad: 40 }), []);
});

test('a shares scatter plots the two shares the type rules cut', () => {
  assert.deepEqual(axisNames(), { x: 'Automation', y: 'Amplification' });
  assert.deepEqual(axisNames(SHARES), { x: 'AI can take over', y: 'AI assists' });
  assert.deepEqual(axisTicks(), { low: '1', high: '10' });
  assert.deepEqual(axisTicks(SHARES), { low: '0%', high: '100%' });
  const box = { width: 400, height: 400, pad: 40 };
  const [point] = scatterPoints(sharesIn('2643'), box, SHARES);
  // subtitler: 62% can be taken over, 24% assisted, in a 320 px inner box.
  assert.ok(Math.abs(point.x - (40 + 0.62 * 320)) < 1e-9);
  assert.ok(Math.abs(point.y - (360 - 0.24 * 320)) < 1e-9);
  assert.deepEqual(plotShare(0, 0, box), { x: 40, y: 360 });
  assert.deepEqual(plotShare(1, 1, box), { x: 360, y: 40 });
  assert.match(point.label, /^subtitler, AI can take over 62% · AI assists 24%/);
  assert.match(point.label, /Automation-heavy\.$/);
  assert.doesNotMatch(point.label, QUADRANT_WORDS);
  assert.doesNotMatch(point.label, /substitution|assistance/);
  // A job with no shares cannot be placed on these axes at all.
  assert.deepEqual(scatterPoints([{ t: 'x', s: 'x', a: 5, m: 5, q: 'MIXED' }], box, SHARES), []);
});

test('the rule lines are the cuts of the type rules, each labelled', () => {
  const box = { width: 400, height: 400, pad: 40 };
  const rules = ruleLines(box, SHARES);
  assert.deepEqual(rules.map((rule) => rule.label), ['50%', '30%', '20%', '30%']);
  const [half, third, assisted, insulated] = rules;
  // The two x cuts run the full height; the y cuts only where the rule applies.
  assert.equal(half.x1, half.x2);
  assert.ok(Math.abs(half.x1 - (40 + 0.5 * 320)) < 1e-9);
  assert.equal(third.y1, box.pad);
  assert.equal(assisted.y1, assisted.y2);
  assert.ok(Math.abs(assisted.x1 - (40 + 0.3 * 320)) < 1e-9, 'the 20% rule starts at x = 30%');
  assert.equal(assisted.x2, 360);
  assert.equal(insulated.x1, 40, 'the 30% rule runs left of x = 30%');
  assert.ok(Math.abs(insulated.x2 - (40 + 0.3 * 320)) < 1e-9);
  assert.deepEqual(ruleLines(box), []);
  // C15: both y lines carry a label, and neither sits on top of the other.
  const labelled = rules.map((rule) => `${rule.label}@${Math.round(rule.textX)},${Math.round(rule.textY)}`);
  assert.equal(new Set(labelled).size, 4, labelled.join(' '));
  assert.equal(insulated.anchor, 'end', 'the 30% line reaches the axis, so it labels the tick');
  assert.equal(assisted.anchor, 'start');
  assert.ok(insulated.textX < box.pad, 'and sits in the tick column beside the axis');
  assert.ok(Math.abs(insulated.textY - plotShare(0, 0.3, box).y) <= 6);
  assert.ok(Math.abs(assisted.textY - plotShare(0, 0.2, box).y) <= 6);
});

// C5: the two axes settle three of the seven types; the lines must not promise
// more than that.

test('the chart names only the regions its two axes can settle', () => {
  const box = { width: 400, height: 400, pad: 40 };
  const regions = namedRegions(box, SHARES);
  assert.deepEqual(regions.map((region) => region.code),
    ['AUTOMATION_HEAVY', 'TRANSFORMING', 'AUGMENTED']);
  assert.deepEqual(regions.map((region) => region.label),
    ['Automation-heavy', 'Transforming', 'Augmented']);
  const [heavy, changing, helped] = regions;
  assert.ok(Math.abs(heavy.x - (40 + 0.5 * 320)) < 1e-9, 'right of the 50% line');
  assert.ok(Math.abs(heavy.width - 0.5 * 320) < 1e-9);
  assert.ok(Math.abs(changing.x - (40 + 0.3 * 320)) < 1e-9);
  assert.ok(Math.abs(changing.width - 0.2 * 320) < 1e-9, 'between 30% and 50%');
  assert.ok(Math.abs(changing.height - 0.8 * 320) < 1e-9, 'above the 20% line');
  assert.ok(Math.abs(helped.height - 0.7 * 320) < 1e-9, 'above the 30% line');
  // Each name sits inside its own region.
  for (const region of regions) {
    assert.ok(region.textX > region.x && region.textX < region.x + region.width);
    assert.ok(region.textY >= region.y && region.textY <= region.y + region.height);
  }
  assert.deepEqual(namedRegions(box), []);
});

test('the rest of the plot is shaded and says what decides it', () => {
  const box = { width: 400, height: 400, pad: 40 };
  const area = undecidedArea(box, SHARES);
  assert.equal(area.rects.length, 2);
  const [left, strip] = area.rects;
  assert.ok(Math.abs(left.width - 0.3 * 320) < 1e-9, 'left of 30%');
  assert.ok(Math.abs(left.height - 0.3 * 320) < 1e-9, 'below 30% assisted');
  assert.ok(Math.abs(strip.width - 0.2 * 320) < 1e-9, '30% to 50%');
  assert.ok(Math.abs(strip.height - 0.2 * 320) < 1e-9, 'below 20% assisted');
  assert.equal(left.y + left.height, box.height - box.pad, 'both reach the x axis');
  assert.equal(strip.y + strip.height, box.height - box.pad);
  assert.match(area.label, /other two shares/);
  assert.equal(undecidedArea(box), null);
});

test('every job inside a named region really is that type', () => {
  // The three rules of docs/scoring-v2.md, read on these two axes only.
  assert.equal(regionOf([0.62, 0.24, 0.02, 0.12]), 'AUTOMATION_HEAVY');
  assert.equal(regionOf([0.38, 0.34, 0.03, 0.25]), 'TRANSFORMING');
  assert.equal(regionOf([0.22, 0.46, 0.02, 0.3]), 'AUGMENTED');
  for (const row of SHARE_INDEX) {
    const region = regionOf(row.sh);
    if (region) assert.equal(region, row.q, `${row.t} sits in ${region} but is ${row.q}`);
  }
  // The four types the chart cannot settle are never claimed by a region.
  for (const row of SHARE_INDEX.filter((r) => !['AUTOMATION_HEAVY', 'TRANSFORMING', 'AUGMENTED'].includes(r.q))) {
    assert.equal(regionOf(row.sh), null, `${row.t} (${row.q}) must not be claimed`);
  }
});

test('a job sitting on a cut is claimed by no region, because rounding moved it', () => {
  assert.equal(REGION_MARGIN, 0.005);
  // Published to two decimals: a true 0.2951 reads 0.30 and is not over the cut.
  assert.equal(regionOf([0.3, 0.64, 0.0, 0.06]), null, 'exactly on the 30% line');
  assert.equal(regionOf([0.5, 0.3, 0.1, 0.1]), null, 'exactly on the 50% line');
  assert.equal(regionOf([0.2, 0.3, 0.2, 0.3]), null, 'exactly on the 30% assisted line');
  assert.equal(regionOf([0.4, 0.2, 0.2, 0.2]), null, 'exactly on the 20% assisted line');
  assert.equal(regionOf([0.4, 0.21, 0.19, 0.2]), 'TRANSFORMING', 'clear of both');
  assert.equal(regionOf(null), null);
  assert.equal(regionOf([0.5, 0.5]), null, 'not four shares at all');
});

test('the caption says which types the lines decide and which they do not', () => {
  const caption = scatterCaption(SHARES);
  assert.match(caption, /three types only/);
  assert.match(caption, /Automation-heavy right of 50%/);
  assert.match(caption, /Transforming between 30% and 50% and above 20%/);
  assert.match(caption, /Augmented left of 30% and above 30%/);
  assert.match(caption, /other four types/);
  assert.match(caption, /machines and stays-human shares/);
  assert.match(caption, /shaded area/);
  assert.doesNotMatch(caption, QUADRANT_WORDS);
  assert.equal(scatterCaption(), '');
});

test('the scatter summary gives the range of each axis it actually draws', () => {
  const sentence = shareAxisSentence(sharesIn('2'));
  assert.equal(sentence, 'AI can take over runs 22% to 62%, AI assists runs 24% to 46%.');
  assert.equal(shareAxisSentence([]), 'No job in this group has shares.');
  const summary = visualSummary('scatter', SHARE_GROUPS['major:2'], sharesIn('2'), SHARES);
  assert.match(summary, /the share AI can take over across and the share AI assists with up/);
  assert.match(summary, /AI can take over runs 22% to 62%/);
  assert.doesNotMatch(summary, /1 to 10/, 'the axes are shares, not the display scores');
});

test('the scatter legend names only the types on screen, in rule order', () => {
  assert.deepEqual(typeLegend(SHARE_INDEX, SHARES).map((entry) => entry.code), TYPE_ORDER);
  assert.deepEqual(typeLegend(sharesIn('2'), SHARES).map((entry) => entry.label),
    ['Automation-heavy', 'Transforming', 'Augmented']);
  // The four boxes name themselves in the corners instead, so no legend there.
  assert.deepEqual(typeLegend(INDEX), []);
});

test('each type has its own marker outline, so colour is never the only signal', () => {
  const shapes = markerShapes(4);
  assert.deepEqual(shapes.map((shape) => shape.code), TYPE_ORDER);
  assert.equal(new Set(shapes.map((shape) => shape.d)).size, TYPE_ORDER.length);
  for (const shape of shapes) {
    assert.match(shape.d, /^M [-\d. ]/);
    assert.doesNotMatch(shape.d, /NaN|undefined/);
    assert.doesNotMatch(shape.d, /\d\.\d{3}/, 'coordinates are rounded');
  }
  assert.notEqual(markerShapes(4)[0].d, markerShapes(8)[0].d);
});

test('the table keeps the shared columns, and adds the shares under this scheme', () => {
  assert.deepEqual(columnsFor(), TABLE_COLUMNS);
  const columns = columnsFor(SHARES);
  assert.deepEqual(columns.map((column) => column.key),
    ['t', 'c', 's0', 's1', 's2', 's3', 'a', 'm', 'k', 'q']);
  assert.deepEqual(columns.map((column) => column.label), [
    'Job', 'ISCO', 'AI can take over', 'AI assists', 'Machines', 'Stays human',
    'AI substitution', 'AI assistance', 'Machine automation', 'Type',
  ]);
  for (const column of columns) assert.doesNotMatch(column.label, QUADRANT_WORDS);
});

test('the table body is the shared one under quadrants, and fuller under shares', () => {
  assert.deepEqual(tableBody(rowsOf('2512')), tableRows(rowsOf('2512')));
  const [row] = tableBody(sharesIn('2643'), SHARES);
  assert.equal(row.slug, 'subtitler');
  assert.deepEqual(row.cells.map((cell) => cell.text),
    ['subtitler', '2643', '62%', '24%', '2%', '12%', '8.1', '6.2', '1.5', 'Automation-heavy']);
  const [gap] = tableBody([{ t: 'mystery', s: 'm', c: '0000' }], SHARES);
  assert.deepEqual(gap.cells.map((cell) => cell.text).slice(2, 9),
    ['Not scored', 'Not scored', 'Not scored', 'Not scored', 'Not scored', 'Not scored',
      'Not scored']);
});

test('the table sorts by any column, both ways, ties by title', () => {
  const rows = rowsOf('');
  assert.equal(sortByColumn(rows, 'a', true)[0].s, 'bookkeeper');
  assert.equal(sortByColumn(rows, 'a', false)[0].s, 'sommeliere');
  assert.equal(sortByColumn(rows, 'c', false)[0].s, 'nursery-school-head-teacher');
  assert.equal(sortByColumn(rows, 'q', false)[0].q, 'EVOLVE');
  assert.equal(sortByColumn(rows, 't', false)[0].s, 'bookkeeper');
  assert.equal(sortByColumn(rows, 'a', true).length, rows.length);
});

test('the share columns and machine automation sort too', () => {
  assert.equal(sortByColumn(SHARE_INDEX, 's0', true)[0].s, 'subtitler');
  assert.equal(sortByColumn(SHARE_INDEX, 's3', true)[0].s, 'rescue-diver');
  // Both sit at 2% machines, so the tie breaks by title, as every column does.
  assert.deepEqual(sortByColumn(SHARE_INDEX, 's2', false).slice(0, 2).map((row) => row.s),
    ['subtitler', 'translator']);
  assert.equal(sortByColumn(SHARE_INDEX, 'k', true)[0].s, 'welder');
  assert.equal(sortByColumn(SHARE_INDEX, 'q', false)[0].q, 'AUGMENTED');
  const withGap = [...SHARE_INDEX, { t: 'mystery', s: 'mystery', c: '0000' }];
  assert.equal(sortByColumn(withGap, 's0', true).at(-1).s, 'mystery');
  assert.equal(sortByColumn(withGap, 's0', false).at(-1).s, 'mystery');
});

test('sorting the table never mutates the caller\'s array', () => {
  const before = INDEX.map((row) => row.s);
  sortByColumn(INDEX, 'a', true);
  assert.deepEqual(INDEX.map((row) => row.s), before);
  const shares = SHARE_INDEX.map((row) => row.s);
  rankedRows(SHARE_INDEX, 'substituted', SHARES);
  assert.deepEqual(SHARE_INDEX.map((row) => row.s), shares);
});

test('unscored rows sort last, whichever way the column goes', () => {
  const rows = [{ t: 'a', a: null }, { t: 'b', a: 3 }];
  assert.equal(sortByColumn(rows, 'a', false)[0].t, 'b');
  assert.equal(sortByColumn(rows, 'a', true)[1].t, 'a');
});

/* --- treemap -------------------------------------------------------------- */

test('a group with children gives group tiles, sized by job count', () => {
  const tiles = treemapTiles(GROUPS, INDEX, 'major:2');
  assert.deepEqual(tiles.map((tile) => tile.kind), ['group']);
  assert.equal(tiles[0].value, 2);
  assert.equal(tiles[0].id, 'minor:251');
  assert.match(tiles[0].detail, /2 jobs · mostly Transform/);
});

test('a unit group gives one tile per job', () => {
  const tiles = treemapTiles(GROUPS, INDEX, 'unit:2512');
  assert.deepEqual(tiles.map((tile) => tile.id),
    ['software-developer', 'embedded-systems-software-developer']);
  assert.deepEqual(tiles.map((tile) => tile.value), [1, 1]);
  assert.equal(tiles[0].kind, 'job');
});

test('the live region says what the focused tile is and what Enter does', () => {
  const [group] = treemapTiles(GROUPS, INDEX, 'major:2');
  assert.equal(describeTile(group),
    'Software and applications developers, 2 jobs, average automation 6.3. '
    + 'Press Enter to drill down.');
  const [job] = treemapTiles(GROUPS, INDEX, 'unit:2512');
  assert.match(describeTile(job), /Press Enter to open the job page\.$/);
  assert.equal(describeTile(null), '');
});

test('a shares treemap tile says what the jobs under it are made of', () => {
  const [group] = treemapTiles(SHARE_GROUPS, SHARE_INDEX, 'all', SHARES);
  assert.match(group.detail, /^3 jobs · mostly /);
  assert.equal(describeTile(group),
    'Professionals, 3 jobs, on average AI can take over 41%. Press Enter to drill down.');
  const [job] = treemapTiles(SHARE_GROUPS, SHARE_INDEX, 'major:5', SHARES);
  assert.match(job.detail, /^AI can take over \d+% · /);
  assert.match(describeTile(job), /AI can take over \d+% · AI assists/);
  assert.match(describeTile(job), /Press Enter to open the job page\.$/);
  for (const text of [group.detail, job.detail, describeTile(group), describeTile(job)]) {
    assert.doesNotMatch(text, QUADRANT_WORDS);
  }
});

test('tile text always clears 4.5:1 against its own tile, in every mode', () => {
  const modes = [...colourModes(), ...colourModes(SHARES)].map((mode) => mode.key);
  const scores = [null, 1, 2.5, 4, 5.5, 6, 7.5, 9, 10];
  const codes = ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK', ...TYPE_ORDER, null];
  for (const quadrant of codes) {
    for (const score of scores) {
      for (const mode of modes) {
        const tile = {
          quadrant, automation: score, amplification: score, mechanical: score,
        };
        const colour = tileColour(tile, mode);
        assert.ok(
          contrastRatio(colour.rgb, colour.text === '#ffffff' ? [255, 255, 255] : [0, 0, 0]) >= 4.5,
          `${mode} ${quadrant} ${score} -> ${colour.fill} with ${colour.text}`,
        );
      }
    }
  }
});

test('the colour modes follow the scheme, and never include AI exposure', () => {
  assert.deepEqual(colourModes().map((mode) => mode.key),
    ['quadrant', 'automation', 'amplification']);
  assert.deepEqual(colourModes(SHARES).map((mode) => mode.key),
    ['type', 'substitution', 'assistance', 'mechanical']);
  assert.deepEqual(colourModes(SHARES).map((mode) => mode.label),
    ['Type', 'AI substitution', 'AI assistance', 'Machine automation']);
  assert.equal(defaultColour(), 'quadrant');
  assert.equal(defaultColour(SHARES), 'type');
  // A22: colouring by class paints one flat colour when one class owns the group.
  const flat = { q: { TRANSFORM: 151, STABLE: 79, EVOLVE: 638, SHRINK: 1 } };
  assert.equal(defaultColour(undefined, flat), 'automation');
  assert.equal(defaultColour(undefined, GROUPS['major:2']), 'quadrant');
  assert.equal(defaultColour(SHARES, SHARE_GROUPS.all), 'type');
  assert.equal(defaultColour(SHARES, { q: { AUGMENTED: 9, MIXED: 1 } }), 'substitution');
  const labels = [...colourModes(), ...colourModes(SHARES)].map((mode) => mode.label);
  assert.ok(!labels.some((label) => /exposure/i.test(label)));
  for (const mode of colourModes(SHARES)) assert.doesNotMatch(mode.label, QUADRANT_WORDS);
});

test('a high score is painted differently from a low one, in each mode', () => {
  for (const [mode, field] of [['automation', 'automation'], ['substitution', 'automation'],
    ['assistance', 'amplification'], ['mechanical', 'mechanical']]) {
    const low = tileColour({ [field]: 1 }, mode);
    const high = tileColour({ [field]: 10 }, mode);
    assert.notEqual(low.fill, high.fill, mode);
  }
  assert.notEqual(tileColour({ quadrant: 'AUGMENTED' }, 'type').fill,
    tileColour({ quadrant: 'MIXED' }, 'type').fill);
  assert.equal(textColourFor([255, 255, 255]), '#000000');
  assert.equal(textColourFor([0, 0, 0]), '#ffffff');
});

/* --- summaries ------------------------------------------------------------ */

test('every visual gets a summary with the mix and the range in it', () => {
  const rows = rowsOf('2');
  for (const kind of ['ranked', 'scatter', 'treemap', 'table']) {
    const summary = visualSummary(kind, GROUPS['major:2'], rows);
    assert.match(summary, /Professionals/);
    assert.match(summary, /50% Transform/);
    assert.match(summary, /cut-off on both axes is 6/);
  }
  assert.match(visualSummary('scatter', GROUPS['major:2'], rows), /automation across/);
});

test('a group with no scores says so instead of printing NaN', () => {
  assert.equal(rangeSentence([]), 'No job in this group has scores.');
  assert.equal(rangeSentence([], SHARES), 'No job in this group has scores.');
});

test('a shares summary names three scores, seven types and no cut-off', () => {
  const rows = sharesIn('2');
  const range = rangeSentence(rows, SHARES);
  assert.match(range, /AI substitution runs 6.0 to 8.1/);
  assert.match(range, /AI assistance runs 6.2 to 7.9/);
  assert.match(range, /machine automation runs 1.5 to 2.0/);
  for (const kind of ['ranked', 'scatter', 'treemap', 'table']) {
    const summary = visualSummary(kind, SHARE_GROUPS['major:2'], rows, SHARES);
    assert.match(summary, /Professionals/);
    assert.match(summary, /33% Augmented/);
    assert.doesNotMatch(summary, QUADRANT_WORDS);
    assert.doesNotMatch(summary, /\bautomation risk\b|amplification/i);
  }
  assert.match(visualSummary('scatter', SHARE_GROUPS['major:2'], rows, SHARES),
    /the share AI can take over across and the share AI assists with up/);
});

/* --- lists ---------------------------------------------------------------- */

test('most and least exposed resolve their titles through the search index', () => {
  const lists = exposureLists({ group: GROUPS.all, bySlug: indexBySlug(INDEX) });
  assert.deepEqual(lists.top.map((job) => job.title), ['bookkeeper', 'software developer']);
  assert.equal(lists.top[0].automation, 8.6);
  assert.deepEqual(lists.bottom.map((job) => job.slug),
    ['sommeliere', 'nursery-school-head-teacher']);
  assert.deepEqual(exposureLists({ bySlug: indexBySlug(INDEX) }), { top: [], bottom: [] });
  assert.deepEqual(exposureHeadings(),
    { top: 'Most exposed to automation', bottom: 'Least exposed' });
});

test('under shares the two lists rank by the share AI can take over', () => {
  const lists = exposureLists({ scheme: SHARES, rows: SHARE_INDEX }, 3);
  assert.deepEqual(lists.top.map((job) => job.slug),
    ['subtitler', 'data-analyst', 'translator']);
  assert.deepEqual(lists.bottom.map((job) => job.slug),
    ['rescue-diver', 'child-care-worker', 'welder']);
  assert.deepEqual(lists.top[0].shares, [0.62, 0.24, 0.02, 0.12]);
  assert.deepEqual(exposureLists({ scheme: SHARES, rows: [] }), { top: [], bottom: [] });
  const headings = exposureHeadings(SHARES);
  assert.deepEqual(headings, {
    top: 'Most of the work AI can take over', bottom: 'Least of the work AI can take over',
  });
  assert.doesNotMatch(`${headings.top} ${headings.bottom}`, QUADRANT_WORDS);
});

test('driving skills wait for their titles rather than inventing them', () => {
  const pending = drivingSkills(GROUPS.all, null);
  assert.deepEqual(pending.columns.map((column) => column.heading),
    ['Most automatable', 'Most amplified']);
  assert.equal(pending.pending, false);
  assert.equal(pending.columns[0].entries[0].id, 'aaaa1111');
  assert.equal(pending.columns[0].entries[0].title, null);
  assert.equal(pending.columns[0].entries[0].count, 'in 12 jobs');
  const named = drivingSkills(GROUPS.all, new Map([['aaaa1111', 'record test data']]));
  assert.equal(named.columns[0].entries[0].title, 'record test data');
  assert.deepEqual(drivingSkills(GROUPS['unit:2512'], null).columns.map((c) => c.entries),
    [[], []]);
});

test('under shares the skills split by their own class, not by a score', () => {
  const skills = drivingSkills(SHARE_GROUPS.all, SKILLS, SHARES);
  assert.deepEqual(skills.columns.map((column) => column.heading), [
    'Skills AI can take over most often here', 'Skills AI assists with most often here',
  ]);
  assert.equal(skills.pending, false);
  assert.deepEqual(skills.columns[0].entries.map((entry) => entry.title), ['record test data']);
  assert.deepEqual(skills.columns[1].entries.map((entry) => entry.title),
    ['liaise with colleagues']);
  assert.equal(skills.columns[0].entries[0].count, 'in 5 jobs');
  // Before skill_index arrives the classes are unknown, so nothing is guessed.
  const waiting = drivingSkills(SHARE_GROUPS.all, null, SHARES);
  assert.equal(waiting.pending, true);
  assert.deepEqual(waiting.columns.map((column) => column.entries), [[], []]);
  assert.equal(drivingSkills(SHARE_GROUPS['major:5'], null, SHARES).pending, false);
  assert.deepEqual(skillIdsOf(SHARE_GROUPS.all),
    ['aaaa1111', 'bbbb2222', 'cccc3333', 'dddd4444']);
  assert.deepEqual(skillIdsOf(SHARE_GROUPS['major:5']), []);
  // A class with nothing in this group says so, rather than "none listed".
  const oneClass = drivingSkills(
    { skills: { auto: [{ id: 'cccc3333', n: 2 }], amp: [] } }, SKILLS, SHARES,
  );
  assert.deepEqual(oneClass.columns.map((column) => column.entries), [[], []]);
  assert.match(oneClass.columns[0].empty, /None of this group's most common skills/);
  assert.match(skillsNote(SHARES), /^Taken from the skills these jobs list most often/);
  assert.equal(skillsNote(), '');
  for (const column of [...drivingSkills(GROUPS.all, null).columns, ...oneClass.columns]) {
    assert.doesNotMatch(column.empty, QUADRANT_WORDS);
  }
});

/* --- comparison ----------------------------------------------------------- */

test('two groups compare on the same four quadrants', () => {
  const comparison = compareGroups(GROUPS, 'major:2', 'major:1');
  assert.equal(comparison.a.label, 'Professionals');
  assert.equal(comparison.b.label, 'Managers');
  assert.deepEqual(comparison.a.medians, { automation: 5.7, amplification: 7.4 });
  assert.equal(medianSentence(comparison.a),
    'Median automation 5.7 · median amplification 7.4');
  assert.deepEqual(comparison.shareDifferences, []);
  assert.deepEqual(comparison.differences.map((row) => row.code),
    ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
  assert.equal(comparison.differences[0].delta, 0.5); // 50% transform vs 0%
  assert.equal(comparison.differences[2].delta, 0.5 - 1);
  assert.match(comparison.differences[0].text, /50% in Professionals, 0% in Managers/);
  assert.equal(comparison.differences[0].deltaText, '50 percentage points more in Professionals');
  assert.equal(comparison.differences[2].deltaText, '50 percentage points fewer in Professionals');
  assert.equal(comparison.differences[1].deltaText, 'the same share');
  const single = compareGroups(
    { x: { label: 'X', q: { TRANSFORM: 1, STABLE: 99 } }, y: { label: 'Y', q: { STABLE: 100 } } },
    'x', 'y',
  );
  assert.equal(single.differences[0].deltaText, '1 percentage point more in X');
});

test('under shares two groups differ by their mean shares and their type mix', () => {
  const comparison = compareGroups(SHARE_GROUPS, 'major:2', 'major:5', SHARES);
  assert.deepEqual(comparison.a.shares, [0.41, 0.35, 0.02, 0.22]);
  assert.deepEqual(comparison.shareDifferences.map((row) => row.code), ['S', 'A', 'M', 'I']);
  assert.deepEqual(comparison.shareDifferences.map((row) => row.label),
    ['AI can take over', 'AI assists', 'Machines can do', 'Stays human']);
  assert.ok(Math.abs(comparison.shareDifferences[0].delta - (0.41 - 0.07)) < 1e-9);
  assert.equal(comparison.shareDifferences[0].deltaText,
    '34 percentage points more in Professionals');
  assert.match(comparison.shareDifferences[3].deltaText, /fewer in Professionals$/);
  assert.equal(comparison.differences.length, 7);
  assert.deepEqual(comparison.differences.map((row) => row.code), TYPE_ORDER);
  assert.equal(medianSentence(comparison.a, SHARES),
    'Median AI substitution 6.8 · median AI assistance 7.2 · median machine automation 1.8');
  for (const row of [...comparison.shareDifferences, ...comparison.differences]) {
    assert.doesNotMatch(`${row.label} ${row.text} ${row.deltaText}`, QUADRANT_WORDS);
  }
});

test('comparing with a group we do not have gives nothing', () => {
  assert.equal(compareGroups(GROUPS, 'major:2', 'major:9'), null);
  assert.equal(compareGroups(GROUPS, 'nope', 'major:1'), null);
});

test('the shares legend names all four classes, never a bare letter', () => {
  assert.deepEqual(shareClassLegend(), [
    { code: 'S', label: 'AI can take over' },
    { code: 'A', label: 'AI assists' },
    { code: 'M', label: 'Machines can do' },
    { code: 'I', label: 'Stays human' },
  ]);
});
