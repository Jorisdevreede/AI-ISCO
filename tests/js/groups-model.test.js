import test from 'node:test';
import assert from 'node:assert/strict';

import {
  COLOUR_MODES,
  NARROW_WIDTH,
  breadcrumb,
  childGroups,
  compareGroups,
  compareHref,
  contrastRatio,
  defaultView,
  describeTile,
  drivingSkills,
  exposureLists,
  groupSubtitle,
  indexBySlug,
  matchGroups,
  mixSentence,
  nearLineCaveat,
  plotPoint,
  quadrantBars,
  rangeSentence,
  rankedRows,
  readState,
  resolveGroup,
  scatterPoints,
  sortByColumn,
  textColourFor,
  thresholdPoint,
  tileColour,
  treemapTiles,
  unknownGroupMessage,
  visualSummary,
} from '../../site/js/pages/groups-model.js';
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
  occupations: 3043,
  near_line: { count: 1509, share: 0.4959 },
};

const rowsOf = (code) => INDEX.filter((row) => row.c.startsWith(code));

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
  const bars = quadrantBars(GROUPS['major:2']);
  assert.deepEqual(bars.map((bar) => bar.code), ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
  assert.deepEqual(bars.map((bar) => bar.label), ['Transform', 'Stable', 'Evolve', 'Shrink']);
  assert.deepEqual(bars.map((bar) => bar.count), [2, 0, 2, 0]);
  assert.deepEqual(bars.map((bar) => bar.percent), ['50%', '0%', '50%', '0%']);
  assert.equal(bars[0].text, '2 of 4 jobs');
});

test('an empty group has zero shares rather than NaN', () => {
  const bars = quadrantBars({ q: {} });
  assert.deepEqual(bars.map((bar) => bar.percent), ['0%', '0%', '0%', '0%']);
  assert.equal(mixSentence({ q: {} }), '0% Transform, 0% Stable, 0% Evolve, 0% Shrink.');
});

test('the caveat quotes the near-the-line share and claims nothing else', () => {
  const caveat = nearLineCaveat(STATS);
  assert.match(caveat, /1,509 of 3,043 jobs \(50%\)/);
  assert.match(caveat, /band, not a count/);
  assert.doesNotMatch(caveat, /second|run|agree|4 in 10/i);
  assert.match(nearLineCaveat(null), /band, not a count/);
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

test('the table sorts by any column, both ways, ties by title', () => {
  const rows = rowsOf('');
  assert.equal(sortByColumn(rows, 'a', true)[0].s, 'bookkeeper');
  assert.equal(sortByColumn(rows, 'a', false)[0].s, 'sommeliere');
  assert.equal(sortByColumn(rows, 'c', false)[0].s, 'nursery-school-head-teacher');
  assert.equal(sortByColumn(rows, 'q', false)[0].q, 'EVOLVE');
  assert.equal(sortByColumn(rows, 't', false)[0].s, 'bookkeeper');
  assert.equal(sortByColumn(rows, 'a', true).length, rows.length);
});

test('sorting the table never mutates the caller\'s array', () => {
  const before = INDEX.map((row) => row.s);
  sortByColumn(INDEX, 'a', true);
  assert.deepEqual(INDEX.map((row) => row.s), before);
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

test('tile text always clears 4.5:1 against its own tile', () => {
  const modes = COLOUR_MODES.map((mode) => mode.key);
  const scores = [null, 1, 2.5, 4, 5.5, 6, 7.5, 9, 10];
  for (const quadrant of ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK', null]) {
    for (const score of scores) {
      for (const mode of modes) {
        const tile = { quadrant, automation: score, amplification: score };
        const colour = tileColour(tile, mode);
        assert.ok(
          contrastRatio(colour.rgb, colour.text === '#ffffff' ? [255, 255, 255] : [0, 0, 0]) >= 4.5,
          `${mode} ${quadrant} ${score} -> ${colour.fill} with ${colour.text}`,
        );
      }
    }
  }
});

test('the colour modes never include AI exposure', () => {
  assert.deepEqual(COLOUR_MODES.map((mode) => mode.key),
    ['quadrant', 'automation', 'amplification']);
});

test('a high score is painted differently from a low one', () => {
  const low = tileColour({ automation: 1 }, 'automation');
  const high = tileColour({ automation: 10 }, 'automation');
  assert.notEqual(low.fill, high.fill);
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
});

/* --- lists ---------------------------------------------------------------- */

test('most and least exposed resolve their titles through the search index', () => {
  const lists = exposureLists(GROUPS.all, indexBySlug(INDEX));
  assert.deepEqual(lists.top.map((job) => job.title), ['bookkeeper', 'software developer']);
  assert.equal(lists.top[0].automation, 8.6);
  assert.deepEqual(lists.bottom.map((job) => job.slug),
    ['sommeliere', 'nursery-school-head-teacher']);
  assert.deepEqual(exposureLists(null, indexBySlug(INDEX)), { top: [], bottom: [] });
});

test('driving skills wait for their titles rather than inventing them', () => {
  const pending = drivingSkills(GROUPS.all, null);
  assert.equal(pending.automation[0].id, 'aaaa1111');
  assert.equal(pending.automation[0].title, null);
  assert.equal(pending.automation[0].count, 'in 12 jobs');
  const named = drivingSkills(GROUPS.all, new Map([['aaaa1111', 'record test data']]));
  assert.equal(named.automation[0].title, 'record test data');
  assert.deepEqual(drivingSkills(GROUPS['unit:2512'], null), { automation: [], amplification: [] });
});

/* --- comparison ----------------------------------------------------------- */

test('two groups compare on the same four quadrants', () => {
  const comparison = compareGroups(GROUPS, 'major:2', 'major:1');
  assert.equal(comparison.a.label, 'Professionals');
  assert.equal(comparison.b.label, 'Managers');
  assert.deepEqual(comparison.a.medians, { automation: 5.7, amplification: 7.4 });
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

test('comparing with a group we do not have gives nothing', () => {
  assert.equal(compareGroups(GROUPS, 'major:2', 'major:9'), null);
  assert.equal(compareGroups(GROUPS, 'nope', 'major:1'), null);
});
