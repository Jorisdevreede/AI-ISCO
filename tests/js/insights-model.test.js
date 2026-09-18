import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ARTICLE, articleValues, blockText, buildArticle, deriveInsights, fillTemplate,
} from '../../site/js/pages/insights-model.js';

// A synthetic stats.json: 20 occupations, shares that do not round to the counts
// by accident, so a hard-coded percentage in the prose would show up at once.
const STATS = {
  built: '2026-01-02',
  threshold: 6,
  occupations: 20,
  skills_scored: 120,
  quadrants: {
    counts: { EVOLVE: 9, STABLE: 5, TRANSFORM: 4, SHRINK: 2 },
    shares: { EVOLVE: 0.45, STABLE: 0.25, TRANSFORM: 0.2, SHRINK: 0.1 },
  },
  near_line: { count: 7, share: 0.35 },
};

// A synthetic groups.json, only the levels and fields the model reads.
const GROUPS = {
  all: { label: 'All occupations', level: 'all', n: 20, q: {} },
  'major:1': {
    label: 'Managers', level: 'major', n: 8,
    q: { EVOLVE: 6, TRANSFORM: 2 },
    auto: { mean: 5.0 }, amp: { mean: 7.2 },
  },
  'major:2': {
    label: 'Professionals', level: 'major', n: 7,
    q: { EVOLVE: 3, TRANSFORM: 2, STABLE: 1, SHRINK: 1 },
    auto: { mean: 4.4 }, amp: { mean: 7.5 },
  },
  'major:9': {
    label: 'Elementary occupations', level: 'major', n: 5,
    q: { STABLE: 4, SHRINK: 1 },
    auto: { mean: 6.1 }, amp: { mean: 4.2 },
  },
  'sub:21': { label: 'Science professionals', level: 'sub', n: 3, q: { EVOLVE: 3 } },
};

// A synthetic search_index.json.
const INDEX = [
  { t: 'typist', s: 'typist', c: '4131', mg: 'Clerical', a: 8.5, m: 8.0, q: 'TRANSFORM' },
  { t: 'data clerk', s: 'data-clerk', c: '4132', mg: 'Clerical', a: 8.1, m: 8.9, q: 'TRANSFORM' },
  { t: 'soap chipper', s: 'soap-chipper', c: '8160', mg: 'Plant', a: 7.9, m: 3.8, q: 'SHRINK' },
  { t: 'wax bleacher', s: 'wax-bleacher', c: '8160', mg: 'Plant', a: 6.7, m: 3.4, q: 'SHRINK' },
  { t: 'geneticist', s: 'geneticist', c: '2131', mg: 'Prof', a: 4.4, m: 8.4, q: 'EVOLVE' },
  { t: 'nurse', s: 'nurse', c: '2221', mg: 'Prof', a: 4.5, m: 6.0, q: 'EVOLVE' },
  { t: 'puppeteer', s: 'puppeteer', c: '2655', mg: 'Prof', a: 2.1, m: 4.7, q: 'STABLE' },
  { t: 'stagehand', s: 'stagehand', c: '2659', mg: 'Prof', a: 2.6, m: 3.7, q: 'STABLE' },
];

const facts = deriveInsights({ stats: STATS, groups: GROUPS, index: INDEX });
const article = buildArticle(facts);
const prose = article.map(blockText).join(' ');

test('the headline totals come straight from stats.json', () => {
  assert.equal(facts.occupations, 20);
  assert.equal(facts.skills, 120);
  assert.equal(facts.threshold, 6);
  assert.deepEqual(facts.nearLine, { count: 7, share: 0.35 });
});

test('quadrants come back biggest first, with counts and shares paired', () => {
  assert.deepEqual(facts.quadrants.map((row) => row.code),
    ['EVOLVE', 'STABLE', 'TRANSFORM', 'SHRINK']);
  assert.deepEqual(
    facts.quadrants.map((row) => [row.countText, row.shareText]),
    [['9', '45%'], ['5', '25%'], ['4', '20%'], ['2', '10%']],
  );
});

test('only major groups are tabled, in ISCO order', () => {
  assert.deepEqual(facts.majors.map((row) => row.key), ['major:1', 'major:2', 'major:9']);
  assert.equal(facts.majors[0].href, 'groups.html#g=major:1');
  assert.equal(facts.majors[2].shrinkShare, 0.2);
});

test('group highlights are read off the data, not asserted', () => {
  const { topAmplification, topAutomation, topShrinkShare, largestZeroShrink } = facts.highlights;
  assert.equal(topAmplification.label, 'Professionals'); // 7.5 beats 7.2
  assert.equal(topAutomation.label, 'Elementary occupations'); // 6.1
  assert.equal(topShrinkShare.label, 'Elementary occupations'); // 1 of 5
  assert.equal(largestZeroShrink.label, 'Managers');
});

test('the ranked lists sort by the thing each section is about', () => {
  assert.deepEqual(facts.lists.mostExposed.slice(0, 2).map((row) => row.s),
    ['typist', 'data-clerk']);
  assert.equal(facts.lists.leastExposed[0].s, 'puppeteer');
  assert.equal(facts.lists.shrink[0].s, 'soap-chipper'); // widest automation-amplification gap
  assert.equal(facts.lists.transform[0].s, 'data-clerk'); // highest on both axes
  assert.equal(facts.lists.stable[0].s, 'stagehand'); // lowest on both axes together
  assert.equal(facts.lists.evolve[0].s, 'geneticist');
});

test('evolve-per-shrink is derived, never written down', () => {
  assert.equal(facts.evolvePerShrink, 5); // 9 / 2, rounded
  assert.equal(deriveInsights({ stats: { quadrants: { counts: {} } } }).evolvePerShrink, null);
});

test('a template splits into text and link segments', () => {
  const segments = fillTemplate('a {one} b {two}!', {
    one: 'X', two: { text: 'Y', href: 'job.html#y' },
  });
  assert.deepEqual(segments, [
    { text: 'a ' }, { text: 'X' }, { text: ' b ' },
    { text: 'Y', href: 'job.html#y' }, { text: '!' },
  ]);
});

test('an unknown placeholder survives as itself rather than as "undefined"', () => {
  assert.equal(blockText({ segments: fillTemplate('a {nope} b') }), 'a {nope} b');
});

test('every placeholder in the article has a value', () => {
  const values = articleValues(facts);
  for (const block of ARTICLE) {
    for (const [, name] of block.text.matchAll(/\{(\w+)\}/g)) {
      assert.ok(values[name] !== undefined, `no value for {${name}}`);
    }
  }
  assert.equal(prose.includes('{'), false);
});

test('the prose states the fixture numbers, not the real ones', () => {
  assert.match(prose, /120 ESCO skills behind 20 European occupations/);
  assert.match(prose, /10% of occupations fell into Shrink — 2 jobs/);
  assert.match(prose, /largest share, 45%, landed in\s+Evolve/);
  assert.match(prose, /9 of 20 occupations sit in Evolve/);
  assert.match(prose, /For every occupation in Shrink, 5 sit in Evolve/);
  assert.match(prose, /hard cut at 6 on both axes, and 7 occupations — 35% of them/);
  assert.equal(prose.includes('51%'), false); // the old hard-coded share
  assert.equal(prose.includes('13,939'), false); // the old hard-coded skill count
});

test('the prose names the occupations and groups the data picked', () => {
  assert.match(prose, /soap chipper, at 7\.9 on automation risk against 3\.8/);
  assert.match(prose, /data clerk scores 8\.1 .* and 8\.9 on amplification/);
  assert.match(prose, /typist carries the highest automation risk in the dataset, 8\.5/);
  assert.match(prose, /puppeteer sits at the bottom of the exposure ranking, 2\.1/);
  assert.match(prose, /Not one of the 8 occupations in Managers falls into Shrink/);
  assert.match(prose, /Professionals show the highest average amplification, 7\.5/);
  assert.match(prose, /Elementary occupations carry the highest average automation risk at 6\.1/);
});

test('occupation names in the prose link to their job page, groups to theirs', () => {
  const links = article.flatMap((block) => block.segments).filter((segment) => segment.href);
  const hrefs = links.map((segment) => segment.href);
  assert.ok(hrefs.includes('job.html#soap-chipper'));
  assert.ok(hrefs.includes('job.html#puppeteer'));
  assert.ok(hrefs.includes('groups.html#g=major:1'));
  assert.ok(hrefs.includes('method.html'));
  assert.equal(links.every((segment) => segment.text.length > 0), true);
});

test('when every major group has a Shrink occupation the clause changes', () => {
  const groups = {
    'major:1': { ...GROUPS['major:1'], q: { EVOLVE: 5, SHRINK: 3 } },
    'major:9': GROUPS['major:9'],
  };
  const text = buildArticle(deriveInsights({ stats: STATS, groups, index: INDEX }))
    .map(blockText).join(' ');
  assert.match(text, /smallest share of Shrink sits in Elementary occupations, at 20%/);
  assert.equal(text.includes('Not one of the'), false);
});

test('no figure from the private second scoring run appears anywhere', () => {
  const text = `${prose} ${JSON.stringify(ARTICLE)}`;
  for (const forbidden of ['57.8', '81.8', '671', '0.93', '0.94', 'second scoring run',
    'agreement', 'changed box', 'correlat']) {
    assert.equal(text.includes(forbidden), false, `article mentions ${forbidden}`);
  }
});
