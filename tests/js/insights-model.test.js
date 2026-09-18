import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ARTICLE, SHARES_ARTICLE, SHARES_NOTES, articleValues, blockText, buildArticle,
  deriveInsights, fillTemplate,
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
  const text = `${prose} ${JSON.stringify(ARTICLE)} ${JSON.stringify(SHARES_ARTICLE)}`
    + ` ${JSON.stringify(SHARES_NOTES)} ${sharesProse}`;
  for (const forbidden of ['57.8', '81.8', '671', '0.93', '0.94', 'second scoring run',
    'agreement', 'changed box', 'correlat']) {
    assert.equal(text.includes(forbidden), false, `article mentions ${forbidden}`);
  }
});

/* --- the shares scheme ---------------------------------------------------- */

// A synthetic stats_v2.json: seven types and four skill classes, with counts
// that no percentage in the prose could match by accident.
const SHARES_STATS = {
  built: '2026-02-03',
  scheme: 'shares',
  model: 'jev-test',
  occupations: 40,
  skills_scored: 200,
  types: {
    order: ['AUTOMATION_HEAVY', 'TRANSFORMING', 'AUGMENTED', 'MECHANISABLE',
      'INSULATED_PHYSICAL', 'INSULATED_PEOPLE', 'MIXED'],
    counts: {
      AUTOMATION_HEAVY: 2,
      TRANSFORMING: 3,
      AUGMENTED: 12,
      MECHANISABLE: 4,
      INSULATED_PHYSICAL: 14,
      INSULATED_PEOPLE: 4,
      MIXED: 1,
    },
    shares: {
      AUTOMATION_HEAVY: 0.05,
      TRANSFORMING: 0.075,
      AUGMENTED: 0.3,
      MECHANISABLE: 0.1,
      INSULATED_PHYSICAL: 0.35,
      INSULATED_PEOPLE: 0.1,
      MIXED: 0.025,
    },
  },
  skill_classes: {
    counts: { S: 25, A: 38, M: 19, I: 118 },
    shares: { S: 0.125, A: 0.19, M: 0.095, I: 0.59 },
  },
  near_line: { count: 11, share: 0.275 },
};

const SHARES_GROUPS = {
  all: { label: 'All occupations', level: 'all', n: 40, q: {} },
  'major:3': {
    label: 'Technicians', level: 'major', n: 15,
    q: { AUGMENTED: 9, TRANSFORMING: 3, MIXED: 3 },
    auto: { mean: 5.4 }, amp: { mean: 6.2 }, sh: [0.21, 0.46, 0.05, 0.28],
  },
  'major:7': {
    label: 'Craft workers', level: 'major', n: 15,
    q: { INSULATED_PHYSICAL: 11, MECHANISABLE: 4 },
    auto: { mean: 3.9 }, amp: { mean: 4.1 }, sh: [0.08, 0.14, 0.31, 0.47],
  },
  'major:9': {
    label: 'Elementary occupations', level: 'major', n: 10,
    q: { INSULATED_PHYSICAL: 6, INSULATED_PEOPLE: 4 },
    auto: { mean: 3.2 }, amp: { mean: 3.6 }, sh: [0.06, 0.1, 0.18, 0.66],
  },
};

const share = (s, t, q, sh) => ({ t, s, c: '0000', mg: 'Test', a: 5, m: 5, k: 3, q, sh, nl: false });

const SHARES_INDEX = [
  share('data-typist', 'data typist', 'AUTOMATION_HEAVY', [0.72, 0.14, 0.0, 0.14]),
  share('ledger-clerk', 'ledger clerk', 'AUTOMATION_HEAVY', [0.61, 0.2, 0.04, 0.15]),
  share('tax-adviser', 'tax adviser', 'TRANSFORMING', [0.38, 0.29, 0.02, 0.31]),
  share('ward-nurse', 'ward nurse', 'AUGMENTED', [0.12, 0.48, 0.02, 0.38]),
  share('press-operator', 'press operator', 'MECHANISABLE', [0.05, 0.1, 0.55, 0.3]),
  share('stone-mason', 'stone mason', 'INSULATED_PHYSICAL', [0.02, 0.06, 0.09, 0.83]),
  share('roof-tiler', 'roof tiler', 'INSULATED_PHYSICAL', [0.04, 0.08, 0.12, 0.76]),
  share('youth-worker', 'youth worker', 'INSULATED_PEOPLE', [0.03, 0.15, 0.01, 0.81]),
  share('site-foreman', 'site foreman', 'MIXED', [0.22, 0.24, 0.26, 0.28]),
];

const sharesFacts = deriveInsights({
  stats: SHARES_STATS, groups: SHARES_GROUPS, index: SHARES_INDEX,
});
const sharesArticle = buildArticle(sharesFacts);
const sharesProse = sharesArticle.map(blockText).join(' ');

test('a shares set derives its own totals, classes and largest class', () => {
  assert.equal(sharesFacts.scheme, 'shares');
  assert.equal(sharesFacts.occupations, 40);
  assert.equal(sharesFacts.threshold, null);
  assert.deepEqual(sharesFacts.classes.map((row) => row.code), ['S', 'A', 'M', 'I']);
  assert.deepEqual(sharesFacts.classes.map((row) => row.shareText),
    ['13%', '19%', '9%', '59%']);
  assert.equal(sharesFacts.topClass.code, 'I');
});

test('the seven types come back largest first', () => {
  assert.deepEqual(sharesFacts.quadrants.slice(0, 3).map((row) => row.code),
    ['INSULATED_PHYSICAL', 'AUGMENTED', 'MECHANISABLE']);
  assert.equal(sharesFacts.quadrants[0].name, 'Insulated by physical work');
  assert.equal(sharesFacts.quadrants.length, 7);
});

test('each ranked list is ranked by its own share', () => {
  assert.deepEqual(sharesFacts.lists.substituted.slice(0, 2).map((row) => row.s),
    ['data-typist', 'ledger-clerk']);
  assert.equal(sharesFacts.lists.assisted[0].s, 'ward-nurse');
  assert.equal(sharesFacts.lists.mechanised[0].s, 'press-operator');
  assert.equal(sharesFacts.lists.insulated[0].s, 'stone-mason');
  assert.equal(sharesFacts.lists.substituted.every((row) => Array.isArray(row.sh)), true);
});

test('the examples are the clearest case of each type, not the list heads', () => {
  assert.equal(sharesFacts.examples.substituted.s, 'data-typist');
  assert.equal(sharesFacts.examples.physical.s, 'stone-mason');
  assert.equal(sharesFacts.examples.people.s, 'youth-worker');
});

test('a major group carries its mean shares and its largest type', () => {
  const craft = sharesFacts.majors.find((row) => row.key === 'major:7');
  assert.deepEqual(craft.sh, [0.08, 0.14, 0.31, 0.47]);
  assert.equal(craft.topType, 'INSULATED_PHYSICAL');
  assert.equal(sharesFacts.groupShares.assisted.label, 'Technicians');
  assert.equal(sharesFacts.groupShares.mechanised.label, 'Craft workers');
  assert.equal(sharesFacts.groupShares.insulated.label, 'Elementary occupations');
});

test('every placeholder in the shares article and its notes has a value', () => {
  const values = articleValues(sharesFacts);
  const templates = [...SHARES_ARTICLE.map((block) => block.text),
    ...Object.values(SHARES_NOTES)];
  for (const template of templates) {
    for (const [, name] of template.matchAll(/\{(\w+)\}/g)) {
      assert.ok(values[name] !== undefined, `no value for {${name}}`);
    }
  }
  assert.equal(sharesProse.includes('{'), false);
});

test('no shares template carries a figure of its own: every digit is a slot', () => {
  const templates = [...SHARES_ARTICLE.map((block) => block.text),
    ...Object.values(SHARES_NOTES)];
  for (const template of templates) {
    const outsideSlots = template.replace(/\{\w+\}/g, '');
    assert.doesNotMatch(outsideSlots, /\d/, `typed-in figure in: ${template}`);
  }
});

test('the shares prose states the fixture numbers and names its own picks', () => {
  assert.match(sharesProse, /200 ESCO skills behind these 40 occupations/);
  assert.match(sharesProse, /largest single type is Insulated by physical work/);
  assert.match(sharesProse, /14 occupations, 35% of the total, 5 points ahead of Augmented at 30%/);
  assert.match(sharesProse, /59% of all scored skills are the kind this rubric calls “Stays human”/);
  assert.match(sharesProse, /data typist is the clearest case/);
  assert.match(sharesProse, /2 occupations, 5% of them, meet the first rule/);
  assert.match(sharesProse, /stone mason keeps 83% of its skill weight/);
  assert.match(sharesProse, /youth worker keeps 81%/);
  assert.match(sharesProse, /Craft workers carry the largest mechanised share .*, 31%/);
  assert.match(sharesProse, /Technicians carry the largest assisted share at 46%/);
  assert.match(sharesProse, /1 occupation, 2% of them, come out Mixed/);
  assert.match(sharesProse, /11 occupations, 28% of them, sit near enough/);
});

test('the shares prose never uses the words of the other scheme', () => {
  assert.doesNotMatch(sharesProse, /quadrant|\bbox\b|cut at 6|at risk/i);
  assert.equal(sharesProse.includes('undefined'), false);
  assert.equal(sharesProse.includes('NaN'), false);
});

test('the shares article says what it must about machinery and about proof', () => {
  assert.match(sharesProse, /Machinery is asked about separately from AI, on purpose/);
  assert.match(sharesProse, /Read the mechanised share as a floor/);
  assert.match(sharesProse, /residual class and not a finding that AI will leave those jobs alone/);
  assert.match(sharesProse, /not a job AI cannot help with/);
  assert.match(sharesProse, /None of this was measured/);
  assert.match(sharesProse, /no ground truth/);
});

test('the shares article links every job and group it names', () => {
  const hrefs = sharesArticle.flatMap((block) => block.segments)
    .filter((segment) => segment.href).map((segment) => segment.href);
  assert.ok(hrefs.includes('job.html#data-typist'));
  assert.ok(hrefs.includes('job.html#stone-mason'));
  assert.ok(hrefs.includes('groups.html#g=major:7'));
  assert.ok(hrefs.includes('method.html'));
});

test('a quadrant set still gets the quadrant article, unchanged', () => {
  assert.equal(facts.scheme, 'quadrants');
  assert.equal(buildArticle(facts).length, ARTICLE.length);
  assert.match(prose, /The great rebalancing|great rebalancing|Shrink/);
});

/* --- C7: every printed split adds up to 100 ------------------------------- */

function sumOfPercents(rows) {
  return rows.reduce((total, row) => total + Number(row.shareText.replace('%', '')), 0);
}

test('the type table and the class table each sum to 100 (C7)', () => {
  assert.equal(sumOfPercents(sharesFacts.quadrants), 100);
  assert.equal(sumOfPercents(sharesFacts.classes), 100);
  // Naive rounding of this fixture gives 101: 5 + 8 + 30 + 10 + 35 + 10 + 3.
  const naive = sharesFacts.quadrants
    .reduce((total, row) => total + Math.round(row.share * 100), 0);
  assert.notEqual(naive, 100);
});

test('a quadrant set is rounded the same way (C7)', () => {
  assert.equal(sumOfPercents(facts.quadrants), 100);
});

test('a class with something in it never reads "0%" (C7)', () => {
  const tiny = deriveInsights({
    stats: {
      ...SHARES_STATS,
      occupations: 4000,
      types: {
        order: SHARES_STATS.types.order,
        counts: {
          AUTOMATION_HEAVY: 1,
          TRANSFORMING: 1,
          AUGMENTED: 1999,
          MECHANISABLE: 0,
          INSULATED_PHYSICAL: 1999,
          INSULATED_PEOPLE: 0,
          MIXED: 0,
        },
        shares: {},
      },
    },
    groups: SHARES_GROUPS,
    index: SHARES_INDEX,
  });
  const byCode = Object.fromEntries(tiny.quadrants.map((row) => [row.code, row]));
  assert.equal(byCode.AUTOMATION_HEAVY.shareText, '<1%');
  assert.equal(byCode.MECHANISABLE.shareText, '0%');
  assert.equal(sumOfPercents(tiny.quadrants.filter((row) => row.count > 1)), 100);
});

test('no count slot in the article can read "1 occupations" (A13)', () => {
  const one = { ...SHARES_STATS, near_line: { count: 1, share: 0.025 } };
  const prose = buildArticle(deriveInsights({
    stats: one, groups: SHARES_GROUPS, index: SHARES_INDEX,
  })).map(blockText).join(' ');
  assert.doesNotMatch(prose, /\b1 (?:occupations|jobs|skills)\b/);
  assert.match(prose, /\b1 occupation\b/);
});
