import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MOVE_MARGIN, addRecent, backTarget, compareMove, essentialSkills, evolutionPaths,
  findOccupation, gapSkills, groupChain, occupationIndex, occupationSkills,
  resolveSkills, scatterPoints, scatterSummary, splitSkills, unitGroupKey,
} from '../../site/js/pages/job-model.js';

// Synthetic fixture, hand-written. It copies the SHAPE of portfolio_data.json
// and the published scores of two occupations so the numbers read plausibly;
// every skill, rationale and adjacency below is made up for this test.
const DATA = {
  skills: {
    aaa11111: { t: 'write computer code', a: 8.2, m: 9.0, r: 'A made-up rationale.' },
    bbb22222: { t: 'design software architecture', a: 4.1, m: 8.8, r: null },
    ccc33333: { t: 'carry equipment', a: 2.0, m: 3.0, r: 'Another made-up rationale.' },
    ddd44444: { t: 'skill with no amplification score', a: 7.0, m: null },
    eee55555: { t: 'skill exactly on the line', a: 6.0, m: 5.9 },
    fff66666: { t: 'gap skill, strongly amplified', a: 3.0, m: 9.0 },
    ggg77777: { t: 'gap skill, mildly amplified', a: 3.0, m: 6.5 },
    hhh88888: { t: 'gap skill with no score', a: 3.0, m: null },
  },
  occupations: [
    {
      t: 'software developer', s: 'software-developer', c: '2512',
      cat: 'Information and communications technology professionals', mg: 'Professionals',
      q: 'TRANSFORM', e: 5.7, ar: 6.4, ap: 8.9,
      se: ['aaa11111', 'bbb22222', 'ddd44444'],
      so: ['ccc33333', 'eee55555', 'not-a-real-id'],
      adj: [
        { s: 'web-developer', t: 'web developer', ov: 0.32, q: 'TRANSFORM', gap: ['fff66666'] },
        {
          s: 'language-engineer', t: 'language engineer', ov: 0.25, q: 'TRANSFORM',
          gap: ['fff66666', 'ggg77777', 'hhh88888', 'aaa11111'],
        },
        {
          s: 'ict-application-configurator', t: 'ICT application configurator',
          ov: 0.37, q: 'TRANSFORM', gap: ['ggg77777'],
        },
      ],
      n: { story: 'A made-up story.', adv: 'Made-up advice.' },
    },
    { t: 'web developer', s: 'web-developer', c: '2513', q: 'TRANSFORM', ar: 6.9, ap: 8.9 },
    {
      t: 'language engineer', s: 'language-engineer', c: '2643', q: 'TRANSFORM',
      ar: 6.8, ap: 8.9,
    },
    {
      t: 'ICT application configurator', s: 'ict-application-configurator', c: '2512',
      q: 'TRANSFORM', ar: 6.6, ap: 8.9,
    },
    {
      t: 'nurse responsible for general care', s: 'nurse-responsible-for-general-care',
      c: '2221', mg: 'Professionals', q: 'EVOLVE', e: 1.9, ar: 3.0, ap: 6.4,
      se: ['ccc33333'], so: [],
      adj: [
        { s: 'biomedical-scientist', t: 'biomedical scientist', ov: 0.18, q: 'EVOLVE', gap: [] },
        { s: 'counsellor', t: 'counsellor', ov: 0.2, q: 'EVOLVE', gap: [] },
        { s: 'ward clerk', t: 'ward clerk', ov: 0.2, q: 'SHRINK', gap: [] },
      ],
    },
    { t: 'biomedical scientist', s: 'biomedical-scientist', c: '2131', q: 'EVOLVE', ar: 4.6, ap: 7.3 },
    { t: 'counsellor', s: 'counsellor', c: '2635', q: 'EVOLVE', ar: 2.1, ap: 7.4 },
    { t: 'ward clerk', s: 'ward clerk', c: '4226', q: 'SHRINK', ar: 7.8, ap: 5.0 },
    { t: 'lift operator', s: 'lift-operator', c: '9629', q: 'STABLE', ar: 3.0 },
  ],
};

const GROUPS = {
  all: { label: 'All occupations', n: 3043 },
  'major:2': { label: 'Professionals', n: 869 },
  'minor:251': { label: 'Software and applications developers and analysts', n: 55 },
  'unit:2512': { label: 'Software developers', n: 10 },
  'unit:2221': { label: 'Nursing professionals', n: 3 },
};

const developer = () => findOccupation(DATA, 'software-developer');
const nurse = () => findOccupation(DATA, 'nurse-responsible-for-general-care');

test('an occupation is found by slug, and an unknown slug is not an error', () => {
  assert.equal(developer().t, 'software developer');
  assert.equal(findOccupation(DATA, 'tea-taster-to-the-queen'), null);
  assert.equal(findOccupation(DATA, ''), null);
  assert.equal(findOccupation({}, 'software-developer'), null);
});

test('skill ids resolve to objects, and ids the data does not know are dropped', () => {
  const resolved = resolveSkills(DATA.skills, ['aaa11111', 'not-a-real-id'], true);
  assert.equal(resolved.length, 1);
  assert.deepEqual(resolved[0], {
    id: 'aaa11111',
    title: 'write computer code',
    auto: 8.2,
    amp: 9.0,
    rationale: 'A made-up rationale.',
    rationaleFrom: null,
    essential: true,
  });
  assert.deepEqual(resolveSkills(DATA.skills, null), []);
});

test('a rationale another model wrote keeps who wrote it and for which scores', () => {
  const skills = { bbb: { t: 'sort post', a: 7.4, m: 2.8, r: 'Borrowed.', rf: { s: 'gemini', a: 9, m: 3 } } };
  const [skill] = resolveSkills(skills, ['bbb']);
  assert.equal(skill.rationale, 'Borrowed.');
  assert.deepEqual(skill.rationaleFrom, { s: 'gemini', a: 9, m: 3 });
});

test('a half-scored skill keeps its scored half and nulls the other', () => {
  const [skill] = resolveSkills(DATA.skills, ['ddd44444']);
  assert.equal(skill.auto, 7.0);
  assert.equal(skill.amp, null);
  assert.equal(skill.essential, false);
});

test('the skills of an occupation are essential first, optional after', () => {
  const skills = occupationSkills(DATA, developer());
  assert.deepEqual(skills.map((skill) => skill.id),
    ['aaa11111', 'bbb22222', 'ddd44444', 'ccc33333', 'eee55555']);
  assert.deepEqual(skills.map((skill) => skill.essential),
    [true, true, true, false, false]);
});

test('the rationale cards lead with the highest-tension essential skill', () => {
  assert.deepEqual(essentialSkills(DATA, developer()).map((skill) => skill.id),
    ['aaa11111', 'bbb22222', 'ddd44444']);
});

test('the two skill lists cut at 6, not at 5.5', () => {
  const { depreciating, appreciating } = splitSkills(occupationSkills(DATA, developer()));
  assert.deepEqual(depreciating.map((skill) => skill.id),
    ['aaa11111', 'ddd44444', 'eee55555']);
  assert.deepEqual(appreciating.map((skill) => skill.id), ['aaa11111', 'bbb22222']);
  // 5.9 is below the cut-off even though the old page drew the line at 5.5.
  assert.ok(!appreciating.some((skill) => skill.id === 'eee55555'));
});

test('the scatter plots only the skills that carry both scores', () => {
  const points = scatterPoints(occupationSkills(DATA, developer()));
  assert.deepEqual(points.map((point) => point.id),
    ['aaa11111', 'bbb22222', 'ccc33333', 'eee55555']);
  assert.deepEqual(points.map((point) => point.q),
    ['TRANSFORM', 'EVOLVE', 'STABLE', 'SHRINK']);
});

test('the aria-label counts what the picture shows', () => {
  const points = scatterPoints(occupationSkills(DATA, developer()));
  const label = scatterSummary('software developer', points);
  assert.match(label, /Scatter plot of 4 skills in software developer/);
  assert.match(label, /both cut-offs are drawn at 6/);
  assert.match(label, /2 skills reach 6 on automation risk/);
  assert.match(label, /2 reach it on amplification/);
  assert.match(label, /2 are essential to the job and 2 optional/);
  assert.match(scatterSummary('lift operator', []), /the plot is empty/);
});

test('ISCO codes give the group keys an occupation sits in', () => {
  assert.equal(unitGroupKey('2512'), 'unit:2512');
  assert.equal(unitGroupKey('251'), null);
  assert.deepEqual(groupChain('2512'), ['unit:2512', 'minor:251', 'sub:25', 'major:2', 'all']);
  assert.deepEqual(groupChain(''), ['all']);
});

test('from= decides the back link, with the label and count from groups.json', () => {
  const back = backTarget('unit:2512', developer(), GROUPS);
  assert.equal(back.text, '← Back to Software developers (10 jobs)');
  assert.equal(back.href, 'groups.html#g=unit:2512');
  assert.equal(back.count, 10);
  assert.equal(backTarget('unit:2221', nurse(), GROUPS).text,
    '← Back to Nursing professionals (3 jobs)');
  assert.equal(backTarget('major:2', developer(), GROUPS).text,
    '← Back to Professionals (869 jobs)');
});

test('without from= the back link comes from the occupation itself', () => {
  const back = backTarget(null, developer(), GROUPS);
  assert.equal(back.text, '← All jobs in Software developers');
  assert.equal(back.href, 'groups.html#g=unit:2512');
});

test('a back link is never a dead end', () => {
  // from= naming a group that is not in groups.json falls back to the occupation.
  assert.equal(backTarget('unit:9999', developer(), GROUPS).key, 'unit:2512');
  // No unit group of its own: walk up to the widest group that does exist.
  const sparse = { 'major:2': GROUPS['major:2'] };
  assert.equal(backTarget(null, developer(), sparse).text,
    '← All jobs in Professionals');
  // Nothing at all still points somewhere real.
  const last = backTarget(null, { c: '9629' }, {});
  assert.equal(last.text, '← All jobs');
  assert.equal(last.href, 'groups.html#g=all');
});

test('a difference smaller than the margin is not a move', () => {
  assert.equal(MOVE_MARGIN, 0.5);
  const from = { a: 6.4, m: 8.9 };
  assert.equal(compareMove(from, { a: 6.8, m: 8.9 }).kind, 'sideways');
  assert.equal(compareMove(from, { a: 6.0, m: 8.9 }).kind, 'sideways');
  assert.match(compareMove(from, { a: 6.8, m: 8.9 }).label, /sideways move/);
});

test('a move is named for what actually improves', () => {
  const from = { a: 6.4, m: 8.9 };
  assert.deepEqual(compareMove(from, { a: 5.0, m: 9.5 }),
    { kind: 'better', label: 'Less exposed and more amplified' });
  assert.deepEqual(compareMove(from, { a: 5.0, m: 8.9 }),
    { kind: 'better', label: 'Less exposed to automation' });
  assert.deepEqual(compareMove(from, { a: 6.4, m: 9.6 }),
    { kind: 'better', label: 'More amplified by AI' });
});

test('a trade-off is labelled as one rather than sold as a path', () => {
  assert.equal(compareMove({ a: 3.0, m: 6.4 }, { a: 4.6, m: 7.3 }).kind, 'trade');
  assert.equal(compareMove({ a: 6.4, m: 8.9 }, { a: 5.0, m: 7.0 }).kind, 'trade');
  assert.equal(compareMove({ a: 3.0, m: 6.4 }, { a: 7.8, m: 5.0 }).kind, 'exposed');
  assert.equal(compareMove({ a: 3.0, m: 6.4 }, { a: null, m: 5.0 }).kind, 'unknown');
  assert.equal(compareMove(null, { a: 3, m: 3 }).kind, 'unknown');
});

test('software developer gets no paths, because none of its neighbours differ', () => {
  const index = occupationIndex(DATA);
  const { paths, others, total } = evolutionPaths(developer(), index);
  assert.equal(total, 3);
  assert.deepEqual(paths, []);
  assert.deepEqual(others.map((card) => card.kind), ['sideways', 'sideways', 'exposed']);
  assert.deepEqual(others.map((card) => card.slug).sort(),
    ['ict-application-configurator', 'language-engineer', 'web-developer']);
});

test('a path card carries what the link needs to render', () => {
  const index = occupationIndex(DATA);
  const { paths, others } = evolutionPaths(nurse(), index);
  assert.deepEqual(paths.map((card) => card.slug), ['counsellor']);
  assert.deepEqual(others.map((card) => card.kind), ['trade', 'exposed']);
  assert.deepEqual(paths[0], {
    slug: 'counsellor',
    title: 'counsellor',
    auto: 2.1,
    amp: 7.4,
    q: 'EVOLVE',
    overlap: 0.2,
    gapCount: 0,
    kind: 'better',
    label: 'Less exposed and more amplified',
  });
});

test('gap skills skip what the job already has, and what has no score', () => {
  const found = gapSkills(DATA, developer());
  assert.deepEqual(found.map((skill) => skill.id), ['fff66666', 'ggg77777']);
  assert.equal(found[0].fromTitle, 'web developer');   // first adjacency wins the credit
  assert.equal(found[0].amp, 9.0);
  // aaa11111 is already an essential skill of the job; hhh88888 has no score.
  assert.ok(!found.some((skill) => ['aaa11111', 'hhh88888'].includes(skill.id)));
});

test('gap skills honour the limit and survive an occupation with no neighbours', () => {
  assert.equal(gapSkills(DATA, developer(), 1).length, 1);
  assert.deepEqual(gapSkills(DATA, { se: [], so: [], adj: [] }), []);
  assert.deepEqual(gapSkills(DATA, null), []);
});

test('recently viewed keeps five slugs, most recent first, without duplicates', () => {
  assert.deepEqual(addRecent([], 'a'), ['a']);
  assert.deepEqual(addRecent(['b', 'c'], 'a'), ['a', 'b', 'c']);
  assert.deepEqual(addRecent(['a', 'b'], 'a'), ['a', 'b']);
  assert.deepEqual(addRecent(['b', 'a', 'c'], 'a'), ['a', 'b', 'c']);
  assert.deepEqual(addRecent(['b', 'c', 'd', 'e', 'f'], 'a'), ['a', 'b', 'c', 'd', 'e']);
  assert.deepEqual(addRecent(null, 'a'), ['a']);
  assert.deepEqual(addRecent([1, null, 'b', ''], 'a'), ['a', 'b']);
});
