import test from 'node:test';
import assert from 'node:assert/strict';

import {
  LARGE_SUBSTITUTED_SHARE, MOVE_MARGIN, SHARE_MARGIN, UNITS_FILE, addRecent, advicePosition,
  backTarget,
  classSkillLists, compareMove, compareShareMove, essentialSkills, evolutionPaths,
  findOccupation, gapSkills, groupChain, occupationIndex, occupationSkills, resolveSkills,
  chanceLine, classTaskLists, decidingChances, largestShareSentence, scatterCells,
  scatterColumns, scatterPoints, scatterSummary, schemeOfSet, scoreCards, shareMeanings,
  skillListSpecs, skillNear, splitSkills, staysHumanNote, unitFile, unitGroupKey,
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
  all: { label: 'All occupations', n: 3039 },
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
    mech: null,
    cls: null,
    probs: null,
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
    mech: null,
    q: 'EVOLVE',
    sh: null,
    nl: false,
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
  assert.equal(gapSkills(DATA, developer(), { limit: 1 }).length, 1);
  assert.deepEqual(gapSkills(DATA, { se: [], so: [], adj: [] }), []);
  assert.deepEqual(gapSkills(DATA, null), []);
});

// --- the shares scheme -------------------------------------------------------
//
// A second synthetic fixture, in the shape a `_v2` set has: every skill carries
// a class `c`, a third score `k` and the three probabilities `p`; every
// occupation carries `sh`, `nl`, `why`, `ak` and a type code in `q`. Every
// number below is invented for this test.
const SHARES_DATA = {
  skills: {
    s1: { t: 'draft a report', a: 8.0, m: 7.0, k: 1.5, c: 'S', p: [0.71, 0.55, 0.02], r: 'Made up.' },
    s2: { t: 'write computer code', a: 7.5, m: 9.0, k: 1.2, c: 'S', p: [0.62, 0.80, 0.01] },
    a1: { t: 'plan a project', a: 4.0, m: 8.5, k: 1.0, c: 'A', p: [0.20, 0.77, 0.03] },
    a2: { t: 'advise a client', a: 3.0, m: 7.6, k: 1.0, c: 'A', p: [0.11, 0.63, 0.01] },
    m1: { t: 'operate a press', a: 3.5, m: 3.0, k: 7.2, c: 'M', p: [0.05, 0.10, 0.66] },
    i1: { t: 'lift a patient', a: 1.5, m: 2.0, k: 2.0, c: 'I', p: [0.02, 0.21, 0.14] },
    i2: { t: 'calm a relative', a: 2.0, m: 4.0, k: 1.5, c: 'I', p: [0.03, 0.30, 0.02] },
    i3: { t: 'skill with no assistance score', a: 3.0, m: null, k: 1.0, c: 'I', p: [0.01, 0.20, 0.05] },
    g1: { t: 'gap skill AI assists with', a: 3.0, m: 8.8, k: 1.0, c: 'A', p: [0.10, 0.90, 0.00] },
    g2: { t: 'gap skill that stays human', a: 2.0, m: 5.0, k: 1.0, c: 'I', p: [0.05, 0.30, 0.10] },
    g3: { t: 'gap skill AI can take over', a: 9.0, m: 9.5, k: 1.0, c: 'S', p: [0.95, 0.90, 0.00] },
  },
  occupations: [
    {
      t: 'technical writer', s: 'technical-writer', c: '2642', mg: 'Professionals',
      q: 'TRANSFORMING', e: 5.0, ar: 6.5, ap: 7.8, ak: 1.4,
      sh: [0.40, 0.25, 0.05, 0.30], nl: false, why: 'people',
      se: ['s1', 's2', 'a1'], so: ['a2', 'm1', 'i1', 'i2', 'i3'],
      adj: [
        { s: 'content-manager', t: 'content manager', ov: 0.4, q: 'AUGMENTED', gap: ['g1', 'g2'] },
        { s: 'copy-editor', t: 'copy editor', ov: 0.3, q: 'AUTOMATION_HEAVY', gap: ['g3'] },
        { s: 'documentation-clerk', t: 'documentation clerk', ov: 0.2, q: 'TRANSFORMING', gap: [] },
      ],
      n: { adv: 'Made-up advice.' },
    },
    {
      t: 'content manager', s: 'content-manager', c: '2431', q: 'AUGMENTED',
      ar: 4.4, ap: 8.2, ak: 1.1, sh: [0.15, 0.50, 0.02, 0.33], nl: false, why: 'people',
    },
    {
      t: 'copy editor', s: 'copy-editor', c: '2642', q: 'AUTOMATION_HEAVY',
      ar: 8.1, ap: 7.0, ak: 1.0, sh: [0.62, 0.18, 0.00, 0.20], nl: true, why: 'people',
    },
    {
      t: 'documentation clerk', s: 'documentation-clerk', c: '4110', q: 'TRANSFORMING',
      ar: 6.6, ap: 7.6, ak: 1.2, sh: [0.44, 0.22, 0.04, 0.30], nl: false, why: 'other',
    },
    {
      t: 'welder', s: 'welder', c: '7212', q: 'MECHANISABLE', ar: 2.0, ap: 3.0, ak: 6.0,
      sh: [0.01, 0.06, 0.30, 0.63], nl: false, why: 'physical',
      se: ['m1'], so: [], adj: [],
    },
  ],
};

const SHARES = 'shares';
const writer = () => findOccupation(SHARES_DATA, 'technical-writer');
const welder = () => findOccupation(SHARES_DATA, 'welder');
const writerSkills = () => occupationSkills(SHARES_DATA, writer());

test('the scheme comes from the stats file, and from the data when that is missing', () => {
  assert.equal(schemeOfSet({ scheme: 'shares' }, DATA), 'shares');
  assert.equal(schemeOfSet({ threshold: 6 }, SHARES_DATA), 'quadrants');
  assert.equal(schemeOfSet(null, SHARES_DATA), 'shares');
  assert.equal(schemeOfSet(null, DATA), 'quadrants');
  assert.equal(schemeOfSet(null, null), 'quadrants');
});

test('a skill carries its class, its third score and its probabilities', () => {
  const [skill] = resolveSkills(SHARES_DATA.skills, ['s1'], true);
  assert.equal(skill.cls, 'S');
  assert.equal(skill.mech, 1.5);
  assert.deepEqual(skill.probs, [0.71, 0.55, 0.02]);
  // The same call against a quadrant set leaves all three null rather than NaN.
  const [old] = resolveSkills(DATA.skills, ['aaa11111']);
  assert.deepEqual([old.cls, old.mech, old.probs], [null, null, null]);
});

test('skills group by class, each group led by the score that class is about', () => {
  const lists = classSkillLists(writerSkills());
  assert.deepEqual(lists.S.map((skill) => skill.id), ['s1', 's2']);
  assert.deepEqual(lists.A.map((skill) => skill.id), ['a1', 'a2']);
  assert.deepEqual(lists.M.map((skill) => skill.id), ['m1']);
  assert.deepEqual(lists.I.map((skill) => skill.id), ['i2', 'i1', 'i3']);
});

test('under shares the page shows one list per class, counted from the data', () => {
  const specs = skillListSpecs(writerSkills(), SHARES);
  assert.deepEqual(specs.map((spec) => spec.key), ['class-S', 'class-A', 'class-M']);
  assert.deepEqual(specs.map((spec) => spec.headingKey),
    ['classSHeading', 'classAHeading', 'classMHeading']);
  assert.deepEqual(specs.map((spec) => spec.primary), ['auto', 'amp', 'mech']);
  assert.match(specs[0].note, /2 of the 8 skills in this job are ones AI can take over/);
  assert.match(specs[0].note, /All three scores are shown, AI substitution first/);
  assert.doesNotMatch(specs.map((spec) => spec.note).join(' '), /cut-off|quadrant|\bbox\b/i);
});

test('a job with no mechanised skills gets no "machines can do" list', () => {
  const without = writerSkills().filter((skill) => skill.cls !== 'M');
  assert.deepEqual(skillListSpecs(without, SHARES).map((spec) => spec.code), ['S', 'A']);
  assert.equal(skillListSpecs([], SHARES)[0].note, 'No skill in this job is one AI can take over.');
});

test('the quadrant lists are untouched by the shares branch', () => {
  const specs = skillListSpecs(occupationSkills(DATA, developer()));
  assert.deepEqual(specs.map((spec) => spec.key), ['dep', 'app']);
  assert.deepEqual(specs[0].skills.map((skill) => skill.id),
    splitSkills(occupationSkills(DATA, developer())).depreciating.map((skill) => skill.id));
  assert.match(specs[0].note, /score 6 or more for automation risk/);
});

test('the skills no list covers are counted, and "stays human" is not "AI is no help"', () => {
  const note = staysHumanNote(writerSkills());
  assert.match(note, /The other 3 of the 8 skills in this job stay human/);
  assert.match(note, /AI can still help with parts of it/);
  // C14: the reassurance is a positive sentence, not a negated negative.
  assert.doesNotMatch(note, /not the same as|no help/);
  assert.equal(staysHumanNote([]), '');
});

test('the shares scatter has no boxes: a dot carries its class instead', () => {
  const points = scatterPoints(writerSkills(), SHARES);
  assert.deepEqual(points.map((point) => point.id), ['s1', 's2', 'a1', 'a2', 'm1', 'i1', 'i2']);
  assert.equal(points.every((point) => point.q === null), true);
  assert.deepEqual(points.map((point) => point.cls), ['S', 'S', 'A', 'A', 'M', 'I', 'I']);
});

test('the shares aria-label counts the classes and draws no cut-off', () => {
  const label = scatterSummary('technical writer', scatterPoints(writerSkills(), SHARES), SHARES);
  assert.match(label, /Scatter plot of 7 skills in technical writer/);
  assert.match(label, /AI substitution 1 to 10 runs left to right/);
  assert.match(label, /2 of the skills are ones AI can take over, 2 ones AI assists with, 1 ones machines can do and 2 stay human/);
  assert.match(label, /3 are essential to the job and 4 optional/);
  assert.doesNotMatch(label, /cut-off|quadrant|\bbox\b/i);
});

test('the table alternative lists the class and all three scores', () => {
  assert.deepEqual(scatterColumns(SHARES).map((column) => column.label), [
    'Skill', 'What AI can do with it', 'AI substitution (out of 10)',
    'AI assistance (out of 10)', 'Machine automation (out of 10)', 'In this job',
  ]);
  const [point] = scatterPoints(writerSkills(), SHARES);
  assert.deepEqual(scatterCells(point, SHARES), [
    { value: 'AI can take over', numeric: false },
    { value: 8.0, numeric: true },
    { value: 7.0, numeric: true },
    { value: 1.5, numeric: true },
    { value: 'Essential', numeric: false },
  ]);
  // The quadrant table is exactly the three cells it always had.
  assert.deepEqual(scatterColumns().map((column) => column.label),
    ['Skill', 'Automation risk (out of 10)', 'Amplification (out of 10)', 'In this job']);
  assert.equal(scatterCells(point).length, 3);
});

test('a move under shares is judged on the shares, with a margin of its own', () => {
  assert.equal(SHARE_MARGIN, 0.10);
  const from = { sh: [0.40, 0.25, 0.05, 0.30] };
  assert.deepEqual(compareShareMove(from, { sh: [0.15, 0.50, 0.02, 0.33] }),
    { kind: 'better', label: 'Less of it can be taken over, and more of it is assisted' });
  assert.equal(compareShareMove(from, { sh: [0.25, 0.25, 0.10, 0.40] }).kind, 'better');
  assert.equal(compareShareMove(from, { sh: [0.40, 0.40, 0.05, 0.15] }).kind, 'better');
  assert.equal(compareShareMove(from, { sh: [0.62, 0.18, 0.00, 0.20] }).kind, 'exposed');
  assert.equal(compareShareMove(from, { sh: [0.44, 0.22, 0.04, 0.30] }).kind, 'sideways');
  assert.equal(compareShareMove(from, { a: 3, m: 4 }).kind, 'unknown');
});

test('a sideways move is never sold as a step up', () => {
  const from = { sh: [0.40, 0.25, 0.05, 0.30] };
  const sideways = compareShareMove(from, { sh: [0.44, 0.22, 0.04, 0.30] });
  assert.match(sideways.label, /sideways move/);
  assert.doesNotMatch(sideways.label, /better|less|more|step up/i);
  const trade = compareShareMove(from, { sh: [0.55, 0.40, 0.00, 0.05] });
  assert.equal(trade.kind, 'trade');
  assert.match(trade.label, /but more can be taken over too/);
});

test('the neighbours of a shares job sort into moves, sideways and more exposed', () => {
  const index = occupationIndex(SHARES_DATA);
  const { paths, others, total } = evolutionPaths(writer(), index, { scheme: SHARES });
  assert.equal(total, 3);
  assert.deepEqual(paths.map((card) => card.slug), ['content-manager']);
  assert.deepEqual(others.map((card) => [card.slug, card.kind]),
    [['documentation-clerk', 'sideways'], ['copy-editor', 'exposed']]);
  assert.deepEqual(paths[0].sh, [0.15, 0.50, 0.02, 0.33]);
  assert.equal(paths[0].q, 'AUGMENTED');
  assert.equal(others[1].nl, true);
});

test('gap skills rank by net gain and leave out what AI can take over (A21)', () => {
  const found = gapSkills(SHARES_DATA, writer(), { scheme: SHARES });
  // g1 gains 8.8 - 3.0; g2 gains 5.0 - 2.0; g3 is class S and is not offered.
  assert.deepEqual(found.map((skill) => skill.id), ['g1', 'g2']);
  assert.deepEqual(found.map((skill) => skill.gain), [5.8, 3]);
  assert.equal(found.some((skill) => skill.id === 'g3'), false);
  assert.equal(gapSkills(SHARES_DATA, writer(), { scheme: SHARES, limit: 1 }).length, 1);
  // The quadrant rule cuts on the score instead: g3's 9.0 is over the cut-off.
  const quadrant = gapSkills(SHARES_DATA, writer());
  assert.deepEqual(quadrant.map((skill) => skill.id), ['g1', 'g2']);
});

test('the learn list is not one neighbour’s wish list (A21)', () => {
  // Four skills from one job beat the fifth on net gain; the fifth is the only
  // one from a second job, so it takes the last of the three places.
  const data = {
    skills: {
      x1: { t: 'x1', a: 1, m: 9 }, x2: { t: 'x2', a: 1, m: 8 },
      x3: { t: 'x3', a: 1, m: 7 }, y1: { t: 'y1', a: 1, m: 4 },
    },
    occupations: [{
      t: 'one', s: 'one', c: '1111', se: [], so: [],
      adj: [
        { s: 'a-job', t: 'a job', gap: ['x1', 'x2', 'x3'] },
        { s: 'b-job', t: 'b job', gap: ['y1'] },
      ],
    }],
  };
  const job = findOccupation(data, 'one');
  const picked = gapSkills(data, job, { limit: 3 });
  assert.deepEqual(picked.map((skill) => skill.id), ['x1', 'x2', 'y1']);
  assert.equal(new Set(picked.map((skill) => skill.fromSlug)).size, 2);
  // With only one source job there is nothing to spread, and nothing is dropped.
  const alone = { ...data, occupations: [{ ...job, adj: [job.adj[0]] }] };
  assert.deepEqual(gapSkills(alone, alone.occupations[0], { limit: 3 })
    .map((skill) => skill.id), ['x1', 'x2', 'x3']);
});

test('a score card cannot be read as the share above it (C6)', () => {
  const cards = scoreCards(SHARES);
  assert.deepEqual(cards.map((card) => card.key), ['ar', 'ap', 'ak']);
  for (const card of cards) {
    assert.match(card.label, /^Average .*score across this job’s skills$/, card.key);
    assert.equal(card.unit, '1 to 10');
  }
  // None of them may be named as one of the four shares.
  assert.doesNotMatch(cards.map((card) => card.label).join(' '),
    /AI can take over|AI assists|Machines can do|Stays human/);
  assert.deepEqual(scoreCards().map((card) => card.label),
    ['Automation risk', 'Amplification']);
  assert.deepEqual(scoreCards().map((card) => card.unit), ['out of 10', 'out of 10']);
  assert.doesNotMatch(scoreCards(SHARES).map((card) => card.hint).join(' '), /at risk|quadrant/i);
});

test('each share gets a plain sentence with the job’s own percentage in it', () => {
  const parts = shareMeanings(writer().sh, writer().why);
  assert.deepEqual(parts.map((part) => part.percent), [40, 25, 5, 30]);
  assert.deepEqual(parts.map((part) => part.name),
    ['AI can take over', 'AI assists', 'Machines can do', 'Stays human']);
  assert.match(parts[0].text, /^40% of the work this job is built from/);
  // C14: said positively, with no double negative.
  assert.match(parts[3].text, /30% is work that neither AI nor machinery gets most of the way through/);
  assert.match(parts[3].text, /AI can still help with parts of it/);
  assert.doesNotMatch(parts[3].text, /not the same as|no help/);
  assert.match(parts[3].text, /with and for other people/);
  assert.match(shareMeanings(welder().sh, welder().why)[3].text, /on things, in a place/);
  assert.deepEqual(shareMeanings(null, 'people'), []);
});

test('a job with a large substituted share leads with its next step', () => {
  assert.equal(LARGE_SUBSTITUTED_SHARE, 0.30);
  assert.equal(advicePosition(writer(), SHARES), 'top');
  assert.equal(advicePosition(welder(), SHARES), 'bottom');
  assert.equal(advicePosition({ sh: null }, SHARES), 'bottom');
  // The quadrant rule is untouched: software developer's 6.4 still leads.
  assert.equal(advicePosition(developer()), 'top');
  assert.equal(advicePosition(nurse()), 'bottom');
});

// --- the chances that decide a class (C1, C12) -------------------------------

test('a classed row can show the chances that decided it (C1)', () => {
  const parts = decidingChances([0.48, 0.37, 0.07]);
  assert.deepEqual(parts.map((part) => part.percent), ['48%', '37%', '7%']);
  assert.deepEqual(parts.map((part) => part.text), [
    'chance AI could do nearly all of it', 'clear gain with AI alongside',
    'machinery does it',
  ]);
  assert.equal(chanceLine([0.48, 0.37, 0.07]),
    '48% chance AI could do nearly all of it · 37% clear gain with AI alongside '
    + '· 7% machinery does it');
  assert.deepEqual(decidingChances(null), []);
  assert.deepEqual(decidingChances([0.1, 0.2]), []);
  assert.equal(chanceLine(undefined), '');
});

test('a skill whose class hangs on a rounding says so (C12)', () => {
  // technical drawings: substitution 0.48, class I — it misses S by 0.02.
  assert.equal(skillNear({ cls: 'I', probs: [0.48, 0.37, 0.07] }), true);
  assert.equal(skillNear({ cls: 'I', probs: [0.10, 0.20, 0.05] }), false);
  // "AI can take over" was settled by the first comparison alone.
  assert.equal(skillNear({ cls: 'S', probs: [0.53, 0.05, 0.02] }), true);
  assert.equal(skillNear({ cls: 'S', probs: [0.90, 0.48, 0.02] }), false);
  assert.equal(skillNear({ cls: null, probs: [0.5, 0.5, 0.5] }), false);
  assert.equal(skillNear({ cls: 'A' }), false);
});

// --- the task lists come from this job's own classes (C3) --------------------

test('what AI takes on is this job’s own skills, not another model’s tasks (C3)', () => {
  const lists = classTaskLists(writerSkills());
  assert.deepEqual(lists.map((list) => list.key), ['takes', 'amplifies']);
  assert.deepEqual(lists.map((list) => list.heading),
    ['What AI could take on', 'What AI amplifies']);
  // Ranked by the chance that put them in the class, not by the display score.
  assert.deepEqual(lists[0].items.map((item) => item.id), ['s1', 's2']);
  assert.deepEqual(lists[0].items.map((item) => item.percent), ['71%', '62%']);
  assert.deepEqual(lists[1].items.map((item) => item.id), ['a1', 'a2']);
  // Every named skill links out by its own id.
  assert.equal(lists[0].items.every((item) => item.id && item.title), true);
  // A job with no skills of a class simply has no list for it.
  assert.deepEqual(classTaskLists([]), []);
});

test('the first screen names one quantity: the largest share (C6)', () => {
  assert.equal(largestShareSentence([0.62, 0.25, 0.0, 0.13]),
    'Most of this job — 62% — is work an AI system could carry out almost entirely by itself.');
  // Under half it is the largest part, not "most".
  assert.match(largestShareSentence([0.40, 0.25, 0.05, 0.30]),
    /^The largest part of this job — 40% —/);
  assert.match(largestShareSentence([0.01, 0.06, 0.30, 0.63]),
    /neither AI nor machinery gets most of the way through\.$/);
  assert.equal(largestShareSentence(null), '');
});

// --- one unit group per request (A1) -----------------------------------------

test('a slug names the one small file the page has to fetch', () => {
  assert.equal(UNITS_FILE, 'jobs/units.json');
  const units = { 'software-developer': '2512', welder: '7212' };
  assert.equal(unitFile(units, 'software-developer'), 'jobs/2512');
  assert.equal(unitFile(units, 'welder'), 'jobs/7212');
  // An unknown slug is answered without a request, so it keeps its error state.
  assert.equal(unitFile(units, 'tea-taster-to-the-queen'), null);
  assert.equal(unitFile(units, ''), null);
  assert.equal(unitFile(null, 'welder'), null);
});

test('a shard answers for its own jobs and for the neighbours it carries', () => {
  const shard = {
    occupations: [{ t: 'software developer', s: 'software-developer', c: '2512', ar: 6.4, ap: 8.9 }],
    neighbours: [{ t: 'language engineer', s: 'language-engineer', c: '2643', ar: 6.8, ap: 8.9 }],
    skills: {},
  };
  assert.equal(findOccupation(shard, 'software-developer').c, '2512');
  assert.equal(findOccupation(shard, 'language-engineer').c, '2643');
  assert.equal(findOccupation(shard, 'nobody'), null);
  const index = occupationIndex(shard);
  assert.equal(index.size, 2);
  assert.equal(index.get('language-engineer').ap, 8.9);
  // A shard without neighbours is the shape the old file had, and still works.
  assert.equal(occupationIndex({ occupations: shard.occupations }).size, 1);
  assert.equal(occupationIndex(null).size, 0);
});

test('a path card reads the neighbour record the shard brought with it', () => {
  const shard = {
    occupations: [{
      t: 'one', s: 'one', c: '1111', ar: 3.0, ap: 6.4,
      adj: [{ s: 'far-away', t: 'far away', ov: 0.2, q: 'EVOLVE', gap: [] }],
    }],
    neighbours: [{ t: 'far away', s: 'far-away', c: '9999', q: 'EVOLVE', ar: 2.1, ap: 7.4 }],
    skills: {},
  };
  const { paths } = evolutionPaths(findOccupation(shard, 'one'), occupationIndex(shard));
  assert.deepEqual(paths.map((card) => [card.slug, card.auto, card.amp]),
    [['far-away', 2.1, 7.4]]);
  assert.equal(paths[0].kind, 'better');
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
