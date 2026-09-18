import test from 'node:test';
import assert from 'node:assert/strict';

import {
  QUADRANTS, SHARES, SKILL_CLASS_ORDER, TYPE_DESCRIPTIONS, TYPE_NAMES, TYPE_ORDER,
  SKILL_NEAR_MARGIN, TYPE_RULES, TYPE_SHORT, colorVarOf, explain, explainType, isNear, isSkillNear, orderForCounts,
  orderOf, schemeOf, schemeOfCode, skillClassName, skillClassOf, splitOf, thresholdOf,
  typeDescription, typeLabel, typeRule, typeShortLabel,
} from '../../site/js/scheme.js';

const JOB = {
  t: 'bookkeeper', q: 'TRANSFORMING', sh: [0.61, 0.22, 0.0, 0.17], nl: true,
};

test('the scheme comes from the stats file, and a file without it is an old set', () => {
  assert.equal(schemeOf({ scheme: 'shares' }), SHARES);
  assert.equal(schemeOf({ scheme: 'quadrants' }), QUADRANTS);
  assert.equal(schemeOf({}), QUADRANTS);
  assert.equal(schemeOf(null), QUADRANTS);
  assert.equal(schemeOf({ scheme: 'nonsense' }), QUADRANTS);
});

test('every type has a name, a short name, a colour, a rule and a description', () => {
  assert.equal(TYPE_ORDER.length, 7);
  for (const code of TYPE_ORDER) {
    assert.ok(TYPE_NAMES[code], `${code} has no name`);
    assert.ok(TYPE_SHORT[code], `${code} has no short name`);
    assert.match(colorVarOf(code), /^--type-/);
    assert.ok(TYPE_RULES[code].length > 20, `${code} has no rule`);
    assert.ok(TYPE_DESCRIPTIONS[code].endsWith('.'), `${code} description is not a sentence`);
  }
});

test('no wording calls a job "at risk"', () => {
  const copy = [...Object.values(TYPE_DESCRIPTIONS), ...Object.values(TYPE_RULES),
    ...explainType(JOB).sentences].join(' ');
  assert.equal(/at risk/i.test(copy), false);
});

test('rule 1 claims the substituted share and nothing about spread', () => {
  // The spread condition was dropped after the full run (docs/scoring-v2.md).
  const job = { q: 'AUTOMATION_HEAVY', sh: [0.6, 0.2, 0.1, 0.1] };
  const copy = [TYPE_RULES.AUTOMATION_HEAVY, ...explainType(job).sentences].join(' ');
  assert.doesNotMatch(copy, /agree|spread|sigma|deviation/i);
  assert.match(copy, /half or more/i);
});

test('a code names itself, whichever scheme it belongs to', () => {
  assert.equal(typeLabel('TRANSFORM'), 'Transform');
  assert.equal(typeLabel('TRANSFORMING'), 'Transforming');
  assert.equal(typeLabel('INSULATED_PEOPLE'), 'Insulated by work with people');
  assert.equal(typeShortLabel('INSULATED_PEOPLE'), 'People work');
  assert.equal(typeLabel(undefined), 'Not scored');
  assert.equal(typeLabel('NONSENSE'), 'Not scored');
});

test('an explicit scheme keeps the other scheme\'s codes out', () => {
  assert.equal(typeLabel('TRANSFORMING', QUADRANTS), 'Not scored');
  assert.equal(typeLabel('TRANSFORM', SHARES), 'Not scored');
  assert.equal(colorVarOf('TRANSFORM', QUADRANTS), '--q-transform');
  assert.equal(colorVarOf('TRANSFORM', SHARES), null);
});

test('the two code sets are disjoint, so a bare code resolves its scheme', () => {
  for (const code of TYPE_ORDER) assert.equal(schemeOfCode(code), SHARES);
  for (const code of orderOf(QUADRANTS)) assert.equal(schemeOfCode(code), QUADRANTS);
  assert.equal(schemeOfCode('NONSENSE'), null);
  assert.equal(schemeOfCode(undefined), null);
});

test('the order of a scheme is a fresh array the caller may not corrupt', () => {
  const order = orderOf(SHARES);
  order.push('OOPS');
  assert.equal(orderOf(SHARES).length, 7);
  assert.deepEqual(orderOf(QUADRANTS), ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
  assert.deepEqual(orderOf(undefined), ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
});

test('a counts object says which order it is keyed by', () => {
  assert.deepEqual(orderForCounts({ MIXED: 3, AUGMENTED: 1 }), TYPE_ORDER);
  assert.deepEqual(orderForCounts({ TRANSFORM: 2 }), orderOf(QUADRANTS));
  assert.deepEqual(orderForCounts({}), orderOf(QUADRANTS));
  assert.deepEqual(orderForCounts(null), orderOf(QUADRANTS));
});

test('the split reads whichever field the stats file carries', () => {
  const shares = splitOf({
    scheme: 'shares',
    types: { order: TYPE_ORDER, counts: { MIXED: 5 }, shares: { MIXED: 0.5 } },
  });
  assert.equal(shares.scheme, SHARES);
  assert.equal(shares.counts.MIXED, 5);
  assert.deepEqual(shares.order, TYPE_ORDER);

  const quadrants = splitOf({ quadrants: { counts: { STABLE: 2 }, shares: { STABLE: 0.2 } } });
  assert.equal(quadrants.scheme, QUADRANTS);
  assert.equal(quadrants.counts.STABLE, 2);
  assert.deepEqual(quadrants.order, orderOf(QUADRANTS));
});

test('a stats file with no split at all yields empty counts, not undefined', () => {
  const split = splitOf({ scheme: 'shares' });
  assert.deepEqual(split.counts, {});
  assert.deepEqual(split.shares, {});
  assert.deepEqual(splitOf(null).counts, {});
});

test('a missing cut-off is null, never a number the scheme does not have', () => {
  assert.equal(thresholdOf({ threshold: 6 }), 6);
  assert.equal(thresholdOf({ scheme: 'shares' }), null);
  assert.equal(thresholdOf(null), null);
});

test('skill classes have names and colours; the quadrant scheme has none', () => {
  assert.deepEqual(SKILL_CLASS_ORDER, ['S', 'A', 'M', 'I']);
  assert.equal(skillClassName('S'), 'AI can take over');
  assert.equal(skillClassName('I'), 'Stays human');
  assert.equal(skillClassName('x'), 'Not scored');
  assert.equal(skillClassOf({ c: 'M' }), 'M');
  assert.equal(skillClassOf({ c: 'M' }, QUADRANTS), null);
  assert.equal(skillClassOf({ c: '2512' }), null, 'an ISCO code is not a class');
  assert.equal(skillClassOf(null), null);
});

test('explainType says which rule matched, in words, with the job\'s own numbers', () => {
  const explanation = explainType(JOB);
  assert.equal(explanation.heading, 'Why "Transforming"?');
  assert.equal(explanation.nearLine, true);
  assert.match(explanation.sentences[0], /AI can take over covers 61%/);
  assert.match(explanation.sentences[0], /AI assists another 22%/);
  assert.match(explanation.sentences[0], /rule for "Transforming"/);
  assert.match(explanation.sentences[1], /AI can take over 61% · AI assists 22%/);
  assert.match(explanation.sentences[2], /near a cut-off/);
  assert.match(explanation.sentences[3], /model estimates/);
});

test('a job away from the cut-offs says so instead of hedging', () => {
  const explanation = explainType({ ...JOB, nl: false });
  assert.equal(explanation.nearLine, false);
  assert.match(explanation.sentences[2], /No share is within 5 points/);
});

test('Mixed is explained as a residual class, not apologised for', () => {
  const explanation = explainType({ t: 'x', q: 'MIXED', sh: [0.2, 0.2, 0.3, 0.3], nl: false });
  const last = explanation.sentences.at(-1);
  assert.match(last, /residual class/);
  assert.match(last, /not a finding that AI will leave the job alone/);
});

test('every type produces a complete explanation', () => {
  for (const code of TYPE_ORDER) {
    const explanation = explainType({ t: 'x', q: code, sh: [0.25, 0.25, 0.25, 0.25] });
    assert.equal(explanation.heading, `Why "${TYPE_NAMES[code]}"?`);
    assert.ok(explanation.sentences.length >= 4, `${code} says too little`);
    for (const sentence of explanation.sentences) assert.ok(sentence.endsWith('.'));
  }
});

test('an unscored job gets no type and says so', () => {
  for (const job of [{}, { q: 'MIXED' }, { q: 'MIXED', sh: [0.5, 0.5] }, null]) {
    const explanation = explainType(job);
    assert.equal(explanation.heading, 'Not scored');
    assert.equal(explanation.nearLine, false);
  }
});

test('the explanation quotes nothing but the job itself', () => {
  const numbers = explainType(JOB).sentences.join(' ').match(/\d+(\.\d+)?/g) || [];
  const allowed = new Set(['61', '22', '0', '17', '5']);
  for (const found of numbers) assert.ok(allowed.has(found), `${found} came from nowhere`);
});

test('one entry point serves both schemes', () => {
  assert.equal(explain(JOB).heading, 'Why "Transforming"?');
  assert.equal(explain({ t: 'x', a: 8.6, m: 5.2, q: 'SHRINK' }).heading, 'Why "Shrink"?');
  assert.equal(explain({ t: 'x', a: 8.6, m: 5.2 }, QUADRANTS).heading, 'Why "Shrink"?');
});

test('near the line reads nl under shares and the distance rule under quadrants', () => {
  assert.equal(isNear(JOB), true);
  assert.equal(isNear({ ...JOB, nl: false }), false);
  assert.equal(isNear({ q: 'SHRINK', a: 5.5, m: 1 }), true);
  assert.equal(isNear({ q: 'STABLE', a: 5.4, m: 1 }), false);
  assert.equal(isNear({ a: 5.5, m: 1 }, QUADRANTS), true);
  assert.equal(isNear(null), false);
});

test('descriptions and rules are reachable one at a time too', () => {
  assert.equal(typeDescription('MIXED'), TYPE_DESCRIPTIONS.MIXED);
  assert.equal(typeDescription('TRANSFORM'), '');
  assert.equal(typeRule('AUGMENTED'), TYPE_RULES.AUGMENTED);
  assert.equal(typeRule('nope'), '');
});


test('a skill is near the line when a deciding chance sits within the margin of the cut', () => {
  assert.equal(SKILL_NEAR_MARGIN, 0.05);
  // "technical drawings": misses "AI can take over" by 0.02 and is printed "Stays human".
  assert.equal(isSkillNear({ c: 'I', p: [0.48, 0.2, 0.01] }), true);
  assert.equal(isSkillNear({ c: 'S', p: [0.55, 0.9, 0.0] }), true);
  assert.equal(isSkillNear({ c: 'S', p: [0.56, 0.5, 0.5] }), false);
  // Only the comparisons made before the class was settled count.
  assert.equal(isSkillNear({ c: 'A', p: [0.1, 0.9, 0.5] }), false);
  assert.equal(isSkillNear({ c: 'M', p: [0.1, 0.46, 0.9] }), true);
  assert.equal(isSkillNear({ c: 'I', p: [0.1, 0.2, 0.3] }), false);
});

test('a skill without published chances is never called near the line', () => {
  assert.equal(isSkillNear({ c: 'S' }), false);
  assert.equal(isSkillNear({ a: 6.1, m: 5.9 }), false);
  assert.equal(isSkillNear(null), false);
  assert.equal(isSkillNear({ c: 'S', p: [0.62] }, 0.6), true);
});
