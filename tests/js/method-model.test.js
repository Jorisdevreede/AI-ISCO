import test from 'node:test';
import assert from 'node:assert/strict';

import { compareScoreSets } from '../../site/js/agreement.js';
import { splitOf } from '../../site/js/scheme.js';
import {
  SKILL_CLASS_RULE_TEXT, TYPE_RULE_TEXT, agreementTable, agreementValues, methodValues,
  quadrantRuleRows, skillClassRows, typeRuleRows,
} from '../../site/js/pages/method-model.js';

const row = (s, a, m, q) => ({ s, a, m, q });
const FIRST = [row('nurse', 3, 7, 'EVOLVE'), row('clerk', 8, 7, 'TRANSFORM'),
  row('mason', 2, 3, 'STABLE'), row('packer', 7, 3, 'SHRINK')];
const SECOND = [row('nurse', 6.5, 6.4, 'TRANSFORM'), row('clerk', 8.5, 6.2, 'TRANSFORM'),
  row('mason', 3, 2.5, 'STABLE'), row('packer', 7.5, 2, 'SHRINK')];
const LABELS = ['Gemini', 'TypeSafe'];

test('every slot gets text computed from the comparison, none typed in', () => {
  const values = agreementValues(compareScoreSets(FIRST, SECOND), LABELS);

  assert.equal(values.share, '75%');
  assert.equal(values.n, '4');
  assert.equal(values.moved, '1');
  assert.equal(values.rhoAutomation, '1.00');
  assert.equal(values.rhoAmplification, '0.89');
  assert.equal(values.meanAutomation, '5.0 under Gemini and 6.4 under TypeSafe');
  assert.equal(values.firstLabel, 'Gemini');
  assert.equal(values.secondLabel, 'TypeSafe');
});

test('the largest move is worded with both models named', () => {
  const values = agreementValues(compareScoreSets(FIRST, SECOND), LABELS);

  assert.equal(values.move,
    'The largest single move is 1 occupation from Evolve under Gemini to Transform under TypeSafe.');
});

test('two sets that agree everywhere get no move sentence and an undefined-safe correlation', () => {
  const same = compareScoreSets(FIRST, FIRST);
  const flat = compareScoreSets([row('a', 5, 5, 'STABLE'), row('b', 5, 5, 'STABLE')],
    [row('a', 4, 4, 'STABLE'), row('b', 6, 6, 'TRANSFORM')]);

  assert.equal(agreementValues(same, LABELS).move, '');
  assert.equal(agreementValues(same, LABELS).share, '100%');
  assert.equal(agreementValues(flat, LABELS).rhoAutomation, 'not defined');
});

test('the table has a row and a column per box and marks the diagonal', () => {
  const table = agreementTable(compareScoreSets(FIRST, SECOND));

  assert.deepEqual(table.columns, ['Transform', 'Stable', 'Evolve', 'Shrink']);
  assert.deepEqual(table.rows.map((entry) => entry.label), table.columns);
  const evolve = table.rows[2];
  assert.deepEqual(evolve.cells.map((cell) => cell.text), ['1', '0', '0', '0']);
  assert.deepEqual(evolve.cells.map((cell) => cell.agrees), [false, false, true, false]);
});

/* --- what the page says about each scheme --------------------------------- */

const QUADRANT_STATS = {
  built: '2026-06-10',
  threshold: 6,
  occupations: 3039,
  skills_scored: 13475,
  quadrants: {
    counts: { EVOLVE: 1561, SHRINK: 184, STABLE: 790, TRANSFORM: 508 },
    shares: { EVOLVE: 0.513, SHRINK: 0.0605, STABLE: 0.2596, TRANSFORM: 0.1669 },
  },
  near_line: { count: 1509, share: 0.4959 },
};

const SHARES_STATS = {
  built: '2026-09-18',
  scheme: 'shares',
  model: 'jev-1.13.0',
  occupations: 3039,
  skills_scored: 13939,
  types: {
    order: ['AUTOMATION_HEAVY', 'TRANSFORMING', 'AUGMENTED', 'MECHANISABLE',
      'INSULATED_PHYSICAL', 'INSULATED_PEOPLE', 'MIXED'],
    counts: {
      AUTOMATION_HEAVY: 137,
      TRANSFORMING: 207,
      AUGMENTED: 943,
      MECHANISABLE: 274,
      INSULATED_PHYSICAL: 1004,
      INSULATED_PEOPLE: 387,
      MIXED: 91,
    },
    shares: {
      AUTOMATION_HEAVY: 0.045,
      TRANSFORMING: 0.068,
      AUGMENTED: 0.31,
      MECHANISABLE: 0.09,
      INSULATED_PHYSICAL: 0.33,
      INSULATED_PEOPLE: 0.127,
      MIXED: 0.03,
    },
  },
  skill_classes: {
    counts: { S: 1742, A: 2620, M: 1338, I: 8239 },
    shares: { S: 0.125, A: 0.188, M: 0.096, I: 0.591 },
  },
  near_line: { count: 802, share: 0.2636 },
};

test('the quadrant slots keep the wording and the formatting they had', () => {
  const values = methodValues(QUADRANT_STATS, 0.5);
  assert.equal(values.skills, '13,475');
  assert.equal(values.occupations, '3,039');
  assert.equal(values.threshold, '6');
  assert.equal(values.nearLineCount, '1,509');
  assert.equal(values.nearLineShare, '50%');
  assert.equal(values.nearLineDistance, '0.5');
  assert.equal(values.built, '2026-06-10');
});

test('a shares set never prints 6 as a cut-off, and names its model', () => {
  const values = methodValues(SHARES_STATS, 0.5);
  assert.equal(values.threshold, 'none');
  assert.equal(values.classCut, 'one half');
  assert.equal(values.model, 'jev-1.13.0');
  assert.equal(values.mixedCount, '91');
  assert.equal(values.mixedShare, '3%');
  assert.equal(values.nearLineCount, '802');
  // A set that does publish the probability cut-off prints that instead.
  assert.equal(methodValues({ ...SHARES_STATS, threshold: 0.5 }, 0.5).classCut, '0.5');
});

test('a stats file with nothing in it degrades rather than throwing', () => {
  const values = methodValues({}, 0.5);
  assert.equal(values.model, 'not recorded');
  assert.equal(values.built, 'not recorded');
  assert.equal(values.classCut, 'one half');
  assert.equal(values.skills, 'Not scored');
});

test('the quadrant rule table is the four boxes, counted and shared as before', () => {
  const rows = quadrantRuleRows(splitOf(QUADRANT_STATS));
  assert.deepEqual(rows.map((row) => row.code), ['TRANSFORM', 'EVOLVE', 'STABLE', 'SHRINK']);
  assert.deepEqual(rows.map((row) => row.count), ['508', '1,561', '790', '184']);
  assert.deepEqual(rows.map((row) => row.share), ['17%', '51%', '26%', '6%']);
  assert.match(rows[0].rule, /Both scores at or above the cut-off/);
});

test('the type rule table follows the rules the pipeline actually ran', () => {
  const rows = typeRuleRows(splitOf(SHARES_STATS));
  assert.equal(rows.length, 7);
  assert.deepEqual(rows.map((row) => row.code), SHARES_STATS.types.order);
  assert.equal(rows[0].count, '137');
  assert.equal(rows[4].share, '33%');
  // Rule 1 lost its spread condition after the full run: it must not be claimed.
  assert.equal(TYPE_RULE_TEXT.AUTOMATION_HEAVY,
    'Half or more of the skill weight is work AI can take over.');
  assert.doesNotMatch(TYPE_RULE_TEXT.AUTOMATION_HEAVY, /agree|spread|sigma|deviation/i);
  assert.equal(rows.every((row) => row.rule.length > 0), true);
});

test('the class rule table chains, so "first match wins" is visible', () => {
  const rows = skillClassRows(SHARES_STATS);
  assert.deepEqual(rows.map((row) => row.code), ['S', 'A', 'M', 'I']);
  assert.deepEqual(rows.map((row) => row.name),
    ['AI can take over', 'AI assists', 'Machines can do', 'Stays human']);
  assert.deepEqual(rows.map((row) => row.count), ['1,742', '2,620', '1,338', '8,239']);
  assert.deepEqual(rows.map((row) => row.share), ['13%', '19%', '10%', '59%']);
  assert.match(SKILL_CLASS_RULE_TEXT.A, /clear gain across most of the work/);
  assert.match(SKILL_CLASS_RULE_TEXT.I, /None of the three/);
});

test('the class table falls back to counts when the file publishes no shares', () => {
  const rows = skillClassRows({ skill_classes: { counts: { S: 1, A: 1, M: 0, I: 2 } } });
  assert.deepEqual(rows.map((row) => row.share), ['25%', '25%', '0%', '50%']);
  assert.deepEqual(skillClassRows({}).map((row) => row.count), ['0', '0', '0', '0']);
});

test('no rule text on this page mentions the other scheme', () => {
  const text = `${Object.values(TYPE_RULE_TEXT).join(' ')} `
    + `${Object.values(SKILL_CLASS_RULE_TEXT).join(' ')}`;
  assert.doesNotMatch(text, /quadrant|\bbox\b|cut at 6|at risk/i);
  assert.doesNotMatch(text, /\d/);
});

test('the agreement slots carry the subset and the whole population (A8)', () => {
  const values = agreementValues(compareScoreSets(FIRST, SECOND), LABELS);
  assert.equal(values.n, '4');
  assert.equal(values.total, '4');
  // A run where one file scores an occupation the other does not.
  const wider = compareScoreSets(
    [...FIRST, row('extra', 5, 5, 'STABLE')],
    [...SECOND, row('extra', null, 5, null)],
  );
  assert.equal(agreementValues(wider, LABELS).n, '4');
  assert.equal(agreementValues(wider, LABELS).total, '5');
});

test('the comparison names its population only when it is not the whole (A8)', () => {
  const result = compareScoreSets(FIRST, SECOND);

  // Two runs that cover every occupation the site has: nothing to explain, so
  // the note just names the total. It reads after a <strong> holding the share.
  assert.equal(agreementValues(result, LABELS, 4).note,
    'of the 4 occupations land in the same box under both models. '
    + 'The other 1 change box.');

  // A site with more occupations than the two runs line up: say so first.
  assert.equal(agreementValues(result, LABELS, 8).note,
    '4 of the 8 occupations are scored in both runs. Of those, 75% land in the '
    + 'same box under both models. The other 1 change box.');

  // With no site total it falls back to what the two files themselves show.
  assert.equal(agreementValues(result, LABELS).total, '4');
  assert.match(agreementValues(result, LABELS).note, /^of the 4 occupations/);
});

test('no figure in the agreement note is typed rather than computed (A8)', () => {
  const result = compareScoreSets(FIRST, SECOND);
  for (const total of [4, 8]) {
    const note = agreementValues(result, LABELS, total).note;
    const computed = new Set([String(result.n), String(total),
      String(result.n - result.same), String(Math.round(result.share * 100))]);
    for (const figure of note.match(/\d+/g) || []) {
      assert.ok(computed.has(figure), `${figure} in "${note}" is not from the data`);
    }
  }
});
