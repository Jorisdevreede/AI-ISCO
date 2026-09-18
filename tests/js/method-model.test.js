import test from 'node:test';
import assert from 'node:assert/strict';

import { compareScoreSets } from '../../site/js/agreement.js';
import { agreementTable, agreementValues } from '../../site/js/pages/method-model.js';

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
