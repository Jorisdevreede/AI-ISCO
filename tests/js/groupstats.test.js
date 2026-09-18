import test from 'node:test';
import assert from 'node:assert/strict';

import {
  TABLE_COLUMNS, groupPrefix, occupationsInGroup, quadrantMix, sortOccupations, tableRows,
} from '../../site/js/groupstats.js';
import { INDEX } from './fixture.js';

const slugs = (rows) => rows.map((row) => row.s);

test('a group key becomes an ISCO prefix', () => {
  assert.equal(groupPrefix('major:2'), '2');
  assert.equal(groupPrefix('sub:25'), '25');
  assert.equal(groupPrefix('unit:2512'), '2512');
  assert.equal(groupPrefix('all'), '');
  assert.equal(groupPrefix(''), '');
});

test('filtering by group key works at every level', () => {
  assert.equal(occupationsInGroup(INDEX, 'all').length, INDEX.length);
  assert.deepEqual(slugs(occupationsInGroup(INDEX, 'unit:2512')),
    ['software-developer', 'embedded-systems-software-developer']);
  assert.equal(occupationsInGroup(INDEX, 'major:2').length, 4);
  assert.deepEqual(occupationsInGroup(INDEX, 'unit:9999'), []);
});

test('the ranked view sorts by automation, most exposed first', () => {
  const ranked = sortOccupations(occupationsInGroup(INDEX, 'all'));
  assert.equal(ranked[0].s, 'bookkeeper');
  assert.equal(ranked.at(-1).s, 'sommeliere');
});

test('sorting never mutates the caller\'s array', () => {
  const before = slugs(INDEX);
  sortOccupations(INDEX, 'amplification');
  assert.deepEqual(slugs(INDEX), before);
});

test('other sorts are available and A to Z is alphabetical', () => {
  assert.equal(sortOccupations(INDEX, 'amplification')[0].s, 'software-developer');
  assert.equal(sortOccupations(INDEX, 'title')[0].s, 'bookkeeper');
  assert.equal(sortOccupations(INDEX, 'nonsense')[0].s, 'bookkeeper'); // falls back
});

test('an explicit direction overrides the sort\'s own', () => {
  assert.equal(sortOccupations(INDEX, 'automation', false)[0].s, 'sommeliere');
});

test('the quadrant mix counts and shares every box, including empty ones', () => {
  const mix = quadrantMix(occupationsInGroup(INDEX, 'unit:2512'));
  assert.equal(mix.total, 2);
  assert.equal(mix.counts.TRANSFORM, 2);
  assert.equal(mix.counts.SHRINK, 0);
  assert.equal(mix.shares.TRANSFORM, 1);
  assert.deepEqual(mix.order, ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK']);
});

test('an empty group has shares of zero rather than NaN', () => {
  const mix = quadrantMix([]);
  assert.equal(mix.total, 0);
  assert.equal(mix.shares.EVOLVE, 0);
});

test('table rows carry the slug and formatted cells', () => {
  const rows = tableRows(occupationsInGroup(INDEX, 'unit:2512'));
  assert.equal(rows[0].slug, 'software-developer');
  assert.deepEqual(rows[0].cells.map((cell) => cell.text),
    ['software developer', '2512', '6.4', '8.9', 'Transform']);
  assert.deepEqual(rows[0].cells.map((cell) => cell.key),
    TABLE_COLUMNS.map((column) => column.key));
  assert.deepEqual(rows[0].cells.map((cell) => cell.numeric),
    [false, false, true, true, false]);
});

test('an unscored row says so in the table', () => {
  const rows = tableRows([{ t: 'mystery', s: 'm', c: '0000', a: null, m: null, q: null }]);
  assert.deepEqual(rows[0].cells.map((cell) => cell.text),
    ['mystery', '0000', 'Not scored', 'Not scored', 'Not scored']);
});
