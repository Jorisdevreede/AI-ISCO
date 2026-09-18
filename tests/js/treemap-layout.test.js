import test from 'node:test';
import assert from 'node:assert/strict';

import { layoutTreemap, worstAspectRatio } from '../../site/js/treemap-layout.js';

const RECT = { x: 0, y: 0, width: 600, height: 400 };
const TOLERANCE = 1e-6;

/** The example from the Squarified Treemaps paper, section 3. */
const PAPER = [6, 6, 4, 3, 2, 2, 1];

const area = (tile) => tile.width * tile.height;
const sumAreas = (tiles) => tiles.reduce((total, tile) => total + area(tile), 0);

function overlapArea(a, b) {
  const wide = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const tall = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  return Math.max(0, wide) * Math.max(0, tall);
}

function largestOverlap(tiles) {
  let worst = 0;
  for (let i = 0; i < tiles.length; i += 1) {
    for (let j = i + 1; j < tiles.length; j += 1) {
      worst = Math.max(worst, overlapArea(tiles[i], tiles[j]));
    }
  }
  return worst;
}

test('every area is proportional to its value', () => {
  const tiles = layoutTreemap(PAPER, RECT);
  const unit = area(tiles[0]) / PAPER[0];
  for (const tile of tiles) {
    assert.ok(Math.abs(area(tile) / tile.value - unit) < TOLERANCE * unit,
      `tile ${tile.index} has area ${area(tile)} for value ${tile.value}`);
  }
});

test('the tiles fill the container with no gap and no overlap', () => {
  const tiles = layoutTreemap(PAPER, { x: 12, y: 20, width: 600, height: 400 });
  assert.ok(Math.abs(sumAreas(tiles) - 600 * 400) < TOLERANCE);
  assert.ok(largestOverlap(tiles) < TOLERANCE);
  for (const tile of tiles) {
    assert.ok(tile.x >= 12 - TOLERANCE && tile.y >= 20 - TOLERANCE);
    assert.ok(tile.x + tile.width <= 612 + TOLERANCE);
    assert.ok(tile.y + tile.height <= 420 + TOLERANCE);
    assert.ok(tile.width > 0 && tile.height > 0);
  }
});

test('a lopsided container is still tiled exactly', () => {
  const tiles = layoutTreemap([40, 1, 1, 1, 1, 1], { width: 1200, height: 90 });
  assert.equal(tiles.length, 6);
  assert.ok(Math.abs(sumAreas(tiles) - 1200 * 90) < TOLERANCE);
  assert.ok(largestOverlap(tiles) < TOLERANCE);
});

test('aspect ratios stay bounded for a typical input', () => {
  assert.ok(worstAspectRatio(layoutTreemap(PAPER, RECT)) < 3,
    'the paper example should stay well away from slivers');
  const many = Array.from({ length: 120 }, (_, i) => 120 - i);
  assert.ok(worstAspectRatio(layoutTreemap(many, RECT)) < 6);
});

test('squarifying beats a naive strip layout on the paper example', () => {
  const tiles = layoutTreemap(PAPER, RECT);
  const stripes = PAPER.map((value, index) => ({
    width: RECT.width * (value / 24), height: RECT.height, index,
  }));
  assert.ok(worstAspectRatio(tiles) < worstAspectRatio(stripes));
});

test('zero, negative, missing and non-numeric values get no tile', () => {
  const tiles = layoutTreemap(
    [{ value: 4 }, { value: 0 }, { value: -3 }, { value: null }, { value: 'x' }, { value: 4 }],
    RECT,
  );
  assert.deepEqual(tiles.map((tile) => tile.index), [0, 5]);
  assert.ok(Math.abs(sumAreas(tiles) - 600 * 400) < TOLERANCE);
});

test('empty and impossible inputs give an empty layout', () => {
  assert.deepEqual(layoutTreemap([], RECT), []);
  assert.deepEqual(layoutTreemap(null, RECT), []);
  assert.deepEqual(layoutTreemap([0, -1], RECT), []);
  assert.deepEqual(layoutTreemap([1, 2], { width: 0, height: 400 }), []);
  assert.deepEqual(layoutTreemap([1, 2], { width: 600, height: -1 }), []);
  assert.deepEqual(layoutTreemap([1, 2], undefined), []);
});

test('one item fills the whole container', () => {
  const [tile] = layoutTreemap([{ value: 9 }], RECT);
  assert.deepEqual(
    { x: tile.x, y: tile.y, width: tile.width, height: tile.height },
    { x: 0, y: 0, width: 600, height: 400 },
  );
});

test('the order of the input is kept, whatever the values do', () => {
  const items = [{ value: 1 }, { value: 9 }, { value: 1 }, { value: 9 }];
  const tiles = layoutTreemap(items, RECT);
  assert.deepEqual(tiles.map((tile) => tile.index), [0, 1, 2, 3]);
  assert.deepEqual(tiles.map((tile) => tile.item), items);
  assert.deepEqual(layoutTreemap(items, RECT), tiles); // same input, same output
});

test('equal values keep the order they came in', () => {
  const items = [{ id: 'a', value: 5 }, { id: 'b', value: 5 }, { id: 'c', value: 5 }];
  assert.deepEqual(
    layoutTreemap(items, RECT).map((tile) => tile.item.id),
    ['a', 'b', 'c'],
  );
});
