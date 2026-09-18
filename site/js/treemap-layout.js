// A squarified treemap layout. Pure: no DOM, no globals, no dependencies.
//
// Written from the published algorithm rather than from any implementation of
// it: Mark Bruls, Kees Huizing and Jarke J. van Wijk, "Squarified Treemaps",
// in Data Visualization 2000 (Proceedings of the Joint Eurographics and IEEE
// TCVG Symposium on Visualization), Springer, pp. 33-42.
//
// The recipe in words. Give every item an area proportional to its value.
// Walk the items in the order given, adding each one to the row being built
// while that keeps the row's worst aspect ratio from getting worse. When it
// would get worse, lay the finished row along the shorter side of the free
// rectangle, cut that strip off, and carry on with what is left. For a row of
// areas laid along a side of length `side`, the worst aspect ratio is
//
//     max( side^2 * max(areas) / sum(areas)^2 ,
//          sum(areas)^2 / (side^2 * min(areas)) )
//
// which is the paper's `worst` function written with areas instead of the
// normalised values it uses.

/** @typedef {{x: number, y: number, width: number, height: number}} Rect */
/** @typedef {{item: *, index: number, value: number} & Rect} Tile */

const EPSILON = 1e-12;

/** A drawable value, or 0 for anything missing, zero, negative or not a number. */
function valueOf(item) {
  const value = typeof item === 'number' ? item : Number(item && item.value);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

/** The items worth drawing, each keeping the index it had in the input. */
function scaleEntries(items, area) {
  const kept = [];
  let total = 0;
  (items || []).forEach((item, index) => {
    const value = valueOf(item);
    if (value <= 0) return;
    kept.push({ item, index, value, area: 0 });
    total += value;
  });
  for (const entry of kept) entry.area = (entry.value / total) * area;
  return kept;
}

function normalise(rect) {
  const { x = 0, y = 0, width = 0, height = 0 } = rect || {};
  return { x, y, width, height };
}

function newRow() {
  return { entries: [], count: 0, sum: 0, min: Infinity, max: 0 };
}

function statsWith(row, area) {
  return {
    count: row.count + 1,
    sum: row.sum + area,
    min: Math.min(row.min, area),
    max: Math.max(row.max, area),
  };
}

function addToRow(row, entry) {
  Object.assign(row, statsWith(row, entry.area));
  row.entries.push(entry);
}

/** The worst (largest) aspect ratio in a row of areas laid along `side`. */
function worstRatio(stats, side) {
  if (!stats.count || stats.sum <= EPSILON || side <= EPSILON) return Infinity;
  const sum2 = stats.sum * stats.sum;
  const side2 = side * side;
  return Math.max((side2 * stats.max) / sum2, sum2 / (side2 * stats.min));
}

function tileFor(entry, free, geometry) {
  const { vertical, offset, span, thickness } = geometry;
  return {
    item: entry.item,
    index: entry.index,
    value: entry.value,
    x: vertical ? free.x : free.x + offset,
    y: vertical ? free.y + offset : free.y,
    width: vertical ? thickness : span,
    height: vertical ? span : thickness,
  };
}

function shrink(free, thickness, vertical) {
  if (vertical) {
    return { x: free.x + thickness, y: free.y, width: free.width - thickness, height: free.height };
  }
  return { x: free.x, y: free.y + thickness, width: free.width, height: free.height - thickness };
}

/** How thick the strip holding `row` is. The last row takes all that is left. */
function rowThickness(row, free, vertical, last) {
  const across = vertical ? free.width : free.height;
  if (last) return across;
  const extent = vertical ? free.height : free.width;
  return Math.min(row.sum / extent, across);
}

/**
 * Lay one row along the shorter side of `free`, appending its tiles to `out`.
 * The last tile is snapped to the end of the strip so rows tile exactly.
 * @returns {Rect} what is left of `free`
 */
function placeRow(row, free, out, last) {
  const vertical = free.width >= free.height; // the row becomes a column
  const extent = vertical ? free.height : free.width;
  const thickness = rowThickness(row, free, vertical, last);
  let offset = 0;
  row.entries.forEach((entry, position) => {
    const isLast = position === row.entries.length - 1;
    const span = isLast ? extent - offset : entry.area / thickness;
    out.push(tileFor(entry, free, { vertical, offset, span, thickness }));
    offset += span;
  });
  return shrink(free, thickness, vertical);
}

/**
 * Squarified treemap rectangles for a list of items.
 *
 * Areas are proportional to the values, the tiles tile `rect` exactly with no
 * overlap and no gap, and the output keeps the input order. Items with a zero,
 * negative, missing or non-numeric value get no tile at all; `index` says where
 * each tile's item sat in the input.
 *
 * @param {Array<number|{value: number}>} items in the order they should be laid
 * @param {Rect} rect the container, `x` and `y` default to 0
 * @returns {Tile[]}
 */
export function layoutTreemap(items, rect) {
  const box = normalise(rect);
  if (box.width <= 0 || box.height <= 0) return [];
  const entries = scaleEntries(items, box.width * box.height);
  if (!entries.length) return [];

  const out = [];
  let free = box;
  let row = newRow();
  for (const entry of entries) {
    const side = Math.min(free.width, free.height);
    if (row.count && worstRatio(statsWith(row, entry.area), side) > worstRatio(row, side)) {
      free = placeRow(row, free, out, false);
      row = newRow();
    }
    addToRow(row, entry);
  }
  placeRow(row, free, out, true);
  return out;
}

/** The worst aspect ratio in a laid-out set of tiles. Handy for tests. */
export function worstAspectRatio(tiles) {
  return (tiles || []).reduce((worst, tile) => {
    if (tile.width <= 0 || tile.height <= 0) return Infinity;
    return Math.max(worst, tile.width / tile.height, tile.height / tile.width);
  }, 0);
}
