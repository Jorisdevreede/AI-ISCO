// The group overview, as pure functions: no DOM, no globals, no fetching.
//
// Everything groups.html shows is derived here so `node --test` can check it:
// which group the URL asks for, its trail and children, the rows behind each of
// the four views, the summaries a screen reader is given, the colours the
// treemap paints with, and the two-group comparison.

import { formatCount, formatPercent, formatScore, formatShare, isScored } from '../format.js';
import { QUADRANT_NAMES, THRESHOLD } from '../quadrant.js';
import { QUADRANT_ORDER, occupationsInGroup, sortOccupations } from '../groupstats.js';
import { fold } from '../search.js';
import { buildHash, parseHash } from '../urlstate.js';

/** The four views, in tab order. */
export const VIEWS = ['ranked', 'scatter', 'treemap', 'table'];

export const VIEW_LABELS = {
  ranked: 'Ranked', scatter: 'Scatter', treemap: 'Treemap', table: 'Table',
};

/** Below this viewport width the ranked list is the default view. */
export const NARROW_WIDTH = 700;

/** The group key an empty hash means. */
export const ROOT_KEY = 'all';

/** How the treemap may be coloured. AI exposure is deliberately not offered. */
export const COLOUR_MODES = [
  { key: 'quadrant', label: 'Quadrant' },
  { key: 'automation', label: 'Automation' },
  { key: 'amplification', label: 'Amplification' },
];

const LEVEL_LABELS = {
  all: 'All occupations',
  major: 'Major group',
  sub: 'Sub-major group',
  minor: 'Minor group',
  unit: 'Unit group',
};

const UNSCORED_RGB = [138, 138, 152];

const QUADRANT_RGB = {
  TRANSFORM: [232, 185, 59],
  SHRINK: [239, 106, 94],
  EVOLVE: [85, 192, 119],
  STABLE: [106, 169, 232],
};

// Low-to-high ramps for the two score colourings. Both start on the page's own
// slate so a low score never reads as "nothing here".
const RAMPS = {
  automation: [[44, 62, 92], [232, 185, 59]],
  amplification: [[44, 62, 92], [85, 192, 119]],
};

/* --- URL state ----------------------------------------------------------- */

/**
 * The view a visitor gets when the URL does not name one.
 * @param {number} width viewport width in CSS pixels
 * @returns {'ranked'|'scatter'}
 */
export function defaultView(width) {
  return Number(width) < NARROW_WIDTH ? 'ranked' : 'scatter';
}

/**
 * Read the page state out of a location hash.
 * @param {string} hash
 * @param {number} width viewport width, for the default view
 * @returns {{key: string, view: string, compare: {a: string, b: string}|null}}
 */
export function readState(hash, width) {
  const { params } = parseHash(hash);
  const named = VIEWS.includes(params.view) ? params.view : null;
  return {
    key: params.g || ROOT_KEY,
    view: named || defaultView(width),
    compare: params.a && params.b ? { a: params.a, b: params.b } : null,
  };
}

/** The href of a two-group comparison. urlstate.js has no helper for this one. */
export function compareHref(a, b) {
  return `groups.html${buildHash(null, { a, b })}`;
}

/* --- the group itself ----------------------------------------------------- */

/**
 * The group a key names, or null when nothing has that key.
 * @param {Object} groups groups.json
 * @param {string} key
 * @returns {Object|null}
 */
export function resolveGroup(groups, key) {
  return (groups && groups[key]) || null;
}

/** "Major group 2 · ISCO-08 · 869 jobs" — level, code and size in one line. */
export function groupSubtitle(group) {
  const level = LEVEL_LABELS[group.level] || 'Group';
  const code = group.code ? `${level} ${group.code} · ISCO-08` : level;
  return `${code} · ${formatCount(group.n)} jobs`;
}

/**
 * The trail from "all" down to this group, this group last.
 * @returns {Array<{key: string, label: string}>}
 */
export function breadcrumb(groups, key) {
  const trail = [];
  let current = key;
  while (current && groups[current] && trail.length < 8) {
    trail.unshift({ key: current, label: groups[current].label });
    current = groups[current].parent;
  }
  return trail;
}

/**
 * The groups one level down, as links.
 * @returns {Array<{key: string, label: string, n: number}>}
 */
export function childGroups(groups, key) {
  const group = resolveGroup(groups, key);
  return ((group && group.children) || [])
    .filter((child) => groups[child])
    .map((child) => ({ key: child, label: groups[child].label, n: groups[child].n }));
}

/** What the picker says when a hash names a group we do not have. */
export function unknownGroupMessage(key) {
  return `We don't have a group called “${key}”.`;
}

/**
 * Group labels matching a typed query, best first.
 * @param {Object} groups groups.json
 * @param {string} query
 * @param {number} [limit=10]
 * @returns {Array<{key: string, label: string, n: number, level: string}>}
 */
export function matchGroups(groups, query, limit = 10) {
  const needle = fold(query || '');
  if (!needle) return [];
  const hits = [];
  for (const [key, group] of Object.entries(groups || {})) {
    const rank = labelRank(fold(group.label), needle);
    if (rank >= 0) hits.push({ key, label: group.label, n: group.n, level: group.level, rank });
  }
  hits.sort((a, b) => a.rank - b.rank || b.n - a.n || a.label.localeCompare(b.label));
  return hits.slice(0, limit).map(({ rank, ...hit }) => hit);
}

function labelRank(label, needle) {
  if (label === needle) return 0;
  if (label.startsWith(needle)) return 1;
  return label.includes(needle) ? 2 : -1;
}

/* --- how this group splits ------------------------------------------------ */

/**
 * The four quadrant shares, always in the same order, each carrying its own
 * label, count and percentage so nothing is conveyed by colour alone.
 * @param {{q: Object}} group a groups.json entry
 * @returns {Array<{code, label, count, share, percent, text}>}
 */
export function quadrantBars(group) {
  const counts = (group && group.q) || {};
  const total = QUADRANT_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0);
  return QUADRANT_ORDER.map((code) => {
    const count = counts[code] || 0;
    const share = total ? count / total : 0;
    return {
      code,
      label: QUADRANT_NAMES[code],
      count,
      share,
      percent: formatPercent(share),
      text: `${formatCount(count)} of ${formatCount(total)} jobs`,
    };
  });
}

/** "36% Transform, 28% Stable, 18% Evolve, 17% Shrink." */
export function mixSentence(group) {
  return `${quadrantBars(group).map((bar) => `${bar.percent} ${bar.label}`).join(', ')}.`;
}

/**
 * The one permanent caveat beside the split, built from stats.json.
 * Says only what the published data shows: how many jobs sit near a cut-off.
 */
export function nearLineCaveat(stats) {
  const near = stats && stats.near_line;
  if (!near || !isScored(near.count)) {
    return 'Jobs sitting near a cut-off can fall either side of it, so read this split '
      + 'as a band, not a count.';
  }
  return `${formatShare(near.count, stats.occupations, 'jobs')} sit within 0.5 of a `
    + `cut-off, and a small change in the scores moves those into another box: read this `
    + 'split as a band, not a count.';
}

/* --- ranked view ---------------------------------------------------------- */

/** Where a 1-10 score sits on the shared axis, as a percentage. */
export function axisPercent(score) {
  if (!isScored(score)) return null;
  return ((Math.min(10, Math.max(1, score)) - 1) / 9) * 100;
}

function dot(value) {
  return { value, text: formatScore(value), percent: axisPercent(value) };
}

function toRankedRow(row) {
  return {
    slug: row.s,
    title: row.t,
    code: row.c,
    quadrant: row.q,
    quadrantLabel: QUADRANT_NAMES[row.q] || 'Not scored',
    automation: dot(row.a),
    amplification: dot(row.m),
  };
}

/**
 * One row per job for the dot plot, sorted.
 * @param {Array<Object>} rows search_index rows
 * @param {string} sortKey a key of groupstats SORTS
 * @param {boolean} [descending]
 */
export function rankedRows(rows, sortKey, descending) {
  return sortOccupations(rows, sortKey, descending).map(toRankedRow);
}

/* --- scatter view --------------------------------------------------------- */

/**
 * Where a pair of scores lands in a plotting box.
 * @param {number} automation x
 * @param {number} amplification y, drawn upwards
 * @param {{width: number, height: number, pad: number}} box
 * @returns {{x: number, y: number}}
 */
export function plotPoint(automation, amplification, box) {
  const inner = { width: box.width - 2 * box.pad, height: box.height - 2 * box.pad };
  return {
    x: box.pad + (axisPercent(automation) / 100) * inner.width,
    y: box.height - box.pad - (axisPercent(amplification) / 100) * inner.height,
  };
}

/**
 * One dot per scored job.
 * @returns {Array<{slug, title, quadrant, x, y, label}>}
 */
export function scatterPoints(rows, box) {
  return (rows || [])
    .filter((row) => isScored(row.a) && isScored(row.m))
    .map((row) => ({
      slug: row.s,
      title: row.t,
      quadrant: row.q,
      ...plotPoint(row.a, row.m, box),
      label: pointLabel(row),
    }));
}

function pointLabel(row) {
  return `${row.t}, automation ${formatScore(row.a)}, amplification ${formatScore(row.m)}, `
    + `${QUADRANT_NAMES[row.q] || 'Not scored'}.`;
}

/** Where the two cut-off lines cross, in the same box. */
export function thresholdPoint(box) {
  return plotPoint(THRESHOLD, THRESHOLD, box);
}

/* --- table view ----------------------------------------------------------- */

function cellValue(row, key) {
  const value = row[key];
  if (key === 'q') return QUADRANT_NAMES[value] || '';
  return value;
}

function compareCells(a, b, key) {
  const left = cellValue(a, key);
  const right = cellValue(b, key);
  if (isScored(left) && isScored(right)) return left - right;
  return String(left ?? '').localeCompare(String(right ?? ''));
}

function isPresent(value) {
  if (value === null || value === undefined || value === '') return false;
  return typeof value === 'number' ? Number.isFinite(value) : true;
}

/** Rows with nothing in this column go last, whichever way the column runs. */
function missingLast(a, b, key) {
  const left = isPresent(cellValue(a, key));
  if (left === isPresent(cellValue(b, key))) return 0;
  return left ? -1 : 1;
}

/**
 * Sort search_index rows by any table column. Never mutates the input; ties
 * break by title, so the same click always gives the same order.
 */
export function sortByColumn(rows, columnKey, descending) {
  const direction = descending ? -1 : 1;
  return [...(rows || [])].sort((a, b) => (
    missingLast(a, b, columnKey)
    || compareCells(a, b, columnKey) * direction
    || String(a.t ?? '').localeCompare(String(b.t ?? ''))
  ));
}

/* --- treemap -------------------------------------------------------------- */

function dominantQuadrant(counts) {
  const tally = counts || {};
  return QUADRANT_ORDER.reduce(
    (best, code) => ((tally[code] || 0) > (tally[best] || 0) ? code : best),
    QUADRANT_ORDER[0],
  );
}

function groupTile(key, group) {
  const quadrant = dominantQuadrant(group.q);
  return {
    id: key,
    kind: 'group',
    label: group.label,
    value: group.n,
    automation: group.auto && group.auto.mean,
    amplification: group.amp && group.amp.mean,
    quadrant,
    detail: `${formatCount(group.n)} jobs · mostly ${QUADRANT_NAMES[quadrant]}`,
  };
}

function jobTile(row) {
  return {
    id: row.s,
    kind: 'job',
    label: row.t,
    value: 1,
    automation: row.a,
    amplification: row.m,
    quadrant: row.q,
    detail: `automation ${formatScore(row.a)} · ${QUADRANT_NAMES[row.q] || 'Not scored'}`,
  };
}

/**
 * The tiles of the treemap: the child groups when there are any, otherwise the
 * jobs themselves. Sized by the number of jobs, so search_index is enough and
 * the page never touches the big data files.
 */
export function treemapTiles(groups, rows, key) {
  const children = childGroups(groups, key);
  if (children.length) return children.map((child) => groupTile(child.key, groups[child.key]));
  return occupationsInGroup(rows, key).map(jobTile);
}

/** What the live region says about the tile that has the focus ring. */
export function describeTile(tile) {
  if (!tile) return '';
  if (tile.kind === 'group') {
    return `${tile.label}, ${formatCount(tile.value)} jobs, average automation `
      + `${formatScore(tile.automation)}. Press Enter to drill down.`;
  }
  return `${tile.label}, automation ${formatScore(tile.automation)}, amplification `
    + `${formatScore(tile.amplification)}, ${QUADRANT_NAMES[tile.quadrant] || 'Not scored'}. `
    + 'Press Enter to open the job page.';
}

/* --- tile colour, and text that can be read on it ------------------------- */

function channelLuminance(value) {
  const channel = value / 255;
  return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

/** WCAG relative luminance of an [r, g, b] triple. */
export function relativeLuminance(rgb) {
  const [r, g, b] = rgb.map(channelLuminance);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two [r, g, b] triples, 1 to 21. */
export function contrastRatio(one, other) {
  const [light, dark] = [relativeLuminance(one), relativeLuminance(other)].sort((a, b) => b - a);
  return (light + 0.05) / (dark + 0.05);
}

/**
 * Black or white, whichever reads better on this fill. One of the two always
 * clears 4.5:1, which is how the audit's 2.7:1 tile text gets fixed.
 */
export function textColourFor(rgb) {
  return contrastRatio(rgb, [255, 255, 255]) >= contrastRatio(rgb, [0, 0, 0])
    ? '#ffffff'
    : '#000000';
}

function rampColour(ramp, score) {
  if (!isScored(score)) return UNSCORED_RGB;
  const position = Math.min(1, Math.max(0, (score - 1) / 9));
  const [low, high] = ramp;
  return low.map((channel, index) => Math.round(channel + (high[index] - channel) * position));
}

/**
 * The fill and text colour of one tile.
 * @param {Object} tile from treemapTiles
 * @param {'quadrant'|'automation'|'amplification'} mode
 * @returns {{rgb: number[], fill: string, text: string}}
 */
export function tileColour(tile, mode) {
  const rgb = mode === 'quadrant'
    ? QUADRANT_RGB[tile.quadrant] || UNSCORED_RGB
    : rampColour(RAMPS[mode] || RAMPS.automation, tile[mode]);
  return { rgb, fill: `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`, text: textColourFor(rgb) };
}

/* --- summaries for the visuals -------------------------------------------- */

function scoreRange(rows, key) {
  const values = (rows || []).map((row) => row[key]).filter(isScored);
  if (!values.length) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

/** "Automation runs 1.9 to 8.6, amplification 3.0 to 10.0. Cut-off at 6." */
export function rangeSentence(rows) {
  const automation = scoreRange(rows, 'a');
  const amplification = scoreRange(rows, 'm');
  if (!automation || !amplification) return 'No job in this group has scores.';
  return `Automation runs ${formatScore(automation.min)} to ${formatScore(automation.max)}, `
    + `amplification ${formatScore(amplification.min)} to ${formatScore(amplification.max)}. `
    + `The cut-off on both axes is ${THRESHOLD}.`;
}

const SUMMARY_OPENERS = {
  ranked: (group, rows) => `Dot plot of ${formatCount(rows.length)} jobs in ${group.label}, `
    + 'automation and amplification on a shared 1 to 10 axis.',
  scatter: (group, rows) => `Scatter plot of ${formatCount(rows.length)} jobs in `
    + `${group.label}, automation across and amplification up.`,
  treemap: (group, rows) => `Treemap of ${group.label}, ${formatCount(rows.length)} jobs, `
    + 'each tile sized by the number of jobs it holds.',
  table: (group, rows) => `Table of ${formatCount(rows.length)} jobs in ${group.label}.`,
};

/**
 * The aria-label every visual carries.
 * @param {'ranked'|'scatter'|'treemap'|'table'} kind
 * @param {Object} group a groups.json entry
 * @param {Array<Object>} rows the group's occupations
 */
export function visualSummary(kind, group, rows) {
  const opener = (SUMMARY_OPENERS[kind] || SUMMARY_OPENERS.table)(group, rows || []);
  return `${opener} ${mixSentence(group)} ${rangeSentence(rows)}`;
}

/* --- most and least exposed, and the skills behind them ------------------- */

/** slug -> search_index row, for resolving the slug lists in groups.json. */
export function indexBySlug(rows) {
  return new Map((rows || []).map((row) => [row.s, row]));
}

function resolveSlugs(slugs, bySlug, limit) {
  return (slugs || [])
    .map((slug) => bySlug.get(slug))
    .filter(Boolean)
    .slice(0, limit)
    .map((row) => ({
      slug: row.s, title: row.t, quadrant: row.q, automation: row.a, amplification: row.m,
    }));
}

/**
 * The five most and five least automation-exposed jobs of a group, with their
 * titles resolved through search_index.
 */
export function exposureLists(group, bySlug, limit = 5) {
  return {
    top: resolveSlugs(group && group.top, bySlug, limit),
    bottom: resolveSlugs(group && group.bottom, bySlug, limit),
  };
}

function skillEntries(list, titles) {
  return (list || []).map((entry) => ({
    id: entry.id,
    n: entry.n,
    title: (titles && titles.get(entry.id)) || null,
    count: `in ${formatCount(entry.n)} jobs`,
  }));
}

/**
 * The skills driving a group, titles resolved through skill_index when it has
 * arrived; `title` stays null until then so the page can show a placeholder.
 */
export function drivingSkills(group, titles) {
  const skills = (group && group.skills) || {};
  return {
    automation: skillEntries(skills.auto, titles),
    amplification: skillEntries(skills.amp, titles),
  };
}

/* --- comparing two groups ------------------------------------------------- */

function comparisonSide(groups, key) {
  const group = resolveGroup(groups, key);
  if (!group) return null;
  return {
    key,
    label: group.label,
    subtitle: groupSubtitle(group),
    bars: quadrantBars(group),
    medians: {
      automation: group.auto && group.auto.p50,
      amplification: group.amp && group.amp.p50,
    },
  };
}

function differenceRow(a, b, position) {
  const left = a.bars[position];
  const right = b.bars[position];
  const delta = left.share - right.share;
  const points = Math.round(Math.abs(delta) * 100);
  return {
    code: left.code,
    label: left.label,
    a: left,
    b: right,
    delta,
    text: `${left.percent} in ${a.label}, ${right.percent} in ${b.label}`,
    deltaText: points === 0
      ? 'the same share'
      : `${points} percentage point${points === 1 ? '' : 's'} `
        + `${delta > 0 ? 'more' : 'fewer'} in ${a.label}`,
  };
}

/**
 * Two groups side by side: each one's quadrant split and medians, plus the
 * difference per quadrant for the diverging bars.
 * @returns {{a: Object, b: Object, differences: Array<Object>}|null}
 */
export function compareGroups(groups, aKey, bKey) {
  const a = comparisonSide(groups, aKey);
  const b = comparisonSide(groups, bKey);
  if (!a || !b) return null;
  const differences = QUADRANT_ORDER.map((code, position) => differenceRow(a, b, position));
  return { a, b, differences };
}
