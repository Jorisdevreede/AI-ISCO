// Everything tree.html decides before it touches the DOM. Pure: no DOM, no
// globals, unit-tested with `node --test tests/js/tree-model.test.js`.
//
// It works on the shapes the published index files already have:
//   groups.json        {"<level>:<code>": {label, level, code, n, q, parent, children}}
//   search_index.json  [{t, s, c, mg, a, m, q, alt[]}]
//   portfolio_data.json {skills: {<id>: {t, a, m, r}}, occupations: [{s, se[], so[]}]}
//
// The tree is never flattened in full: `visibleRows` walks only the branches the
// visitor has opened, so a page showing ten major groups builds ten rows, not
// three thousand.

import { NOT_SCORED, formatCount, formatPercent, formatScore, formatShare, isScored } from '../format.js';
import { QUADRANT_ORDER } from '../groupstats.js';
import { QUADRANT_NAMES, quadrantOf } from '../quadrant.js';
import { fold, rankOccupations } from '../search.js';
import { buildHash, parseHash } from '../urlstate.js';

/** The group key the tree starts from. Its children are the ten major groups. */
export const ROOT_KEY = 'all';

/** How many matching jobs the filter draws at once, so the DOM stays small. */
export const FILTER_LIMIT = 300;

/** How long two keystrokes stay one type-ahead word, in milliseconds. */
export const TYPE_AHEAD_MS = 800;

const LEVEL_LABELS = {
  all: 'All occupations',
  major: 'Major group',
  sub: 'Sub-major group',
  minor: 'Minor group',
  unit: 'Unit group',
};

/** ISCO codes nest by digit: 2512 sits in 251, in 25, in 2. */
const LEVEL_WIDTHS = [['unit', 4], ['minor', 3], ['sub', 2], ['major', 1]];

/* --- the model ------------------------------------------------------------ */

function byTitle(a, b) {
  return fold(a.t).localeCompare(fold(b.t));
}

function bucketJobs(index) {
  const byUnit = new Map();
  for (const row of index || []) {
    const key = `unit:${String(row.c || '')}`;
    if (!byUnit.has(key)) byUnit.set(key, []);
    byUnit.get(key).push(row);
  }
  for (const rows of byUnit.values()) rows.sort(byTitle);
  return byUnit;
}

/**
 * Index the two published files once, so the tree can be walked cheaply.
 * @param {Object} groups groups.json
 * @param {Array<Object>} index search_index.json
 * @returns {{groups: Object, index: Array, jobsByUnit: Map, bySlug: Map}}
 */
export function buildModel(groups, index) {
  const rows = index || [];
  return {
    groups: groups || {},
    index: rows,
    jobsByUnit: bucketJobs(rows),
    bySlug: new Map(rows.map((row) => [row.s, row])),
  };
}

/**
 * The group keys an ISCO code belongs to, narrowest first.
 * @param {string} code
 * @returns {Array<string>}
 */
export function unitChain(code) {
  const digits = String(code || '').replace(/\D/g, '');
  return LEVEL_WIDTHS
    .filter(([, width]) => digits.length >= width)
    .map(([level, width]) => `${level}:${digits.slice(0, width)}`);
}

/* --- nodes ---------------------------------------------------------------- */

function groupNode(model, key) {
  const group = model.groups[key];
  return {
    id: key,
    kind: 'group',
    key,
    label: group.label,
    level: group.level,
    code: group.code,
    n: group.n,
    counts: group.q || {},
  };
}

function jobNode(row) {
  return {
    id: `job:${row.s}`,
    kind: 'job',
    slug: row.s,
    label: row.t,
    code: row.c,
    a: row.a,
    m: row.m,
    q: row.q,
  };
}

/**
 * The nodes one level under a group: its child groups, or — for a unit group —
 * the jobs themselves.
 *
 * @param {Object} model from buildModel
 * @param {string} key group key
 * @param {{keys: Set<string>, slugs: Set<string>}|null} [filter]
 * @returns {Array<Object>} node descriptors
 */
export function childrenOf(model, key, filter = null) {
  const group = model.groups[key];
  if (!group) return [];
  const children = (group.children || [])
    .filter((child) => model.groups[child] && (!filter || filter.keys.has(child)));
  if (children.length) return children.map((child) => groupNode(model, child));
  return (model.jobsByUnit.get(key) || [])
    .filter((row) => !filter || filter.slugs.has(row.s))
    .map(jobNode);
}

/* --- which rows are on screen --------------------------------------------- */

function isExpanded(view, key) {
  if (view.filter) return view.filter.keys.has(key);
  return Boolean(view.expanded && view.expanded.has(key));
}

function pushLevel(walk, key, level) {
  const nodes = childrenOf(walk.model, key, walk.view.filter);
  for (const [position, node] of nodes.entries()) {
    const row = {
      ...node,
      level,
      posinset: position + 1,
      setsize: nodes.length,
      expanded: false,
      selected: node.id === walk.view.selectedId,
    };
    walk.rows.push(row);
    if (node.kind !== 'group' || !isExpanded(walk.view, node.key)) continue;
    const before = walk.rows.length;
    pushLevel(walk, node.key, level + 1);
    row.expanded = walk.rows.length > before;
  }
}

/**
 * Every row currently on screen, depth first, with the ARIA numbers each one
 * needs. Collapsed branches are never walked, so this stays cheap.
 *
 * @param {Object} model from buildModel
 * @param {{expanded: Set<string>, filter: Object|null, selectedId: string|null}} view
 * @returns {Array<Object>} rows carrying level, posinset, setsize, expanded
 */
export function visibleRows(model, view) {
  const walk = { model, view, rows: [] };
  pushLevel(walk, ROOT_KEY, 1);
  return walk.rows;
}

/** Where a row id sits in a row list, or -1. */
export function rowIndexOf(rows, id) {
  return (rows || []).findIndex((row) => row.id === id);
}

/* --- filtering ------------------------------------------------------------ */

/**
 * The jobs whose title or alternative label matches, and every group above
 * them. Capped so a one-letter query cannot ask for three thousand rows.
 *
 * @param {Object} model from buildModel
 * @param {string} query raw input; empty returns null (no filter)
 * @param {number} [limit=FILTER_LIMIT]
 * @returns {{slugs: Set, keys: Set, total: number, shown: number, capped: boolean}|null}
 */
export function filterMatches(model, query, limit = FILTER_LIMIT) {
  if (!fold(query)) return null;
  const ranked = rankOccupations(model.index, query, Infinity);
  const slugs = new Set();
  const keys = new Set();
  for (const hit of ranked.slice(0, limit)) {
    slugs.add(hit.row.s);
    for (const key of unitChain(hit.row.c)) if (model.groups[key]) keys.add(key);
  }
  return {
    slugs, keys, total: ranked.length, shown: slugs.size, capped: ranked.length > slugs.size,
  };
}

/** What the live region says while the filter is on. */
export function matchMessage(result) {
  if (!result) return '';
  if (!result.total) return 'No jobs match. Clear the filter to see the whole tree.';
  const jobs = result.total === 1 ? '1 job matches' : `${formatCount(result.total)} jobs match`;
  if (!result.capped) return `${jobs}.`;
  return `${jobs}. Showing the first ${formatCount(result.shown)} — type more to narrow it.`;
}

/* --- URL state ------------------------------------------------------------ */

/** The row id a selection points at, so rows and selections compare directly. */
export function selectionId(selection) {
  if (!selection) return null;
  return selection.kind === 'job' ? `job:${selection.id}` : selection.id;
}

/** The selection a row id means. Inverse of selectionId. */
export function selectionOf(id) {
  if (!id) return null;
  return id.startsWith('job:')
    ? { kind: 'job', id: id.slice(4) }
    : { kind: 'group', id };
}

/**
 * Read the page state out of a location hash.
 *   tree.html#job=<slug>         a job is selected
 *   tree.html#g=<level>:<code>   a group is selected and expanded
 *   ...&q=<text>                 the filter box carries text
 * @param {string} hash
 * @returns {{selection: {kind: string, id: string}|null, query: string}}
 */
export function readState(hash) {
  const { params } = parseHash(hash);
  const query = params.q || '';
  if (params.job) return { selection: { kind: 'job', id: params.job }, query };
  if (params.g) return { selection: { kind: 'group', id: params.g }, query };
  return { selection: null, query };
}

/** The hash a state writes. Inverse of readState. */
export function treeHash(selection, query) {
  const params = {};
  if (selection && selection.kind === 'job') params.job = selection.id;
  if (selection && selection.kind === 'group') params.g = selection.id;
  if (query) params.q = query;
  return buildHash(null, params);
}

/** The href of one state of this page. */
export function treeHref(selection, query) {
  return `tree.html${treeHash(selection, query)}`;
}

function chainOf(model, key) {
  const keys = [];
  let current = key;
  while (current && model.groups[current] && keys.length < 8) {
    keys.push(current);
    current = model.groups[current].parent;
  }
  return keys;
}

/**
 * The groups that have to be open for a selection to be on screen. A selected
 * group is included, so its own children show.
 * @param {Object} model
 * @param {{kind: string, id: string}|null} selection
 * @returns {Array<string>}
 */
export function ancestorKeys(model, selection) {
  if (!selection) return [];
  if (selection.kind !== 'job') return chainOf(model, selection.id);
  const row = model.bySlug.get(selection.id);
  return row ? chainOf(model, `unit:${String(row.c || '')}`) : [];
}

/* --- keyboard ------------------------------------------------------------- */

/**
 * Where Up, Down, Home and End land.
 * @returns {number} the new index, or -1 when the key does not move
 */
export function moveIndex(rows, from, key) {
  const last = (rows || []).length - 1;
  if (last < 0) return -1;
  if (key === 'Home') return 0;
  if (key === 'End') return last;
  if (key === 'ArrowDown') return Math.min(from + 1, last);
  if (key === 'ArrowUp') return Math.max(from - 1, 0);
  return -1;
}

function parentIndex(rows, from) {
  for (let at = from - 1; at >= 0; at -= 1) {
    if (rows[at].level < rows[from].level) return at;
  }
  return -1;
}

function rightAction(rows, from) {
  const row = rows[from];
  if (row.kind === 'group' && !row.expanded) return { action: 'expand', key: row.key };
  if (row.expanded && from < rows.length - 1) return { action: 'move', index: from + 1 };
  return null;
}

function leftAction(rows, from) {
  const row = rows[from];
  if (row.kind === 'group' && row.expanded) return { action: 'collapse', key: row.key };
  const parent = parentIndex(rows, from);
  return parent === -1 ? null : { action: 'move', index: parent };
}

/**
 * What Right and Left do: open a closed group, step into an open one, close an
 * open group, or step out to the parent.
 * @returns {{action: 'expand'|'collapse'|'move', key?: string, index?: number}|null}
 */
export function horizontalMove(rows, from, key) {
  const row = (rows || [])[from];
  if (!row) return null;
  if (key === 'ArrowRight') return rightAction(rows, from);
  if (key === 'ArrowLeft') return leftAction(rows, from);
  return null;
}

/**
 * The next row whose label starts with what was typed, wrapping around. A
 * single letter always steps on, so pressing "n" repeatedly cycles.
 * @returns {number} index, or -1 when nothing matches
 */
export function typeAheadIndex(rows, from, prefix) {
  const needle = fold(prefix);
  const total = (rows || []).length;
  if (!needle || !total) return -1;
  const offset = needle.length > 1 ? 0 : 1;
  for (let step = 0; step < total; step += 1) {
    const at = (((from + offset + step) % total) + total) % total;
    if (fold(rows[at].label).startsWith(needle)) return at;
  }
  return -1;
}

/* --- the quadrant mix ----------------------------------------------------- */

/**
 * The four quadrant shares of a group, always in the same order, each carrying
 * its own label, count and percentage so nothing is said by colour alone.
 * @param {Object} counts the `q` map of a groups.json entry
 * @returns {Array<{code, label, count, share, percent, text}>}
 */
export function mixSegments(counts) {
  const tally = counts || {};
  const total = QUADRANT_ORDER.reduce((sum, code) => sum + (tally[code] || 0), 0);
  return QUADRANT_ORDER.map((code) => {
    const count = tally[code] || 0;
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

/** "51% Evolve, 26% Stable, 17% Transform, 6% Shrink" — the bar said out loud. */
export function mixText(segments) {
  const shown = (segments || []).filter((segment) => segment.count > 0);
  if (!shown.length) return 'no scored jobs';
  return [...shown]
    .sort((a, b) => b.share - a.share)
    .map((segment) => `${segment.percent} ${segment.label}`)
    .join(', ');
}

/** "Unit group 2512 · ISCO-08 · 10 jobs" — level, code and size in one line. */
export function groupSubtitle(group) {
  const level = LEVEL_LABELS[group.level] || 'Group';
  const code = group.code ? `${level} ${group.code} · ISCO-08` : level;
  return `${code} · ${formatCount(group.n)} jobs`;
}

/**
 * The one permanent caveat beside a split, built from stats.json. Says only
 * what the published data shows: how many jobs sit near a cut-off.
 */
export function nearLineCaveat(stats) {
  const near = stats && stats.near_line;
  if (!near || !isScored(near.count)) {
    return 'Jobs sitting near a cut-off can fall either side of it, so read this split '
      + 'as a band, not a count.';
  }
  return `${formatShare(near.count, stats.occupations, 'jobs')} sit within 0.5 of a `
    + 'cut-off, and a small change in the scores moves those into another box: read this '
    + 'split as a band, not a count.';
}

/** "Transform, automation 6.4, amplification 8.9" — a leaf said out loud. */
export function jobSummary(node) {
  return `${QUADRANT_NAMES[node.q] || NOT_SCORED}, automation ${formatScore(node.a)}, `
    + `amplification ${formatScore(node.m)}`;
}

/* --- the skills of one job ------------------------------------------------ */

function resolveSkills(skills, ids, essential) {
  return (ids || [])
    .map((id) => [id, skills[id]])
    .filter(([, skill]) => Boolean(skill))
    .map(([id, skill]) => ({
      id,
      title: skill.t,
      auto: isScored(skill.a) ? skill.a : null,
      amp: isScored(skill.m) ? skill.m : null,
      essential,
      quadrant: quadrantOf(skill.a, skill.m),
    }));
}

/** The occupation a slug names inside portfolio_data.json, or null. */
export function findOccupation(portfolio, slug) {
  const rows = (portfolio && portfolio.occupations) || [];
  return rows.find((row) => row.s === slug) || null;
}

/**
 * Every skill of one occupation, essential first, unknown ids dropped.
 * @param {Object} portfolio parsed portfolio_data.json
 * @param {{se?: Array<string>, so?: Array<string>}} occupation
 * @returns {Array<{id, title, auto, amp, essential, quadrant}>}
 */
export function skillRows(portfolio, occupation) {
  const skills = (portfolio && portfolio.skills) || {};
  return [
    ...resolveSkills(skills, occupation && occupation.se, true),
    ...resolveSkills(skills, occupation && occupation.so, false),
  ];
}

/**
 * How the job's own skills split across the four boxes, on the same cut-off
 * the occupations use.
 * @param {Array<Object>} rows from skillRows
 * @returns {{total, essential, scored, unscored, bars: Array}}
 */
export function skillMix(rows) {
  const all = rows || [];
  const counts = {};
  for (const row of all) {
    if (row.quadrant) counts[row.quadrant] = (counts[row.quadrant] || 0) + 1;
  }
  const scored = QUADRANT_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0);
  const bars = mixSegments(counts).map((segment) => ({
    ...segment,
    text: `${formatCount(segment.count)} of ${formatCount(scored)} skills`,
  }));
  return {
    total: all.length,
    essential: all.filter((row) => row.essential).length,
    scored,
    unscored: all.length - scored,
    bars,
  };
}

/** "108 skills, 24 of them essential. 108 carry both scores." */
export function skillSummary(mix) {
  const essential = `${formatCount(mix.essential)} of them essential`;
  const head = `${formatCount(mix.total)} skill${mix.total === 1 ? '' : 's'}, ${essential}.`;
  if (!mix.unscored) return head;
  return `${head} ${formatCount(mix.unscored)} have no scores yet.`;
}

/* --- the skills table ----------------------------------------------------- */

/** The table's columns, in order. */
export const SKILL_COLUMNS = [
  { key: 'title', label: 'Skill', numeric: false },
  { key: 'essential', label: 'In this job', numeric: false },
  { key: 'auto', label: 'Automation risk', numeric: true },
  { key: 'amp', label: 'Amplification', numeric: true },
  { key: 'quadrant', label: 'Quadrant', numeric: false },
];

/** The three states of the Essential / Optional / All toggle. */
export const SKILL_FILTERS = [
  { key: 'essential', label: 'Essential' },
  { key: 'optional', label: 'Optional' },
  { key: 'all', label: 'All' },
];

const SKILL_VALUES = {
  title: (row) => fold(row.title),
  essential: (row) => (row.essential ? 1 : 0),
  auto: (row) => row.auto,
  amp: (row) => row.amp,
  quadrant: (row) => QUADRANT_NAMES[row.quadrant] || '',
};

function isPresent(value) {
  if (value === null || value === undefined || value === '') return false;
  return typeof value === 'number' ? Number.isFinite(value) : true;
}

function compareValues(left, right) {
  if (typeof left === 'number' && typeof right === 'number') return left - right;
  return String(left ?? '').localeCompare(String(right ?? ''));
}

/**
 * Sort the skill rows by any column. Never mutates; rows with nothing in the
 * column go last whichever way it runs, and ties break by skill name.
 */
export function sortSkillRows(rows, key, descending) {
  const read = SKILL_VALUES[key] || SKILL_VALUES.title;
  const direction = descending ? -1 : 1;
  return [...(rows || [])].sort((a, b) => {
    const left = read(a);
    const right = read(b);
    if (isPresent(left) !== isPresent(right)) return isPresent(left) ? -1 : 1;
    return compareValues(left, right) * direction || fold(a.title).localeCompare(fold(b.title));
  });
}

/** Essential, optional, or everything. */
export function filterSkillRows(rows, mode) {
  if (mode === 'essential') return (rows || []).filter((row) => row.essential);
  if (mode === 'optional') return (rows || []).filter((row) => !row.essential);
  return [...(rows || [])];
}

/**
 * One table row per skill, each cell already formatted, so the page never
 * decides what a missing score looks like.
 */
export function skillTableRows(rows) {
  return (rows || []).map((row) => ({
    id: row.id,
    title: row.title,
    quadrant: row.quadrant,
    cells: [
      { key: 'title', text: row.title },
      { key: 'essential', text: row.essential ? 'Essential' : 'Optional' },
      { key: 'auto', text: formatScore(row.auto), numeric: true },
      { key: 'amp', text: formatScore(row.amp), numeric: true },
      { key: 'quadrant', text: QUADRANT_NAMES[row.quadrant] || NOT_SCORED },
    ],
  }));
}
