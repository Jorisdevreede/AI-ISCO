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

import {
  NOT_SCORED, formatCount, formatScore, isScored, largestRemainder, plural,
} from '../format.js';
import { QUADRANT_ORDER } from '../groupstats.js';
import {
  QUADRANTS, SHARES, SKILL_CLASS_NAMES, SKILL_CLASS_ORDER, SKILL_NEAR_NOTE, isSkillNear,
  nearLineCaveat as sharedNearLineCaveat, orderForCounts, schemeOfCode, skillClassName,
  typeLabel, typeShortLabel,
} from '../scheme.js';
import { quadrantOf } from '../quadrant.js';
import { isShares, sharePercents, shareSentence } from '../shares.js';
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
const LEVEL_WIDTHS = [
  { level: 'unit', width: 4 },
  { level: 'minor', width: 3 },
  { level: 'sub', width: 2 },
  { level: 'major', width: 1 },
];

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
    .filter(({ width }) => digits.length >= width)
    .map(({ level, width }) => `${level}:${digits.slice(0, width)}`);
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
    k: row.k,
    q: row.q,
    sh: row.sh,
    nl: row.nl,
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
  return Boolean(view.expanded?.has(key));
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
  const hits = ranked.slice(0, limit);
  for (const hit of hits) {
    slugs.add(hit.row.s);
    for (const key of unitChain(hit.row.c)) if (model.groups[key]) keys.add(key);
  }
  // A Set collapses any two rows that share a slug, so its size is not what is
  // on screen. What is shown is one row per hit, and only the cap shortens it.
  return {
    slugs, keys, hits, total: ranked.length, shown: hits.length,
    capped: ranked.length > hits.length,
  };
}

/**
 * The matching jobs as a flat list, best match first: what someone who typed a
 * word actually asked for. The hierarchy that holds them is a second question,
 * so it goes behind a disclosure rather than around every result.
 *
 * @param {Object} model from buildModel
 * @param {Object|null} filter from filterMatches
 * @returns {Array<{slug, title, code, group, alt}>} empty without a filter
 */
export function resultRows(model, filter) {
  if (!filter?.hits) return [];
  return filter.hits.map((hit) => {
    const key = `unit:${String(hit.row.c || '')}`;
    const group = model.groups[key];
    return {
      slug: hit.row.s,
      title: hit.row.t,
      code: hit.row.c,
      group: group ? group.label : '',
      alt: hit.alt || null,
    };
  });
}

/** "also matches “programmer”" — why a row without the word in its title is here. */
export function alsoMatches(row) {
  return row?.alt ? `also matches “${row.alt}”` : '';
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
  if (selection?.kind === 'job') params.job = selection.id;
  if (selection?.kind === 'group') params.g = selection.id;
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

/* --- the class mix -------------------------------------------------------- */

// One shape for every labelled bar on this page, whichever thing it counts:
// the four boxes, the seven types or the four skill classes. `attribute` is the
// data-* name the view paints the colour from, so the view never has to know
// which scheme it is drawing.
// A count with the right noun ("1 job", "3,039 jobs") lives in format.js now.
// It is re-exported because the view and the tests read it from here.
export { plural };

/** "0%" is a lie about a group that has one job in that class. */
function percentText(percent, count) {
  return count > 0 && percent === 0 ? '<1%' : `${percent}%`;
}

function barSegment(part, total) {
  const share = total ? part.count / total : 0;
  return {
    code: part.code,
    label: part.label,
    attribute: part.attribute,
    count: part.count,
    share,
    percent: percentText(part.percent, part.count),
    text: `${formatCount(part.count)} of ${plural(total, part.noun)}`,
  };
}

/**
 * One labelled bar per class, with whole percentages that add up to 100.
 * Rounding each share on its own gives 99 or 101 often enough to notice — it
 * broke 77 of 604 groups — so the remainder is shared out by largestRemainder.
 *
 * @param {Array<string>} order the class codes, in display order
 * @param {Object} counts code -> count
 * @param {{attribute: string, noun: string, label: Function}} shape
 * @returns {Array<Object>}
 */
function barSegments(order, counts, shape) {
  const total = order.reduce((sum, code) => sum + (counts[code] || 0), 0);
  const percents = largestRemainder(order.map((code) => counts[code] || 0));
  return order.map((code, position) => barSegment({
    code,
    label: shape.label(code),
    count: counts[code] || 0,
    percent: percents[position],
    attribute: shape.attribute,
    noun: shape.noun,
  }, total));
}

/**
 * How a set of jobs splits over the classes of its scheme — the four boxes or
 * the seven types — each part carrying its own label, count and percentage so
 * nothing is said by colour alone.
 * @param {Object} counts the `q` map of a groups.json entry
 * @returns {Array<{code, label, attribute, count, share, percent, text}>}
 */
export function mixSegments(counts) {
  const tally = counts || {};
  const order = orderForCounts(tally);
  const attribute = schemeOfCode(order[0]) === SHARES ? 'data-type' : 'data-quadrant';
  return barSegments(order, tally, { attribute, noun: 'job', label: typeLabel });
}

/**
 * How every scored skill splits over the four classes, from stats.skill_classes.
 * @param {Object} stats the active set's stats file
 * @returns {Array<Object>} empty under a scheme that has no skill classes
 */
export function classSegments(stats) {
  const counts = stats?.skill_classes?.counts || null;
  if (!counts) return [];
  return barSegments(SKILL_CLASS_ORDER, counts,
    { attribute: 'data-class', noun: 'skill', label: skillClassName });
}

/** "51% Evolve, 26% Stable, 17% Transform, 6% Shrink" — the bar said out loud. */
export function mixText(segments) {
  const shown = (segments || []).filter((segment) => segment.count > 0);
  if (!shown.length) return 'no scored jobs';
  const parts = [...shown]
    .sort((a, b) => b.share - a.share)
    .map((segment) => `${segment.percent} ${segment.label}`)
    .join(', ');
  return shown[0].attribute === 'data-type' ? `job types: ${parts}` : parts;
}

/** "Unit group 2512 · ISCO-08 · 10 jobs" — level, code and size in one line. */
export function groupSubtitle(group) {
  const level = LEVEL_LABELS[group.level] || 'Group';
  const code = group.code ? `${level} ${group.code} · ISCO-08` : level;
  return `${code} · ${plural(group.n, 'job')}`;
}

/**
 * The visible key for the figures every job row prints. Sighted visitors had
 * only the screen-reader label to go on, which is no key at all.
 * @param {string} [scheme]
 * @returns {string}
 */
export function scoreKeyLine(scheme) {
  if (scheme === SHARES) {
    return 'Each job shows its type and how much of its work AI can take over.';
  }
  return 'Each job shows automation risk / amplification, both out of 10.';
}

/**
 * The one permanent caveat beside a split, built from stats.json. Says only
 * what the published data shows: how many jobs sit near a cut-off.
 */
export function nearLineCaveat(stats) {
  return sharedNearLineCaveat(stats);
}

/* --- one job, as a leaf --------------------------------------------------- */

function isSharesNode(node) {
  return schemeOfCode(node?.q) === SHARES || isShares(node?.sh);
}

/**
 * The figure beside a leaf: how much of the work AI can take over under the
 * shares scheme, the pair of scores under quadrants.
 * @param {Object} node a job node
 * @returns {string}
 */
export function leafFigure(node) {
  if (!isSharesNode(node)) {
    return `${formatScore(node?.a)} / ${formatScore(node?.m)}`;
  }
  const parts = sharePercents(node.sh);
  return parts.length ? `${parts[0].percent}% AI can take over` : NOT_SCORED;
}

/**
 * What a leaf paints and what it names: the data-* attribute that carries the
 * colour, the code, the short class name and the figure beside it.
 * @param {Object} node a job node
 * @returns {{attribute: string, code: string, name: string, figure: string}}
 */
export function leafParts(node) {
  const shares = isSharesNode(node);
  return {
    attribute: shares ? 'data-type' : 'data-quadrant',
    code: node?.q || '',
    name: shares ? typeShortLabel(node.q, SHARES) : typeLabel(node.q, QUADRANTS),
    figure: leafFigure(node),
  };
}

/** "Transform, automation 6.4, amplification 8.9" — a leaf said out loud. */
export function jobSummary(node) {
  if (!isSharesNode(node)) {
    return `${typeLabel(node.q)}, automation ${formatScore(node.a)}, `
      + `amplification ${formatScore(node.m)}`;
  }
  const sentence = shareSentence(node.sh);
  const name = typeShortLabel(node.q, SHARES);
  return sentence ? `${name}. ${sentence}.` : `${name}. Shares not scored.`;
}

// What the insulated part of a job is made of, in the words of `why_insulated`.
const WHY_SENTENCES = {
  physical: 'The part that stays human here is mostly work done on things, in a place.',
  people: 'The part that stays human here is mostly work done with and for other people.',
  other: 'The part that stays human here is mostly desk work, done through software or '
    + 'on paper.',
};

/**
 * The sentence about a job's insulated part, or '' where there is none to
 * explain: the run records a reason for every occupation, including the ones
 * whose stays-human share is zero, and explaining an empty part would be noise.
 *
 * @param {{why?: string, sh?: number[]}} occupation a portfolio_data occupation
 * @returns {string}
 */
export function whyLine(occupation) {
  if (!occupation || !isShares(occupation.sh) || occupation.sh[3] <= 0) return '';
  return WHY_SENTENCES[occupation.why] || '';
}

/* --- the skills of one job ------------------------------------------------ */

function resolveSkills(skills, ids, essential, cut) {
  return (ids || [])
    .map((id) => [id, skills[id]])
    .filter(([, skill]) => Boolean(skill))
    .map(([id, skill]) => ({
      id,
      title: skill.t,
      auto: isScored(skill.a) ? skill.a : null,
      amp: isScored(skill.m) ? skill.m : null,
      mech: isScored(skill.k) ? skill.k : null,
      cls: SKILL_CLASS_NAMES[skill.c] ? skill.c : null,
      // The job badge says when a job sits near a cut-off; a skill's class is
      // decided the same way and deserves the same warning.
      near: isSkillNear(skill, cut),
      essential,
      quadrant: quadrantOf(skill.a, skill.m),
    }));
}

/** The occupation a slug names inside portfolio_data.json, or null. */
export function findOccupation(portfolio, slug) {
  const rows = portfolio?.occupations || [];
  return rows.find((row) => row.s === slug) || null;
}

/**
 * Every skill of one occupation, essential first, unknown ids dropped.
 * @param {Object} portfolio a `jobs/<unit>` shard, shaped like portfolio_data
 * @param {{se?: Array<string>, so?: Array<string>}} occupation
 * @param {number} [cut] the class cut-off, `thresholdOf(stats)`
 * @returns {Array<{id, title, auto, amp, mech, cls, near, essential, quadrant}>}
 */
export function skillRows(portfolio, occupation, cut) {
  const skills = portfolio?.skills || {};
  return [
    ...resolveSkills(skills, occupation?.se, true, cut),
    ...resolveSkills(skills, occupation?.so, false, cut),
  ];
}

/** What a near-the-line skill says about itself, for the chip's title. */
export const SKILL_NEAR_TEXT = SKILL_NEAR_NOTE;

function countBy(rows, read) {
  const counts = {};
  for (const row of rows) {
    const code = read(row);
    if (code) counts[code] = (counts[code] || 0) + 1;
  }
  return counts;
}

// Under shares a skill carries its own class, so the split is the split of the
// classes — the same four parts the job's shares are built from.
function classBars(rows) {
  const counts = countBy(rows, (row) => row.cls);
  return {
    scored: SKILL_CLASS_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0),
    bars: barSegments(SKILL_CLASS_ORDER, counts,
      { attribute: 'data-class', noun: 'skill', label: skillClassName }),
  };
}

function quadrantBars(rows) {
  const counts = countBy(rows, (row) => row.quadrant);
  return {
    scored: QUADRANT_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0),
    bars: barSegments(QUADRANT_ORDER, counts,
      { attribute: 'data-quadrant', noun: 'skill', label: typeLabel }),
  };
}

/**
 * How the job's own skills split: by class under the shares scheme, across the
 * four boxes under quadrants.
 * @param {Array<Object>} rows from skillRows
 * @param {string} [scheme] 'quadrants' (default) or 'shares'
 * @returns {{total, essential, scored, unscored, bars: Array}}
 */
export function skillMix(rows, scheme) {
  const all = rows || [];
  const { scored, bars } = scheme === SHARES ? classBars(all) : quadrantBars(all);
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

/** The table's columns under the quadrant scheme, in order. */
export const SKILL_COLUMNS = [
  { key: 'title', label: 'Skill', numeric: false },
  { key: 'essential', label: 'In this job', numeric: false },
  { key: 'auto', label: 'Automation risk', numeric: true },
  { key: 'amp', label: 'Amplification', numeric: true },
  { key: 'quadrant', label: 'Quadrant', numeric: false },
];

/** The same table under the shares scheme: the class, then three scores. */
export const SHARES_SKILL_COLUMNS = [
  { key: 'title', label: 'Skill', numeric: false },
  { key: 'essential', label: 'In this job', numeric: false },
  { key: 'cls', label: 'Class', numeric: false },
  { key: 'auto', label: 'AI substitution', numeric: true },
  { key: 'amp', label: 'AI assistance', numeric: true },
  { key: 'mech', label: 'Machine automation', numeric: true },
];

/**
 * The columns of whichever scheme the active set uses.
 * @param {string} [scheme]
 * @returns {Array<{key: string, label: string, numeric: boolean}>}
 */
export function skillColumns(scheme) {
  return scheme === SHARES ? SHARES_SKILL_COLUMNS : SKILL_COLUMNS;
}

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
  mech: (row) => row.mech,
  quadrant: (row) => (row.quadrant ? typeLabel(row.quadrant) : ''),
  cls: (row) => (row.cls ? skillClassName(row.cls) : ''),
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

const SKILL_CELLS = {
  title: (row) => row.title,
  essential: (row) => (row.essential ? 'Essential' : 'Optional'),
  auto: (row) => formatScore(row.auto),
  amp: (row) => formatScore(row.amp),
  mech: (row) => formatScore(row.mech),
  quadrant: (row) => typeLabel(row.quadrant),
  cls: (row) => (row.cls ? skillClassName(row.cls) : NOT_SCORED),
};

/**
 * One table row per skill, each cell already formatted, so the page never
 * decides what a missing score looks like.
 * @param {Array<Object>} rows from skillRows
 * @param {string} [scheme] picks the column set
 */
export function skillTableRows(rows, scheme) {
  const columns = skillColumns(scheme);
  return (rows || []).map((row) => ({
    id: row.id,
    title: row.title,
    quadrant: row.quadrant,
    cls: row.cls,
    near: Boolean(row.near),
    cells: columns.map((column) => ({
      key: column.key,
      text: SKILL_CELLS[column.key](row),
      numeric: column.numeric,
    })),
  }));
}
