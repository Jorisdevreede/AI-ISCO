// The group overview, as pure functions: no DOM, no globals, no fetching.
//
// Everything groups.html shows is derived here so `node --test` can check it:
// which group the URL asks for, its trail and children, the rows behind each of
// the four views, the summaries a screen reader is given, the colours the
// treemap paints with, and the two-group comparison.
//
// Both score schemes run through these same functions. The page reads
// `schemeOf(stats)` once and passes it down; a `scheme` of 'shares' picks the
// seven types, the four shares and their wording, anything else keeps the four
// boxes exactly as they were. Nothing here sniffs a file name or a URL, and no
// sentence written for one scheme can reach the other.

import {
  NOT_SCORED, formatCount, formatScore, formatShare, isScored, largestRemainder,
} from '../format.js';
import { THRESHOLD } from '../quadrant.js';
import {
  SHARES, SKILL_CLASS_ORDER, nearLineCaveat as sharedNearLineCaveat, orderForCounts, orderOf,
  skillClassName, typeLabel, typeShortLabel,
} from '../scheme.js';
import { isShares, sharePercents, shareSentence } from '../shares.js';
import { occupationsInGroup, sortOccupations, tableColumns } from '../groupstats.js';
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

/**
 * How the treemap may be coloured, per scheme. AI exposure is deliberately not
 * offered, and neither scheme's score names leak into the other.
 */
const COLOUR_MODES_BY_SCHEME = {
  quadrants: [
    { key: 'quadrant', label: 'Quadrant' },
    { key: 'automation', label: 'Automation' },
    { key: 'amplification', label: 'Amplification' },
  ],
  shares: [
    { key: 'type', label: 'Type' },
    { key: 'substitution', label: 'AI substitution' },
    { key: 'assistance', label: 'AI assistance' },
    { key: 'mechanical', label: 'Machine automation' },
  ],
};

const LEVEL_LABELS = {
  all: 'All occupations',
  major: 'Major group',
  sub: 'Sub-major group',
  minor: 'Minor group',
  unit: 'Unit group',
};

const UNSCORED_RGB = [138, 138, 152];

/** The eleven class colours of css/app.css, as triples the canvas can paint. */
const CODE_RGB = {
  TRANSFORM: [232, 185, 59],
  SHRINK: [239, 106, 94],
  EVOLVE: [85, 192, 119],
  STABLE: [106, 169, 232],
  AUTOMATION_HEAVY: [239, 106, 94],
  TRANSFORMING: [232, 185, 59],
  AUGMENTED: [106, 169, 232],
  MECHANISABLE: [199, 155, 240],
  INSULATED_PHYSICAL: [85, 192, 119],
  INSULATED_PEOPLE: [79, 208, 196],
  MIXED: [185, 185, 200],
};

// Low-to-high ramps for the score colourings, one per colour mode. Each starts
// on the page's own slate so a low score never reads as "nothing here", and ends
// on the accent the same quantity carries everywhere else on the page.
const RAMPS = {
  automation: { field: 'automation', ramp: [[44, 62, 92], [232, 185, 59]] },
  amplification: { field: 'amplification', ramp: [[44, 62, 92], [85, 192, 119]] },
  substitution: { field: 'automation', ramp: [[44, 62, 92], [239, 106, 94]] },
  assistance: { field: 'amplification', ramp: [[44, 62, 92], [106, 169, 232]] },
  mechanical: { field: 'mechanical', ramp: [[44, 62, 92], [199, 155, 240]] },
};

const CATEGORICAL_MODES = ['quadrant', 'type'];

const isShared = (scheme) => scheme === SHARES;

const byTitle = (a, b) => String(a.t ?? '').localeCompare(String(b.t ?? ''));

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
 *
 * `sort` and `dir` are the table's column and direction: the page has a Copy
 * link button one section above the table, so the sorted view has to be in the
 * link. They are absent while the table is in its default order.
 *
 * @param {string} hash
 * @param {number} width viewport width, for the default view
 * @returns {{key, view, compare, sort: string|null, descending: boolean|null}}
 */
export function readState(hash, width) {
  const { params } = parseHash(hash);
  const named = VIEWS.includes(params.view) ? params.view : null;
  return {
    key: params.g || ROOT_KEY,
    view: named || defaultView(width),
    compare: params.a && params.b ? { a: params.a, b: params.b } : null,
    sort: params.sort || null,
    descending: params.dir ? params.dir !== 'asc' : null,
  };
}

// A shared link says which column it is sorted by in words. "sort=s1" meant
// "AI assists" only to someone reading the source; the old keys still resolve,
// so links people already copied keep working.
const SORT_WORDS = {
  quadrants: {
    t: 'title', c: 'isco', a: 'automation', m: 'amplification', q: 'quadrant',
  },
  shares: {
    t: 'title',
    c: 'isco',
    s0: 'takeover',
    s1: 'assists',
    s2: 'machines',
    s3: 'human',
    a: 'substitution',
    m: 'assistance',
    k: 'mechanisation',
    q: 'type',
  },
};

const sortWords = (scheme) => SORT_WORDS[isShared(scheme) ? 'shares' : 'quadrants'];

/**
 * The word a shared link uses for a column.
 * @param {string} key a column key of `columnsFor(scheme)`
 * @param {string} [scheme]
 * @returns {string}
 */
export function sortWordOf(key, scheme) {
  return sortWords(scheme)[key] || key;
}

/**
 * The column a shared link names, by word or by the key older links carry.
 * @param {string} word
 * @param {string} [scheme]
 * @returns {string|null} null when it names nothing this scheme has
 */
export function sortKeyOf(word, scheme) {
  const words = sortWords(scheme);
  const found = Object.keys(words).find((key) => words[key] === word);
  if (found) return found;
  return words[word] ? word : null;
}

/**
 * The table's sort, from the URL where it says one and from the scheme's own
 * default where it does not.
 * @param {{sort: string|null, descending: boolean|null}} here from `readState`
 * @param {string} [scheme]
 * @returns {{key: string, descending: boolean}}
 */
export function tableSortOf(here, scheme) {
  const wanted = sortKeyOf(here?.sort, scheme);
  const named = columnsFor(scheme).find((column) => column.key === wanted);
  if (!named) return { key: isShared(scheme) ? 's0' : 'a', descending: true };
  return { key: named.key, descending: here.descending === null ? true : here.descending };
}

/**
 * The href of one view of a group, carrying a table sort when it is not the
 * default one. A tab link never carries it, so switching view drops it.
 * @param {string} key group key
 * @param {string} [view]
 * @param {{key: string, descending: boolean}} [sort] omit for the default order
 * @param {string} [scheme]
 * @returns {string}
 */
export function groupViewHref(key, view, sort, scheme) {
  const params = { g: key, view };
  if (sort) {
    params.sort = sortWordOf(sort.key, scheme);
    params.dir = sort.descending ? 'desc' : 'asc';
  }
  return `groups.html${buildHash(null, params)}`;
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
  return groups?.[key] || null;
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
  return (group?.children || [])
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

// Whole percentages that add up to 100: see largestRemainder in format.js.
export { largestRemainder };

/** "0%" is a lie about a group that has one job in that class. */
function percentText(percent, count) {
  return count > 0 && percent === 0 ? '<1%' : `${percent}%`;
}

/**
 * The classes of the scheme in display order — four boxes or seven types —
 * each carrying its own full name, count and percentage, so nothing on the bar
 * chart is conveyed by colour alone.
 * @param {{q: Object}} group a groups.json entry
 * @param {string} [scheme] omit to read the scheme off the counts
 * @returns {Array<{code, label, count, share, percent, text}>}
 */
export function mixBars(group, scheme) {
  const counts = group?.q || {};
  const order = scheme ? orderOf(scheme) : orderForCounts(counts);
  const total = order.reduce((sum, code) => sum + (counts[code] || 0), 0);
  const percents = largestRemainder(order.map((code) => counts[code] || 0));
  return order.map((code, position) => {
    const count = counts[code] || 0;
    return {
      code,
      label: typeLabel(code, scheme),
      count,
      share: total ? count / total : 0,
      percent: percentText(percents[position], count),
      text: `${formatCount(count)} of ${formatCount(total)} jobs`,
    };
  });
}

/** One bar of the split as words: "36% Transform". */
const barText = (bar) => `${bar.percent} ${bar.label}`;

/** "36% Transform, 28% Stable, 18% Evolve, 17% Shrink." */
export function mixSentence(group, scheme) {
  return `${mixBars(group, scheme).map(barText).join(', ')}.`;
}

/**
 * The group's mean shares as one sentence, for the big bar above the split.
 * Empty under a scheme that has no shares, so the page can hide the figure.
 * @param {{n: number, sh: number[]}} group
 * @returns {string}
 */
export function meanShareSentence(group) {
  const shares = group?.sh;
  if (!isShares(shares)) return '';
  return `Across the ${formatCount(group.n)} jobs in this group, on average `
    + `${shareSentence(shares)}.`;
}

/** The accessible name of that bar: what it is, then the four parts. */
export function meanShareLabel(group) {
  const sentence = meanShareSentence(group);
  return sentence ? `What the jobs in ${group.label} are made of. ${sentence}` : '';
}

/**
 * The one permanent caveat beside the split, built from stats.json.
 * Says only what the published data shows: how many jobs sit near a cut-off.
 */
export function nearLineCaveat(stats) {
  return sharedNearLineCaveat(stats);
}

/**
 * The same caveat about THIS group, from its own `near` count.
 *
 * The site-wide figure printed inside a group panel reads as a fact about that
 * group, and it is usually the wrong one: the near-the-line share runs from
 * under a third to over a half depending on the group. A build that does not
 * carry `near` yet gets no sentence at all rather than a borrowed one.
 *
 * @param {{n: number, near?: number}} group a groups.json entry
 * @param {string} [scheme]
 * @returns {string} '' when the group has no `near` count
 */
export function groupNearSentence(group, scheme) {
  const near = group?.near;
  if (!isScored(near) || !group.n) return '';
  const share = formatShare(near, group.n, 'jobs');
  if (isShared(scheme)) {
    return `${share} in this group sit near a cut-off, where moving any one of their four `
      + 'shares by 5 points would give them a different type: read this split as a band, '
      + 'not a count.';
  }
  return `${share} in this group sit within 0.5 of a cut-off, and a small change in the `
    + 'scores moves those into another box: read this split as a band, not a count.';
}

/* --- ranked view ---------------------------------------------------------- */

/** The sort options of each scheme, the default first. */
const SORT_OPTIONS = {
  quadrants: [
    { key: 'automation', label: 'Most automation-exposed' },
    { key: 'amplification', label: 'Most amplified' },
    { key: 'title', label: 'A to Z' },
  ],
  shares: [
    { key: 'substituted', label: 'Most can be taken over' },
    { key: 'assisted', label: 'Most assisted' },
    { key: 'mechanised', label: 'Most mechanical' },
    { key: 'insulated', label: 'Most stays human' },
    { key: 'title', label: 'A to Z' },
  ],
};

const SHARE_SORT_AT = { substituted: 0, assisted: 1, mechanised: 2, insulated: 3 };

/**
 * The sort options the ranked view offers under this scheme.
 * @param {string} [scheme]
 * @returns {Array<{key: string, label: string}>}
 */
export function rankedSorts(scheme) {
  return [...SORT_OPTIONS[isShared(scheme) ? 'shares' : 'quadrants']];
}

/** The sort a visitor gets before touching the control. */
export function defaultSort(scheme) {
  return rankedSorts(scheme)[0].key;
}

/** Where a 1-10 score sits on the shared axis, as a percentage. */
export function axisPercent(score) {
  if (!isScored(score)) return null;
  return ((Math.min(10, Math.max(1, score)) - 1) / 9) * 100;
}

function dot(value) {
  return { value, text: formatScore(value), percent: axisPercent(value) };
}

function shareAt(row, position) {
  const shares = row?.sh;
  return isShares(shares) ? shares[position] : null;
}

function sharePercentText(row, position) {
  const parts = sharePercents(row?.sh);
  return parts.length ? `${parts[position].percent}%` : NOT_SCORED;
}

function sortByShare(rows, sortKey) {
  if (sortKey === 'title') return [...(rows || [])].sort(byTitle);
  const at = SHARE_SORT_AT[sortKey] ?? SHARE_SORT_AT.substituted;
  return [...(rows || [])].sort((a, b) => {
    const left = shareAt(a, at);
    const right = shareAt(b, at);
    if (left === null || right === null) return (left === null ? 1 : 0) - (right === null ? 1 : 0);
    return (right - left) || byTitle(a, b);
  });
}

function toRankedRow(row, scheme) {
  return {
    slug: row.s,
    title: row.t,
    code: row.c,
    quadrant: row.q,
    quadrantLabel: typeLabel(row.q, scheme),
    typeShort: typeShortLabel(row.q, scheme),
    shares: isShares(row.sh) ? row.sh : null,
    shareText: shareSentence(row.sh),
    automation: dot(row.a),
    amplification: dot(row.m),
    mechanical: dot(row.k),
  };
}

/**
 * One row per job for the ranked view, sorted.
 * @param {Array<Object>} rows search_index rows
 * @param {string} sortKey a key of `rankedSorts(scheme)`
 * @param {string} [scheme]
 */
export function rankedRows(rows, sortKey, scheme) {
  const sorted = isShared(scheme)
    ? sortByShare(rows, sortKey)
    : sortOccupations(rows, sortKey);
  return sorted.map((row) => toRankedRow(row, scheme));
}

/**
 * What a ranked row says to a screen reader, in the words of its scheme.
 *
 * It starts with a separator: this text follows the job title and its ISCO
 * code inside the same link, and without one the accessible name ran them
 * together ("subtitler 26437.8 / 8.4").
 */
export function rankedRowSummary(row, scheme) {
  if (isShared(scheme)) {
    return `, ${row.shareText || 'shares not scored'}, ${row.quadrantLabel}`;
  }
  return `, automation ${row.automation.text}, amplification ${row.amplification.text}, `
    + `${row.quadrantLabel}`;
}

/** How many ranked rows are drawn before the visitor asks for more. */
export const RANKED_PAGE = 50;

/**
 * The slice of the ranked list that is on screen.
 *
 * 869 rows is 114,000 px of phone: a list nobody can reach the end of, and a
 * lot of DOM for a page that also has to draw a treemap. The page starts at
 * one screenful of rows and lets the visitor ask for the rest.
 *
 * @param {Array<Object>} rows every ranked row of the group
 * @param {number} [shown] how many the visitor has asked for so far
 * @returns {{rows: Array, visible: number, total: number, more: number, next: number}}
 */
export function rankedPage(rows, shown) {
  const list = rows || [];
  const total = list.length;
  const wanted = Number.isFinite(shown) && shown > 0 ? shown : RANKED_PAGE;
  const visible = Math.min(wanted, total);
  return {
    rows: list.slice(0, visible),
    visible,
    total,
    more: total - visible,
    next: Math.min(RANKED_PAGE, total - visible),
  };
}

/** "Showing 50 of 869 jobs." — announced politely after each step. */
export function rankedCountText(page) {
  if (!page.more) {
    return `Showing all ${formatCount(page.total)} ${page.total === 1 ? 'job' : 'jobs'}.`;
  }
  return `Showing ${formatCount(page.visible)} of ${formatCount(page.total)} jobs.`;
}

/** The two buttons under a shortened list, or none when it is whole. */
export function rankedMoreButtons(page) {
  if (!page.more) return [];
  const buttons = [{ key: 'more', label: `Show ${formatCount(page.next)} more`, step: page.next }];
  if (page.more > page.next) {
    buttons.push({ key: 'all', label: `Show all ${formatCount(page.total)}`, step: page.more });
  }
  return buttons;
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
 * Where a pair of shares lands in the same box, both axes running 0 to 100%.
 * @param {number} across share on the x axis, 0 to 1
 * @param {number} up share on the y axis, 0 to 1
 * @param {{width: number, height: number, pad: number}} box
 */
export function plotShare(across, up, box) {
  const inner = { width: box.width - 2 * box.pad, height: box.height - 2 * box.pad };
  return {
    x: box.pad + Math.min(1, Math.max(0, across)) * inner.width,
    y: box.height - box.pad - Math.min(1, Math.max(0, up)) * inner.height,
  };
}

function pointLabel(row, scheme) {
  if (isShared(scheme)) {
    return `${row.t}, ${shareSentence(row.sh)}, ${typeLabel(row.q, scheme)}.`;
  }
  return `${row.t}, automation ${formatScore(row.a)}, amplification ${formatScore(row.m)}, `
    + `${typeLabel(row.q, scheme)}.`;
}

function sharePoint(row, box, scheme) {
  return {
    slug: row.s,
    title: row.t,
    quadrant: row.q,
    ...plotShare(row.sh[0], row.sh[1], box),
    label: pointLabel(row, scheme),
  };
}

/**
 * One dot per scored job.
 *
 * Under the shares scheme the axes are the two shares the type rules cut, not
 * the display scores: those two scores are strongly correlated, so every group
 * drew the same diagonal smear in one corner of the box and the picture said
 * nothing the badge did not.
 *
 * @returns {Array<{slug, title, quadrant, x, y, label}>}
 */
export function scatterPoints(rows, box, scheme) {
  if (isShared(scheme)) {
    return (rows || []).filter((row) => isShares(row.sh))
      .map((row) => sharePoint(row, box, scheme));
  }
  return (rows || [])
    .filter((row) => isScored(row.a) && isScored(row.m))
    .map((row) => ({
      slug: row.s,
      title: row.t,
      quadrant: row.q,
      ...plotPoint(row.a, row.m, box),
      label: pointLabel(row, scheme),
    }));
}

/** Where the two cut-off lines cross, in the same box. Quadrants only. */
export function thresholdPoint(box) {
  return plotPoint(THRESHOLD, THRESHOLD, box);
}

/** What the two axes are called, in the words of the scheme. */
export function axisNames(scheme) {
  return isShared(scheme)
    ? { x: 'AI can take over', y: 'AI assists' }
    : { x: 'Automation', y: 'Amplification' };
}

/** What the ends of the two axes are labelled. */
export function axisTicks(scheme) {
  return isShared(scheme) ? { low: '0%', high: '100%' } : { low: '1', high: '10' };
}

// The cuts the type rules make on these two axes (docs/scoring-v2.md): half and
// three in ten of the work AI can take over, and two or three in ten assisted,
// depending on which side of the 30% line the job sits.
// The cuts the type rules make on these two axes. `onAxis` says the label
// belongs in the tick column beside the y axis, because the line reaches it;
// the 20% line starts at x = 30%, so its label goes inside the plot at that
// end. Both y lines carry a label of their own.
const RULE_CUTS = [
  { axis: 'x', at: 0.5, from: 0, to: 1 },
  { axis: 'x', at: 0.3, from: 0, to: 1 },
  { axis: 'y', at: 0.2, from: 0.3, to: 1, onAxis: false },
  { axis: 'y', at: 0.3, from: 0, to: 0.3, onAxis: true },
];

function verticalRule(cut, box) {
  const { x } = plotShare(cut.at, 0, box);
  return {
    x1: x, y1: box.pad, x2: x, y2: box.height - box.pad,
    textX: x + 3, textY: box.height - box.pad + 16, anchor: 'start',
  };
}

function horizontalRule(cut, box) {
  const { y } = plotShare(0, cut.at, box);
  const from = plotShare(cut.from, 0, box).x;
  const label = cut.onAxis
    ? { textX: box.pad - 8, textY: y + 4, anchor: 'end' }
    : { textX: from + 3, textY: y - 4, anchor: 'start' };
  return { x1: from, y1: y, x2: plotShare(cut.to, 0, box).x, y2: y, ...label };
}

/**
 * The type rules as lines across the plot, each with its own label.
 * Empty under quadrants, where the cut-off lines are drawn instead.
 * @returns {Array<{label, x1, y1, x2, y2, textX, textY, anchor}>}
 */
export function ruleLines(box, scheme) {
  if (!isShared(scheme)) return [];
  return RULE_CUTS.map((cut) => ({
    label: `${Math.round(cut.at * 100)}%`,
    ...(cut.axis === 'x' ? verticalRule(cut, box) : horizontalRule(cut, box)),
  }));
}

/* --- which types these two axes can actually decide ----------------------- */

/**
 * Half a rounding step of the published shares.
 *
 * `sh` is rounded to two decimals, so a job whose true share was 0.2951 reads
 * 0.30 here and sits on a line it is not actually over. A region only claims a
 * dot that is clear of every cut bounding it by more than that, which is what
 * makes "every dot in this region is this type" true rather than nearly true.
 */
export const REGION_MARGIN = 0.005;

// Only three of the seven types are settled by the two shares this chart plots:
// the other four turn on the machines and stays-human shares, which are not on
// either axis. These are the rules of docs/scoring-v2.md read on this plane.
const REGIONS = [
  { code: 'AUTOMATION_HEAVY', x: [0.5, 1], y: [0, 1] },
  { code: 'TRANSFORMING', x: [0.3, 0.5], y: [0.2, 1] },
  { code: 'AUGMENTED', x: [0, 0.3], y: [0.3, 1] },
];

function within(value, [low, high]) {
  const lowEnough = high >= 1 || value <= high - REGION_MARGIN;
  return value >= low + (low > 0 ? REGION_MARGIN : 0) && lowEnough;
}

/**
 * The type this chart can prove for a job, or null when its two shares cannot.
 * @param {number[]} shares `sh`
 * @returns {string|null}
 */
export function regionOf(shares) {
  if (!isShares(shares)) return null;
  const found = REGIONS.find((region) => within(shares[0], region.x) && within(shares[1], region.y));
  return found ? found.code : null;
}

function regionBox(region, box) {
  const topLeft = plotShare(region.x[0], region.y[1], box);
  const bottomRight = plotShare(region.x[1], region.y[0], box);
  return {
    x: topLeft.x,
    y: topLeft.y,
    width: bottomRight.x - topLeft.x,
    height: bottomRight.y - topLeft.y,
  };
}

/**
 * The three regions the lines really do decide, each with a place for its name.
 * @returns {Array<{code, label, x, y, width, height, textX, textY}>}
 */
export function namedRegions(box, scheme) {
  if (!isShared(scheme)) return [];
  return REGIONS.map((region) => {
    const rect = regionBox(region, box);
    return {
      code: region.code,
      label: typeShortLabel(region.code, SHARES),
      ...rect,
      textX: rect.x + rect.width / 2,
      textY: box.pad + 14,
    };
  });
}

/**
 * The rest of the plot, where these two axes do not settle the type: an L along
 * the bottom left, drawn as two rectangles so it can be tinted.
 * @returns {{rects: Array<Object>, label: string, textX: number, textY: number}|null}
 */
export function undecidedArea(box, scheme) {
  if (!isShared(scheme)) return null;
  const rects = [
    regionBox({ x: [0, 0.3], y: [0, 0.3] }, box),
    regionBox({ x: [0.3, 0.5], y: [0, 0.2] }, box),
  ];
  return {
    rects,
    label: 'Decided by the other two shares',
    textX: rects[0].x + 6,
    textY: box.height - box.pad - 8,
  };
}

/** What the lines mean, and what they cannot mean, in words, under the chart. */
export function scatterCaption(scheme) {
  if (!isShared(scheme)) return '';
  return 'The lines are the cuts for three types only: Automation-heavy right of 50%, '
    + 'Transforming between 30% and 50% and above 20%, Augmented left of 30% and above 30%. '
    + 'The other four types — Mechanisable, the two insulated ones and Mixed — are decided by '
    + 'the machines and stays-human shares, which this chart does not plot; their jobs sit in '
    + 'the shaded area.';
}

function shareRange(rows, position) {
  const values = (rows || []).filter((row) => isShares(row.sh)).map((row) => row.sh[position]);
  if (!values.length) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

function rangeText(name, range) {
  return `${name} runs ${Math.round(range.min * 100)}% to ${Math.round(range.max * 100)}%`;
}

/** "AI can take over runs 5% to 79%, AI assists 10% to 67%." */
export function shareAxisSentence(rows) {
  const across = shareRange(rows, 0);
  const up = shareRange(rows, 1);
  if (!across || !up) return 'No job in this group has shares.';
  return `${rangeText('AI can take over', across)}, ${rangeText('AI assists', up)}.`;
}

/**
 * The classes actually present in these rows, for the scatter legend. Empty
 * under quadrants, where the four boxes are named in the corners instead.
 * @returns {Array<{code: string, label: string}>}
 */
export function typeLegend(rows, scheme) {
  if (!isShared(scheme)) return [];
  const present = new Set((rows || []).map((row) => row.q));
  return orderOf(scheme)
    .filter((code) => present.has(code))
    .map((code) => ({ code, label: typeLabel(code, scheme) }));
}

/** Coordinates to two decimals, so a path string stays short and exact. */
const at = (value) => Number(value.toFixed(2));

const MARKERS = {
  AUTOMATION_HEAVY: (r) => `M 0 ${at(-r)} L ${at(r)} ${at(r * 0.8)} L ${at(-r)} ${at(r * 0.8)} Z`,
  TRANSFORMING: (r) => `M 0 ${at(-r)} L ${at(r)} 0 L 0 ${at(r)} L ${at(-r)} 0 Z`,
  AUGMENTED: (r) => `M ${at(-r)} 0 a ${at(r)} ${at(r)} 0 1 0 ${at(r * 2)} 0 `
    + `a ${at(r)} ${at(r)} 0 1 0 ${at(-r * 2)} 0 Z`,
  MECHANISABLE: (r) => `M ${at(-r * 0.85)} ${at(-r * 0.85)} H ${at(r * 0.85)} `
    + `V ${at(r * 0.85)} H ${at(-r * 0.85)} Z`,
  INSULATED_PHYSICAL: (r) => `M 0 ${at(r)} L ${at(r)} ${at(-r * 0.8)} L ${at(-r)} ${at(-r * 0.8)} Z`,
  INSULATED_PEOPLE: (r) => `M 0 ${at(-r)} L ${at(r)} ${at(-r * 0.31)} L ${at(r * 0.62)} `
    + `${at(r * 0.81)} L ${at(-r * 0.62)} ${at(r * 0.81)} L ${at(-r)} ${at(-r * 0.31)} Z`,
  MIXED: (r) => `M ${at(-r / 3)} ${at(-r)} H ${at(r / 3)} V ${at(-r / 3)} H ${at(r)} `
    + `V ${at(r / 3)} H ${at(r / 3)} V ${at(r)} H ${at(-r / 3)} V ${at(r / 3)} `
    + `H ${at(-r)} V ${at(-r / 3)} H ${at(-r / 3)} Z`,
};

/**
 * One marker outline per type, centred on the origin, so the scatter tells the
 * seven types apart by shape as well as by colour.
 * @param {number} radius
 * @returns {Array<{code: string, d: string}>}
 */
export function markerShapes(radius) {
  return orderOf(SHARES).map((code) => ({ code, d: MARKERS[code](radius) }));
}

/* --- table view ----------------------------------------------------------- */

const SHARE_COLUMNS = [
  { key: 's0', label: 'AI can take over', numeric: true },
  { key: 's1', label: 'AI assists', numeric: true },
  { key: 's2', label: 'Machines', numeric: true },
  { key: 's3', label: 'Stays human', numeric: true },
];

const SHARES_LABELS = {
  a: 'AI substitution',
  m: 'AI assistance',
};

function sharesColumn(column) {
  return SHARES_LABELS[column.key] ? { ...column, label: SHARES_LABELS[column.key] } : column;
}

/**
 * The columns of the table view: the shared five under quadrants, and under
 * shares the same five with the score names of this scheme, the four share
 * percentages and machine automation.
 * @param {string} [scheme]
 * @returns {Array<{key: string, label: string, numeric: boolean}>}
 */
export function columnsFor(scheme) {
  const base = tableColumns(scheme);
  if (!isShared(scheme)) return base;
  const scores = base.filter((column) => column.key === 'a' || column.key === 'm')
    .map(sharesColumn)
    .concat([{ key: 'k', label: 'Machine automation', numeric: true }]);
  const head = base.filter((column) => column.key === 't' || column.key === 'c');
  const tail = base.filter((column) => column.key === 'q');
  return [...head, ...SHARE_COLUMNS, ...scores, ...tail];
}

const CELL_TEXT = {
  t: (row) => String(row.t ?? ''),
  c: (row) => String(row.c ?? ''),
  a: (row) => formatScore(row.a),
  m: (row) => formatScore(row.m),
  k: (row) => formatScore(row.k),
  s0: (row) => sharePercentText(row, 0),
  s1: (row) => sharePercentText(row, 1),
  s2: (row) => sharePercentText(row, 2),
  s3: (row) => sharePercentText(row, 3),
};

function cellValue(row, key) {
  if (key === 'q') return typeLabel(row.q);
  if (key.startsWith('s') && key.length === 2) return shareAt(row, Number(key[1]));
  return row[key];
}

/**
 * Rows for the table view, one cell per column of `columnsFor(scheme)`.
 * @param {Array<Object>} rows
 * @param {string} [scheme]
 */
export function tableBody(rows, scheme) {
  const columns = columnsFor(scheme);
  return (rows || []).map((row) => ({
    slug: row.s,
    title: String(row.t ?? ''),
    cells: columns.map((column) => ({
      key: column.key,
      text: column.key === 'q' ? typeLabel(row.q, scheme) : CELL_TEXT[column.key](row),
      numeric: column.numeric,
    })),
  }));
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

/**
 * The colour modes the treemap offers under this scheme, the default first.
 * @param {string} [scheme]
 */
export function colourModes(scheme) {
  return [...COLOUR_MODES_BY_SCHEME[isShared(scheme) ? 'shares' : 'quadrants']];
}

/** Above this share of one class, colouring by class paints one flat colour. */
const DOMINANT = 0.6;

/**
 * The colour mode a visitor gets before touching the chips.
 *
 * Colouring by class is the informative default — until one class holds most
 * of the group, when it draws a uniform green that carries nothing. Seven in
 * ten Professionals sit in one box, so that group opened on a flat treemap. In
 * that case the score ramp says more, so the page opens on it instead.
 *
 * @param {string} [scheme]
 * @param {Object} [group] the group about to be drawn
 */
export function defaultColour(scheme, group) {
  const modes = colourModes(scheme);
  if (!group) return modes[0].key;
  const biggest = mixBars(group, scheme).reduce((top, bar) => Math.max(top, bar.share), 0);
  return biggest > DOMINANT ? modes[1].key : modes[0].key;
}

function dominantCode(counts, scheme) {
  const tally = counts || {};
  const order = scheme ? orderOf(scheme) : orderForCounts(tally);
  return order.reduce(
    (best, code) => ((tally[code] || 0) > (tally[best] || 0) ? code : best),
    order[0],
  );
}

function groupTile(key, group, scheme) {
  const quadrant = dominantCode(group.q, scheme);
  return {
    id: key,
    kind: 'group',
    label: group.label,
    value: group.n,
    automation: group.auto?.mean,
    amplification: group.amp?.mean,
    mechanical: group.mech?.mean,
    shares: isShares(group.sh) ? group.sh : null,
    quadrant,
    detail: `${formatCount(group.n)} jobs · mostly ${typeShortLabel(quadrant, scheme)}`,
  };
}

function jobDetail(row, scheme) {
  if (isShared(scheme) && isShares(row.sh)) {
    return `AI can take over ${sharePercentText(row, 0)} · ${typeShortLabel(row.q, scheme)}`;
  }
  return `automation ${formatScore(row.a)} · ${typeShortLabel(row.q, scheme)}`;
}

function jobTile(row, scheme) {
  return {
    id: row.s,
    kind: 'job',
    label: row.t,
    value: 1,
    automation: row.a,
    amplification: row.m,
    mechanical: row.k,
    shares: isShares(row.sh) ? row.sh : null,
    quadrant: row.q,
    detail: jobDetail(row, scheme),
  };
}

/**
 * The tiles of the treemap: the child groups when there are any, otherwise the
 * jobs themselves. Sized by the number of jobs, so search_index is enough and
 * the page never touches the big data files.
 */
export function treemapTiles(groups, rows, key, scheme) {
  const children = childGroups(groups, key);
  if (children.length) {
    return children.map((child) => groupTile(child.key, groups[child.key], scheme));
  }
  return occupationsInGroup(rows, key).map((row) => jobTile(row, scheme));
}

function describeGroupTile(tile) {
  const average = tile.shares
    ? `on average AI can take over ${sharePercents(tile.shares)[0].percent}%`
    : `average automation ${formatScore(tile.automation)}`;
  return `${tile.label}, ${formatCount(tile.value)} jobs, ${average}. `
    + 'Press Enter to drill down.';
}

function describeJobTile(tile) {
  const detail = tile.shares
    ? shareSentence(tile.shares)
    : `automation ${formatScore(tile.automation)}, amplification `
      + `${formatScore(tile.amplification)}`;
  return `${tile.label}, ${detail}, ${typeLabel(tile.quadrant)}. `
    + 'Press Enter to open the job page.';
}

/** What the live region says about the tile that has the focus ring. */
export function describeTile(tile) {
  if (!tile) return '';
  return tile.kind === 'group' ? describeGroupTile(tile) : describeJobTile(tile);
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

function rampColour(entry, tile) {
  const score = tile[entry.field];
  if (!isScored(score)) return UNSCORED_RGB;
  const position = Math.min(1, Math.max(0, (score - 1) / 9));
  const [low, high] = entry.ramp;
  return low.map((channel, index) => Math.round(channel + (high[index] - channel) * position));
}

/**
 * The fill and text colour of one tile.
 * @param {Object} tile from treemapTiles
 * @param {string} mode a key of `colourModes(scheme)`
 * @returns {{rgb: number[], fill: string, text: string}}
 */
export function tileColour(tile, mode) {
  const rgb = CATEGORICAL_MODES.includes(mode)
    ? CODE_RGB[tile.quadrant] || UNSCORED_RGB
    : rampColour(RAMPS[mode] || RAMPS.automation, tile);
  return { rgb, fill: `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`, text: textColourFor(rgb) };
}

/* --- summaries for the visuals -------------------------------------------- */

function scoreRange(rows, key) {
  const values = (rows || []).map((row) => row[key]).filter(isScored);
  if (!values.length) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

function clause(name, range) {
  return range ? `${name} runs ${formatScore(range.min)} to ${formatScore(range.max)}` : null;
}

function shareRangeSentence(rows) {
  const parts = [
    clause('AI substitution', scoreRange(rows, 'a')),
    clause('AI assistance', scoreRange(rows, 'm')),
    clause('machine automation', scoreRange(rows, 'k')),
  ].filter(Boolean);
  if (!parts.length) return 'No job in this group has scores.';
  return `${parts.join(', ')}, each on a 1 to 10 scale.`;
}

/** "Automation runs 1.9 to 8.6, amplification 3.0 to 10.0. Cut-off at 6." */
export function rangeSentence(rows, scheme) {
  if (isShared(scheme)) return shareRangeSentence(rows);
  const automation = scoreRange(rows, 'a');
  const amplification = scoreRange(rows, 'm');
  if (!automation || !amplification) return 'No job in this group has scores.';
  return `Automation runs ${formatScore(automation.min)} to ${formatScore(automation.max)}, `
    + `amplification ${formatScore(amplification.min)} to ${formatScore(amplification.max)}. `
    + `The cut-off on both axes is ${THRESHOLD}.`;
}

const OPENERS = {
  quadrants: {
    ranked: (group, rows) => `Dot plot of ${formatCount(rows.length)} jobs in ${group.label}, `
      + 'automation and amplification on a shared 1 to 10 axis.',
    scatter: (group, rows) => `Scatter plot of ${formatCount(rows.length)} jobs in `
      + `${group.label}, automation across and amplification up.`,
    treemap: (group, rows) => `Treemap of ${group.label}, ${formatCount(rows.length)} jobs, `
      + 'each tile sized by the number of jobs it holds.',
    table: (group, rows) => `Table of ${formatCount(rows.length)} jobs in ${group.label}.`,
  },
  shares: {
    ranked: (group, rows) => `Ranked list of ${formatCount(rows.length)} jobs in ${group.label}, `
      + 'each one split into the four shares of its skills.',
    scatter: (group, rows) => `Scatter plot of ${formatCount(rows.length)} jobs in `
      + `${group.label}, the share AI can take over across and the share AI assists with up, `
      + 'each dot shaped and coloured by the job\'s type.',
  },
};

function opener(kind, scheme, group, rows) {
  const set = OPENERS[isShared(scheme) ? 'shares' : 'quadrants'];
  const build = set[kind] || OPENERS.quadrants[kind] || OPENERS.quadrants.table;
  return build(group, rows);
}

/** The scatter is the one view whose axes are not the three display scores. */
function summaryRange(kind, rows, scheme) {
  return kind === 'scatter' && isShared(scheme)
    ? shareAxisSentence(rows)
    : rangeSentence(rows, scheme);
}

/**
 * The aria-label every visual carries.
 * @param {'ranked'|'scatter'|'treemap'|'table'} kind
 * @param {Object} group a groups.json entry
 * @param {Array<Object>} rows the group's occupations
 * @param {string} [scheme]
 */
export function visualSummary(kind, group, rows, scheme) {
  const head = opener(kind, scheme, group, rows || []);
  return `${head} ${mixSentence(group, scheme)} ${summaryRange(kind, rows, scheme)}`;
}

/* --- most and least exposed, and the skills behind them ------------------- */

/** slug -> search_index row, for resolving the slug lists in groups.json. */
export function indexBySlug(rows) {
  return new Map((rows || []).map((row) => [row.s, row]));
}

function exposureRow(row) {
  return {
    slug: row.s,
    title: row.t,
    quadrant: row.q,
    typeLabel: typeLabel(row.q),
    automation: row.a,
    amplification: row.m,
    shares: isShares(row.sh) ? row.sh : null,
  };
}

function resolveSlugs(slugs, bySlug, limit) {
  return (slugs || [])
    .map((slug) => bySlug.get(slug))
    .filter(Boolean)
    .slice(0, limit)
    .map(exposureRow);
}

function bySubstitutedShare(rows, limit) {
  const ranked = (rows || [])
    .filter((row) => isShares(row.sh))
    .sort((a, b) => (b.sh[0] - a.sh[0]) || byTitle(a, b));
  return {
    top: ranked.slice(0, limit).map(exposureRow),
    bottom: ranked.slice(-limit).reverse().map(exposureRow),
  };
}

/**
 * The five jobs of a group at each end of the exposure order: the pipeline's
 * own lists under quadrants, and the largest and smallest "AI can take over"
 * share under shares.
 * @param {{scheme?: string, group?: Object, rows?: Array, bySlug?: Map}} context
 * @param {number} [limit=5]
 * @returns {{top: Array<Object>, bottom: Array<Object>}}
 */
export function exposureLists(context, limit = 5) {
  const { scheme, group, rows, bySlug } = context || {};
  if (isShared(scheme)) return bySubstitutedShare(rows, limit);
  return {
    top: resolveSlugs(group?.top, bySlug, limit),
    bottom: resolveSlugs(group?.bottom, bySlug, limit),
  };
}

/** The headings above those two lists, in the words of the scheme. */
export function exposureHeadings(scheme) {
  return isShared(scheme)
    ? { top: 'Most of the work AI can take over', bottom: 'Least of the work AI can take over' }
    : { top: 'Most exposed to automation', bottom: 'Least exposed' };
}

/* --- the skills behind a group -------------------------------------------- */

const NO_SKILLS = 'No skills listed for this group.';

const SKILL_COLUMNS = {
  quadrants: [
    { key: 'auto', heading: 'Most automatable', empty: NO_SKILLS },
    { key: 'amp', heading: 'Most amplified', empty: NO_SKILLS },
  ],
  shares: [
    {
      key: 'S',
      heading: 'Skills AI can take over most often here',
      empty: 'None of this group\'s most common skills is one AI can take over.',
    },
    {
      key: 'A',
      heading: 'Skills AI assists with most often here',
      empty: 'None of this group\'s most common skills is one AI assists with.',
    },
  ],
};

/** Where these two lists come from, said plainly. Empty under quadrants. */
export function skillsNote(scheme) {
  return isShared(scheme)
    ? 'Taken from the skills these jobs list most often, sorted into classes by the '
      + 'same rule as everywhere else on this site.'
    : '';
}

function skillRow(skills, id) {
  const found = skills?.get(id);
  return typeof found === 'string' ? { t: found } : found || null;
}

function skillEntries(list, skills) {
  return (list || []).map((entry) => ({
    id: entry.id,
    n: entry.n,
    title: skillRow(skills, entry.id)?.t || null,
    count: `in ${formatCount(entry.n)} jobs`,
  }));
}

/** Every skill the group's two lists name, the larger count per id kept. */
function pooledSkills(group) {
  const skills = group?.skills || {};
  const pool = new Map();
  for (const entry of [...(skills.auto || []), ...(skills.amp || [])]) {
    const seen = pool.get(entry.id);
    if (!seen || entry.n > seen.n) pool.set(entry.id, entry);
  }
  return [...pool.values()].sort((a, b) => (b.n - a.n) || a.id.localeCompare(b.id));
}

function byClass(pool, skills, code, limit) {
  return pool.filter((entry) => skillRow(skills, entry.id)?.c === code).slice(0, limit);
}

function sharesSkillColumns(group, skills, limit) {
  const pool = pooledSkills(group);
  const pending = pool.length > 0 && !skills;
  return {
    pending,
    columns: SKILL_COLUMNS.shares.map((column) => ({
      ...column,
      entries: pending ? [] : skillEntries(byClass(pool, skills, column.key, limit), skills),
    })),
  };
}

/**
 * The skills driving a group. Under quadrants these are the pipeline's two
 * lists, their titles resolved through skill_index when it has arrived; under
 * shares the same pool is split by the skill's own class, which needs that file
 * too, so `pending` says whether the page should show a placeholder.
 *
 * @param {Object} group a groups.json entry
 * @param {Map<string, Object|string>|null} skills skill_index rows by id
 * @param {string} [scheme]
 * @param {number} [limit=5]
 * @returns {{pending: boolean, columns: Array<{key, heading, entries}>}}
 */
export function drivingSkills(group, skills, scheme, limit = 5) {
  if (isShared(scheme)) return sharesSkillColumns(group, skills, limit);
  const lists = group?.skills || {};
  return {
    pending: false,
    columns: SKILL_COLUMNS.quadrants.map((column) => ({
      ...column,
      entries: skillEntries((lists[column.key] || []).slice(0, limit), skills),
    })),
  };
}

/** Which skill ids a page still has to look up, so it loads skill_index once. */
export function skillIdsOf(group) {
  return pooledSkills(group).map((entry) => entry.id);
}

/* --- comparing two groups ------------------------------------------------- */

function shareBars(group) {
  const parts = sharePercents(group?.sh);
  return parts.map((part) => ({
    code: part.code,
    label: skillClassName(part.code),
    share: part.share,
    percent: `${part.percent}%`,
    text: `${part.percent}% of the skills in this group`,
  }));
}

function medians(group, scheme) {
  const base = {
    automation: group.auto?.p50,
    amplification: group.amp?.p50,
  };
  return isShared(scheme) ? { ...base, mechanical: group.mech?.p50 } : base;
}

/** "Median AI substitution 6.0 · median AI assistance 6.7 · …" */
export function medianSentence(side, scheme) {
  const { automation, amplification, mechanical } = side.medians;
  if (!isShared(scheme)) {
    return `Median automation ${formatScore(automation)} · `
      + `median amplification ${formatScore(amplification)}`;
  }
  return `Median AI substitution ${formatScore(automation)} · median AI assistance `
    + `${formatScore(amplification)} · median machine automation ${formatScore(mechanical)}`;
}

function comparisonSide(groups, key, scheme) {
  const group = resolveGroup(groups, key);
  if (!group) return null;
  return {
    key,
    label: group.label,
    subtitle: groupSubtitle(group),
    bars: mixBars(group, scheme),
    shares: isShares(group.sh) ? group.sh : null,
    shareBars: shareBars(group),
    shareSentence: meanShareSentence(group),
    medians: medians(group, scheme),
  };
}

const pointPlural = (points) => (points === 1 ? '' : 's');

const moreOrFewer = (delta) => (delta > 0 ? 'more' : 'fewer');

/** "3 percentage points more in Managers", or "the same share" when they tie. */
function deltaSentence(label, delta, points) {
  if (points === 0) return 'the same share';
  return `${points} percentage point${pointPlural(points)} ${moreOrFewer(delta)} in ${label}`;
}

function differenceRow(a, b, left, right) {
  const delta = left.share - right.share;
  const points = Math.round(Math.abs(delta) * 100);
  return {
    code: left.code,
    label: left.label,
    a: left,
    b: right,
    delta,
    text: `${left.percent} in ${a.label}, ${right.percent} in ${b.label}`,
    deltaText: deltaSentence(a.label, delta, points),
  };
}

function differencesBetween(a, b, field) {
  return a[field].map((bar, position) => differenceRow(a, b, bar, b[field][position]));
}

/**
 * Two groups side by side: each one's split and medians, the difference per
 * class for the diverging bars, and under shares the difference per share too.
 * @returns {{a, b, differences, shareDifferences}|null}
 */
export function compareGroups(groups, aKey, bKey, scheme) {
  const a = comparisonSide(groups, aKey, scheme);
  const b = comparisonSide(groups, bKey, scheme);
  if (!a || !b) return null;
  const shared = isShared(scheme) && a.shareBars.length && b.shareBars.length;
  return {
    a,
    b,
    differences: differencesBetween(a, b, 'bars'),
    shareDifferences: shared ? differencesBetween(a, b, 'shareBars') : [],
  };
}

/** The four skill classes, for a legend that names every colour it uses. */
export function shareClassLegend() {
  return SKILL_CLASS_ORDER.map((code) => ({ code, label: skillClassName(code) }));
}
