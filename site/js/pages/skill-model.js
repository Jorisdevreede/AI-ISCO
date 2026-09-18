// Looking a skill up, and relating it to other skills. Pure: no DOM, no globals.
//
// The skill index is a flat list of 13,475 rows {id, t, a, m, ne, no}; a shares
// set adds {k, c, p}. ESCO phrases a skill as a verb ("manage budgets"), so the
// ranking is title-only — there are no alternative labels to fall back on,
// which is why the no-match state has to teach the wording instead of guessing
// at it.
//
// Everything that differs between the two schemes is one small function per
// scheme behind a dispatcher taking `scheme`, so skill.js stays a thin wiring
// layer and both schemes are tested here.

import { NOT_SCORED, formatCount, formatPercent, formatScore, isScored } from '../format.js';
import { SORTS, sortOccupations } from '../groupstats.js';
import { THRESHOLD } from '../quadrant.js';
import {
  QUADRANTS, SHARES, SKILL_CLASS_ORDER, schemeOf, skillClassName, skillClassOf,
  typeShortLabel,
} from '../scheme.js';
import { isShares, sharePercents } from '../shares.js';
import { fold } from '../search.js';

/** Match tiers, best first. Exposed so the UI can say why a row matched. */
export const SKILL_TIERS = { EXACT: 0, PREFIX: 1, WORD_PREFIX: 2, SUBSTRING: 3 };

/** Shown when nothing matched: the phrasing is the thing people get wrong. */
export const NO_MATCH_HINT =
  "Skills use ESCO wording — try 'manage budgets' rather than 'budgeting'.";

const WORD_BREAK = /[^a-z0-9]/;

function wordPrefix(title, query) {
  return title.split(WORD_BREAK).some((word) => word && word.startsWith(query));
}

/** The tier a folded title earns for a folded query, or null for no match. */
function tierOf(title, query) {
  if (title === query) return SKILL_TIERS.EXACT;
  if (title.startsWith(query)) return SKILL_TIERS.PREFIX;
  if (wordPrefix(title, query)) return SKILL_TIERS.WORD_PREFIX;
  return title.includes(query) ? SKILL_TIERS.SUBSTRING : null;
}

function compareMatches(a, b) {
  return a.tier - b.tier
    || a.title.length - b.title.length
    || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);
}

/**
 * Rank skills for a query.
 *
 * Order: exact title, title prefix, a word inside the title, then substring.
 * Ties break by title length then alphabetically, so equal input always gives
 * the same order.
 *
 * @param {Array<{id: string, t: string}>} index skill_index.json rows
 * @param {string} query raw user input; empty or whitespace returns []
 * @param {number} [limit=12] pass Infinity for every match
 * @returns {Array<{row: Object, tier: number}>}
 */
export function rankSkills(index, query, limit = 12) {
  const folded = fold(query);
  if (!folded) return [];
  const matches = [];
  for (const row of index || []) {
    const title = fold(row.t);
    const tier = tierOf(title, folded);
    if (tier !== null) matches.push({ row, tier, title });
  }
  matches.sort(compareMatches);
  return matches.slice(0, limit).map(({ row, tier }) => ({ row, tier }));
}

/**
 * How many occupations list this skill at all.
 * @param {{ne?: number, no?: number}} row a skill_index.json row
 * @returns {number}
 */
export function usedIn(row) {
  return Number(row?.ne || 0) + Number(row?.no || 0);
}

/**
 * The occupation slugs of one skill_occupations.json entry, essential first.
 * @param {{e?: string[], o?: string[]}} entry
 * @returns {string[]} de-duplicated, order preserved
 */
export function occupationSlugs(entry) {
  return [...new Set([...(entry?.e || []), ...(entry?.o || [])])];
}

function countOverlap(entry, target) {
  let overlap = 0;
  for (const slug of occupationSlugs(entry)) {
    if (target.has(slug)) overlap += 1;
  }
  return overlap;
}

function byCountThenId(a, b) {
  return b.n - a.n || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
}

/**
 * The skills that most often turn up in the same occupations as this one.
 *
 * Counted client-side from skill_occupations.json: every other skill is scored
 * by how many of this skill's occupations it also appears in. The work is capped
 * by `maxOccupations` so a skill used in hundreds of jobs stays as cheap as one
 * used in five; the whole index is ~126,000 slug entries, a few milliseconds.
 *
 * @param {Object<string, {e?: string[], o?: string[]}>} skillOccupations
 * @param {string} id the skill to relate to
 * @param {{limit?: number, maxOccupations?: number}} [options]
 * @returns {Array<{id: string, n: number}>} most shared occupations first
 */
export function coOccurringSkills(skillOccupations, id, options = {}) {
  const { limit = 8, maxOccupations = 120 } = options;
  const target = new Set(occupationSlugs(skillOccupations?.[id]).slice(0, maxOccupations));
  if (!target.size) return [];
  const counts = [];
  for (const [other, entry] of Object.entries(skillOccupations)) {
    const overlap = other === id ? 0 : countOverlap(entry, target);
    if (overlap > 0) counts.push({ id: other, n: overlap });
  }
  counts.sort(byCountThenId);
  return counts.slice(0, limit);
}

/**
 * Resolve occupation slugs to search_index rows so titles and scores show.
 *
 * A slug with no row still comes back — as a row carrying the slug as its title
 * and no scores — so a link is never dropped silently.
 *
 * @param {string[]} slugs
 * @param {Map<string, Object>} bySlug search_index rows keyed by slug
 * @returns {Array<Object>} search_index-shaped rows
 */
export function resolveOccupations(slugs, bySlug) {
  return (slugs || []).map((slug) => bySlug.get(slug)
    || { t: slug, s: slug, c: '', mg: '', a: null, m: null, q: null });
}

/**
 * Both occupation lists of a skill, resolved and ready to render.
 * @param {{e?: string[], o?: string[]}} entry a skill_occupations.json entry
 * @param {Map<string, Object>} bySlug search_index rows keyed by slug
 * @returns {{essential: Array<Object>, optional: Array<Object>}}
 */
export function neededBy(entry, bySlug) {
  return {
    essential: resolveOccupations(entry?.e, bySlug),
    optional: resolveOccupations(entry?.o, bySlug),
  };
}

/**
 * A count with a noun that agrees with it: "1 occupation", "319 occupations".
 * @param {number} count
 * @param {string} one the singular noun
 * @param {string} many the plural noun
 * @returns {string}
 */
export function countNoun(count, one, many) {
  return `${formatCount(count)} ${count === 1 ? one : many}`;
}

/* --- the measures a skill carries ----------------------------------------- */

/**
 * The scheme of the set the page loaded.
 *
 * The stats file states it; it is fetched beside the index and may fail on its
 * own, so when it is missing the index rows settle it — only a shares set gives
 * a skill a class — rather than the page guessing from the URL or a file name.
 *
 * @param {Object|null} stats the active set's stats file, or null
 * @param {Array<Object>} rows skill_index rows
 * @returns {'quadrants'|'shares'}
 */
export function schemeOfSkillSet(stats, rows) {
  if (stats) return schemeOf(stats);
  return (rows || []).some((row) => skillClassOf(row)) ? SHARES : QUADRANTS;
}

const QUADRANT_AXES = [
  { key: 'a', axis: 'automation', label: 'Automation risk' },
  { key: 'm', axis: 'amplification', label: 'Amplification' },
];

const SHARE_AXES = [
  { key: 'a', axis: 'substituted', label: 'AI substitution' },
  { key: 'm', axis: 'assisted', label: 'AI assistance' },
  { key: 'k', axis: 'mechanised', label: 'Machine automation' },
];

/**
 * The bars the detail view draws: two under quadrants, three under shares.
 * @param {string} [scheme]
 * @returns {Array<{key: string, axis: string, label: string}>}
 */
export function scoreAxes(scheme) {
  return scheme === SHARES ? SHARE_AXES : QUADRANT_AXES;
}

// Under quadrants the site cuts at 6, so calling exactly 6.0 "medium" tells the
// reader the opposite of what the method page tells them. Under shares these
// three scores cut nothing, so a plain band is the honest word.
const SHARES_BANDS = [[7, 'high'], [4, 'medium'], [0, 'low']];

/**
 * The word beside a 1-10 score.
 * @param {number} value
 * @param {string} [scheme]
 * @returns {string} '' when the value is not a score
 */
export function scoreBand(value, scheme) {
  if (!isScored(value)) return '';
  const shown = Math.round(value * 10) / 10;
  if (scheme === SHARES) {
    return (SHARES_BANDS.find(([floor]) => shown >= floor) || [0, ''])[1];
  }
  if (shown === THRESHOLD) return 'on the cut-off';
  return shown > THRESHOLD ? 'above the cut-off' : 'below the cut-off';
}

const QUADRANT_SCORES_NOTE = 'Automation risk is how much of this skill a machine could '
  + 'take over. Amplification is how much more someone could do with AI helping. Both are '
  + 'model estimates on a 1 to 10 scale, read off the skill’s ESCO description.';

const SHARES_SCORES_NOTE = 'AI substitution is how much of this skill an AI system could '
  + 'carry out itself. AI assistance is how much better the person doing it gets with such '
  + 'a system alongside. Machine automation is how much of it physical equipment does, with '
  + 'no AI involved. All three are model estimates on a 1 to 10 scale, read off the skill’s '
  + 'ESCO description — and none of them decides the class above, which comes from the '
  + 'model’s own probabilities.';

/**
 * The note under the score bars, in the words of the active scheme.
 * @param {string} [scheme]
 * @returns {string}
 */
export function scoresNote(scheme) {
  return scheme === SHARES ? SHARES_SCORES_NOTE : QUADRANT_SCORES_NOTE;
}

const QUADRANT_SCORES_HEADING = 'Both scores';
const SHARES_SCORES_HEADING = 'What AI can do with this skill';

/**
 * The heading of the scores section.
 * @param {string} [scheme]
 * @returns {string}
 */
export function scoresHeading(scheme) {
  return scheme === SHARES ? SHARES_SCORES_HEADING : QUADRANT_SCORES_HEADING;
}

/**
 * What each skill class means, in one sentence.
 *
 * "Stays human" says out loud that it is not a finding that AI is no help:
 * about six skills in ten land there, and a reader must not read the label as
 * "AI is irrelevant to this work".
 */
export const CLASS_MEANINGS = {
  S: 'The model is more likely than not that an AI system could carry out nearly all of '
    + 'this work by itself, and that what comes out of it is data or a document.',
  A: 'The person keeps this work, and the model is more likely than not that an AI system '
    + 'gives them a clear gain across most of it.',
  M: 'The model is more likely than not that physical equipment — production lines, '
    + 'machines, robots — does nearly all of this work.',
  I: 'None of the three reached that point. That is not the same as AI being no help here: '
    + 'many skills in this class still gain clearly on part of the work.',
};

/**
 * The one-sentence meaning of a skill class.
 * @param {string} code one of S, A, M, I
 * @returns {string} '' for anything else
 */
export function classMeaning(code) {
  return CLASS_MEANINGS[code] || '';
}

// `p` is [sub, comp, mech]: the three threshold probabilities a class is read
// off, in the order docs/scoring-v2.md stores them.
const PROBABILITY_PHRASES = [
  'an AI system being able to do nearly all of this work, with a result that is data or a '
    + 'document',
  'the person keeping it and getting a clear gain across most of it with such a system '
    + 'alongside',
  'physical equipment doing nearly all of it',
];

/**
 * The three probabilities behind a skill's class, as percentages and phrases.
 * @param {number[]} probs the skill's `p`
 * @returns {Array<{percent: string, text: string}>} empty when `p` is unusable
 */
export function probabilityParts(probs) {
  if (!Array.isArray(probs) || probs.length !== 3) return [];
  if (!probs.every((value) => typeof value === 'number' && Number.isFinite(value))) return [];
  return probs.map((value, index) => ({
    percent: formatPercent(value),
    text: PROBABILITY_PHRASES[index],
  }));
}

/**
 * What the model put on each of the three questions a class is read from.
 * @param {number[]} probs the skill's `p`
 * @returns {string} '' when `p` is unusable
 */
export function probabilitySentence(probs) {
  const parts = probabilityParts(probs);
  if (!parts.length) return '';
  const listed = parts.map((part) => `${part.percent} on ${part.text}`);
  return `Behind that class: the model puts ${listed[0]}, ${listed[1]}, and ${listed[2]}. `
    + 'A skill takes the first of those three the model is more likely than not to say yes '
    + 'to. They are the model’s own estimates from the skill’s ESCO description, not '
    + 'measurements of any workplace.';
}

/* --- the occupations that need a skill ------------------------------------ */

function shareAt(row, index) {
  return isShares(row && row.sh) ? row.sh[index] : -1;
}

function numberOf(value) {
  return isScored(value) ? value : -1;
}

/** The sorts the "Needed by" lists offer under the shares scheme. */
export const SHARE_SORTS = {
  substituted: {
    label: 'Most can be taken over', descending: true, value: (row) => shareAt(row, 0),
  },
  assisted: {
    label: 'Most assisted', descending: true, value: (row) => shareAt(row, 1),
  },
  mechanical: {
    label: 'Most mechanical', descending: true, value: (row) => numberOf(row && row.k),
  },
  title: {
    label: 'A to Z', descending: false, value: (row) => String((row && row.t) ?? ''),
  },
};

const SHARE_SORT_ORDER = ['substituted', 'assisted', 'mechanical', 'title'];

/**
 * The options of the sort control, in display order.
 * @param {string} [scheme]
 * @returns {Array<{key: string, label: string}>}
 */
export function sortOptions(scheme) {
  if (scheme !== SHARES) {
    return Object.entries(SORTS).map(([key, sort]) => ({ key, label: sort.label }));
  }
  return SHARE_SORT_ORDER.map((key) => ({ key, label: SHARE_SORTS[key].label }));
}

/**
 * The sort a page starts on.
 * @param {string} [scheme]
 * @returns {string}
 */
export function defaultSort(scheme) {
  return scheme === SHARES ? 'substituted' : 'automation';
}

function compareBy(sort, a, b) {
  const left = sort.value(a);
  const right = sort.value(b);
  const order = typeof left === 'number' && typeof right === 'number'
    ? left - right
    : String(left).localeCompare(String(right));
  return order * (sort.descending ? -1 : 1);
}

/**
 * Sort the occupations that need a skill. Never mutates the input; ties break
 * by title, so the same input always gives the same order.
 *
 * @param {Array<Object>} rows search_index rows
 * @param {string} sortKey a key of SORTS or of SHARE_SORTS
 * @param {string} [scheme]
 * @returns {Array<Object>} a new array
 */
export function sortSkillOccupations(rows, sortKey, scheme) {
  if (scheme !== SHARES) return sortOccupations(rows, sortKey);
  const sort = SHARE_SORTS[sortKey] || SHARE_SORTS.substituted;
  return [...(rows || [])].sort((a, b) => compareBy(sort, a, b)
    || String(a.t ?? '').localeCompare(String(b.t ?? '')));
}

function shareParts(row) {
  const parts = sharePercents(row && row.sh);
  const type = { label: 'type ', text: typeShortLabel(row && row.q, SHARES) };
  if (!parts.length) return [type, { label: 'shares ', text: NOT_SCORED }];
  return [
    type,
    { label: 'AI can take over ', text: `${parts[0].percent}%` },
    { label: 'AI assists ', text: `${parts[1].percent}%` },
  ];
}

/**
 * What one occupation row shows beside its title: two scores under quadrants,
 * the job's type and its two telling shares under shares.
 *
 * @param {Object} row a search_index row
 * @param {string} [scheme]
 * @returns {{separator: string, parts: Array<{label: string, text: string}>}}
 *   `label` is for a visually-hidden span, so the numbers are never bare
 */
export function occupationScores(row, scheme) {
  if (scheme === SHARES) return { separator: ' · ', parts: shareParts(row) };
  return {
    separator: ' / ',
    parts: [
      { label: 'automation ', text: formatScore(row && row.a) },
      { label: 'amplification ', text: formatScore(row && row.m) },
    ],
  };
}

const QUADRANT_NEEDED_NOTE = 'Essential skills are required for the occupation; optional '
  + 'ones are common but not required. Scores are automation / amplification.';

const SHARES_NEEDED_NOTE = 'Essential skills are required for the occupation; optional ones '
  + 'are common but not required. Each job shows its type, then how much of it AI can take '
  + 'over and how much of it AI assists with.';

/**
 * The note above the two occupation lists.
 * @param {string} [scheme]
 * @returns {string}
 */
export function neededNote(scheme) {
  return scheme === SHARES ? SHARES_NEEDED_NOTE : QUADRANT_NEEDED_NOTE;
}

/**
 * The meta line beside a result in the search box: both scores under quadrants,
 * the skill's class under shares. Never a bare code.
 * @param {Object} row a skill_index row
 * @param {string} [scheme]
 * @returns {string}
 */
export function optionMeta(row, scheme) {
  if (scheme === SHARES) return skillClassName(row && row.c);
  return `${formatScore(row && row.a)} / ${formatScore(row && row.m)}`;
}

const QUADRANT_LOOKUP_INTRO = 'Every skill in the ESCO classification was scored on two '
  + 'axes: how much of it AI could do, and how much AI could amplify the person doing it. '
  + 'Search by the skill’s ESCO name.';

const SHARES_LOOKUP_INTRO = 'Every skill in the ESCO classification was asked the same '
  + 'questions: how much of it an AI system could do by itself, how much better the person '
  + 'doing it gets with one alongside, and how much of it physical equipment does. Search '
  + 'by the skill’s ESCO name.';

/**
 * The paragraph above the search box in the empty state.
 * @param {string} [scheme]
 * @returns {string}
 */
export function lookupIntro(scheme) {
  return scheme === SHARES ? SHARES_LOOKUP_INTRO : QUADRANT_LOOKUP_INTRO;
}

/* --- the table of every scored skill -------------------------------------- */

/** Rows of the all-skills table per page. Never more than one page in the DOM. */
export const PAGE_SIZE = 50;

/** The hash id the list state lives under, so a filtered view can be shared. */
export const LIST_ID = 'list';

/** How a skill is mainly exercised (`mo`), in plain words. */
export const MODE_NAMES = {
  t: 'On things',
  p: 'With people',
  d: 'Directing others',
  s: 'Through software',
  a: 'On paper, in place',
};

/** Whether a row is an activity or a body of knowledge (`ty`). */
export const ITEM_NAMES = { s: 'Skill', k: 'Knowledge item' };

const CLASS_RANK = { S: 0, A: 1, M: 2, I: 3 };

function sortableScore(value) {
  return isScored(value) ? value : -1;
}

function column(key, label, numeric, text, sort) {
  return { key, label, numeric, text, sort };
}

function scoreColumn(key, label) {
  return column(key, label, true, (row) => formatScore(row[key]),
    (row) => sortableScore(row[key]));
}

const SKILL_COLUMN = column('t', 'Skill', false,
  (row) => String(row.t ?? ''), (row) => fold(row.t));
const CLASS_COLUMN = column('c', 'What AI can do with it', false,
  (row) => skillClassName(row.c), (row) => (CLASS_RANK[row.c] ?? 9));
const MODE_COLUMN = column('mo', 'How it is exercised', false,
  (row) => MODE_NAMES[row.mo] || NOT_SCORED, (row) => MODE_NAMES[row.mo] || '~');
const ITEM_COLUMN = column('ty', 'Skill or knowledge', false,
  (row) => ITEM_NAMES[row.ty] || NOT_SCORED, (row) => ITEM_NAMES[row.ty] || '~');
const JOBS_COLUMN = column('n', 'Jobs that need it', true,
  (row) => formatCount(usedIn(row)), (row) => usedIn(row));

function carries(rows, key) {
  return (rows || []).some((row) => row && row[key]);
}

/**
 * The columns of the all-skills table.
 *
 * A quadrant set has two scores and no classes. A shares set has the class and
 * three scores, plus the mode and the skill/knowledge columns when the rows
 * actually carry them, so a set built before those fields existed shows a
 * shorter table rather than a column of "Not scored".
 *
 * @param {string} [scheme]
 * @param {Array<Object>} [rows] skill_index rows
 * @returns {Array<{key, label, numeric, text: Function, sort: Function}>}
 */
export function listColumns(scheme, rows) {
  if (scheme !== SHARES) {
    return [SKILL_COLUMN, scoreColumn('a', 'Automation risk'),
      scoreColumn('m', 'Amplification'), JOBS_COLUMN];
  }
  return [
    SKILL_COLUMN, CLASS_COLUMN,
    scoreColumn('a', 'AI substitution'), scoreColumn('m', 'AI assistance'),
    scoreColumn('k', 'Machine automation'),
    ...(carries(rows, 'mo') ? [MODE_COLUMN] : []),
    ...(carries(rows, 'ty') ? [ITEM_COLUMN] : []),
    JOBS_COLUMN,
  ];
}

/** A column sorts the way it reads: biggest first for a number, A to Z for a name. */
function naturalDirection(columns, key) {
  const found = columns.find((item) => item.key === key);
  return found && found.numeric ? 'desc' : 'asc';
}

function positiveInt(value) {
  const number = Number.parseInt(value, 10);
  return Number.isFinite(number) && number > 0 ? number : 1;
}

function oneOf(value, allowed) {
  return allowed[value] ? value : '';
}

/**
 * The list state a hash carries, with every value checked against the data.
 *
 * @param {Object<string, string>} params from parseHash
 * @param {Array<Object>} columns from listColumns
 * @returns {{query, cls, item, mode, sort, dir, page}}
 */
export function parseListState(params, columns) {
  const given = params || {};
  const keys = columns.map((item) => item.key);
  const sort = keys.includes(given.sort) ? given.sort : 't';
  return {
    query: String(given.q || ''),
    cls: oneOf(given.class, { S: 1, A: 1, M: 1, I: 1 }),
    item: oneOf(given.item, ITEM_NAMES),
    mode: oneOf(given.mode, MODE_NAMES),
    sort,
    dir: ['asc', 'desc'].includes(given.dir) ? given.dir : naturalDirection(columns, sort),
    page: positiveInt(given.page),
  };
}

/**
 * The hash parameters of a list state, with anything at its default left out so
 * a plain view stays `skill.html#list`.
 * @param {Object} state from parseListState
 * @param {Array<Object>} columns
 * @returns {Object<string, string>}
 */
export function listParams(state, columns) {
  const params = {};
  if (state.query) params.q = state.query;
  if (state.cls) params.class = state.cls;
  if (state.item) params.item = state.item;
  if (state.mode) params.mode = state.mode;
  if (state.sort !== 't') params.sort = state.sort;
  if (state.dir !== naturalDirection(columns, state.sort)) params.dir = state.dir;
  if (state.page > 1) params.page = String(state.page);
  return params;
}

/**
 * The rows a list state selects.
 * @param {Array<Object>} rows skill_index rows
 * @param {Object} state from parseListState
 * @returns {Array<Object>}
 */
export function filterSkills(rows, state) {
  const query = fold(state.query);
  return (rows || []).filter((row) => (!state.cls || row.c === state.cls)
    && (!state.item || row.ty === state.item)
    && (!state.mode || row.mo === state.mode)
    && (!query || fold(row.t).includes(query)));
}

function compareRows(sort, a, b) {
  const left = sort(a);
  const right = sort(b);
  if (typeof left === 'number' && typeof right === 'number') return left - right;
  return String(left).localeCompare(String(right));
}

/**
 * Sort by any column, either way. Ties break by title, so the order is stable.
 * @param {Array<Object>} rows
 * @param {Array<Object>} columns
 * @param {Object} state
 * @returns {Array<Object>} a new array
 */
export function sortListRows(rows, columns, state) {
  const sorted = columns.find((item) => item.key === state.sort) || columns[0];
  const factor = state.dir === 'desc' ? -1 : 1;
  return [...(rows || [])].sort((a, b) => (compareRows(sorted.sort, a, b) * factor)
    || fold(a.t).localeCompare(fold(b.t)));
}

/**
 * One page of rows, and what the page says about itself.
 * @param {Array<Object>} rows the filtered and sorted rows
 * @param {number} page 1-based
 * @param {number} [size=PAGE_SIZE]
 * @returns {{rows, page, pages, from, to, total}}
 */
export function pageOf(rows, page, size = PAGE_SIZE) {
  const all = rows || [];
  const pages = Math.max(1, Math.ceil(all.length / size));
  const current = Math.min(Math.max(1, page || 1), pages);
  const start = (current - 1) * size;
  return {
    rows: all.slice(start, start + size),
    page: current,
    pages,
    from: all.length ? start + 1 : 0,
    to: Math.min(start + size, all.length),
    total: all.length,
  };
}

/**
 * The line above the table, and what the live region announces after a filter.
 * @param {Object} view from pageOf
 * @returns {string}
 */
export function pageSummary(view) {
  if (!view.total) return 'No skill matches these filters.';
  return `Showing ${formatCount(view.from)}–${formatCount(view.to)} of `
    + `${countNoun(view.total, 'skill', 'skills')}`;
}

/**
 * The options of one filter, counted over the rows, in the order the names are
 * declared. Values nothing carries are left out rather than offered as an
 * option that empties the table.
 *
 * @param {Array<Object>} rows
 * @param {string} key the row field, such as 'c' or 'mo'
 * @param {Object<string, string>} names code -> plain name
 * @returns {Array<{value: string, label: string}>} the "all" option first
 */
export function filterOptions(rows, key, names) {
  const counts = {};
  for (const row of rows || []) {
    if (names[row[key]]) counts[row[key]] = (counts[row[key]] || 0) + 1;
  }
  const options = Object.keys(names)
    .filter((code) => counts[code])
    .map((code) => ({ value: code, label: `${names[code]} (${formatCount(counts[code])})` }));
  return [{ value: '', label: `All (${formatCount((rows || []).length)})` }, ...options];
}

/** The class filter's options, named as the site names the four classes. */
export function classFilterOptions(rows) {
  const names = {};
  for (const code of SKILL_CLASS_ORDER) names[code] = skillClassName(code);
  return filterOptions(rows, 'c', names);
}

/* --- how a skill is scored ------------------------------------------------ */

function choicesOf(source) {
  if (Array.isArray(source && source.options)) {
    return source.options.map((option) => ({ name: option.name, text: option.text }));
  }
  return ((source && source.levels) || []).map((text) => ({ name: null, text }));
}

/**
 * The six questions as the page shows them: one shape whether the rubric wrote
 * them as ordered levels or as named choices, with the knowledge-item variant
 * beside the ordinary one where the rubric has one.
 *
 * @param {Object} rubric parsed rubric_v2.json
 * @returns {Array<{id, label, kind, instructions, choices, knowledge}>}
 */
export function rubricQuestions(rubric) {
  return ((rubric && rubric.questions) || []).map((question) => ({
    id: question.id,
    label: question.label || question.id,
    kind: question.kind,
    instructions: question.instructions || '',
    choices: choicesOf(question),
    knowledge: question.knowledge
      ? {
        instructions: question.knowledge.instructions || '',
        choices: choicesOf(question.knowledge),
      }
      : null,
  }));
}

// The rubric states each rule as an identifier and a number ("SUB >= 0.5").
// That is the right form for the repository and the wrong one for a reader, so
// the page says the same thing in words. The codes and their order still come
// from the file, so a rule the pipeline adds cannot be missed here.
const CLASS_RULES_IN_WORDS = {
  S: 'when the model puts the chance that an AI system could do nearly all of this work '
    + 'above one in two',
  A: 'when that chance is not reached, and the model puts the chance of a clear gain across '
    + 'most of the work above one in two',
  M: 'when neither of those is reached, and the model puts the chance that machinery does '
    + 'nearly all of it above one in two',
  I: 'when none of the three chances reaches one in two',
};

/**
 * The four class rules: the name the site uses, the rule in plain words, and
 * what the rule means for the skill.
 *
 * The sentence for "Stays human" is the one in CLASS_MEANINGS, which says out
 * loud that the class is not a finding that AI is of no help.
 *
 * @param {Object} rubric parsed rubric_v2.json
 * @returns {Array<{code, siteName, rule, meaning}>}
 */
export function rubricClasses(rubric) {
  return ((rubric && rubric.classes) || []).map((entry) => ({
    code: entry.code,
    siteName: skillClassName(entry.code),
    rule: CLASS_RULES_IN_WORDS[entry.code] || '',
    meaning: classMeaning(entry.code),
  }));
}

/**
 * The paragraph above the questions. Everything the rubric wrote is quoted, not
 * retyped; the count comes from the stats file.
 *
 * @param {Object} rubric parsed rubric_v2.json
 * @param {number} [scored] stats.skills_scored
 * @returns {{lead: string, preamble: string, display: string}}
 */
export function rubricIntro(rubric, scored) {
  const model = (rubric && rubric.model) || 'a judgment model';
  const count = isScored(scored) ? `${formatCount(scored)} skills` : 'every skill';
  return {
    lead: `Each of the ${count} went to ${model} on its own, as six questions. The model `
      + 'answers with a probability for every level rather than picking one, so what comes '
      + 'back is how sure it is, not a verdict.',
    preamble: (rubric && rubric.preamble) || '',
    display: `The 1 to 10 scores on this page are the answers put on a scale — `
      + `${(rubric && rubric.display) || ''} — where position is the probability-weighted `
      + 'level. No class depends on them.',
  };
}

/* --- how the model answered one skill ------------------------------------- */

/** Which field of an answers shard each question was stored in. */
const ANSWER_KEYS = {
  digital_output: 'd',
  ai_substitution: 's',
  mechanical: 'k',
  complementarity: 'c',
  mode: 'mo',
  deployment: 'dp',
};

/** Which confidence a question has; digital output was not asked for one. */
const CONFIDENCE_KEYS = {
  ai_substitution: 's', mechanical: 'k', complementarity: 'c', mode: 'mo', deployment: 'dp',
};

/**
 * The shard a skill's answers and its rationale live in: the first two
 * characters of its id. `skill_answers_v2/<xx>.json`, `skill_notes/<xx>*.json`.
 * @param {string} id
 * @returns {string|null}
 */
export function shardOf(id) {
  const text = String(id || '');
  return text.length >= 2 ? text.slice(0, 2).toLowerCase() : null;
}

function share(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function marked(bars) {
  const best = bars.reduce((top, bar) => (bar.share > top ? bar.share : top), -1);
  let taken = false;
  return bars.map((bar) => {
    const modal = !taken && bar.share === best && best > 0;
    taken = taken || modal;
    return { ...bar, modal, percent: formatPercent(bar.share) };
  });
}

function yesNoBars(question, value) {
  if (typeof value !== 'number') return [];
  const yes = share(value);
  const named = question.choices.map((choice, index) => ({
    label: choice.text,
    share: choice.name === 'false' || (choice.name === null && index === 0) ? 1 - yes : yes,
  }));
  return marked(named.length ? named : [{ label: 'Yes', share: yes }]);
}

function choiceBars(question, value) {
  if (!value || typeof value !== 'object') return [];
  return marked(question.choices.map((choice) => ({
    label: choice.text, share: share(value[choice.name]),
  })));
}

function levelBars(choices, value) {
  if (!Array.isArray(value)) return [];
  return marked(choices.map((choice, index) => ({
    label: choice.text, share: share(value[index]),
  })));
}

const CONFIDENCE_WORDS = [[0.8, 'high'], [0.6, 'moderate'], [0, 'low']];

/**
 * How sure the model was, in a word.
 * @param {number} value
 * @returns {string} '' when there is no usable number
 */
export function confidenceWord(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';
  return (CONFIDENCE_WORDS.find(([floor]) => value >= floor) || [0, ''])[1];
}

function confidenceOf(question, entry) {
  const key = CONFIDENCE_KEYS[question.id];
  const value = key && entry.cf && entry.cf[key];
  if (typeof value !== 'number') return null;
  return { share: value, percent: formatPercent(value), word: confidenceWord(value) };
}

function barsFor(question, entry, knowledge) {
  const value = entry[ANSWER_KEYS[question.id]];
  if (question.kind === 'yesno') return yesNoBars(question, value);
  if (question.kind === 'choice') return choiceBars(question, value);
  return levelBars(knowledge ? question.knowledge.choices : question.choices, value);
}

/**
 * What the model answered for one skill, question by question.
 *
 * @param {Object} rubric parsed rubric_v2.json
 * @param {Object} entry the skill's entry of its answers shard
 * @returns {Array<{id, label, knowledge, confidence, bars}>} empty without either
 */
export function answerSections(rubric, entry) {
  if (!entry || typeof entry !== 'object') return [];
  return rubricQuestions(rubric).map((question) => {
    const knowledge = entry.ty === 'k' && Boolean(question.knowledge);
    return {
      id: question.id,
      label: question.label,
      knowledge,
      confidence: confidenceOf(question, entry),
      bars: barsFor(question, entry, knowledge),
    };
  }).filter((section) => section.bars.length);
}

/** True when the skill was asked the knowledge-item wording of a question. */
export function askedAsKnowledge(entry) {
  return Boolean(entry && entry.ty === 'k');
}
