// Everything job.html decides before it touches the DOM. Pure: no DOM, no
// globals, unit-tested with `node --test tests/js/job-model.test.js`.
//
// It works on the shapes portfolio_data.json already has:
//   occupation { t, s, c, cat, mg, q, e, ar, ap, se[], so[], adj[], n{} }
//   skill      { t, a, m, r }            keyed by 8-character id
//   adjacency  { s, t, ov, e, q, gap[] }
//
// Two rules the brief insists on live here:
//   - the quadrant cut-off is quadrant.js's THRESHOLD (6), never 5.5;
//   - an adjacent occupation is only called a move when the two scores
//     genuinely separate it from this job. The audit found eight "paths" for
//     software developer that all sat within 0.5 of it.

import { formatCount, isScored, NOT_SCORED } from '../format.js';
import { THRESHOLD, quadrantOf } from '../quadrant.js';
import { groupHref } from '../urlstate.js';

/** A move has to beat this margin on a score to count as a real difference. */
export const MOVE_MARGIN = 0.5;

/** How many gap skills the page offers at once. */
export const LEARN_LIMIT = 8;

/** How many slugs "recently viewed" keeps. */
export const RECENT_LIMIT = 5;

/** localStorage key for recently viewed slugs, most recent first. */
export const RECENT_KEY = 'ai-isco-recent';

/**
 * Index the occupations by slug once, so adjacency lookups are not linear.
 * @param {{occupations?: Array<Object>}} data parsed portfolio_data.json
 * @returns {Map<string, Object>}
 */
export function occupationIndex(data) {
  const rows = (data && data.occupations) || [];
  return new Map(rows.map((row) => [row.s, row]));
}

/**
 * The occupation a hash slug names.
 * @param {{occupations?: Array<Object>}} data
 * @param {string} slug
 * @returns {Object|null} null when the slug is not in the data
 */
export function findOccupation(data, slug) {
  if (!slug) return null;
  const rows = (data && data.occupations) || [];
  return rows.find((row) => row.s === slug) || null;
}

/**
 * Resolve skill ids into the objects the page shows.
 * @param {Object<string, Object>} skills the `skills` map of portfolio_data
 * @param {Array<string>} ids
 * @param {boolean} [essential=false] marks the skill as required by the job
 * @returns {Array<{id, title, auto, amp, rationale, rationaleFrom, essential}>}
 *   unknown ids dropped; `rationaleFrom` is set when another model wrote the rationale
 */
export function resolveSkills(skills, ids, essential = false) {
  const map = skills || {};
  return (ids || [])
    .map((id) => [id, map[id]])
    .filter(([, skill]) => Boolean(skill))
    .map(([id, skill]) => ({
      id,
      title: skill.t,
      auto: isScored(skill.a) ? skill.a : null,
      amp: isScored(skill.m) ? skill.m : null,
      rationale: skill.r || null,
      rationaleFrom: skill.rf || null,
      essential,
    }));
}

/**
 * Every skill of one occupation, essential first.
 * @param {{skills?: Object}} data
 * @param {{se?: Array<string>, so?: Array<string>}} occupation
 * @returns {Array<Object>}
 */
export function occupationSkills(data, occupation) {
  const skills = (data && data.skills) || {};
  return [
    ...resolveSkills(skills, occupation && occupation.se, true),
    ...resolveSkills(skills, occupation && occupation.so, false),
  ];
}

/** Highest tension first: the skills where both scores are large. */
function byTension(a, b) {
  return ((b.auto || 0) * (b.amp || 0)) - ((a.auto || 0) * (a.amp || 0));
}

/**
 * The essential skills, ordered for the rationale cards.
 * @param {Object} data
 * @param {Object} occupation
 * @returns {Array<Object>}
 */
export function essentialSkills(data, occupation) {
  const skills = (data && data.skills) || {};
  return resolveSkills(skills, occupation && occupation.se, true).sort(byTension);
}

/**
 * Split skills into the ones automation reaches and the ones AI amplifies. The
 * cut-off is the model's threshold, so the lists agree with the scatter lines.
 * @param {Array<Object>} skills
 * @returns {{depreciating: Array<Object>, appreciating: Array<Object>}}
 */
export function splitSkills(skills) {
  const rows = skills || [];
  return {
    depreciating: rows
      .filter((skill) => isScored(skill.auto) && skill.auto >= THRESHOLD)
      .sort((a, b) => b.auto - a.auto),
    appreciating: rows
      .filter((skill) => isScored(skill.amp) && skill.amp >= THRESHOLD)
      .sort((a, b) => b.amp - a.amp),
  };
}

/**
 * The dots of the scatter plot: every skill that carries both scores.
 * @param {Array<Object>} skills
 * @returns {Array<{id, title, auto, amp, essential, q}>}
 */
export function scatterPoints(skills) {
  return (skills || [])
    .filter((skill) => isScored(skill.auto) && isScored(skill.amp))
    .map((skill) => ({ ...skill, q: quadrantOf(skill.auto, skill.amp) }));
}

function countAbove(points, key) {
  return points.filter((point) => point[key] >= THRESHOLD).length;
}

function plural(count, noun) {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}

/**
 * The aria-label that says out loud what the canvas draws.
 * @param {string} title the occupation title
 * @param {Array<Object>} points from scatterPoints
 * @returns {string}
 */
export function scatterSummary(title, points) {
  const rows = points || [];
  if (!rows.length) return `No skill in ${title} carries both scores, so the plot is empty.`;
  const essential = rows.filter((point) => point.essential).length;
  return `Scatter plot of ${plural(rows.length, 'skill')} in ${title}. `
    + `Automation risk 1 to 10 runs left to right, amplification 1 to 10 runs bottom to `
    + `top, and both cut-offs are drawn at ${THRESHOLD}. `
    + `${formatCount(countAbove(rows, 'auto'))} skills reach ${THRESHOLD} on automation risk `
    + `and ${formatCount(countAbove(rows, 'amp'))} reach it on amplification. `
    + `${formatCount(essential)} are essential to the job and `
    + `${formatCount(rows.length - essential)} optional. The table below lists them all.`;
}

/** ISCO codes nest by digit: 2512 sits in 251, in 25, in 2. */
const GROUP_LEVELS = [['unit', 4], ['minor', 3], ['sub', 2], ['major', 1]];

/**
 * The unit group key of an ISCO code, or null when there is no four-digit code.
 * @param {string} code
 * @returns {string|null}
 */
export function unitGroupKey(code) {
  const digits = String(code || '').replace(/\D/g, '');
  return digits.length >= 4 ? `unit:${digits.slice(0, 4)}` : null;
}

/**
 * The group keys an occupation belongs to, narrowest first, ending at "all".
 * @param {string} code ISCO code
 * @returns {Array<string>}
 */
export function groupChain(code) {
  const digits = String(code || '').replace(/\D/g, '');
  const keys = GROUP_LEVELS
    .filter(([, length]) => digits.length >= length)
    .map(([level, length]) => `${level}:${digits.slice(0, length)}`);
  return [...keys, 'all'];
}

function groupBackLink(key, group, text) {
  return { key, href: groupHref(key), label: group.label, count: group.n, text };
}

function allJobsText(key, group) {
  return key === 'all' ? '← All jobs' : `← All jobs in ${group.label}`;
}

/**
 * Where the back link above the title points, and what it says.
 *
 * `from=` wins, because it is the trail the visitor actually walked. Without it
 * the occupation's own ISCO code gives a group, so a cold deep link is never a
 * dead end.
 *
 * @param {string|null} fromKey the `from=` hash parameter
 * @param {Object} occupation
 * @param {Object<string, {label: string, n: number}>} groups groups.json
 * @returns {{key, href, label, count, text}}
 */
export function backTarget(fromKey, occupation, groups) {
  const table = groups || {};
  const named = fromKey && table[fromKey];
  if (named) {
    return groupBackLink(fromKey, named,
      `← Back to ${named.label} (${plural(named.n, 'job')})`);
  }
  for (const key of groupChain(occupation && occupation.c)) {
    if (table[key]) return groupBackLink(key, table[key], allJobsText(key, table[key]));
  }
  return {
    key: 'all', href: groupHref('all'), label: 'all occupations', count: null,
    text: '← All jobs',
  };
}

function bothScored(scores) {
  return Boolean(scores) && isScored(scores.a) && isScored(scores.m);
}

function moveFlags(from, to, margin) {
  return {
    lessAutomated: to.a <= from.a - margin,
    moreAutomated: to.a >= from.a + margin,
    moreAmplified: to.m >= from.m + margin,
    lessAmplified: to.m <= from.m - margin,
  };
}

// Read top to bottom: the first rule that fits wins. Only "better" is a move.
const MOVE_RULES = [
  { kind: 'better', label: 'Less exposed and more amplified',
    fits: (f) => f.lessAutomated && f.moreAmplified },
  { kind: 'better', label: 'Less exposed to automation',
    fits: (f) => f.lessAutomated && !f.lessAmplified },
  { kind: 'better', label: 'More amplified by AI',
    fits: (f) => f.moreAmplified && !f.moreAutomated },
  { kind: 'trade', label: 'More amplified, but more exposed to automation',
    fits: (f) => f.moreAmplified && f.moreAutomated },
  { kind: 'trade', label: 'Less exposed, but less amplified',
    fits: (f) => f.lessAutomated && f.lessAmplified },
  { kind: 'exposed', label: 'More exposed to automation than this job',
    fits: (f) => f.moreAutomated || f.lessAmplified },
];

const SIDEWAYS = {
  kind: 'sideways',
  label: `A sideways move: within ${MOVE_MARGIN} on both scores`,
};

/**
 * How one occupation compares with another on the two scores.
 * @param {{a: number, m: number}} from the job being read
 * @param {{a: number, m: number}} to the adjacent job
 * @param {number} [margin=MOVE_MARGIN] below which a difference is noise
 * @returns {{kind: 'better'|'trade'|'exposed'|'sideways'|'unknown', label: string}}
 */
export function compareMove(from, to, margin = MOVE_MARGIN) {
  if (!bothScored(from) || !bothScored(to)) {
    return { kind: 'unknown', label: NOT_SCORED };
  }
  const flags = moveFlags(from, to, margin);
  const rule = MOVE_RULES.find((candidate) => candidate.fits(flags)) || SIDEWAYS;
  return { kind: rule.kind, label: rule.label };
}

function adjacencyCard(adjacency, index, from, margin) {
  const target = index.get(adjacency.s) || null;
  const scores = { a: target && target.ar, m: target && target.ap };
  const move = compareMove(from, scores, margin);
  return {
    slug: adjacency.s,
    title: adjacency.t || (target && target.t) || adjacency.s,
    auto: isScored(scores.a) ? scores.a : null,
    amp: isScored(scores.m) ? scores.m : null,
    q: adjacency.q || (target && target.q) || null,
    overlap: typeof adjacency.ov === 'number' ? adjacency.ov : null,
    gapCount: (adjacency.gap || []).length,
    kind: move.kind,
    label: move.label,
  };
}

const KIND_ORDER = { better: 0, trade: 1, sideways: 2, exposed: 3, unknown: 4 };

function compareCards(a, b) {
  const byKind = KIND_ORDER[a.kind] - KIND_ORDER[b.kind];
  if (byKind !== 0) return byKind;
  return (b.amp || 0) - (a.amp || 0) || (a.auto || 0) - (b.auto || 0);
}

/**
 * The adjacent occupations, split into the ones worth calling a move and the
 * ones that are honestly sideways or more exposed.
 *
 * @param {Object} occupation
 * @param {Map<string, Object>} index from occupationIndex
 * @param {number} [margin=MOVE_MARGIN]
 * @returns {{paths: Array<Object>, others: Array<Object>, total: number}}
 */
export function evolutionPaths(occupation, index, margin = MOVE_MARGIN) {
  const from = { a: occupation && occupation.ar, m: occupation && occupation.ap };
  const cards = ((occupation && occupation.adj) || [])
    .map((adjacency) => adjacencyCard(adjacency, index, from, margin))
    .sort(compareCards);
  return {
    paths: cards.filter((card) => card.kind === 'better'),
    others: cards.filter((card) => card.kind !== 'better'),
    total: cards.length,
  };
}

function gapCandidate(id, skills, adjacency) {
  const skill = skills[id];
  if (!skill || !isScored(skill.m)) return null;
  return {
    id,
    title: skill.t,
    amp: skill.m,
    auto: isScored(skill.a) ? skill.a : null,
    fromTitle: adjacency.t,
    fromSlug: adjacency.s,
  };
}

function gapIdsOf(occupation) {
  const own = new Set([...(occupation.se || []), ...(occupation.so || [])]);
  const wanted = ((occupation.adj || [])
    .filter((adjacency) => adjacency.q === 'TRANSFORM' || adjacency.q === 'EVOLVE'));
  return wanted.flatMap((adjacency) => (adjacency.gap || [])
    .filter((id) => !own.has(id))
    .map((id) => ({ id, adjacency })));
}

/**
 * The skills a nearby job needs that this one does not list, most amplified
 * first. Unscored skills are left out rather than shown as a blank.
 *
 * @param {Object} data parsed portfolio_data.json
 * @param {Object} occupation
 * @param {number} [limit=LEARN_LIMIT]
 * @returns {Array<{id, title, amp, auto, fromTitle, fromSlug}>}
 */
export function gapSkills(data, occupation, limit = LEARN_LIMIT) {
  const skills = (data && data.skills) || {};
  const seen = new Set();
  const found = [];
  for (const { id, adjacency } of gapIdsOf(occupation || {})) {
    if (seen.has(id)) continue;
    seen.add(id);
    const candidate = gapCandidate(id, skills, adjacency);
    if (candidate) found.push(candidate);
  }
  return found.sort((a, b) => b.amp - a.amp).slice(0, limit);
}

/**
 * Recently viewed slugs: most recent first, no duplicates, capped.
 * @param {Array<string>} list what localStorage held
 * @param {string} slug the slug just opened
 * @param {number} [limit=RECENT_LIMIT]
 * @returns {Array<string>} a new array
 */
export function addRecent(list, slug, limit = RECENT_LIMIT) {
  const kept = (Array.isArray(list) ? list : [])
    .filter((item) => typeof item === 'string' && item && item !== slug);
  return (slug ? [slug, ...kept] : kept).slice(0, limit);
}
