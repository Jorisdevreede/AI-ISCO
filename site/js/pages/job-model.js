// Everything job.html decides before it touches the DOM. Pure: no DOM, no
// globals, unit-tested with `node --test tests/js/job-model.test.js`.
//
// It works on one unit group's slice of the portfolio, `jobs/<unit><suffix>.json`,
// which has the record shapes the whole portfolio file has:
//   occupation { t, s, c, cat, mg, q, e, ar, ap, se[], so[], adj[], n{} }
//              and, in a shares set, also { ak, sh[4], nl, why }
//   skill      { t, a, m, r }  and, in a shares set, also { k, c, p[3] }
//   adjacency  { s, t, ov, e, q, gap[] }
// plus `neighbours`: the full records of the adjacency targets that live in
// another unit group, so a path card never needs a second file. `jobs/units.json`
// maps every slug to its unit group, which is how the page knows what to fetch.
// Nothing here loads the 15 MB portfolio file any more.
//
// Two rules the brief insists on live here:
//   - the quadrant cut-off is quadrant.js's THRESHOLD (6), never 5.5;
//   - an adjacent occupation is only called a move when the scores genuinely
//     separate it from this job. The audit found eight "paths" for software
//     developer that all sat within 0.5 of it.
//
// Every decision that differs between the two schemes is one small function per
// scheme behind a dispatcher, so neither the page nor a test has to fork.

import { formatCount, formatPercent, isScored, NOT_SCORED } from '../format.js';
import { THRESHOLD, quadrantOf } from '../quadrant.js';
import {
  QUADRANTS, SHARES, isSkillNear, schemeOf, schemeOfCode, skillClassName, skillClassOf,
} from '../scheme.js';
import { isShares, largestShare, sharePercents } from '../shares.js';
import { groupHref } from '../urlstate.js';

/** A move has to beat this margin on a score to count as a real difference. */
export const MOVE_MARGIN = 0.5;

/**
 * Under the shares scheme a move has to beat this much of a share instead: a
 * tenth of the job's weight, which is a fifth of the narrowest type rule.
 */
export const SHARE_MARGIN = 0.10;

/**
 * A job with at least this much of its weight in "AI can take over" leads with
 * a next step rather than ending on one. It is the substituted share the
 * "Transforming" rule asks for, so the page and the type table agree.
 */
export const LARGE_SUBSTITUTED_SHARE = 0.30;

/** How many gap skills the page offers at once. */
export const LEARN_LIMIT = 8;

/** How many slugs "recently viewed" keeps. */
export const RECENT_LIMIT = 5;

/** localStorage key for recently viewed slugs, most recent first. */
export const RECENT_KEY = 'ai-isco-recent';

/**
 * The scheme of the set the page loaded.
 *
 * The stats file is the source: it is the one file that states the scheme. It
 * is loaded as an optional extra, though, so when it is missing the occupations
 * themselves settle it — their `q` codes belong to one scheme or the other —
 * rather than the page guessing from the URL or a file name.
 *
 * @param {Object|null} stats the active set's stats file, or null
 * @param {Object} data parsed portfolio_data.json
 * @returns {'quadrants'|'shares'}
 */
export function schemeOfSet(stats, data) {
  if (stats) return schemeOf(stats);
  const rows = (data && data.occupations) || [];
  return rows.some((row) => schemeOfCode(row.q) === SHARES) ? SHARES : QUADRANTS;
}

/** Where the slug-to-unit map lives. Scorer-independent: one file for both sets. */
export const UNITS_FILE = 'jobs/units.json';

/**
 * The file holding one job's unit group, or null when the slug is unknown.
 * @param {Object<string, string>} units parsed jobs/units.json
 * @param {string} slug
 * @returns {string|null} a name for loadJSON, without ".json"
 */
export function unitFile(units, slug) {
  const unit = units && slug ? units[slug] : null;
  return unit ? `jobs/${unit}` : null;
}

function allRecords(data) {
  return [...((data && data.occupations) || []), ...((data && data.neighbours) || [])];
}

/**
 * Index a unit group's occupations by slug, neighbours included, so an
 * adjacency lookup is neither linear nor a second request.
 * @param {{occupations?: Array<Object>, neighbours?: Array<Object>}} data
 * @returns {Map<string, Object>}
 */
export function occupationIndex(data) {
  return new Map(allRecords(data).map((row) => [row.s, row]));
}

/**
 * The occupation a hash slug names. Its own unit group first; a neighbour
 * record answers a slug the shard carries only as a path target.
 * @param {{occupations?: Array<Object>, neighbours?: Array<Object>}} data
 * @param {string} slug
 * @returns {Object|null} null when the slug is not in this shard
 */
export function findOccupation(data, slug) {
  if (!slug) return null;
  return allRecords(data).find((row) => row.s === slug) || null;
}

function skillFields(id, skill, essential) {
  return {
    id,
    title: skill.t,
    auto: isScored(skill.a) ? skill.a : null,
    amp: isScored(skill.m) ? skill.m : null,
    mech: isScored(skill.k) ? skill.k : null,
    cls: skillClassOf(skill),
    probs: Array.isArray(skill.p) ? skill.p : null,
    rationale: skill.r || null,
    rationaleFrom: skill.rf || null,
    essential,
  };
}

/**
 * Resolve skill ids into the objects the page shows.
 * @param {Object<string, Object>} skills the `skills` map of portfolio_data
 * @param {Array<string>} ids
 * @param {boolean} [essential=false] marks the skill as required by the job
 * @returns {Array<Object>} unknown ids dropped; `mech`, `cls` and `probs` are
 *   null in a set that does not carry them, `rationaleFrom` is set when another
 *   model wrote the rationale
 */
export function resolveSkills(skills, ids, essential = false) {
  const map = skills || {};
  return (ids || [])
    .map((id) => [id, map[id]])
    .filter(([, skill]) => Boolean(skill))
    .map(([id, skill]) => skillFields(id, skill, essential));
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
 * Quadrant sets only: a shares set classifies a skill, it does not cut it.
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

// C1. A class comes from three probabilities compared with one cut; the page
// used to print the class and the three 1-10 display scores and never the
// probability, so a skill could read "Stays human" at 6.2 five rows under one
// reading "AI can take over" at 6.7. `p` is [substitution, assistance, machinery].
const CHANCE_PHRASES = [
  'chance AI could do nearly all of it',
  'clear gain with AI alongside',
  'machinery does it',
];

/**
 * The three chances behind a skill's class, as percentages and short phrases.
 * @param {number[]} probs the skill's `p`
 * @returns {Array<{percent: string, text: string}>} empty when `p` is unusable
 */
export function decidingChances(probs) {
  if (!Array.isArray(probs) || probs.length !== 3) return [];
  if (!probs.every((value) => typeof value === 'number' && Number.isFinite(value))) return [];
  return probs.map((value, index) => ({
    percent: formatPercent(value),
    text: CHANCE_PHRASES[index],
  }));
}

/**
 * The same three chances as one line, for a row that also shows a class.
 * @param {number[]} probs
 * @returns {string} '' when `p` is unusable
 */
export function chanceLine(probs) {
  return decidingChances(probs)
    .map((part) => `${part.percent} ${part.text}`)
    .join(' · ');
}

/**
 * Does one of the comparisons that settled this skill's class sit within a
 * whisker of the cut? Adapts a resolved skill to the shared row shape.
 *
 * @param {{cls?: string, probs?: number[]}} skill from resolveSkills
 * @param {number} [cut] the class cut from the stats file
 * @returns {boolean}
 */
export function skillNear(skill, cut) {
  return isSkillNear({ c: skill && skill.cls, p: skill && skill.probs }, cut);
}

/** Which of the three scores leads a row of each class. */
const CLASS_PRIMARY = { S: 'auto', A: 'amp', M: 'mech', I: 'amp' };

/**
 * The skills of a job grouped by the class the model gave each one, every group
 * led by the score that class is about. A skill with no class is left out.
 * @param {Array<Object>} skills from occupationSkills
 * @returns {{S: Array<Object>, A: Array<Object>, M: Array<Object>, I: Array<Object>}}
 */
export function classSkillLists(skills) {
  const lists = { S: [], A: [], M: [], I: [] };
  for (const skill of skills || []) {
    if (lists[skill.cls]) lists[skill.cls].push(skill);
  }
  for (const [code, list] of Object.entries(lists)) {
    const key = CLASS_PRIMARY[code];
    list.sort((a, b) => (b[key] || 0) - (a[key] || 0));
  }
  return lists;
}

function plural(count, noun) {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}

const QUADRANT_LISTS = [
  { key: 'dep', headingKey: 'depreciatingHeading', primary: 'auto', noun: 'automation risk' },
  { key: 'app', headingKey: 'appreciatingHeading', primary: 'amp', noun: 'amplification' },
];

function quadrantNote(count, total, noun) {
  if (!count) return `No skill in this job reaches ${THRESHOLD} for ${noun}.`;
  return `${formatCount(count)} of the ${plural(total, 'skill')} in this job score `
    + `${THRESHOLD} or more for ${noun}. Both scores are shown, ${noun} first.`;
}

function quadrantSpecs(skills) {
  const split = splitSkills(skills);
  const rows = { dep: split.depreciating, app: split.appreciating };
  return QUADRANT_LISTS.map((spec) => ({
    key: spec.key,
    code: null,
    headingKey: spec.headingKey,
    primary: spec.primary,
    skills: rows[spec.key],
    note: quadrantNote(rows[spec.key].length, skills.length, spec.noun),
  }));
}

const CLASS_LISTS = [
  {
    code: 'S', headingKey: 'classSHeading', lead: 'AI substitution',
    phrase: 'ones AI can take over', none: 'No skill in this job is one AI can take over.',
  },
  {
    code: 'A', headingKey: 'classAHeading', lead: 'AI assistance',
    phrase: 'ones AI assists with', none: 'No skill in this job is one AI assists with.',
  },
  {
    code: 'M', headingKey: 'classMHeading', lead: 'Machine automation',
    phrase: 'ones machines can do', none: 'No skill in this job is one machines can do.',
  },
];

function classNote(count, total, spec) {
  if (!count) return spec.none;
  return `${formatCount(count)} of the ${plural(total, 'skill')} in this job are `
    + `${spec.phrase}. All three scores are shown, ${spec.lead} first.`;
}

function classSpecs(skills) {
  const lists = classSkillLists(skills);
  return CLASS_LISTS
    .filter((spec) => spec.code !== 'M' || lists.M.length > 0)
    .map((spec) => ({
      key: `class-${spec.code}`,
      code: spec.code,
      headingKey: spec.headingKey,
      primary: CLASS_PRIMARY[spec.code],
      skills: lists[spec.code],
      note: classNote(lists[spec.code].length, skills.length, spec),
    }));
}

/**
 * The skill lists the page shows, in order: two cut at the threshold under
 * quadrants, one per class that has members under shares. Each spec names the
 * wording key of its heading and which score leads a row.
 *
 * @param {Array<Object>} skills from occupationSkills
 * @param {string} [scheme] 'quadrants' (default) or 'shares'
 * @returns {Array<{key, code, headingKey, primary, skills, note}>}
 */
export function skillListSpecs(skills, scheme) {
  const rows = skills || [];
  return scheme === SHARES ? classSpecs(rows) : quadrantSpecs(rows);
}

/**
 * What the page says about the skills no list above covers.
 * @param {Array<Object>} skills from occupationSkills
 * @returns {string} '' when every skill is already in a list
 */
export function staysHumanNote(skills) {
  const rows = skills || [];
  const count = classSkillLists(rows).I.length;
  if (!count) return '';
  return `The other ${formatCount(count)} of the ${plural(rows.length, 'skill')} in this `
    + 'job stay human: neither an AI system nor machinery reaches most of that work. That is '
    + 'not the same as AI being no help — many of these skills still gain clearly on part of '
    + 'the work.';
}

/** How many skills each rebuilt task list names. */
export const TASK_LIMIT = 5;

const TASK_LISTS = [
  {
    code: 'S', key: 'takes', index: 0, heading: 'What AI could take on',
    note: 'The skills in this job the model is most confident an AI system could carry '
      + 'out almost entirely by itself.',
  },
  {
    code: 'A', key: 'amplifies', index: 1, heading: 'What AI amplifies',
    note: 'The skills the person keeps, where the model is most confident an AI system '
      + 'makes them clearly better at the work.',
  },
];

function taskItems(skills, spec) {
  return skills
    .filter((skill) => skill.cls === spec.code && Array.isArray(skill.probs))
    .sort((a, b) => b.probs[spec.index] - a.probs[spec.index])
    .slice(0, TASK_LIMIT)
    .map((skill) => ({
      id: skill.id,
      title: skill.title,
      percent: formatPercent(skill.probs[spec.index]),
    }));
}

/**
 * What AI takes on and what it amplifies, read off this job's own classes.
 *
 * The Gemini narrative named tasks its own run had scored, so the list could
 * say "generating technical drawings" one scroll above a row classing
 * `technical drawings` as one that stays human. These are the same skills the
 * classes came from, so the two cannot disagree.
 *
 * @param {Array<Object>} skills from occupationSkills
 * @returns {Array<{key, code, heading, note, items}>} lists with no members dropped
 */
export function classTaskLists(skills) {
  return TASK_LISTS
    .map((spec) => ({ ...spec, items: taskItems(skills || [], spec) }))
    .filter((list) => list.items.length);
}

/**
 * The dots of the scatter plot: every skill that carries both scores.
 * @param {Array<Object>} skills
 * @param {string} [scheme] under shares a dot is coloured by class, not by box
 * @returns {Array<Object>} each with `q` (quadrants) or `cls` (shares)
 */
export function scatterPoints(skills, scheme) {
  return (skills || [])
    .filter((skill) => isScored(skill.auto) && isScored(skill.amp))
    .map((skill) => ({
      ...skill,
      q: scheme === SHARES ? null : quadrantOf(skill.auto, skill.amp),
    }));
}

function countAbove(points, key) {
  return points.filter((point) => point[key] >= THRESHOLD).length;
}

function classCounts(points) {
  const counts = { S: 0, A: 0, M: 0, I: 0 };
  for (const point of points) {
    if (counts[point.cls] !== undefined) counts[point.cls] += 1;
  }
  return counts;
}

function quadrantSummary(title, rows) {
  const essential = rows.filter((point) => point.essential).length;
  return `Scatter plot of ${plural(rows.length, 'skill')} in ${title}. `
    + `Automation risk 1 to 10 runs left to right, amplification 1 to 10 runs bottom to `
    + `top, and both cut-offs are drawn at ${THRESHOLD}. `
    + `${formatCount(countAbove(rows, 'auto'))} skills reach ${THRESHOLD} on automation risk `
    + `and ${formatCount(countAbove(rows, 'amp'))} reach it on amplification. `
    + `${formatCount(essential)} are essential to the job and `
    + `${formatCount(rows.length - essential)} optional. The table below lists them all.`;
}

function sharesSummary(title, rows) {
  const counts = classCounts(rows);
  const essential = rows.filter((point) => point.essential).length;
  return `Scatter plot of ${plural(rows.length, 'skill')} in ${title}. `
    + 'AI substitution 1 to 10 runs left to right and AI assistance 1 to 10 runs bottom to '
    + 'top. No lines are drawn across it: a skill’s class comes from the model’s own '
    + 'probabilities rather than from these two numbers. '
    + `${formatCount(counts.S)} of the skills are ones AI can take over, `
    + `${formatCount(counts.A)} ones AI assists with, ${formatCount(counts.M)} ones machines `
    + `can do and ${formatCount(counts.I)} stay human. ${formatCount(essential)} are essential `
    + `to the job and ${formatCount(rows.length - essential)} optional. The table below lists `
    + 'every one of them with its class and all three scores.';
}

/**
 * The aria-label that says out loud what the canvas draws.
 * @param {string} title the occupation title
 * @param {Array<Object>} points from scatterPoints
 * @param {string} [scheme]
 * @returns {string}
 */
export function scatterSummary(title, points, scheme) {
  const rows = points || [];
  if (!rows.length) return `No skill in ${title} carries both scores, so the plot is empty.`;
  return scheme === SHARES ? sharesSummary(title, rows) : quadrantSummary(title, rows);
}

const QUADRANT_COLUMNS = [
  { label: 'Skill', numeric: false },
  { label: 'Automation risk (out of 10)', numeric: true },
  { label: 'Amplification (out of 10)', numeric: true },
  { label: 'In this job', numeric: false },
];

const SHARE_COLUMNS = [
  { label: 'Skill', numeric: false },
  { label: 'What AI can do with it', numeric: false },
  { label: 'AI substitution (out of 10)', numeric: true },
  { label: 'AI assistance (out of 10)', numeric: true },
  { label: 'Machine automation (out of 10)', numeric: true },
  { label: 'In this job', numeric: false },
];

/**
 * The columns of the table that stands in for the plot. The first one is always
 * the skill itself, which the page renders as a link.
 * @param {string} [scheme]
 * @returns {Array<{label: string, numeric: boolean}>}
 */
export function scatterColumns(scheme) {
  return scheme === SHARES ? SHARE_COLUMNS : QUADRANT_COLUMNS;
}

/**
 * The cells of one table row, after the skill's own name.
 * @param {Object} point from scatterPoints
 * @param {string} [scheme]
 * @returns {Array<{value: number|string, numeric: boolean}>} `value` is a number
 *   for a score cell, so the page can format it, and a string otherwise
 */
export function scatterCells(point, scheme) {
  const place = { value: point.essential ? 'Essential' : 'Optional', numeric: false };
  if (scheme !== SHARES) {
    return [
      { value: point.auto, numeric: true }, { value: point.amp, numeric: true }, place,
    ];
  }
  return [
    { value: skillClassName(point.cls), numeric: false },
    { value: point.auto, numeric: true },
    { value: point.amp, numeric: true },
    { value: point.mech, numeric: true },
    place,
  ];
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

function shareFlags(from, to, margin) {
  return {
    lessSubstituted: to.sh[0] <= from.sh[0] - margin,
    moreSubstituted: to.sh[0] >= from.sh[0] + margin,
    moreAssisted: to.sh[1] >= from.sh[1] + margin,
    lessAssisted: to.sh[1] <= from.sh[1] - margin,
  };
}

// The same ladder under the shares scheme, read off the two shares that say
// what happens to the work. Nothing here calls a sideways move a step up.
const SHARE_RULES = [
  { kind: 'better', label: 'Less of it can be taken over, and more of it is assisted',
    fits: (f) => f.lessSubstituted && f.moreAssisted },
  { kind: 'better', label: 'Less of it can be taken over by AI',
    fits: (f) => f.lessSubstituted && !f.lessAssisted },
  { kind: 'better', label: 'More of it is work AI assists with',
    fits: (f) => f.moreAssisted && !f.moreSubstituted },
  { kind: 'trade', label: 'More of it is assisted, but more can be taken over too',
    fits: (f) => f.moreAssisted && f.moreSubstituted },
  { kind: 'trade', label: 'Less can be taken over, but less of it is assisted',
    fits: (f) => f.lessSubstituted && f.lessAssisted },
  { kind: 'exposed', label: 'More of it can be taken over than in this job',
    fits: (f) => f.moreSubstituted || f.lessAssisted },
];

const SHARE_SIDEWAYS = {
  kind: 'sideways',
  label: 'A sideways move: the same work, split about the same way',
};

/**
 * How one occupation compares with another on its four shares.
 *
 * Only the substituted and assisted shares decide, because they are the two
 * that say what happens to the work; a difference under SHARE_MARGIN is noise.
 *
 * @param {{sh: number[]}} from the job being read
 * @param {{sh: number[]}} to the adjacent job
 * @param {number} [margin=SHARE_MARGIN]
 * @returns {{kind: 'better'|'trade'|'exposed'|'sideways'|'unknown', label: string}}
 */
export function compareShareMove(from, to, margin = SHARE_MARGIN) {
  if (!isShares(from && from.sh) || !isShares(to && to.sh)) {
    return { kind: 'unknown', label: NOT_SCORED };
  }
  const flags = shareFlags(from, to, margin);
  const rule = SHARE_RULES.find((candidate) => candidate.fits(flags)) || SHARE_SIDEWAYS;
  return { kind: rule.kind, label: rule.label };
}

function moveOf(from, scores, context) {
  return context.scheme === SHARES
    ? compareShareMove(from, scores, context.margin)
    : compareMove(from, scores, context.margin);
}

function cardScores(target) {
  return {
    auto: isScored(target && target.ar) ? target.ar : null,
    amp: isScored(target && target.ap) ? target.ap : null,
    mech: isScored(target && target.ak) ? target.ak : null,
    sh: isShares(target && target.sh) ? target.sh : null,
    nl: Boolean(target && target.nl),
  };
}

function adjacencyCard(adjacency, index, context) {
  const target = index.get(adjacency.s) || null;
  const scores = cardScores(target);
  const move = moveOf(context.from, { a: scores.auto, m: scores.amp, sh: scores.sh }, context);
  return {
    ...scores,
    slug: adjacency.s,
    title: adjacency.t || (target && target.t) || adjacency.s,
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

function moveContext(occupation, options) {
  const scheme = options.scheme === SHARES ? SHARES : undefined;
  const fallback = scheme === SHARES ? SHARE_MARGIN : MOVE_MARGIN;
  return {
    scheme,
    margin: options.margin === undefined ? fallback : options.margin,
    from: {
      a: occupation && occupation.ar,
      m: occupation && occupation.ap,
      sh: occupation && occupation.sh,
    },
  };
}

/**
 * The adjacent occupations, split into the ones worth calling a move and the
 * ones that are honestly sideways or more exposed.
 *
 * @param {Object} occupation
 * @param {Map<string, Object>} index from occupationIndex
 * @param {{scheme?: string, margin?: number}} [options]
 * @returns {{paths: Array<Object>, others: Array<Object>, total: number}}
 */
export function evolutionPaths(occupation, index, options = {}) {
  const context = moveContext(occupation, options);
  const cards = ((occupation && occupation.adj) || [])
    .map((adjacency) => adjacencyCard(adjacency, index, context))
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
  const auto = isScored(skill.a) ? skill.a : null;
  return {
    id,
    title: skill.t,
    amp: skill.m,
    auto,
    gain: Math.round((skill.m - (auto || 0)) * 100) / 100,
    mech: isScored(skill.k) ? skill.k : null,
    cls: skillClassOf(skill),
    fromTitle: adjacency.t,
    fromSlug: adjacency.s,
  };
}

function gapIdsOf(occupation) {
  const own = new Set([...(occupation.se || []), ...(occupation.so || [])]);
  return (occupation.adj || []).flatMap((adjacency) => (adjacency.gap || [])
    .filter((id) => !own.has(id))
    .map((id) => ({ id, adjacency })));
}

/**
 * Is this skill worth sending a person towards?
 *
 * A skill AI can take over is not: learning it buys the person the thing the
 * page has just said is being taken over. Shares read the class the model gave;
 * quadrants have only the score, cut where the rest of that scheme cuts.
 */
function worthLearning(skill, scheme) {
  if (scheme === SHARES) return skill.cls !== 'S';
  return !(isScored(skill.auto) && skill.auto >= THRESHOLD);
}

function byNetGain(a, b) {
  return (b.gain - a.gain) || (b.amp - a.amp);
}

/**
 * Keep the list from being one neighbour's wish list: when every pick came from
 * the same job and another job has a candidate, give it the last place.
 */
function spreadSources(sorted, limit) {
  const picked = sorted.slice(0, limit);
  if (picked.length < limit) return picked;
  const sources = new Set(picked.map((skill) => skill.fromSlug));
  if (sources.size > 1) return picked;
  const other = sorted.slice(limit).find((skill) => !sources.has(skill.fromSlug));
  return other ? [...picked.slice(0, limit - 1), other] : picked;
}

/**
 * The skills a nearby job needs that this one does not list, best first.
 *
 * Ranked by net gain (how much better AI makes the person at it, less how much
 * of it AI can take over), with the skills AI can take over left out and at
 * least two source jobs in the list where the data allows one. Ranking by
 * amplification alone put "use word processing software" at the top of a
 * software developer's page, all of it from one neighbour.
 *
 * @param {Object} data one unit group's slice of the portfolio
 * @param {Object} occupation
 * @param {{scheme?: string, limit?: number}} [options]
 * @returns {Array<{id, title, amp, auto, gain, mech, cls, fromTitle, fromSlug}>}
 */
export function gapSkills(data, occupation, options = {}) {
  const { scheme, limit = LEARN_LIMIT } = options;
  const skills = (data && data.skills) || {};
  const seen = new Set();
  const found = [];
  for (const { id, adjacency } of gapIdsOf(occupation || {})) {
    if (seen.has(id)) continue;
    seen.add(id);
    const candidate = gapCandidate(id, skills, adjacency);
    if (candidate && worthLearning(candidate, scheme)) found.push(candidate);
  }
  return spreadSources(found.sort(byNetGain), limit);
}

const QUADRANT_SCORE_CARDS = [
  {
    key: 'ar', modifier: 'auto', label: 'Automation risk', unit: 'out of 10',
    hint: 'How much of the work a machine could take over, read off the skills this job '
      + 'is built from.',
  },
  {
    key: 'ap', modifier: 'amp', label: 'Amplification', unit: 'out of 10',
    hint: 'How much more someone in this job could get done with AI helping, read off the '
      + 'same skills.',
  },
];

// C6. "Machines can do 0%" and "Machine automation 1.9" sat 200 px apart: the
// same words, two scales, no bridge. The card now says what it averages, so it
// cannot be read as the share above it.
const SHARE_SCORE_CARDS = [
  {
    key: 'ar', modifier: 'sub', unit: '1 to 10',
    label: 'Average AI-substitution score across this job’s skills',
    hint: 'A per-skill score averaged over the job. It is not the share above: a skill '
      + 'counts towards this whatever class it ended up in.',
  },
  {
    key: 'ap', modifier: 'assist', unit: '1 to 10',
    label: 'Average AI-assistance score across this job’s skills',
    hint: 'How much better the person gets at the work with an AI system alongside, '
      + 'averaged the same way.',
  },
  {
    key: 'ak', modifier: 'mech', unit: '1 to 10',
    label: 'Average machine-automation score across this job’s skills',
    hint: 'How much of the work physical equipment does — production lines, machines, '
      + 'robots — averaged the same way.',
  },
];

/**
 * The secondary score cards, and which field of the occupation each one reads.
 * @param {string} [scheme]
 * @returns {Array<{key: string, modifier: string, label: string, hint: string}>}
 */
export function scoreCards(scheme) {
  return scheme === SHARES ? SHARE_SCORE_CARDS : QUADRANT_SCORE_CARDS;
}

const SHARE_MEANINGS = {
  S: (percent) => `${percent}% of the work this job is built from is work an AI system `
    + 'could carry out almost entirely by itself.',
  A: (percent) => `${percent}% stays with the person and gets a clear gain across most of `
    + 'that work with an AI system alongside.',
  M: (percent) => `${percent}% is work physical equipment does, with no AI involved.`,
  I: (percent) => `${percent}% is work that neither AI nor machinery gets most of the way `
    + 'through. AI can still help with parts of it.',
};

// C6. The first screen names one quantity: the largest share, in a sentence.
// "Most of this job" is only said when the share really is most of it.
const LARGEST_PHRASES = {
  S: 'work an AI system could carry out almost entirely by itself',
  A: 'work that stays with the person and gets clearly better with an AI system alongside',
  M: 'work physical equipment does, with no AI involved',
  I: 'work that neither AI nor machinery gets most of the way through',
};

/**
 * The one sentence the first screen shows under the shares scheme.
 * @param {number[]} shares the occupation's `sh`
 * @returns {string} '' when the shares are not usable
 */
export function largestShareSentence(shares) {
  const part = largestShare(shares);
  if (!part || !LARGEST_PHRASES[part.code]) return '';
  const lead = part.percent >= 50 ? 'Most of this job' : 'The largest part of this job';
  return `${lead} — ${part.percent}% — is ${LARGEST_PHRASES[part.code]}.`;
}

const WHY_TAILS = {
  physical: ' Most of that is work done on things, in a place.',
  people: ' Most of that is work done with and for other people.',
};

/**
 * One plain sentence per share, using the job's own percentages.
 *
 * @param {number[]} shares the occupation's `sh`
 * @param {string} [why] the occupation's `why_insulated`
 * @returns {Array<{code: string, name: string, percent: number, text: string}>}
 *   an empty array when the shares are not usable
 */
export function shareMeanings(shares, why) {
  return sharePercents(shares).map((part) => {
    const tail = part.code === 'I' && part.percent > 0 ? (WHY_TAILS[why] || '') : '';
    return {
      code: part.code,
      name: skillClassName(part.code),
      percent: part.percent,
      text: SHARE_MEANINGS[part.code](part.percent) + tail,
    };
  });
}

/**
 * Whether the next step goes above the detail or below it. A job with a large
 * part of its work in "AI can take over" always leads with it.
 *
 * @param {Object} occupation
 * @param {string} [scheme]
 * @returns {'top'|'bottom'}
 */
export function advicePosition(occupation, scheme) {
  if (scheme === SHARES) {
    const shares = occupation && occupation.sh;
    return isShares(shares) && shares[0] >= LARGE_SUBSTITUTED_SHARE ? 'top' : 'bottom';
  }
  const score = occupation && occupation.ar;
  return isScored(score) && score >= THRESHOLD ? 'top' : 'bottom';
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
