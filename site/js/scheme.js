// Which classification the active score set uses, and the names that go with it.
// Pure: no DOM, no globals.
//
// Two schemes exist side by side:
//   'quadrants'  two 1-10 scores cut at 6 -> four boxes (site/js/quadrant.js)
//   'shares'     four shares per occupation -> seven types (docs/scoring-v2.md)
//
// A page reads `scheme` from the stats file of the set it loaded and passes it
// down. Where a code is unambiguous — no quadrant code is also a type code — the
// scheme argument may be left out and the code resolves itself, so a call site
// that only needs a label needs no plumbing.
//
// Copy rule, inherited from quadrant.js: every number in an explanation is
// computed from the job's own data. Nothing here may quote another scoring run,
// and no wording calls a person's job "at risk".

import { formatShare, isScored } from './format.js';
import { explainQuadrant, isNearLine, QUADRANT_NAMES } from './quadrant.js';
import { shareSentence, sharePercents } from './shares.js';

/** The two schemes a stats file can declare. */
export const QUADRANTS = 'quadrants';
export const SHARES = 'shares';

/** Type codes in the order the spec's table lists them. */
export const TYPE_ORDER = [
  'AUTOMATION_HEAVY',
  'TRANSFORMING',
  'AUGMENTED',
  'MECHANISABLE',
  'INSULATED_PHYSICAL',
  'INSULATED_PEOPLE',
  'MIXED',
];

/** Full names, as the spec's type table writes them. */
export const TYPE_NAMES = {
  AUTOMATION_HEAVY: 'Automation-heavy',
  TRANSFORMING: 'Transforming',
  AUGMENTED: 'Augmented',
  MECHANISABLE: 'Mechanisable',
  INSULATED_PHYSICAL: 'Insulated by physical work',
  INSULATED_PEOPLE: 'Insulated by work with people',
  MIXED: 'Mixed',
};

/** Short names, for a badge, a table cell or a bar segment. */
export const TYPE_SHORT = {
  AUTOMATION_HEAVY: 'Automation-heavy',
  TRANSFORMING: 'Transforming',
  AUGMENTED: 'Augmented',
  MECHANISABLE: 'Mechanisable',
  INSULATED_PHYSICAL: 'Physical work',
  INSULATED_PEOPLE: 'People work',
  MIXED: 'Mixed',
};

/** The custom property in css/app.css that colours each type. */
export const TYPE_COLOR_VARS = {
  AUTOMATION_HEAVY: '--type-automation-heavy',
  TRANSFORMING: '--type-transforming',
  AUGMENTED: '--type-augmented',
  MECHANISABLE: '--type-mechanisable',
  INSULATED_PHYSICAL: '--type-insulated-physical',
  INSULATED_PEOPLE: '--type-insulated-people',
  MIXED: '--type-mixed',
};

/** One sentence per type: what it means for the job. Never "at risk". */
export const TYPE_DESCRIPTIONS = {
  AUTOMATION_HEAVY: 'Most of the skills this job is built from are ones an AI system '
    + 'could carry out itself, and they point the same way.',
  TRANSFORMING: 'A large part of the work can be taken over and another large part gets '
    + 'better with AI alongside, so the job carries on in a different shape.',
  AUGMENTED: 'The work stays with the person and gets faster or better with AI alongside.',
  MECHANISABLE: 'Much of this work is done by physical equipment rather than by software.',
  INSULATED_PHYSICAL: 'Most of the work happens in a place, with hands and equipment, '
    + 'which a system with no body cannot do.',
  INSULATED_PEOPLE: 'Most of the work is done with and for other people, which a system '
    + 'cannot stand in for.',
  MIXED: 'The skills in this job point in different directions, so no single label '
    + 'describes it.',
};

/**
 * The rule each type matched, in plain words, in the order the pipeline tries
 * them. Kept in step with the type table of docs/scoring-v2.md: rule 1 asks for
 * the substituted share alone, since the full run showed a spread condition on
 * top of it to be ill-posed.
 */
export const TYPE_RULES = {
  AUTOMATION_HEAVY: 'Half or more of the skill weight is work AI can take over.',
  TRANSFORMING: 'At least three in ten of the skill weight is work AI can take over, and at '
    + 'least two in ten is work AI assists.',
  AUGMENTED: 'At least three in ten is work AI assists, and fewer than three in ten is work '
    + 'AI can take over.',
  MECHANISABLE: 'At least three in ten is work machines can do, and fewer than three in ten '
    + 'is work AI can take over.',
  INSULATED_PHYSICAL: 'At least four in ten stays human, and most of that is exercised on '
    + 'things, in a place.',
  INSULATED_PEOPLE: 'At least four in ten stays human, and most of that is exercised with or '
    + 'for other people.',
  MIXED: 'No rule above matched, so the skills point in different directions.',
};

/** Skill class codes, in the order the shares are stored (`sh`, `p`). */
export const SKILL_CLASS_ORDER = ['S', 'A', 'M', 'I'];

/** What the site calls each skill class. */
export const SKILL_CLASS_NAMES = {
  S: 'AI can take over',
  A: 'AI assists',
  M: 'Machines can do',
  I: 'Stays human',
};

/** The custom property in css/app.css that colours each skill class. */
export const SKILL_CLASS_COLOR_VARS = {
  S: '--class-substituted',
  A: '--class-assisted',
  M: '--class-mechanised',
  I: '--class-insulated',
};

/**
 * The scheme a stats file declares. A file without the field is an older set.
 * @param {{scheme?: string}} stats
 * @returns {'quadrants'|'shares'}
 */
export function schemeOf(stats) {
  return stats && stats.scheme === SHARES ? SHARES : QUADRANTS;
}

/**
 * The scheme a bare class code belongs to, or null when it belongs to neither.
 * The two code sets are disjoint, so a label never needs a scheme argument.
 * @param {string} code
 * @returns {'quadrants'|'shares'|null}
 */
export function schemeOfCode(code) {
  if (TYPE_NAMES[code]) return SHARES;
  if (QUADRANT_NAMES[code]) return QUADRANTS;
  return null;
}

/**
 * The class codes of a scheme, in display order.
 * @param {string} [scheme]
 * @returns {string[]}
 */
export function orderOf(scheme) {
  return scheme === SHARES ? [...TYPE_ORDER] : ['TRANSFORM', 'STABLE', 'EVOLVE', 'SHRINK'];
}

/**
 * The order a counts object is keyed by, read off the codes it actually holds.
 * Lets a bar or a tally follow the data instead of a four-box constant.
 * @param {Object<string, number>} counts
 * @returns {string[]}
 */
export function orderForCounts(counts) {
  const keys = Object.keys(counts || {});
  return orderOf(keys.some((code) => TYPE_NAMES[code]) ? SHARES : QUADRANTS);
}

/**
 * The split of all occupations, from whichever field the stats file carries.
 * @param {Object} stats the active set's stats file
 * @returns {{scheme: string, order: string[], counts: Object, shares: Object}}
 */
export function splitOf(stats) {
  const scheme = schemeOf(stats);
  const source = (scheme === SHARES ? stats && stats.types : stats && stats.quadrants) || {};
  return {
    scheme,
    order: Array.isArray(source.order) ? source.order : orderOf(scheme),
    counts: source.counts || {},
    shares: source.shares || {},
  };
}

/**
 * The cut-off the set used, or null where the scheme has none.
 * @param {Object} stats
 * @returns {number|null}
 */
export function thresholdOf(stats) {
  const value = stats && stats.threshold;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The rule a type matched, in plain words.
 * @param {string} code
 * @returns {string} '' for a code that is not a type
 */
export function typeRule(code) {
  return TYPE_RULES[code] || '';
}

function nameIn(code, scheme, typeNames) {
  if (scheme === QUADRANTS) return QUADRANT_NAMES[code] || null;
  if (scheme === SHARES) return typeNames[code] || null;
  return typeNames[code] || QUADRANT_NAMES[code] || null;
}

/**
 * The full name of a quadrant or a type. "Not scored" for a missing code.
 * @param {string} code
 * @param {string} [scheme] omit to resolve from the code itself
 * @returns {string}
 */
export function typeLabel(code, scheme) {
  return nameIn(code, scheme, TYPE_NAMES) || 'Not scored';
}

/**
 * The short name, for a badge or a narrow table cell.
 * @param {string} code
 * @param {string} [scheme] omit to resolve from the code itself
 * @returns {string}
 */
export function typeShortLabel(code, scheme) {
  return nameIn(code, scheme, TYPE_SHORT) || 'Not scored';
}

/**
 * The CSS custom property that colours a quadrant or a type.
 * @param {string} code
 * @param {string} [scheme] omit to resolve from the code itself
 * @returns {string|null}
 */
export function colorVarOf(code, scheme) {
  const quadrants = {
    TRANSFORM: '--q-transform',
    SHRINK: '--q-shrink',
    EVOLVE: '--q-evolve',
    STABLE: '--q-stable',
  };
  if (scheme === QUADRANTS) return quadrants[code] || null;
  if (scheme === SHARES) return TYPE_COLOR_VARS[code] || null;
  return TYPE_COLOR_VARS[code] || quadrants[code] || null;
}

/**
 * The one-sentence description of a type.
 * @param {string} code
 * @returns {string} '' for a code that is not a type
 */
export function typeDescription(code) {
  return TYPE_DESCRIPTIONS[code] || '';
}

/**
 * The name of a skill class.
 * @param {string} code one of S, A, M, I
 * @returns {string}
 */
export function skillClassName(code) {
  return SKILL_CLASS_NAMES[code] || 'Not scored';
}

/**
 * The CSS custom property that colours a skill class.
 * @param {string} code
 * @returns {string|null}
 */
export function skillClassColorVar(code) {
  return SKILL_CLASS_COLOR_VARS[code] || null;
}

/**
 * The class of one skill row. The quadrant scheme has no classes.
 * @param {{c?: string}} row a skill_index or portfolio skill row
 * @param {string} [scheme]
 * @returns {'S'|'A'|'M'|'I'|null}
 */
export function skillClassOf(row, scheme) {
  if (scheme === QUADRANTS || !row) return null;
  const code = row.c;
  return SKILL_CLASS_NAMES[code] ? code : null;
}

/** How close to the cut a deciding probability must sit for a skill to be "near the line". */
export const SKILL_NEAR_MARGIN = 0.05;

// The probabilities that were compared with the cut before a class was settled,
// as indexes into `p` = [substitution, assistance, machinery]. First match wins,
// so "AI assists" was reached only after substitution was turned down, and so on.
const DECIDING = { S: [0], A: [0, 1], M: [0, 1, 2], I: [0, 1, 2] };

/**
 * Does one of the comparisons that settled this skill's class sit within
 * SKILL_NEAR_MARGIN of the cut? The class itself is read from `c`, never
 * recomputed: `p` is rounded for publication and the pipeline used the full value.
 *
 * @param {{c?: string, p?: number[]}} row a skill_index or job-shard skill row
 * @param {number} [cut] the class cut, `thresholdOf(stats)`; 0.5 when omitted
 * @returns {boolean} false under the quadrant scheme or when `p` is missing
 */
export function isSkillNear(row, cut = 0.5) {
  const deciding = DECIDING[row && row.c];
  if (!deciding || !Array.isArray(row.p)) return false;
  return deciding.some((index) => Math.abs(row.p[index] - cut) <= SKILL_NEAR_MARGIN + 1e-9);
}

/** What a near-the-line skill says about itself, for a chip's title or a note. */
export const SKILL_NEAR_NOTE = 'Near the line: one of the chances that decide this class is '
  + 'within 5 points of one in two, so a small change would give it a different class.';

// --- explaining a type ---------------------------------------------------------

const RULE_SENTENCES = {
  AUTOMATION_HEAVY: (p) => `AI can take over covers ${p.S} of this job's skills, which is `
    + 'half or more: the rule for "Automation-heavy".',
  TRANSFORMING: (p) => `AI can take over covers ${p.S} of this job's skills and AI assists `
    + `another ${p.A}, which is the rule for "Transforming".`,
  AUGMENTED: (p) => `AI assists covers ${p.A} of this job's skills while AI can take over `
    + `stays at ${p.S}, which is the rule for "Augmented".`,
  MECHANISABLE: (p) => `Machines can do covers ${p.M} of this job's skills while AI can `
    + `take over stays at ${p.S}, which is the rule for "Mechanisable".`,
  INSULATED_PHYSICAL: (p) => `${p.I} of this job's skills stay human, and most of those are `
    + 'exercised on things in a place, which is the rule for "Insulated by physical work".',
  INSULATED_PEOPLE: (p) => `${p.I} of this job's skills stay human, and most of those are `
    + 'exercised with or for other people, which is the rule for "Insulated by work with people".',
  MIXED: (p) => `No rule matched: AI can take over reaches ${p.S}, AI assists ${p.A}, `
    + `machines ${p.M} and ${p.I} stays human, so this job lands in "Mixed".`,
};

const ESTIMATE_SENTENCE = 'These are model estimates. Nobody measured a real job, and '
  + 'there is no ground truth here.';

const MIXED_SENTENCE = 'Mixed is a residual class, not a finding that AI will leave the '
  + 'job alone: the skills point in different directions and one label would mislead.';

function percentsByCode(shares) {
  const map = {};
  for (const part of sharePercents(shares)) map[part.code] = `${part.percent}%`;
  return map;
}

function nearSentence(job) {
  const name = job.t || 'This job';
  if (job.nl) {
    return `${name} sits near a cut-off: moving any one of those four shares by 5 points `
      + 'would give it a different type.';
  }
  return `No share is within 5 points of a cut-off, so ${job.t ? 'this type' : 'the type'} `
    + 'does not hang on a rounding.';
}

function unscoredExplanation() {
  return {
    heading: 'Not scored',
    sentences: ['This occupation has no shares in the current run, so it has no type.'],
    nearLine: false,
  };
}

/**
 * The sentences the type badge shows, built only from the job's own data.
 *
 * @param {{t?: string, q?: string, sh?: number[], nl?: boolean}} job
 * @returns {{heading: string, sentences: string[], nearLine: boolean}}
 */
export function explainType(job) {
  const rule = job && RULE_SENTENCES[job.q];
  if (!rule || !Array.isArray(job.sh) || job.sh.length !== 4) return unscoredExplanation();
  const sentences = [
    rule(percentsByCode(job.sh)),
    `Across this job's skills the split is ${shareSentence(job.sh)}.`,
    nearSentence(job),
    ESTIMATE_SENTENCE,
  ];
  if (job.q === 'MIXED') sentences.push(MIXED_SENTENCE);
  return { heading: `Why "${TYPE_NAMES[job.q]}"?`, sentences, nearLine: Boolean(job.nl) };
}

/**
 * The one permanent caveat beside a split, in the words of the active scheme.
 *
 * Says only what the published data shows: how many occupations sit close
 * enough to a cut-off that they could fall the other side of it. The quadrant
 * wording is unchanged; the shares wording names the rule that scheme uses.
 *
 * @param {Object} stats the active set's stats file
 * @returns {string}
 */
export function nearLineCaveat(stats) {
  const near = stats && stats.near_line;
  if (!near || !isScored(near.count)) {
    return 'Jobs sitting near a cut-off can fall either side of it, so read this split '
      + 'as a band, not a count.';
  }
  const share = formatShare(near.count, stats.occupations, 'jobs');
  if (schemeOf(stats) === SHARES) {
    return `${share} sit near a cut-off, where moving any one of their four shares by 5 `
      + 'points would give them a different type: read this split as a band, not a count.';
  }
  return `${share} sit within 0.5 of a cut-off, and a small change in the scores moves `
    + 'those into another box: read this split as a band, not a count.';
}

/**
 * One entry point for both schemes: the wording behind a badge.
 * @param {Object} job an index row or portfolio occupation
 * @param {string} [scheme] omit to resolve from `job.q`
 * @returns {{heading: string, sentences: string[], nearLine: boolean}}
 */
export function explain(job, scheme) {
  const resolved = scheme || schemeOfCode(job && job.q) || QUADRANTS;
  return resolved === SHARES ? explainType(job) : explainQuadrant(job);
}

/**
 * Is this job near the line its scheme draws?
 * Quadrants use the distance rule; shares read the pipeline's own `nl`.
 * @param {Object} job
 * @param {string} [scheme] omit to resolve from `job.q`
 * @returns {boolean}
 */
export function isNear(job, scheme) {
  const resolved = scheme || schemeOfCode(job && job.q) || QUADRANTS;
  if (resolved === SHARES) return Boolean(job && job.nl);
  return isNearLine(job && job.a, job && job.m);
}
