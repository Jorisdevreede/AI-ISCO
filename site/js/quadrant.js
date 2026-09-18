// Quadrant rules and the honest wording that goes with them. Pure: no DOM.
//
// The cut-off is a hard 6 on both axes, the same number aiisco/rollup.py uses.
// Nothing here may state or imply a result from a second scoring run: the only
// uncertainty the site is allowed to quote is what the published data shows on
// its own (model estimates, a hard cut, and how close this job sits to it).

import { formatScore, isScored } from './format.js';

/** The cut-off on both axes. Mirrors aiisco/rollup.py QUADRANT_THRESHOLD. */
export const THRESHOLD = 6;

/** A job within this of the cut-off on either axis is "near the line". */
export const NEAR_LINE = 0.5;

/** Display names, keyed by the code stored in the data. */
export const QUADRANT_NAMES = {
  TRANSFORM: 'Transform',
  SHRINK: 'Shrink',
  EVOLVE: 'Evolve',
  STABLE: 'Stable',
};

/**
 * The quadrant for a pair of scores.
 * @param {number} automation
 * @param {number} amplification
 * @returns {'TRANSFORM'|'SHRINK'|'EVOLVE'|'STABLE'|null} null when unscored
 */
export function quadrantOf(automation, amplification) {
  if (!isScored(automation) || !isScored(amplification)) return null;
  if (automation >= THRESHOLD) {
    return amplification >= THRESHOLD ? 'TRANSFORM' : 'SHRINK';
  }
  return amplification >= THRESHOLD ? 'EVOLVE' : 'STABLE';
}

/**
 * How far the closer of the two scores sits from the cut-off.
 *
 * Rounded to two decimals: the scores carry one, and 6.9 - 6 in binary floating
 * point is 0.9000000000000004, which no caller should have to format around.
 *
 * @param {number} automation
 * @param {number} amplification
 * @returns {number|null} always >= 0
 */
export function distanceToLine(automation, amplification) {
  if (!isScored(automation) || !isScored(amplification)) return null;
  const gap = Math.min(
    Math.abs(automation - THRESHOLD),
    Math.abs(amplification - THRESHOLD),
  );
  return Math.round(gap * 100) / 100;
}

/**
 * True when either score is within 0.5 of the cut-off.
 * @param {number} automation
 * @param {number} amplification
 * @returns {boolean}
 */
export function isNearLine(automation, amplification) {
  const distance = distanceToLine(automation, amplification);
  return distance !== null && distance <= NEAR_LINE;
}

/**
 * "low" under 4, "medium" under 7, "high" from 7 up.
 * @param {number} score
 * @returns {'low'|'medium'|'high'|null}
 */
export function exposureWord(score) {
  if (!isScored(score)) return null;
  if (score < 4) return 'low';
  if (score < 7) return 'medium';
  return 'high';
}

function sideWord(score) {
  return score >= THRESHOLD ? 'above' : 'below';
}

function placementSentence(job) {
  const above = [];
  if (job.a >= THRESHOLD) above.push('automation risk');
  if (job.m >= THRESHOLD) above.push('amplification');
  return `Automation risk ${formatScore(job.a)} sits ${sideWord(job.a)} the cut-off of `
    + `${THRESHOLD}, and amplification ${formatScore(job.m)} sits ${sideWord(job.m)} it`
    + `${above.length === 2 ? ' too' : ''}.`;
}

function distanceSentence(job) {
  const gap = distanceToLine(job.a, job.m);
  const name = job.t || 'This job';
  if (gap <= NEAR_LINE) {
    return `${name} is only ${gap.toFixed(1)} from the cut-off, so it is near the line: `
      + 'a small change in the scores would move it into a different box.';
  }
  return `${name} is ${gap.toFixed(1)} from the nearest cut-off.`;
}

/**
 * The sentences the badge popover shows.
 *
 * Copy rule: model estimates, a hard cut at 6, and how far this job sits from
 * the line. Never a claim about any other scoring run.
 *
 * @param {{t?: string, a: number, m: number, q?: string}} job
 * @returns {{heading: string, sentences: string[], nearLine: boolean}}
 */
export function explainQuadrant(job) {
  const code = job.q || quadrantOf(job.a, job.m);
  const name = QUADRANT_NAMES[code] || 'Not scored';
  if (!isScored(job.a) || !isScored(job.m)) {
    return {
      heading: 'Not scored',
      sentences: ['This occupation has no scores in the current run, so it has no box.'],
      nearLine: false,
    };
  }
  return {
    heading: `Why "${name}"?`,
    sentences: [
      placementSentence(job),
      distanceSentence(job),
      'These are model estimates. Nobody measured a real job, and there is no ground '
        + 'truth here.',
      `The boxes are a hard cut at ${THRESHOLD}, so a job at 5.9 and a job at 6.1 get `
        + 'different labels. Treat the box as a direction, not a verdict.',
    ],
    nearLine: isNearLine(job.a, job.m),
  };
}
