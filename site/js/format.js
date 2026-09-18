// Number and score formatting. Pure: no DOM, no globals.
//
// One place decides what a missing score looks like, so no page ever prints a
// bare "?" (the audit found 80% of skill rows doing exactly that).

/** Shown wherever a score is null, undefined or not a finite number. */
export const NOT_SCORED = 'Not scored';

/**
 * One decimal, or "Not scored".
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function formatScore(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return NOT_SCORED;
  return value.toFixed(1);
}

/**
 * True when a score is missing, so callers can style it as absent.
 * @param {*} value
 * @returns {boolean}
 */
export function isScored(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * A count with thousands separators: 3043 -> "3,043".
 * @param {number} value
 * @returns {string}
 */
export function formatCount(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return NOT_SCORED;
  return Math.round(value).toLocaleString('en-GB');
}

/**
 * A share (0..1) as a percentage string: 0.513 -> "51%".
 * @param {number} share
 * @param {number} [digits=0] decimal places
 * @returns {string}
 */
export function formatPercent(share, digits = 0) {
  if (typeof share !== 'number' || !Number.isFinite(share)) return NOT_SCORED;
  return `${(share * 100).toFixed(digits)}%`;
}

/**
 * "12 of 3,043 jobs" style phrase for captions under a chart.
 * @param {number} part
 * @param {number} whole
 * @param {string} [noun='jobs']
 * @returns {string}
 */
export function formatShare(part, whole, noun = 'jobs') {
  const share = whole ? formatPercent(part / whole) : formatPercent(0);
  return `${formatCount(part)} of ${formatCount(whole)} ${noun} (${share})`;
}
