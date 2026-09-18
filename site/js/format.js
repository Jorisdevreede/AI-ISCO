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
 * A count with thousands separators: 3039 -> "3,039".
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
 * "12 of 3,039 jobs" style phrase for captions under a chart.
 * @param {number} part
 * @param {number} whole
 * @param {string} [noun='jobs']
 * @returns {string}
 */
export function formatShare(part, whole, noun = 'jobs') {
  const share = whole ? formatPercent(part / whole) : formatPercent(0);
  return `${formatCount(part)} of ${formatCount(whole)} ${noun} (${share})`;
}

/**
 * Whole percentages that add up to 100. Rounding each part on its own gives 99
 * or 101 for a fair share of splits, and the same wrong total is then read out
 * in every aria-label. The remainder goes to the largest fractions (the
 * largest-remainder method), earlier parts first on a tie.
 *
 * @param {number[]} parts counts or shares; anything unusable reads as 0
 * @returns {number[]} one whole percentage per part, summing to 100 (or all 0)
 */
export function largestRemainder(parts) {
  const values = (parts || []).map((value) => (Number.isFinite(value) && value > 0 ? value : 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!total) return values.map(() => 0);
  const exact = values.map((value, index) => {
    const percent = (value / total) * 100;
    return { index, floor: Math.floor(percent), fraction: percent - Math.floor(percent) };
  });
  let left = 100 - exact.reduce((sum, part) => sum + part.floor, 0);
  for (const part of [...exact].sort((a, b) => (b.fraction - a.fraction) || (a.index - b.index))) {
    part.floor += left > 0 ? 1 : 0;
    left -= left > 0 ? 1 : 0;
  }
  return exact.map((part) => part.floor);
}

/**
 * A count with its noun, pluralised: 1 -> "1 occupation", 3039 -> "3,039 occupations".
 * Only the regular -s plural, which is all the nouns the pages use.
 *
 * @param {number} count
 * @param {string} [noun='occupation']
 * @returns {string}
 */
export function plural(count, noun = 'occupation') {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}
