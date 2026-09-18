// The four shares of an occupation, as text and as bar segments. Pure: no DOM.
//
// `sh` is [substituted, assisted, mechanised, insulated] and sums to 1. The
// percentages shown always sum to 100: rounding each one on its own gives 99 or
// 101 often enough that a visitor would notice, so the remainder goes to the
// largest fractions (the largest-remainder method), which keeps the order of the
// four parts intact.

/** The class codes, in the order `sh` stores them. */
export const SHARE_ORDER = ['S', 'A', 'M', 'I'];

/** How each part reads inside the one-line sentence. */
export const SHARE_LABELS = {
  S: 'AI can take over',
  A: 'AI assists',
  M: 'machines',
  I: 'stays human',
};

/**
 * True for a usable shares array: four finite numbers that add up to about 1.
 * @param {*} shares
 * @returns {boolean}
 */
export function isShares(shares) {
  if (!Array.isArray(shares) || shares.length !== 4) return false;
  if (!shares.every((value) => typeof value === 'number' && Number.isFinite(value))) return false;
  const total = shares.reduce((sum, value) => sum + value, 0);
  return Math.abs(total - 1) <= 0.02;
}

function remainderOrder(parts) {
  return [...parts].sort((a, b) => (b.fraction - a.fraction) || (a.index - b.index));
}

function rawParts(shares) {
  return shares.map((share, index) => {
    const exact = share * 100;
    const floor = Math.floor(exact);
    return { index, code: SHARE_ORDER[index], floor, fraction: exact - floor };
  });
}

/**
 * The four shares as whole percentages that add up to 100.
 *
 * @param {number[]} shares four numbers summing to 1
 * @returns {Array<{code: string, label: string, share: number, percent: number}>}
 *   an empty array when `shares` is not usable
 */
export function sharePercents(shares) {
  if (!isShares(shares)) return [];
  const parts = rawParts(shares);
  let left = 100 - parts.reduce((sum, part) => sum + part.floor, 0);
  for (const part of remainderOrder(parts)) {
    part.floor += left > 0 ? 1 : 0;
    left -= left > 0 ? 1 : 0;
  }
  return parts.map((part) => ({
    code: part.code,
    label: SHARE_LABELS[part.code],
    share: shares[part.index],
    percent: part.floor,
  }));
}

/**
 * The shares as one sentence:
 * "AI can take over 42% · AI assists 31% · machines 5% · stays human 22%".
 *
 * @param {number[]} shares
 * @returns {string} '' when `shares` is not usable
 */
export function shareSentence(shares) {
  return sharePercents(shares)
    .map((part) => `${part.label} ${part.percent}%`)
    .join(' · ');
}

/**
 * The same four parts, phrased for a screen reader reading a bar.
 * @param {number[]} shares
 * @param {string} [name] the job's title, when there is one
 * @returns {string}
 */
export function shareAriaLabel(shares, name) {
  const sentence = shareSentence(shares);
  if (!sentence) return 'Shares not scored';
  const subject = name ? `What ${name} is made of` : 'What this job is made of';
  return `${subject}: ${sentence}.`;
}

/**
 * Segments for a stacked bar: the parts that are worth drawing, widest first in
 * the data but kept in class order so the colours never move between jobs.
 *
 * @param {number[]} shares
 * @returns {Array<{code: string, label: string, percent: number, width: string}>}
 */
export function shareSegments(shares) {
  return sharePercents(shares)
    .filter((part) => part.percent > 0)
    .map((part) => ({ ...part, width: `${part.percent}%` }));
}

/**
 * The largest of the four parts, for a one-word summary in a dense row.
 * @param {number[]} shares
 * @returns {{code: string, label: string, percent: number}|null}
 */
export function largestShare(shares) {
  const parts = sharePercents(shares);
  if (!parts.length) return null;
  return parts.reduce((best, part) => (part.percent > best.percent ? part : best));
}
