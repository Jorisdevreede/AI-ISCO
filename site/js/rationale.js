// Who wrote a rationale. Pure: no DOM.
//
// A scorer that returns numbers only writes no text. For its scores the build
// borrows the rationale another model wrote, and marks the skill with
//   rf: { s: <scorer>, a: <its automation score>, m: <its amplification score> }
// (see build_portfolio_data.py). The pages show the borrowed text with an "i"
// that says so; a skill without `rf` has a rationale written for the scores on
// screen and gets no note.

import { formatScore, isScored } from './format.js';

const SCORER_NAMES = { gemini: 'Gemini' };

/** Display name of a scorer key: a known name, or the key with a capital. */
export function scorerName(key) {
  const text = String(key || '').trim();
  if (!text) return 'another model';
  return SCORER_NAMES[text.toLowerCase()] || text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * The sentence above a list of rationales that says who wrote them.
 * @param {Array<{rationaleFrom?: Object|null}>} skills
 */
export function rationaleWriterLine(skills) {
  const borrowed = (skills || []).find((skill) => skill?.rationaleFrom);
  if (!borrowed) return 'Each rationale is what the model wrote while it scored that skill.';
  return `Each explanation below was written by ${scorerName(borrowed.rationaleFrom.s)} for `
    + 'its own, older scores. Each one says which score it was written for, and where that '
    + 'score is far from the one shown here the explanation is left out rather than '
    + 'stretched to fit.';
}

/**
 * How far a borrowed explanation's own score has to sit from the one on screen
 * before the explanation stops being about the same thing.
 *
 * Measured over the published set: the median gap is 1.3 points and 17% of
 * skills differ by 3 or more. At 3 the sentences start contradicting the label
 * beside them — a skill classed "Stays human" carrying "AI can automate this
 * entirely", written for a 9.0.
 */
export const BORROWED_GAP_LIMIT = 3;

/**
 * The gap between the score a borrowed explanation was written for and the one
 * on screen.
 * @param {{a?: number}|null} source the skill's `rf`
 * @param {number} score the AI-substitution score this page shows
 * @returns {number|null} null when either side is missing
 */
export function borrowedGap(source, score) {
  if (!source || !isScored(source.a) || !isScored(score)) return null;
  return Math.round(Math.abs(source.a - score) * 10) / 10;
}

/**
 * Whether to print a borrowed explanation, and what to say around it.
 *
 * A rationale written for a very different score argues with the class beside
 * it, and a disclaimer does not stop a reader taking the sentence as the
 * explanation of the label. So past BORROWED_GAP_LIMIT it is not printed at
 * all, and under it the gap leads rather than a generic note.
 *
 * @param {{text?: string, source?: Object|null, score?: number}} skill
 * @returns {{show: boolean, lead: string, after: string, why: string}}
 */
function suppressed(wrote, score) {
  return {
    show: false,
    lead: '',
    after: '',
    why: `${wrote}, against the ${formatScore(score)} shown here. The only explanation we `
      + 'have was written for that older score, so it is not shown: it would be arguing '
      + 'with the numbers beside it.',
  };
}

export function borrowedRationale({ text, source, score } = {}) {
  const blank = { show: false, lead: '', after: '', why: '' };
  if (!text) return blank;
  const gap = borrowedGap(source, score);
  if (gap === null) return { ...blank, show: true };
  const wrote = `${scorerName(source.s)} scored this ${formatScore(source.a)} `
    + 'for AI substitution';
  if (gap >= BORROWED_GAP_LIMIT) return suppressed(wrote, score);
  return {
    show: true,
    lead: `${wrote} and wrote:`,
    after: `This page shows ${formatScore(score)}.`,
    why: '',
  };
}

/**
 * The note behind the "i" of a borrowed rationale, or null when the rationale
 * was written for the scores on screen.
 * @param {{s?: string, a?: number|null, m?: number|null}|null|undefined} source the skill's `rf`
 * @returns {{writer: string, label: string, text: string}|null}
 */
export function borrowedRationaleNote(source) {
  if (!source || typeof source !== 'object') return null;
  const writer = scorerName(source.s);
  return {
    writer,
    label: `About this explanation: written by ${writer}`,
    text: `${writer} wrote this explanation for its own scores: automation risk `
      + `${formatScore(source.a)}, amplification ${formatScore(source.m)}. The scores on `
      + 'screen come from a model that returns numbers only and writes no text, so the '
      + 'explanation may not match them.',
  };
}
