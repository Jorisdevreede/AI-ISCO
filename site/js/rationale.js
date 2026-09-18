// Who wrote a rationale. Pure: no DOM.
//
// A scorer that returns numbers only writes no text. For its scores the build
// borrows the rationale another model wrote, and marks the skill with
//   rf: { s: <scorer>, a: <its automation score>, m: <its amplification score> }
// (see build_portfolio_data.py). The pages show the borrowed text with an "i"
// that says so; a skill without `rf` has a rationale written for the scores on
// screen and gets no note.

import { formatScore } from './format.js';

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
  const borrowed = (skills || []).find((skill) => skill && skill.rationaleFrom);
  if (!borrowed) return 'Each rationale is what the model wrote while it scored that skill.';
  return `Each rationale was written by ${scorerName(borrowed.rationaleFrom.s)} for its own `
    + 'scores, not for the scores shown here: the i beside it says which.';
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
