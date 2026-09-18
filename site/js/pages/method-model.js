// What the method page says, as text. Pure: no DOM.
//
// Two jobs. The agreement half words the numbers compareScoreSets
// (site/js/agreement.js) computes about two scorings of the ORIGINAL rubric.
// The shares half words the current rubric: what is asked, how a skill gets its
// class, how a job gets its four shares and its type.
//
// The wording of the type rules is scheme.js's TYPE_RULES, the same text a type
// badge shows, so the page that explains the rules cannot drift from the badges.

import { formatCount, formatPercent } from '../format.js';
import { QUADRANT_ORDER } from '../groupstats.js';
import { QUADRANT_NAMES } from '../quadrant.js';
import {
  SKILL_CLASS_ORDER, TYPE_RULES, skillClassName, thresholdOf,
} from '../scheme.js';

function correlation(value) {
  return value === null ? 'not defined' : value.toFixed(2);
}

function levels(axis, labels) {
  return `${axis.meanFirst.toFixed(1)} under ${labels[0]} and `
    + `${axis.meanSecond.toFixed(1)} under ${labels[1]}`;
}

function moveSentence(move, labels) {
  if (!move) return '';
  const noun = move.count === 1 ? 'occupation' : 'occupations';
  return `The largest single move is ${formatCount(move.count)} ${noun} from `
    + `${QUADRANT_NAMES[move.from]} under ${labels[0]} to ${QUADRANT_NAMES[move.to]} `
    + `under ${labels[1]}.`;
}

/**
 * The whole note beside the big share, as one computed sentence.
 *
 * It follows a `<strong>` holding the share, so it opens mid-sentence. When the
 * two runs cover every occupation the site has, there is nothing to explain and
 * it simply names the total. When they do not, it says so first, because a page
 * that quotes the smaller number alone contradicts itself two sections above.
 *
 * @param {Object} result from compareScoreSets
 * @param {number} total how many occupations the site has
 * @returns {string}
 */
function agreementNote(result, total) {
  const moved = formatCount(result.n - result.same);
  const tail = `The other ${moved} change box.`;
  if (result.n >= total) {
    return `of the ${formatCount(total)} occupations land in the same box under both `
      + `models. ${tail}`;
  }
  return `${formatCount(result.n)} of the ${formatCount(total)} occupations are scored in `
    + `both runs. Of those, ${formatPercent(result.share)} land in the same box under both `
    + `models. ${tail}`;
}

/**
 * Text for every [data-agree] slot of the page.
 * @param {Object} result from compareScoreSets
 * @param {[string, string]} labels names of the default and the second scorer
 * @param {number} [siteTotal] stats.occupations, so the page agrees with itself
 * @returns {Object<string, string>}
 */
export function agreementValues(result, labels, siteTotal) {
  const total = Number.isFinite(siteTotal) ? siteTotal : (result.total ?? result.n);
  return {
    firstLabel: labels[0],
    secondLabel: labels[1],
    n: formatCount(result.n),
    total: formatCount(total),
    note: agreementNote(result, total),
    share: formatPercent(result.share),
    moved: formatCount(result.n - result.same),
    rhoAutomation: correlation(result.automation.rho),
    rhoAmplification: correlation(result.amplification.rho),
    meanAutomation: levels(result.automation, labels),
    meanAmplification: levels(result.amplification, labels),
    move: moveSentence(result.move, labels),
  };
}

/**
 * The box-by-box table: one row per quadrant of the first set.
 * @returns {{columns: string[], rows: Array<{label: string, cells: Array<{text: string, agrees: boolean}>}>}}
 */
export function agreementTable(result) {
  return {
    columns: QUADRANT_ORDER.map((quadrant) => QUADRANT_NAMES[quadrant]),
    rows: QUADRANT_ORDER.map((from) => ({
      label: QUADRANT_NAMES[from],
      cells: QUADRANT_ORDER.map((to) => ({
        text: formatCount(result.matrix[from][to]),
        agrees: from === to,
      })),
    })),
  };
}

/* --- the shares scheme ---------------------------------------------------- */

/** The rule each of the seven types matched: scheme.js's wording, under the page's name. */
export const TYPE_RULE_TEXT = TYPE_RULES;

/**
 * The rule each skill class matched, in the order the pipeline tries them.
 * Every one is "first match wins", which is why the wording chains.
 */
export const SKILL_CLASS_RULE_TEXT = {
  S: 'The chance that an AI system could carry the work out itself passes the cut.',
  A: 'It did not, and the chance that the person gets a clear gain across most of the work '
    + 'with that system beside them passes the cut.',
  M: 'Neither did, and the chance that physical equipment does the work passes the cut.',
  I: 'None of the three passed the cut.',
};

function ruleRow(code, rule, split) {
  return {
    code,
    rule,
    count: formatCount(split.counts[code] ?? 0),
    share: formatPercent(split.shares[code] ?? 0),
  };
}

/**
 * One row per type, for the method page's rule table.
 * @param {Object} split from splitOf(stats)
 * @returns {Array<{code, rule, count, share}>}
 */
export function typeRuleRows(split) {
  return (split.order || []).map((code) => ruleRow(code, TYPE_RULE_TEXT[code] || '', split));
}

/** The four boxes, in the order the quadrant table has always listed them. */
export const QUADRANT_RULE_TEXT = [
  ['TRANSFORM', 'Both scores at or above the cut-off'],
  ['EVOLVE', 'Automation below the cut-off, amplification at or above it'],
  ['STABLE', 'Both scores below the cut-off'],
  ['SHRINK', 'Automation at or above the cut-off, amplification below it'],
];

/**
 * One row per box, for the same table under the quadrant scheme.
 * @param {Object} split from splitOf(stats)
 * @returns {Array<{code, rule, count, share}>}
 */
export function quadrantRuleRows(split) {
  return QUADRANT_RULE_TEXT.map(([code, rule]) => ruleRow(code, rule, split));
}

/**
 * One row per skill class, for the method page's class table.
 * @param {Object} stats the active set's stats file
 * @returns {Array<{code, name, rule, count, share}>}
 */
export function skillClassRows(stats) {
  const source = (stats && stats.skill_classes) || {};
  const counts = source.counts || {};
  const shares = source.shares || {};
  const total = SKILL_CLASS_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0);
  return SKILL_CLASS_ORDER.map((code) => ({
    code,
    name: skillClassName(code),
    rule: SKILL_CLASS_RULE_TEXT[code] || '',
    count: formatCount(counts[code] ?? 0),
    share: formatPercent(shares[code] ?? (total ? (counts[code] || 0) / total : 0)),
  }));
}

/**
 * Every value a [data-stat] slot on the page can ask for.
 *
 * `classCut` is the cut-off on a model probability, in words where the stats
 * file does not publish one, so the page can never print the quadrant scheme's
 * 6 beside a probability.
 *
 * @param {Object} stats the active set's stats file
 * @param {number} nearDistance the quadrant scheme's near-the-line distance
 * @returns {Object<string, string>}
 */
export function methodValues(stats, nearDistance) {
  const nearLine = (stats && stats.near_line) || {};
  const threshold = thresholdOf(stats);
  const types = ((stats && stats.types) || {}).counts || {};
  const typeShares = ((stats && stats.types) || {}).shares || {};
  return {
    skills: formatCount(stats.skills_scored),
    occupations: formatCount(stats.occupations),
    threshold: threshold === null ? 'none' : String(threshold),
    classCut: threshold === null ? 'one half' : String(threshold),
    nearLineCount: formatCount(nearLine.count),
    nearLineShare: formatPercent(nearLine.share),
    nearLineDistance: String(nearDistance),
    mixedCount: formatCount(types.MIXED ?? 0),
    mixedShare: formatPercent(typeShares.MIXED ?? 0),
    model: stats.model || 'not recorded',
    built: stats.built || 'not recorded',
  };
}
