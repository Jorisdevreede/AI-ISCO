// What the method page says about two score sets, as text. Pure: no DOM.
//
// The numbers come from compareScoreSets (site/js/agreement.js); this module only
// words them, so the wording can be tested without a browser.

import { formatCount, formatPercent } from '../format.js';
import { QUADRANT_ORDER } from '../groupstats.js';
import { QUADRANT_NAMES } from '../quadrant.js';

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
 * Text for every [data-agree] slot of the page.
 * @param {Object} result from compareScoreSets
 * @param {[string, string]} labels names of the default and the second scorer
 * @returns {Object<string, string>}
 */
export function agreementValues(result, labels) {
  return {
    firstLabel: labels[0],
    secondLabel: labels[1],
    n: formatCount(result.n),
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
