// "How sure is this?" — audit flow D5, within the brief's restrictions.
//
// The page's prose is static; every number in it is computed at load, so a
// rebuild of the data rewrites the page instead of contradicting it. The counts
// come from stats.json. Where a second score set is deployed, the agreement
// between the two sets is computed from both search indexes (site/js/agreement.js);
// without it that section stays hidden.

import { renderChrome } from '../chrome.js';
import { compareScoreSets } from '../agreement.js';
import { loadBothSets, loadJSON, whenSlow } from '../data.js';
import { formatCount, formatPercent } from '../format.js';
import { NEAR_LINE, QUADRANT_NAMES, THRESHOLD } from '../quadrant.js';
import { agreementTable, agreementValues } from './method-model.js';

const SLOW_MS = 200;

/** The four boxes, in the order the table lists them. */
const QUADRANT_RULES = [
  { code: 'TRANSFORM', rule: 'Both scores at or above the cut-off' },
  { code: 'EVOLVE', rule: 'Automation below the cut-off, amplification at or above it' },
  { code: 'STABLE', rule: 'Both scores below the cut-off' },
  { code: 'SHRINK', rule: 'Automation at or above the cut-off, amplification below it' },
];

const status = document.getElementById('method-status');
const errorSlot = document.getElementById('method-error');
const table = document.getElementById('quadrant-table');

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(text) {
  status.textContent = text || '';
  status.hidden = !text;
}

/** Every value a [data-stat] slot in the page can ask for. */
function statValues(stats) {
  const nearLine = stats.near_line || {};
  return {
    skills: formatCount(stats.skills_scored),
    occupations: formatCount(stats.occupations),
    threshold: String(stats.threshold ?? THRESHOLD),
    nearLineCount: formatCount(nearLine.count),
    nearLineShare: formatPercent(nearLine.share),
    nearLineDistance: String(NEAR_LINE),
    built: stats.built || 'not recorded',
  };
}

function fillSlots(values) {
  for (const slot of document.querySelectorAll('[data-stat]')) {
    const value = values[slot.dataset.stat];
    if (value !== undefined) slot.textContent = value;
  }
}

function quadrantRow(entry, stats) {
  const counts = stats.quadrants?.counts || {};
  const shares = stats.quadrants?.shares || {};
  const row = el('tr');
  const name = el('th', null, QUADRANT_NAMES[entry.code]);
  name.scope = 'row';
  row.append(name, el('td', null, entry.rule),
    el('td', 'numeric', formatCount(counts[entry.code] ?? 0)),
    el('td', 'numeric', formatPercent(shares[entry.code] ?? 0)));
  return row;
}

function renderTable(stats) {
  const body = table.querySelector('tbody');
  body.replaceChildren(...QUADRANT_RULES.map((entry) => quadrantRow(entry, stats)));
}

function showError(error, retry) {
  const block = el('div', 'error-block');
  block.append(el('h2', null, 'The numbers did not load'), el('p', null, error.message),
    el('p', null, 'The explanation below is unchanged; only the counts are missing.'));
  const again = el('button', 'button button--quiet', 'Try again');
  again.type = 'button';
  again.addEventListener('click', retry);
  block.appendChild(again);
  errorSlot.replaceChildren(block);
}

// --- the second model ---------------------------------------------------------

function fillAgreementSlots(values) {
  for (const slot of document.querySelectorAll('[data-agree]')) {
    const value = values[slot.dataset.agree];
    if (value !== undefined) slot.textContent = value;
  }
}

function headerRow(columns) {
  const row = el('tr');
  row.appendChild(el('td'));
  for (const label of columns) {
    const cell = el('th', 'numeric', label);
    cell.scope = 'col';
    row.appendChild(cell);
  }
  return row;
}

function matrixRow(entry) {
  const row = el('tr');
  const name = el('th', null, entry.label);
  name.scope = 'row';
  row.appendChild(name);
  for (const cell of entry.cells) {
    row.appendChild(el('td', cell.agrees ? 'numeric agrees' : 'numeric', cell.text));
  }
  return row;
}

function renderAgreement(sets) {
  const result = sets && compareScoreSets(sets.first, sets.second);
  if (!result) return;
  fillAgreementSlots(agreementValues(result, sets.labels));
  const matrix = agreementTable(result);
  const grid = document.getElementById('agreement-table');
  grid.querySelector('thead').replaceChildren(headerRow(matrix.columns));
  grid.querySelector('tbody').replaceChildren(...matrix.rows.map(matrixRow));
  document.getElementById('agreement').hidden = false;
}

/** The section is a bonus: if its files fail to load it stays hidden, the page stands. */
function loadAgreement() {
  loadBothSets('search_index').then(renderAgreement).catch(() => {});
}

function load() {
  errorSlot.replaceChildren();
  whenSlow(loadJSON('stats'), SLOW_MS, () => setStatus('Loading the current counts…'))
    .then((stats) => {
      setStatus('');
      fillSlots(statValues(stats));
      renderTable(stats);
    })
    .catch((error) => {
      setStatus('');
      showError(error, load);
    });
}

renderChrome({ active: 'method' });
setStatus('');
load();
loadAgreement();
