// "How sure is this?" — audit flow D5, within the brief's restrictions.
//
// The page's prose is static and every number in it is computed at load, so a
// rebuild of the data rewrites the page instead of contradicting it. Two score
// sets of two different shapes are published, so the prose comes in two
// versions: elements carrying data-scheme belong to one of them, everything
// else belongs to both, and applyScheme shows the right half.
//
// The counts come from stats.json. Where a second scoring of the ORIGINAL
// rubric is deployed, the agreement between those two sets is computed from
// both search indexes (site/js/agreement.js); without it that section stays
// hidden, and compareScoreSets refuses anything that is not that rubric.

import { renderChrome } from '../chrome.js';
import { compareScoreSets } from '../agreement.js';
import { loadBothSets, loadJSON, whenSlow } from '../data.js';
import { NEAR_LINE } from '../quadrant.js';
import { SHARES, splitOf, typeLabel } from '../scheme.js';
import {
  agreementTable, agreementValues, methodValues, quadrantRuleRows, skillClassRows, typeRuleRows,
} from './method-model.js';

const SLOW_MS = 200;

const status = document.getElementById('method-status');
const errorSlot = document.getElementById('method-error');

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

function fillSlots(values) {
  for (const slot of document.querySelectorAll('[data-stat]')) {
    const value = values[slot.dataset.stat];
    if (value !== undefined) slot.textContent = value;
  }
}

/** Show the half of the prose that belongs to the scheme on screen. */
function applyScheme(scheme) {
  for (const node of document.querySelectorAll('[data-scheme]')) {
    node.hidden = node.dataset.scheme !== scheme;
  }
}

/* --- the rule tables ------------------------------------------------------ */

// Every rule table on this page has the same four columns: the class, the rule
// it matched in plain words, and the two figures the current run gives it.
function ruleRow(entry) {
  const row = el('tr');
  const name = el('th', null, entry.name || typeLabel(entry.code));
  name.scope = 'row';
  row.append(name, el('td', null, entry.rule), el('td', 'numeric', entry.count),
    el('td', 'numeric', entry.share));
  return row;
}

function fillTable(id, rows) {
  document.querySelector(`#${id} tbody`).replaceChildren(...rows.map(ruleRow));
}

function renderTables(stats) {
  const split = splitOf(stats);
  if (split.scheme !== SHARES) {
    fillTable('quadrant-table', quadrantRuleRows(split));
    return;
  }
  fillTable('type-table', typeRuleRows(split));
  fillTable('class-table', skillClassRows(stats));
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

// --- the second scoring of the original rubric --------------------------------

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

// The site states one occupation total everywhere; this comparison covers
// slightly fewer, so it is given that total and says so rather than quietly
// printing a smaller number the page contradicts two sections above.
let siteOccupations = null;

const FAILED = 'The comparison did not load, so it is left out. Nothing else on this page '
  + 'depends on it.';

function setAgreementStatus(text) {
  const slot = document.getElementById('agreement-status');
  slot.textContent = text || '';
  slot.hidden = !text;
}

function renderAgreement(sets) {
  const result = sets && compareScoreSets(sets.first, sets.second);
  if (!result) {
    setAgreementStatus(FAILED);
    return;
  }
  fillAgreementSlots(agreementValues(result, sets.labels, siteOccupations));
  const matrix = agreementTable(result);
  const grid = document.getElementById('agreement-table');
  grid.querySelector('thead').replaceChildren(headerRow(matrix.columns));
  grid.querySelector('tbody').replaceChildren(...matrix.rows.map(matrixRow));
  setAgreementStatus('');
  document.getElementById('agreement-body').hidden = false;
}

/** The section is a bonus: if its files fail to load it says so and the page stands. */
function fetchAgreement() {
  const sets = loadBothSets('search_index');
  whenSlow(sets, SLOW_MS, () => setAgreementStatus('Loading both score files…')).catch(() => {});
  sets.then(renderAgreement).catch(() => setAgreementStatus(FAILED));
}

// Two whole search indexes, and the largest download on this page. Nobody pays
// for them until the section they belong to is about to be read.
function watchAgreement() {
  const sentinel = document.getElementById('agreement-sentinel');
  if (!sentinel || typeof window.IntersectionObserver !== 'function') {
    fetchAgreement();
    return;
  }
  const observer = new window.IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    fetchAgreement();
  }, { rootMargin: '300px' });
  observer.observe(sentinel);
}

// The heading and the note saying which rubric this compares cost nothing, so
// they stand as soon as we know a second run of it is deployed here at all.
function loadAgreement() {
  Promise.resolve(typeof window !== 'undefined' && window.scorerAlternative)
    .then((alternative) => {
      if (!alternative) return;
      document.getElementById('agreement').hidden = false;
      watchAgreement();
    })
    .catch(() => {});
}

function load() {
  errorSlot.replaceChildren();
  whenSlow(loadJSON('stats'), SLOW_MS, () => setStatus('Loading the current counts…'))
    .then((stats) => {
      setStatus('');
      siteOccupations = stats.occupations ?? null;
      applyScheme(splitOf(stats).scheme);
      fillSlots(methodValues(stats, NEAR_LINE));
      renderTables(stats);
    })
    .catch((error) => {
      setStatus('');
      showError(error, load);
    })
    // Either way the comparison runs; it only ever needed the total for wording.
    .finally(loadAgreement);
}

renderChrome({ active: 'method' });
setStatus('');
load();
