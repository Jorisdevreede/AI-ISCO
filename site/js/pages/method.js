// "How sure is this?" — audit flow D5, within the brief's restrictions.
//
// The page's prose is static; every number in it is read from stats.json at
// load, so a rebuild of the data rewrites the page instead of contradicting it.
// Nothing here may state or imply a result from any second scoring run: the
// only uncertainty this site is allowed to quote is what the published data
// shows on its own.

import { renderChrome } from '../chrome.js';
import { loadJSON, whenSlow } from '../data.js';
import { formatCount, formatPercent } from '../format.js';
import { NEAR_LINE, QUADRANT_NAMES, THRESHOLD } from '../quadrant.js';

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
