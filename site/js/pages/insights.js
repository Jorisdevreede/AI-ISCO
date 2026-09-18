// "What we found" — the article, with every figure computed from the data.
// Thin DOM wiring; insights-model.js derives every number and fills the prose.
//
// The audit (B5) found this page showing a hero count and a contradicting share
// two lines below it, because the prose carried constants. There are no number
// literals in this file, and the page no longer touches data.json or
// portfolio_data.json: stats.json, groups.json and search_index.json are all it
// loads.

import { renderChrome } from '../chrome.js';
import { loadJSON, whenSlow } from '../data.js';
import { NOT_SCORED, formatCount, formatScore } from '../format.js';
import { SHARES, schemeOfCode, typeLabel } from '../scheme.js';
import { renderSharesBar } from '../shares-bar.js';
import { sharePercents } from '../shares.js';
import { jobHref, withScorer } from '../urlstate.js';
import {
  SHARES_NOTES, articleValues, buildArticle, deriveInsights, fillTemplate,
} from './insights-model.js';

const SLOW_MS = 200;

/** Section copy. Every figure is a placeholder that insights-model.js fills. */
const NOTES = {
  'hero-sub': 'Every skill in ESCO read by a language model and scored on two axes, then '
    + 'rolled up to {occupations} occupations. {skills} skills, no measurement of any real '
    + 'workplace. Every number on this page is computed from the published data as the page '
    + 'loads.',
  'split-note': 'The boxes are a hard cut at {threshold} on both axes, so read the shares as '
    + 'a band rather than a count: {nearLineCount} occupations, {nearLineShare} of them, sit '
    + 'within half a point of a cut-off. {methodLink}',
  'extremes-note': 'The ends of the automation-risk axis. High exposure is not a verdict on '
    + 'a job — it says how much of the skill list a machine could take on.',
  'shrink-note': 'High automation risk with low amplification: the work is automatable, and '
    + 'the same technology gives little back to the person doing it. {shrinkCount} '
    + 'occupations, {shrinkShare} of the total, land here. Ranked by the gap between the two '
    + 'scores.',
  'stable-note': 'The lowest scores on both axes — work these scores barely touch. '
    + '{stableCount} occupations, {stableShare} of the total, land here.',
  'transform-note': 'High on both axes: automatable work where AI also makes the person '
    + 'doing it far more productive. {transformCount} occupations, {transformShare} of the '
    + 'total. Ranked by the two scores together.',
  'groups-note': 'The ten ISCO major groups, with the average of each score across the '
    + 'occupations in them. {topAmpGroup} average the highest amplification, {topAmpGroupAmp} '
    + 'out of 10; {topAutoGroup} the highest automation risk, {topAutoGroupAuto}.',
};

const LISTS = [
  { id: 'most-exposed', key: 'mostExposed' },
  { id: 'least-exposed', key: 'leastExposed' },
  { id: 'shrink-list', key: 'shrink' },
  { id: 'stable-list', key: 'stable' },
  { id: 'transform-list', key: 'transform' },
];

/** The sections that only make sense under one scheme. */
const QUADRANT_SECTIONS = ['extremes-section', 'shrink-section', 'stable-section',
  'transform-section'];
const SHARES_SECTIONS = ['classes-section', 'substituted-section', 'assisted-section',
  'mechanised-section', 'insulated-section'];

/** The four ranked lists of a shares set, each ranked by one of the four shares. */
const SHARES_LISTS = [
  { id: 'substituted-list', key: 'substituted', at: 0 },
  { id: 'assisted-list', key: 'assisted', at: 1 },
  { id: 'mechanised-list', key: 'mechanised', at: 2 },
  { id: 'insulated-list', key: 'insulated', at: 3 },
];

const SHARES_GROUP_COLUMNS = [
  { label: 'Major group', numeric: false },
  { label: 'Jobs', numeric: true },
  { label: 'What the average job is made of', numeric: false },
  { label: 'Largest type', numeric: false },
];

const SHARES_GROUP_CAPTION = 'The average make-up of a job in each ISCO major group. '
  + 'Each group links to its sector page.';

const SHARES_ARTICLE_TITLE = 'The shape of the work';

const SHARES_ARTICLE_BYLINE = 'A data-led look at what an AI system could take over, what it '
  + 'only assists with, what machinery already does, and what stays with the person';

const status = document.getElementById('insights-status');
const errorSlot = document.getElementById('insights-error');

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function anchor(href, text, className) {
  const node = el('a', className, text);
  node.href = withScorer(href, window.location.search);
  return node;
}

function setStatus(text) {
  status.textContent = text || '';
  status.hidden = !text;
}

/* --- prose ---------------------------------------------------------------- */

function segmentNodes(segments) {
  return segments.map((segment) => (segment.href
    ? anchor(segment.href, segment.text)
    : document.createTextNode(segment.text)));
}

function renderNotes(values, templates) {
  for (const [id, template] of Object.entries(templates)) {
    const node = document.getElementById(id);
    if (node) node.replaceChildren(...segmentNodes(fillTemplate(template, values)));
  }
}

function articleBlock(block) {
  const paragraph = el('p', block.type === 'pullquote' ? 'pullquote' : block.className);
  paragraph.append(...segmentNodes(block.segments));
  return paragraph;
}

function renderArticle(facts) {
  document.getElementById('article-body')
    .replaceChildren(...buildArticle(facts).map(articleBlock));
}

/* --- tables and lists ----------------------------------------------------- */

function heroStat(label, value) {
  const item = el('div', 'hero-stat');
  item.append(el('dd', null, value), el('dt', null, label));
  return item;
}

function renderHero(facts) {
  const items = [
    heroStat('Occupations scored', formatCount(facts.occupations)),
    heroStat('ESCO skills scored', formatCount(facts.skills)),
    ...facts.quadrants.map((row) => heroStat(`in ${row.name}`, row.countText)),
  ];
  document.getElementById('hero-stats').replaceChildren(...items);
}

// One drawn bar for a split table, painted from whichever code the row carries:
// a quadrant, one of the seven types, or one of the four skill classes.
function splitBar(row) {
  const bar = el('span', 'q-bar');
  if (row.code.length === 1) bar.dataset.class = row.code;
  else if (schemeOfCode(row.code) === SHARES) bar.dataset.type = row.code;
  else bar.dataset.quadrant = row.code;
  bar.style.width = `${Math.max(2, row.share * 100).toFixed(1)}%`;
  return bar;
}

function quadrantRow(row) {
  const tr = el('tr');
  const name = el('th', null, row.name);
  name.scope = 'row';
  const barCell = el('td', 'q-bar-cell');
  barCell.setAttribute('aria-hidden', 'true');
  barCell.appendChild(splitBar(row));
  tr.append(name, el('td', 'numeric', row.countText), el('td', 'numeric', row.shareText),
    barCell);
  return tr;
}

function scorePair(row, className) {
  const node = el('span', className);
  node.append(el('span', 'visually-hidden', 'automation '), formatScore(row.a), ' / ',
    el('span', 'visually-hidden', 'amplification '), formatScore(row.m));
  return node;
}

function rankItem(row) {
  const item = el('li');
  item.append(anchor(jobHref(row.s), row.t), scorePair(row, 'rank-scores'));
  return item;
}

function renderLists(facts) {
  for (const { id, key } of LISTS) {
    document.getElementById(id).replaceChildren(...facts.lists[key].map(rankItem));
  }
}

function largestBox(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return '—';
  const best = entries.reduce((top, entry) => (entry[1] > top[1] ? entry : top));
  return typeLabel(best[0]);
}

function groupRow(major) {
  const tr = el('tr');
  const name = el('th');
  name.scope = 'row';
  name.appendChild(anchor(major.href, major.label));
  tr.append(name, el('td', 'numeric', formatCount(major.n)),
    el('td', 'numeric', formatScore(major.auto)), el('td', 'numeric', formatScore(major.amp)),
    el('td', null, largestBox(major.counts)));
  return tr;
}

function renderGroups(facts) {
  document.querySelector('#group-table tbody')
    .replaceChildren(...facts.majors.map(groupRow));
}

/* --- the shares scheme ---------------------------------------------------- */

// Under a shares set the four boxes do not exist, so the sections built around
// them are hidden and four sections built around the four shares take their
// place. Nothing is filled with the zeros the other scheme's lookups return.

function showSections(ids, visible) {
  for (const id of ids) {
    const section = document.getElementById(id);
    if (section) section.hidden = !visible;
  }
}

function retitleSplitTable() {
  const table = document.getElementById('quadrant-table');
  table.querySelector('caption').textContent = 'Occupations of each type, largest first.';
  table.querySelector('thead th').textContent = 'Type';
}

function renderClasses(facts) {
  document.querySelector('#class-table tbody')
    .replaceChildren(...facts.classes.map(quadrantRow));
}

// A ranked row leads with the job, then the share it is ranked by, then the
// whole make-up as a compact bar — so the rank is never the only thing said.
function shareRankItem(row, at) {
  const item = el('li');
  const part = sharePercents(row.sh)[at];
  const figure = el('span', 'rank-figure numeric');
  figure.append(el('span', 'visually-hidden', `${part.label}: `), `${part.percent}%`);
  item.append(anchor(jobHref(row.s), row.t), figure,
    renderSharesBar(row.sh, { name: row.t, compact: true }));
  return item;
}

function renderShareLists(facts) {
  for (const { id, key, at } of SHARES_LISTS) {
    document.getElementById(id)
      .replaceChildren(...facts.lists[key].map((row) => shareRankItem(row, at)));
  }
}

function shareGroupRow(major) {
  const tr = el('tr');
  const name = el('th');
  name.scope = 'row';
  name.appendChild(anchor(major.href, major.label));
  const madeOf = el('td', 'shares-cell');
  madeOf.appendChild(major.sh
    ? renderSharesBar(major.sh, { name: `the average job in ${major.label}`, compact: true })
    : el('span', 'muted', NOT_SCORED));
  tr.append(name, el('td', 'numeric', formatCount(major.n)), madeOf,
    el('td', null, typeLabel(major.topType)));
  return tr;
}

function renderShareGroups(facts) {
  const table = document.getElementById('group-table');
  table.querySelector('caption').textContent = SHARES_GROUP_CAPTION;
  table.querySelector('thead').replaceChildren(el('tr'));
  const head = table.querySelector('thead tr');
  for (const column of SHARES_GROUP_COLUMNS) {
    const cell = el('th', column.numeric ? 'numeric' : null, column.label);
    cell.scope = 'col';
    head.appendChild(cell);
  }
  table.querySelector('tbody').replaceChildren(...facts.majors.map(shareGroupRow));
}

function retitleArticle() {
  document.getElementById('article-heading').textContent = SHARES_ARTICLE_TITLE;
  document.querySelector('.article-byline').textContent = SHARES_ARTICLE_BYLINE;
}

function adaptToShares(facts) {
  showSections(QUADRANT_SECTIONS, false);
  showSections(SHARES_SECTIONS, true);
  retitleSplitTable();
  renderClasses(facts);
  renderShareLists(facts);
  renderShareGroups(facts);
  retitleArticle();
}

/* --- boot ----------------------------------------------------------------- */

function renderAll(facts) {
  const shares = facts.scheme === SHARES;
  renderHero(facts);
  renderNotes(articleValues(facts), shares ? SHARES_NOTES : NOTES);
  document.querySelector('#quadrant-table tbody')
    .replaceChildren(...facts.quadrants.map(quadrantRow));
  if (shares) adaptToShares(facts);
  else {
    renderLists(facts);
    renderGroups(facts);
  }
  renderArticle(facts);
}

function showError(error, retry) {
  const block = el('div', 'error-block');
  block.append(el('h2', null, 'The findings did not load'), el('p', null, error.message));
  const actions = el('p', 'inline-actions');
  const again = el('button', 'button button--quiet', 'Try again');
  again.type = 'button';
  again.addEventListener('click', retry);
  actions.append(again, anchor('index.html', 'Find a job instead'));
  block.appendChild(actions);
  errorSlot.replaceChildren(block);
}

function load() {
  errorSlot.replaceChildren();
  const sources = Promise.all([loadJSON('stats'), loadJSON('groups'), loadJSON('search_index')]);
  whenSlow(sources, SLOW_MS, () => setStatus('Loading the findings…'))
    .then(([stats, groups, index]) => {
      setStatus('');
      renderAll(deriveInsights({ stats, groups, index }));
    })
    .catch((error) => {
      setStatus('');
      showError(error, load);
    });
}

renderChrome({ active: 'insights' });
setStatus('');
load();
