// Landing page wiring: "Find a job". Thin on purpose — every decision (which
// chips, what the recently-viewed list is, when a query has too many or no
// matches, what the copy says) lives in the pure landing-model.js next door.
//
// This page loads search_index.json and stats.json and nothing else: the 13 MB
// portfolio file belongs to job.html.

import { renderChrome } from '../chrome.js';
import { createCombobox } from '../combobox.js';
import { LoadError, loadJSON, whenSlow } from '../data.js';
import { groupHref, jobHref, skillHref, withScorer } from '../urlstate.js';
import {
  POPULAR_CHIPS, loadingText, noMatchText, optionMeta, overflowText, readRecent,
  recentRows, removeRecent, rowBySlug, rowTypeLabel, searchOutcome, sentenceCase, subtitleText,
  suggestionHeading,
} from './landing-model.js';

/** Show a loading state only once the wait is real. */
const SLOW_MS = 200;

const state = { index: null, stats: null, combo: null };

const el = (id) => document.getElementById(id);

/** Every in-site href goes through here, so ?scorer= is never dropped. */
const link = (href) => withScorer(href, window.location.search);

const groupsHref = () => link(groupHref('all'));

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function anchor(href, text, className) {
  const element = node('a', className, text);
  element.href = href;
  return element;
}

/** localStorage throws in private mode; the page works without it. */
function store() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function openJob(slug) {
  window.location.assign(link(jobHref(slug)));
}

// --- search results -------------------------------------------------------

function renderOption(result, item) {
  item.append(
    node('span', 'code', result.row.c || ''),
    node('span', 'title', sentenceCase(result.row.t)),
    node('span', 'meta', optionMeta(result.row)),
  );
  if (result.alt) item.appendChild(node('span', 'also', `also matches “${result.alt}”`));
}

function setHint(nodes) {
  const hint = el('search-hint');
  hint.replaceChildren(...nodes);
  hint.hidden = nodes.length === 0;
}

function manyHint(outcome) {
  const line = node('p', null, `${overflowText(outcome.hidden, outcome.query)} `);
  line.append(anchor(groupsHref(), 'browse by sector'), document.createTextNode('.'));
  return [line];
}

function nearestItem(row) {
  const item = node('li');
  const target = anchor(link(jobHref(row.s)), sentenceCase(row.t));
  target.appendChild(node('span', 'code', row.c || ''));
  item.appendChild(target);
  return item;
}

function noMatchHint(outcome) {
  const tail = node('p');
  tail.appendChild(anchor(groupsHref(), 'Browse by sector instead →'));
  const lead = node('p', null, noMatchText(outcome.query));
  if (!outcome.nearest.length) return [lead, tail];
  const list = node('ul', 'nearest-list');
  list.append(...outcome.nearest.map(nearestItem));
  return [lead, node('p', null, suggestionHeading(outcome)), list, tail];
}

const HINTS = { many: manyHint, 'no-match': noMatchHint };

function getResults(query) {
  if (!state.index) {
    setHint([node('p', null, loadingText(state.stats))]);
    return [];
  }
  const outcome = searchOutcome(state.index, query);
  const build = HINTS[outcome.state];
  setHint(build ? build(outcome) : []);
  return outcome.results;
}

/** Enter without having arrowed down opens the best match. */
function onEnter(event) {
  if (event.key !== 'Enter' || event.defaultPrevented || !state.index) return;
  const [best] = searchOutcome(state.index, event.currentTarget.value).results;
  if (best) openJob(best.row.s);
}

// --- chips and recently viewed --------------------------------------------

/**
 * One chip: the job it opens, and once the index is in, what kind of job the
 * active score set says it is. The class name sits beside the link rather than
 * inside it, so the chip never promises a title it does not open.
 */
function chipNode(chip) {
  const target = anchor(link(jobHref(chip.slug)), sentenceCase(chip.label), 'chip');
  const kind = rowTypeLabel(rowBySlug(state.index, chip.slug));
  if (!kind) return target;
  const item = node('span', 'chip-item');
  item.append(target, node('span', 'chip-meta', kind));
  return item;
}

function renderChips() {
  el('chip-row').replaceChildren(...POPULAR_CHIPS.map(chipNode));
}

function forget(slug) {
  removeRecent(store(), slug);
  renderRecent();
  const next = el('recent-list').querySelector('.recent-remove');
  (next || el('job-search')).focus();
}

function removeButton(row) {
  const button = node('button', 'recent-remove', '×');
  button.type = 'button';
  button.setAttribute('aria-label', `Remove ${sentenceCase(row.t)} from recently viewed`);
  button.addEventListener('click', () => forget(row.s));
  return button;
}

function recentItem(row) {
  const item = node('li', 'recent-item');
  const kind = rowTypeLabel(row);
  item.append(anchor(link(jobHref(row.s)), sentenceCase(row.t)));
  if (kind) item.append(node('span', 'recent-meta', kind));
  item.append(removeButton(row));
  return item;
}

function renderRecent() {
  const rows = recentRows(readRecent(store()), state.index);
  el('recent-list').replaceChildren(...rows.map(recentItem));
  el('recent-section').hidden = rows.length === 0;
}

// --- loading --------------------------------------------------------------

function errorActions() {
  const actions = node('p', 'error-actions');
  const retry = node('button', 'button', 'Try again');
  retry.type = 'button';
  retry.addEventListener('click', () => {
    el('search-error').replaceChildren();
    load();
  });
  actions.append(retry, anchor(groupsHref(), 'Browse by sector instead', 'button button--quiet'));
  return actions;
}

function showError(error) {
  el('search-loading').hidden = true;
  const block = node('div', 'error-block');
  block.append(
    node('h2', null, 'Couldn’t load the job list'),
    node('p', null, error instanceof LoadError ? error.message : String(error)),
    errorActions(),
  );
  el('search-error').replaceChildren(block);
}

function indexReady(index) {
  state.index = index;
  el('search-loading').hidden = true;
  renderChips();
  renderRecent();
  state.combo.refresh();
}

function applyStats(stats) {
  state.stats = stats;
  el('subtitle').textContent = subtitleText(stats);
  el('search-loading').textContent = loadingText(stats);
}

function load() {
  whenSlow(loadJSON('search_index'), SLOW_MS, () => { el('search-loading').hidden = false; })
    .then(indexReady)
    .catch(showError);
  // The subtitle keeps its number-free wording if this one fails.
  loadJSON('stats').then(applyStats).catch(() => {});
}

function wireLinks() {
  el('subtitle').textContent = subtitleText(null);
  el('browse-link').href = groupsHref();
  el('skill-link').href = link(skillHref(''));
}

function wireSearch() {
  const input = el('job-search');
  state.combo = createCombobox({
    input,
    listbox: el('job-listbox'),
    status: el('search-status'),
    getResults,
    renderOption,
    onSelect: (result) => openJob(result.row.s),
  });
  input.addEventListener('keydown', onEnter);
  // The combobox short-circuits an emptied box, so clear the hint here.
  input.addEventListener('input', () => { if (!input.value.trim()) setHint([]); });
}

function init() {
  renderChrome({ active: 'index' });
  wireLinks();
  renderChips();
  wireSearch();
  load();
}

init();
