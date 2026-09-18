// "Look up a skill" — audit flow D3. Thin DOM wiring; the logic is in
// skill-model.js and the shared pure modules.
//
// The page paints in three waves, so nothing waits on a file it does not need:
//   1. skill_index.json  -> title, both scores, how many occupations need it
//   2. skill_occupations.json + search_index.json -> the two lists and the
//      "often appears with" chips
//   3. portfolio_data.json -> the written rationale, which lives nowhere else
// Wave 3 is 13 MB for one sentence, so it never blocks the first paint and its
// failure costs the visitor nothing but that sentence.

import { renderChrome } from '../chrome.js';
import { createCombobox } from '../combobox.js';
import { loadJSON, whenSlow } from '../data.js';
import { NOT_SCORED, formatCount, formatScore, isScored } from '../format.js';
import { renderInfoNote } from '../info-note.js';
import { borrowedRationaleNote } from '../rationale.js';
import { SORTS, sortOccupations } from '../groupstats.js';
import { exposureWord } from '../quadrant.js';
import { jobHref, parseHash, skillHref, withScorer } from '../urlstate.js';
import {
  NO_MATCH_HINT, coOccurringSkills, neededBy, rankSkills, usedIn,
} from './skill-model.js';

/** Example skills for the empty state. Every id is checked against the index. */
const EXAMPLE_IDS = ['7677a630', '4b25e9dd', 'd7cf3a29', '234750db', '5ea98e3c'];

/** How many occupations each list shows before "Show all". */
const FIRST_SHOWN = 20;

/** A wait shorter than this needs no loading state. */
const SLOW_MS = 200;

const AXES = [
  { key: 'a', axis: 'automation', label: 'Automation risk' },
  { key: 'm', axis: 'amplification', label: 'Amplification' },
];

const dom = {
  title: document.getElementById('skill-title'),
  meta: document.getElementById('skill-meta'),
  status: document.getElementById('skill-status'),
  error: document.getElementById('skill-error'),
  detail: document.getElementById('skill-detail'),
  heading: document.getElementById('lookup-heading'),
  intro: document.getElementById('lookup-intro'),
  input: document.getElementById('skill-input'),
  listbox: document.getElementById('skill-listbox'),
  comboStatus: document.getElementById('skill-combobox-status'),
  hint: document.getElementById('skill-hint'),
  examples: document.getElementById('skill-examples'),
};

const state = {
  skills: null,
  byId: null,
  bySlug: null,
  occupations: null,
  current: undefined,
  sort: 'automation',
  showAll: { essential: false, optional: false },
  columns: {},
  slots: {},
};

/* --- small DOM helpers ---------------------------------------------------- */

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

function section(id, title) {
  const node = el('section', 'section');
  node.setAttribute('aria-labelledby', id);
  const heading = el('h2', null, title);
  heading.id = id;
  node.appendChild(heading);
  return node;
}

function skillChip(row) {
  const chip = anchor(skillHref(row.id), row.t, 'chip');
  chip.dataset.skillLink = '';
  return chip;
}

function setStatus(text) {
  dom.status.textContent = text || '';
  dom.status.hidden = !text;
}

/* --- states --------------------------------------------------------------- */

function showError(error, retry) {
  const block = el('div', 'error-block');
  block.append(el('h2', null, 'That did not load'), el('p', null, error.message));
  const actions = el('p', 'inline-actions');
  const again = el('button', 'button button--quiet', 'Try again');
  again.type = 'button';
  again.addEventListener('click', retry);
  actions.append(again, anchor('index.html', 'Find a job instead'));
  block.appendChild(actions);
  dom.error.replaceChildren(block);
}

function showUnknown(id) {
  dom.title.textContent = 'No skill with that id';
  document.title = 'Skill not found — AI-ISCO';
  dom.meta.hidden = true;
  const block = el('div', 'error-block');
  block.append(
    el('h2', null, `Nothing here has the id “${id}”.`),
    el('p', null, 'The link may be out of date. Search for the skill by its ESCO name below.'),
  );
  dom.detail.replaceChildren(block);
}

function showEmpty() {
  dom.title.textContent = 'Which skill do you want to look up?';
  document.title = 'Look up a skill — AI-ISCO';
  dom.meta.hidden = true;
  dom.heading.textContent = 'Search ESCO skills';
  dom.heading.classList.add('visually-hidden');
  dom.intro.hidden = false;
}

/* --- the scores ----------------------------------------------------------- */

function track(value, axis) {
  const outer = el('span', 'score-track');
  outer.setAttribute('aria-hidden', 'true');
  const fill = el('span', 'score-fill');
  fill.dataset.axis = axis;
  fill.style.width = `${isScored(value) ? Math.max(0, Math.min(10, value)) * 10 : 0}%`;
  outer.appendChild(fill);
  return outer;
}

function scoreValue(value) {
  const node = el('span', 'score-value');
  if (!isScored(value)) {
    node.textContent = NOT_SCORED;
    return node;
  }
  node.append(`${formatScore(value)} out of 10 `);
  node.appendChild(el('span', 'score-word', `· ${exposureWord(value)}`));
  return node;
}

function scoreRow(row, axis) {
  const wrapper = el('div', 'score-row');
  wrapper.append(el('span', 'score-label', axis.label), track(row[axis.key], axis.axis),
    scoreValue(row[axis.key]));
  return wrapper;
}

function scoresSection(row) {
  const node = section('scores-heading', 'Both scores');
  const list = el('div', 'score-list');
  for (const axis of AXES) list.appendChild(scoreRow(row, axis));
  node.append(list, el('p', 'section-note',
    'Automation risk is how much of this skill a machine could take over. Amplification is '
    + 'how much more someone could do with AI helping. Both are model estimates on a 1 to 10 '
    + 'scale, read off the skill’s ESCO description.'));
  return node;
}

/* --- the rationale -------------------------------------------------------- */

function whySection() {
  const node = section('why-heading', 'Why these scores');
  state.slots.rationale = el('div');
  node.appendChild(state.slots.rationale);
  return node;
}

const RATIONALE_CAVEAT = 'It is reasoning about the description, not a measurement of any workplace.';

/** Who wrote the explanation: the model on screen, or the one it was borrowed from. */
function rationaleSource(borrowed) {
  if (!borrowed) {
    return [el('p', 'rationale-source',
      `The model’s own explanation, written as it scored the skill. ${RATIONALE_CAVEAT}`)];
  }
  const { button, note } = renderInfoNote(borrowed);
  const source = el('p', 'rationale-source',
    `Written by ${borrowed.writer} as it scored the skill. ${RATIONALE_CAVEAT} `);
  source.appendChild(button);
  return [source, note];
}

function rationaleBlock(data, id) {
  const skill = data && data.skills && data.skills[id];
  if (!skill || !skill.r) {
    return el('p', 'muted small', 'The model wrote no explanation for this skill.');
  }
  const quote = el('blockquote', 'rationale');
  quote.append(el('p', null, skill.r), ...rationaleSource(borrowedRationaleNote(skill.rf)));
  return quote;
}

function loadRationale(id) {
  const fill = (node) => {
    if (state.current === id) state.slots.rationale.replaceChildren(node);
  };
  whenSlow(loadJSON('portfolio_data'), SLOW_MS,
    () => fill(el('p', 'loading', 'Loading the model’s explanation…')))
    .then((data) => fill(rationaleBlock(data, id)))
    .catch(() => fill(el('p', 'muted small',
      'The model’s explanation could not be loaded. The scores above do not depend on it.')));
}

/* --- needed by ------------------------------------------------------------ */

function sortControl() {
  const tools = el('div', 'list-tools');
  const label = el('label', null, 'Sort by');
  label.htmlFor = 'needed-sort';
  const select = el('select');
  select.id = 'needed-sort';
  for (const [key, sort] of Object.entries(SORTS)) {
    const option = el('option', null, sort.label);
    option.value = key;
    select.appendChild(option);
  }
  select.value = state.sort;
  select.addEventListener('change', () => {
    state.sort = select.value;
    renderLists();
  });
  tools.append(label, select);
  return tools;
}

function listColumn(kind, label, count) {
  const column = el('div');
  const list = el('ul', 'occ-list');
  const more = el('button', 'button button--quiet', 'Show all');
  more.type = 'button';
  more.hidden = true;
  more.addEventListener('click', () => {
    state.showAll[kind] = true;
    renderLists();
    list.querySelector('a')?.focus();
  });
  column.append(el('h3', null, `${label} (${formatCount(count || 0)})`), list, more);
  state.columns[kind] = { list, more };
  return column;
}

function scorePair(row) {
  const node = el('span', 'occ-scores');
  node.append(el('span', 'visually-hidden', 'automation '), formatScore(row.a), ' / ',
    el('span', 'visually-hidden', 'amplification '), formatScore(row.m));
  return node;
}

function occupationItem(row) {
  const item = el('li');
  const link = anchor(jobHref(row.s), undefined, 'occ-link');
  link.append(el('span', 'occ-title', row.t), scorePair(row));
  item.appendChild(link);
  return item;
}

function fillColumn(kind, rows) {
  const column = state.columns[kind];
  if (!column) return;
  const sorted = sortOccupations(rows, state.sort);
  const shown = state.showAll[kind] ? sorted : sorted.slice(0, FIRST_SHOWN);
  column.list.replaceChildren(...shown.map(occupationItem));
  if (!sorted.length) {
    column.list.appendChild(el('li', 'muted small', 'No occupation lists this skill here.'));
  }
  column.more.hidden = sorted.length <= shown.length;
  column.more.textContent = `Show all ${formatCount(sorted.length)}`;
}

function renderLists() {
  const entry = state.occupations && state.occupations[state.current];
  if (!entry || !state.bySlug) return;
  const lists = neededBy(entry, state.bySlug);
  fillColumn('essential', lists.essential);
  fillColumn('optional', lists.optional);
}

function neededSection(row) {
  const node = section('needed-heading', 'Needed by');
  state.slots.needed = el('p', 'loading');
  const split = el('div', 'split');
  split.append(listColumn('essential', 'Essential', row.ne),
    listColumn('optional', 'Optional', row.no));
  node.append(el('p', 'section-note', 'Essential skills are required for the occupation; '
    + 'optional ones are common but not required. Scores are automation / amplification.'),
  sortControl(), state.slots.needed, split);
  return node;
}

/* --- often appears with --------------------------------------------------- */

function relatedSection() {
  const node = section('related-heading', 'Often appears with');
  state.slots.related = el('div', 'chip-row');
  node.append(el('p', 'section-note',
    'The skills that turn up most often in the same occupations as this one.'),
  state.slots.related);
  return node;
}

function renderRelated(id) {
  const chips = coOccurringSkills(state.occupations, id, { limit: 8 })
    .map((item) => state.byId.get(item.id))
    .filter(Boolean)
    .map(skillChip);
  state.slots.related.replaceChildren(...(chips.length ? chips
    : [el('p', 'muted small', 'No other skill shares these occupations.')]));
}

function loadOccupations(id) {
  const both = Promise.all([loadJSON('skill_occupations'), loadJSON('search_index')]);
  whenSlow(both, SLOW_MS, () => {
    if (state.current === id) {
      state.slots.needed.textContent = 'Loading the occupations that need this skill…';
    }
  }).then(([occupations, index]) => {
    state.occupations = occupations;
    state.bySlug = new Map(index.map((row) => [row.s, row]));
    if (state.current !== id) return;
    state.slots.needed.textContent = '';
    renderLists();
    renderRelated(id);
  }).catch((error) => {
    if (state.current === id) state.slots.needed.textContent = `${error.message}`;
  });
}

/* --- routing -------------------------------------------------------------- */

function showSkill(id) {
  dom.heading.textContent = 'Look up another skill';
  dom.heading.classList.remove('visually-hidden');
  dom.intro.hidden = true;
  const row = state.byId && state.byId.get(id);
  if (!row) {
    showUnknown(id);
    return;
  }
  dom.title.textContent = row.t;
  document.title = `${row.t} — AI-ISCO`;
  dom.meta.textContent = `ESCO skill · used in ${formatCount(usedIn(row))} occupations`;
  dom.meta.hidden = false;
  dom.detail.replaceChildren(scoresSection(row), whySection(), neededSection(row),
    relatedSection());
  loadOccupations(id);
  loadRationale(id);
}

function render() {
  const { id } = parseHash(window.location.hash);
  if (id === state.current) return;
  state.current = id;
  state.showAll = { essential: false, optional: false };
  state.columns = {};
  dom.error.replaceChildren();
  dom.detail.replaceChildren();
  if (id) showSkill(id);
  else showEmpty();
}

function go(href) {
  window.history.pushState({}, '', withScorer(href, window.location.search));
  render();
  dom.title.focus();
}

function onDocumentClick(event) {
  const link = event.target.closest ? event.target.closest('a[data-skill-link]') : null;
  if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
  event.preventDefault();
  go(link.getAttribute('href'));
}

/* --- search --------------------------------------------------------------- */

function paintOption(result, node) {
  node.append(el('span', 'occ-title', result.row.t),
    el('span', 'code', `${formatScore(result.row.a)} / ${formatScore(result.row.m)}`));
}

function updateHint() {
  const query = dom.input.value.trim();
  const none = query.length > 0 && rankSkills(state.skills, query, 1).length === 0;
  dom.hint.textContent = none ? `No skill called “${query}”. ${NO_MATCH_HINT}` : '';
}

function setupSearch() {
  dom.examples.replaceChildren(...EXAMPLE_IDS
    .map((id) => state.byId.get(id)).filter(Boolean).map(skillChip));
  createCombobox({
    input: dom.input,
    listbox: dom.listbox,
    status: dom.comboStatus,
    getResults: (query) => rankSkills(state.skills, query, 12),
    renderOption: paintOption,
    onSelect: (result) => go(skillHref(result.row.id)),
  });
  dom.input.addEventListener('input', updateHint);
}

/* --- boot ----------------------------------------------------------------- */

function loadIndex() {
  dom.error.replaceChildren();
  whenSlow(loadJSON('skill_index'), SLOW_MS, () => setStatus('Loading the skill list…'))
    .then((rows) => {
      state.skills = rows;
      state.byId = new Map(rows.map((row) => [row.id, row]));
      setStatus('');
      setupSearch();
      state.current = undefined;
      render();
    })
    .catch((error) => {
      setStatus('');
      showError(error, loadIndex);
    });
}

renderChrome({ active: 'skill' });
setStatus('');
window.addEventListener('hashchange', render);
window.addEventListener('popstate', render);
document.addEventListener('click', onDocumentClick);
loadIndex();
