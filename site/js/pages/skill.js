// "Look up a skill" — audit flow D3. Thin DOM wiring; the logic is in
// skill-model.js and the shared pure modules.
//
// The page paints in waves, so nothing waits on a file it does not need:
//   1. skill_index.json + stats -> the table, the title, the scores, the class
//   2. rubric -> "How a skill is scored", only with no skill selected
//   3. skill_occupations.json + search_index.json -> the two lists and the
//      "often appears with" chips, only once a skill IS selected
//   4. skill_notes/<xx> -> the written rationale; skill_answers_v2/<xx> -> the
//      model's own answers. Both are a few kilobytes and both are per skill.
// Nothing here loads the 15 MB portfolio file: waves 3 and 4 never start until
// a skill is open, so the list and the explainer cost one small index.

import { renderChrome } from '../chrome.js';
import { createCombobox } from '../combobox.js';
import { loadJSON, whenSlow } from '../data.js';
import { NOT_SCORED, formatCount, formatScore, isScored } from '../format.js';
import { borrowedRationale } from '../rationale.js';
import {
  SHARES, SKILL_NEAR_NOTE, isSkillNear, skillClassName, skillClassOf, thresholdOf,
} from '../scheme.js';
import { buildHash, jobHref, parseHash, skillHref, withScorer } from '../urlstate.js';
import {
  ITEM_NAMES, LIST_ID, MODE_NAMES, NO_MATCH_HINT, answerSections, askedAsKnowledge,
  classFilterOptions, classMeaning, coOccurringSkills, countNoun, defaultSort,
  filterOptions, filterSkills, listColumns, listParams, lookupIntro, neededBy, neededNote,
  occupationScores, optionMeta, pageOf, pageSummary, parseListState, probabilitySentence,
  rankSkills, rubricClasses, rubricIntro, rubricQuestions, scoreAxes, scoreBand,
  schemeOfSkillSet, scoresHeading, scoresNote, shardOf, sortListRows, sortOptions,
  sortSkillOccupations, usedIn,
} from './skill-model.js';

/** Example skills for the empty state. Every id is checked against the index. */
const EXAMPLE_IDS = ['7677a630', '4b25e9dd', 'd7cf3a29', '234750db', '5ea98e3c'];

/** How many occupations each list shows before "Show all". */
const FIRST_SHOWN = 20;

/** A wait shorter than this needs no loading state. */
const SLOW_MS = 200;

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
  scoring: document.getElementById('skill-scoring'),
  table: document.getElementById('skill-table'),
};

const state = {
  skills: null,
  byId: null,
  bySlug: null,
  occupations: null,
  stats: null,
  current: undefined,
  scheme: undefined,
  sort: 'automation',
  showAll: { essential: false, optional: false },
  columns: {},
  slots: {},
  // The all-skills table: its columns, the state its hash carries, and the
  // elements that are built once and refilled on every change.
  listColumns: [],
  list: null,
  table: null,
  watcher: null,
};

function shares() {
  return state.scheme === SHARES;
}

/** A skill class, named. The bare letter is never printed. */
function classChip(code) {
  const chip = el('span', 'class-chip', skillClassName(code));
  chip.dataset.class = code || '';
  return chip;
}

/** C12. The marker the job badge carries, for a skill that sits as close. */
function nearChip() {
  const chip = el('span', 'near-chip', 'near the line');
  chip.title = SKILL_NEAR_NOTE;
  chip.appendChild(el('span', 'visually-hidden', `. ${SKILL_NEAR_NOTE}`));
  return chip;
}

function nearSkill(row) {
  return shares() && isSkillNear(row, thresholdOf(state.stats) ?? undefined);
}

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
  dom.title.textContent = 'Skill scores';
  document.title = 'Skill scores — AI-ISCO';
  dom.meta.hidden = true;
  dom.heading.textContent = 'Look up a skill';
  dom.intro.hidden = false;
  showList();
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
  node.appendChild(el('span', 'score-word', `· ${scoreBand(value, state.scheme)}`));
  return node;
}

function scoreRow(row, axis) {
  const wrapper = el('div', 'score-row');
  wrapper.append(el('span', 'score-label', axis.label), track(row[axis.key], axis.axis),
    scoreValue(row[axis.key]));
  return wrapper;
}

/** The class, named, with what it means and the probabilities it was read from. */
function classBlock(row) {
  const code = skillClassOf(row, state.scheme);
  const block = el('div', 'skill-class');
  const head = el('p', 'skill-class-head');
  head.appendChild(classChip(code));
  if (nearSkill(row)) head.appendChild(nearChip());
  block.append(head, el('p', 'skill-class-meaning', classMeaning(code)));
  const sentence = probabilitySentence(row.p);
  if (sentence) block.appendChild(el('p', 'skill-class-probs', sentence));
  return block;
}

function scoresSection(row) {
  const node = section('scores-heading', scoresHeading(state.scheme));
  if (shares()) node.appendChild(classBlock(row));
  const list = el('div', 'score-list');
  for (const axis of scoreAxes(state.scheme)) list.appendChild(scoreRow(row, axis));
  node.append(list, el('p', 'section-note', scoresNote(state.scheme)));
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

/**
 * C2. The same rule as the job page: an explanation written for a score far
 * from the one on screen is not printed, because it argues with the class above
 * it. Where it is printed, the gap leads instead of a generic note.
 */
function rationaleBlock(note, row) {
  const verdict = borrowedRationale({
    text: note?.r, source: note?.rf, score: row?.a,
  });
  if (!verdict.show) {
    return el('p', 'muted small', verdict.why
      || 'The model wrote no explanation for this skill.');
  }
  const quote = el('blockquote', 'rationale');
  if (verdict.lead) quote.appendChild(el('p', 'rationale-lead', verdict.lead));
  quote.appendChild(el('p', null, note.r));
  quote.appendChild(el('p', 'rationale-source', verdict.after
    ? `${verdict.after} ${RATIONALE_CAVEAT}`
    : `The model’s own explanation, written as it scored the skill. ${RATIONALE_CAVEAT}`));
  return quote;
}

function loadRationale(id) {
  const fill = (node) => {
    if (state.current === id) state.slots.rationale.replaceChildren(node);
  };
  return whenSlow(loadJSON(`skill_notes/${shardOf(id)}`), SLOW_MS,
    () => fill(el('p', 'loading', 'Loading the model’s explanation…')))
    .then((notes) => fill(rationaleBlock(notes?.[id], state.byId.get(id))))
    .catch(() => fill(el('p', 'muted small',
      'The model’s explanation could not be loaded. The scores above do not depend on it.')));
}

/* --- how the model answered ------------------------------------------------ */

// The answers shards live in a directory that carries the set's own suffix
// (docs/scoring-v2.md), and data.js's loader cannot express that: it appends the
// suffix to the last segment of the name. This section exists only under the
// shares scheme and the v2 set is the one shares set published, so the directory
// is named here rather than derived. Everything else still goes through loadJSON.
const ANSWERS_DIR = 'skill_answers_v2';

const ANSWERS_NOTE = 'Each question was answered with a probability for every level rather '
  + 'than a pick. These are the model’s judgments about the skill’s ESCO description, not '
  + 'measurements of any workplace.';

const KNOWLEDGE_NOTE = 'This is a knowledge item rather than an activity, so the questions '
  + 'about AI and about machinery were put in their knowledge wording: about the work the '
  + 'knowledge is applied in.';

function answersSection() {
  const node = section('answers-heading', 'How the model answered');
  state.slots.answers = el('div', 'answers');
  node.append(el('p', 'section-note', ANSWERS_NOTE), state.slots.answers);
  return node;
}

function answerBar(bar) {
  const item = el('li', bar.modal ? 'answer-bar is-modal' : 'answer-bar');
  const track = el('span', 'answer-track');
  track.setAttribute('aria-hidden', 'true');
  const fill = el('span', 'answer-fill');
  fill.style.width = bar.percent;
  track.appendChild(fill);
  const label = el('span', 'answer-label', bar.label);
  if (bar.modal) label.appendChild(el('span', 'answer-chosen', ' — the model’s answer'));
  item.append(el('span', 'answer-value numeric', bar.percent), track, label);
  return item;
}

function confidenceLine(confidence) {
  return `How sure the model was about this question: ${confidence.word} (${confidence.percent}).`;
}

function answerBlock(block) {
  const node = el('div', 'answer');
  node.appendChild(el('h3', null, block.label));
  const list = el('ul', 'answer-bars');
  list.replaceChildren(...block.bars.map(answerBar));
  node.appendChild(list);
  if (block.confidence) {
    node.appendChild(el('p', 'answer-confidence', confidenceLine(block.confidence)));
  }
  return node;
}

const ANSWER_TABLE_COLUMNS = [
  ['Question', false], ['Answer the model was offered', false],
  ['Probability', true], ['The model’s answer', false],
];

function answerTableHead() {
  const head = el('thead');
  const row = el('tr');
  for (const [label, numeric] of ANSWER_TABLE_COLUMNS) {
    const cell = el('th', numeric ? 'numeric' : null, label);
    cell.scope = 'col';
    row.appendChild(cell);
  }
  head.appendChild(row);
  return head;
}

function answerTableRows(sections) {
  const body = el('tbody');
  for (const block of sections) {
    for (const bar of block.bars) {
      const row = el('tr');
      row.append(el('td', null, block.label), el('td', null, bar.label),
        el('td', 'numeric', bar.percent), el('td', null, bar.modal ? 'Yes' : 'No'));
      body.appendChild(row);
    }
  }
  return body;
}

function answerTable(sections) {
  const wrap = el('div', 'table-scroll');
  wrap.id = 'answers-table';
  wrap.hidden = true;
  const table = el('table');
  table.append(el('caption', null, 'Every answer the model was offered, and the probability '
    + 'it put on each.'), answerTableHead(), answerTableRows(sections));
  wrap.appendChild(table);
  return wrap;
}

function answerTableToggle(wrap) {
  const button = el('button', 'button button--quiet', 'View as table');
  button.type = 'button';
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', 'answers-table');
  button.addEventListener('click', () => {
    wrap.hidden = !wrap.hidden;
    button.setAttribute('aria-expanded', String(!wrap.hidden));
    button.textContent = wrap.hidden ? 'View as table' : 'Hide the table';
  });
  return button;
}

function answerNodes(rubric, entry) {
  const sections = answerSections(rubric, entry);
  if (!sections.length) {
    return [el('p', 'muted small', 'No stored answers for this skill.')];
  }
  const wrap = answerTable(sections);
  const nodes = sections.map(answerBlock);
  if (askedAsKnowledge(entry)) nodes.unshift(el('p', 'answers-kind', KNOWLEDGE_NOTE));
  return [...nodes, answerTableToggle(wrap), wrap];
}

function fetchAnswers(id) {
  const shard = shardOf(id);
  return fetch(`${ANSWERS_DIR}/${shard}.json`).then((response) => {
    if (!response.ok) throw new Error(`the server answered ${response.status}`);
    return response.json();
  }).then((rows) => rows[id] || null);
}

function loadAnswers(id) {
  const fill = (nodes) => {
    if (state.current === id) state.slots.answers.replaceChildren(...nodes);
  };
  const both = Promise.all([loadJSON('rubric'), fetchAnswers(id)]);
  return whenSlow(both, SLOW_MS, () => fill([el('p', 'loading', 'Loading the model’s answers…')]))
    .then(([rubric, entry]) => fill(answerNodes(rubric, entry)))
    .catch(() => fill([el('p', 'muted small', 'The model’s answers could not be loaded. '
      + 'The scores above do not depend on them.')]));
}

/* --- needed by ------------------------------------------------------------ */

function sortControl() {
  const tools = el('div', 'list-tools');
  const label = el('label', null, 'Sort by');
  label.htmlFor = 'needed-sort';
  const select = el('select');
  select.id = 'needed-sort';
  for (const sort of sortOptions(state.scheme)) {
    const option = el('option', null, sort.label);
    option.value = sort.key;
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
  const { separator, parts } = occupationScores(row, state.scheme);
  const node = el('span', `occ-scores${shares() ? ' occ-scores--shares' : ''}`);
  parts.forEach((part, index) => {
    if (index) node.append(separator);
    node.append(el('span', 'visually-hidden', part.label), part.text);
  });
  return node;
}

function occupationItem(row) {
  const item = el('li');
  const link = anchor(jobHref(row.s), undefined, 'occ-link');
  link.append(el('span', 'occ-title', row.t), el('span', 'visually-hidden', ', '),
    scorePair(row));
  item.appendChild(link);
  return item;
}

function fillColumn(kind, rows) {
  const column = state.columns[kind];
  if (!column) return;
  const sorted = sortSkillOccupations(rows, state.sort, state.scheme);
  const shown = state.showAll[kind] ? sorted : sorted.slice(0, FIRST_SHOWN);
  column.list.replaceChildren(...shown.map(occupationItem));
  if (!sorted.length) {
    column.list.appendChild(el('li', 'muted small', 'No occupation lists this skill here.'));
  }
  column.more.hidden = sorted.length <= shown.length;
  column.more.textContent = `Show all ${formatCount(sorted.length)}`;
}

function renderLists() {
  const entry = state.occupations?.[state.current];
  if (!entry || !state.bySlug) return;
  const lists = neededBy(entry, state.bySlug);
  fillColumn('essential', lists.essential);
  fillColumn('optional', lists.optional);
}

/**
 * C4. `skill_occupations` is 3.5 MB and only these two sections need it, so the
 * fetch waits until "Needed by" is about to be on screen. A browser without an
 * IntersectionObserver loads it as before.
 */
function whenNeededVisible(id, node) {
  if (state.watcher) state.watcher.disconnect();
  if (typeof window.IntersectionObserver !== 'function') {
    loadOccupations(id);
    return;
  }
  state.watcher = new window.IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    state.watcher.disconnect();
    state.watcher = null;
    if (state.current === id) loadOccupations(id);
  }, { rootMargin: '300px' });
  state.watcher.observe(node);
}

/**
 * The sections above "Needed by" fill in from their own fetches, so for the
 * first frames it sits near the top of a short page. Watching it then fires at
 * once and the deferral buys nothing, so the watch starts once they hold their
 * content.
 */
function watchNeeded(id, pending) {
  Promise.allSettled(pending).then(() => {
    const node = document.getElementById('needed-heading');
    if (state.current === id && node) whenNeededVisible(id, node);
  });
}

function neededSection(row) {
  const node = section('needed-heading', 'Needed by');
  state.slots.needed = el('p', 'loading');
  const split = el('div', 'split');
  split.append(listColumn('essential', 'Essential', row.ne),
    listColumn('optional', 'Optional', row.no));
  node.append(el('p', 'section-note', neededNote(state.scheme)),
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

/* --- how a skill is scored ------------------------------------------------- */

const METHOD_LINK = 'The full method, its limits and the two choices made after the run';

function levelList(choices) {
  const list = el('ol', 'rubric-levels');
  for (const choice of choices) list.appendChild(el('li', null, choice.text));
  return list;
}

function questionBlock(question) {
  const block = el('details', 'rubric-question');
  block.appendChild(el('summary', null, question.label));
  const body = el('div', 'rubric-body');
  body.append(el('p', 'rubric-instructions', question.instructions), levelList(question.choices));
  if (question.knowledge) {
    body.append(
      el('p', 'rubric-variant', 'A knowledge item is asked this instead:'),
      el('p', 'rubric-instructions', question.knowledge.instructions),
      levelList(question.knowledge.choices),
    );
  }
  block.appendChild(body);
  return block;
}

function classRuleRow(entry, list) {
  const term = el('dt');
  term.append(classChip(entry.code), el('span', 'rubric-rule', entry.rule));
  list.append(term, el('dd', null, entry.meaning));
}

function classRules(rubric) {
  const block = el('div', 'rubric-classes');
  block.appendChild(el('h3', null, 'What the answers make'));
  const list = el('dl', 'rubric-class-list');
  for (const entry of rubricClasses(rubric)) classRuleRow(entry, list);
  block.append(list, el('p', 'small muted', 'The first rule that fits wins, so a skill '
    + 'gets exactly one of the four.'));
  return block;
}

function scoringSection(rubric) {
  const node = section('scoring-heading', 'How a skill is scored');
  const intro = rubricIntro(rubric, state.stats?.skills_scored);
  const preamble = el('blockquote', 'rubric-preamble');
  preamble.appendChild(el('p', null, intro.preamble));
  node.append(
    el('p', 'section-note', intro.lead),
    el('p', 'small muted', 'Every question that mentions AI opens with the same preamble, '
      + 'which fixes what "the system" is:'),
    preamble,
  );
  for (const question of rubricQuestions(rubric)) node.appendChild(questionBlock(question));
  const more = el('p');
  more.appendChild(anchor('method.html', METHOD_LINK));
  node.append(classRules(rubric), el('p', 'small muted', intro.display), more);
  return node;
}

function renderScoring() {
  if (!shares()) {
    dom.scoring.replaceChildren();
    return;
  }
  loadJSON('rubric')
    .then((rubric) => dom.scoring.replaceChildren(scoringSection(rubric)))
    .catch(() => dom.scoring.replaceChildren(el('p', 'muted small',
      'The question set could not be loaded. The scores on this page do not depend on it.')));
}

/* --- all skill scores ------------------------------------------------------ */

function field(id, label, control) {
  const wrap = el('div', 'list-field');
  const tag = el('label', null, label);
  tag.htmlFor = id;
  control.id = id;
  wrap.append(tag, control);
  return wrap;
}

function filterSelect(id, label, options, onChange) {
  const select = el('select');
  for (const option of options) {
    const node = el('option', null, option.label);
    node.value = option.value;
    select.appendChild(node);
  }
  select.addEventListener('change', onChange);
  return { node: field(id, label, select), select };
}

function pushList(changes, replace) {
  const next = { ...state.list, page: 1, ...changes };
  const href = `skill.html${buildHash(LIST_ID, listParams(next, state.listColumns))}`;
  const url = withScorer(href, window.location.search);
  if (replace) window.history.replaceState({}, '', url);
  else window.history.pushState({}, '', url);
  renderList();
}

/** The direction a column sorts in when it is picked up fresh. */
function firstDir(column) {
  return column.numeric ? 'desc' : 'asc';
}

function flipDir(dir) {
  return dir === 'asc' ? 'desc' : 'asc';
}

/** Clicking the column that already sorts, still in its first direction, reverses it. */
function nextDir(column) {
  const first = firstDir(column);
  if (state.list.sort !== column.key || state.list.dir !== first) return first;
  return flipDir(first);
}

function sortBy(column) {
  pushList({ sort: column.key, dir: nextDir(column) });
  const header = state.table.head.querySelector(`th[data-key="${column.key}"] button`);
  if (header) header.focus();
}

function headerCell(column) {
  const cell = el('th', column.numeric ? 'numeric' : null);
  cell.scope = 'col';
  cell.dataset.key = column.key;
  cell.setAttribute('aria-sort', 'none');
  const button = el('button', 'sort-button', column.label);
  button.type = 'button';
  button.addEventListener('click', () => sortBy(column));
  cell.appendChild(button);
  return cell;
}

function listCell(column, row, first) {
  const cell = el(first ? 'th' : 'td', column.numeric ? 'numeric' : null);
  if (first) {
    cell.scope = 'row';
    const link = anchor(skillHref(row.id), row.t);
    link.dataset.skillLink = '';
    cell.appendChild(link);
  } else if (column.key === 'c') {
    cell.appendChild(classChip(row.c));
    if (nearSkill(row)) cell.appendChild(nearChip());
  } else {
    cell.textContent = column.text(row);
  }
  return cell;
}

function listRow(row) {
  const node = el('tr');
  state.listColumns.forEach((column, index) => {
    node.appendChild(listCell(column, row, index === 0));
  });
  return node;
}

function pagerButton(label, step) {
  const button = el('button', 'button button--quiet', label);
  button.type = 'button';
  button.addEventListener('click', () => {
    const page = state.list.page + step;
    pushList({ page });
    if (button.disabled) state.table.summary.focus();
  });
  return button;
}

function buildPager() {
  const pager = el('div', 'list-pager');
  const previous = pagerButton('← Previous', -1);
  const next = pagerButton('Next →', 1);
  const label = el('p', 'list-page', '');
  pager.append(previous, label, next);
  return { pager, previous, next, label };
}

function searchBox() {
  const search = el('input');
  search.type = 'search';
  search.autocomplete = 'off';
  search.placeholder = 'Part of a skill name';
  return search;
}

/** Only a shares set has classes, kinds and modes to filter on. */
function filterSpecs(rows) {
  if (!shares()) return [];
  return [
    ['class', 'What AI can do with it', classFilterOptions(rows)],
    ['item', 'Skill or knowledge', filterOptions(rows, 'ty', ITEM_NAMES)],
    ['mode', 'How it is exercised', filterOptions(rows, 'mo', MODE_NAMES)],
  ].filter(([, , options]) => options.length > 1);
}

function addFilter(bar, controls, [key, label, options]) {
  const stateKey = key === 'class' ? 'cls' : key;
  const built = filterSelect(`list-${key}`, label, options,
    () => pushList({ [stateKey]: built.select.value }));
  controls.selects[key] = built.select;
  bar.appendChild(built.node);
}

function buildFilters(rows) {
  const controls = { search: searchBox(), selects: {} };
  const bar = el('div', 'list-tools');
  bar.appendChild(field('list-query', 'Filter by name', controls.search));
  for (const spec of filterSpecs(rows)) addFilter(bar, controls, spec);
  return { bar, controls };
}

/** The count above the table is also what a screen reader hears after a filter. */
function buildSummary() {
  const summary = el('p', 'list-summary');
  summary.tabIndex = -1;
  summary.setAttribute('role', 'status');
  summary.setAttribute('aria-live', 'polite');
  return summary;
}

function buildGrid() {
  const wrap = el('div', 'table-scroll');
  const table = el('table', 'list-table');
  const head = el('thead');
  const row = el('tr');
  row.replaceChildren(...state.listColumns.map(headerCell));
  head.appendChild(row);
  const body = el('tbody');
  table.append(head, body);
  wrap.appendChild(table);
  return { wrap, head, body };
}

function buildTable(rows) {
  const node = section('list-heading', 'All skill scores');
  node.appendChild(el('p', 'section-note', 'Every scored ESCO skill. Filter it, sort it by '
    + 'any column, and open one to see how the model answered.'));
  const { bar, controls } = buildFilters(rows);
  const summary = buildSummary();
  const { wrap, head, body } = buildGrid();
  const pager = buildPager();
  node.append(bar, summary, wrap, pager.pager);
  return { node, controls, summary, head, body, pager };
}

let typingTimer = null;

function onTyping() {
  window.clearTimeout(typingTimer);
  typingTimer = window.setTimeout(
    () => pushList({ query: state.table.controls.search.value.trim() }, true), 250,
  );
}

function syncControls() {
  const { search, selects } = state.table.controls;
  if (document.activeElement !== search) search.value = state.list.query;
  const values = { class: state.list.cls, item: state.list.item, mode: state.list.mode };
  for (const [key, select] of Object.entries(selects)) {
    if (select.value !== values[key]) select.value = values[key];
  }
}

/** The aria-sort value one header cell carries. */
function sortAttr(active) {
  if (!active) return 'none';
  return state.list.dir === 'asc' ? 'ascending' : 'descending';
}

function markSort() {
  for (const cell of state.table.head.querySelectorAll('th')) {
    cell.setAttribute('aria-sort', sortAttr(cell.dataset.key === state.list.sort));
  }
}

function renderList() {
  const { params } = parseHash(window.location.hash);
  state.list = parseListState(params, state.listColumns);
  syncControls();
  const rows = sortListRows(filterSkills(state.skills, state.list),
    state.listColumns, state.list);
  const view = pageOf(rows, state.list.page);
  state.table.summary.textContent = pageSummary(view);
  state.table.body.replaceChildren(...view.rows.map(listRow));
  state.table.pager.previous.disabled = view.page <= 1;
  state.table.pager.next.disabled = view.page >= view.pages;
  state.table.pager.label.textContent = `Page ${formatCount(view.page)} of `
    + `${formatCount(view.pages)}`;
  markSort();
}

function showList() {
  if (!state.table) {
    state.listColumns = listColumns(state.scheme, state.skills);
    state.table = buildTable(state.skills);
    state.table.controls.search.addEventListener('input', onTyping);
    dom.table.replaceChildren(state.table.node);
    renderScoring();
  }
  dom.table.hidden = false;
  dom.scoring.hidden = false;
  renderList();
}

/* --- routing -------------------------------------------------------------- */

function detailSections(row) {
  const sections = [scoresSection(row), whySection()];
  if (shares()) sections.push(answersSection());
  return [...sections, neededSection(row), relatedSection()];
}

function showSkill(id) {
  dom.heading.textContent = 'Look up another skill';
  dom.intro.hidden = true;
  dom.table.hidden = true;
  dom.scoring.hidden = true;
  const row = state.byId?.get(id);
  if (!row) {
    showUnknown(id);
    return;
  }
  dom.title.textContent = row.t;
  document.title = `${row.t} — AI-ISCO`;
  dom.meta.textContent = 'ESCO skill · used in '
    + countNoun(usedIn(row), 'occupation', 'occupations');
  dom.meta.hidden = false;
  dom.detail.replaceChildren(...detailSections(row));
  watchNeeded(id, [loadRationale(id), shares() ? loadAnswers(id) : null]);
}

/** The hash carries either a skill id or the state of the all-skills list. */
function skillInHash() {
  const { id } = parseHash(window.location.hash);
  return id && id !== LIST_ID ? id : null;
}

function render() {
  const id = skillInHash();
  if (id === state.current) {
    if (!id) renderList();
    return;
  }
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
    el('span', 'code', optionMeta(result.row, state.scheme)));
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

function receiveIndex([rows, stats]) {
  state.skills = rows;
  state.byId = new Map(rows.map((row) => [row.id, row]));
  state.stats = stats;
  state.scheme = schemeOfSkillSet(stats, rows);
  state.sort = defaultSort(state.scheme);
  dom.intro.textContent = lookupIntro(state.scheme);
  setStatus('');
  setupSearch();
  state.current = undefined;
  render();
}

function loadIndex() {
  dom.error.replaceChildren();
  const sources = Promise.all([
    loadJSON('skill_index'), loadJSON('stats').catch(() => null),
  ]);
  whenSlow(sources, SLOW_MS, () => setStatus('Loading the skill list…'))
    .then(receiveIndex)
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
