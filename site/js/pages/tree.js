// tree.html — DOM wiring only.
//
// The tree itself lives in tree-view.js and every string, number and ordering
// in tree-model.js. This file loads the three small index files, keeps the URL
// as the state (#job=<slug> or #g=<level>:<code>, plus &q=<filter>), and fills
// the detail pane beside the tree.
//
// portfolio_data.json is 13 MB and only the skills table needs it, so it is
// fetched when the browser first goes idle *after* the tree has painted. The
// tree, the filter and every group detail work without it.

import { renderQuadrantBadge } from '../badge.js';
import { renderChrome } from '../chrome.js';
import { loadJSON, whenSlow } from '../data.js';
import { formatCount, formatScore, isScored } from '../format.js';
import { groupHref, jobHref, skillHref, withScorer } from '../urlstate.js';
import * as model from './tree-model.js';
import { TreeView } from './tree-view.js';

/** Below this width the detail pane sits under the tree, not beside it. */
const NARROW = '(max-width: 860px)';

const SCORE_CARDS = [
  { key: 'a', modifier: 'auto', label: 'Automation risk' },
  { key: 'm', modifier: 'amp', label: 'Amplification' },
];

const state = {
  model: null,
  stats: null,
  expanded: new Set(),
  selection: null,
  query: '',
  filter: null,
  portfolio: null,
  portfolioError: null,
  portfolioPending: false,
  skills: null,
  skillSort: { key: 'essential', descending: true },
  skillFilter: 'all',
};

let tree = null;

/* --- tiny DOM helpers ----------------------------------------------------- */

const byId = (id) => document.getElementById(id);

function apply(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'class') node.setAttribute('class', value);
    else node.setAttribute(key, value);
  }
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  apply(node, props);
  for (const child of [].concat(children)) if (child !== null) node.append(child);
  return node;
}

function link(href, text, props = {}) {
  return el('a', { href: withScorer(href, window.location.search), text, ...props });
}

function show(node, visible) {
  node.hidden = !visible;
}

function isNarrow() {
  return window.matchMedia(NARROW).matches;
}

/* --- boot ----------------------------------------------------------------- */

function failed(error) {
  show(byId('tree-status'), false);
  const retry = el('button', { type: 'button', class: 'button', text: 'Try again' });
  retry.addEventListener('click', () => window.location.reload());
  byId('tree-error').replaceChildren(
    el('h2', { text: 'We could not load the tree' }),
    el('p', { text: error.message }),
    el('p', { text: 'Nothing on your side is wrong. Try again, or start from a job title.' }),
    el('p', {}, [retry, ' ', link('index.html', 'Find a job instead', { class: 'button button--quiet' })]),
  );
  show(byId('tree-error'), true);
}

function ready(files) {
  const [groups, index, stats] = files;
  state.model = model.buildModel(groups, index);
  state.stats = stats;
  show(byId('tree-status'), false);
  show(byId('tree-layout'), true);
  onRoute();
  schedulePrefetch();
}

function boot() {
  renderChrome({ active: 'tree' });
  tree = new TreeView(byId('tree'), { onActivate, onToggle });
  byId('tree-q').addEventListener('input', onFilterInput);
  byId('tree-collapse').addEventListener('click', collapseAll);
  byId('detail-back').addEventListener('click', backToTree);
  window.addEventListener('popstate', onRoute);
  window.addEventListener('hashchange', onRoute);
  const pending = Promise.all([loadJSON('groups'), loadJSON('search_index'), loadJSON('stats')]);
  whenSlow(pending, 200, () => show(byId('tree-status'), true)).then(ready).catch(failed);
}

/* --- routing -------------------------------------------------------------- */

function onRoute() {
  if (!state.model) return;
  const here = model.readState(window.location.hash);
  state.selection = here.selection;
  if (here.query !== state.query) applyQuery(here.query);
  for (const key of model.ancestorKeys(state.model, here.selection)) state.expanded.add(key);
  renderTree();
  renderDetail();
  const id = model.selectionId(state.selection);
  if (id) tree.reveal(id);
}

function pushState() {
  const hash = model.treeHash(state.selection, state.query);
  if (hash === window.location.hash) return;
  window.history.pushState(null, '',
    withScorer(model.treeHref(state.selection, state.query), window.location.search));
}

function select(selection) {
  state.selection = selection;
  renderTree();
  renderDetail();
  pushState();
  if (isNarrow()) focusDetail();
}

function onActivate(row) {
  if (row.kind === 'job') {
    select({ kind: 'job', id: row.slug });
    return;
  }
  if (row.expanded) state.expanded.delete(row.key);
  else state.expanded.add(row.key);
  select({ kind: 'group', id: row.key });
}

function onToggle(key, open) {
  if (open) state.expanded.add(key);
  else state.expanded.delete(key);
  renderTree();
  tree.reveal(key, { focus: true });
}

function collapseAll() {
  state.expanded.clear();
  renderTree();
  byId('tree-collapse').focus();
}

function focusDetail() {
  const pane = byId('detail');
  pane.focus();
  pane.scrollIntoView({ block: 'start' });
}

function backToTree() {
  const id = model.selectionId(state.selection);
  if (id && tree.indexOf(id) !== -1) tree.reveal(id, { focus: true });
  else byId('tree-q').focus();
}

/* --- the tree ------------------------------------------------------------- */

function renderTree() {
  tree.setRows(model.visibleRows(state.model, {
    expanded: state.expanded,
    filter: state.filter,
    selectedId: model.selectionId(state.selection),
  }));
}

function applyQuery(query) {
  state.query = query;
  state.filter = model.filterMatches(state.model, query);
  byId('tree-q').value = query;
  byId('tree-count').textContent = model.matchMessage(state.filter);
}

function onFilterInput() {
  applyQuery(byId('tree-q').value);
  renderTree();
  const href = withScorer(model.treeHref(state.selection, state.query), window.location.search);
  window.history.replaceState(null, '', href);
}

/* --- detail: shared pieces ------------------------------------------------ */

function mixRow(segment) {
  const width = segment.count ? Math.max(segment.share * 100, 1.5) : 0;
  return el('li', { 'data-quadrant': segment.code }, [
    el('span', { class: 'mix-name', text: segment.label }),
    el('span', { class: 'mix-track' }, [el('span', { class: 'mix-fill', style: `width: ${width}%` })]),
    el('span', { class: 'mix-figure', text: `${segment.text} · ${segment.percent}` }),
  ]);
}

function mixList(segments) {
  return el('ul', { class: 'mix-bars' }, segments.map(mixRow));
}

function caveat() {
  return el('p', { class: 'mix-caveat small', text: model.nearLineCaveat(state.stats) });
}

function setTitle(text, documentTitle) {
  byId('detail-title').textContent = text;
  document.title = `${documentTitle} — Job tree — AI-ISCO`;
}

function renderDetail() {
  const host = byId('detail-body');
  if (!state.selection) renderEmptyDetail(host);
  else if (state.selection.kind === 'group') renderGroupDetail(host);
  else renderJobDetail(host);
}

/* --- detail: nothing selected --------------------------------------------- */

function renderEmptyDetail(host) {
  setTitle('Pick a job in the tree to see its skills', 'Job tree');
  const counts = (state.stats && state.stats.quadrants && state.stats.quadrants.counts) || {};
  host.replaceChildren(
    el('p', { text: 'Open a group to walk down ISCO-08. Selecting a group shows how its jobs '
      + 'split; selecting a job shows its skills and where each one sits.' }),
    el('h3', { text: `How all ${formatCount(state.stats.occupations)} jobs split` }),
    mixList(model.mixSegments(counts)),
    caveat(),
  );
}

/* --- detail: a group ------------------------------------------------------ */

function renderGroupDetail(host) {
  const key = state.selection.id;
  const group = state.model.groups[key];
  if (!group) {
    setTitle('Group not found', 'Group not found');
    host.replaceChildren(el('div', { class: 'error-block' }, [
      el('p', { text: `We don't have a group called “${key}”. Pick one from the tree.` }),
    ]));
    return;
  }
  setTitle(group.label, group.label);
  host.replaceChildren(
    el('p', { class: 'muted small', text: model.groupSubtitle(group) }),
    el('h3', { text: 'How this group splits' }),
    mixList(model.mixSegments(group.q)),
    caveat(),
    el('p', {}, [link(groupHref(key), 'Open this sector', { class: 'button button--quiet' })]),
  );
}

/* --- detail: a job -------------------------------------------------------- */

function scoreCard(spec, row) {
  const value = row[spec.key];
  const percent = isScored(value) ? ((Math.min(10, Math.max(1, value)) - 1) / 9) * 100 : 0;
  return el('div', { class: `score-card score-card--${spec.modifier}` }, [
    el('p', { class: 'score-label', text: `${spec.label}, out of 10` }),
    el('p', {
      class: isScored(value) ? 'score-value numeric' : 'score-value is-missing',
      text: formatScore(value),
    }),
    el('span', { class: 'score-bar', 'aria-hidden': 'true' },
      [el('span', { class: 'score-fill', style: `width: ${percent.toFixed(1)}%` })]),
  ]);
}

function unitLine(row) {
  const key = `unit:${String(row.c || '')}`;
  const group = state.model.groups[key];
  if (!group) return el('p', { class: 'muted small', text: `ISCO-08 code ${row.c}` });
  return el('p', { class: 'muted small' }, [
    'ISCO-08 unit group ',
    link(model.treeHref({ kind: 'group', id: key }, state.query), `${group.label} (${key.slice(5)})`),
  ]);
}

function jobHeader(row) {
  const badge = renderQuadrantBadge({ t: row.t, a: row.a, m: row.m, q: row.q },
    { search: window.location.search });
  return [
    unitLine(row),
    el('div', { class: 'badge-line' }, [badge]),
    el('div', { class: 'score-grid' }, SCORE_CARDS.map((spec) => scoreCard(spec, row))),
  ];
}

function renderJobDetail(host) {
  const row = state.model.bySlug.get(state.selection.id);
  if (!row) {
    setTitle('Job not found', 'Job not found');
    host.replaceChildren(el('div', { class: 'error-block' }, [
      el('p', { text: `We don't have a job with the slug “${state.selection.id}”.` }),
    ]));
    return;
  }
  setTitle(row.t, row.t);
  const skills = el('div', { id: 'skills-block' });
  host.replaceChildren(
    ...jobHeader(row),
    el('h3', { text: 'Skills in this job' }),
    skills,
    el('p', {}, [link(jobHref(row.s, { from: `unit:${row.c}` }), 'Open the full job page',
      { class: 'button button--quiet' })]),
  );
  renderSkills(row);
}

/* --- the big file --------------------------------------------------------- */

function startPortfolio() {
  if (state.portfolio || state.portfolioPending) return;
  state.portfolioPending = true;
  state.portfolioError = null;
  loadJSON('portfolio_data')
    .then((data) => { state.portfolio = data; })
    .catch((error) => { state.portfolioError = error; })
    .finally(() => {
      state.portfolioPending = false;
      state.skills = null;
      refreshSkills();
    });
}

function schedulePrefetch() {
  const start = () => {
    if (typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(startPortfolio, { timeout: 4000 });
    } else {
      window.setTimeout(startPortfolio, 1200);
    }
  };
  window.requestAnimationFrame(() => window.setTimeout(start, 0));
}

function refreshSkills() {
  if (!state.selection || state.selection.kind !== 'job') return;
  const row = state.model.bySlug.get(state.selection.id);
  if (row && byId('skills-block')) renderSkills(row);
}

function skillsError(host) {
  const retry = el('button', { type: 'button', class: 'button button--quiet', text: 'Retry' });
  retry.addEventListener('click', startPortfolio);
  host.replaceChildren(el('div', { class: 'error-block' }, [
    el('p', { text: state.portfolioError.message }),
    el('p', { text: 'The scores above still stand. The full job page has the same skills.' }),
    el('p', {}, [retry]),
  ]));
}

function currentSkills(row) {
  if (state.skills && state.skills.slug === row.s) return state.skills.rows;
  const occupation = model.findOccupation(state.portfolio, row.s);
  state.skills = { slug: row.s, rows: model.skillRows(state.portfolio, occupation) };
  return state.skills.rows;
}

function renderSkills(row) {
  const host = byId('skills-block');
  if (state.portfolioError) {
    skillsError(host);
    return;
  }
  if (!state.portfolio) {
    startPortfolio();
    host.replaceChildren(el('p', { class: 'loading', text: 'Loading the skills. They live in one '
      + 'large file, so this can take a few seconds…' }));
    return;
  }
  const rows = currentSkills(row);
  if (!rows.length) {
    host.replaceChildren(el('p', { class: 'muted', text: 'We have no skill list for this job.' }));
    return;
  }
  host.replaceChildren(...skillSections(rows));
}

function skillSections(rows) {
  const mix = model.skillMix(rows);
  const shown = model.filterSkillRows(rows, state.skillFilter);
  return [
    el('p', { class: 'small muted', text: model.skillSummary(mix) }),
    el('h4', { text: 'Where the skills sit' }),
    mixList(mix.bars),
    el('h4', { text: 'Every skill' }),
    filterToggle(),
    el('div', { class: 'table-scroll' }, [skillTable(shown, rows.length)]),
  ];
}

/* --- the skills table ----------------------------------------------------- */

function filterButton(option) {
  const button = el('button', {
    type: 'button',
    class: 'chip',
    'aria-pressed': String(state.skillFilter === option.key),
    text: option.label,
  });
  button.addEventListener('click', () => {
    state.skillFilter = option.key;
    refreshSkills();
    const again = [...byId('skills-block').querySelectorAll('[aria-pressed]')]
      .find((node) => node.textContent === option.label);
    if (again) again.focus();
  });
  return button;
}

function filterToggle() {
  return el('div', { class: 'skill-toolbar' }, [
    el('span', { class: 'section-heading', id: 'skill-filter-label', text: 'Show' }),
    el('div', { class: 'chip-row', role: 'group', 'aria-labelledby': 'skill-filter-label' },
      model.SKILL_FILTERS.map(filterButton)),
  ]);
}

function sortBy(column) {
  const same = state.skillSort.key === column.key;
  state.skillSort = {
    key: column.key,
    descending: same ? !state.skillSort.descending : Boolean(column.numeric),
  };
  refreshSkills();
  const header = byId('skills-block').querySelector(`th[data-key="${column.key}"] button`);
  if (header) header.focus();
}

function headerCell(column) {
  const sorted = state.skillSort.key === column.key;
  const cell = el('th', {
    scope: 'col',
    'data-key': column.key,
    class: column.numeric ? 'numeric' : null,
    'aria-sort': sorted ? (state.skillSort.descending ? 'descending' : 'ascending') : 'none',
  });
  const button = el('button', { type: 'button', class: 'th-sort', text: column.label });
  button.addEventListener('click', () => sortBy(column));
  cell.append(button);
  return cell;
}

function bodyCell(cell, entry) {
  if (cell.key === 'title') return el('td', {}, [link(skillHref(entry.id), cell.text)]);
  if (cell.key !== 'quadrant') {
    return el('td', { class: cell.numeric ? 'numeric' : null, text: cell.text });
  }
  return el('td', { class: 'quad-cell', 'data-quadrant': entry.quadrant || '', text: cell.text });
}

function bodyRow(entry) {
  return el('tr', {}, entry.cells.map((cell) => bodyCell(cell, entry)));
}

function tableCaption(shown, total) {
  const suffix = shown === total ? '' : ` of ${formatCount(total)}`;
  return `${formatCount(shown)}${suffix} skill${shown === 1 ? '' : 's'}. Every column sorts.`;
}

function skillTable(rows, total) {
  const sorted = model.sortSkillRows(rows, state.skillSort.key, state.skillSort.descending);
  return el('table', { class: 'skill-table' }, [
    el('caption', { text: tableCaption(rows.length, total) }),
    el('thead', {}, [el('tr', {}, model.SKILL_COLUMNS.map(headerCell))]),
    el('tbody', {}, model.skillTableRows(sorted).map(bodyRow)),
  ]);
}

boot();
