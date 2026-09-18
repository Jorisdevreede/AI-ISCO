// tree.html — DOM wiring only.
//
// The tree itself lives in tree-view.js and every string, number and ordering
// in tree-model.js. This file loads the three small index files, keeps the URL
// as the state (#job=<slug> or #g=<level>:<code>, plus &q=<filter>), and fills
// the detail pane beside the tree.
//
// No page fetches the 15 MB portfolio file any more. The skills of one job come
// from its unit group's own shard, `jobs/<unit>`, and the tree already knows the
// unit code of every job — so nothing is downloaded until a job is selected, and
// then only that group's slice. The tree, the filter and every group detail work
// without any of it.

import { renderTypeBadge } from '../badge.js';
import { renderChrome } from '../chrome.js';
import { loadJSON, whenSlow } from '../data.js';
import { formatCount, formatScore, isScored } from '../format.js';
import { SHARES, schemeOf, splitOf, thresholdOf } from '../scheme.js';
import { renderSharesBar } from '../shares-bar.js';
import { isShares } from '../shares.js';
import { groupHref, jobHref, skillHref, withScorer } from '../urlstate.js';
import * as model from './tree-model.js';
import { TreeView } from './tree-view.js';

/** Below this width the detail pane sits under the tree, not beside it. */
const NARROW = '(max-width: 860px)';

// The display scores, which are secondary detail: the shares bar leads.
const QUADRANT_CARDS = [
  { key: 'a', modifier: 'auto', label: 'Automation risk' },
  { key: 'm', modifier: 'amp', label: 'Amplification' },
];

const SHARES_CARDS = [
  { key: 'a', modifier: 'auto', label: 'AI substitution' },
  { key: 'm', modifier: 'amp', label: 'AI assistance' },
  { key: 'k', modifier: 'mech', label: 'Machine automation' },
];

/** The page's own title, which the static markup already carries. */
const BASE_TITLE = 'Job tree — AI-ISCO';

/** When a wait is worth a word, and when it is worth an apology. */
const SLOW_MS = 200;
const VERY_SLOW_MS = 3000;

const state = {
  model: null,
  stats: null,
  scheme: null,
  cut: null,
  expanded: new Set(),
  selection: null,
  query: '',
  filter: null,
  // One entry per unit group asked for: { rows } once loaded, { error } if not.
  shards: new Map(),
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
  state.scheme = schemeOf(stats);
  state.cut = thresholdOf(stats);
  byId('tree-key').textContent = model.scoreKeyLine(state.scheme);
  show(byId('tree-status'), false);
  show(byId('tree-layout'), true);
  onRoute();
  renderResults();
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

function resultItem(entry) {
  const also = model.alsoMatches(entry);
  const href = model.treeHref({ kind: 'job', id: entry.slug }, state.query);
  return el('li', {}, [
    el('a', { href: withScorer(href, window.location.search), class: 'result-link' }, [
      el('span', { class: 'result-title', text: entry.title }),
      el('span', { class: 'result-group', text: entry.group }),
      also ? el('span', { class: 'result-also', text: also }) : null,
    ]),
  ]);
}

// With a filter on, the answer is a list of jobs; the hierarchy that holds them
// waits behind the disclosure, which is closed so the results are what is read.
function renderResults() {
  const entries = model.resultRows(state.model, state.filter);
  const box = byId('tree-results');
  const hierarchy = byId('tree-hierarchy');
  byId('tree-results-list').replaceChildren(...entries.map(resultItem));
  show(box, entries.length > 0);
  hierarchy.classList.toggle('tree-disclosure--plain', !state.filter);
  hierarchy.open = !state.filter;
}

function applyQuery(query) {
  state.query = query;
  state.filter = model.filterMatches(state.model, query);
  byId('tree-q').value = query;
  byId('tree-count').textContent = model.matchMessage(state.filter);
  renderResults();
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
  return el('li', { [segment.attribute]: segment.code }, [
    el('span', { class: 'mix-name', text: segment.label }),
    el('span', { class: 'mix-track' }, [el('span', { class: 'mix-fill', style: `width: ${width}%` })]),
    el('span', { class: 'mix-figure', text: `${segment.text} · ${segment.percent}` }),
  ]);
}

// Type and class names are whole phrases, so those lists get a wider name
// column; the four box names fit the narrow one they have always had.
function mixList(segments) {
  const wide = segments.length > 0 && segments[0].attribute !== 'data-quadrant';
  return el('ul', { class: wide ? 'mix-bars mix-bars--wide' : 'mix-bars' }, segments.map(mixRow));
}

function sharesFigure(shares, name) {
  return el('div', { class: 'shares-block' }, [renderSharesBar(shares, { name })]);
}

function caveat() {
  return el('p', { class: 'mix-caveat small', text: model.nearLineCaveat(state.stats) });
}

// The static <title> already names the page, so only a selection adds to it.
function setTitle(text, documentTitle) {
  byId('detail-title').textContent = text;
  document.title = documentTitle ? `${documentTitle} — ${BASE_TITLE}` : BASE_TITLE;
}

function renderDetail() {
  const host = byId('detail-body');
  if (!state.selection) renderEmptyDetail(host);
  else if (state.selection.kind === 'group') renderGroupDetail(host);
  else renderJobDetail(host);
}

/* --- detail: nothing selected --------------------------------------------- */

function skillSplitBlock() {
  const segments = model.classSegments(state.stats);
  if (!segments.length) return [];
  return [
    el('h3', { text: `How all ${formatCount(state.stats.skills_scored)} skills split` }),
    el('p', { class: 'small muted', text: 'Every scored skill falls into exactly one class. '
      + 'The shares of these four over a job’s own skills decide its type.' }),
    mixList(segments),
  ];
}

function renderEmptyDetail(host) {
  setTitle('Pick a job in the tree to see its skills', null);
  const counts = splitOf(state.stats).counts;
  host.replaceChildren(
    el('p', { text: 'Open a group to walk down ISCO-08. Selecting a group shows how its jobs '
      + 'split; selecting a job shows its skills and where each one sits.' }),
    el('h3', { text: `How all ${formatCount(state.stats.occupations)} jobs split` }),
    mixList(model.mixSegments(counts)),
    caveat(),
    ...skillSplitBlock(),
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
    ...groupSharesBlock(group),
    caveat(),
    el('p', {}, [link(groupHref(key), 'Open this sector', { class: 'button button--quiet' })]),
  );
}

function groupSharesBlock(group) {
  if (!isShares(group.sh)) return [];
  return [
    el('h3', { text: 'What the average job here is made of' }),
    sharesFigure(group.sh, `the average job in ${group.label}`),
  ];
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
  return el('p', { class: 'muted small unit-line' }, [
    'ISCO-08 unit group ',
    link(model.treeHref({ kind: 'group', id: key }, state.query), `${group.label} (${key.slice(5)})`),
  ]);
}

// Under shares the four-share bar leads and the three display scores follow it;
// under quadrants the two scores are all there is.
function jobHeader(row) {
  const shares = state.scheme === SHARES;
  const badge = renderTypeBadge(
    { t: row.t, a: row.a, m: row.m, q: row.q, sh: row.sh, nl: row.nl },
    { search: window.location.search },
  );
  const cards = shares ? SHARES_CARDS : QUADRANT_CARDS;
  return [
    unitLine(row),
    el('div', { class: 'badge-line' }, [badge]),
    ...(shares ? [sharesFigure(row.sh, row.t)] : []),
    el('div', { class: 'score-grid' }, cards.map((spec) => scoreCard(spec, row))),
    ...(shares ? [el('p', { id: 'why-line', class: 'why-line small', hidden: 'hidden' })] : []),
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

/* --- one unit group's skills ---------------------------------------------- */

// Nothing is fetched until a job is selected, and then only the shard of its own
// unit group. Each shard is kept, so walking a group costs one request.

function unitOf(row) {
  return String(row.c || '');
}

function keepShard(unit, entry) {
  state.shards.set(unit, entry);
  state.skills = null;
  refreshSkills();
}

function requestShard(unit) {
  if (state.shards.has(unit)) return;
  state.shards.set(unit, { pending: true });
  const load = loadJSON(`jobs/${unit}`);
  // Two thresholds on one request; the first chain's rejection is the second's
  // to report, so it is swallowed here rather than left unhandled.
  whenSlow(load, SLOW_MS, () => setLoadingStage('start')).catch(() => {});
  whenSlow(load, VERY_SLOW_MS, () => setLoadingStage('slow'))
    .then((data) => keepShard(unit, { data }))
    .catch((error) => keepShard(unit, { error }));
}

function retryShard(unit) {
  state.shards.delete(unit);
  refreshSkills();
}

function refreshSkills() {
  if (!state.selection || state.selection.kind !== 'job') return;
  const row = state.model.bySlug.get(state.selection.id);
  if (row && byId('skills-block')) renderSkills(row);
}

// Two stages, both true. The first says what is happening; the second admits the
// wait and offers a way on rather than repeating itself.
function setLoadingStage(stage) {
  const host = byId('skills-block');
  if (!host?.querySelector('.loading')) return;
  if (stage === 'start') {
    host.replaceChildren(el('p', { class: 'loading', text: 'Loading this job’s skills…' }));
    return;
  }
  host.replaceChildren(el('p', { class: 'loading' }, [
    'Still loading. The skills of this job come from a separate file, and on a slow '
      + 'connection that can take a while. ',
    link(groupHref(`unit:${unitOf(state.model.bySlug.get(state.selection.id))}`),
      'Browse this sector instead'),
  ]));
}

function skillsError(host, unit, error) {
  const retry = el('button', { type: 'button', class: 'button button--quiet', text: 'Retry' });
  retry.addEventListener('click', () => retryShard(unit));
  host.replaceChildren(el('div', { class: 'error-block' }, [
    el('p', { text: error.message }),
    el('p', { text: 'The scores above still stand. The full job page has the same skills.' }),
    el('p', {}, [retry]),
  ]));
}

// `why_insulated` lives in the shard, so the line fills in when that arrives.
function setWhyLine(occupation) {
  const slot = byId('why-line');
  if (!slot) return;
  const text = model.whyLine(occupation);
  slot.textContent = text;
  show(slot, Boolean(text));
}

function currentSkills(row, shard) {
  if (!state.skills || state.skills.slug !== row.s) {
    const occupation = model.findOccupation(shard, row.s);
    state.skills = {
      slug: row.s, occupation, rows: model.skillRows(shard, occupation, state.cut),
    };
  }
  setWhyLine(state.skills.occupation);
  return state.skills.rows;
}

function renderSkills(row) {
  const host = byId('skills-block');
  const unit = unitOf(row);
  const entry = state.shards.get(unit);
  if (!entry) {
    host.replaceChildren(el('p', { class: 'loading', text: 'Loading this job’s skills…' }));
    requestShard(unit);
    return;
  }
  if (entry.pending) return;
  if (entry.error) {
    skillsError(host, unit, entry.error);
    return;
  }
  const rows = currentSkills(row, entry.data);
  if (!rows.length) {
    host.replaceChildren(el('p', { class: 'muted', text: 'We have no skill list for this job.' }));
    return;
  }
  host.replaceChildren(...skillSections(rows));
}

function skillSections(rows) {
  const mix = model.skillMix(rows, state.scheme);
  const shown = model.filterSkillRows(rows, state.skillFilter);
  return [
    el('p', { class: 'small muted', text: model.skillSummary(mix) }),
    el('h4', { text: state.scheme === SHARES ? 'How these skills split' : 'Where the skills sit' }),
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

// The three values aria-sort takes: only the sorted column claims a direction.
function ariaSortValue(sorted, descending) {
  if (!sorted) return 'none';
  return descending ? 'descending' : 'ascending';
}

function headerCell(column) {
  const sorted = state.skillSort.key === column.key;
  const cell = el('th', {
    scope: 'col',
    'data-key': column.key,
    class: column.numeric ? 'numeric' : null,
    'aria-sort': ariaSortValue(sorted, state.skillSort.descending),
  });
  const button = el('button', { type: 'button', class: 'th-sort', text: column.label });
  button.addEventListener('click', () => sortBy(column));
  cell.append(button);
  return cell;
}

// A skill whose deciding chance sits within five points of the cut carries the
// same "near the line" marker the job badge uses.
function classCell(cell, entry) {
  const td = el('td', { class: 'class-cell', 'data-class': entry.cls || '' },
    [el('span', { text: cell.text })]);
  if (!entry.near) return td;
  td.append(' ', el('span', {
    class: 'near-chip', text: 'near the line', title: model.SKILL_NEAR_TEXT,
  }));
  return td;
}

function bodyCell(cell, entry) {
  if (cell.key === 'title') return el('td', {}, [link(skillHref(entry.id), cell.text)]);
  if (cell.key === 'quadrant') {
    return el('td', { class: 'quad-cell', 'data-quadrant': entry.quadrant || '', text: cell.text });
  }
  if (cell.key === 'cls') return classCell(cell, entry);
  return el('td', { class: cell.numeric ? 'numeric' : null, text: cell.text });
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
  const columns = model.skillColumns(state.scheme);
  return el('table', { class: 'skill-table' }, [
    el('caption', { text: tableCaption(rows.length, total) }),
    el('thead', {}, [el('tr', {}, columns.map(headerCell))]),
    el('tbody', {}, model.skillTableRows(sorted, state.scheme).map(bodyRow)),
  ]);
}

boot();
