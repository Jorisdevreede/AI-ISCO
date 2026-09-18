// groups.html — DOM wiring only.
//
// Every string, number and geometry comes from groups-model.js; the treemap
// canvas lives in treemap-view.js. This file listens, builds nodes and pushes
// history entries, so the URL stays the state: #g=<level>:<code>&view=<view>,
// or #a=<key>&b=<key> for a comparison.

import { renderChrome } from '../chrome.js';
import { createCombobox } from '../combobox.js';
import { loadJSON, whenSlow } from '../data.js';
import { formatCount, formatScore } from '../format.js';
import { SORTS, TABLE_COLUMNS, occupationsInGroup, tableRows } from '../groupstats.js';
import { groupHref, jobHref, skillHref, withScorer } from '../urlstate.js';
import * as model from './groups-model.js';
import { createTreemapView } from './treemap-view.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const SCATTER_BOX = { width: 420, height: 420, pad: 44 };

const state = {
  groups: null,
  index: null,
  stats: null,
  bySlug: null,
  skills: null,
  skillsPending: false,
  colour: 'quadrant',
  rankedSort: 'automation',
  tableSort: { key: 'a', descending: true },
  teardown: null,
  focusChart: false,
};

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

function svgEl(tag, props = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
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

function go(href) {
  window.history.pushState(null, '', withScorer(href, window.location.search));
  render();
}

function openJob(slug, groupKey) {
  window.location.assign(withScorer(jobHref(slug, { from: groupKey }), window.location.search));
}

/* --- boot ----------------------------------------------------------------- */

function ready(files) {
  const [groups, index, stats] = files;
  Object.assign(state, { groups, index, stats, bySlug: model.indexBySlug(index) });
  show(byId('page-status'), false);
  wirePickers();
  render();
}

function failed(error) {
  show(byId('page-status'), false);
  const retry = el('button', { type: 'button', class: 'button', text: 'Try again' });
  retry.addEventListener('click', () => window.location.reload());
  byId('page-error').replaceChildren(
    el('h2', { text: 'We could not load the groups' }),
    el('p', { text: error.message }),
    el('p', { text: 'Nothing on your side is wrong. Try again, or start from a job title.' }),
    el('p', {}, [retry, ' ', link('index.html', 'Find a job instead', { class: 'button button--quiet' })]),
  );
  byId('group-subtitle').textContent = '';
  show(byId('page-error'), true);
}

function boot() {
  renderChrome({ active: 'groups' });
  byId('copy-link').addEventListener('click', copyLink);
  byId('tablist').addEventListener('keydown', onTabKey);
  window.addEventListener('hashchange', render);
  window.addEventListener('popstate', render);
  const pending = Promise.all([loadJSON('groups'), loadJSON('search_index'), loadJSON('stats')]);
  whenSlow(pending, 200, () => show(byId('page-status'), true)).then(ready).catch(failed);
}

/* --- routing -------------------------------------------------------------- */

function currentState() {
  return model.readState(window.location.hash, window.innerWidth);
}

function currentKey() {
  const here = currentState();
  return here.compare ? here.compare.a : here.key;
}

function render() {
  if (!state.groups) return;
  if (state.teardown) state.teardown();
  state.teardown = null;
  const here = currentState();
  if (here.compare) renderCompare(here.compare);
  else renderGroup(here);
}

function renderGroup(here) {
  const group = model.resolveGroup(state.groups, here.key);
  show(byId('compare-body'), false);
  show(byId('unknown-body'), !group);
  show(byId('group-body'), Boolean(group));
  show(byId('crumbs-nav'), Boolean(group));
  if (!group) {
    renderUnknown(here.key);
    return;
  }
  const view = { key: here.key, view: here.view, group, rows: occupationsInGroup(state.index, here.key) };
  renderHead(view);
  renderMix(view);
  renderTabs(view);
  renderPanel(view);
  renderLists(view);
}

function renderUnknown(key) {
  byId('group-title').textContent = 'Group not found';
  byId('group-subtitle').textContent = 'Pick a group above, or start from all occupations.';
  document.title = 'Group not found — Browse sectors — AI-ISCO';
  show(byId('children-block'), false);
  byId('unknown-body').replaceChildren(el('div', { class: 'error-block' }, [
    el('h2', { text: model.unknownGroupMessage(key) }),
    el('p', { text: 'The link may be incomplete, or the group may have been renamed.' }),
    link(groupHref(model.ROOT_KEY), 'See all occupations', { class: 'button' }),
  ]));
}

/* --- header --------------------------------------------------------------- */

function renderHead(view) {
  byId('group-title').textContent = view.group.label;
  byId('group-subtitle').textContent = model.groupSubtitle(view.group);
  document.title = `${view.group.label} — Browse sectors — AI-ISCO`;
  renderCrumbs(view.key);
  renderChildren(view.key);
}

function crumbItem(step, isLast) {
  const anchor = link(groupHref(step.key), step.label);
  if (isLast) anchor.setAttribute('aria-current', 'page');
  return el('li', {}, [anchor]);
}

function renderCrumbs(key) {
  const trail = model.breadcrumb(state.groups, key);
  byId('crumbs').replaceChildren(
    ...trail.map((step, position) => crumbItem(step, position === trail.length - 1)),
  );
}

function childItem(child) {
  const anchor = link(groupHref(child.key), child.label, { class: 'chip' });
  anchor.append(el('span', { class: 'count', text: ` ${formatCount(child.n)} jobs` }));
  return el('li', {}, [anchor]);
}

function renderChildren(key) {
  const children = model.childGroups(state.groups, key);
  show(byId('children-block'), children.length > 0);
  byId('children').replaceChildren(...children.map(childItem));
}

function copyLink() {
  const status = byId('copy-status');
  const url = window.location.href;
  const fallback = () => { status.textContent = `Copy this link: ${url}`; };
  if (!navigator.clipboard) return fallback();
  return navigator.clipboard.writeText(url)
    .then(() => { status.textContent = 'Link copied.'; }, fallback);
}

function pickerOption(hit, node) {
  node.append(el('span', { text: hit.label }), el('span', { class: 'code', text: `${formatCount(hit.n)} jobs` }));
}

function wirePicker(name, onSelect) {
  const input = byId(name);
  createCombobox({
    input,
    listbox: byId(`${name}-list`),
    status: byId(`${name}-status`),
    getResults: (query) => model.matchGroups(state.groups, query, 10),
    renderOption: pickerOption,
    onSelect: (hit) => { input.value = ''; onSelect(hit); },
  });
}

function wirePickers() {
  wirePicker('group-picker', (hit) => go(groupHref(hit.key)));
  wirePicker('compare-picker', (hit) => go(model.compareHref(currentKey(), hit.key)));
}

/* --- how this group splits ------------------------------------------------ */

function mixRow(bar) {
  const width = bar.count ? Math.max(bar.share * 100, 1.5) : 0;
  return el('li', { 'data-quadrant': bar.code }, [
    el('span', { class: 'mix-name', text: bar.label }),
    el('span', { class: 'mix-track' }, [el('span', { class: 'mix-fill', style: `width: ${width}%` })]),
    el('span', { class: 'mix-figure', text: `${bar.text} · ${bar.percent}` }),
  ]);
}

function renderMix(view) {
  byId('mix-bars').replaceChildren(...model.quadrantBars(view.group).map(mixRow));
  byId('mix-caveat').textContent = model.nearLineCaveat(state.stats);
}

/* --- tabs ----------------------------------------------------------------- */

function tabButton(name, view) {
  const selected = name === view.view;
  const button = el('button', {
    type: 'button',
    class: 'tab',
    role: 'tab',
    id: `tab-${name}`,
    'aria-selected': String(selected),
    'aria-controls': 'panel',
    tabindex: selected ? '0' : '-1',
    text: model.VIEW_LABELS[name],
  });
  button.addEventListener('click', () => go(groupHref(view.key, name)));
  return button;
}

function renderTabs(view) {
  byId('tablist').replaceChildren(...model.VIEWS.map((name) => tabButton(name, view)));
}

function nextTab(tabs, from, key) {
  const steps = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 };
  if (key === 'Home') return 0;
  if (key === 'End') return tabs.length - 1;
  if (!steps[key]) return null;
  return (from + steps[key] + tabs.length) % tabs.length;
}

function onTabKey(event) {
  const tabs = [...byId('tablist').querySelectorAll('[role="tab"]')];
  const from = tabs.indexOf(document.activeElement);
  const next = from < 0 ? null : nextTab(tabs, from, event.key);
  if (next === null) return;
  event.preventDefault();
  tabs[next].focus();
}

/* --- the four views ------------------------------------------------------- */

function summaryLine(kind, view) {
  return el('p', { class: 'view-summary', text: model.visualSummary(kind, view.group, view.rows) });
}

function tableButton(view) {
  const button = el('button', { type: 'button', class: 'button button--quiet', text: 'View as table' });
  button.addEventListener('click', () => go(groupHref(view.key, 'table')));
  return button;
}

function renderPanel(view) {
  const panel = byId('panel');
  if (state.teardown) state.teardown();
  state.teardown = null;
  panel.setAttribute('aria-labelledby', `tab-${view.view}`);
  panel.replaceChildren();
  if (!view.rows.length) {
    panel.append(el('p', { class: 'empty-note', text: 'No jobs sit in this group yet. Try the group above it.' }));
    return;
  }
  ({
    ranked: renderRanked, scatter: renderScatter, treemap: renderTreemap, table: renderTable,
  })[view.view](panel, view);
}

/* Ranked ------------------------------------------------------------------- */

function sortSelect(view) {
  const options = Object.entries(SORTS).map(([key, sort]) => el('option', { value: key, text: sort.label }));
  const select = el('select', { id: 'ranked-sort' }, options);
  select.value = state.rankedSort;
  select.addEventListener('change', () => {
    state.rankedSort = select.value;
    renderPanel(view);
    byId('ranked-sort').focus();
  });
  return el('div', { class: 'view-toolbar' }, [el('label', { for: 'ranked-sort', text: 'Sort by' }), select]);
}

function legendItem(className, text) {
  return el('span', {}, [el('span', { class: `dot ${className}` }), text]);
}

function rankedLegend() {
  return el('p', { class: 'dot-legend' }, [
    legendItem('dot--auto', ' automation'),
    legendItem('dot--amp', ' amplification'),
    el('span', { text: `The line on each row is the cut-off of 6 on a 1 to 10 axis.` }),
  ]);
}

function dotTrack(row) {
  const dots = [
    row.automation.percent === null ? null
      : el('span', { class: 'dot dot--auto', style: `left: ${row.automation.percent}%` }),
    row.amplification.percent === null ? null
      : el('span', { class: 'dot dot--amp', style: `left: ${row.amplification.percent}%` }),
  ].filter(Boolean);
  return el('span', { class: 'dot-track', 'aria-hidden': 'true' }, dots);
}

function rankedItem(row, groupKey) {
  const scores = `${row.automation.text} / ${row.amplification.text} · ${row.quadrantLabel}`;
  const anchor = link(jobHref(row.slug, { from: groupKey }), null);
  anchor.append(
    el('span', { class: 'ranked-title' }, [row.title, ' ', el('span', { class: 'ranked-code', text: row.code })]),
    dotTrack(row),
    el('span', { class: 'ranked-scores', 'aria-hidden': 'true', text: scores }),
    el('span', {
      class: 'visually-hidden',
      text: `automation ${row.automation.text}, amplification ${row.amplification.text}, ${row.quadrantLabel}`,
    }),
  );
  return el('li', {}, [anchor]);
}

function renderRanked(panel, view) {
  const rows = model.rankedRows(view.rows, state.rankedSort);
  panel.append(
    sortSelect(view),
    summaryLine('ranked', view),
    rankedLegend(),
    el('ol', { class: 'ranked' }, rows.map((row) => rankedItem(row, view.key))),
  );
}

/* Scatter ------------------------------------------------------------------ */

function svgLine(points, className) {
  return svgEl('line', { ...points, class: className });
}

function svgText(at, value, className, extra = {}) {
  return svgEl('text', { ...at, class: className, ...extra }, [value]);
}

function scatterFrame(cut) {
  const { width, height, pad } = SCATTER_BOX;
  return [
    svgLine({ x1: pad, y1: pad, x2: pad, y2: height - pad }, 'grid-line'),
    svgLine({ x1: pad, y1: height - pad, x2: width - pad, y2: height - pad }, 'grid-line'),
    svgLine({ x1: cut.x, y1: pad, x2: cut.x, y2: height - pad }, 'cut-line'),
    svgLine({ x1: pad, y1: cut.y, x2: width - pad, y2: cut.y }, 'cut-line'),
  ];
}

function scatterTicks(cut) {
  const { width, height, pad } = SCATTER_BOX;
  const middle = height / 2 + 34;
  return [
    svgText({ x: pad - 4, y: height - pad + 16 }, '1', 'axis-text'),
    svgText({ x: cut.x - 3, y: height - pad + 16 }, '6', 'axis-text'),
    svgText({ x: width - pad - 8, y: height - pad + 16 }, '10', 'axis-text'),
    svgText({ x: width / 2 - 30, y: height - 8 }, 'Automation', 'axis-text'),
    svgText({ x: pad - 24, y: height - pad + 4 }, '1', 'axis-text'),
    svgText({ x: pad - 24, y: cut.y + 4 }, '6', 'axis-text'),
    svgText({ x: pad - 30, y: pad + 4 }, '10', 'axis-text'),
    svgText({ x: 14, y: middle }, 'Amplification', 'axis-text', { transform: `rotate(-90 14 ${middle})` }),
  ];
}

function scatterQuadrants(cut) {
  const { width, height, pad } = SCATTER_BOX;
  return [
    svgText({ x: pad + 6, y: pad + 14 }, 'EVOLVE', 'quad-text'),
    svgText({ x: cut.x + 6, y: pad + 14 }, 'TRANSFORM', 'quad-text'),
    svgText({ x: pad + 6, y: height - pad - 6 }, 'STABLE', 'quad-text'),
    svgText({ x: cut.x + 6, y: height - pad - 6 }, 'SHRINK', 'quad-text'),
  ];
}

function scatterDot(point, index) {
  return svgEl('circle', {
    class: 'dot-point',
    'data-quadrant': point.quadrant || '',
    'data-index': String(index),
    cx: point.x.toFixed(2),
    cy: point.y.toFixed(2),
    r: 3.4,
  });
}

function buildScatter(points, summary) {
  const cut = model.thresholdPoint(SCATTER_BOX);
  return svgEl('svg', {
    class: 'scatter',
    viewBox: `0 0 ${SCATTER_BOX.width} ${SCATTER_BOX.height}`,
    role: 'img',
    tabindex: '0',
    'aria-label': summary,
    'aria-describedby': 'chart-caption',
  }, [
    ...scatterFrame(cut), ...scatterTicks(cut), ...scatterQuadrants(cut),
    ...points.map(scatterDot),
    svgEl('circle', { class: 'dot-marker', r: 7, cx: -30, cy: -30 }),
  ]);
}

/** Roving focus over the dots: one tab stop, arrow keys, Enter opens. */
function wireScatter(svg, points, groupKey) {
  const order = points.map((point, index) => index).sort((a, b) => points[a].x - points[b].x);
  const marker = svg.querySelector('.dot-marker');
  const caption = byId('chart-caption');
  const cursor = { at: -1 };
  const moveTo = (position) => {
    cursor.at = Math.max(0, Math.min(position, order.length - 1));
    const point = points[order[cursor.at]];
    marker.setAttribute('cx', point.x.toFixed(2));
    marker.setAttribute('cy', point.y.toFixed(2));
    caption.textContent = `${point.label} Press Enter to open the job page.`;
    byId('chart-live').textContent = caption.textContent;
  };
  svg.addEventListener('focus', () => { if (cursor.at < 0) moveTo(0); });
  svg.addEventListener('keydown', (event) => onScatterKey(event, { cursor, moveTo, order, points, groupKey }));
  svg.addEventListener('pointerover', (event) => {
    const index = event.target.dataset && event.target.dataset.index;
    if (index !== undefined) moveTo(order.indexOf(Number(index)));
  });
}

function onScatterKey(event, scatter) {
  const steps = { ArrowRight: 1, ArrowLeft: -1, ArrowUp: 1, ArrowDown: -1 };
  const { cursor, moveTo, order, points, groupKey } = scatter;
  if (steps[event.key]) {
    event.preventDefault();
    moveTo(cursor.at + steps[event.key]);
  } else if (event.key === 'Home' || event.key === 'End') {
    event.preventDefault();
    moveTo(event.key === 'Home' ? 0 : order.length - 1);
  } else if (event.key === 'Enter' && cursor.at >= 0) {
    event.preventDefault();
    openJob(points[order[cursor.at]].slug, groupKey);
  }
}

function chartCaption(text) {
  return el('p', { class: 'chart-caption', id: 'chart-caption', text });
}

function renderScatter(panel, view) {
  const points = model.scatterPoints(view.rows, SCATTER_BOX);
  const svg = buildScatter(points, model.visualSummary('scatter', view.group, view.rows));
  panel.append(
    summaryLine('scatter', view),
    el('div', { class: 'scatter-wrap' }, [svg,
      chartCaption('Hover a dot, or focus the chart and use the arrow keys, to name a job.')]),
    el('div', { class: 'chart-actions' }, [tableButton(view)]),
  );
  wireScatter(svg, points, view.key);
}

/* Treemap ------------------------------------------------------------------ */

function colourButton(mode, view) {
  const button = el('button', {
    type: 'button',
    class: 'chip',
    'aria-pressed': String(state.colour === mode.key),
    text: mode.label,
  });
  button.addEventListener('click', () => {
    state.colour = mode.key;
    renderPanel(view);
    const again = [...byId('panel').querySelectorAll('[aria-pressed]')]
      .find((node) => node.textContent === mode.label);
    if (again) again.focus();
  });
  return button;
}

function colourToolbar(view) {
  return el('div', { class: 'view-toolbar' }, [
    el('span', { class: 'section-heading', id: 'colour-label', text: 'Colour by' }),
    el('div', { class: 'chip-row', role: 'group', 'aria-labelledby': 'colour-label' },
      model.COLOUR_MODES.map((mode) => colourButton(mode, view))),
  ]);
}

function renderTreemap(panel, view) {
  const canvas = el('canvas', {
    id: 'treemap-canvas',
    role: 'img',
    tabindex: '0',
    'aria-label': model.visualSummary('treemap', view.group, view.rows),
    'aria-describedby': 'chart-caption',
  });
  const caption = chartCaption('Click a tile to drill into a group or open a job. By keyboard: '
    + 'focus the treemap, arrow keys move, Enter opens, Escape goes up a level.');
  panel.append(
    colourToolbar(view), summaryLine('treemap', view),
    el('div', { class: 'treemap-wrap' }, [canvas, caption]),
    el('div', { class: 'chart-actions' }, [tableButton(view)]),
  );
  startTreemap({ canvas, caption, view });
}

function startTreemap({ canvas, caption, view }) {
  const treemap = createTreemapView({
    canvas,
    caption,
    live: byId('chart-live'),
    onActivate: (tile) => activateTile(tile, view.key),
    onUp: () => goUp(view.key),
  });
  treemap.setMode(state.colour);
  treemap.setTiles(model.treemapTiles(state.groups, state.index, view.key));
  state.teardown = treemap.destroy;
  if (state.focusChart) canvas.focus();
  state.focusChart = false;
}

function activateTile(tile, groupKey) {
  if (tile.kind !== 'group') {
    openJob(tile.id, groupKey);
    return;
  }
  state.focusChart = true;
  go(groupHref(tile.id, 'treemap'));
}

function goUp(groupKey) {
  const parent = (state.groups[groupKey] || {}).parent;
  if (!parent) return;
  state.focusChart = true;
  go(groupHref(parent, 'treemap'));
}

/* Table -------------------------------------------------------------------- */

function sortTable(column, view) {
  const same = state.tableSort.key === column.key;
  state.tableSort = {
    key: column.key,
    descending: same ? !state.tableSort.descending : Boolean(column.numeric),
  };
  renderPanel(view);
  const header = byId('panel').querySelector(`th[data-key="${column.key}"] button`);
  if (header) header.focus();
}

function headerCell(column, view) {
  const sorted = state.tableSort.key === column.key;
  const direction = state.tableSort.descending ? 'descending' : 'ascending';
  const cell = el('th', {
    scope: 'col',
    'data-key': column.key,
    class: column.numeric ? 'numeric' : null,
    'aria-sort': sorted ? direction : 'none',
  });
  const button = el('button', { type: 'button', class: 'th-sort', text: column.label });
  button.addEventListener('click', () => sortTable(column, view));
  cell.append(button);
  return cell;
}

function bodyRow(entry, groupKey) {
  return el('tr', {}, entry.cells.map((cell, position) => (position === 0
    ? el('td', {}, [link(jobHref(entry.slug, { from: groupKey }), cell.text)])
    : el('td', { class: cell.numeric ? 'numeric' : null, text: cell.text }))));
}

function renderTable(panel, view) {
  const sorted = model.sortByColumn(view.rows, state.tableSort.key, state.tableSort.descending);
  const table = el('table', {}, [
    el('caption', { text: `${formatCount(sorted.length)} jobs in ${view.group.label}. Every column sorts.` }),
    el('thead', {}, [el('tr', {}, TABLE_COLUMNS.map((column) => headerCell(column, view)))]),
    el('tbody', {}, tableRows(sorted).map((entry) => bodyRow(entry, view.key))),
  ]);
  panel.append(summaryLine('table', view), el('div', { class: 'table-scroll' }, [table]));
}

/* --- lists ---------------------------------------------------------------- */

function jobItem(job, groupKey) {
  return el('li', {}, [
    link(jobHref(job.slug, { from: groupKey }), job.title),
    el('span', {
      class: 'scores',
      text: `automation ${formatScore(job.automation)} · amplification ${formatScore(job.amplification)}`,
    }),
  ]);
}

function fillList(id, jobs, groupKey) {
  const items = jobs.length
    ? jobs.map((job) => jobItem(job, groupKey))
    : [el('li', { class: 'empty-note', text: 'No jobs listed for this group.' })];
  byId(id).replaceChildren(...items);
}

function skillItem(entry) {
  return el('li', {}, [
    entry.title
      ? link(skillHref(entry.id), entry.title)
      : el('span', { class: 'muted', text: 'Loading skill name…' }),
    el('span', { class: 'scores', text: entry.count }),
  ]);
}

function skillColumn(heading, entries) {
  const items = entries.length
    ? entries.map(skillItem)
    : [el('li', { class: 'empty-note', text: 'No skills listed for this group.' })];
  return el('div', {}, [el('h3', { text: heading }), el('ul', { class: 'job-list' }, items)]);
}

function ensureSkillTitles(view) {
  const skills = model.drivingSkills(view.group, null);
  const wanted = skills.automation.length + skills.amplification.length;
  if (!wanted || state.skills || state.skillsPending) return;
  state.skillsPending = true;
  loadJSON('skill_index')
    .then((rows) => {
      state.skills = new Map(rows.map((row) => [row.id, row.t]));
      renderSkills(view);
    })
    .catch(() => { state.skillsPending = false; });
}

function renderSkills(view) {
  const skills = model.drivingSkills(view.group, state.skills);
  byId('skill-columns').replaceChildren(
    skillColumn('Most automatable', skills.automation),
    skillColumn('Most amplified', skills.amplification),
  );
  ensureSkillTitles(view);
}

function renderLists(view) {
  const lists = model.exposureLists(view.group, state.bySlug);
  fillList('top-list', lists.top, view.key);
  fillList('bottom-list', lists.bottom, view.key);
  renderSkills(view);
}

/* --- comparison ----------------------------------------------------------- */

function compareCard(side) {
  return el('section', { class: 'card' }, [
    el('h2', {}, [link(groupHref(side.key), side.label)]),
    el('p', { class: 'muted small', text: side.subtitle }),
    el('ul', { class: 'mix-bars' }, side.bars.map(mixRow)),
    el('p', {
      class: 'medians',
      text: `Median automation ${formatScore(side.medians.automation)} · `
        + `median amplification ${formatScore(side.medians.amplification)}`,
    }),
  ]);
}

function divergeRow(row) {
  const size = Math.abs(row.delta) * 50;
  const style = row.delta >= 0 ? `left: 50%; width: ${size}%;` : `right: 50%; width: ${size}%;`;
  return el('li', { 'data-quadrant': row.code }, [
    el('span', { class: 'mix-name', text: row.label }),
    el('span', { class: 'diverge-track' }, [el('span', { class: 'diverge-fill', style })]),
    el('span', { class: 'diverge-figure', text: `${row.text} · ${row.deltaText}` }),
  ]);
}

function compareSections(comparison) {
  return [
    el('div', { class: 'compare-columns' }, [compareCard(comparison.a), compareCard(comparison.b)]),
    el('section', { class: 'card' }, [
      el('h2', { text: 'Where they differ' }),
      el('p', {
        class: 'small muted',
        text: `Each bar is the share in ${comparison.a.label} minus the share in ${comparison.b.label}.`,
      }),
      el('ul', { class: 'diverge' }, comparison.differences.map(divergeRow)),
      el('p', { class: 'mix-caveat small', text: model.nearLineCaveat(state.stats) }),
    ]),
  ];
}

function renderCompare(compare) {
  const comparison = model.compareGroups(state.groups, compare.a, compare.b);
  show(byId('group-body'), false);
  show(byId('crumbs-nav'), false);
  show(byId('children-block'), false);
  show(byId('unknown-body'), !comparison);
  show(byId('compare-body'), Boolean(comparison));
  if (!comparison) {
    renderUnknown(`${compare.a} and ${compare.b}`);
    return;
  }
  byId('group-title').textContent = `${comparison.a.label} compared with ${comparison.b.label}`;
  byId('group-subtitle').textContent = 'Two groups, side by side, from the same scoring run.';
  document.title = `${comparison.a.label} compared with ${comparison.b.label} — AI-ISCO`;
  const back = el('p', {}, [link(groupHref(comparison.a.key), `Back to ${comparison.a.label}`)]);
  byId('compare-body').replaceChildren(...compareSections(comparison), back);
}

boot();
