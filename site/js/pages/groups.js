// groups.html — DOM wiring only.
//
// Every string, number and geometry comes from groups-model.js; the treemap
// canvas lives in treemap-view.js. This file listens, builds nodes and pushes
// history entries, so the URL stays the state: #g=<level>:<code>&view=<view>,
// or #a=<key>&b=<key> for a comparison.
//
// The scheme of the active score set is read once, from its stats file, and
// handed to the model with every call; nothing here decides what a class is
// called, which order the classes go in or which colour belongs to which code.

import { renderChrome } from '../chrome.js';
import { createCombobox } from '../combobox.js';
import { loadJSON, whenSlow } from '../data.js';
import { formatCount, formatScore } from '../format.js';
import { occupationsInGroup } from '../groupstats.js';
import { SHARES, schemeOf } from '../scheme.js';
import { renderSharesBar } from '../shares-bar.js';
import { groupHref, jobHref, skillHref, withScorer } from '../urlstate.js';
import * as model from './groups-model.js';
import { createTreemapView } from './treemap-view.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const SCATTER_BOX = { width: 420, height: 420, pad: 44 };
const DOT_RADIUS = 3.6;
const LEGEND_RADIUS = 6;

const state = {
  groups: null,
  index: null,
  stats: null,
  scheme: undefined,
  bySlug: null,
  skills: null,
  skillsPending: false,
  skillsWatch: null,
  colour: null,
  colourChosen: false,
  rankedSort: 'automation',
  rankedShown: model.RANKED_PAGE,
  rankedFor: null,
  teardown: null,
  focusChart: false,
  focusSort: null,
};

const shares = () => state.scheme === SHARES;

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

/** A class code goes on the attribute its own scheme's CSS paints. */
function codeAttr(code) {
  return shares() ? { 'data-type': code } : { 'data-quadrant': code };
}

function go(href) {
  window.history.pushState(null, '', withScorer(href, window.location.search));
  render();
}

function openJob(slug, groupKey) {
  window.location.assign(withScorer(jobHref(slug, { from: groupKey }), window.location.search));
}

/* --- boot ----------------------------------------------------------------- */

function adoptScheme(stats) {
  state.scheme = schemeOf(stats);
  state.rankedSort = model.defaultSort(state.scheme);
}

function ready(files) {
  const [groups, index, stats] = files;
  Object.assign(state, { groups, index, stats, bySlug: model.indexBySlug(index) });
  adoptScheme(stats);
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

/** Everything one render of a group needs, including the table's sort. */
function groupView(here, group) {
  return {
    key: here.key,
    view: here.view,
    group,
    rows: occupationsInGroup(state.index, here.key),
    sort: model.tableSortOf(here, state.scheme),
  };
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
  if (!state.colourChosen) state.colour = model.defaultColour(state.scheme, group);
  const view = groupView(here, group);
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
  return el('li', codeAttr(bar.code), [
    el('span', { class: 'mix-name', text: bar.label }),
    el('span', { class: 'mix-track' }, [el('span', { class: 'mix-fill', style: `width: ${width}%` })]),
    el('span', { class: 'mix-figure', text: `${bar.text} · ${bar.percent}` }),
  ]);
}

/** The group's mean shares, above the split: one big bar and its sentence. */
function renderMeanShares(group) {
  const block = byId('mean-shares');
  const sentence = model.meanShareSentence(group);
  show(block, Boolean(sentence));
  if (!sentence) return;
  block.replaceChildren(
    el('h3', { class: 'section-heading', text: 'What this group is made of' }),
    renderSharesBar(group.sh, { name: group.label }),
    el('p', { class: 'mean-shares-line', text: sentence }),
  );
}

function renderMix(view) {
  byId('mix-bars').replaceChildren(...model.mixBars(view.group, state.scheme).map(mixRow));
  // The site-wide near-the-line figure reads as a fact about this group, and it
  // is the wrong one; without a per-group count there is no sentence to say.
  const caveat = model.groupNearSentence(view.group, state.scheme);
  byId('mix-caveat').textContent = caveat;
  show(byId('mix-caveat'), Boolean(caveat));
  byId('mix-heading').textContent = shares() ? 'How this group splits by type' : 'How this group splits';
  renderMeanShares(view.group);
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
  return el('p', {
    class: 'view-summary',
    text: model.visualSummary(kind, view.group, view.rows, state.scheme),
  });
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
  const options = model.rankedSorts(state.scheme)
    .map((sort) => el('option', { value: sort.key, text: sort.label }));
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

function classLegendItem(entry) {
  const key = el('span', { class: 'shares-key', 'data-class': entry.code, 'aria-hidden': 'true', text: entry.code });
  return el('li', {}, [key, el('span', { class: 'shares-label', text: entry.label })]);
}

function rankedLegend() {
  if (shares()) {
    return el('ul', { class: 'shares-legend class-legend' },
      model.shareClassLegend().map(classLegendItem));
  }
  return el('p', { class: 'dot-legend' }, [
    legendItem('dot--auto', ' automation'),
    legendItem('dot--amp', ' amplification'),
    el('span', { text: 'The line on each row is the cut-off of 6 on a 1 to 10 axis.' }),
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

/**
 * The shares bar of one row, hidden from the accessibility tree: the row's own
 * visually-hidden summary says the same four numbers, and saying them twice in
 * one link is worse than not drawing the bar at all.
 */
function rowSharesBar(row) {
  const bar = renderSharesBar(row.shares, { name: row.title, compact: true });
  bar.setAttribute('aria-hidden', 'true');
  return bar;
}

/** The middle and right of a row: the four shares, or the two dots and scores. */
function rankedFigure(row) {
  if (shares()) {
    return [
      row.shares
        ? rowSharesBar(row)
        : el('span', { class: 'muted small', text: 'Shares not scored' }),
      el('span', { class: 'ranked-type', ...codeAttr(row.quadrant), 'aria-hidden': 'true', text: row.typeShort }),
    ];
  }
  return [
    dotTrack(row),
    el('span', {
      class: 'ranked-scores',
      'aria-hidden': 'true',
      text: `${row.automation.text} / ${row.amplification.text} · ${row.quadrantLabel}`,
    }),
  ];
}

function rankedItem(row, groupKey) {
  const anchor = link(jobHref(row.slug, { from: groupKey }), null);
  anchor.append(
    el('span', { class: 'ranked-title' }, [row.title, ' ', el('span', { class: 'ranked-code', text: row.code })]),
    ...rankedFigure(row),
    el('span', { class: 'visually-hidden', text: model.rankedRowSummary(row, state.scheme) }),
  );
  return el('li', {}, [anchor]);
}

/** A new group or a new sort starts the list at its first page again. */
function rankedReset(view) {
  const token = `${view.key}:${state.rankedSort}`;
  if (state.rankedFor === token) return;
  state.rankedFor = token;
  state.rankedShown = model.RANKED_PAGE;
}

function focusAfterMore(panel) {
  const button = panel.querySelector('.ranked-more button');
  const last = panel.querySelector('.ranked li:last-child a');
  if (button) button.focus();
  else if (last) last.focus();
}

function moreButton(step, view) {
  const button = el('button', { type: 'button', class: 'button button--quiet', text: step.label });
  button.addEventListener('click', () => {
    state.rankedShown += step.step;
    renderPanel(view);
    byId('chart-live').textContent = model.rankedCountText(rankedSlice(view));
    focusAfterMore(byId('panel'));
  });
  return button;
}

function rankedSlice(view) {
  const rows = model.rankedRows(view.rows, state.rankedSort, state.scheme);
  return model.rankedPage(rows, state.rankedShown);
}

function rankedFooter(page, view) {
  const buttons = model.rankedMoreButtons(page);
  if (!buttons.length) return null;
  return el('div', { class: 'ranked-more' }, [
    el('p', { class: 'small muted', text: model.rankedCountText(page) }),
    ...buttons.map((step) => moreButton(step, view)),
  ]);
}

function renderRanked(panel, view) {
  rankedReset(view);
  const page = rankedSlice(view);
  const footer = rankedFooter(page, view);
  panel.append(
    sortSelect(view),
    summaryLine('ranked', view),
    rankedLegend(),
    el('ol', { class: shares() ? 'ranked ranked--shares' : 'ranked' },
      page.rows.map((row) => rankedItem(row, view.key))),
  );
  if (footer) panel.append(footer);
}

/* Scatter ------------------------------------------------------------------ */

function svgLine(points, className) {
  return svgEl('line', { ...points, class: className });
}

function svgText(at, value, className, extra = {}) {
  return svgEl('text', { ...at, class: className, ...extra }, [value]);
}

function axisFrame() {
  const { width, height, pad } = SCATTER_BOX;
  return [
    svgLine({ x1: pad, y1: pad, x2: pad, y2: height - pad }, 'grid-line'),
    svgLine({ x1: pad, y1: height - pad, x2: width - pad, y2: height - pad }, 'grid-line'),
  ];
}

function cutLines(cut) {
  const { width, height, pad } = SCATTER_BOX;
  return [
    svgLine({ x1: cut.x, y1: pad, x2: cut.x, y2: height - pad }, 'cut-line'),
    svgLine({ x1: pad, y1: cut.y, x2: width - pad, y2: cut.y }, 'cut-line'),
  ];
}

/** The ends of both axes, and what each axis measures. */
function endTicks(names, ticks) {
  const { width, height, pad } = SCATTER_BOX;
  const middle = height / 2 + 34;
  return [
    svgText({ x: pad - 4, y: height - pad + 16 }, ticks.low, 'axis-text'),
    svgText({ x: width - pad - 8, y: height - pad + 16 }, ticks.high, 'axis-text'),
    svgText({ x: width / 2 - 30, y: height - 8 }, names.x, 'axis-text'),
    svgText({ x: pad - 24, y: height - pad + 4 }, ticks.low, 'axis-text'),
    svgText({ x: pad - 30, y: pad + 4 }, ticks.high, 'axis-text'),
    svgText({ x: 14, y: middle }, names.y, 'axis-text', { transform: `rotate(-90 14 ${middle})` }),
  ];
}

/** The same ticks with the cut-off marked between them, in the old order. */
function quadrantTicks(cut, names, ticks) {
  const marks = endTicks(names, ticks);
  const { height, pad } = SCATTER_BOX;
  marks.splice(1, 0, svgText({ x: cut.x - 3, y: height - pad + 16 }, '6', 'axis-text'));
  marks.splice(5, 0, svgText({ x: pad - 24, y: cut.y + 4 }, '6', 'axis-text'));
  return marks;
}

const at = (value) => value.toFixed(1);

/** The type rules, drawn lightly across the plot and labelled with their cut. */
function ruleMarks() {
  return model.ruleLines(SCATTER_BOX, state.scheme).flatMap((rule) => [
    svgLine({ x1: at(rule.x1), y1: at(rule.y1), x2: at(rule.x2), y2: at(rule.y2) }, 'rule-line'),
    svgText({ x: at(rule.textX), y: at(rule.textY) }, rule.label, 'rule-text',
      { 'text-anchor': rule.anchor }),
  ]);
}

/** The part of the plot these two axes cannot settle, tinted under the dots. */
function shadedArea() {
  const area = model.undecidedArea(SCATTER_BOX, state.scheme);
  if (!area) return [];
  return area.rects.map((rect) => svgEl('rect', {
    class: 'undecided',
    x: at(rect.x),
    y: at(rect.y),
    width: at(rect.width),
    height: at(rect.height),
  }));
}

/**
 * The names of the three regions the lines really do decide, and one for the
 * rest. Drawn last so they sit above the dots; the CSS gives them a halo.
 */
function regionLabels() {
  const area = model.undecidedArea(SCATTER_BOX, state.scheme);
  const labels = model.namedRegions(SCATTER_BOX, state.scheme).map((region) => svgText(
    { x: at(region.textX), y: at(region.textY) }, region.label, 'region-text',
    { 'text-anchor': 'middle' },
  ));
  if (area) {
    labels.push(svgText({ x: at(area.textX), y: at(area.textY) }, area.label,
      'region-text region-quiet'));
  }
  return labels;
}

function cornerLabels(cut) {
  const { height, pad } = SCATTER_BOX;
  return [
    svgText({ x: pad + 6, y: pad + 14 }, 'EVOLVE', 'quad-text'),
    svgText({ x: cut.x + 6, y: pad + 14 }, 'TRANSFORM', 'quad-text'),
    svgText({ x: pad + 6, y: height - pad - 6 }, 'STABLE', 'quad-text'),
    svgText({ x: cut.x + 6, y: height - pad - 6 }, 'SHRINK', 'quad-text'),
  ];
}

/** Under quadrants the two axes are class boundaries; under shares they are not. */
function scatterFrame() {
  const names = model.axisNames(state.scheme);
  const ticks = model.axisTicks(state.scheme);
  if (shares()) return [...axisFrame(), ...ruleMarks(), ...endTicks(names, ticks)];
  const cut = model.thresholdPoint(SCATTER_BOX);
  return [...axisFrame(), ...cutLines(cut), ...quadrantTicks(cut, names, ticks),
    ...cornerLabels(cut)];
}

const markerIds = new Set(model.markerShapes(DOT_RADIUS).map((shape) => shape.code));

/** The four skill classes colour the diverging bars of a share comparison. */
const CLASS_CODES = new Set(model.shareClassLegend().map((entry) => entry.code));

function markerDefs() {
  const shapes = model.markerShapes(DOT_RADIUS)
    .map((shape) => svgEl('path', { id: `marker-${shape.code}`, d: shape.d }));
  shapes.push(svgEl('circle', { id: 'marker-unscored', r: DOT_RADIUS }));
  return svgEl('defs', {}, shapes);
}

function scatterDot(point, index) {
  if (!shares()) {
    return svgEl('circle', {
      class: 'dot-point',
      'data-quadrant': point.quadrant || '',
      'data-index': String(index),
      cx: point.x.toFixed(2),
      cy: point.y.toFixed(2),
      r: 3.4,
    });
  }
  return svgEl('use', {
    class: 'dot-point',
    'data-type': point.quadrant || '',
    'data-index': String(index),
    href: `#marker-${markerIds.has(point.quadrant) ? point.quadrant : 'unscored'}`,
    x: point.x.toFixed(2),
    y: point.y.toFixed(2),
  });
}

function buildScatter(points, summary) {
  return svgEl('svg', {
    class: 'scatter',
    viewBox: `0 0 ${SCATTER_BOX.width} ${SCATTER_BOX.height}`,
    role: 'img',
    tabindex: '0',
    'aria-label': summary,
    'aria-describedby': 'chart-caption',
  }, [
    shares() ? markerDefs() : null,
    ...shadedArea(),
    ...scatterFrame(),
    ...points.map(scatterDot),
    ...regionLabels(),
    svgEl('circle', { class: 'dot-marker', r: 7, cx: -30, cy: -30 }),
  ]);
}

function legendMark(shape) {
  return svgEl('svg', {
    class: 'legend-mark', viewBox: '-8 -8 16 16', 'aria-hidden': 'true', 'data-type': shape.code,
  }, [svgEl('path', { d: shape.d })]);
}

/** Which types the dots on screen stand for, by shape as well as by colour. */
function scatterLegend(view) {
  const entries = model.typeLegend(view.rows, state.scheme);
  if (!entries.length) return null;
  const shapes = new Map(model.markerShapes(LEGEND_RADIUS).map((shape) => [shape.code, shape]));
  return el('ul', { class: 'type-legend' }, entries.map((entry) => el('li', codeAttr(entry.code), [
    legendMark(shapes.get(entry.code)),
    el('span', { text: entry.label }),
  ])));
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
    const index = event.target.dataset?.index;
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
  const points = model.scatterPoints(view.rows, SCATTER_BOX, state.scheme);
  const svg = buildScatter(points, model.visualSummary('scatter', view.group, view.rows, state.scheme));
  const legend = scatterLegend(view);
  const note = model.scatterCaption(state.scheme);
  panel.append(
    summaryLine('scatter', view),
    el('div', { class: 'scatter-wrap' }, [svg,
      chartCaption('Hover a dot, or focus the chart and use the arrow keys, to name a job.')]),
  );
  if (note) panel.append(el('p', { class: 'chart-note small muted', text: note }));
  if (legend) panel.append(legend);
  panel.append(el('div', { class: 'chart-actions' }, [tableButton(view)]));
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
    state.colourChosen = true;
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
      model.colourModes(state.scheme).map((mode) => colourButton(mode, view))),
  ]);
}

function renderTreemap(panel, view) {
  const canvas = el('canvas', {
    id: 'treemap-canvas',
    role: 'img',
    tabindex: '0',
    'aria-label': model.visualSummary('treemap', view.group, view.rows, state.scheme),
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
  treemap.setTiles(model.treemapTiles(state.groups, state.index, view.key, state.scheme));
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
  const parent = state.groups[groupKey]?.parent;
  if (!parent) return;
  state.focusChart = true;
  go(groupHref(parent, 'treemap'));
}

/* Table -------------------------------------------------------------------- */

/** The sorted table is in the URL, so the Copy link button really copies it. */
function sortTable(column, view) {
  const same = view.sort.key === column.key;
  const sort = {
    key: column.key,
    descending: same ? !view.sort.descending : Boolean(column.numeric),
  };
  state.focusSort = column.key;
  go(model.groupViewHref(view.key, 'table', sort, state.scheme));
}

function headerCell(column, view) {
  const sorted = view.sort.key === column.key;
  const direction = view.sort.descending ? 'descending' : 'ascending';
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

function restoreSortFocus(panel) {
  if (!state.focusSort) return;
  const header = panel.querySelector(`th[data-key="${state.focusSort}"] button`);
  state.focusSort = null;
  if (header) header.focus();
}

function renderTable(panel, view) {
  const sorted = model.sortByColumn(view.rows, view.sort.key, view.sort.descending);
  const table = el('table', { class: shares() ? 'table--wide' : null }, [
    el('caption', { text: `${formatCount(sorted.length)} jobs in ${view.group.label}. Every column sorts.` }),
    el('thead', {}, [el('tr', {}, model.columnsFor(state.scheme).map((column) => headerCell(column, view)))]),
    el('tbody', {}, model.tableBody(sorted, state.scheme).map((entry) => bodyRow(entry, view.key))),
  ]);
  panel.append(summaryLine('table', view), el('div', { class: 'table-scroll' }, [table]));
  restoreSortFocus(panel);
}

/* --- lists ---------------------------------------------------------------- */

function jobFigure(job) {
  if (shares() && job.shares) {
    return renderSharesBar(job.shares, { name: job.title, compact: true });
  }
  return el('span', {
    class: 'scores',
    text: shares()
      ? 'Shares not scored'
      : `automation ${formatScore(job.automation)} · amplification ${formatScore(job.amplification)}`,
  });
}

function jobItem(job, groupKey) {
  return el('li', shares() ? { class: 'job-row' } : {}, [
    link(jobHref(job.slug, { from: groupKey }), job.title),
    jobFigure(job),
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

function skillColumn(column, pending) {
  const items = column.entries.length
    ? column.entries.map(skillItem)
    : [el('li', { class: 'empty-note', text: pending ? 'Loading skills…' : column.empty })];
  return el('div', {}, [el('h3', { text: column.heading }), el('ul', { class: 'job-list' }, items)]);
}

function loadSkillTitles(view) {
  if (state.skills || state.skillsPending) return;
  state.skillsPending = true;
  loadJSON('skill_index')
    .then((rows) => {
      state.skills = new Map(rows.map((row) => [row.id, row]));
      renderSkills(view);
    })
    .catch(() => { state.skillsPending = false; });
}

/**
 * skill_index.json is a megabyte spent on two lists of five names, so it is
 * only fetched once the section that needs it is actually on screen. Without
 * IntersectionObserver the load simply starts at once.
 */
function watchSkills(view) {
  if (state.skillsWatch) state.skillsWatch.disconnect();
  state.skillsWatch = null;
  if (!model.skillIdsOf(view.group).length || state.skills || state.skillsPending) return;
  if (typeof IntersectionObserver !== 'function') {
    loadSkillTitles(view);
    return;
  }
  state.skillsWatch = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    state.skillsWatch.disconnect();
    state.skillsWatch = null;
    loadSkillTitles(view);
  }, { rootMargin: '200px' });
  state.skillsWatch.observe(byId('skills-section'));
}

function renderSkills(view) {
  const skills = model.drivingSkills(view.group, state.skills, state.scheme);
  const note = model.skillsNote(state.scheme);
  byId('skills-note').textContent = note;
  show(byId('skills-note'), Boolean(note));
  byId('skill-columns').replaceChildren(
    ...skills.columns.map((column) => skillColumn(column, skills.pending)),
  );
  watchSkills(view);
}

function renderLists(view) {
  const headings = model.exposureHeadings(state.scheme);
  byId('top-heading').textContent = headings.top;
  byId('bottom-heading').textContent = headings.bottom;
  const lists = model.exposureLists({
    scheme: state.scheme, group: view.group, rows: view.rows, bySlug: state.bySlug,
  });
  fillList('top-list', lists.top, view.key);
  fillList('bottom-list', lists.bottom, view.key);
  renderSkills(view);
}

/* --- comparison ----------------------------------------------------------- */

function compareShares(side) {
  if (!side.shares) return null;
  return el('div', { class: 'compare-shares' }, [
    renderSharesBar(side.shares, { name: side.label, compact: true }),
  ]);
}

function compareCard(side) {
  return el('section', { class: 'card' }, [
    el('h2', {}, [link(groupHref(side.key), side.label)]),
    el('p', { class: 'muted small', text: side.subtitle }),
    compareShares(side),
    el('ul', { class: 'mix-bars' }, side.bars.map(mixRow)),
    el('p', { class: 'medians', text: model.medianSentence(side, state.scheme) }),
  ]);
}

function divergeRow(row) {
  const size = Math.abs(row.delta) * 50;
  const style = row.delta >= 0 ? `left: 50%; width: ${size}%;` : `right: 50%; width: ${size}%;`;
  const attr = CLASS_CODES.has(row.code) ? { 'data-class': row.code } : codeAttr(row.code);
  return el('li', attr, [
    el('span', { class: 'mix-name', text: row.label }),
    el('span', { class: 'diverge-track' }, [el('span', { class: 'diverge-fill', style })]),
    el('span', { class: 'diverge-figure', text: `${row.text} · ${row.deltaText}` }),
  ]);
}

function divergeSection(heading, lead, rows) {
  return el('section', { class: 'card' }, [
    el('h2', { text: heading }),
    el('p', { class: 'small muted', text: lead }),
    el('ul', { class: 'diverge' }, rows.map(divergeRow)),
  ]);
}

/** The caveat belongs to the comparison, not to each half of it. */
function caveatLine() {
  return el('p', { class: 'mix-caveat small', text: model.nearLineCaveat(state.stats) });
}

function compareSections(comparison) {
  const names = `${comparison.a.label} minus the share in ${comparison.b.label}`;
  const sections = [
    el('div', { class: 'compare-columns' }, [compareCard(comparison.a), compareCard(comparison.b)]),
  ];
  if (comparison.shareDifferences.length) {
    sections.push(divergeSection('Where the work differs',
      `Each bar is the mean share of the skills in ${names}.`, comparison.shareDifferences));
  }
  const mix = divergeSection(shares() ? 'Where the types differ' : 'Where they differ',
    shares() ? `Each bar is the share of jobs in ${names}.` : `Each bar is the share in ${names}.`,
    comparison.differences);
  mix.append(caveatLine());
  sections.push(mix);
  return sections;
}

function renderCompare(compare) {
  const comparison = model.compareGroups(state.groups, compare.a, compare.b, state.scheme);
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
