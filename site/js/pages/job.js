// job.html — the one canonical job page. DOM wiring only: every decision this
// file makes is taken in ./job-model.js or ./framing.js, which are pure and
// unit-tested.
//
// What the page owes the visitor, and where each promise is kept:
//   - a real back link, always            renderBack / backTarget
//   - the class, hedged                   renderTypeBadge from ../badge.js
//   - what the job is made of             renderSharesBar + shareMeanings
//   - a chart with a text alternative     renderScatter + renderScatterTable
//   - "your job" or neutral wording       ./framing.js, `&for=other` in the hash
//   - a next step, always                 renderAdvice
//   - Back that walks the real trail      history.pushState + popstate/hashchange
//
// The active set decides between two schemes once, in `receive`, and every
// branch below reads `state.scheme`: a quadrant set renders exactly as it did
// before, a shares set leads with the four shares and names three scores.

import { renderChrome } from '../chrome.js';
import { LoadError, loadJSON, whenSlow } from '../data.js';
import { createCombobox } from '../combobox.js';
import { rankOccupations, nearestTitles } from '../search.js';
import { renderTypeBadge } from '../badge.js';
import { renderInfoNote } from '../info-note.js';
import { renderSharesBar } from '../shares-bar.js';
import { borrowedRationale, rationaleWriterLine } from '../rationale.js';
import { THRESHOLD } from '../quadrant.js';
import {
  SHARES, SKILL_CLASS_ORDER, SKILL_NEAR_NOTE, skillClassName, thresholdOf, typeShortLabel,
} from '../scheme.js';
import { formatCount, formatPercent, formatScore, isScored } from '../format.js';
import { groupHref, jobHref, parseHash, skillHref, withScorer } from '../urlstate.js';
import {
  MOVE_MARGIN, RECENT_KEY, SHARE_MARGIN, UNITS_FILE, addRecent, advicePosition, backTarget,
  essentialSkills, evolutionPaths, findOccupation, gapSkills, occupationIndex,
  occupationSkills, scatterCells, scatterColumns, scatterPoints, scatterSummary,
  chanceLine, classTaskLists, largestShareSentence, schemeOfSet, scoreCards, shareMeanings,
  skillListSpecs, skillNear, staysHumanNote, unitFile, unitGroupKey,
} from './job-model.js';
import {
  framingParam, isNeutral, normaliseFraming, otherFraming, recordsRecent, wordingFor,
} from './framing.js';

const state = {
  units: null, data: null, groups: null, index: null, indexPromise: null, stats: null,
  scheme: undefined, byslug: new Map(), points: [], started: false, pending: null,
};

/** A wait shorter than this needs no loading state at all. */
const SLOW_MS = 200;

/** And past this one, saying only "loading" stops being honest. */
const VERY_SLOW_MS = 3000;

const NEUTRAL_NOTE = 'Neutral wording: this page is written about the job, not to the '
  + 'person doing it.';

// --- small DOM helpers ------------------------------------------------------

const byId = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function link(href, text, className) {
  const anchor = el('a', className, text);
  anchor.href = withScorer(href, window.location.search);
  return anchor;
}

/** A link to another job page, intercepted by the click handler below. */
function jobAnchor(slug, context, text) {
  const anchor = link(jobHref(slug, context), text);
  anchor.dataset.jobLink = '';
  return anchor;
}

function scoreSpan(value, name, className) {
  const span = el('span', className, formatScore(value));
  span.appendChild(el('span', 'visually-hidden', ` ${name}, out of 10. `));
  return span;
}

/** The skill class as a named chip. The letter is never printed on its own. */
function classChip(code) {
  const chip = el('span', 'class-chip', skillClassName(code));
  chip.dataset.class = code;
  return chip;
}

/** C12. The same marker the job badge carries, for a skill that sits as close. */
function nearChip() {
  const chip = el('span', 'near-chip', 'near the line');
  chip.title = SKILL_NEAR_NOTE;
  chip.appendChild(el('span', 'visually-hidden', `. ${SKILL_NEAR_NOTE}`));
  return chip;
}

function isNearSkill(skill) {
  return shares() && skillNear(skill, thresholdOf(state.stats) ?? undefined);
}

function shares() {
  return state.scheme === SHARES;
}

function capitalise(text) {
  const value = String(text || '');
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function plural(count, noun) {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}

// --- route and navigation ---------------------------------------------------

function readRoute() {
  const { id, params } = parseHash(window.location.hash);
  return {
    slug: id, framing: normaliseFraming(params.for), from: params.from || null,
  };
}

/** What a link out of this page should carry: the trail and the wording mode. */
function linkContext(route, occupation) {
  return {
    for: framingParam(route.framing),
    from: route.from || unitGroupKey(occupation && occupation.c) || '',
  };
}

function go(href) {
  window.history.pushState(null, '', withScorer(href, window.location.search));
  render(true);
}

function onDocumentClick(event) {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const target = event.target instanceof Element ? event.target : null;
  const anchor = target && target.closest('a[data-job-link]');
  if (!anchor) return;
  event.preventDefault();
  go(anchor.getAttribute('href'));
}

// --- page states ------------------------------------------------------------

function setState(name) {
  byId('job-article').hidden = name !== 'job';
  byId('job-intro').hidden = name === 'job';
  byId('job-tools').hidden = name !== 'job';
  if (name !== 'job') byId('job-note').hidden = true;
}

function chipRow(...children) {
  const row = el('p', 'chip-row');
  row.append(...children);
  return row;
}

function showPrompt() {
  document.title = 'Will AI change this job? — AI-ISCO';
  byId('job-title').textContent = 'Pick a job to see how AI could change it';
  byId('job-meta').replaceChildren();
  byId('job-intro').replaceChildren(
    el('p', null, 'This page shows one occupation at a time: what its skills are made '
      + 'of, what changes in the work, and where someone could go next.'),
    chipRow(
      link('index.html', 'Find a job →', 'chip'),
      link('groups.html', 'Browse sectors →', 'chip'),
    ),
  );
  setState('intro');
}

function suggestionList(query) {
  const rows = state.index ? nearestTitles(state.index, query, 3) : [];
  if (!rows.length) return null;
  const list = el('ul');
  for (const row of rows) {
    const item = el('li');
    item.appendChild(jobAnchor(row.s, {}, capitalise(row.t)));
    list.appendChild(item);
  }
  return list;
}

function showNotFound(slug) {
  const spoken = slug.replace(/-/g, ' ');
  document.title = 'Job not found — AI-ISCO';
  byId('job-title').textContent = `We don’t have a job called “${spoken}”`;
  byId('job-meta').replaceChildren();
  const intro = byId('job-intro');
  intro.replaceChildren(el('p', null, 'The titles come from ESCO, which uses formal '
    + 'wording, so this job may be listed under another name. Search with the box at the '
    + 'top of this page, or browse by sector.'));
  intro.appendChild(chipRow(link('groups.html', 'Browse sectors →', 'chip')));
  setState('intro');
  suggestTitles(intro, spoken);
}

/**
 * The closest titles we do have, under the "not found" text. The index is
 * 1.3 MB and only this state and the finder need it, so it is fetched here
 * rather than on every job page.
 */
function suggestTitles(intro, spoken) {
  ensureIndex().then(() => {
    if (state.pending !== null || !state.index) return;
    const suggestions = suggestionList(spoken);
    if (!suggestions) return;
    intro.insertBefore(el('p', 'small muted', 'The closest titles we do have:'),
      intro.lastElementChild);
    intro.insertBefore(suggestions, intro.lastElementChild);
  });
}

function errorActions() {
  const retry = el('button', 'chip', 'Try again');
  retry.type = 'button';
  retry.addEventListener('click', () => {
    byId('job-error').replaceChildren();
    start();
  });
  return chipRow(retry, link('groups.html', 'Browse sectors instead →', 'chip'));
}

function showError(error) {
  const block = el('div', 'error-block');
  block.append(
    el('h2', null, 'The data file didn’t arrive'),
    el('p', null, error instanceof LoadError ? error.message : String(error)),
    el('p', null, 'The job data is one large file, so a slow or interrupted connection is '
      + 'the usual cause. Nothing here is broken by trying again.'),
    errorActions(),
  );
  byId('job-error').replaceChildren(block);
  byId('job-title').textContent = 'We couldn’t load this job';
  byId('job-meta').replaceChildren();
  setState('intro');
  byId('job-intro').replaceChildren();
}

// --- header -----------------------------------------------------------------

function renderBack(route, occupation) {
  const target = backTarget(route.from, occupation, state.groups);
  const anchor = link(target.href, target.text);
  byId('job-back').replaceChildren(anchor);
}

function metaLinks(occupation) {
  const digits = String(occupation.c || '').replace(/\D/g, '');
  const wanted = [
    { key: unitGroupKey(occupation.c), text: `ISCO ${occupation.c}` },
    { key: digits.length >= 2 ? `sub:${digits.slice(0, 2)}` : null, text: occupation.cat },
    { key: digits.length >= 1 ? `major:${digits.slice(0, 1)}` : null, text: occupation.mg },
  ];
  return wanted
    .filter((item) => item.key && item.text)
    .map((item) => link(groupHref(item.key), capitalise(item.text)));
}

function renderMeta(occupation) {
  const meta = byId('job-meta');
  // The separators are ::before content on the links (css/job.css), so a "·"
  // can never wrap onto a line of its own when a group name is long.
  meta.replaceChildren(...metaLinks(occupation));
  meta.appendChild(renderTypeBadge(
    {
      t: occupation.t, a: occupation.ar, m: occupation.ap, q: occupation.q,
      sh: occupation.sh, nl: occupation.nl,
    },
    { scheme: state.scheme, search: window.location.search },
  ));
}

function renderTools(route, words) {
  const toggle = byId('job-framing');
  toggle.textContent = words.toggleLabel;
  toggle.href = withScorer(
    jobHref(route.slug, {
      for: framingParam(otherFraming(route.framing)), from: route.from || '',
    }),
    window.location.search,
  );
  toggle.dataset.jobLink = '';
  byId('job-copy-status').textContent = '';
  const note = byId('job-note');
  note.hidden = !isNeutral(route.framing);
  note.textContent = NEUTRAL_NOTE;
}

function copyLink() {
  const status = byId('job-copy-status');
  const done = (message) => { status.textContent = message; };
  const url = window.location.href;
  if (!navigator.clipboard) {
    done('Copying is not available here. The link is in the address bar.');
    return;
  }
  navigator.clipboard.writeText(url).then(
    () => done('Link copied.'),
    () => done('Couldn’t copy. The link is in the address bar.'),
  );
}

// --- what the job is made of ------------------------------------------------

function scoreBar(value) {
  const bar = el('div', 'score-bar');
  const fill = el('div', 'score-fill');
  fill.style.width = `${Math.max(0, Math.min(100, (value / 10) * 100))}%`;
  bar.appendChild(fill);
  return bar;
}

function scoreCard(spec, occupation) {
  const value = occupation[spec.key];
  const scored = isScored(value);
  const card = el('div', `score-card score-card--${spec.modifier}`);
  card.append(
    el('p', 'score-label', `${spec.label}, ${spec.unit}`),
    el('p', scored ? 'score-value numeric' : 'score-value is-missing', formatScore(value)),
  );
  if (scored) card.appendChild(scoreBar(value));
  card.appendChild(el('p', 'score-hint', spec.hint));
  return card;
}

function shareLine(part) {
  const item = el('li');
  const name = el('span', 'share-line-name', part.name);
  name.dataset.class = part.code;
  item.append(name, el('span', 'share-line-text', part.text));
  return item;
}

/**
 * C6. The first screen carries the bar and ONE sentence: the largest share,
 * named. Everything else that is a number — the other three share sentences and
 * the three averaged scores — moves into "The numbers behind this", because a
 * first screen of twelve quantities is not an answer to "will AI change my job".
 */
function renderShares(occupation) {
  const box = byId('job-shares');
  box.hidden = !shares();
  if (!shares()) {
    box.replaceChildren();
    return;
  }
  box.replaceChildren(
    renderSharesBar(occupation.sh, { name: occupation.t }),
    el('p', 'job-headline', largestShareSentence(occupation.sh)),
  );
}

/** The rest of the share sentences, and the score cards, behind one disclosure. */
function renderNumbers(occupation) {
  const details = byId('job-numbers');
  const body = byId('job-numbers-body');
  const list = byId('score-list');
  details.hidden = !shares();
  if (!shares()) {
    body.replaceChildren();
    list.hidden = false;
    details.parentNode.insertBefore(list, details);
    return;
  }
  const lines = el('ul', 'share-lines');
  lines.replaceChildren(...shareMeanings(occupation.sh, occupation.why).map(shareLine));
  list.hidden = false;
  body.replaceChildren(lines, list);
}

function renderScores(occupation, words) {
  byId('scores-heading').textContent = words.scoresHeading;
  byId('scores-intro').textContent = words.scoresIntro;
  renderShares(occupation);
  byId('score-list').replaceChildren(
    ...scoreCards(state.scheme).map((spec) => scoreCard(spec, occupation)),
  );
  renderNumbers(occupation);
}

function adviceText(occupation, words) {
  const narrative = occupation.n || {};
  return narrative.adv || words.adviceFallback;
}

function renderAdvice(occupation, words) {
  const card = el('section', 'card card--advice');
  card.append(
    el('h2', null, words.adviceHeading),
    el('p', null, adviceText(occupation, words)),
    el('p', 'small muted', shares()
      ? words.geminiNote
      : 'Written by the same model pass as the story below.'),
  );
  const top = advicePosition(occupation, state.scheme) === 'top';
  byId('advice-top').replaceChildren(...(top ? [card] : []));
  byId('advice-bottom').replaceChildren(...(top ? [] : [card]));
}

// --- scatter plot -----------------------------------------------------------

const PAD = { top: 18, right: 18, bottom: 46, left: 46 };
let palette = null;

function colours() {
  if (palette) return palette;
  const style = getComputedStyle(document.documentElement);
  const read = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
  palette = {
    bg: read('--bg3', '#1b1b26'),
    grid: 'rgba(255,255,255,0.06)',
    axis: 'rgba(255,255,255,0.28)',
    text: 'rgba(255,255,255,0.62)',
    TRANSFORM: read('--q-transform', '#e8b93b'),
    SHRINK: read('--q-shrink', '#ef6a5e'),
    EVOLVE: read('--q-evolve', '#55c077'),
    STABLE: read('--q-stable', '#6aa9e8'),
    S: read('--class-substituted', '#ef6a5e'),
    A: read('--class-assisted', '#6aa9e8'),
    M: read('--class-mechanised', '#c79bf0'),
    I: read('--class-insulated', '#55c077'),
  };
  return palette;
}

function plotBox(size) {
  return {
    x: PAD.left,
    y: PAD.top,
    w: Math.max(40, size.width - PAD.left - PAD.right),
    h: Math.max(40, size.height - PAD.top - PAD.bottom),
  };
}

const toX = (box, value) => box.x + ((value - 1) / 9) * box.w;
const toY = (box, value) => box.y + box.h - ((value - 1) / 9) * box.h;

function prepareCanvas(canvas) {
  const width = Math.max(240, canvas.parentElement.clientWidth || 320);
  const size = { width, height: Math.max(260, Math.min(430, width * 0.62)) };
  const ratio = window.devicePixelRatio || 1;
  canvas.width = size.width * ratio;
  canvas.height = size.height * ratio;
  canvas.style.height = `${size.height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, size };
}

function drawGrid(ctx, size, box) {
  const ink = colours();
  ctx.fillStyle = ink.bg;
  ctx.fillRect(0, 0, size.width, size.height);
  ctx.strokeStyle = ink.grid;
  ctx.lineWidth = 1;
  for (let value = 1; value <= 10; value += 1) {
    ctx.beginPath();
    ctx.moveTo(toX(box, value), box.y);
    ctx.lineTo(toX(box, value), box.y + box.h);
    ctx.moveTo(box.x, toY(box, value));
    ctx.lineTo(box.x + box.w, toY(box, value));
    ctx.stroke();
  }
}

// Only the quadrant scheme has a cut-off to draw: under shares the two axes are
// display scores, and a skill's class comes from the model's probabilities.
function drawThreshold(ctx, box) {
  if (shares()) return;
  ctx.save();
  ctx.strokeStyle = colours().axis;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(toX(box, THRESHOLD), box.y);
  ctx.lineTo(toX(box, THRESHOLD), box.y + box.h);
  ctx.moveTo(box.x, toY(box, THRESHOLD));
  ctx.lineTo(box.x + box.w, toY(box, THRESHOLD));
  ctx.stroke();
  ctx.restore();
}

function drawAxes(ctx, box) {
  const ink = colours();
  ctx.fillStyle = ink.text;
  ctx.font = '400 10px -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let value = 1; value <= 10; value += 2) {
    ctx.fillText(String(value), toX(box, value), box.y + box.h + 6);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let value = 1; value <= 10; value += 2) {
    ctx.fillText(String(value), box.x - 8, toY(box, value));
  }
}

function axisTitles() {
  return shares()
    ? ['AI substitution, 1 to 10', 'AI assistance, 1 to 10']
    : ['Automation risk, 1 to 10', 'Amplification, 1 to 10'];
}

function drawAxisTitles(ctx, box) {
  const [across, up] = axisTitles();
  ctx.save();
  ctx.fillStyle = colours().text;
  ctx.font = '500 11px -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(across, box.x + box.w / 2, box.y + box.h + 24);
  ctx.translate(12, box.y + box.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText(up, 0, 0);
  ctx.restore();
}

// Four shapes as well as four colours, so the classes stay apart in greyscale.
function circlePath(ctx, x, y, r) {
  ctx.arc(x, y, r, 0, Math.PI * 2);
}

function trianglePath(ctx, x, y, r) {
  ctx.moveTo(x, y - r * 1.1);
  ctx.lineTo(x + r, y + r * 0.8);
  ctx.lineTo(x - r, y + r * 0.8);
  ctx.closePath();
}

function squarePath(ctx, x, y, r) {
  ctx.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7);
}

function diamondPath(ctx, x, y, r) {
  ctx.moveTo(x, y - r * 1.2);
  ctx.lineTo(x + r * 1.2, y);
  ctx.lineTo(x, y + r * 1.2);
  ctx.lineTo(x - r * 1.2, y);
  ctx.closePath();
}

const CLASS_SHAPES = { S: circlePath, A: trianglePath, M: squarePath, I: diamondPath };

function dotPath(ctx, point, box, radius) {
  const shape = (shares() && CLASS_SHAPES[point.cls]) || circlePath;
  shape(ctx, toX(box, point.auto), toY(box, point.amp), radius);
}

function dotColour(point) {
  const ink = colours();
  return (shares() ? ink[point.cls] : ink[point.q]) || ink.text;
}

function drawDot(ctx, box, point) {
  const colour = dotColour(point);
  ctx.beginPath();
  dotPath(ctx, point, box, point.essential ? 6 : 4.5);
  if (point.essential) {
    ctx.fillStyle = colour;
    ctx.globalAlpha = 0.8;
    ctx.fill();
    ctx.globalAlpha = 1;
    return;
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.8;
  ctx.stroke();
}

function drawScatter() {
  const canvas = byId('scatter');
  if (!canvas || byId('job-article').hidden) return;
  const { ctx, size } = prepareCanvas(canvas);
  const box = plotBox(size);
  drawGrid(ctx, size, box);
  drawThreshold(ctx, box);
  drawAxes(ctx, box);
  drawAxisTitles(ctx, box);
  const ordered = [...state.points].sort((a, b) => Number(a.essential) - Number(b.essential));
  for (const point of ordered) drawDot(ctx, box, point);
}

function nearestPoint(box, x, y) {
  let best = null;
  let bestDistance = 14;
  for (const point of state.points) {
    const distance = Math.hypot(toX(box, point.auto) - x, toY(box, point.amp) - y);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = point;
    }
  }
  return best;
}

function tipText(point) {
  const place = point.essential ? 'essential' : 'optional';
  if (shares()) {
    return `${skillClassName(point.cls)} · AI substitution ${formatScore(point.auto)}, `
      + `AI assistance ${formatScore(point.amp)}, machine automation `
      + `${formatScore(point.mech)}, out of 10 · ${place}`;
  }
  return `Automation risk ${formatScore(point.auto)}, amplification `
    + `${formatScore(point.amp)}, out of 10 · ${place}`;
}

function showTip(point, x, y) {
  const tip = byId('scatter-tip');
  tip.replaceChildren(el('strong', null, point.title), el('span', null, tipText(point)));
  tip.style.left = `${Math.max(0, x - 40)}px`;
  tip.style.top = `${Math.max(0, y - 70)}px`;
  tip.hidden = false;
}

function onScatterMove(event) {
  const canvas = byId('scatter');
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const point = nearestPoint(plotBox({ width: rect.width, height: rect.height }), x, y);
  if (point) showTip(point, x, y);
  else byId('scatter-tip').hidden = true;
}

function headCell(column) {
  const cell = el('th', column.numeric ? 'numeric' : null, column.label);
  cell.scope = 'col';
  return cell;
}

function bodyCell(cell) {
  return el('td', cell.numeric ? 'numeric' : null,
    cell.numeric ? formatScore(cell.value) : cell.value);
}

function skillRow(point) {
  const row = el('tr');
  const name = el('td');
  name.appendChild(link(skillHref(point.id), point.title));
  row.append(name, ...scatterCells(point, state.scheme).map(bodyCell));
  return row;
}

function renderScatterTable(occupation) {
  byId('scatter-head').replaceChildren(...scatterColumns(state.scheme).map(headCell));
  byId('scatter-caption').textContent = `Every skill in ${occupation.t} that carries both `
    + `scores: ${plural(state.points.length, 'skill')}, one for each dot in the plot above.`;
  byId('scatter-rows').replaceChildren(...state.points.map(skillRow));
}

function classLegendItem(code) {
  const item = el('span');
  const mark = el('span', 'legend-mark');
  mark.dataset.class = code;
  item.append(mark, el('span', null, skillClassName(code)));
  return item;
}

function renderClassLegend() {
  const legend = byId('scatter-classes');
  legend.hidden = !shares();
  legend.replaceChildren(...(shares() ? SKILL_CLASS_ORDER.map(classLegendItem) : []));
}

function renderScatter(occupation, words) {
  byId('scatter-heading').textContent = words.scatterHeading;
  byId('scatter-intro').textContent = words.scatterIntro;
  byId('scatter-threshold').textContent = shares()
    ? '' : `Dashed lines: the cut-off of ${THRESHOLD}`;
  renderClassLegend();
  byId('scatter').setAttribute('aria-label',
    scatterSummary(occupation.t, state.points, state.scheme));
  renderScatterTable(occupation);
  drawScatter();
}

function toggleScatterTable() {
  const wrap = byId('scatter-table-wrap');
  const button = byId('scatter-toggle');
  wrap.hidden = !wrap.hidden;
  button.setAttribute('aria-expanded', String(!wrap.hidden));
  button.textContent = wrap.hidden ? 'View as table' : 'Hide the table';
}

function openScatterTable() {
  const wrap = byId('scatter-table-wrap');
  if (wrap.hidden) toggleScatterTable();
  const first = wrap.querySelector('a');
  (first || byId('scatter-toggle')).focus();
}

// --- skill lists and rationale cards ---------------------------------------

const SCORE_KEYS = ['auto', 'amp', 'mech'];

function scoreNames() {
  return shares()
    ? { auto: 'AI substitution', amp: 'AI assistance', mech: 'machine automation' }
    : { auto: 'automation risk', amp: 'amplification' };
}

function shownKeys() {
  return shares() ? SCORE_KEYS : ['auto', 'amp'];
}

/** The scores of one skill, the one its list is about first. */
function skillScores(skill, primary) {
  const names = scoreNames();
  const keys = [primary, ...shownKeys().filter((key) => key !== primary)];
  return keys.map((key, index) => scoreSpan(
    skill[key], names[key], `sk-score sk-score--${index ? 'muted' : primary}`,
  ));
}

function skillItem(skill, primary) {
  const item = el('li');
  item.appendChild(link(skillHref(skill.id), skill.title));
  item.append(...skillScores(skill, primary));
  return item;
}

function skillListCard(spec, words) {
  const card = el('section', 'card');
  const heading = el('h2', null, words[spec.headingKey]);
  heading.id = `${spec.key}-heading`;
  const note = el('p', 'small muted', spec.note);
  note.id = `${spec.key}-count`;
  const list = el('ul', 'skill-list');
  list.id = `${spec.key}-list`;
  list.replaceChildren(...spec.skills.map((skill) => skillItem(skill, spec.primary)));
  card.append(heading, note, list);
  return card;
}

function staysHumanNodes(skills) {
  const text = shares() ? staysHumanNote(skills) : '';
  if (!text) return [];
  const button = el('button', 'button button--quiet', 'See every skill in the table');
  button.type = 'button';
  button.addEventListener('click', openScatterTable);
  return [document.createTextNode(`${text} `), button];
}

function renderSkillLists(skills, words) {
  const specs = skillListSpecs(skills, state.scheme);
  byId('skill-columns').replaceChildren(...specs.map((spec) => skillListCard(spec, words)));
  const note = byId('stays-human');
  const nodes = staysHumanNodes(skills);
  note.hidden = !nodes.length;
  note.replaceChildren(...nodes);
}

function cardScores(skill) {
  const names = scoreNames();
  return shownKeys().map((key) => scoreSpan(skill[key], names[key], `sk-score sk-score--${key}`));
}

/**
 * C1. The row that carries a class carries the chances that decided it, in
 * words, and the three 1-10 scores come after them. Without this the page shows
 * a label and three numbers that look like they disagree with it, and never the
 * quantity the label actually came from.
 */
function cardSummary(skill) {
  const summary = el('summary');
  const row = el('span', 'sc-row');
  const head = el('span', 'sc-head');
  head.appendChild(el('span', 'sc-name', skill.title));
  if (shares() && skill.cls) head.appendChild(classChip(skill.cls));
  if (isNearSkill(skill)) head.appendChild(nearChip());
  row.appendChild(head);
  const chances = shares() ? chanceLine(skill.probs) : '';
  if (chances) row.appendChild(el('span', 'sc-chances', chances));
  const scores = el('span', 'sc-scores');
  scores.append(...cardScores(skill));
  row.appendChild(scores);
  summary.appendChild(row);
  return summary;
}

function skillCard(skill) {
  const card = el('details', 'skill-card');
  const more = el('p');
  more.appendChild(link(skillHref(skill.id), `Look up ${skill.title} →`));
  const body = el('div', 'skill-card-body');
  body.append(...rationaleNodes(skill), more);
  card.append(cardSummary(skill), body);
  return card;
}

/**
 * C2. A borrowed explanation is printed only while it is still about the same
 * skill: past the gap limit it argued with the class beside it, so the page
 * says whose score it was written for and leaves the sentence out.
 */
function rationaleNodes(skill) {
  const verdict = borrowedRationale({
    text: skill.rationale, source: skill.rationaleFrom, score: skill.auto,
  });
  if (!verdict.show) {
    return [el('p', 'rationale-missing', verdict.why
      || 'The model wrote no rationale for this skill.')];
  }
  const nodes = [];
  if (verdict.lead) nodes.push(el('p', 'rationale-lead', verdict.lead));
  nodes.push(el('p', null, skill.rationale));
  if (verdict.after) nodes.push(el('p', 'rationale-after', verdict.after));
  return nodes;
}

function renderCards(occupation, words) {
  byId('cards-heading').textContent = words.cardsHeading;
  byId('cards-key').textContent = words.cardsKey;
  const skills = essentialSkills(state.data, occupation);
  const list = byId('cards-list');
  if (!skills.length) {
    list.replaceChildren(el('p', 'muted', 'This job lists no essential skills.'));
    return;
  }
  byId('cards-writer').textContent = rationaleWriterLine(skills);
  list.replaceChildren(...skills.map(skillCard));
}

// --- the narrative ----------------------------------------------------------

function paragraphs(text) {
  return String(text)
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => el('p', null, line));
}

function storyFacts(narrative) {
  const facts = [];
  if (narrative.tl) facts.push(el('span', 'story-fact', `Over about ${narrative.tl}`));
  if (isScored(narrative.ts)) {
    facts.push(el('span', 'story-fact', `~${narrative.ts}% of routine time freed`));
  }
  return facts;
}

/**
 * C11. The fact pills ("over about 2-3 years", "~60% of routine time freed")
 * are stored strings from the Gemini run and cannot be reproduced from any v2
 * file; one of them sat 300 px from a different ~60% meaning something else. So
 * they are not shown under shares, and the story says whose it is.
 */
function renderStory(narrative, words) {
  byId('sec-story').hidden = !narrative.story;
  if (!narrative.story) return;
  byId('story-heading').textContent = words.storyHeading;
  const note = byId('story-note');
  const attribution = [words.storyNote, shares() ? words.geminiNote : ''].filter(Boolean);
  note.hidden = !attribution.length;
  note.textContent = attribution.join(' ');
  byId('story-facts').replaceChildren(...(shares() ? [] : storyFacts(narrative)));
  byId('story-text').replaceChildren(...paragraphs(narrative.story));
}

const GEMINI_TASK_LISTS = [
  { key: 'auto', heading: 'What AI takes on' },
  { key: 'amp', heading: 'What AI amplifies' },
  { key: 'tools', heading: 'Tools already doing it' },
];

function geminiTaskList(spec, narrative) {
  const items = narrative[spec.key] || [];
  if (!items.length) return null;
  const block = el('div', 'ai-list');
  const list = el('ul');
  for (const item of items) list.appendChild(el('li', null, item));
  block.append(el('h3', null, spec.heading), list);
  return block;
}

/** C3. One row per skill, named and linked, with the chance that ranked it. */
function classTaskItem(item) {
  const row = el('li');
  row.append(link(skillHref(item.id), item.title),
    el('span', 'ai-chance numeric', item.percent));
  return row;
}

function classTaskList(spec) {
  const block = el('div', 'ai-list');
  const list = el('ul', 'ai-list-skills');
  list.replaceChildren(...spec.items.map(classTaskItem));
  block.append(el('h3', null, spec.heading), el('p', 'small muted', spec.note), list);
  return block;
}

/**
 * C3. Under shares these two lists are this job's own skills, so they cannot
 * name a task as one AI takes on while a row below classes it as staying human.
 * The Gemini lists stay under the scheme they were written for.
 */
function taskBlocks(narrative, skills) {
  if (!shares()) {
    return GEMINI_TASK_LISTS.map((spec) => geminiTaskList(spec, narrative)).filter(Boolean);
  }
  return classTaskLists(skills).map(classTaskList);
}

function renderTasks(narrative, words, skills) {
  const blocks = taskBlocks(narrative, skills);
  byId('sec-tasks').hidden = !blocks.length;
  byId('tasks-heading').textContent = words.tasksHeading;
  byId('tasks-grid').replaceChildren(...blocks);
}

function weekColour(index) {
  const ink = colours();
  return [ink.STABLE, ink.EVOLVE, ink.TRANSFORM, ink.SHRINK, ink.text][index % 5];
}

function weekBlock(heading, data) {
  const block = el('div', 'week-block');
  const bar = el('div', 'week-bar');
  bar.setAttribute('aria-hidden', 'true');
  const legend = el('ul', 'week-legend');
  Object.entries(data).forEach(([key, share], index) => {
    const colour = weekColour(index);
    const segment = el('div', 'week-seg');
    segment.style.flexBasis = `${share}%`;
    segment.style.background = colour;
    bar.appendChild(segment);
    legend.appendChild(weekLegendItem(colour, key, share));
  });
  block.append(el('h3', null, heading), bar, legend);
  return block;
}

function weekLegendItem(colour, key, share) {
  const item = el('li');
  const swatch = el('span', 'week-swatch');
  swatch.style.background = colour;
  item.append(swatch, el('span', null, `${key.replace(/_/g, ' ')} ${share}%`));
  return item;
}

function renderWeek(narrative, words) {
  const week = narrative.week || {};
  const usable = Boolean(week.before && week.after) && !shares();
  byId('sec-week').hidden = !usable;
  if (!usable) return;
  byId('week-heading').textContent = words.weekHeading;
  byId('week-bars').replaceChildren(
    weekBlock('Before AI', week.before),
    weekBlock('After AI', week.after),
  );
}

// --- paths and gap skills ---------------------------------------------------

function pathStats(card) {
  const overlap = card.overlap === null
    ? 'skill overlap not scored'
    : `${formatPercent(card.overlap)} of its skills shared with this job`;
  const tail = `${overlap} · ${plural(card.gapCount, 'new skill')}`;
  if (shares()) return `${typeShortLabel(card.q, SHARES)} · ${tail}`;
  return `Automation risk ${formatScore(card.auto)}, amplification ${formatScore(card.amp)}, `
    + `out of 10 · ${tail}`;
}

function pathCard(card, context) {
  const anchor = jobAnchor(card.slug, context, '');
  anchor.className = 'path-card';
  const tag = el('span', 'path-tag', card.label);
  tag.dataset.kind = card.kind;
  anchor.append(el('span', 'path-title', card.title),
    el('span', 'visually-hidden', ': '), tag, el('span', 'visually-hidden', '. '));
  if (shares()) {
    anchor.appendChild(renderSharesBar(card.sh, { name: card.title, compact: true }));
  }
  anchor.appendChild(el('p', 'path-stats', pathStats(card)));
  return anchor;
}

function noPathsText(total) {
  if (shares()) {
    return `No move came out of the ${plural(total, 'nearby job')} beside this one: to count `
      + `as a move, a job needs at least ${Math.round(SHARE_MARGIN * 100)} points less of `
      + 'its work taken over by AI, or at least that much more of it assisted, without '
      + 'losing the other. They are listed below as what they are.';
  }
  return `None of the ${plural(total, 'nearby job')} is clearly less exposed or more `
    + `amplified than this one: every one of them sits within ${MOVE_MARGIN} of it on `
    + 'both scores. They are listed below as what they are.';
}

function pathsIntro(words, found, total) {
  if (!total) return 'The data lists no occupation next to this one.';
  return found ? words.pathsIntro : noPathsText(total);
}

/** With no move to show, the section ends on the lever that does exist. */
function renderNeedle(found, total, words) {
  const box = byId('paths-needle');
  const show = !found && total > 0;
  box.hidden = !show;
  box.replaceChildren(...(show
    ? [el('h3', null, words.needleHeading), el('p', null, words.needleText)]
    : []));
}

function renderPaths(occupation, context, words) {
  const { paths, others, total } = evolutionPaths(occupation, state.byslug,
    { scheme: state.scheme });
  byId('paths-heading').textContent = paths.length
    ? words.pathsHeading : words.pathsHeadingNone;
  byId('paths-intro').textContent = pathsIntro(words, paths.length, total);
  byId('paths-grid').replaceChildren(...paths.map((card) => pathCard(card, context)));
  byId('sideways-heading').textContent = others.length ? words.sidewaysHeading : '';
  byId('sideways-note').textContent = others.length ? words.sidewaysNote : '';
  byId('sideways-grid').replaceChildren(...others.map((card) => pathCard(card, context)));
  renderNeedle(paths.length, total, words);
}

function learnItem(item, context) {
  const row = el('li');
  row.append(
    scoreSpan(item.amp, shares() ? 'AI assistance' : 'amplification', 'learn-amp numeric'),
    link(skillHref(item.id), item.title),
  );
  if (shares() && item.cls) row.appendChild(classChip(item.cls));
  const from = el('p', 'learn-from');
  from.append(document.createTextNode('Needed by '), jobAnchor(item.fromSlug, context, item.fromTitle));
  row.appendChild(from);
  return row;
}

function renderLearn(occupation, context, words) {
  byId('learn-heading').textContent = words.learnHeading;
  const items = gapSkills(state.data, occupation, { scheme: state.scheme });
  byId('learn-intro').textContent = items.length
    ? words.learnIntro
    : 'The nearby jobs need no skill this one does not already list.';
  byId('learn-list').replaceChildren(...items.map((item) => learnItem(item, context)));
}

// --- recently viewed --------------------------------------------------------

function readRecent() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(RECENT_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function rememberRecent(slug, framing) {
  if (!recordsRecent(framing)) return;
  try {
    window.localStorage.setItem(RECENT_KEY, JSON.stringify(addRecent(readRecent(), slug)));
  } catch (error) {
    /* private mode: the page works without a history */
  }
}

// --- rendering one job ------------------------------------------------------

function renderNarrative(occupation, words, skills) {
  const narrative = occupation.n || {};
  renderStory(narrative, words);
  renderTasks(narrative, words, skills);
  renderWeek(narrative, words);
}

function showJob(occupation, route) {
  const words = wordingFor(route.framing, state.scheme);
  const context = linkContext(route, occupation);
  const skills = occupationSkills(state.data, occupation);
  state.points = scatterPoints(skills, state.scheme);
  document.title = `${capitalise(occupation.t)} — AI-ISCO`;
  byId('job-title').textContent = occupation.t;
  renderMeta(occupation);
  renderTools(route, words);
  setState('job');
  renderScores(occupation, words);
  renderAdvice(occupation, words);
  renderScatter(occupation, words);
  renderSkillLists(skills, words);
  renderCards(occupation, words);
  renderNarrative(occupation, words, skills);
  renderPaths(occupation, context, words);
  renderLearn(occupation, context, words);
  rememberRecent(occupation.s, route.framing);
}

function finishRender(moveFocus) {
  if (!moveFocus) return;
  window.scrollTo({ top: 0, behavior: 'auto' });
  byId('job-title').focus();
}

// --- loading one job --------------------------------------------------------

/**
 * What the page says while a job's own file is on its way.
 *
 * Two stages, because one honest sentence cannot cover both cases: under a
 * moment it says nothing at all, past 200 ms it says what it is doing, and past
 * three seconds it stops pretending the wait is normal and offers a way out.
 * The old copy promised "a few seconds" in front of a 73-second download.
 */
function setLoading(...nodes) {
  const box = byId('job-loading');
  box.replaceChildren(...nodes);
  box.hidden = false;
}

function loadingSlowly() {
  setLoading(
    document.createTextNode('Still loading this job. Its file is a small one, so a slow '
      + 'connection is the likely cause. '),
    link('groups.html', 'Browse sectors instead →'),
  );
}

function openJob(route, file, moveFocus) {
  const wanted = route.slug;
  state.pending = wanted;
  renderBack(route, { c: state.units[wanted] });
  const pending = loadJSON(file);
  const stage = (show) => { if (state.pending === wanted) show(); };
  whenSlow(pending, SLOW_MS, () => stage(() => setLoading('Loading this job…')));
  whenSlow(pending, VERY_SLOW_MS, () => stage(loadingSlowly));
  pending
    .then((data) => { if (state.pending === wanted) receiveJob(data, route, moveFocus); })
    .catch((error) => { if (state.pending === wanted) showError(error); });
}

function receiveJob(data, route, moveFocus) {
  byId('job-loading').hidden = true;
  state.data = data;
  state.scheme = schemeOfSet(state.stats, data);
  state.byslug = occupationIndex(data);
  const occupation = findOccupation(data, route.slug);
  renderBack(route, occupation);
  if (occupation) showJob(occupation, route);
  else showNotFound(route.slug);
  finishRender(moveFocus);
}

function render(moveFocus) {
  if (!state.units) return;
  const route = readRoute();
  const file = route.slug ? unitFile(state.units, route.slug) : null;
  if (file) {
    openJob(route, file, moveFocus);
    return;
  }
  state.pending = null;
  byId('job-loading').hidden = true;
  renderBack(route, null);
  if (route.slug) showNotFound(route.slug);
  else showPrompt();
  finishRender(moveFocus);
}

// --- the compact finder -----------------------------------------------------

function renderOption(result, node) {
  const code = el('span', 'code', result.row.c);
  const title = el('span', null, capitalise(result.row.t));
  node.append(code, title);
  if (result.alt) node.appendChild(el('span', 'also', `also matches "${result.alt}"`));
}

/**
 * The search index, fetched once and only when something needs it.
 *
 * It is the biggest file the page can ask for, and a visitor who came to read
 * one job never types in the finder. So the combobox resolves it on the first
 * keystroke — `createCombobox` takes a promise and drops stale answers — and
 * focus starts the fetch a moment earlier so the first keystroke feels instant.
 */
function ensureIndex() {
  if (!state.indexPromise) {
    state.indexPromise = loadJSON('search_index').catch(() => null);
    state.indexPromise.then((index) => { state.index = index; });
  }
  return state.indexPromise;
}

function findResults(query) {
  return ensureIndex().then((index) => (index ? rankOccupations(index, query, 8) : []));
}

function setupFinder() {
  const input = byId('job-find');
  input.addEventListener('focus', ensureIndex, { once: true });
  createCombobox({
    input,
    listbox: byId('job-find-list'),
    status: byId('job-find-status'),
    getResults: findResults,
    renderOption,
    onSelect: (result) => {
      input.value = '';
      go(jobHref(result.row.s, { for: framingParam(readRoute().framing) }));
    },
  });
}

// --- boot -------------------------------------------------------------------

function receive([units, groups, stats]) {
  byId('job-loading').hidden = true;
  state.units = units;
  state.groups = groups || {};
  state.stats = stats;
  state.scheme = schemeOfSet(stats, null);
  setupFinder();
  render(false);
}

/**
 * The slug-to-unit map. One file for both score sets, so it is fetched
 * directly: the shared loader would append the active set's suffix to it.
 */
function loadUnits() {
  return fetch(UNITS_FILE).then((response) => {
    if (!response.ok) {
      throw new LoadError('jobs/units', `The server answered ${response.status}.`);
    }
    return response.json();
  });
}

function start() {
  byId('job-loading').hidden = true;
  const optional = (name) => loadJSON(name).catch(() => null);
  const everything = Promise.all([loadUnits(), optional('groups'), optional('stats')]);
  whenSlow(everything, SLOW_MS, () => setLoading('Loading the job list…'))
    .then(receive)
    .catch(showError);
}

function bind() {
  if (state.started) return;
  state.started = true;
  document.addEventListener('click', onDocumentClick);
  window.addEventListener('popstate', () => render(false));
  window.addEventListener('hashchange', () => render(false));
  window.addEventListener('resize', drawScatter);
  byId('job-copy').addEventListener('click', copyLink);
  byId('scatter-toggle').addEventListener('click', toggleScatterTable);
  byId('scatter').addEventListener('pointermove', onScatterMove);
  byId('scatter').addEventListener('pointerleave', () => { byId('scatter-tip').hidden = true; });
}

// No nav item owns this page: it is reached from all of them.
renderChrome();
bind();
start();
