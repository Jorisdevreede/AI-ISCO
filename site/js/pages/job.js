// job.html — the one canonical job page. DOM wiring only: every decision this
// file makes is taken in ./job-model.js or ./framing.js, which are pure and
// unit-tested.
//
// What the page owes the visitor, and where each promise is kept:
//   - a real back link, always            renderBack / backTarget
//   - the quadrant, hedged                renderQuadrantBadge from ../badge.js
//   - a chart with a text alternative     renderScatter + renderScatterTable
//   - "your job" or neutral wording       ./framing.js, `&for=other` in the hash
//   - a next step, always                 renderAdvice
//   - Back that walks the real trail      history.pushState + popstate/hashchange

import { renderChrome } from '../chrome.js';
import { LoadError, loadJSON, whenSlow } from '../data.js';
import { createCombobox } from '../combobox.js';
import { rankOccupations, nearestTitles } from '../search.js';
import { renderQuadrantBadge } from '../badge.js';
import { THRESHOLD } from '../quadrant.js';
import { NOT_SCORED, formatCount, formatPercent, formatScore, isScored } from '../format.js';
import { groupHref, jobHref, parseHash, skillHref, withScorer } from '../urlstate.js';
import {
  MOVE_MARGIN, RECENT_KEY, addRecent, backTarget, essentialSkills, evolutionPaths,
  findOccupation, gapSkills, occupationIndex, occupationSkills, scatterPoints,
  scatterSummary, splitSkills, unitGroupKey,
} from './job-model.js';
import {
  framingParam, isNeutral, normaliseFraming, otherFraming, recordsRecent, wordingFor,
} from './framing.js';

const state = {
  data: null, groups: null, index: null, byslug: new Map(), points: [], started: false,
};

const NEUTRAL_NOTE = 'Neutral wording: this page is written about the job, not to the '
  + 'person doing it.';
const SIDEWAYS_NOTE = 'These share skills with this job, but the two scores do not make '
  + 'them a step up. Each one says what actually differs.';
const ADVICE_FALLBACK = 'Start with the skills AI could amplify, further down this page: '
  + 'they are the parts of this job that get more valuable, not less. The skills nearby '
  + 'jobs need are listed at the end.';

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
  span.appendChild(el('span', 'visually-hidden', ` ${name}, out of 10`));
  return span;
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
  document.title = 'Job details — AI-ISCO';
  byId('job-title').textContent = 'Pick a job to see how AI could change it';
  byId('job-meta').replaceChildren();
  byId('job-intro').replaceChildren(
    el('p', null, 'This page shows one occupation at a time: its two scores, the skills '
      + 'behind them, what changes in the work, and where someone could go next.'),
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
  const suggestions = suggestionList(spoken);
  if (suggestions) {
    intro.append(el('p', 'small muted', 'The closest titles we do have:'), suggestions);
  }
  intro.appendChild(chipRow(link('groups.html', 'Browse sectors →', 'chip')));
  setState('intro');
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
  meta.replaceChildren();
  metaLinks(occupation).forEach((anchor, index) => {
    if (index) meta.appendChild(el('span', 'job-meta-sep', '·'));
    meta.appendChild(anchor);
  });
  meta.appendChild(renderQuadrantBadge(
    { t: occupation.t, a: occupation.ar, m: occupation.ap, q: occupation.q },
    { search: window.location.search },
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

// --- the two scores ---------------------------------------------------------

const SCORE_CARDS = [
  {
    key: 'ar', modifier: 'auto', label: 'Automation risk',
    hint: 'How much of the work a machine could take over, read off the skills this job '
      + 'is built from.',
  },
  {
    key: 'ap', modifier: 'amp', label: 'Amplification',
    hint: 'How much more someone in this job could get done with AI helping, read off the '
      + 'same skills.',
  },
];

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
    el('p', 'score-label', `${spec.label}, out of 10`),
    el('p', scored ? 'score-value numeric' : 'score-value is-missing', formatScore(value)),
  );
  if (scored) card.appendChild(scoreBar(value));
  card.appendChild(el('p', 'score-hint', spec.hint));
  return card;
}

function exposureSentence(occupation) {
  if (!isScored(occupation.e)) return `AI exposure: ${NOT_SCORED}.`;
  return `AI exposure ${formatScore(occupation.e)} out of 10: the two scores multiplied, `
    + 'so it measures how much AI touches this job at all. It is not a verdict, and '
    + 'nothing on this page is sorted or coloured by it.';
}

function renderScores(occupation, words) {
  byId('scores-heading').textContent = words.scoresHeading;
  byId('scores-intro').textContent = words.scoresIntro;
  byId('score-list').replaceChildren(
    ...SCORE_CARDS.map((spec) => scoreCard(spec, occupation)),
  );
  byId('exposure-line').textContent = exposureSentence(occupation);
}

function adviceText(occupation) {
  const narrative = occupation.n || {};
  return narrative.adv || ADVICE_FALLBACK;
}

function renderAdvice(occupation, words) {
  const card = el('section', 'card card--advice');
  card.append(
    el('h2', null, words.adviceHeading),
    el('p', null, adviceText(occupation)),
    el('p', 'small muted', 'Written by the same model pass as the story below.'),
  );
  const exposed = isScored(occupation.ar) && occupation.ar >= THRESHOLD;
  byId('advice-top').replaceChildren(...(exposed ? [card] : []));
  byId('advice-bottom').replaceChildren(...(exposed ? [] : [card]));
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

function drawThreshold(ctx, box) {
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

function drawAxisTitles(ctx, box) {
  ctx.save();
  ctx.fillStyle = colours().text;
  ctx.font = '500 11px -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Automation risk, 1 to 10', box.x + box.w / 2, box.y + box.h + 24);
  ctx.translate(12, box.y + box.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText('Amplification, 1 to 10', 0, 0);
  ctx.restore();
}

function drawDot(ctx, box, point) {
  const colour = colours()[point.q] || colours().text;
  ctx.beginPath();
  ctx.arc(toX(box, point.auto), toY(box, point.amp), point.essential ? 6 : 4.5, 0, Math.PI * 2);
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

function showTip(point, x, y) {
  const tip = byId('scatter-tip');
  tip.replaceChildren(
    el('strong', null, point.title),
    el('span', null, `Automation risk ${formatScore(point.auto)}, amplification `
      + `${formatScore(point.amp)}, out of 10 · `
      + `${point.essential ? 'essential' : 'optional'}`),
  );
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

function skillRow(point) {
  const row = el('tr');
  const name = el('td');
  name.appendChild(link(skillHref(point.id), point.title));
  row.append(
    name,
    el('td', 'numeric', formatScore(point.auto)),
    el('td', 'numeric', formatScore(point.amp)),
    el('td', null, point.essential ? 'Essential' : 'Optional'),
  );
  return row;
}

function renderScatterTable(occupation) {
  byId('scatter-caption').textContent = `Every skill in ${occupation.t} that carries both `
    + `scores: ${plural(state.points.length, 'skill')}, one for each dot in the plot above.`;
  byId('scatter-rows').replaceChildren(...state.points.map(skillRow));
}

function renderScatter(occupation, words) {
  byId('scatter-heading').textContent = words.scatterHeading;
  byId('scatter-intro').textContent = words.scatterIntro;
  byId('scatter-threshold').textContent = `Dashed lines: the cut-off of ${THRESHOLD}`;
  byId('scatter').setAttribute('aria-label', scatterSummary(occupation.t, state.points));
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

// --- skill lists and rationale cards ---------------------------------------

function skillItem(skill, primary) {
  const item = el('li');
  item.appendChild(link(skillHref(skill.id), skill.title));
  const leading = primary === 'auto' ? skill.auto : skill.amp;
  const trailing = primary === 'auto' ? skill.amp : skill.auto;
  const names = primary === 'auto'
    ? ['automation risk', 'amplification']
    : ['amplification', 'automation risk'];
  item.append(
    scoreSpan(leading, names[0], `sk-score sk-score--${primary}`),
    scoreSpan(trailing, names[1], 'sk-score sk-score--muted'),
  );
  return item;
}

function fillSkillList(listId, countId, skills, spec) {
  byId(countId).textContent = skills.length
    ? `${formatCount(skills.length)} of the ${plural(spec.total, 'skill')} in this job score `
      + `${THRESHOLD} or more for ${spec.noun}. Both scores are shown, ${spec.noun} first.`
    : `No skill in this job reaches ${THRESHOLD} for ${spec.noun}.`;
  byId(listId).replaceChildren(...skills.map((skill) => skillItem(skill, spec.primary)));
}

function renderSkillLists(skills, words) {
  const { depreciating, appreciating } = splitSkills(skills);
  byId('dep-heading').textContent = words.depreciatingHeading;
  byId('app-heading').textContent = words.appreciatingHeading;
  fillSkillList('dep-list', 'dep-count', depreciating,
    { total: skills.length, noun: 'automation risk', primary: 'auto' });
  fillSkillList('app-list', 'app-count', appreciating,
    { total: skills.length, noun: 'amplification', primary: 'amp' });
}

function skillCard(skill) {
  const card = el('details', 'skill-card');
  const summary = el('summary');
  const row = el('span', 'sc-row');
  row.append(
    el('span', 'sc-name', skill.title),
    scoreSpan(skill.auto, 'automation risk', 'sk-score sk-score--auto'),
    scoreSpan(skill.amp, 'amplification', 'sk-score sk-score--amp'),
  );
  summary.appendChild(row);
  const more = el('p');
  more.appendChild(link(skillHref(skill.id), `Look up ${skill.title} →`));
  const body = el('div', 'skill-card-body');
  body.append(el('p', null, skill.rationale || 'The model wrote no rationale for this skill.'), more);
  card.append(summary, body);
  return card;
}

function renderCards(occupation, words) {
  byId('cards-heading').textContent = words.cardsHeading;
  const skills = essentialSkills(state.data, occupation);
  const list = byId('cards-list');
  if (!skills.length) {
    list.replaceChildren(el('p', 'muted', 'This job lists no essential skills.'));
    return;
  }
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

function renderStory(narrative, words) {
  byId('sec-story').hidden = !narrative.story;
  if (!narrative.story) return;
  byId('story-heading').textContent = words.storyHeading;
  const note = byId('story-note');
  note.hidden = !words.storyNote;
  note.textContent = words.storyNote;
  byId('story-facts').replaceChildren(...storyFacts(narrative));
  byId('story-text').replaceChildren(...paragraphs(narrative.story));
}

const TASK_LISTS = [
  { key: 'auto', heading: 'What AI takes on' },
  { key: 'amp', heading: 'What AI amplifies' },
  { key: 'tools', heading: 'Tools already doing it' },
];

function taskList(spec, narrative) {
  const items = narrative[spec.key] || [];
  if (!items.length) return null;
  const block = el('div', 'ai-list');
  const list = el('ul');
  for (const item of items) list.appendChild(el('li', null, item));
  block.append(el('h3', null, spec.heading), list);
  return block;
}

function renderTasks(narrative, words) {
  const blocks = TASK_LISTS.map((spec) => taskList(spec, narrative)).filter(Boolean);
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
  const usable = Boolean(week.before && week.after);
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
  return `Automation risk ${formatScore(card.auto)}, amplification ${formatScore(card.amp)}, `
    + `out of 10 · ${overlap} · ${plural(card.gapCount, 'new skill')}`;
}

function pathCard(card, context) {
  const anchor = jobAnchor(card.slug, context, '');
  anchor.className = 'path-card';
  const tag = el('span', 'path-tag', card.label);
  tag.dataset.kind = card.kind;
  anchor.append(el('span', 'path-title', card.title), tag, el('p', 'path-stats', pathStats(card)));
  return anchor;
}

function pathsIntro(words, found, total) {
  if (!total) return 'The data lists no occupation next to this one.';
  if (!found) {
    return `None of the ${plural(total, 'nearby job')} is clearly less exposed or more `
      + `amplified than this one: every one of them sits within ${MOVE_MARGIN} of it on `
      + 'both scores. They are listed below as what they are.';
  }
  return words.pathsIntro;
}

function renderPaths(occupation, context, words) {
  const { paths, others, total } = evolutionPaths(occupation, state.byslug);
  byId('paths-heading').textContent = words.pathsHeading;
  byId('paths-intro').textContent = pathsIntro(words, paths.length, total);
  byId('paths-grid').replaceChildren(...paths.map((card) => pathCard(card, context)));
  byId('sideways-heading').textContent = others.length ? words.sidewaysHeading : '';
  byId('sideways-note').textContent = others.length ? SIDEWAYS_NOTE : '';
  byId('sideways-grid').replaceChildren(...others.map((card) => pathCard(card, context)));
}

function learnItem(item, context) {
  const row = el('li');
  row.append(
    scoreSpan(item.amp, 'amplification', 'learn-amp numeric'),
    link(skillHref(item.id), item.title),
  );
  const from = el('p', 'learn-from');
  from.append(document.createTextNode('Needed by '), jobAnchor(item.fromSlug, context, item.fromTitle));
  row.appendChild(from);
  return row;
}

function renderLearn(occupation, context, words) {
  byId('learn-heading').textContent = words.learnHeading;
  const items = gapSkills(state.data, occupation);
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

function showJob(occupation, route) {
  const words = wordingFor(route.framing);
  const context = linkContext(route, occupation);
  const skills = occupationSkills(state.data, occupation);
  state.points = scatterPoints(skills);
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
  renderStory(occupation.n || {}, words);
  renderTasks(occupation.n || {}, words);
  renderWeek(occupation.n || {}, words);
  renderPaths(occupation, context, words);
  renderLearn(occupation, context, words);
  rememberRecent(occupation.s, route.framing);
}

function render(moveFocus) {
  if (!state.data) return;
  const route = readRoute();
  const occupation = route.slug ? findOccupation(state.data, route.slug) : null;
  renderBack(route, occupation);
  if (!route.slug) showPrompt();
  else if (!occupation) showNotFound(route.slug);
  else showJob(occupation, route);
  if (moveFocus) {
    window.scrollTo({ top: 0, behavior: 'auto' });
    byId('job-title').focus();
  }
}

// --- the compact finder -----------------------------------------------------

function renderOption(result, node) {
  const code = el('span', 'code', result.row.c);
  const title = el('span', null, capitalise(result.row.t));
  node.append(code, title);
  if (result.alt) node.appendChild(el('span', 'also', `also matches "${result.alt}"`));
}

function setupFinder(index) {
  const input = byId('job-find');
  createCombobox({
    input,
    listbox: byId('job-find-list'),
    status: byId('job-find-status'),
    getResults: (query) => rankOccupations(index, query, 8),
    renderOption,
    onSelect: (result) => {
      input.value = '';
      go(jobHref(result.row.s, { for: framingParam(readRoute().framing) }));
    },
  });
}

// --- boot -------------------------------------------------------------------

function receive([data, groups, index]) {
  byId('job-loading').hidden = true;
  state.data = data;
  state.groups = groups || {};
  state.index = index;
  state.byslug = occupationIndex(data);
  if (index) setupFinder(index);
  else byId('job-find').closest('.job-finder').hidden = true;
  render(false);
}

function start() {
  byId('job-loading').hidden = true;
  const optional = (name) => loadJSON(name).catch(() => null);
  const everything = Promise.all([
    loadJSON('portfolio_data'), optional('groups'), optional('search_index'),
  ]);
  whenSlow(everything, 200, () => { byId('job-loading').hidden = false; })
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
