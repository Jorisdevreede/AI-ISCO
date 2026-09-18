// The class badge and its popover. Thin DOM module.
//
// The badge is a real <button>, so it is reachable by keyboard and announced as
// pressable. Its popover carries the only uncertainty wording the site is
// allowed to use, which quadrant.js owns for the four boxes and scheme.js for
// the seven types.
//
// Two entry points, one body: renderQuadrantBadge is the four-box badge exactly
// as it has always rendered, renderTypeBadge follows the active scheme.

import { QUADRANT_NAMES } from './quadrant.js';
import { QUADRANTS, explain, isNear, schemeOfCode, typeShortLabel } from './scheme.js';
import { withScorer } from './urlstate.js';

let badgeCount = 0;

/** Where "How we scored this" points. */
export const METHOD_URL = 'method.html';

function schemeFor(job, scheme) {
  return scheme || schemeOfCode(job?.q) || QUADRANTS;
}

function buildPopover(id, explanation, search) {
  const popover = document.createElement('div');
  popover.id = id;
  popover.className = 'badge-popover';
  popover.hidden = true;
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-label', explanation.heading);

  const heading = document.createElement('h3');
  heading.textContent = explanation.heading;
  popover.appendChild(heading);
  for (const sentence of explanation.sentences) {
    const paragraph = document.createElement('p');
    paragraph.textContent = sentence;
    popover.appendChild(paragraph);
  }
  const link = document.createElement('a');
  link.href = withScorer(METHOD_URL, search);
  link.textContent = 'How we scored this';
  popover.appendChild(link);
  return popover;
}

function labelButton(button, job, scheme) {
  if (scheme === QUADRANTS) {
    button.className = 'quadrant-badge';
    button.dataset.quadrant = job.q || '';
    return QUADRANT_NAMES[job.q] || 'Not scored';
  }
  button.className = 'quadrant-badge type-badge';
  button.dataset.type = job.q || '';
  return typeShortLabel(job.q, scheme);
}

function buildButton(job, popoverId, scheme) {
  const button = document.createElement('button');
  button.type = 'button';
  const label = labelButton(button, job, scheme);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', popoverId);

  const name = document.createElement('span');
  name.className = 'quadrant-badge-name';
  name.textContent = label;
  button.appendChild(name);

  if (isNear(job, scheme)) {
    const marker = document.createElement('span');
    marker.className = 'quadrant-badge-near';
    marker.textContent = 'near the line';
    button.appendChild(marker);
  }
  const hint = document.createElement('span');
  hint.className = 'quadrant-badge-hint';
  hint.setAttribute('aria-hidden', 'true');
  hint.textContent = 'i';
  button.appendChild(hint);
  return button;
}

// Open popovers, so ONE document listener can close them on an outside press
// however many badges a page renders.
const openBadges = new Map();
let outsideCloseBound = false;

function bindOutsideClose() {
  if (outsideCloseBound) return;
  outsideCloseBound = true;
  document.addEventListener('pointerdown', (event) => {
    openBadges.forEach((close, wrapper) => {
      if (!wrapper.contains(event.target)) close();
    });
  });
}

function wireBadge(wrapper, button, popover) {
  const setOpen = (open) => {
    popover.hidden = !open;
    button.setAttribute('aria-expanded', String(open));
    wrapper.classList.toggle('is-open', open);
    if (open) openBadges.set(wrapper, () => setOpen(false));
    else openBadges.delete(wrapper);
  };
  button.addEventListener('click', () => setOpen(popover.hidden));
  wrapper.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || popover.hidden) return;
    event.stopPropagation();
    setOpen(false);
    button.focus();
  });
  bindOutsideClose();
}

function renderBadge(job, scheme, search) {
  badgeCount += 1;
  const query = search === undefined ? window.location.search : search;
  const popoverId = `quadrant-popover-${badgeCount}`;
  const wrapper = document.createElement('span');
  wrapper.className = 'quadrant-badge-wrap';
  const popover = buildPopover(popoverId, explain(job, scheme), query);
  const button = buildButton(job, popoverId, scheme);
  wireBadge(wrapper, button, popover);
  wrapper.append(button, popover);
  return wrapper;
}

/**
 * A quadrant badge with an explaining popover. Four boxes only: a job from a
 * shares set renders as "Not scored" here, which is what renderTypeBadge is for.
 *
 * @param {{t?: string, a: number, m: number, q?: string}} job
 * @param {{search?: string}} [options] `search` defaults to location.search and
 *   carries ?scorer= onto the method link
 * @returns {HTMLElement} a wrapper holding the button and its popover
 */
export function renderQuadrantBadge(job, { search } = {}) {
  return renderBadge(job, QUADRANTS, search);
}

/**
 * The badge of whichever scheme the active set uses: the quadrant badge under
 * `quadrants`, a type badge explained from the job's own shares under `shares`.
 * Same button, same popover, same keyboard behaviour.
 *
 * @param {{t?: string, q?: string, a?: number, m?: number, sh?: number[], nl?: boolean}} job
 * @param {{scheme?: string, search?: string}} [options] `scheme` may be left out
 *   when `job.q` carries a code, which resolves the scheme on its own
 * @returns {HTMLElement} a wrapper holding the button and its popover
 */
export function renderTypeBadge(job, { scheme, search } = {}) {
  return renderBadge(job, schemeFor(job, scheme), search);
}
