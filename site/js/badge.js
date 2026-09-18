// The quadrant badge and its popover. Thin DOM module.
//
// The badge is a real <button>, so it is reachable by keyboard and announced as
// pressable. Its popover carries the only uncertainty wording the site is
// allowed to use, which quadrant.js owns.

import { QUADRANT_NAMES, explainQuadrant, isNearLine } from './quadrant.js';
import { withScorer } from './urlstate.js';

let badgeCount = 0;

/** Where "How we scored this" points. */
export const METHOD_URL = 'method.html';

function buildPopover(id, job, search) {
  const explanation = explainQuadrant(job);
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

function buildButton(job, popoverId) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'quadrant-badge';
  button.dataset.quadrant = job.q || '';
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', popoverId);

  const name = document.createElement('span');
  name.className = 'quadrant-badge-name';
  name.textContent = QUADRANT_NAMES[job.q] || 'Not scored';
  button.appendChild(name);

  if (isNearLine(job.a, job.m)) {
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

/**
 * A quadrant badge with an explaining popover.
 *
 * @param {{t?: string, a: number, m: number, q?: string}} job
 * @param {{search?: string}} [options] `search` defaults to location.search and
 *   carries ?scorer= onto the method link
 * @returns {HTMLElement} a wrapper holding the button and its popover
 */
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

export function renderQuadrantBadge(job, { search } = {}) {
  badgeCount += 1;
  const query = search === undefined ? window.location.search : search;
  const popoverId = `quadrant-popover-${badgeCount}`;
  const wrapper = document.createElement('span');
  wrapper.className = 'quadrant-badge-wrap';
  const popover = buildPopover(popoverId, job, query);
  const button = buildButton(job, popoverId);
  wireBadge(wrapper, button, popover);
  wrapper.append(button, popover);
  return wrapper;
}
