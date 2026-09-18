// The four-share stacked bar. Thin DOM module: the arithmetic lives in shares.js.
//
// Colour never carries the meaning on its own. The bar is one role="img" whose
// aria-label is the whole sentence, and beside it sits a legend in plain text
// with the same numbers, so the figure reads the same to a screen reader, to a
// visitor who cannot tell the segments apart by hue, and in print.

import { shareAriaLabel, shareSegments, shareSentence, sharePercents } from './shares.js';
import { skillClassName } from './scheme.js';

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildBar(shares, name) {
  const bar = element('div', 'shares-bar');
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', shareAriaLabel(shares, name));
  for (const segment of shareSegments(shares)) {
    const part = element('span', 'shares-seg');
    part.dataset.class = segment.code;
    part.style.width = segment.width;
    bar.appendChild(part);
  }
  return bar;
}

function legendItem(part) {
  const item = element('li');
  const key = element('span', 'shares-key', part.code);
  key.dataset.class = part.code;
  key.setAttribute('aria-hidden', 'true');
  item.append(key, element('span', 'shares-label', skillClassName(part.code)),
    element('span', 'shares-value numeric', `${part.percent}%`));
  return item;
}

function buildLegend(shares) {
  const legend = element('ul', 'shares-legend');
  for (const part of sharePercents(shares)) legend.appendChild(legendItem(part));
  return legend;
}

/**
 * The stacked bar with its legend.
 *
 * @param {number[]} shares `sh`: [substituted, assisted, mechanised, insulated]
 * @param {{name?: string, compact?: boolean}} [options]
 *   `name` is the job title, which goes into the accessible name; `compact`
 *   swaps the four-row legend for the one-line sentence, for a list row.
 * @returns {HTMLElement} '' shares render as a "Not scored" note, never an empty bar
 */
export function renderSharesBar(shares, { name, compact } = {}) {
  const wrapper = element('div', compact ? 'shares shares--compact' : 'shares');
  const sentence = shareSentence(shares);
  if (!sentence) {
    wrapper.appendChild(element('p', 'shares-empty muted small', 'Shares not scored'));
    return wrapper;
  }
  wrapper.appendChild(buildBar(shares, name));
  wrapper.appendChild(compact
    ? element('p', 'shares-line small', sentence)
    : buildLegend(shares));
  return wrapper;
}
