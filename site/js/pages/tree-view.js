// The tree widget of tree.html: the WAI-ARIA tree pattern, and nothing else.
//
// It draws only the rows tree-model.js says are on screen, so opening a branch
// builds that branch and no more. Every decision about *which* rows those are
// belongs to the model; this file turns them into nodes and into key presses.
//
// The row is the treeitem, as the tree pattern requires: one tab stop for the
// whole tree, arrow keys to move, and no focusable element inside a row. The
// real links out of this page live in the detail pane beside it.

import { formatCount, formatScore } from '../format.js';
import { QUADRANT_NAMES } from '../quadrant.js';
import {
  TYPE_AHEAD_MS, horizontalMove, jobSummary, mixSegments, mixText, moveIndex, rowIndexOf,
  typeAheadIndex,
} from './tree-model.js';

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

function twisty(kind) {
  return el('span', { class: `tree-twisty tree-twisty--${kind}`, 'aria-hidden': 'true' });
}

function mixBar(segments) {
  const fills = segments.map((segment) => el('span', {
    class: 'tree-mix-fill',
    'data-quadrant': segment.code,
    style: `width: ${(segment.share * 100).toFixed(2)}%`,
  }));
  return el('span', { class: 'tree-mix', 'aria-hidden': 'true' }, fills);
}

// A row is two blocks — the name, then the figures — so that on a narrow screen
// the figures wrap as one piece instead of scattering across two lines.
function row(main, meta, spoken) {
  return el('span', { class: 'tree-row' }, [
    el('span', { class: 'tree-main' }, main),
    el('span', { class: 'tree-meta' }, meta),
    el('span', { class: 'visually-hidden', text: spoken }),
  ]);
}

function groupRow(node) {
  const segments = mixSegments(node.counts);
  return row(
    [twisty('group'), el('span', { class: 'tree-label', text: node.label })],
    [
      el('span', { class: 'tree-count numeric', text: `${formatCount(node.n)} jobs` }),
      mixBar(segments),
    ],
    mixText(segments),
  );
}

function jobRow(node) {
  const name = QUADRANT_NAMES[node.q] || 'Not scored';
  return row(
    [
      twisty('leaf'),
      el('span', { class: 'tree-dot', 'data-quadrant': node.q || '', 'aria-hidden': 'true' }),
      el('span', { class: 'tree-label', text: node.label }),
    ],
    [
      el('span', { class: 'tree-quad', 'data-quadrant': node.q || '', text: name }),
      el('span', {
        class: 'tree-scores numeric',
        'aria-hidden': 'true',
        text: `${formatScore(node.a)} / ${formatScore(node.m)}`,
      }),
    ],
    jobSummary(node),
  );
}

function treeItem(row) {
  const item = el('li', {
    role: 'treeitem',
    class: `tree-item tree-item--${row.kind}`,
    'data-id': row.id,
    'aria-level': String(row.level),
    'aria-posinset': String(row.posinset),
    'aria-setsize': String(row.setsize),
    'aria-selected': String(Boolean(row.selected)),
    tabindex: '-1',
  }, [row.kind === 'group' ? groupRow(row) : jobRow(row)]);
  if (row.kind === 'group') item.setAttribute('aria-expanded', String(row.expanded));
  return item;
}

function renderRows(rows, root) {
  const items = [];
  root.replaceChildren();
  const stack = [root];
  for (const row of rows) {
    stack.length = row.level;
    const item = treeItem(row);
    items.push(item);
    stack[row.level - 1].append(item);
    if (row.expanded) stack.push(item.appendChild(el('ul', { role: 'group' })));
  }
  return items;
}

function isTypeAhead(event) {
  return event.key.length === 1 && event.key !== ' '
    && !event.ctrlKey && !event.metaKey && !event.altKey;
}

/**
 * The tree. Owns focus and key handling; the page owns what is expanded and
 * what is selected, and hands in a new row list whenever that changes.
 */
export class TreeView {
  /**
   * @param {HTMLElement} root the element carrying role="tree"
   * @param {{onActivate: Function, onToggle: Function}} callbacks
   *   `onActivate(row)` selects; `onToggle(key, open)` opens or closes a group.
   */
  constructor(root, callbacks) {
    this.root = root;
    this.callbacks = callbacks;
    this.rows = [];
    this.items = [];
    this.typed = { text: '', at: 0 };
    this.root.addEventListener('click', (event) => this.onClick(event));
    this.root.addEventListener('keydown', (event) => this.onKeyDown(event));
  }

  /** Draw a new set of visible rows and put the single tab stop back. */
  setRows(rows) {
    this.rows = rows || [];
    this.items = renderRows(this.rows, this.root);
    this.updateTabStop();
  }

  indexOf(id) {
    return rowIndexOf(this.rows, id);
  }

  updateTabStop() {
    const selected = this.rows.findIndex((row) => row.selected);
    const stop = selected === -1 ? 0 : selected;
    this.items.forEach((item, at) => { item.tabIndex = at === stop ? 0 : -1; });
  }

  /** Scroll a row into view, and optionally take the focus with it. */
  reveal(id, { focus = false } = {}) {
    const at = this.indexOf(id);
    if (at === -1) return;
    this.items[at].scrollIntoView({ block: 'nearest' });
    if (focus) this.focusAt(at);
  }

  focusAt(index) {
    const item = this.items[index];
    if (!item) return;
    this.items.forEach((node, at) => { node.tabIndex = at === index ? 0 : -1; });
    item.focus();
    item.scrollIntoView({ block: 'nearest' });
  }

  focusIndex() {
    const item = document.activeElement && document.activeElement.closest('[role="treeitem"]');
    return item ? this.items.indexOf(item) : -1;
  }

  onClick(event) {
    const item = event.target.closest('[role="treeitem"]');
    const at = item ? this.items.indexOf(item) : -1;
    if (at === -1) return;
    this.focusAt(at);
    this.callbacks.onActivate(this.rows[at]);
  }

  onKeyDown(event) {
    const at = this.focusIndex();
    if (at === -1) return;
    if (this.handleMove(event, at) || this.handleHorizontal(event, at)) return;
    if (this.handleActivate(event, at)) return;
    this.handleTypeAhead(event, at);
  }

  handleMove(event, at) {
    const next = moveIndex(this.rows, at, event.key);
    if (next === -1) return false;
    event.preventDefault();
    this.focusAt(next);
    return true;
  }

  handleHorizontal(event, at) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return false;
    event.preventDefault();
    const move = horizontalMove(this.rows, at, event.key);
    if (!move) return true;
    if (move.action === 'move') this.focusAt(move.index);
    else this.callbacks.onToggle(move.key, move.action === 'expand');
    return true;
  }

  handleActivate(event, at) {
    if (event.key !== 'Enter' && event.key !== ' ') return false;
    event.preventDefault();
    this.callbacks.onActivate(this.rows[at]);
    return true;
  }

  handleTypeAhead(event, at) {
    if (!isTypeAhead(event)) return;
    const now = Date.now();
    this.typed.text = now - this.typed.at > TYPE_AHEAD_MS ? event.key : this.typed.text + event.key;
    this.typed.at = now;
    const next = typeAheadIndex(this.rows, at, this.typed.text);
    if (next === -1) return;
    event.preventDefault();
    this.focusAt(next);
  }
}
