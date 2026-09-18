// The treemap canvas: drawing, hit-testing and keyboard navigation.
//
// The layout comes from treemap-layout.js and every string comes from
// groups-model.js, so this module only paints and listens. The canvas is a
// single focusable element with a drawn focus ring, which is how 3,000 tiles
// stay keyboard reachable without 3,000 tab stops; a polite live region says
// which tile the ring is on.

import { layoutTreemap } from '../treemap-layout.js';
import { describeTile, tileColour } from './groups-model.js';

const GAP = 2;
const PADDING = 8;
const MIN_LABEL = { width: 54, height: 26 };

const STEPS = {
  ArrowRight: { dx: 1, dy: 0, fallback: 1 },
  ArrowLeft: { dx: -1, dy: 0, fallback: -1 },
  ArrowDown: { dx: 0, dy: 1, fallback: 1 },
  ArrowUp: { dx: 0, dy: -1, fallback: -1 },
};

const centre = (rect) => ({ x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 });

const ratio = () => window.devicePixelRatio || 1;

function canvasHeight(width) {
  return Math.round(Math.min(560, Math.max(260, width * 0.62)));
}

function fontSize(rect, tile) {
  const cap = tile.kind === 'group' ? 19 : 14;
  return Math.max(10, Math.min(cap, rect.width / 8, rect.height / 2.6));
}

/** `text`, shortened with an ellipsis until it fits `room` pixels. */
function fitText(ctx, text, room) {
  if (ctx.measureText(text).width <= room) return text;
  let cut = text.length;
  while (cut > 1 && ctx.measureText(`${text.slice(0, cut)}…`).width > room) cut -= 1;
  return `${text.slice(0, cut).trimEnd()}…`;
}

function drawLabel(ctx, rect, tile, colours) {
  if (rect.width < MIN_LABEL.width || rect.height < MIN_LABEL.height) return;
  const size = fontSize(rect, tile);
  const room = rect.width - 14;
  ctx.save();
  ctx.fillStyle = colours.text;
  ctx.textBaseline = 'top';
  ctx.font = `${tile.kind === 'group' ? 700 : 500} ${size}px -apple-system, system-ui, sans-serif`;
  ctx.fillText(fitText(ctx, tile.label, room), rect.x + 7, rect.y + 5);
  if (rect.height > size * 2 + 16 && rect.width > 96) {
    ctx.font = `400 ${Math.max(10, size - 3)}px -apple-system, system-ui, sans-serif`;
    ctx.fillText(fitText(ctx, tile.detail, room), rect.x + 7, rect.y + size + 9);
  }
  ctx.restore();
}

function drawRing(ctx, rect) {
  ctx.strokeStyle = '#0a0a0f';
  ctx.lineWidth = 5;
  ctx.strokeRect(rect.x + 2.5, rect.y + 2.5, rect.width - 5, rect.height - 5);
  ctx.strokeStyle = '#e8b93b';
  ctx.lineWidth = 3;
  ctx.strokeRect(rect.x + 2.5, rect.y + 2.5, rect.width - 5, rect.height - 5);
}

function drawHover(ctx, box) {
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x + 1, box.y + 1, box.width - 2, box.height - 2);
}

function inset(rect) {
  return {
    x: rect.x + GAP / 2,
    y: rect.y + GAP / 2,
    width: Math.max(0, rect.width - GAP),
    height: Math.max(0, rect.height - GAP),
  };
}

function hitIndex(rects, x, y) {
  for (let index = rects.length - 1; index >= 0; index -= 1) {
    const rect = rects[index];
    if (x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height) {
      return index;
    }
  }
  return -1;
}

/** The tile nearest `from` in the direction of an arrow key. */
function geometricNeighbour(rects, from, step) {
  const origin = centre(rects[from]);
  let best = -1;
  let bestScore = Infinity;
  rects.forEach((rect, index) => {
    const point = centre(rect);
    const along = (point.x - origin.x) * step.dx + (point.y - origin.y) * step.dy;
    const across = Math.abs((point.x - origin.x) * step.dy + (point.y - origin.y) * step.dx);
    const score = along + across * 2;
    if (index !== from && along > 1 && score < bestScore) {
      bestScore = score;
      best = index;
    }
  });
  return best;
}

function neighbour(rects, from, key) {
  const step = STEPS[key];
  const found = geometricNeighbour(rects, from, step);
  if (found >= 0) return found;
  const wrapped = from + step.fallback;
  return wrapped >= 0 && wrapped < rects.length ? wrapped : from;
}

/** One canvas painted as a treemap. `createTreemapView` is the way in. */
class TreemapView {
  constructor({ canvas, caption, live, onActivate, onUp }) {
    this.canvas = canvas;
    this.caption = caption;
    this.live = live;
    this.onActivate = onActivate;
    this.onUp = onUp;
    this.context = canvas.getContext('2d');
    this.tiles = [];
    this.rects = [];
    this.mode = 'quadrant';
    this.focus = -1;
    this.hover = -1;
    this.focused = false;
    this.handlers = this.listen();
    this.observer = this.observe();
  }

  listen() {
    const handlers = {
      keydown: (event) => this.onKeyDown(event),
      pointermove: (event) => this.onPointerMove(event),
      pointerleave: () => this.onPointerLeave(),
      click: (event) => this.onClick(event),
      focus: () => this.onFocus(),
      blur: () => this.onBlur(),
    };
    for (const [type, handler] of Object.entries(handlers)) {
      this.canvas.addEventListener(type, handler);
    }
    return handlers;
  }

  observe() {
    if (typeof ResizeObserver !== 'function') return null;
    const observer = new ResizeObserver(() => this.redraw());
    observer.observe(this.canvas.parentElement);
    return observer;
  }

  tileAt(index) {
    return index >= 0 ? this.tiles[index] : null;
  }

  /** The tile under a rect index, or null when there is no such rect. */
  tileOf(index) {
    const rect = this.rects[index];
    return this.tileAt(rect ? rect.index : -1);
  }

  paint() {
    const scale = ratio();
    this.context.setTransform(scale, 0, 0, scale, 0, 0);
    this.context.clearRect(0, 0, this.canvas.width / scale, this.canvas.height / scale);
    this.rects.forEach((rect, index) => this.paintTile(rect, index));
  }

  paintTile(rect, index) {
    const tile = this.tiles[rect.index];
    const colours = tileColour(tile, this.mode);
    const box = inset(rect);
    this.context.fillStyle = colours.fill;
    this.context.fillRect(box.x, box.y, box.width, box.height);
    drawLabel(this.context, box, tile, colours);
    if (index === this.hover) drawHover(this.context, box);
    if (index === this.focus && this.focused) drawRing(this.context, box);
  }

  resize(width, height) {
    const scale = ratio();
    this.canvas.width = Math.round(width * scale);
    this.canvas.height = Math.round(height * scale);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
  }

  say(index) {
    const text = describeTile(this.tileOf(index));
    if (this.live) this.live.textContent = text;
    if (this.caption) this.caption.textContent = text;
  }

  setFocus(index) {
    this.focus = index;
    this.paint();
    if (index >= 0) this.say(index);
  }

  activate(index) {
    const rect = this.rects[index];
    if (rect) this.onActivate(this.tiles[rect.index]);
  }

  /** The rect under a pointer event, or -1. */
  indexAt(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return hitIndex(this.rects, event.clientX - bounds.left, event.clientY - bounds.top);
  }

  onKeyDown(event) {
    if (STEPS[event.key] && this.rects.length) {
      event.preventDefault();
      this.setFocus(neighbour(this.rects, Math.max(0, this.focus), event.key));
    } else if (event.key === 'Enter' && this.focus >= 0) {
      event.preventDefault();
      this.activate(this.focus);
    } else if (event.key === 'Escape' || event.key === 'Backspace') {
      event.preventDefault();
      this.onUp();
    }
  }

  onPointerMove(event) {
    const index = this.indexAt(event);
    if (index === this.hover) return;
    this.hover = index;
    if (index >= 0 && this.caption) this.caption.textContent = describeTile(this.tileOf(index));
    this.paint();
  }

  onPointerLeave() {
    this.hover = -1;
    this.paint();
  }

  onClick(event) {
    const index = this.indexAt(event);
    if (index >= 0) this.activate(index);
  }

  onFocus() {
    this.focused = true;
    this.setFocus(this.focus >= 0 ? this.focus : 0);
  }

  onBlur() {
    this.focused = false;
    this.paint();
  }

  setTiles(tiles) {
    this.tiles = tiles || [];
    this.focus = -1;
    this.hover = -1;
    this.redraw();
  }

  setMode(mode) {
    this.mode = mode;
    this.paint();
  }

  /** Size the canvas to its parent, lay the tiles out again and repaint. */
  redraw() {
    const width = Math.max(200, this.canvas.parentElement.clientWidth - 2);
    const height = canvasHeight(width);
    this.resize(width, height);
    const box = { x: PADDING, y: PADDING, width: width - PADDING * 2, height: height - PADDING * 2 };
    this.rects = layoutTreemap(this.tiles, box);
    this.paint();
  }

  destroy() {
    for (const [type, handler] of Object.entries(this.handlers)) {
      this.canvas.removeEventListener(type, handler);
    }
    if (this.observer) this.observer.disconnect();
  }
}

/**
 * Wire a canvas up as a treemap.
 *
 * @param {Object} options
 * @param {HTMLCanvasElement} options.canvas focusable, role="img"
 * @param {HTMLElement} options.caption visible line under the canvas
 * @param {HTMLElement} options.live a polite live region
 * @param {(tile: Object) => void} options.onActivate click or Enter
 * @param {() => void} options.onUp Escape or Backspace
 * @returns {{setTiles: Function, setMode: Function, redraw: Function, destroy: Function}}
 */
export function createTreemapView(options) {
  const view = new TreemapView(options);
  return {
    setTiles: (tiles) => view.setTiles(tiles),
    setMode: (mode) => view.setMode(mode),
    redraw: () => view.redraw(),
    destroy: () => view.destroy(),
  };
}
