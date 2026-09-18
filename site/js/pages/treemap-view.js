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
export function createTreemapView({ canvas, caption, live, onActivate, onUp }) {
  const context = canvas.getContext('2d');
  const state = { tiles: [], rects: [], mode: 'quadrant', focus: -1, hover: -1, focused: false };

  function tileAt(index) {
    return index >= 0 ? state.tiles[index] : null;
  }

  function paint() {
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    context.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    context.clearRect(0, 0, width, height);
    state.rects.forEach((rect, index) => {
      const tile = state.tiles[rect.index];
      const colours = tileColour(tile, state.mode);
      const box = inset(rect);
      context.fillStyle = colours.fill;
      context.fillRect(box.x, box.y, box.width, box.height);
      drawLabel(context, box, tile, colours);
      if (index === state.hover) drawHover(context, box);
      if (index === state.focus && state.focused) drawRing(context, box);
    });
  }

  function drawHover(ctx, box) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x + 1, box.y + 1, box.width - 2, box.height - 2);
  }

  function relayout() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(200, canvas.parentElement.clientWidth - 2);
    const height = canvasHeight(width);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const box = { x: PADDING, y: PADDING, width: width - PADDING * 2, height: height - PADDING * 2 };
    state.rects = layoutTreemap(state.tiles, box);
    paint();
  }

  function say(index) {
    const tile = tileAt(state.rects[index] ? state.rects[index].index : -1);
    if (live) live.textContent = describeTile(tile);
    if (caption) caption.textContent = describeTile(tile);
  }

  function setFocus(index) {
    state.focus = index;
    paint();
    if (index >= 0) say(index);
  }

  function activate(index) {
    const rect = state.rects[index];
    if (rect) onActivate(state.tiles[rect.index]);
  }

  function onKeyDown(event) {
    if (STEPS[event.key] && state.rects.length) {
      event.preventDefault();
      setFocus(neighbour(state.rects, Math.max(0, state.focus), event.key));
    } else if (event.key === 'Enter' && state.focus >= 0) {
      event.preventDefault();
      activate(state.focus);
    } else if (event.key === 'Escape' || event.key === 'Backspace') {
      event.preventDefault();
      onUp();
    }
  }

  function onPointerMove(event) {
    const bounds = canvas.getBoundingClientRect();
    const index = hitIndex(state.rects, event.clientX - bounds.left, event.clientY - bounds.top);
    if (index === state.hover) return;
    state.hover = index;
    if (index >= 0 && caption) caption.textContent = describeTile(tileAt(state.rects[index].index));
    paint();
  }

  function onPointerLeave() {
    state.hover = -1;
    paint();
  }

  function onClick(event) {
    const bounds = canvas.getBoundingClientRect();
    const index = hitIndex(state.rects, event.clientX - bounds.left, event.clientY - bounds.top);
    if (index >= 0) activate(index);
  }

  function onFocus() {
    state.focused = true;
    setFocus(state.focus >= 0 ? state.focus : 0);
  }

  function onBlur() {
    state.focused = false;
    paint();
  }

  const listeners = [
    ['keydown', onKeyDown], ['pointermove', onPointerMove], ['pointerleave', onPointerLeave],
    ['click', onClick], ['focus', onFocus], ['blur', onBlur],
  ];
  for (const [type, handler] of listeners) canvas.addEventListener(type, handler);

  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(relayout) : null;
  if (observer) observer.observe(canvas.parentElement);

  return {
    setTiles(tiles) {
      state.tiles = tiles || [];
      state.focus = -1;
      state.hover = -1;
      relayout();
    },
    setMode(mode) {
      state.mode = mode;
      paint();
    },
    redraw: relayout,
    destroy() {
      for (const [type, handler] of listeners) canvas.removeEventListener(type, handler);
      if (observer) observer.disconnect();
    },
  };
}
