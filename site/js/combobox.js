// An accessible autocomplete, ARIA 1.2 combobox pattern. Thin DOM module.
//
// Used by the landing page, the skill page and the group picker. Keep ranking
// out of here: the caller supplies getResults and renderOption.

let comboboxCount = 0;

const KEYS = {
  ArrowDown: (combo) => combo.move(1),
  ArrowUp: (combo) => combo.move(-1),
  Home: (combo) => combo.moveTo(0),
  End: (combo) => combo.moveTo(combo.results.length - 1),
};

// Long enough for a tap on an option to land before the blur closes the list.
const BLUR_MS = 120;

function optionId(prefix, index) {
  return `${prefix}-option-${index}`;
}

// Where a move starts counting from. With nothing active, Down goes to the
// first option and Up to the last.
function startIndex(active, step) {
  if (active >= 0) return active;
  return step > 0 ? -1 : 0;
}

function describe(count, query) {
  if (!query) return '';
  if (count === 0) return `No results for ${query}.`;
  return `${count} result${count === 1 ? '' : 's'} available. `
    + 'Use the up and down arrow keys to review, Enter to open.';
}

function defaultRenderOption(result, node) {
  node.textContent = result?.row ? result.row.t : String(result);
}

function prepare(input, listbox, prefix) {
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-haspopup', 'listbox');
  input.setAttribute('autocomplete', 'off');
  if (!listbox.id) listbox.id = `${prefix}-listbox`;
  input.setAttribute('aria-controls', listbox.id);
  listbox.setAttribute('role', 'listbox');
  listbox.hidden = true;
}

/** One input wired to one listbox. `createCombobox` is the way in. */
class Combobox {
  constructor({
    input, listbox, status, getResults, renderOption = defaultRenderOption, onSelect,
  }) {
    comboboxCount += 1;
    this.input = input;
    this.listbox = listbox;
    this.status = status;
    this.getResults = getResults;
    this.renderOption = renderOption;
    this.onSelect = onSelect;
    this.prefix = input.id || `combobox-${comboboxCount}`;
    this.results = [];
    this.active = -1;
    this.token = 0;
    prepare(input, listbox, this.prefix);
    this.handlers = this.listen();
  }

  listen() {
    const handlers = {
      input: () => this.refresh(),
      keydown: (event) => this.onKeyDown(event),
      focus: () => this.onFocus(),
      blur: () => setTimeout(() => this.close(), BLUR_MS),
    };
    for (const [type, handler] of Object.entries(handlers)) {
      this.input.addEventListener(type, handler);
    }
    return handlers;
  }

  setActive(index) {
    this.active = index;
    for (const [i, node] of [...this.listbox.children].entries()) {
      const on = i === index;
      node.setAttribute('aria-selected', String(on));
      node.classList.toggle('is-active', on);
    }
    this.pointAt(index);
  }

  pointAt(index) {
    if (index < 0) {
      this.input.removeAttribute('aria-activedescendant');
      return;
    }
    this.input.setAttribute('aria-activedescendant', optionId(this.prefix, index));
    this.listbox.children[index].scrollIntoView({ block: 'nearest' });
  }

  close() {
    this.listbox.hidden = true;
    this.input.setAttribute('aria-expanded', 'false');
    this.setActive(-1);
  }

  open() {
    if (!this.results.length) {
      this.close();
      return;
    }
    this.listbox.hidden = false;
    this.input.setAttribute('aria-expanded', 'true');
  }

  choose(index) {
    const result = this.results[index];
    if (result === undefined) return;
    this.close();
    this.onSelect(result);
  }

  option(result, index) {
    const node = document.createElement('li');
    node.id = optionId(this.prefix, index);
    node.setAttribute('role', 'option');
    node.setAttribute('aria-selected', 'false');
    this.renderOption(result, node);
    node.addEventListener('pointerdown', (event) => {
      event.preventDefault(); // keep focus in the input
      this.choose(index);
    });
    return node;
  }

  paint(results, query) {
    this.results = results;
    this.listbox.replaceChildren();
    for (const [index, result] of results.entries()) {
      this.listbox.appendChild(this.option(result, index));
    }
    if (this.status) this.status.textContent = describe(results.length, query);
    this.setActive(-1);
    this.open();
  }

  clear() {
    this.results = [];
    this.listbox.replaceChildren();
    if (this.status) this.status.textContent = '';
    this.close();
  }

  // An emptied box short-circuits: it clears the list without asking the caller
  // for results at all. The landing page relies on that.
  refresh() {
    const query = this.input.value.trim();
    const token = (this.token += 1);
    if (!query) return this.clear();
    return Promise.resolve(this.getResults(query)).then((results) => {
      if (token === this.token) this.paint(results || [], query);
    });
  }

  move(step) {
    if (this.listbox.hidden) this.open();
    const count = this.results.length;
    if (!count) return;
    const from = startIndex(this.active, step);
    this.setActive((from + step + count) % count);
  }

  moveTo(index) {
    if (!this.results.length) return;
    this.open();
    this.setActive(Math.max(0, Math.min(index, this.results.length - 1)));
  }

  onKeyDown(event) {
    if (KEYS[event.key]) {
      event.preventDefault();
      KEYS[event.key](this);
      return;
    }
    if (event.key === 'Enter' && this.active >= 0) {
      event.preventDefault();
      this.choose(this.active);
    } else if (event.key === 'Escape') {
      this.close();
    }
  }

  onFocus() {
    if (this.results.length) this.open();
  }

  destroy() {
    for (const [type, handler] of Object.entries(this.handlers)) {
      this.input.removeEventListener(type, handler);
    }
    this.close();
    this.listbox.replaceChildren();
  }
}

/**
 * Wire an input and a list into a combobox.
 *
 * @param {Object} options
 * @param {HTMLInputElement} options.input the text field
 * @param {HTMLElement} options.listbox a <ul> that will hold the options
 * @param {HTMLElement} [options.status] a polite live region for result counts
 * @param {(query: string) => any[]|Promise<any[]>} options.getResults
 * @param {(result: any, node: HTMLElement) => void} [options.renderOption]
 * @param {(result: any) => void} options.onSelect called on Enter, click or tap
 * @returns {{open: Function, close: Function, refresh: Function, destroy: Function}}
 */
export function createCombobox(options) {
  const combo = new Combobox(options);
  return {
    open: () => combo.open(),
    close: () => combo.close(),
    refresh: () => combo.refresh(),
    destroy: () => combo.destroy(),
  };
}
