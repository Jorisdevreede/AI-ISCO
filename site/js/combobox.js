// An accessible autocomplete, ARIA 1.2 combobox pattern. Thin DOM module.
//
// Used by the landing page, the skill page and the group picker. Keep ranking
// out of here: the caller supplies getResults and renderOption.

let comboboxCount = 0;

const KEYS = {
  ArrowDown: (state) => state.move(1),
  ArrowUp: (state) => state.move(-1),
  Home: (state) => state.moveTo(0),
  End: (state) => state.moveTo(state.results.length - 1),
};

function optionId(prefix, index) {
  return `${prefix}-option-${index}`;
}

function describe(count, query) {
  if (!query) return '';
  if (count === 0) return `No results for ${query}.`;
  return `${count} result${count === 1 ? '' : 's'} available. `
    + 'Use the up and down arrow keys to review, Enter to open.';
}

function defaultRenderOption(result, node) {
  node.textContent = result && result.row ? result.row.t : String(result);
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
export function createCombobox({
  input, listbox, status, getResults, renderOption = defaultRenderOption, onSelect,
}) {
  comboboxCount += 1;
  const prefix = input.id || `combobox-${comboboxCount}`;
  prepare(input, listbox, prefix);

  const state = { results: [], active: -1, token: 0 };

  function setActive(index) {
    state.active = index;
    for (const [i, node] of [...listbox.children].entries()) {
      const on = i === index;
      node.setAttribute('aria-selected', String(on));
      node.classList.toggle('is-active', on);
    }
    if (index < 0) input.removeAttribute('aria-activedescendant');
    else {
      input.setAttribute('aria-activedescendant', optionId(prefix, index));
      listbox.children[index].scrollIntoView({ block: 'nearest' });
    }
  }

  function close() {
    listbox.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    setActive(-1);
  }

  function open() {
    if (!state.results.length) return close();
    listbox.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    return undefined;
  }

  function choose(index) {
    const result = state.results[index];
    if (result === undefined) return;
    close();
    onSelect(result);
  }

  function paint(results, query) {
    state.results = results;
    listbox.replaceChildren();
    for (const [index, result] of results.entries()) {
      const node = document.createElement('li');
      node.id = optionId(prefix, index);
      node.setAttribute('role', 'option');
      node.setAttribute('aria-selected', 'false');
      renderOption(result, node);
      node.addEventListener('pointerdown', (event) => {
        event.preventDefault(); // keep focus in the input
        choose(index);
      });
      listbox.appendChild(node);
    }
    if (status) status.textContent = describe(results.length, query);
    setActive(-1);
    open();
  }

  function refresh() {
    const query = input.value.trim();
    const token = (state.token += 1);
    if (!query) {
      state.results = [];
      listbox.replaceChildren();
      if (status) status.textContent = '';
      return close();
    }
    return Promise.resolve(getResults(query)).then((results) => {
      if (token === state.token) paint(results || [], query);
    });
  }

  state.move = (step) => {
    if (listbox.hidden) open();
    const count = state.results.length;
    if (!count) return;
    // From nothing active, Down goes to the first option and Up to the last.
    const from = state.active < 0 ? (step > 0 ? -1 : 0) : state.active;
    setActive((from + step + count) % count);
  };
  state.moveTo = (index) => {
    if (state.results.length) {
      open();
      setActive(Math.max(0, Math.min(index, state.results.length - 1)));
    }
  };

  function onKeyDown(event) {
    if (KEYS[event.key]) {
      event.preventDefault();
      KEYS[event.key](state);
      return;
    }
    if (event.key === 'Enter' && state.active >= 0) {
      event.preventDefault();
      choose(state.active);
    } else if (event.key === 'Escape') {
      close();
    }
  }

  const onFocus = () => {
    if (state.results.length) open();
  };
  const onBlur = () => setTimeout(close, 120); // let a tap on an option land first

  input.addEventListener('input', refresh);
  input.addEventListener('keydown', onKeyDown);
  input.addEventListener('focus', onFocus);
  input.addEventListener('blur', onBlur);

  return {
    open,
    close,
    refresh,
    destroy() {
      input.removeEventListener('input', refresh);
      input.removeEventListener('keydown', onKeyDown);
      input.removeEventListener('focus', onFocus);
      input.removeEventListener('blur', onBlur);
      close();
      listbox.replaceChildren();
    },
  };
}
