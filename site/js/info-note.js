// An "i" button that shows and hides a short note. Thin DOM module.
//
// A disclosure, not a tooltip: it works by tap and by keyboard, the note stays
// open until it is closed, and a screen reader hears the button's state.

let nextId = 0;

function noteElement(id, text) {
  const note = document.createElement('span');
  note.id = id;
  note.className = 'info-note-text';
  note.setAttribute('role', 'note');
  note.hidden = true;
  note.textContent = text;
  return note;
}

function buttonElement(id, label) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'info-button';
  button.textContent = 'i';
  button.setAttribute('aria-label', label);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', id);
  return button;
}

/**
 * @param {{label: string, text: string}} note `label` is the button's accessible name
 * @returns {{button: HTMLButtonElement, note: HTMLElement}} place the button after
 *   the text it is about and the note where it may take a full line
 */
export function renderInfoNote({ label, text }) {
  nextId += 1;
  const id = `info-note-${nextId}`;
  const button = buttonElement(id, label);
  const note = noteElement(id, text);
  button.addEventListener('click', () => {
    note.hidden = !note.hidden;
    button.setAttribute('aria-expanded', String(!note.hidden));
  });
  return { button, note };
}
