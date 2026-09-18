// Which set of skill scores the site shows.
//
// Three sets are published side by side:
//
//   v2        TypeSafe's jev model on the scoring-v2 rubric. Files carry a _v2
//             suffix and classify occupations into seven types from four shares.
//             The default wherever it is deployed.
//   gemini    the original rubric scored by Gemini. Files carry no suffix. The
//             alternative in the switch, and the fallback everywhere else.
//   typesafe  the original rubric scored by the same jev model. Files carry a
//             _typesafe suffix. NOT offered in the switch: it exists only for the
//             method page's same-rubric comparison, through scorerAlternative.
//
// Pages load their data with scorerFetch('data') instead of fetch('data.json').
// The choice comes from ?scorer=, then localStorage, then the default. A
// checkout with neither extra set behaves as a plain single-scorer site.
// Load this right after <nav id="site-nav">.
(function () {
  // `kind` says on the switch itself that the two sets are two ways of looking at
  // a job, not the same numbers from two vendors.
  const SCORERS = {
    v2: { label: 'TypeSafe', kind: 'four shares', suffix: '_v2' },
    gemini: { label: 'Gemini', kind: 'two scores', suffix: '' },
    typesafe: { label: 'TypeSafe', suffix: '_typesafe' },
  };
  // What the switch offers, in the order it shows them.
  const OFFERED = ['v2', 'gemini'];
  const FALLBACK = 'gemini';
  const KEY = 'ai-isco-scorer';

  // A link shared before the v2 set existed says ?scorer=typesafe and meant
  // "the jev scores"; it now lands on the jev scores of the current rubric.
  const ALIASES = { typesafe: 'v2' };

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(value) {
    try { localStorage.setItem(KEY, value); } catch (e) { /* private mode */ }
  }
  function normalise(name) {
    const resolved = ALIASES[name] || name;
    return OFFERED.indexOf(resolved) === -1 ? null : resolved;
  }

  const fromUrl = normalise(new URLSearchParams(location.search).get('scorer'));
  const wanted = fromUrl || normalise(stored());
  if (fromUrl) store(fromUrl);

  // Which sets are deployed here at all? One HEAD request per set decides what
  // the switch offers and what the default is, so a checkout without the files
  // never shows a dead option.
  const probe = (name) => fetch(`data${SCORERS[name].suffix}.json`, { method: 'HEAD' })
    .then((response) => response.ok, () => false);
  const hasV2 = probe('v2');
  const hasTypesafe = probe('typesafe');

  const defaultKey = hasV2.then((ok) => (ok ? 'v2' : FALLBACK));
  // Without the v2 files there is nothing to switch to, so the choice is moot.
  const active = defaultKey.then((fallback) => (
    fallback === FALLBACK ? FALLBACK : wanted || fallback));

  /** The key of the set every page is reading. */
  window.scorerActive = active;

  window.scorerFetch = (name) =>
    active.then((scorer) => fetch(name + SCORERS[scorer].suffix + '.json'));

  // For the method page's same-rubric comparison: the Gemini scores against the
  // same rubric scored by jev. Independent of which set the switch is on.
  window.scorerAlternative = hasTypesafe.then((ok) => (ok
    ? { suffix: SCORERS.typesafe.suffix, labels: [SCORERS.gemini.label, SCORERS.typesafe.label] }
    : null));

  function choose(name, fallback) {
    store(name);
    const url = new URL(location.href);
    if (name === fallback) url.searchParams.delete('scorer');
    else url.searchParams.set('scorer', name);
    location.href = url.toString();
  }

  // One note per set, shown whichever set is on: a visitor who flips the switch
  // needs to hear that the whole way of describing a job changed with it.
  const NOTES = {
    v2: 'Scored by TypeSafe’s jev model: each job is split into four shares of work and '
      + 'lands in one of seven types. The model returns numbers only, so the stories and '
      + 'skill explanations are the ones Gemini wrote for its own, older scores.',
    gemini: 'These are the original scores: Gemini rated every skill on two scales, and '
      + 'each job falls into one of four boxes. The default view scores the same skills on '
      + 'a newer rubric, so switching changes the way a job is described, not only the '
      + 'numbers.',
  };
  const NOTE_ID = 'scorer-note';

  function addStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .scorer-toggle { margin-left: auto; display: flex; align-items: center; gap: 8px;
        font-size: 12px; color: var(--fg2, #888894); }
      .scorer-toggle button { font: inherit; font-weight: 500; color: var(--fg2, #888894);
        background: transparent; border: 1px solid rgba(255,255,255,0.12); padding: 4px 10px;
        cursor: pointer; transition: color 0.15s, background 0.15s; }
      .scorer-toggle button:first-of-type { border-radius: 6px 0 0 6px; }
      .scorer-toggle button:last-of-type { border-radius: 0 6px 6px 0; margin-left: -9px; }
      .scorer-toggle button:hover { color: var(--fg, #e0e0e8); }
      .scorer-toggle button[aria-pressed="true"] { color: #d4a017;
        background: rgba(212,160,23,0.1); border-color: rgba(212,160,23,0.4); }
      .scorer-note { padding: 8px 28px; font-size: 12px; line-height: 1.5; color: #d4a017;
        background: rgba(212,160,23,0.08); border-bottom: 1px solid rgba(212,160,23,0.25); }
      .scorer-note a { color: inherit; }
      .scorer-kind { font-weight: 400; opacity: 0.8; }
      @media (min-width: 701px) and (max-width: 1100px) { .scorer-kind { display: none; } }
      @media (max-width: 700px) { .scorer-caption { display: none; } }
    `;
    document.head.appendChild(style);
  }

  function toggleButton(name, current, fallback) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = SCORERS[name].label;
    const kind = document.createElement('span');
    kind.className = 'scorer-kind';
    kind.textContent = ` · ${SCORERS[name].kind}`;
    button.appendChild(kind);
    button.setAttribute('aria-pressed', String(name === current));
    button.addEventListener('click', () => { if (name !== current) choose(name, fallback); });
    return button;
  }

  function addToggle(nav, current, fallback) {
    const toggle = document.createElement('div');
    toggle.className = 'scorer-toggle';
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Skill scores from');
    toggle.setAttribute('aria-describedby', NOTE_ID);
    const caption = document.createElement('span');
    caption.className = 'scorer-caption';
    caption.textContent = 'Scores from';
    toggle.appendChild(caption);
    OFFERED.forEach((name) => toggle.appendChild(toggleButton(name, current, fallback)));
    nav.appendChild(toggle);
  }

  function methodHref() {
    const url = new URL('method.html', location.href);
    const scorer = new URLSearchParams(location.search).get('scorer');
    if (scorer) url.searchParams.set('scorer', scorer);
    return url.pathname + url.search;
  }

  function addNote(nav, current) {
    const note = document.createElement('div');
    note.className = 'scorer-note';
    note.id = NOTE_ID;
    note.appendChild(document.createTextNode(`${NOTES[current]} `));
    const link = document.createElement('a');
    link.href = methodHref();
    link.textContent = 'How sure is this?';
    note.appendChild(link);
    nav.insertAdjacentElement('afterend', note);
  }

  const nav = document.getElementById('site-nav');
  Promise.all([hasV2, active, defaultKey]).then(([available, current, fallback]) => {
    if (!available || !nav) return;
    addStyles();
    addToggle(nav, current, fallback);
    addNote(nav, current);
  });
})();
