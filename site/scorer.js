// Which set of skill scores the site shows.
//
// The default set is the Gemini scores in data.json, portfolio_data.json and the
// index files. A second set, built with `--scorer typesafe`, sits next to them
// with a _typesafe suffix on every file. The switch below appears wherever that
// second set exists; a checkout without it behaves as a single-scorer site.
//
// Pages load their data with scorerFetch('data') instead of fetch('data.json').
// The choice comes from ?scorer=typesafe, then localStorage, then the default.
// Load this right after <nav id="site-nav">.
(function () {
  const SCORERS = {
    gemini: { label: 'Gemini', suffix: '' },
    typesafe: {
      label: 'TypeSafe',
      suffix: '_typesafe',
      note: 'Scores from TypeSafe’s jev model, same rubric. It returns numbers only, so ' +
        'the narratives and skill rationales are the ones Gemini wrote for its own ' +
        'scores and may not match these numbers.',
    },
  };
  const DEFAULT = 'gemini';
  const KEY = 'ai-isco-scorer';

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(value) {
    try { localStorage.setItem(KEY, value); } catch (e) { /* private mode */ }
  }

  const fromUrl = new URLSearchParams(location.search).get('scorer');
  let wanted = SCORERS[fromUrl] ? fromUrl : stored();
  if (!SCORERS[wanted]) wanted = DEFAULT;
  if (SCORERS[fromUrl]) store(fromUrl);

  // Is the second set deployed here at all? One HEAD request decides whether the
  // switch is offered, so a checkout without those files never shows a dead option.
  const hasAlternative = fetch('data_typesafe.json', { method: 'HEAD' })
    .then(response => response.ok, () => false);
  const active = hasAlternative.then(ok => (ok ? wanted : DEFAULT));

  window.scorerFetch = name =>
    active.then(scorer => fetch(name + SCORERS[scorer].suffix + '.json'));

  function choose(name) {
    store(name);
    const url = new URL(location.href);
    if (name === DEFAULT) url.searchParams.delete('scorer');
    else url.searchParams.set('scorer', name);
    location.href = url.toString();
  }

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
      @media (max-width: 700px) { .scorer-toggle span { display: none; } }
    `;
    document.head.appendChild(style);
  }

  function addToggle(nav, current) {
    const toggle = document.createElement('div');
    toggle.className = 'scorer-toggle';
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', 'Skill scores from');
    const caption = document.createElement('span');
    caption.textContent = 'Scores from';
    toggle.appendChild(caption);
    Object.keys(SCORERS).forEach(name => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = SCORERS[name].label;
      button.setAttribute('aria-pressed', String(name === current));
      button.addEventListener('click', () => { if (name !== current) choose(name); });
      toggle.appendChild(button);
    });
    nav.appendChild(toggle);
  }

  function addNote(nav, text) {
    const note = document.createElement('div');
    note.className = 'scorer-note';
    note.textContent = text;
    nav.insertAdjacentElement('afterend', note);
  }

  const nav = document.getElementById('site-nav');
  Promise.all([hasAlternative, active]).then(([available, current]) => {
    if (!available || !nav) return;
    addStyles();
    addToggle(nav, current);
    if (SCORERS[current].note) addNote(nav, SCORERS[current].note);
  });
})();
