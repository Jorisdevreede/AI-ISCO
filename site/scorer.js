// Which set of skill scores the site shows.
//
// The published site has one set: the Gemini scores in data.json and
// portfolio_data.json. A second set, built with `--scorer typesafe`, can sit next
// to them as data_typesafe.json and portfolio_data_typesafe.json. Those files are
// gitignored (TypeSafe's customer agreement does not allow publishing them), so
// the switch below only appears where they exist, which in practice means a local
// checkout. Without them the site behaves exactly as before.
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
      note: 'Local experiment: scores from TypeSafe, same rubric. Narratives and ' +
        'skill rationales were written from the Gemini scores and may not match ' +
        'these numbers.',
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

  // Is the second set deployed here at all? Only look where it can exist: on a
  // local checkout, or when someone asked for it. The published site never probes.
  const isLocal = ['localhost', '127.0.0.1', ''].includes(location.hostname);
  const hasAlternative = (isLocal || wanted !== DEFAULT)
    ? fetch('data_typesafe.json', { method: 'HEAD' }).then(response => response.ok, () => false)
    : Promise.resolve(false);
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
