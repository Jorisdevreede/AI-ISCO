// The shared navigation and attribution footer. Thin DOM module.
//
// SCRIPT ORDER (see site/js/README.md): the page ships a static, empty
//   <nav id="site-nav"></nav>
// as the first element in <body>. site/scorer.js is loaded with `defer` BEFORE
// the page module and grabs that element by id, so it must exist in the markup
// and must never be replaced. renderChrome only appends its own nodes (marked
// data-chrome) and leaves anything scorer.js added alone.

import { withScorer } from './urlstate.js';

/** Navigation, same on every page. Labels and hrefs are fixed by the brief. */
export const NAV_ITEMS = [
  { key: 'index', href: 'index.html', label: 'Find a job' },
  { key: 'groups', href: 'groups.html', label: 'Browse sectors' },
  { key: 'skill', href: 'skill.html', label: 'Look up a skill' },
  { key: 'insights', href: 'insights.html', label: 'What we found' },
  { key: 'method', href: 'method.html', label: 'How sure is this?' },
];

/** Where "Licensing and attribution" points. */
export const NOTICES_URL =
  'https://github.com/Jorisdevreede/AI-ISCO/blob/master/THIRD-PARTY-NOTICES.md';

// Wording copied verbatim from the <footer class="site-attribution"> of
// site/index.html. ESCO and ILO require these acknowledgements; do not reword.
const FOOTER_TEXT = [
  'This service uses the ESCO classification of the European Commission. What you '
    + 'see is a modified version of ESCO v1.2.1: the scores, quadrants, rationales and '
    + 'narratives are AI-generated additions (Google Gemini Flash) and are not part of '
    + 'ESCO. They are model estimates, meant for exploring, not for predicting.',
  'Occupation groups follow ISCO-08. ESCO states: “Information and data in ESCO '
    + 'is based on an original work published by the ILO under the title International '
    + 'Standard Classification of Occupations, ISCO-08. Structure, Group Definitions and '
    + 'Correspondence Tables. Copyright © 2012 International Labour Organization. '
    + 'Adapted and reproduced with permission.” Neither the European Commission nor '
    + 'the ILO endorses this site.',
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  node.dataset.chrome = '';
  return node;
}

function link(href, label, search) {
  const anchor = element('a', null, label);
  anchor.href = withScorer(href, search);
  return anchor;
}

function buildNavLinks(active, search) {
  const links = element('div', 'nav-links');
  for (const item of NAV_ITEMS) {
    const anchor = link(item.href, item.label, search);
    if (item.key === active || item.href === active) {
      anchor.setAttribute('aria-current', 'page');
    }
    links.appendChild(anchor);
  }
  return links;
}

function buildFooter() {
  const footer = element('footer', 'site-attribution');
  for (const text of FOOTER_TEXT) footer.appendChild(element('p', null, text));
  const last = element('p');
  const anchor = element('a', null, 'Licensing and attribution');
  anchor.href = NOTICES_URL;
  last.appendChild(anchor);
  footer.appendChild(last);
  return footer;
}

function clearOwnNodes(parent) {
  for (const node of [...parent.querySelectorAll(':scope > [data-chrome]')]) node.remove();
}

/**
 * Render the shared navigation and the attribution footer.
 *
 * @param {{active?: string, search?: string}} [options]
 *   `active` is a NAV_ITEMS key ("index", "groups", "skill", "insights",
 *   "method") or an href; that link gets aria-current="page".
 *   `search` defaults to location.search, and any ?scorer= in it is carried
 *   onto every nav link.
 * @returns {{nav: HTMLElement, footer: HTMLElement}}
 */
export function renderChrome({ active, search } = {}) {
  const query = search === undefined ? window.location.search : search;
  const nav = document.getElementById('site-nav') || document.createElement('nav');
  nav.id = 'site-nav';
  if (!nav.isConnected) document.body.insertBefore(nav, document.body.firstChild);
  clearOwnNodes(nav);
  for (const node of document.querySelectorAll('.skip-link[data-chrome]')) node.remove();

  const skip = element('a', 'skip-link', 'Skip to content');
  skip.href = '#main';
  nav.parentNode.insertBefore(skip, nav);

  const brand = link('index.html', 'AI-ISCO', query);
  brand.classList.add('nav-brand');
  // scorer.js may have added its switch already; the brand and links go before it
  // so the switch stays at the right-hand end whichever script finishes first.
  nav.prepend(brand, buildNavLinks(active, query));

  const existing = document.querySelector('footer.site-attribution');
  const footer = existing || buildFooter();
  if (!existing) document.body.appendChild(footer);
  return { nav, footer };
}
