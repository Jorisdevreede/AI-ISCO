// Reading and writing the hash schemes. Pure: no DOM, no globals.
//
//   job.html#software-developer
//   job.html#software-developer&for=other
//   job.html#optical-engineer&from=unit:2149
//   groups.html#g=major:2&view=ranked
//   skill.html#59b27e7b
//
// A hash is an optional bare id followed by &key=value pairs. Everything is
// percent-encoded except ":", which group keys use and which is legal in a
// fragment, so "#g=major:2" stays readable and still round-trips.

/** Encode one id, key or value for a hash. */
function encode(text) {
  return encodeURIComponent(String(text)).replace(/%3A/g, ':');
}

function decode(text) {
  try {
    return decodeURIComponent(text);
  } catch {
    return text; // a hand-typed "%" should not throw the page away
  }
}

/**
 * Parse a location hash.
 * @param {string} hash with or without the leading "#"
 * @returns {{id: string|null, params: Object<string, string>}}
 */
export function parseHash(hash) {
  const body = String(hash ?? '').replace(/^#/, '');
  const params = {};
  let id = null;
  for (const [index, part] of body.split('&').entries()) {
    if (!part) continue;
    const split = part.indexOf('=');
    if (split === -1) {
      if (index === 0) id = decode(part);
      continue;
    }
    params[decode(part.slice(0, split))] = decode(part.slice(split + 1));
  }
  return { id, params };
}

/**
 * Build a hash from an id and parameters. Inverse of parseHash.
 * @param {string|null} id the bare first segment, or null
 * @param {Object<string, string>} [params={}] keys are emitted in insertion order
 * @returns {string} starting with "#", or "" when there is nothing to say
 */
export function buildHash(id, params = {}) {
  const parts = [];
  if (id) parts.push(encode(id));
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    parts.push(`${encode(key)}=${encode(value)}`);
  }
  return parts.length ? `#${parts.join('&')}` : '';
}

/**
 * The href of a job page, carrying the context the job page needs.
 * @param {string} slug
 * @param {{for?: string, from?: string}} [options]
 * @returns {string}
 */
export function jobHref(slug, options = {}) {
  return `job.html${buildHash(slug, options)}`;
}

/**
 * The href of a group overview.
 * @param {string} key group key such as "major:2"
 * @param {string} [view] "ranked" | "scatter" | "treemap" | "table"
 * @returns {string}
 */
export function groupHref(key, view) {
  return `groups.html${buildHash(null, { g: key, view })}`;
}

/**
 * The href of a skill lookup.
 * @param {string} id 8-character skill id
 * @returns {string}
 */
export function skillHref(id) {
  return `skill.html${buildHash(id)}`;
}

/**
 * Carry ?scorer= across a link when the current page has one.
 *
 * The published site never has it; a local checkout with the second score set
 * does, and every in-site link must keep it or the switch silently resets.
 *
 * @param {string} href relative or absolute, may already carry a query or hash
 * @param {string} search the current location.search
 * @returns {string} href, with scorer added when it was missing
 */
export function withScorer(href, search) {
  const scorer = new URLSearchParams(search || '').get('scorer');
  if (!scorer) return href;
  const [beforeHash, ...hashRest] = String(href).split('#');
  const [path, existing] = beforeHash.split('?');
  const query = new URLSearchParams(existing || '');
  query.set('scorer', scorer);
  const hash = hashRest.length ? `#${hashRest.join('#')}` : '';
  return `${path}?${query.toString()}${hash}`;
}
