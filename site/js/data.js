// Fetching the index files. Thin DOM/network layer: keep logic out of here.
//
// Everything goes through window.scorerFetch when site/scorer.js has run, so a
// local second score set keeps working. Falls back to plain fetch so a module
// can be used on a page without the scorer switch (and in the self-test).

const cache = new Map();

/** Thrown with a message a page can show the visitor as-is. */
export class LoadError extends Error {
  constructor(name, reason) {
    super(`Couldn't load ${name}.json. ${reason}`);
    this.name = 'LoadError';
    this.file = name;
  }
}

function fetchFile(name) {
  if (typeof window !== 'undefined' && typeof window.scorerFetch === 'function') {
    return window.scorerFetch(name);
  }
  return fetch(`${name}.json`);
}

/**
 * Load one index file, once per page load.
 *
 * @param {string} name file name without ".json", e.g. "search_index"
 * @returns {Promise<any>} rejects with a LoadError whose message is showable
 */
export function loadJSON(name) {
  if (!cache.has(name)) {
    cache.set(name, fetchFile(name).then((response) => {
      if (!response.ok) {
        throw new LoadError(name, `The server answered ${response.status}.`);
      }
      return response.json();
    }).catch((error) => {
      cache.delete(name); // a retry should really retry
      throw error instanceof LoadError ? error : new LoadError(name, error.message);
    }));
  }
  return cache.get(name);
}

function fetchPlain(file) {
  return fetch(`${file}.json`).then((response) => {
    if (!response.ok) throw new LoadError(file, `The server answered ${response.status}.`);
    return response.json();
  });
}

/**
 * Both score sets of one index file, whichever set the switch is on.
 *
 * @param {string} name file name without ".json", e.g. "search_index"
 * @returns {Promise<null | {first: any, second: any, labels: [string, string]}>}
 *   null where only the default set is deployed (or scorer.js has not run)
 */
export async function loadBothSets(name) {
  const alternative = await (typeof window !== 'undefined' && window.scorerAlternative);
  if (!alternative) return null;
  const [first, second] = await Promise.all(
    [fetchPlain(name), fetchPlain(name + alternative.suffix)],
  );
  return { first, second, labels: alternative.labels };
}

/** Forget everything loaded so far. For tests and the self-test page. */
export function clearCache() {
  cache.clear();
}

/**
 * Call `onSlow` if `promise` has not settled within `ms`, so a page can show a
 * skeleton only when the wait is real.
 *
 * @param {Promise<T>} promise
 * @param {number} ms typically 200
 * @param {Function} onSlow called with no arguments, at most once
 * @returns {Promise<T>} the original promise's result
 * @template T
 */
export function whenSlow(promise, ms, onSlow) {
  let pending = true;
  const timer = setTimeout(() => {
    if (pending) onSlow();
  }, ms);
  const stop = () => {
    pending = false;
    clearTimeout(timer);
  };
  return promise.then(
    (value) => {
      stop();
      return value;
    },
    (error) => {
      stop();
      throw error;
    },
  );
}
