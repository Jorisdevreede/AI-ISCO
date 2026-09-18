import test, { afterEach, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  LoadError, clearCache, loadBothSets, loadJSON, whenSlow,
} from '../../site/js/data.js';

// data.js is the site's only network layer, so every test here stands in for the
// browser: a `window` object that scorer.js may or may not have furnished, and a
// `fetch` that answers from a script instead of the disk. Node has a real global
// fetch and no window at all, so both are saved and put back afterwards.

const REAL_FETCH = globalThis.fetch;

/** A stand-in Response: only `ok`, `status` and `json` are ever read. */
function body(value) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(value) });
}

function refusal(status) {
  return Promise.resolve({
    ok: false,
    status,
    json: () => Promise.reject(new Error('a refusal has no body to read')),
  });
}

/** Replaces `fetch` with `answer` and returns the list of URLs it is asked for. */
function stubFetch(answer) {
  const calls = [];
  globalThis.fetch = (url) => {
    calls.push(String(url));
    return answer(String(url), calls.length);
  };
  return calls;
}

/** Let every pending microtask run, without letting any timer fire. */
const settled = () => new Promise((resolve) => { setImmediate(resolve); });

/** What a rejected promise rejected with, so the error itself can be inspected. */
const reasonOf = (promise) => promise.then(() => null, (error) => error);

beforeEach(() => {
  clearCache();
  delete globalThis.window;
});

afterEach(() => {
  clearCache();
  delete globalThis.window; // node has none of its own, so the stub simply goes
  globalThis.fetch = REAL_FETCH;
});

/* --- loading one file ----------------------------------------------------- */

test('a file is fetched once per page load and every caller gets that same load', async () => {
  const calls = stubFetch(() => body({ rows: ['nurse'] }));
  const first = loadJSON('search_index');
  const second = loadJSON('search_index');
  assert.equal(first, second, 'the second caller must join the load, not start another');
  assert.deepEqual(await first, { rows: ['nurse'] });
  assert.deepEqual(await second, { rows: ['nurse'] });
  assert.deepEqual(await loadJSON('search_index'), { rows: ['nurse'] });
  assert.deepEqual(calls, ['search_index.json'], 'one file, one request');
});

test('two different files are two different loads', async () => {
  const calls = stubFetch((url) => body({ url }));
  assert.deepEqual(await loadJSON('stats'), { url: 'stats.json' });
  assert.deepEqual(await loadJSON('search_index'), { url: 'search_index.json' });
  assert.deepEqual(calls, ['stats.json', 'search_index.json']);
});

test('the scorer switch takes the fetch over whenever scorer.js has run', async () => {
  const asked = [];
  globalThis.window = {
    scorerFetch: (name) => {
      asked.push(name);
      return body({ set: 'the second score set' });
    },
  };
  const calls = stubFetch(() => body({ set: 'the deployed set' }));

  assert.deepEqual(await loadJSON('stats'), { set: 'the second score set' });
  assert.deepEqual(asked, ['stats'], 'scorerFetch is handed the bare name, not the file name');
  assert.deepEqual(calls, [], 'plain fetch must not be used behind the switch');
});

test('a page without the scorer switch falls back to plain fetch', async () => {
  const calls = stubFetch(() => body({ set: 'the deployed set' }));

  // No window at all: a module used on a page that never loaded scorer.js.
  assert.deepEqual(await loadJSON('stats'), { set: 'the deployed set' });

  // A window whose scorerFetch is not callable is no switch either.
  clearCache();
  globalThis.window = { scorerFetch: 'soon' };
  assert.deepEqual(await loadJSON('stats'), { set: 'the deployed set' });
  assert.deepEqual(calls, ['stats.json', 'stats.json']);
});

/* --- when a load fails ---------------------------------------------------- */

test('a server that refuses rejects with a sentence naming the file and the status', async () => {
  stubFetch(() => refusal(404));
  const error = await reasonOf(loadJSON('skill_index'));
  assert.ok(error instanceof LoadError, 'the page shows this text as-is');
  assert.equal(error.name, 'LoadError');
  assert.equal(error.file, 'skill_index');
  assert.equal(error.message, "Couldn't load skill_index.json. The server answered 404.");
});

test('a network failure is wrapped so the page can still say which file went missing', async () => {
  stubFetch(() => Promise.reject(new TypeError('Failed to fetch')));
  const error = await reasonOf(loadJSON('stats'));
  assert.ok(error instanceof LoadError);
  assert.equal(error.file, 'stats', 'a bare TypeError does not say what was being loaded');
  assert.equal(error.message, "Couldn't load stats.json. Failed to fetch");
});

test('an error that already names its file is passed on rather than wrapped twice', async () => {
  const thrown = new LoadError('stats', 'The second score set is not deployed.');
  stubFetch(() => Promise.reject(thrown));
  const error = await reasonOf(loadJSON('stats'));
  assert.equal(error, thrown, 'wrapping again would bury the reason inside a second sentence');
  assert.equal(error.message, "Couldn't load stats.json. The second score set is not deployed.");
});

test('a failed load is forgotten, so asking again really asks again', async () => {
  const calls = stubFetch((url, nth) => (nth === 1 ? refusal(503) : body({ rows: ['nurse'] })));

  await assert.rejects(loadJSON('search_index'), LoadError);
  // The failure must not be cached: a visitor who retries gets a fresh request,
  // and a server that has come back up answers it.
  assert.deepEqual(await loadJSON('search_index'), { rows: ['nurse'] });
  assert.deepEqual(calls, ['search_index.json', 'search_index.json']);
});

test('clearing the cache makes the next load a real request again', async () => {
  const calls = stubFetch(() => body({ n: 1 }));
  await loadJSON('stats');
  await loadJSON('stats');
  assert.equal(calls.length, 1);

  clearCache();
  await loadJSON('stats');
  assert.deepEqual(calls, ['stats.json', 'stats.json']);
});

/* --- both score sets at once ---------------------------------------------- */

test('there are no two sets to compare when only one set is deployed', async () => {
  const calls = stubFetch(() => body({}));

  assert.equal(await loadBothSets('stats'), null, 'scorer.js never ran on this page');

  globalThis.window = {};
  assert.equal(await loadBothSets('stats'), null, 'scorer.js ran, but there is no second set');

  globalThis.window = { scorerAlternative: null };
  assert.equal(await loadBothSets('stats'), null);

  assert.deepEqual(calls, [], 'a set that does not exist is never fetched');
});

test('the second set is asked for while the first is still in flight', async () => {
  globalThis.window = { scorerAlternative: { suffix: '_ai', labels: ['ESCO', 'AI'] } };
  let deliverFirst;
  const held = new Promise((resolve) => { deliverFirst = () => resolve(body({ set: 'first' })); });
  const calls = stubFetch((url, nth) => (nth === 1 ? held : body({ set: 'second' })));

  const both = loadBothSets('stats');
  await settled();
  assert.deepEqual(calls, ['stats.json', 'stats_ai.json'],
    'the second file must not wait for the first to arrive');

  deliverFirst();
  assert.deepEqual(await both, {
    first: { set: 'first' },
    second: { set: 'second' },
    labels: ['ESCO', 'AI'],
  });
});

test('both sets go straight to the files, never through the scorer switch', async () => {
  globalThis.window = {
    scorerAlternative: { suffix: '_ai', labels: ['ESCO', 'AI'] },
    scorerFetch: () => { throw new Error('comparing the sets must not go through the switch'); },
  };
  const calls = stubFetch((url) => body({ url }));
  const both = await loadBothSets('search_index');
  assert.deepEqual(calls, ['search_index.json', 'search_index_ai.json']);
  assert.deepEqual(both.first, { url: 'search_index.json' });
  assert.deepEqual(both.second, { url: 'search_index_ai.json' });
});

test('a second set that is not there names its own file in the failure', async () => {
  globalThis.window = { scorerAlternative: { suffix: '_ai', labels: ['ESCO', 'AI'] } };
  stubFetch((url) => (url === 'stats.json' ? body({}) : refusal(404)));

  const error = await reasonOf(loadBothSets('stats'));
  assert.ok(error instanceof LoadError);
  assert.equal(error.file, 'stats_ai', 'the file that failed, not the one that was asked for');
  assert.equal(error.message, "Couldn't load stats_ai.json. The server answered 404.");
});

/* --- the skeleton timer --------------------------------------------------- */

test('whenSlow asks for a skeleton only once the wait has really happened', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let skeletons = 0;
  let deliver;
  const loading = new Promise((resolve) => { deliver = resolve; });
  const waited = whenSlow(loading, 200, () => { skeletons += 1; });

  t.mock.timers.tick(199);
  assert.equal(skeletons, 0, 'a load that is still inside the budget must not flash anything');
  t.mock.timers.tick(1);
  assert.equal(skeletons, 1);

  deliver(['nurse']);
  assert.deepEqual(await waited, ['nurse'], 'the loaded value still comes through');
  t.mock.timers.tick(10_000);
  assert.equal(skeletons, 1, 'onSlow is called at most once');
});

test('a load that beats the budget never flashes a skeleton, then or later', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let skeletons = 0;
  const value = await whenSlow(Promise.resolve({ rows: 3 }), 200, () => { skeletons += 1; });

  assert.deepEqual(value, { rows: 3 });
  t.mock.timers.tick(10_000);
  assert.equal(skeletons, 0, 'settling first must cancel the timer, not merely outrun it');
});

test('a failed load rethrows its own error and cancels the skeleton with it', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let skeletons = 0;
  const refused = new LoadError('stats', 'The server answered 500.');
  const caught = await reasonOf(
    whenSlow(Promise.reject(refused), 200, () => { skeletons += 1; }),
  );

  assert.equal(caught, refused, 'the page must still be told the real reason');
  t.mock.timers.tick(10_000);
  assert.equal(skeletons, 0, 'a failure is not a slow wait');
});
