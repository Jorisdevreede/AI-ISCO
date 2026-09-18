import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildHash, groupHref, jobHref, parseHash, skillHref, withScorer,
} from '../../site/js/urlstate.js';

test('the brief\'s hashes parse into an id and parameters', () => {
  assert.deepEqual(parseHash('#software-developer'), {
    id: 'software-developer', params: {},
  });
  assert.deepEqual(parseHash('#software-developer&for=other'), {
    id: 'software-developer', params: { for: 'other' },
  });
  assert.deepEqual(parseHash('#optical-engineer&from=unit:2149'), {
    id: 'optical-engineer', params: { from: 'unit:2149' },
  });
  assert.deepEqual(parseHash('#g=major:2&view=ranked'), {
    id: null, params: { g: 'major:2', view: 'ranked' },
  });
  assert.deepEqual(parseHash('#59b27e7b'), { id: '59b27e7b', params: {} });
});

test('an empty or missing hash is not an error', () => {
  for (const hash of ['', '#', null, undefined]) {
    assert.deepEqual(parseHash(hash), { id: null, params: {} });
  }
});

test('buildHash is the inverse of parseHash', () => {
  const cases = [
    ['software-developer', {}],
    ['software-developer', { for: 'other' }],
    ['optical-engineer', { from: 'unit:2149' }],
    [null, { g: 'major:2', view: 'ranked' }],
    ['59b27e7b', {}],
  ];
  for (const [id, params] of cases) {
    assert.deepEqual(parseHash(buildHash(id, params)), { id, params });
  }
});

test('group keys keep their readable colon', () => {
  assert.equal(buildHash(null, { g: 'major:2', view: 'ranked' }), '#g=major:2&view=ranked');
  assert.equal(buildHash('optical-engineer', { from: 'unit:2149' }),
    '#optical-engineer&from=unit:2149');
});

test('odd characters in keys and values round-trip', () => {
  const params = {
    'a key': 'a & b = c',
    'weird#key': 'value#with#hashes',
    unicode: 'sommelière',
    percent: '100%',
    empty_looking: ' ',
  };
  const round = parseHash(buildHash('id with spaces', params));
  assert.equal(round.id, 'id with spaces');
  assert.deepEqual(round.params, params);
});

test('empty values are dropped rather than written as bare keys', () => {
  assert.equal(buildHash('slug', { view: '', from: null, to: undefined }), '#slug');
  assert.equal(buildHash(null, {}), '');
});

test('a malformed percent escape does not throw', () => {
  assert.deepEqual(parseHash('#g=100%'), { id: null, params: { g: '100%' } });
});

test('the href helpers build the brief\'s links', () => {
  assert.equal(jobHref('software-developer'), 'job.html#software-developer');
  assert.equal(jobHref('software-developer', { for: 'other' }),
    'job.html#software-developer&for=other');
  assert.equal(jobHref('optical-engineer', { from: 'unit:2149' }),
    'job.html#optical-engineer&from=unit:2149');
  assert.equal(groupHref('major:2', 'ranked'), 'groups.html#g=major:2&view=ranked');
  assert.equal(groupHref('major:2'), 'groups.html#g=major:2');
  assert.equal(skillHref('59b27e7b'), 'skill.html#59b27e7b');
});

test('withScorer carries the switch across links, and only when present', () => {
  assert.equal(withScorer('job.html#slug', ''), 'job.html#slug');
  assert.equal(withScorer('job.html#slug', '?scorer=typesafe'),
    'job.html?scorer=typesafe#slug');
  assert.equal(withScorer('groups.html', '?scorer=typesafe'),
    'groups.html?scorer=typesafe');
  assert.equal(withScorer('job.html?embed=1#slug', '?scorer=typesafe'),
    'job.html?embed=1&scorer=typesafe#slug');
  assert.equal(withScorer('job.html#a&from=unit:2149', '?scorer=typesafe&other=1'),
    'job.html?scorer=typesafe#a&from=unit:2149');
});

test('withScorer replaces an outdated scorer rather than duplicating it', () => {
  assert.equal(withScorer('job.html?scorer=gemini#slug', '?scorer=typesafe'),
    'job.html?scorer=typesafe#slug');
});
