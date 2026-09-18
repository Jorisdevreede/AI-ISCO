import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_FRAMING, FRAMINGS, WORDING, framingParam, isNeutral, normaliseFraming,
  otherFraming, recordsRecent, wordingFor,
} from '../../site/js/pages/framing.js';

// Every heading the page prints from the wording table. The toggle label is not
// here: in neutral mode it talks to the reader about their own job on purpose.
const HEADINGS = [
  'scoresHeading', 'scoresIntro', 'adviceHeading', 'scatterHeading', 'scatterIntro',
  'depreciatingHeading', 'appreciatingHeading', 'cardsHeading', 'storyHeading',
  'tasksHeading', 'weekHeading', 'pathsHeading', 'pathsIntro', 'sidewaysHeading',
  'learnHeading', 'learnIntro',
];

test('an absent or unknown for= reads as my-job wording', () => {
  for (const value of [undefined, null, '', 'mine', 'MY', 'Other', 'true']) {
    assert.equal(normaliseFraming(value), DEFAULT_FRAMING);
  }
  assert.equal(normaliseFraming('other'), 'other');
  assert.deepEqual(FRAMINGS, ['my', 'other']);
});

test('the switch moves to the mode the visitor is not in', () => {
  assert.equal(otherFraming('my'), 'other');
  assert.equal(otherFraming('other'), 'my');
  assert.equal(otherFraming(undefined), 'other');
  assert.equal(isNeutral('other'), true);
  assert.equal(isNeutral(undefined), false);
});

test('only my-job mode writes to recently viewed', () => {
  assert.equal(recordsRecent('my'), true);
  assert.equal(recordsRecent(undefined), true);
  assert.equal(recordsRecent('other'), false);
});

test('framingParam keeps a my-job link clean', () => {
  assert.equal(framingParam('my'), '');
  assert.equal(framingParam(undefined), '');
  assert.equal(framingParam('other'), 'other');
});

test('both wording tables carry exactly the same keys', () => {
  assert.deepEqual(Object.keys(WORDING.my).sort(), Object.keys(WORDING.other).sort());
  for (const key of HEADINGS) {
    assert.equal(typeof WORDING.my[key], 'string', `my.${key}`);
    assert.ok(WORDING.other[key].length > 0, `other.${key} is empty`);
  }
});

test('neutral wording never addresses the reader as the job holder', () => {
  for (const key of HEADINGS) {
    assert.doesNotMatch(WORDING.other[key], /\b(you|your|yours|yourself)\b/i, key);
  }
});

test('the advice heading is the one the brief specifies', () => {
  assert.equal(wordingFor('my').adviceHeading, 'What to do next');
  assert.equal(wordingFor(undefined).adviceHeading, 'What to do next');
  assert.equal(
    wordingFor('other').adviceHeading,
    'What someone in this role could do next',
  );
});

test('neutral mode says out loud that the story is addressed to the job holder', () => {
  assert.equal(wordingFor('my').storyNote, '');
  assert.match(wordingFor('other').storyNote, /written to the job holder/);
});

test('the switch invites the visitor in whichever mode they are in', () => {
  assert.match(wordingFor('my').toggleLabel, /someone else/i);
  assert.match(wordingFor('other').toggleLabel, /switch back/i);
});
