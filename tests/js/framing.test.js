import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_FRAMING, FRAMINGS, SHARES_WORDING, WORDING, framingParam, isNeutral,
  normaliseFraming, otherFraming, recordsRecent, wordingFor,
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

// --- the shares scheme -------------------------------------------------------

const SHARES = 'shares';

test('both shares tables carry exactly the same keys, and only overrides', () => {
  assert.deepEqual(Object.keys(SHARES_WORDING.my).sort(), Object.keys(SHARES_WORDING.other).sort());
  for (const key of Object.keys(SHARES_WORDING.my)) {
    assert.equal(typeof WORDING.my[key], 'string', `${key} has no quadrant wording to replace`);
    assert.notEqual(SHARES_WORDING.my[key], WORDING.my[key], `${key} is not an override`);
  }
});

test('the shares wording lies over the quadrant wording, not beside it', () => {
  const words = wordingFor('my', SHARES);
  assert.equal(words.scoresHeading, WORDING.my.scoresHeading);
  assert.equal(words.scatterIntro, SHARES_WORDING.my.scatterIntro);
  // No scheme, or the other scheme, leaves the quadrant wording exactly as it was.
  assert.deepEqual(wordingFor('my'), WORDING.my);
  assert.deepEqual(wordingFor('other', 'quadrants'), WORDING.other);
});

test('no shares sentence talks about boxes, cut-offs or a pair of scores', () => {
  for (const mode of ['my', 'other']) {
    for (const [key, text] of Object.entries(wordingFor(mode, SHARES))) {
      if (key.startsWith('depreciating') || key.startsWith('appreciating')) continue;
      assert.doesNotMatch(text, /quadrant|cut-off|\bbox(es)?\b|two scores/i, `${mode}.${key}`);
    }
  }
});

test('neutral shares wording never addresses the reader as the job holder', () => {
  for (const [key, text] of Object.entries(SHARES_WORDING.other)) {
    assert.doesNotMatch(text, /\b(you|your|yours|yourself)\b/i, key);
  }
});

test('with no move to show, nothing promises one (A7)', () => {
  for (const scheme of [undefined, SHARES]) {
    for (const mode of ['my', 'other']) {
      const words = wordingFor(mode, scheme);
      assert.equal(words.pathsHeadingNone, 'Nearby jobs, and how they differ');
      assert.doesNotMatch(words.pathsHeadingNone, /move|step up/i);
      assert.equal(words.needleHeading, 'What would move the needle');
      assert.ok(words.needleText.length > 40, `${mode}/${scheme} needleText`);
    }
  }
  // The heading that DOES promise a move is still there for when there is one.
  assert.match(wordingFor('my').pathsHeading, /move next/);
});

test('the learn list says it ranks by net gain, not by amplification (A21)', () => {
  for (const scheme of [undefined, SHARES]) {
    for (const mode of ['my', 'other']) {
      const intro = wordingFor(mode, scheme).learnIntro;
      assert.match(intro, /net gain/);
      assert.match(intro, /take over/);
      assert.doesNotMatch(intro, /most amplified first/);
    }
  }
});

test('the three class headings name a class, never a letter', () => {
  const words = wordingFor('other', SHARES);
  assert.equal(words.classSHeading, 'Skills AI can take over');
  assert.equal(words.classAHeading, 'Skills AI assists with');
  assert.equal(words.classMHeading, 'Skills machines can do');
});
