import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BORROWED_GAP_LIMIT, borrowedGap, borrowedRationale, borrowedRationaleNote,
  rationaleWriterLine, scorerName,
} from '../../site/js/rationale.js';

test('the line above the rationales says who wrote them', () => {
  const own = [{ rationaleFrom: null }, {}];
  const borrowed = [{ rationaleFrom: null }, { rationaleFrom: { s: 'gemini', a: 9, m: 3 } }];

  assert.match(rationaleWriterLine(own), /what the model wrote while it scored/);
  assert.match(rationaleWriterLine(borrowed), /written by Gemini for its own, older scores/);
  assert.match(rationaleWriterLine(borrowed), /the explanation is left out/);
  assert.match(rationaleWriterLine(undefined), /what the model wrote/);
});

test('a rationale written for the scores on screen gets no note', () => {
  assert.equal(borrowedRationaleNote(undefined), null);
  assert.equal(borrowedRationaleNote(null), null);
  assert.equal(borrowedRationaleNote('gemini'), null);
});

test('a borrowed rationale names its writer and the scores it was written for', () => {
  const note = borrowedRationaleNote({ s: 'gemini', a: 9, m: 3 });

  assert.equal(note.writer, 'Gemini');
  assert.match(note.text, /^Gemini wrote this explanation for its own scores/);
  assert.match(note.text, /automation risk 9\.0, amplification 3\.0\./);
  assert.match(note.text, /returns numbers only and writes no text/);
  assert.match(note.label, /written by Gemini$/);
});

test('a missing source score reads as not scored, never as a blank', () => {
  const note = borrowedRationaleNote({ s: 'gemini', a: null, m: 4.5 });

  assert.match(note.text, /automation risk Not scored, amplification 4\.5\./);
});

test('scorer names: known ones, unknown ones with a capital, and none at all', () => {
  assert.equal(scorerName('gemini'), 'Gemini');
  assert.equal(scorerName('GEMINI'), 'Gemini');
  assert.equal(scorerName('other'), 'Other');
  assert.equal(scorerName(''), 'another model');
  assert.equal(scorerName(undefined), 'another model');
});

// --- C2: a borrowed explanation is only shown while it is still about this skill

test('the gap is the distance between the borrowed score and the one on screen', () => {
  assert.equal(BORROWED_GAP_LIMIT, 3);
  assert.equal(borrowedGap({ a: 9.0 }, 2.4), 6.6);
  assert.equal(borrowedGap({ a: 6.0 }, 6.2), 0.2);
  assert.equal(borrowedGap(null, 6.2), null);
  assert.equal(borrowedGap({ a: 9.0 }, null), null);
});

test('a rationale written for a very different score is not printed (C2)', () => {
  const far = borrowedRationale({
    text: 'AI can automate the conversion of sketches entirely.',
    source: { s: 'gemini', a: 9.0 }, score: 2.4,
  });
  assert.equal(far.show, false);
  assert.match(far.why, /Gemini scored this 9\.0 for AI substitution/);
  assert.match(far.why, /against the 2\.4 shown here/);
  assert.match(far.why, /not shown/);
  // Exactly at the limit is far enough: the sentences start contradicting there.
  assert.equal(borrowedRationale({ text: 'x', source: { a: 9 }, score: 6 }).show, false);
  assert.equal(borrowedRationale({ text: 'x', source: { a: 9 }, score: 6.1 }).show, true);
});

test('a rationale that is printed leads with the gap, not a generic note (C2)', () => {
  const near = borrowedRationale({
    text: 'Generative AI can produce complex technical layouts.',
    source: { s: 'gemini', a: 8.0 }, score: 6.2,
  });
  assert.equal(near.show, true);
  assert.equal(near.lead, 'Gemini scored this 8.0 for AI substitution and wrote:');
  assert.equal(near.after, 'This page shows 6.2.');
  assert.equal(near.why, '');
});

test('a rationale the model wrote for its own scores needs no gap at all', () => {
  const own = borrowedRationale({ text: 'Its own words.', score: 6.2 });
  assert.deepEqual([own.show, own.lead, own.after, own.why], [true, '', '', '']);
  // No text at all is not a rationale to show, and not a suppression to explain.
  assert.deepEqual(borrowedRationale({ source: { a: 9 }, score: 2 }).show, false);
  assert.equal(borrowedRationale({}).why, '');
});
