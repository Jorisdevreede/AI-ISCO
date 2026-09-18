import test from 'node:test';
import assert from 'node:assert/strict';

import {
  borrowedRationaleNote, rationaleWriterLine, scorerName,
} from '../../site/js/rationale.js';

test('the line above the rationales says who wrote them', () => {
  const own = [{ rationaleFrom: null }, {}];
  const borrowed = [{ rationaleFrom: null }, { rationaleFrom: { s: 'gemini', a: 9, m: 3 } }];

  assert.match(rationaleWriterLine(own), /what the model wrote while it scored/);
  assert.match(rationaleWriterLine(borrowed), /written by Gemini for its own scores/);
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
