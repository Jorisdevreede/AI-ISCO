// My-job versus someone-else's-job wording for the job page. Pure: no DOM.
//
// The default is second person ("your job"). `&for=other` in the hash switches
// every heading and label on the page to neutral, third-person wording, and the
// page stops writing to recently viewed.
//
// The narratives themselves are written in the second person and cannot be
// rewritten client side (a third-person pass is a later increment), so neutral
// mode carries a one-line note saying so rather than pretending otherwise.
//
// There are two score schemes as well as two framings. WORDING is the quadrant
// wording, unchanged; SHARES_WORDING holds only the sentences that differ when
// the active set classifies a job by four shares instead of two cut scores, and
// `wordingFor(value, scheme)` lays the second over the first. Both tables of
// both objects carry exactly the same keys, so a missing translation is a test
// failure rather than an `undefined` on screen.

import { SHARES } from '../scheme.js';

/** The two modes. 'my' is the default. */
export const FRAMINGS = ['my', 'other'];

/** What an absent or unknown `for=` means. */
export const DEFAULT_FRAMING = 'my';

/** The hash parameter that carries the mode. */
export const FRAMING_PARAM = 'for';

/** Every wording the page varies, under the quadrant scheme. */
export const WORDING = {
  my: {
    scoresHeading: 'How exposed your job is',
    scoresIntro: 'Two model estimates, each from 1 to 10. They describe the skills '
      + 'your job is built from, not you.',
    adviceHeading: 'What to do next',
    adviceFallback: 'Start with the skills AI could amplify, further down this page: '
      + 'they are the parts of this job that get more valuable, not less. The skills '
      + 'nearby jobs need are listed at the end.',
    scatterHeading: 'Where your skills sit',
    scatterIntro: 'One dot per skill in your job. The lines are the cut-off of 6 '
      + 'the four boxes use.',
    depreciatingHeading: 'Skills more exposed to automation',
    appreciatingHeading: 'Skills AI could amplify',
    classSHeading: 'Skills AI can take over',
    classAHeading: 'Skills AI assists with',
    classMHeading: 'Skills machines can do',
    cardsHeading: 'Why each essential skill scores as it does',
    cardsKey: 'Open one to read it. The red number is automation risk, the blue one '
      + 'amplification, each out of 10.',
    geminiNote: 'Gemini wrote this for its own, older two-score run, not for the '
      + 'four-share scoring on this page.',
    storyHeading: 'How the work could change',
    storyNote: '',
    tasksHeading: 'What AI takes on, and what it amplifies',
    weekHeading: 'Your week, rebalanced',
    pathsHeading: 'Where you could move next',
    pathsHeadingNone: 'Nearby jobs, and how they differ',
    pathsIntro: 'Jobs that share skills with yours. Only the ones our two scores '
      + 'genuinely separate from your job are called a move.',
    needleHeading: 'What would move the needle',
    needleText: 'None of the jobs beside yours is a step up on the two scores, so the '
      + 'lever is the work itself rather than the job title. The skills below are the ones '
      + 'AI makes you most valuable at, net of what it can take over.',
    sidewaysHeading: 'Nearby jobs that sit about where yours does',
    sidewaysNote: 'These share skills with this job, but the two scores do not make '
      + 'them a step up. Each one says what actually differs.',
    learnHeading: 'Skills you could learn',
    learnIntro: 'Skills the nearby jobs need that your job does not list. Best first by '
      + 'net gain: how much better AI makes you at the skill, less how much of it AI can '
      + 'take over. Skills AI can take over are left out.',
    toggleLabel: 'Reading this for someone else? Switch to neutral wording',
  },
  other: {
    scoresHeading: 'How exposed this job is',
    scoresIntro: 'Two model estimates, each from 1 to 10. They describe the skills '
      + 'the job is built from, not the person doing it.',
    adviceHeading: 'What someone in this role could do next',
    adviceFallback: 'Start with the skills AI could amplify, further down this page: '
      + 'they are the parts of this job that get more valuable, not less. The skills '
      + 'nearby jobs need are listed at the end.',
    scatterHeading: 'Where this job’s skills sit',
    scatterIntro: 'One dot per skill in this job. The lines are the cut-off of 6 '
      + 'the four boxes use.',
    depreciatingHeading: 'Skills more exposed to automation',
    appreciatingHeading: 'Skills AI could amplify',
    classSHeading: 'Skills AI can take over',
    classAHeading: 'Skills AI assists with',
    classMHeading: 'Skills machines can do',
    cardsHeading: 'Why each essential skill scores as it does',
    cardsKey: 'Open one to read it. The red number is automation risk, the blue one '
      + 'amplification, each out of 10.',
    geminiNote: 'Gemini wrote this for its own, older two-score run, not for the '
      + 'four-share scoring on this page.',
    storyHeading: 'How the work could change',
    storyNote: 'The story below is written to the job holder, in the second person. '
      + 'It was generated once and is not rewritten here.',
    tasksHeading: 'What AI takes on, and what it amplifies',
    weekHeading: 'The working week, rebalanced',
    pathsHeading: 'Where someone in this role could move next',
    pathsHeadingNone: 'Nearby jobs, and how they differ',
    pathsIntro: 'Jobs that share skills with this one. Only the ones our two scores '
      + 'genuinely separate from it are called a move.',
    needleHeading: 'What would move the needle',
    needleText: 'None of the jobs beside this one is a step up on the two scores, so the '
      + 'lever is the work itself rather than the job title. The skills below are the ones '
      + 'AI makes the person most valuable at, net of what it can take over.',
    sidewaysHeading: 'Nearby jobs that sit about where this one does',
    sidewaysNote: 'These share skills with this job, but the two scores do not make '
      + 'them a step up. Each one says what actually differs.',
    learnHeading: 'Skills someone in this role could learn',
    learnIntro: 'Skills the nearby jobs need that this job does not list. Best first by '
      + 'net gain: how much better AI makes the person at the skill, less how much of it AI '
      + 'can take over. Skills AI can take over are left out.',
    toggleLabel: 'Switch back to “your job” wording',
  },
};

/**
 * What the shares scheme says instead. Only the sentences that would otherwise
 * describe two cut scores are here; everything else falls through to WORDING.
 */
export const SHARES_WORDING = {
  my: {
    scoresIntro: 'What the skills your job is built from are made of, as four shares of '
      + 'the whole job. They describe the skills, not you.',
    adviceFallback: 'Start with the skills AI assists with, further down this page: they '
      + 'are the parts of this job that get more valuable, not less. The skills nearby '
      + 'jobs need are listed at the end.',
    scatterIntro: 'One dot per skill in your job: how much of it AI could carry out '
      + 'itself across, how much better it gets with AI alongside up. Colour and shape '
      + 'say which of the four kinds of skill it is.',
    cardsKey: 'The class comes from the model’s own chances, shown on each row, and not '
      + 'from the three scores after them — so a skill can score above the middle and '
      + 'still stay human. Open one to read the explanation.',
    pathsIntro: 'Jobs that share skills with yours. A move is only called a move when '
      + 'its shares genuinely differ from your job’s; the rest are named as what they are.',
    sidewaysHeading: 'Other nearby jobs, and how they actually differ',
    sidewaysNote: 'These share skills with this job without being a step up from it. Each '
      + 'one says what actually differs.',
    needleText: 'None of the jobs beside yours splits its work differently enough to be a '
      + 'step up, so the lever is the work itself rather than the job title. The skills '
      + 'below are the ones AI assists you with rather than takes over.',
    learnIntro: 'Skills the nearby jobs need that your job does not list, best first by '
      + 'net gain: how much better AI makes you at the skill, less how much of it AI can '
      + 'carry out itself. The skills AI can take over are left out.',
  },
  other: {
    scoresIntro: 'What the skills the job is built from are made of, as four shares of '
      + 'the whole job. They describe the skills, not the person doing it.',
    adviceFallback: 'Start with the skills AI assists with, further down this page: they '
      + 'are the parts of this job that get more valuable, not less. The skills nearby '
      + 'jobs need are listed at the end.',
    scatterIntro: 'One dot per skill in this job: how much of it AI could carry out '
      + 'itself across, how much better it gets with AI alongside up. Colour and shape '
      + 'say which of the four kinds of skill it is.',
    cardsKey: 'The class comes from the model’s own chances, shown on each row, and not '
      + 'from the three scores after them — so a skill can score above the middle and '
      + 'still stay human. Open one to read the explanation.',
    pathsIntro: 'Jobs that share skills with this one. A move is only called a move when '
      + 'its shares genuinely differ from this job’s; the rest are named as what they are.',
    sidewaysHeading: 'Other nearby jobs, and how they actually differ',
    sidewaysNote: 'These share skills with this job without being a step up from it. Each '
      + 'one says what actually differs.',
    needleText: 'None of the jobs beside this one splits its work differently enough to be '
      + 'a step up, so the lever is the work itself rather than the job title. The skills '
      + 'below are the ones AI assists the person with rather than takes over.',
    learnIntro: 'Skills the nearby jobs need that this job does not list, best first by '
      + 'net gain: how much better AI makes the person at the skill, less how much of it AI '
      + 'can carry out itself. The skills AI can take over are left out.',
  },
};

/**
 * Turn whatever the hash carried into one of the two modes.
 * @param {*} value the raw `for=` value
 * @returns {'my'|'other'}
 */
export function normaliseFraming(value) {
  return value === 'other' ? 'other' : DEFAULT_FRAMING;
}

/**
 * The mode the switch control moves to.
 * @param {*} value current mode
 * @returns {'my'|'other'}
 */
export function otherFraming(value) {
  return normaliseFraming(value) === 'other' ? 'my' : 'other';
}

/**
 * True in neutral, someone-else's-job mode.
 * @param {*} value
 * @returns {boolean}
 */
export function isNeutral(value) {
  return normaliseFraming(value) === 'other';
}

/**
 * Whether this mode may write the slug to recently viewed. Reading a page about
 * someone else's job must not fill up the visitor's own history.
 * @param {*} value
 * @returns {boolean}
 */
export function recordsRecent(value) {
  return !isNeutral(value);
}

/**
 * The wording table for a mode, in the words of the active scheme.
 * @param {*} value the raw `for=` value
 * @param {string} [scheme] 'quadrants' (default) or 'shares'
 * @returns {Object<string, string>} a fresh object; the tables are never mutated
 */
export function wordingFor(value, scheme) {
  const mode = normaliseFraming(value);
  if (scheme !== SHARES) return WORDING[mode];
  return { ...WORDING[mode], ...SHARES_WORDING[mode] };
}

/**
 * The `for=` value to write into a hash: nothing at all in the default mode, so
 * a my-job link stays `job.html#slug`.
 * @param {*} value
 * @returns {string} '' or 'other'
 */
export function framingParam(value) {
  return isNeutral(value) ? 'other' : '';
}
