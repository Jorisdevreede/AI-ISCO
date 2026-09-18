// My-job versus someone-else's-job wording for the job page. Pure: no DOM.
//
// The default is second person ("your job"). `&for=other` in the hash switches
// every heading and label on the page to neutral, third-person wording, and the
// page stops writing to recently viewed.
//
// The narratives themselves are written in the second person and cannot be
// rewritten client side (a third-person pass is a later increment), so neutral
// mode carries a one-line note saying so rather than pretending otherwise.

/** The two modes. 'my' is the default. */
export const FRAMINGS = ['my', 'other'];

/** What an absent or unknown `for=` means. */
export const DEFAULT_FRAMING = 'my';

/** The hash parameter that carries the mode. */
export const FRAMING_PARAM = 'for';

/**
 * Every wording the page varies. Both tables carry exactly the same keys, so a
 * missing translation is a test failure rather than an `undefined` on screen.
 */
export const WORDING = {
  my: {
    scoresHeading: 'How exposed your job is',
    scoresIntro: 'Two model estimates, each from 1 to 10. They describe the skills '
      + 'your job is built from, not you.',
    adviceHeading: 'What to do next',
    scatterHeading: 'Where your skills sit',
    scatterIntro: 'One dot per skill in your job. The lines are the cut-off of 6 '
      + 'the four boxes use.',
    depreciatingHeading: 'Skills more exposed to automation',
    appreciatingHeading: 'Skills AI could amplify',
    cardsHeading: 'Why each essential skill scores as it does',
    storyHeading: 'How the work could change',
    storyNote: '',
    tasksHeading: 'What AI takes on, and what it amplifies',
    weekHeading: 'Your week, rebalanced',
    pathsHeading: 'Where you could move next',
    pathsIntro: 'Jobs that share skills with yours. Only the ones our two scores '
      + 'genuinely separate from your job are called a move.',
    sidewaysHeading: 'Nearby jobs that sit about where yours does',
    learnHeading: 'Skills you could learn',
    learnIntro: 'Skills the nearby jobs need that your job does not list, most '
      + 'amplified first.',
    toggleLabel: 'Reading this for someone else? Switch to neutral wording',
  },
  other: {
    scoresHeading: 'How exposed this job is',
    scoresIntro: 'Two model estimates, each from 1 to 10. They describe the skills '
      + 'the job is built from, not the person doing it.',
    adviceHeading: 'What someone in this role could do next',
    scatterHeading: 'Where this job’s skills sit',
    scatterIntro: 'One dot per skill in this job. The lines are the cut-off of 6 '
      + 'the four boxes use.',
    depreciatingHeading: 'Skills more exposed to automation',
    appreciatingHeading: 'Skills AI could amplify',
    cardsHeading: 'Why each essential skill scores as it does',
    storyHeading: 'How the work could change',
    storyNote: 'The story below is written to the job holder, in the second person. '
      + 'It was generated once and is not rewritten here.',
    tasksHeading: 'What AI takes on, and what it amplifies',
    weekHeading: 'The working week, rebalanced',
    pathsHeading: 'Where someone in this role could move next',
    pathsIntro: 'Jobs that share skills with this one. Only the ones our two scores '
      + 'genuinely separate from it are called a move.',
    sidewaysHeading: 'Nearby jobs that sit about where this one does',
    learnHeading: 'Skills someone in this role could learn',
    learnIntro: 'Skills the nearby jobs need that this job does not list, most '
      + 'amplified first.',
    toggleLabel: 'Switch back to “your job” wording',
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
 * The wording table for a mode.
 * @param {*} value
 * @returns {Object<string, string>}
 */
export function wordingFor(value) {
  return WORDING[normaliseFraming(value)];
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
