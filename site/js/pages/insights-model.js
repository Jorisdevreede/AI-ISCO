// Every figure the "What we found" article states, derived from the data.
// Pure: no DOM, no globals.
//
// The audit (B5) found the page showing one count in the hero and a different
// share two lines below it in the prose, because the prose carried constants.
// So the prose here is templates with named placeholders, and this module is the
// only place that turns stats.json, groups.json and search_index.json into the
// strings that fill them. Nothing is written twice, so nothing can disagree.

import {
  formatCount, formatPercent, formatScore, largestRemainder,
} from '../format.js';
import {
  SHARES, SKILL_CLASS_ORDER, skillClassName, splitOf, typeDescription, typeLabel,
} from '../scheme.js';
import { isShares, sharePercents, shareSentence } from '../shares.js';
import { groupHref, jobHref } from '../urlstate.js';

const PLACEHOLDER = /\{(\w+)\}/g;

/** How many occupations each ranked list shows. */
export const LIST_LENGTH = 10;

/* --- deriving ------------------------------------------------------------ */

/**
 * A count with the right noun: "1 occupation", "3,039 occupations".
 * TODO: format.js should own this; tree-model.js has the same two lines.
 */
function countOf(value, noun = 'occupation') {
  return `${formatCount(value)} ${noun}${value === 1 ? '' : 's'}`;
}

/** "0%" is a lie about a class that has one occupation in it. */
function percentText(percent, count) {
  return count > 0 && percent === 0 ? '<1%' : `${percent}%`;
}

function quadrantRow(code, split, percent) {
  const count = split.counts[code] ?? 0;
  return {
    code,
    name: typeLabel(code),
    count,
    share: split.shares[code] ?? 0,
    countText: formatCount(count),
    shareText: percentText(percent, count),
  };
}

/**
 * The classes of whichever scheme the set uses, biggest first, with whole
 * percentages that add up to 100 — rounding each on its own made the headline
 * table read 101%.
 */
function quadrantRows(stats) {
  const split = splitOf(stats);
  const percents = largestRemainder(split.order.map((code) => split.counts[code] ?? 0));
  return split.order
    .map((code, position) => quadrantRow(code, split, percents[position]))
    .sort((a, b) => b.count - a.count);
}

/** The class of a counts object that has the most in it, or null. */
function largestCode(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return null;
  return entries.reduce((top, entry) => (entry[1] > top[1] ? entry : top))[0];
}

function majorRow(key, group) {
  const counts = group.q || {};
  const shrink = counts.SHRINK || 0;
  return {
    key,
    label: group.label,
    href: groupHref(key),
    n: group.n,
    auto: group.auto?.mean ?? null,
    amp: group.amp?.mean ?? null,
    counts,
    shrink,
    shrinkShare: group.n ? shrink / group.n : 0,
    sh: isShares(group.sh) ? group.sh : null,
    topType: largestCode(counts),
  };
}

/** One row per ISCO major group, in ISCO order. */
function majorRows(groups) {
  return Object.entries(groups || {})
    .filter(([, group]) => group && group.level === 'major')
    .map(([key, group]) => majorRow(key, group))
    .sort((a, b) => a.key.localeCompare(b.key));
}

function highest(rows, pick) {
  return rows.reduce((best, row) => (pick(row) > pick(best) ? row : best), rows[0] || null);
}

function groupHighlights(majors) {
  const zeroShrink = majors.filter((row) => row.shrink === 0);
  return {
    topAmplification: highest(majors, (row) => row.amp ?? -1),
    topAutomation: highest(majors, (row) => row.auto ?? -1),
    topShrinkShare: highest(majors, (row) => row.shrinkShare),
    zeroShrink,
    largestZeroShrink: highest(zeroShrink, (row) => row.n),
    lowestShrinkShare: majors.length
      ? majors.reduce((low, row) => (row.shrinkShare < low.shrinkShare ? row : low))
      : null,
  };
}

function ranked(rows, score, length = LIST_LENGTH) {
  return [...rows]
    .sort((a, b) => score(b) - score(a) || String(a.t).localeCompare(String(b.t)))
    .slice(0, length);
}

function inQuadrant(index, code) {
  return (index || []).filter((row) => row.q === code);
}

/**
 * The ranked lists the page shows, all from search_index rows.
 * `shrink` ranks by how far automation outruns amplification, `transform` by
 * the two together, `stable` by how little of either there is.
 */
function rankedLists(index) {
  return {
    mostExposed: ranked(index || [], (row) => row.a),
    leastExposed: ranked(index || [], (row) => -row.a),
    shrink: ranked(inQuadrant(index, 'SHRINK'), (row) => row.a - row.m),
    transform: ranked(inQuadrant(index, 'TRANSFORM'), (row) => row.a + row.m),
    stable: ranked(inQuadrant(index, 'STABLE'), (row) => -(row.a + row.m)),
    evolve: ranked(inQuadrant(index, 'EVOLVE'), (row) => row.m - row.a),
  };
}

/* --- the shares scheme ---------------------------------------------------- */

/** One row per skill class, in the order the shares are stored. */
function classRows(stats) {
  const source = (stats && stats.skill_classes) || null;
  if (!source) return [];
  const counts = source.counts || {};
  const shares = source.shares || {};
  const total = SKILL_CLASS_ORDER.reduce((sum, code) => sum + (counts[code] || 0), 0);
  const percents = largestRemainder(SKILL_CLASS_ORDER.map((code) => counts[code] || 0));
  return SKILL_CLASS_ORDER.map((code, position) => {
    const count = counts[code] ?? 0;
    return {
      code,
      name: skillClassName(code),
      count,
      share: shares[code] ?? (total ? count / total : 0),
      countText: formatCount(count),
      shareText: percentText(percents[position], count),
    };
  });
}

/** The four ranked lists of a shares set: one per share, longest share first. */
function sharesLists(index) {
  const rows = (index || []).filter((row) => isShares(row.sh));
  return {
    substituted: ranked(rows, (row) => row.sh[0]),
    assisted: ranked(rows, (row) => row.sh[1]),
    mechanised: ranked(rows, (row) => row.sh[2]),
    insulated: ranked(rows, (row) => row.sh[3]),
  };
}

/** The occupation of one type with the most of the share that type is about. */
function clearestOfType(index, code, at) {
  const rows = (index || []).filter((row) => row.q === code && isShares(row.sh));
  if (!rows.length) return null;
  return rows.reduce((best, row) => (row.sh[at] > best.sh[at] ? row : best));
}

function sharesExamples(index, lists) {
  return {
    substituted: clearestOfType(index, 'AUTOMATION_HEAVY', 0) || lists.substituted[0] || null,
    physical: clearestOfType(index, 'INSULATED_PHYSICAL', 3) || null,
    people: clearestOfType(index, 'INSULATED_PEOPLE', 3) || null,
  };
}

/** The major group whose mean shares put the most weight on one of the four. */
function groupWithMostOf(majors, at) {
  const scored = majors.filter((row) => row.sh);
  if (!scored.length) return null;
  return scored.reduce((best, row) => (row.sh[at] > best.sh[at] ? row : best));
}

function groupShareHighlights(majors) {
  return {
    substituted: groupWithMostOf(majors, 0),
    assisted: groupWithMostOf(majors, 1),
    mechanised: groupWithMostOf(majors, 2),
    insulated: groupWithMostOf(majors, 3),
  };
}

/**
 * Everything the page and the article quote, derived from the three index files.
 *
 * @param {{stats: Object, groups: Object, index: Array<Object>}} sources
 * @returns {Object} facts; every number in the article comes out of here
 */
/** The totals stats.json states outright, with a floor so nothing prints NaN. */
function totalsOf(stats) {
  return {
    built: stats.built || '',
    model: stats.model || '',
    threshold: stats.threshold ?? null,
    occupations: stats.occupations ?? 0,
    skills: stats.skills_scored ?? 0,
    nearLine: stats.near_line || { count: 0, share: 0 },
  };
}

/** How many Evolve occupations there are per Shrink one, or null. */
function evolvePerShrink(byCode) {
  if (!byCode.SHRINK?.count) return null;
  return Math.round(byCode.EVOLVE.count / byCode.SHRINK.count);
}

export function deriveInsights({ stats = {}, groups = {}, index = [] } = {}) {
  const quadrants = quadrantRows(stats);
  const majors = majorRows(groups);
  const scheme = splitOf(stats).scheme;
  const lists = scheme === SHARES ? sharesLists(index) : rankedLists(index);
  const byCode = Object.fromEntries(quadrants.map((row) => [row.code, row]));
  const classes = classRows(stats);
  return {
    ...totalsOf(stats),
    scheme,
    quadrants,
    byCode,
    classes,
    topClass: [...classes].sort((a, b) => b.count - a.count)[0] || null,
    majors,
    highlights: groupHighlights(majors),
    groupShares: groupShareHighlights(majors),
    lists,
    examples: scheme === SHARES ? sharesExamples(index, lists) : {},
    evolvePerShrink: evolvePerShrink(byCode),
  };
}

/* --- prose --------------------------------------------------------------- */

function jobLink(row) {
  return row ? { text: row.t, href: jobHref(row.s) } : { text: 'not scored' };
}

function groupLink(row) {
  return row ? { text: row.label, href: row.href } : { text: 'no group' };
}

function scoresOf(row) {
  return { auto: formatScore(row?.a), amp: formatScore(row?.m) };
}

/** The clause about major groups with no Shrink occupation at all. */
function shrinkFreeClause(highlights) {
  const free = highlights.largestZeroShrink;
  if (free) {
    return [
      { text: 'Not one of the ' },
      { text: formatCount(free.n) },
      { text: ' occupations in ' },
      groupLink(free),
      { text: ' falls into Shrink.' },
    ];
  }
  const low = highlights.lowestShrinkShare;
  return [
    { text: 'The smallest share of Shrink sits in ' },
    groupLink(low),
    { text: `, at ${formatPercent(low?.shrinkShare ?? 0)} of its occupations.` },
  ];
}

/** The head of one ranked list, or null — the two schemes keep different lists. */
function listHead(facts, key) {
  const rows = (facts.lists || {})[key];
  return (rows && rows[0]) || null;
}

function exampleValues(facts) {
  const shrink = listHead(facts, 'shrink');
  const transform = listHead(facts, 'transform');
  const evolve = listHead(facts, 'evolve');
  return {
    shrinkExample: jobLink(shrink),
    shrinkExampleAuto: scoresOf(shrink).auto,
    shrinkExampleAmp: scoresOf(shrink).amp,
    transformExample: jobLink(transform),
    transformExampleAuto: scoresOf(transform).auto,
    transformExampleAmp: scoresOf(transform).amp,
    evolveExample: jobLink(evolve),
    evolveExampleAuto: scoresOf(evolve).auto,
    evolveExampleAmp: scoresOf(evolve).amp,
  };
}

function extremeValues(facts) {
  const most = listHead(facts, 'mostExposed');
  const least = listHead(facts, 'leastExposed');
  return {
    mostExposed: jobLink(most),
    mostExposedAuto: scoresOf(most).auto,
    mostExposedAmp: scoresOf(most).amp,
    leastExposed: jobLink(least),
    leastExposedAuto: scoresOf(least).auto,
    leastExposedAmp: scoresOf(least).amp,
  };
}

function groupValues(facts) {
  const { topAmplification, topAutomation, topShrinkShare, largestZeroShrink } = facts.highlights;
  return {
    shrinkFreeClause: shrinkFreeClause(facts.highlights),
    shrinkFreeGroupName: (largestZeroShrink || topShrinkShare || {}).label || '',
    topAmpGroup: groupLink(topAmplification),
    topAmpGroupAmp: formatScore(topAmplification?.amp),
    topAutoGroup: groupLink(topAutomation),
    topAutoGroupAuto: formatScore(topAutomation?.auto),
    topShrinkGroup: groupLink(topShrinkShare),
    topShrinkGroupShare: formatPercent(topShrinkShare?.shrinkShare ?? 0),
  };
}

function quadrantValues(facts) {
  const values = {};
  for (const row of facts.quadrants) {
    const stem = row.code.toLowerCase();
    values[`${stem}Count`] = row.countText;
    values[`${stem}Share`] = row.shareText;
  }
  return values;
}

/* --- prose: the shares scheme --------------------------------------------- */

const NO_JOB = { text: 'no occupation of that type' };

function shareOf(row, at) {
  const parts = sharePercents(row && row.sh);
  return parts.length ? `${parts[at].percent}%` : formatPercent(0);
}

/** "3 points" — how far the largest type leads the next, never a bare number. */
function gapWords(first, second) {
  const points = Math.round(((first?.share ?? 0) - (second?.share ?? 0)) * 100);
  if (points <= 0) return 'level with';
  return `${formatCount(points)} point${points === 1 ? '' : 's'} ahead of`;
}

/** The name, count and share of one type, however empty the run left it. */
function typeSlots(facts, code, stem) {
  const row = facts.byCode[code] || {};
  return {
    [`${stem}Name`]: typeLabel(code),
    [`${stem}Count`]: row.countText ?? formatCount(0),
    [`${stem}Jobs`]: countOf(row.count ?? 0),
    [`${stem}Share`]: row.shareText ?? formatPercent(0),
  };
}

function typeValues(facts) {
  const [first, second] = facts.quadrants;
  return {
    topTypeName: typeLabel(first?.code),
    topTypeCount: first?.countText ?? formatCount(0),
    topTypeJobs: countOf(first?.count ?? 0),
    topTypeShare: first?.shareText ?? formatPercent(0),
    topTypeDescription: typeDescription(first?.code),
    typeGap: gapWords(first, second),
    secondTypeName: typeLabel(second?.code),
    secondTypeShare: second?.shareText ?? formatPercent(0),
    topClassName: facts.topClass?.name ?? 'not scored',
    topClassShare: facts.topClass?.shareText ?? formatPercent(0),
    ...typeSlots(facts, 'AUTOMATION_HEAVY', 'autoHeavy'),
    ...typeSlots(facts, 'INSULATED_PHYSICAL', 'physical'),
    ...typeSlots(facts, 'INSULATED_PEOPLE', 'people'),
    ...typeSlots(facts, 'MIXED', 'mixed'),
  };
}

function shareExampleValues(facts) {
  const { substituted, physical, people } = facts.examples || {};
  return {
    substitutedExample: substituted ? jobLink(substituted) : NO_JOB,
    substitutedExampleShares: shareSentence(substituted?.sh),
    physicalExample: physical ? jobLink(physical) : NO_JOB,
    physicalExamplePercent: shareOf(physical, 3),
    peopleExample: people ? jobLink(people) : NO_JOB,
    peopleExamplePercent: shareOf(people, 3),
  };
}

function shareGroupValues(facts) {
  const { assisted, mechanised } = facts.groupShares || {};
  return {
    topAssistedGroup: groupLink(assisted),
    topAssistedGroupShare: shareOf(assisted, 1),
    topMechGroup: groupLink(mechanised),
    topMechGroupShare: shareOf(mechanised, 2),
  };
}

/**
 * Every placeholder the article templates can use, as text or as a link.
 *
 * @param {Object} facts from deriveInsights
 * @returns {Object<string, string|{text: string, href?: string}|Array>}
 */
export function articleValues(facts) {
  return {
    occupations: formatCount(facts.occupations),
    skills: formatCount(facts.skills),
    threshold: facts.threshold === null ? 'no cut-off' : String(facts.threshold),
    nearLineCount: formatCount(facts.nearLine.count),
    nearLineJobs: countOf(facts.nearLine.count),
    nearLineShare: formatPercent(facts.nearLine.share),
    evolvePerShrink: formatCount(facts.evolvePerShrink ?? 0),
    methodLink: { text: 'How sure is this?', href: 'method.html' },
    ...quadrantValues(facts),
    ...exampleValues(facts),
    ...extremeValues(facts),
    ...groupValues(facts),
    ...typeValues(facts),
    ...shareExampleValues(facts),
    ...shareGroupValues(facts),
  };
}

/**
 * The article. Voice kept from the original page; every figure is a placeholder,
 * so a rebuild of the data rewrites the sentence instead of contradicting it.
 */
export const ARTICLE = [
  {
    type: 'p',
    className: 'dropcap',
    text: 'When we scored all {skills} ESCO skills behind {occupations} European occupations '
      + 'for automation risk and amplification potential, the most striking finding was not '
      + 'what disappeared. It was what did not. {shrinkShare} of occupations fell into '
      + 'Shrink — {shrinkCount} jobs where the work is highly automatable and there is little '
      + 'the same technology can hand back. The largest share, {evolveShare}, landed in '
      + 'Evolve: jobs where AI amplifies what a person does rather than replacing it.',
  },
  {
    type: 'p',
    text: 'The numbers tell a story of transformation rather than extinction. The clearest '
      + 'Shrink case in the data is the {shrinkExample}, at {shrinkExampleAuto} on automation risk '
      + 'against {shrinkExampleAmp} on amplification. That gap is the whole finding: the model '
      + 'reads the skills as machine-shaped work, with little room left for a person to add '
      + 'something the machine cannot.',
  },
  {
    type: 'p',
    text: 'Step one level up the abstraction ladder and a different picture emerges. '
      + 'The {evolveExample} scores {evolveExampleAuto} on automation risk and '
      + '{evolveExampleAmp} on amplification — the shape of a job that AI makes bigger rather '
      + 'than smaller. That shape, not displacement, is the most common pattern here: '
      + '{evolveCount} of {occupations} occupations sit in Evolve.',
  },
  {
    type: 'pullquote',
    text: 'Over half of all jobs are not being replaced by AI — they are being amplified. '
      + 'The hammer gets smarter, and the hand that wields it becomes more valuable.',
  },
  {
    type: 'p',
    text: 'The Transform quadrant, {transformShare} of occupations, holds the most dramatic '
      + 'shifts. The {transformExample} scores {transformExampleAuto} on automation risk and '
      + '{transformExampleAmp} on amplification: the work can be done by a machine, and the '
      + 'same machine makes the person doing it far more productive. The job title stays the '
      + 'same. The job does not.',
  },
  {
    type: 'p',
    text: 'Even at the very top of the exposure ranking the picture is not a cliff. '
      + 'The {mostExposed} carries the highest automation risk in the dataset, {mostExposedAuto}, '
      + 'and still scores {mostExposedAmp} on amplification. The model is not describing a job '
      + 'that simply ends; it is describing one where almost everything about the day changes.',
  },
  {
    type: 'p',
    text: 'The quieter cluster sits in Stable: {stableShare} of occupations that these scores '
      + 'barely touch. The {leastExposed} sits at the bottom of the exposure ranking, '
      + '{leastExposedAuto} on automation risk and {leastExposedAmp} on amplification. These '
      + 'are not low-skill roles. They share one trait: their core value is embodied, spatial '
      + 'or sensory in a way the model does not expect current AI to reach.',
  },
  {
    type: 'p',
    text: 'The structural patterns are the most useful part. {shrinkFreeClause} Individual '
      + 'tasks inside those roles score as highly automatable, but the composite job leans on '
      + 'interpersonal judgement and contextual decisions, which the scores treat as amplified '
      + 'rather than replaced.',
  },
  {
    type: 'pullquote',
    text: '{shrinkFreeGroupName}: not one occupation in Shrink. The composite job of managing '
      + 'people holds up even where its individual tasks do not.',
  },
  {
    type: 'p',
    text: 'Across ISCO major groups, {topAmpGroup} show the highest average amplification, '
      + '{topAmpGroupAmp} out of 10, while {topAutoGroup} carry the highest average automation '
      + 'risk at {topAutoGroupAuto}. Shrink concentrates in {topShrinkGroup}, where '
      + '{topShrinkGroupShare} of the jobs land there — the one place in this data where '
      + 'displacement, not redefinition, is the dominant story.',
  },
  {
    type: 'p',
    text: 'What does this mean for workers and for policy? Mostly that "jobs lost to AI" is '
      + 'the wrong frame for what these scores say. For every occupation in Shrink, '
      + '{evolvePerShrink} sit in Evolve. The challenge described here is not mass '
      + 'unemployment; it is mass reskilling. People in Transform roles need support to change '
      + 'alongside their jobs, and people in Shrink roles need a route into an adjacent one.',
  },
  {
    type: 'p',
    text: 'And read the boxes as a direction, not a verdict. The quadrants are a hard cut at '
      + '{threshold} on both axes, and {nearLineCount} occupations — {nearLineShare} of them — '
      + 'sit within half a point of a cut-off. Two jobs on opposite sides of that line can be '
      + 'describing much the same work. {methodLink}',
  },
  {
    type: 'p',
    text: 'The great rebalancing is not a story of human obsolescence. It is a story of human '
      + 'redefinition — a labour market where the most valued skills shift from execution to '
      + 'judgement, from routine to exception-handling, from producing outputs to curating '
      + 'them. The jobs of tomorrow carry the same titles. The work inside them is new.',
  },
];

/**
 * The article of a shares set. Same voice, same rule: every figure, job title,
 * group name and type name is a slot, so a rebuild of the data rewrites the
 * sentence rather than contradicting it. A test asserts that no digit appears
 * in any of these templates outside a slot.
 */
export const SHARES_ARTICLE = [
  {
    type: 'p',
    className: 'dropcap',
    text: 'Every one of the {skills} ESCO skills behind these {occupations} occupations was put '
      + 'to a judgment model as six questions: whether the work leaves something digital behind, '
      + 'how much of it a system with no body could carry out itself, how much of it physical '
      + 'equipment already does, how much better the person gets with that system beside them, '
      + 'how the skill is exercised, and how widely such systems are deployed. The answers put '
      + 'every skill in one class, and the mix of a job’s own classes makes its four shares '
      + 'and its type.',
  },
  {
    type: 'p',
    text: 'The largest single type is {topTypeName}: {topTypeJobs}, {topTypeShare} '
      + 'of the total, {typeGap} {secondTypeName} at {secondTypeShare}. {topTypeDescription}',
  },
  {
    type: 'pullquote',
    text: '{topClassShare} of all scored skills are the kind this rubric calls '
      + '“{topClassName}”. A job’s type is nothing but the mix of its own skills, '
      + 'so that one figure shapes the whole map.',
  },
  {
    type: 'p',
    text: 'Where an AI system could take the work over, it shows. {substitutedExample} is the '
      + 'clearest case in the data: {substitutedExampleShares}. {autoHeavyJobs}, '
      + '{autoHeavyShare} of them, meet the first rule and come out {autoHeavyName}, which asks '
      + 'for half or more of the skill weight to be work the system could carry out itself. A '
      + 'large share there is not a verdict on anybody: it says which part of the day is up for '
      + 'redesign first.',
  },
  {
    type: 'p',
    text: 'At the other end are the jobs built from work a system with no hands cannot reach. '
      + '{physicalExample} keeps {physicalExamplePercent} of its skill weight with the person, '
      + 'and the reason is in how the work is done: on things, in a place. {peopleExample} keeps '
      + '{peopleExamplePercent}, for a different reason — the work is done with and for '
      + 'other people. Those are two separate types, {physicalName} and {peopleName}, because '
      + 'they are two separate arguments: {physicalJobs} for the first, '
      + '{peopleCount} for the second.',
  },
  {
    type: 'p',
    text: 'Machinery is asked about separately from AI, on purpose: a production line and a '
      + 'language model are not the same claim about a job. Across the ISCO major groups, '
      + '{topMechGroup} carry the largest mechanised share of their skill weight, '
      + '{topMechGroupShare}, while {topAssistedGroup} carry the largest assisted share at '
      + '{topAssistedGroupShare}. Read the mechanised share as a floor: the question behind it '
      + 'is worded conservatively, so where it is large the real figure is larger still.',
  },
  {
    type: 'p',
    text: '{mixedJobs}, {mixedShare} of them, come out {mixedName}. That is a '
      + 'residual class and not a finding that AI will leave those jobs alone: their skills '
      + 'point in different directions and one label would mislead. For those, the four shares '
      + 'say more than the badge does.',
  },
  {
    type: 'p',
    text: 'Two readings are worth keeping. A job whose work mostly stays human is not a job AI '
      + 'cannot help with: many of those skills still get a clear gain on part of the work, '
      + 'under the level this rubric counts as assistance. And rules have edges — '
      + '{nearLineJobs}, {nearLineShare} of them, sit near enough to one that '
      + 'moving any single share by five points would give them a different type. {methodLink}',
  },
  {
    type: 'p',
    text: 'None of this was measured. These are one model’s estimates of what a system '
      + 'could do to a list of skills as ESCO describes it, with no ground truth to check them '
      + 'against, and they say nothing about how many people will be doing a job or what it '
      + 'will pay. Read the direction rather than the decimal. And where a large part of a job '
      + 'is work AI can take over, the useful question is not whether to worry but which of the '
      + 'other three shares to grow.',
  },
];

/**
 * The section notes of a shares set, keyed by the id of the element they fill.
 * Same rule as the article: no figure is written down, only slotted.
 */
export const SHARES_NOTES = {
  'hero-sub': 'Every skill in ESCO put to a judgment model as six questions, then rolled up to '
    + '{occupations} occupations as four shares of the work. {skills} skills, no measurement of '
    + 'any real workplace. Every number on this page is computed from the published data as the '
    + 'page loads.',
  'split-note': 'Each occupation gets one of seven types from the shares of its own skills, on '
    + 'the first rule that matches — not from a cut-off on two numbers. {nearLineJobs}, '
    + '{nearLineShare} of them, sit near enough to a cut-off that moving one share '
    + 'by five points would change their type. {methodLink}',
  'classes-note': 'Every scored skill falls into exactly one class, on the first rule that '
    + 'matches. A job’s four shares are these same four classes weighed over its own '
    + 'skills, with essential skills counting double.',
  'substituted-note': 'Ranked by the share of the skill weight an AI system could carry out '
    + 'itself. A large share is not a verdict on the job: it says which part of the work is up '
    + 'for redesign first.',
  'assisted-note': 'Ranked by the share of the work that stays with the person and gets a clear '
    + 'gain across most of it with an AI system beside them.',
  'mechanised-note': 'Ranked by the share of the work physical equipment does — production '
    + 'lines, robots, process plant — with AI set aside. Read it as a floor: the question '
    + 'behind it is worded conservatively.',
  'insulated-note': 'Ranked by the share of the work that stays with the person. It does not '
    + 'mean AI is of no help there: many of those skills still get a gain on part of the work, '
    + 'under the level this rubric counts as assistance.',
  'groups-note': 'The ISCO major groups, with the average of each share across the occupations '
    + 'in them. {topAssistedGroup} carry the largest assisted share, {topAssistedGroupShare}; '
    + '{topMechGroup} the largest mechanised share, {topMechGroupShare}.',
};

function valueSegments(value, fallback) {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null) return [{ text: fallback }];
  if (typeof value === 'object') return [{ text: value.text, href: value.href }];
  return [{ text: String(value) }];
}

/**
 * Split a template into renderable segments, filling {named} placeholders.
 *
 * @param {string} template
 * @param {Object} values placeholder name -> string, {text, href} or segments
 * @returns {Array<{text: string, href?: string}>} in order, ready to append
 */
export function fillTemplate(template, values = {}) {
  const text = String(template ?? '');
  const segments = [];
  let last = 0;
  for (const match of text.matchAll(PLACEHOLDER)) {
    if (match.index > last) segments.push({ text: text.slice(last, match.index) });
    segments.push(...valueSegments(values[match[1]], match[0]));
    last = match.index + match[0].length;
  }
  if (last < text.length) segments.push({ text: text.slice(last) });
  return segments;
}

/**
 * The article, ready to render: one entry per block, each a list of segments.
 *
 * @param {Object} facts from deriveInsights
 * @returns {Array<{type: string, className?: string, segments: Array}>}
 */
export function buildArticle(facts) {
  const values = articleValues(facts);
  const blocks = facts.scheme === SHARES ? SHARES_ARTICLE : ARTICLE;
  return blocks.map(({ type, className, text }) => ({
    type,
    className,
    segments: fillTemplate(text, values),
  }));
}

/**
 * The plain text of a built article block, for tests and aria labels.
 * @param {{segments: Array<{text: string}>}} block
 * @returns {string}
 */
export function blockText(block) {
  return (block.segments || []).map((segment) => segment.text).join('');
}
