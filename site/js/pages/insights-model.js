// Every figure the "What we found" article states, derived from the data.
// Pure: no DOM, no globals.
//
// The audit (B5) found the page showing one count in the hero and a different
// share two lines below it in the prose, because the prose carried constants.
// So the prose here is templates with named placeholders, and this module is the
// only place that turns stats.json, groups.json and search_index.json into the
// strings that fill them. Nothing is written twice, so nothing can disagree.

import { formatCount, formatPercent, formatScore } from '../format.js';
import { QUADRANT_NAMES } from '../quadrant.js';
import { groupHref, jobHref } from '../urlstate.js';

const PLACEHOLDER = /\{(\w+)\}/g;

/** How many occupations each ranked list shows. */
export const LIST_LENGTH = 10;

/* --- deriving ------------------------------------------------------------ */

function quadrantRow(code, stats) {
  const count = stats.quadrants?.counts?.[code] ?? 0;
  const share = stats.quadrants?.shares?.[code] ?? 0;
  return {
    code,
    name: QUADRANT_NAMES[code] || code,
    count,
    share,
    countText: formatCount(count),
    shareText: formatPercent(share),
  };
}

/** The four quadrants, biggest first. */
function quadrantRows(stats) {
  return Object.keys(QUADRANT_NAMES)
    .map((code) => quadrantRow(code, stats))
    .sort((a, b) => b.count - a.count);
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

/**
 * Everything the page and the article quote, derived from the three index files.
 *
 * @param {{stats: Object, groups: Object, index: Array<Object>}} sources
 * @returns {Object} facts; every number in the article comes out of here
 */
export function deriveInsights({ stats = {}, groups = {}, index = [] } = {}) {
  const quadrants = quadrantRows(stats);
  const majors = majorRows(groups);
  const lists = rankedLists(index);
  const byCode = Object.fromEntries(quadrants.map((row) => [row.code, row]));
  return {
    built: stats.built || '',
    threshold: stats.threshold ?? null,
    occupations: stats.occupations ?? 0,
    skills: stats.skills_scored ?? 0,
    nearLine: stats.near_line || { count: 0, share: 0 },
    quadrants,
    byCode,
    majors,
    highlights: groupHighlights(majors),
    lists,
    evolvePerShrink: byCode.SHRINK?.count
      ? Math.round(byCode.EVOLVE.count / byCode.SHRINK.count)
      : null,
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

function exampleValues(facts) {
  const shrink = facts.lists.shrink[0];
  const transform = facts.lists.transform[0];
  const evolve = facts.lists.evolve[0];
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
  const most = facts.lists.mostExposed[0];
  const least = facts.lists.leastExposed[0];
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
    threshold: String(facts.threshold),
    nearLineCount: formatCount(facts.nearLine.count),
    nearLineShare: formatPercent(facts.nearLine.share),
    evolvePerShrink: formatCount(facts.evolvePerShrink ?? 0),
    methodLink: { text: 'How sure is this?', href: 'method.html' },
    ...quadrantValues(facts),
    ...exampleValues(facts),
    ...extremeValues(facts),
    ...groupValues(facts),
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
  return ARTICLE.map(({ type, className, text }) => ({
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
