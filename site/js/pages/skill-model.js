// Looking a skill up, and relating it to other skills. Pure: no DOM, no globals.
//
// The skill index is a flat list of 13,475 rows {id, t, a, m, ne, no}. ESCO
// phrases a skill as a verb ("manage budgets"), so the ranking is title-only —
// there are no alternative labels to fall back on, which is why the no-match
// state has to teach the wording instead of guessing at it.

import { fold } from '../search.js';

/** Match tiers, best first. Exposed so the UI can say why a row matched. */
export const SKILL_TIERS = { EXACT: 0, PREFIX: 1, WORD_PREFIX: 2, SUBSTRING: 3 };

/** Shown when nothing matched: the phrasing is the thing people get wrong. */
export const NO_MATCH_HINT =
  "Skills use ESCO wording — try 'manage budgets' rather than 'budgeting'.";

const WORD_BREAK = /[^a-z0-9]/;

function wordPrefix(title, query) {
  return title.split(WORD_BREAK).some((word) => word && word.startsWith(query));
}

/** The tier a folded title earns for a folded query, or null for no match. */
function tierOf(title, query) {
  if (title === query) return SKILL_TIERS.EXACT;
  if (title.startsWith(query)) return SKILL_TIERS.PREFIX;
  if (wordPrefix(title, query)) return SKILL_TIERS.WORD_PREFIX;
  return title.includes(query) ? SKILL_TIERS.SUBSTRING : null;
}

function compareMatches(a, b) {
  return a.tier - b.tier
    || a.title.length - b.title.length
    || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0);
}

/**
 * Rank skills for a query.
 *
 * Order: exact title, title prefix, a word inside the title, then substring.
 * Ties break by title length then alphabetically, so equal input always gives
 * the same order.
 *
 * @param {Array<{id: string, t: string}>} index skill_index.json rows
 * @param {string} query raw user input; empty or whitespace returns []
 * @param {number} [limit=12] pass Infinity for every match
 * @returns {Array<{row: Object, tier: number}>}
 */
export function rankSkills(index, query, limit = 12) {
  const folded = fold(query);
  if (!folded) return [];
  const matches = [];
  for (const row of index || []) {
    const title = fold(row.t);
    const tier = tierOf(title, folded);
    if (tier !== null) matches.push({ row, tier, title });
  }
  matches.sort(compareMatches);
  return matches.slice(0, limit).map(({ row, tier }) => ({ row, tier }));
}

/**
 * How many occupations list this skill at all.
 * @param {{ne?: number, no?: number}} row a skill_index.json row
 * @returns {number}
 */
export function usedIn(row) {
  return Number(row?.ne || 0) + Number(row?.no || 0);
}

/**
 * The occupation slugs of one skill_occupations.json entry, essential first.
 * @param {{e?: string[], o?: string[]}} entry
 * @returns {string[]} de-duplicated, order preserved
 */
export function occupationSlugs(entry) {
  return [...new Set([...(entry?.e || []), ...(entry?.o || [])])];
}

function countOverlap(entry, target) {
  let overlap = 0;
  for (const slug of occupationSlugs(entry)) {
    if (target.has(slug)) overlap += 1;
  }
  return overlap;
}

function byCountThenId(a, b) {
  return b.n - a.n || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
}

/**
 * The skills that most often turn up in the same occupations as this one.
 *
 * Counted client-side from skill_occupations.json: every other skill is scored
 * by how many of this skill's occupations it also appears in. The work is capped
 * by `maxOccupations` so a skill used in hundreds of jobs stays as cheap as one
 * used in five; the whole index is ~126,000 slug entries, a few milliseconds.
 *
 * @param {Object<string, {e?: string[], o?: string[]}>} skillOccupations
 * @param {string} id the skill to relate to
 * @param {{limit?: number, maxOccupations?: number}} [options]
 * @returns {Array<{id: string, n: number}>} most shared occupations first
 */
export function coOccurringSkills(skillOccupations, id, options = {}) {
  const { limit = 8, maxOccupations = 120 } = options;
  const target = new Set(occupationSlugs(skillOccupations?.[id]).slice(0, maxOccupations));
  if (!target.size) return [];
  const counts = [];
  for (const [other, entry] of Object.entries(skillOccupations)) {
    const overlap = other === id ? 0 : countOverlap(entry, target);
    if (overlap > 0) counts.push({ id: other, n: overlap });
  }
  counts.sort(byCountThenId);
  return counts.slice(0, limit);
}

/**
 * Resolve occupation slugs to search_index rows so titles and scores show.
 *
 * A slug with no row still comes back — as a row carrying the slug as its title
 * and no scores — so a link is never dropped silently.
 *
 * @param {string[]} slugs
 * @param {Map<string, Object>} bySlug search_index rows keyed by slug
 * @returns {Array<Object>} search_index-shaped rows
 */
export function resolveOccupations(slugs, bySlug) {
  return (slugs || []).map((slug) => bySlug.get(slug)
    || { t: slug, s: slug, c: '', mg: '', a: null, m: null, q: null });
}

/**
 * Both occupation lists of a skill, resolved and ready to render.
 * @param {{e?: string[], o?: string[]}} entry a skill_occupations.json entry
 * @param {Map<string, Object>} bySlug search_index rows keyed by slug
 * @returns {{essential: Array<Object>, optional: Array<Object>}}
 */
export function neededBy(entry, bySlug) {
  return {
    essential: resolveOccupations(entry?.e, bySlug),
    optional: resolveOccupations(entry?.o, bySlug),
  };
}
