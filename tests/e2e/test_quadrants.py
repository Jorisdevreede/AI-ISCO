"""`?scorer=gemini`: the older scheme, still whole.

The rest of the suite follows the default — four shares and seven types. This
module is the compact proof that the set the switch offers beside it has not
rotted: two 1-10 scores, a hard cut at 6, four boxes, and nowhere a shares bar
that the older data could not fill.

One or two checks per page. The skill page's half of this lives in
test_skill.py, beside the rubric sections it has to do without.
"""

import pytest
from playwright.sync_api import expect

from .conftest import GEMINI, QUADRANT_NAMES

SLUG = "software-developer"
GROUP = "unit:2512"
QUADRANT_ORDER = ["TRANSFORM", "STABLE", "EVOLVE", "SHRINK"]


def at(path):
    """The same path under the older score set."""
    head, _, hash_part = path.partition("#")
    return f"{head}{GEMINI}" + (f"#{hash_part}" if hash_part else "")


def group(open_ready, session, view, ready):
    """One view of the sample group under the older set. `open_group` builds a
    plain URL, which would land on the default."""
    return open_ready(session, at(f"groups.html#g={GROUP}&view={view}"), ready)


# --- the landing page ------------------------------------------------------


def test_the_landing_page_talks_about_two_scores(desktop, open_ready, by_slug):
    page = open_ready(desktop, at("index.html"), "#chip-row a")
    subtitle = page.locator("#subtitle").inner_text()
    assert "automate" in subtitle and "amplify" in subtitle
    assert "four shares" not in subtitle

    page.get_by_label("Find your job title").fill("programmer")
    page.locator("#job-listbox li").first.wait_for()
    meta = page.locator("#job-listbox li").first.locator(".meta").inner_text()
    assert meta.endswith(QUADRANT_NAMES[by_slug[SLUG]["q"]])
    assert "AI can take over" not in meta


# --- the job page ----------------------------------------------------------


def test_the_job_page_shows_two_scores_a_box_and_no_shares(
        desktop, open_job, by_slug, stats_json):
    page = open_job(desktop, at(f"job.html#{SLUG}"))
    row = by_slug[SLUG]

    assert page.locator(".score-card--auto .score-value").inner_text() == f"{row['a']:.1f}"
    assert page.locator(".score-card--amp .score-value").inner_text() == f"{row['m']:.1f}"
    expect(page.locator(".score-card--mech")).to_have_count(0)

    badge = page.locator("#job-meta .quadrant-badge")
    assert badge.get_attribute("data-quadrant") == row["q"]
    assert badge.locator(".quadrant-badge-name").inner_text() == QUADRANT_NAMES[row["q"]]

    assert not page.locator("#job-shares").is_visible()
    expect(page.locator(".shares-bar")).to_have_count(0)
    assert f"cut-off of {stats_json['threshold']}" in (
        page.locator("#scatter-threshold").inner_text())
    assert desktop.unexpected_errors() == []


def test_the_job_page_cuts_its_skill_lists_at_the_threshold(
        desktop, open_job, stats_json, units_json):
    """Two lists under quadrants, both named for a score and both cut at 6 —
    not one list per class."""
    page = open_job(desktop, at(f"job.html#{SLUG}"))
    lists = page.locator("#skill-columns section")
    expect(lists).to_have_count(2)
    headings = lists.locator("h2").all_inner_texts()
    assert headings == ["Skills more exposed to automation", "Skills AI could amplify"]
    assert f"score {stats_json['threshold']} or more" in (
        page.locator("#dep-count").inner_text())
    assert page.locator("#skill-columns .class-chip").count() == 0
    assert units_json[SLUG]


# --- the group overview ----------------------------------------------------


def test_a_group_splits_over_four_boxes(desktop, open_ready, groups_json):
    page = group(open_ready, desktop, "table", "#panel tbody tr")
    counts = groups_json[GROUP]["q"]
    total = sum(counts.get(code, 0) for code in QUADRANT_ORDER)

    assert page.locator("#mix-heading").inner_text() == "How this group splits"
    rows = page.locator("#mix-bars li")
    expect(rows).to_have_count(len(QUADRANT_ORDER))
    assert rows.locator(".mix-name").all_inner_texts() == [
        QUADRANT_NAMES[code] for code in QUADRANT_ORDER]
    for position, code in enumerate(QUADRANT_ORDER):
        assert rows.nth(position).get_attribute("data-quadrant") == code
        assert rows.nth(position).locator(".mix-figure").inner_text().startswith(
            f"{counts.get(code, 0):,} of {total:,} jobs")
    assert not page.locator("#mean-shares").is_visible()


def test_the_group_scatter_still_draws_the_cut_off(desktop, open_ready, stats_json):
    page = group(open_ready, desktop, "scatter", "#panel svg.scatter")
    dots = page.locator("#panel svg.scatter circle.dot-point")
    assert dots.count() > 0
    assert dots.first.get_attribute("data-quadrant")
    expect(page.locator("#panel svg.scatter use.dot-point")).to_have_count(0)
    assert page.locator("#panel svg.scatter line.cut-line").count() == 2
    corners = page.locator("#panel svg.scatter text.quad-text").all_text_contents()
    assert sorted(corners) == sorted(QUADRANT_ORDER)
    assert str(stats_json["threshold"]) in (
        page.locator("#panel svg.scatter text.axis-text").all_text_contents())


# --- the tree --------------------------------------------------------------


def test_a_tree_leaf_carries_a_quadrant_and_the_pair_of_scores(desktop, by_slug):
    page = desktop.open(at(f"tree.html#job={SLUG}"))
    leaf = page.locator(f'[data-id="job:{SLUG}"]')
    leaf.wait_for()
    row = by_slug[SLUG]
    assert leaf.locator(".tree-quad").get_attribute("data-quadrant") == row["q"]
    assert leaf.locator(".tree-quad").inner_text() == QUADRANT_NAMES[row["q"]]
    assert leaf.locator(".tree-scores").inner_text() == f"{row['a']:.1f} / {row['m']:.1f}"
    assert "automation risk / amplification" in page.locator("#tree-key").inner_text()

    page.wait_for_selector(".skill-table")
    headers = page.locator(".skill-table thead th").all_inner_texts()
    assert headers[-1] == "Quadrant"
    assert "Machine automation" not in headers


# --- what we found, and how sure ------------------------------------------


def test_the_findings_page_shows_the_four_boxes(desktop, open_ready, stats_json):
    page = open_ready(desktop, at("insights.html"), "#quadrant-table tbody tr")
    rows = page.locator("#quadrant-table tbody tr")
    expect(rows).to_have_count(len(QUADRANT_ORDER))
    assert page.locator("#quadrant-table thead th").first.inner_text() == "Box"
    assert not page.locator("#classes-section").is_visible()
    assert page.locator("#extremes-section").is_visible()
    assert not page.locator("#substituted-section").is_visible()
    assert f"cut at {stats_json['threshold']}" in page.locator("#split-note").inner_text()


def test_the_method_page_explains_the_four_boxes(desktop, open_ready, stats_json):
    page = open_ready(desktop, at("method.html"), "#quadrant-table tbody tr")
    expect(page.locator("#quadrant-table tbody tr")).to_have_count(len(QUADRANT_ORDER))
    assert page.locator("#type-table").count() == 0 or not (
        page.locator("#type-table").is_visible())
    assert page.locator("#boxes-heading").is_visible()
    assert not page.locator("#types-heading").is_visible()
    assert page.locator("[data-stat='threshold']").first.inner_text() == (
        str(stats_json["threshold"]))
    assert desktop.unexpected_errors() == []


# --- and nowhere a figure the older data cannot fill -----------------------


@pytest.mark.parametrize("path,ready", [
    (f"job.html#{SLUG}", "#job-article"),
    (f"groups.html#g={GROUP}&view=ranked", "#panel ol.ranked li"),
    ("insights.html", "#group-table tbody tr"),
    (f"tree.html#job={SLUG}", ".skill-table"),
])
def test_no_page_draws_a_shares_bar_under_the_older_set(desktop, open_ready, path, ready):
    page = open_ready(desktop, at(path), ready)
    expect(page.locator(".shares-bar")).to_have_count(0)
    expect(page.locator(".class-chip")).to_have_count(0)
    assert desktop.unexpected_errors() == []
