""""What we found" under the default set.

The page carries two halves. The four-box half — the extremes, the vanishing
jobs, the immovable, the metamorphosis — belongs to the older scheme and is
hidden here; four sections built around the four shares take its place, and the
split table is retitled for the seven types. Nothing is ever filled with the
zeros the other scheme's lookups would return.

Every figure is computed from stats_v2.json, groups_v2.json and
search_index_v2.json, which is where the page gets them too.
"""

import pytest
from playwright.sync_api import expect

from .conftest import (
    SKILL_CLASS_NAMES,
    SKILL_CLASS_ORDER,
    TYPE_NAMES,
    percent_text,
    share_percents,
    whole_percents,
)

READY = "#group-table tbody tr"

#: The sections that only make sense with four boxes, and the four that replace
#: them. site/js/pages/insights.js hides one set and shows the other.
QUADRANT_SECTIONS = ["extremes-section", "shrink-section", "stable-section",
                     "transform-section"]
SHARES_SECTIONS = ["classes-section", "substituted-section", "assisted-section",
                   "mechanised-section", "insulated-section"]

#: Which share each of the four ranked lists is ranked by.
SHARE_LISTS = [("substituted-list", 0), ("assisted-list", 1),
               ("mechanised-list", 2), ("insulated-list", 3)]


@pytest.fixture(scope="module")
def type_split(stats_v2):
    """The split table's rows: largest type first, percentages adding to 100."""
    order = stats_v2["types"]["order"]
    counts = [stats_v2["types"]["counts"].get(code, 0) for code in order]
    rows = list(zip(order, counts, whole_percents(counts)))
    return sorted(rows, key=lambda row: -row[1])


def test_the_shares_half_is_shown_and_the_four_box_half_is_not(desktop, open_ready):
    page = open_ready(desktop, "insights.html", READY)
    for section in SHARES_SECTIONS:
        assert page.locator(f"#{section}").is_visible(), f"#{section} is hidden"
    for section in QUADRANT_SECTIONS:
        assert not page.locator(f"#{section}").is_visible(), f"#{section} is shown"
    assert page.locator("#article-heading").inner_text() == "The shape of the work"
    assert desktop.unexpected_errors() == []


def test_the_split_table_counts_the_seven_types(desktop, open_ready, type_split):
    page = open_ready(desktop, "insights.html", READY)
    table = page.locator("#quadrant-table")
    assert table.locator("thead th").first.inner_text() == "Type"
    assert table.locator("caption").inner_text() == "Occupations of each type, largest first."

    rows = table.locator("tbody tr")
    expect(rows).to_have_count(len(type_split))
    assert sum(percent for _, _, percent in type_split) == 100
    for position, (code, count, percent) in enumerate(type_split):
        cells = rows.nth(position)
        assert cells.locator("th").inner_text() == TYPE_NAMES[code]
        assert cells.locator("td").nth(0).inner_text() == f"{count:,}"
        assert cells.locator("td").nth(1).inner_text() == percent_text(percent, count)


def test_the_class_table_counts_every_scored_skill(desktop, open_ready, stats_v2):
    page = open_ready(desktop, "insights.html", READY)
    counts = [stats_v2["skill_classes"]["counts"].get(code, 0) for code in SKILL_CLASS_ORDER]
    rows = page.locator("#class-table tbody tr")
    expect(rows).to_have_count(len(SKILL_CLASS_ORDER))
    for position, (code, count, percent) in enumerate(
            zip(SKILL_CLASS_ORDER, counts, whole_percents(counts))):
        assert rows.nth(position).locator("th").inner_text() == SKILL_CLASS_NAMES[code]
        assert rows.nth(position).locator("td").nth(0).inner_text() == f"{count:,}"
        assert rows.nth(position).locator("td").nth(1).inner_text() == percent_text(
            percent, count)
    assert sum(whole_percents(counts)) == 100


def test_the_hero_states_what_was_scored(desktop, open_ready, stats_v2):
    page = open_ready(desktop, "insights.html", READY)
    stats = page.locator("#hero-stats").inner_text()
    assert f"{stats_v2['occupations']:,}" in stats
    assert f"{stats_v2['skills_scored']:,}" in stats


@pytest.mark.parametrize("list_id,at", SHARE_LISTS, ids=[i for i, _ in SHARE_LISTS])
def test_a_ranked_list_leads_with_the_share_it_ranks_by(
        desktop, open_ready, by_slug_v2, list_id, at):
    """A rank is never the only thing said: the row carries the share it is
    ranked by and then the whole make-up of the job."""
    page = open_ready(desktop, "insights.html", READY)
    rows = page.locator(f"#{list_id} li")
    assert rows.count() > 0
    first = rows.first
    slug = first.locator("a").get_attribute("href").split("#", 1)[1]
    percents = share_percents(by_slug_v2[slug]["sh"])
    assert first.locator(".rank-figure").inner_text().endswith(f"{percents[at]}%")
    assert first.locator(".shares-line").is_visible()


def test_the_group_table_shows_what_an_average_job_is_made_of(
        desktop, open_ready, groups_v2):
    page = open_ready(desktop, "insights.html", READY)
    headers = page.locator("#group-table thead th").all_inner_texts()
    assert headers == ["Major group", "Jobs", "What the average job is made of",
                       "Largest type"]
    majors = [key for key in groups_v2 if key.startswith("major:")]
    expect(page.locator("#group-table tbody tr")).to_have_count(len(majors))
    first = page.locator("#group-table tbody tr").first
    assert first.locator(".shares-cell .shares-line").is_visible()
    assert first.locator("td").last.inner_text() in TYPE_NAMES.values()
