"""Browse sectors: the four views, the table alternative, and the trail back.

The group used throughout is unit:2512, Software developers, because it is
small enough that its table can be counted against groups.json.
"""

import re

import pytest
from playwright.sync_api import expect

GROUP = "unit:2512"
LABEL = "Software developers"
VIEWS = ["ranked", "scatter", "treemap", "table"]
VIEW_LABELS = {"ranked": "Ranked", "scatter": "Scatter", "treemap": "Treemap", "table": "Table"}

BACK_URL = re.compile(rf"/groups\.html#g={re.escape(GROUP)}$")


def viewed(view):
    return re.compile(rf"view={view}$")


def test_the_table_has_one_row_per_job_in_the_group(desktop, open_group, groups_json):
    page = open_group(desktop, GROUP, "table")
    expect(page.locator("#panel tbody tr")).to_have_count(groups_json[GROUP]["n"])


def test_a_job_link_carries_the_group_it_came_from(desktop, open_group):
    page = open_group(desktop, GROUP, "table")
    first = page.locator("#panel tbody td a").first
    slug = first.get_attribute("href").split("#", 1)[1].split("&", 1)[0]
    first.click()
    page.wait_for_url(re.compile(rf"/job\.html#{slug}&from={re.escape(GROUP)}$"), timeout=15_000)
    expect(page.locator("#job-back a")).to_contain_text(f"Back to {LABEL}", timeout=15_000)


def test_the_back_link_returns_to_the_group(desktop, open_group):
    page = open_group(desktop, GROUP, "table")
    page.locator("#panel tbody td a").first.click()
    back = page.locator("#job-back a")
    back.wait_for(state="visible", timeout=15_000)
    back.click()
    page.wait_for_url(BACK_URL)
    expect(page.locator("h1")).to_have_text(LABEL)


def test_the_back_button_walks_tab_changes(desktop, open_group):
    """Tabs push history entries, so Back undoes a tab change (not a reload)."""
    page = open_group(desktop, GROUP, "ranked")
    page.get_by_role("tab", name="Table").click()
    page.wait_for_url(viewed("table"))
    page.go_back()
    page.wait_for_url(viewed("ranked"))
    expect(page.get_by_role("tab", name="Ranked")).to_have_attribute("aria-selected", "true")


@pytest.mark.parametrize("view", VIEWS)
def test_a_view_can_be_chosen_with_the_mouse(desktop, open_group, view):
    page = open_group(desktop, GROUP, "ranked")
    page.get_by_role("tab", name=VIEW_LABELS[view]).click()
    page.wait_for_url(viewed(view))
    expect(page.get_by_role("tab", name=VIEW_LABELS[view])).to_have_attribute(
        "aria-selected", "true")


@pytest.mark.parametrize("steps,view", list(enumerate(VIEWS[1:] + VIEWS[:1], start=1)))
def test_a_view_can_be_chosen_with_the_keyboard(desktop, open_group, steps, view):
    """Arrow keys rove the tablist; Enter on the focused tab selects it."""
    page = open_group(desktop, GROUP, "ranked")
    page.locator("#tab-ranked").focus()
    for _ in range(steps):
        page.keyboard.press("ArrowRight")
    assert page.evaluate("() => document.activeElement.id") == f"tab-{view}"
    page.keyboard.press("Enter")
    page.wait_for_url(viewed(view))
