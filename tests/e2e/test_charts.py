"""Every drawn chart is reachable without sight and without a mouse.

A canvas or an SVG carries no text, so each one has to say what it shows
(role="img" plus an aria-label), take the focus, and offer the same numbers as
a real table.
"""

import pytest
from playwright.sync_api import expect

PARENT = "major:2"
GROUP = "unit:2512"


def labelled_image(locator):
    """Assert one drawn chart announces itself and can take the focus."""
    expect(locator).to_have_attribute("role", "img")
    expect(locator).to_have_attribute("tabindex", "0")
    assert (locator.get_attribute("aria-label") or "").strip()


def test_the_treemap_is_a_focusable_image_with_a_text_alternative(desktop, open_group):
    page = open_group(desktop, PARENT, "treemap")
    labelled_image(page.locator("#treemap-canvas"))


def test_enter_on_a_focused_treemap_tile_navigates(desktop, open_group):
    page = open_group(desktop, PARENT, "treemap")
    before = page.url
    page.locator("#treemap-canvas").focus()
    expect(page.locator("#chart-caption")).to_contain_text("Press Enter")
    page.keyboard.press("Enter")
    page.wait_for_url(lambda url: url != before)
    assert "#g=" in page.url


@pytest.mark.parametrize("view,chart", [("scatter", "#panel svg.scatter"),
                                        ("treemap", "#panel canvas")])
def test_a_group_chart_offers_a_real_table(desktop, open_group, view, chart):
    page = open_group(desktop, GROUP, view)
    labelled_image(page.locator(chart))
    page.get_by_role("button", name="View as table").click()
    page.wait_for_url(lambda url: url.endswith("view=table"))
    expect(page.locator("#panel table tbody tr").first).to_be_visible()


def test_the_job_page_scatter_offers_a_real_table(desktop, open_job):
    page = open_job(desktop, "job.html#software-developer")
    canvas = page.locator("#scatter")
    expect(canvas).to_have_attribute("role", "img")
    assert (canvas.get_attribute("aria-label") or "").strip()
    toggle = page.locator("#scatter-toggle")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    expect(page.locator("#scatter-table-wrap table tbody tr").first).to_be_visible()


def test_the_job_scatter_table_carries_the_class_and_all_three_scores(desktop, open_job):
    """Under the default the plot is not a class boundary — a skill's class
    comes from the model's own probabilities — so the table has to say both."""
    page = open_job(desktop, "job.html#software-developer")
    page.locator("#scatter-toggle").click()
    headers = page.locator("#scatter-head th").all_inner_texts()
    assert headers == ["Skill", "What AI can do with it", "AI substitution (out of 10)",
                       "AI assistance (out of 10)", "Machine automation (out of 10)",
                       "In this job"]
    legend = page.locator("#scatter-classes")
    expect(legend).to_be_visible()
    assert page.locator("#scatter-threshold").inner_text() == ""
    assert "no lines are drawn across it" in (
        page.locator("#scatter").get_attribute("aria-label").lower())
