"""Landing to a job page, and on to a second job, without touching a mouse.

Nothing in this module clicks, hovers or taps: every step is Tab, a printable
key, an arrow or Enter, which is the only way to prove the path works for
someone who never uses a pointer.
"""

import re

from playwright.sync_api import expect

JOB_URL = re.compile(r"/job\.html#[a-z0-9-]+$")
SECOND = "accountant"


def active_id(page):
    return page.evaluate("() => document.activeElement.id")


def tab_to(page, element_id, limit=20):
    """Press Tab until `element_id` has the focus. True when it got there."""
    for _ in range(limit):
        page.keyboard.press("Tab")
        if active_id(page) == element_id:
            return True
    return False


def type_and_open(page, box_id, query):
    """Type into a combobox, arrow onto the first option and open it."""
    assert tab_to(page, box_id), f"Tab never reached #{box_id}"
    page.keyboard.type(query)
    page.locator("li[role='option']").first.wait_for()
    page.keyboard.press("ArrowDown")
    expect(page.locator(f"#{box_id}")).to_have_attribute(
        "aria-activedescendant", f"{box_id}-option-0")
    page.keyboard.press("Enter")


def test_keyboard_only_from_the_landing_page_to_a_job(desktop):
    page = desktop.open("index.html")
    type_and_open(page, "job-search", "programmer")
    page.wait_for_url(JOB_URL, timeout=15_000)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    assert page.locator("h1").inner_text().strip()
    # A cross-document jump resets the focus to <body>, as it does on every
    # site. What matters is that the next Tab reaches the content.
    page.keyboard.press("Tab")
    assert page.evaluate("() => document.activeElement.getAttribute('href')") == "#main"
    expect(page.locator("#job-title")).to_have_attribute("tabindex", "-1")


def test_the_in_page_finder_moves_the_focus_to_the_new_heading(desktop):
    page = desktop.open("index.html")
    type_and_open(page, "job-search", "programmer")
    page.wait_for_url(JOB_URL, timeout=15_000)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    type_and_open(page, "job-find", SECOND)
    page.wait_for_url(re.compile(rf"/job\.html#{SECOND}$"))
    assert active_id(page) == "job-title"
    expect(page.locator("h1")).to_have_text(re.compile(SECOND, re.IGNORECASE))
