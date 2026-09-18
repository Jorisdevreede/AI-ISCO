"""Reading a job page about someone else, and getting back to your own.

`&for=other` swaps every heading to neutral wording, keeps the page out of the
visitor's own "recently viewed", and has to survive both the Copy link button
and the switch that offers the other wording.
"""

import json
import re

from playwright.sync_api import expect

SLUG = "software-developer"
MINE = f"job.html#{SLUG}"
NEUTRAL = f"job.html#{SLUG}&for=other"

#: The real clipboard needs a permission grant and cannot be read back
#: reliably; this records what the page asked to copy instead.
CLIPBOARD_STUB = """
Object.defineProperty(navigator, 'clipboard', {
  configurable: true,
  value: { writeText: (text) => { window.__copied = text; return Promise.resolve(); } },
});
"""

SECOND_PERSON = re.compile(r"\byou(r|rs)?\b", re.IGNORECASE)


def recent(page):
    """The slugs in localStorage['ai-isco-recent']."""
    return json.loads(page.evaluate(
        "() => window.localStorage.getItem('ai-isco-recent') || '[]'"))


def headings(page):
    return page.locator("main h1, main h2, main h3").all_inner_texts()


def test_my_job_wording_is_first_person_and_is_remembered(desktop, open_job):
    page = open_job(desktop, MINE)
    assert SECOND_PERSON.search(page.locator("#scores-heading").inner_text())
    expect(page.locator("#job-note")).to_be_hidden()
    assert SLUG in recent(page)


def test_neutral_wording_never_addresses_the_reader_and_is_not_remembered(desktop, open_job):
    page = open_job(desktop, NEUTRAL)
    spoken_to = [text for text in headings(page) if SECOND_PERSON.search(text)]
    assert not spoken_to, f"neutral headings still address the reader: {spoken_to}"
    expect(page.locator("#job-note")).to_be_visible()
    expect(page.locator("#job-framing")).to_contain_text("Switch back")
    assert SLUG not in recent(page)


def test_copy_link_keeps_for_other(desktop, open_job):
    desktop.page.add_init_script(CLIPBOARD_STUB)
    page = open_job(desktop, NEUTRAL)
    page.get_by_role("button", name="Copy link").click()
    expect(page.locator("#job-copy-status")).to_have_text("Link copied.")
    assert page.evaluate("() => window.__copied").endswith("&for=other")


def test_the_switch_round_trips_between_the_two_wordings(desktop, open_job):
    page = open_job(desktop, NEUTRAL)
    page.locator("#job-framing").click()
    page.wait_for_url(re.compile(rf"/job\.html#{SLUG}$"))
    assert SECOND_PERSON.search(page.locator("#scores-heading").inner_text())
    assert SLUG in recent(page)
    page.locator("#job-framing").click()
    page.wait_for_url(re.compile(rf"/job\.html#{SLUG}&for=other$"))
    expect(page.locator("#job-note")).to_be_visible()
