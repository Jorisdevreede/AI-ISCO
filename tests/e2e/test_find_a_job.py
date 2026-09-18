"""The landing page: a synonym finds the job, and a chip opens what it names.

Both flows are regressions of audit findings the brief records: "programmer"
never reached Software developer, and five of eight chips opened a different
job than the one on the chip.
"""

import re

import pytest
from playwright.sync_api import expect

#: An ESCO alternative label, not a title: it only matches through `alt`.
SYNONYM = "programmer"
WANTED = "software developer"

#: How many chips are opened in a real browser. The rest are checked by href
#: against search_index.json, which keeps the suite quick.
OPENED = 3

JOB_URL = re.compile(r"/job\.html#[a-z0-9-]+$")


def listbox(page):
    """The landing combobox options, once the list has been painted."""
    options = page.locator("#job-listbox li")
    options.first.wait_for()
    return options


def chip_rows(page):
    """Every chip as (visible label, slug it links to)."""
    chips = page.locator("#chip-row a")
    chips.first.wait_for()
    return [(chips.nth(i).inner_text().strip(),
             chips.nth(i).get_attribute("href").split("#", 1)[1])
            for i in range(chips.count())]


def test_a_synonym_offers_the_job_it_means(desktop):
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(SYNONYM)
    titles = [text.strip().lower() for text in listbox(page).locator(".title").all_inner_texts()]
    assert WANTED in titles, f'"{SYNONYM}" offered {titles}'


def test_choosing_that_option_opens_the_right_job(desktop):
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(SYNONYM)
    titles = [text.strip().lower() for text in listbox(page).locator(".title").all_inner_texts()]
    listbox(page).nth(titles.index(WANTED)).click()
    page.wait_for_url(re.compile(r"/job\.html#software-developer$"))
    expect(page.locator("h1")).to_have_text(re.compile(WANTED, re.IGNORECASE))


def test_every_chip_is_titled_like_the_job_it_links_to(desktop, by_slug):
    rows = chip_rows(desktop.open("index.html"))
    assert len(rows) >= OPENED
    for label, slug in rows:
        assert slug in by_slug, f"chip {label!r} links to unknown slug {slug!r}"
        assert by_slug[slug]["t"].lower() == label.lower()


@pytest.mark.parametrize("position", range(OPENED))
def test_a_chip_opens_the_job_it_names(desktop, position):
    page = desktop.open("index.html")
    label = chip_rows(page)[position][0]
    page.locator("#chip-row a").nth(position).click()
    page.wait_for_url(JOB_URL)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    assert page.locator("h1").inner_text().strip().lower() == label.lower()
