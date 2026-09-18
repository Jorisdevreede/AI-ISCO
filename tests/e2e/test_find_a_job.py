"""The landing page: a synonym finds the job, and a chip opens what it names.

Both flows are regressions of audit findings the brief records: "programmer"
never reached Software developer, and five of eight chips opened a different
job than the one on the chip. The fix wave added a tier for an exact ESCO
synonym, and taught the no-match state to offer only what the visitor's own
words found — or nothing at all.
"""

import re
import unicodedata

import pytest
from playwright.sync_api import expect

from .conftest import TYPE_SHORT, share_percents

#: An ESCO alternative label, not a title: it only matches through `alt`.
SYNONYM = "programmer"
WANTED = "software developer"

#: How many chips are opened in a real browser. The rest are checked by href
#: against search_index_v2.json, which keeps the suite quick.
OPENED = 3

JOB_URL = re.compile(r"/job\.html#[a-z0-9-]+$")

#: Exact synonyms that must beat every job with the word somewhere inside it.
EXACT_SYNONYMS = [
    ("programmer", "software developer"),
    ("software engineer", "software developer"),
    ("nurse", "nurse responsible for general care"),
]

#: Queries with no job of that name: a Dutch word for nurse, and an
#: abbreviation. Whatever the page offers for either has to be something it
#: really found, not a guess dressed as an answer.
HARD_QUERIES = ["verpleegkundige", "GP"]


def fold(text):
    stripped = "".join(part for part in unicodedata.normalize("NFD", str(text))
                       if not unicodedata.combining(part))
    return " ".join(stripped.lower().split())


def listbox(page):
    """The landing combobox options, once the list has been painted."""
    options = page.locator("#job-listbox li")
    options.first.wait_for()
    return options


def titles(page):
    return [text.strip().lower() for text in listbox(page).locator(".title").all_inner_texts()]


def chip_rows(page):
    """Every chip as (visible label, slug it links to)."""
    chips = page.locator("#chip-row a")
    chips.first.wait_for()
    return [(chips.nth(i).inner_text().strip(),
             chips.nth(i).get_attribute("href").split("#", 1)[1])
            for i in range(chips.count())]


@pytest.mark.parametrize("query,wanted", EXACT_SYNONYMS, ids=[q for q, _ in EXACT_SYNONYMS])
def test_an_exact_synonym_offers_that_job_first(desktop, query, wanted):
    """Someone who types a job's exact ESCO synonym means that job, not every
    title with the word somewhere inside it."""
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(query)
    assert titles(page)[0] == wanted, f'"{query}" offered {titles(page)[:4]}'


def test_choosing_that_option_opens_the_right_job(desktop):
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(SYNONYM)
    listbox(page).nth(titles(page).index(WANTED)).click()
    page.wait_for_url(re.compile(r"/job\.html#software-developer$"))
    expect(page.locator("h1")).to_have_text(re.compile(WANTED, re.IGNORECASE))


def test_a_result_says_what_kind_of_job_it_is_under_the_default(desktop, by_slug_v2):
    """A result line carries the job's type and how much of it AI can take
    over, so the list is readable before anything is opened."""
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(SYNONYM)
    meta = listbox(page).first.locator(".meta").inner_text()
    row = by_slug_v2["software-developer"]
    assert TYPE_SHORT[row["q"]] in meta
    assert f"AI can take over {share_percents(row['sh'])[0]}%" in meta


@pytest.mark.parametrize("query", HARD_QUERIES)
def test_a_query_with_no_job_of_that_name_offers_only_what_it_found(desktop, query):
    """Whatever comes back has to be something the query really matched: a row
    that says which label it matched, or a suggestion built from a word the
    visitor typed. Nothing at all is kinder than three unrelated jobs."""
    page = desktop.open("index.html")
    page.get_by_label("Find your job title").fill(query)
    needle = fold(query)

    options = page.locator("#job-listbox li")
    page.wait_for_function(
        "() => document.querySelector('#job-listbox').children.length > 0"
        " || !document.getElementById('search-hint').hidden")
    if options.count():
        for position in range(options.count()):
            shown = fold(options.nth(position).inner_text())
            assert needle in shown, f'"{query}" offered {shown!r} without saying why'
        return

    hint = page.locator("#search-hint")
    expect(hint).to_be_visible()
    assert f"No job called “{query}”" in hint.inner_text()
    suggested = hint.locator(".nearest-list li a")
    if suggested.count() == 0:
        return
    heading = hint.locator("p").nth(1).inner_text()
    assert heading.startswith("No exact match. Jobs matching"), heading
    word = heading.split("“")[1].split("”")[0]
    for title in suggested.all_inner_texts():
        assert word in fold(title), f'"{query}" suggested unrelated {title!r}'


def test_every_chip_is_titled_like_the_job_it_links_to(desktop, by_slug_v2):
    rows = chip_rows(desktop.open("index.html"))
    assert len(rows) >= OPENED
    for label, slug in rows:
        assert slug in by_slug_v2, f"chip {label!r} links to unknown slug {slug!r}"
        assert by_slug_v2[slug]["t"].lower() == label.lower()


def test_every_chip_names_the_kind_of_job_beside_it(desktop, by_slug_v2):
    page = desktop.open("index.html")
    page.locator("#chip-row a").first.wait_for()
    items = page.locator("#chip-row span.chip-item")
    expect(items).to_have_count(len(chip_rows(page)))
    # The "·" before the name is ::before content, so it is not in the text.
    for position, (_, slug) in enumerate(chip_rows(page)):
        assert items.nth(position).locator(".chip-meta").inner_text() == (
            TYPE_SHORT[by_slug_v2[slug]["q"]])


@pytest.mark.parametrize("position", range(OPENED))
def test_a_chip_opens_the_job_it_names(desktop, position):
    page = desktop.open("index.html")
    label = chip_rows(page)[position][0]
    page.locator("#chip-row a").nth(position).click()
    page.wait_for_url(JOB_URL)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    assert page.locator("h1").inner_text().strip().lower() == label.lower()
