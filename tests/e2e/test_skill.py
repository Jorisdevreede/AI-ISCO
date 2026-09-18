"""Look up a skill, then follow it to an occupation that needs it."""

import re

import pytest
from playwright.sync_api import expect

SKILL_TITLE = "manage budgets"
SKILL_URL = re.compile(r"/skill\.html#[0-9a-f]{8}$")


@pytest.fixture(scope="module")
def skill(skill_index):
    """A published skill several occupations need, looked up by its ESCO name."""
    return next(row for row in skill_index
                if row["t"] == SKILL_TITLE and row["ne"] > 0)


def test_searching_by_name_lands_on_the_skill_with_both_scores(desktop, skill):
    page = desktop.open("skill.html")
    page.get_by_label("Search ESCO skills by name").fill(skill["t"])
    options = page.locator("#skill-listbox li")
    options.first.wait_for()
    options.first.click()
    page.wait_for_url(SKILL_URL)
    assert page.url.endswith(f"#{skill['id']}")
    expect(page.locator("h1")).to_have_text(skill["t"])
    rows = page.locator(".score-row").all_inner_texts()
    assert f"{skill['a']:.1f}" in rows[0] and f"{skill['m']:.1f}" in rows[1]


def test_the_first_occupation_that_needs_it_opens_a_job_page(desktop, open_ready, skill):
    page = open_ready(desktop, f"skill.html#{skill['id']}", "#skill-detail .occ-link")
    first = page.locator("#skill-detail .occ-link").first
    slug = first.get_attribute("href").split("#", 1)[1]
    first.click()
    page.wait_for_url(re.compile(rf"/job\.html#{re.escape(slug)}$"), timeout=15_000)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    assert page.locator("h1").inner_text().strip()
