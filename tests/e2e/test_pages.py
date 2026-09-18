"""What every page owes every visitor, page by page.

One list drives three checks: no horizontal scroll on a 390 px phone, a clean
console at desktop width, and the accessibility basics a screen reader needs
to find its way around.
"""

from typing import NamedTuple

import pytest
from playwright.sync_api import expect

#: "manage budgets" — a published skill id, the same one site/js/pages/skill.js
#: offers as an example. test_the_sample_skill_id_is_still_published guards it.
SKILL_ID = "7677a630"


class Page(NamedTuple):
    """One page to check, and how to tell it finished rendering."""

    path: str
    nav: str | None  # the nav label that carries aria-current, if any
    ready: str


PAGES = [
    Page("index.html", "Find a job", "#chip-row a"),
    Page("job.html#software-developer", None, "#job-article"),
    Page("groups.html", "Browse sectors", "#tablist [role='tab']"),
    Page("groups.html#g=major:2&view=table", "Browse sectors", "#panel tbody tr"),
    Page("skill.html", "Look up a skill", "#skill-examples a"),
    Page(f"skill.html#{SKILL_ID}", "Look up a skill", "#skill-detail .occ-link"),
    Page("method.html", "How sure is this?", "#quadrant-table tbody tr"),
    Page("insights.html", "What we found", "#group-table tbody tr"),
]

every_page = pytest.mark.parametrize("page_spec", PAGES, ids=[spec.path for spec in PAGES])


def test_the_sample_skill_id_is_still_published(skill_index):
    assert any(row["id"] == SKILL_ID for row in skill_index)


@every_page
def test_no_horizontal_overflow_on_a_phone(phone, open_ready, page_spec):
    open_ready(phone, page_spec.path, page_spec.ready)
    assert not phone.overflows()


@every_page
def test_no_console_or_page_errors(desktop, open_ready, page_spec):
    open_ready(desktop, page_spec.path, page_spec.ready)
    assert desktop.unexpected_errors() == []


@every_page
def test_accessibility_basics(desktop, open_ready, page_spec):
    page = open_ready(desktop, page_spec.path, page_spec.ready)
    expect(page.locator("h1")).to_have_count(1)
    expect(page.locator("main")).to_have_count(1)
    expect(page.locator("footer.site-attribution")).to_have_count(1)
    assert page.get_attribute("html", "lang") == "en"
    assert page.title().strip()
    current = page.locator("nav#site-nav a[aria-current='page']")
    expect(current).to_have_count(0 if page_spec.nav is None else 1)
    if page_spec.nav:
        expect(current).to_have_text(page_spec.nav)
