"""Old links still land, and the score switch survives a nav click.

portfolio.html and explorer.html are stubs kept for links published before the
rebuild.

Two `?scorer=` values are still worth carrying. `?scorer=gemini` is the one the
switch writes now — choosing the default removes the parameter instead, so it
is the only query a visitor can be holding. `?scorer=typesafe` is the alias
kept for links shared before the v2 set existed; it is not dead, so it is still
tested here. What either one means on screen is test_second_scorer.py's job;
here only the URL is asserted.
"""

import re

import pytest

from .conftest import GEMINI

ALIAS = "?scorer=typesafe"


def test_portfolio_hands_an_old_link_to_the_job_page(desktop, open_job):
    page = open_job(desktop, "portfolio.html#software-developer")
    assert page.url.endswith("/job.html#software-developer")


def test_explorer_hands_the_visitor_to_the_group_overview(desktop, open_ready):
    page = open_ready(desktop, "explorer.html", "#tablist [role='tab']")
    assert page.url.endswith("/groups.html")


@pytest.mark.parametrize("query", [GEMINI, ALIAS])
def test_a_scorer_query_is_carried_across_a_nav_link(desktop, open_ready, query):
    page = open_ready(desktop, f"index.html{query}", "#chip-row a")
    link = page.get_by_role("link", name="Browse sectors", exact=True)
    assert link.get_attribute("href") == f"groups.html{query}"
    link.click()
    page.wait_for_url(re.compile(rf"/groups\.html{re.escape(query)}$"))


def test_a_scorer_query_is_carried_into_a_job_page(desktop, open_ready):
    """A chip is built by the page, not by the chrome, so it needs the query too."""
    page = open_ready(desktop, f"index.html{GEMINI}", "#chip-row a")
    href = page.locator("#chip-row a").first.get_attribute("href")
    assert href.startswith(f"job.html{GEMINI}#")
