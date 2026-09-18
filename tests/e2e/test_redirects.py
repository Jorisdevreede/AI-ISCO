"""Old links still land, and the score switch survives a nav click.

portfolio.html and explorer.html are stubs kept for links published before the
rebuild. ?scorer=typesafe selects the second score set; what that set shows is
covered in test_second_scorer.py, here only the URL is asserted.
"""

import re

SCORER = "?scorer=typesafe"


def test_portfolio_hands_an_old_link_to_the_job_page(desktop, open_job):
    page = open_job(desktop, "portfolio.html#software-developer")
    assert page.url.endswith("/job.html#software-developer")


def test_explorer_hands_the_visitor_to_the_group_overview(desktop, open_ready):
    page = open_ready(desktop, "explorer.html", "#tablist [role='tab']")
    assert page.url.endswith("/groups.html")


def test_a_scorer_query_is_carried_across_a_nav_link(desktop, open_ready):
    page = open_ready(desktop, f"index.html{SCORER}", "#chip-row a")
    link = page.get_by_role("link", name="Browse sectors", exact=True)
    assert link.get_attribute("href") == f"groups.html{SCORER}"
    link.click()
    page.wait_for_url(re.compile(r"/groups\.html\?scorer=typesafe$"))
