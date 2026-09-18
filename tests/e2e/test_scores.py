"""The two numbers on a job page are the two numbers in the search index.

The job page reads portfolio_data.json and the landing search reads
search_index.json. A visitor moving between them must never see the pair
change, so the pages are checked against the published index.
"""

import pytest
from playwright.sync_api import expect

SLUGS = ["software-developer", "chef"]


@pytest.mark.parametrize("slug", SLUGS)
def test_the_job_page_shows_the_published_scores(desktop, open_job, by_slug, slug):
    page = open_job(desktop, f"job.html#{slug}")
    row = by_slug[slug]
    expect(page.locator(".score-card--auto .score-value")).to_have_text(f"{row['a']:.1f}")
    expect(page.locator(".score-card--amp .score-value")).to_have_text(f"{row['m']:.1f}")
