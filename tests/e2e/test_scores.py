"""The numbers on a job page are the numbers in the published index.

The landing search, the group table and the tree all read search_index_v2.json;
the job page reads its own unit shard. A visitor moving between them must never
see a figure change, so the page is checked against the index — three display
scores, four shares and the type, all computed from the file, none typed here.
"""

import pytest
from playwright.sync_api import expect

from .conftest import (
    SKILL_CLASS_NAMES,
    SKILL_CLASS_ORDER,
    TYPE_SHORT,
    reveal_all,
    share_percents,
)

SLUGS = ["software-developer", "chef"]

#: Which score card carries which field of a search_index_v2 row.
CARDS = [("sub", "a"), ("assist", "m"), ("mech", "k")]


@pytest.mark.parametrize("slug", SLUGS)
def test_the_job_page_shows_the_published_scores(desktop, open_job, by_slug_v2, slug):
    """The three 1-10 scores are secondary detail under the shares scheme, so
    they may sit behind a disclosure; what matters is that they are the
    published ones wherever the page keeps them."""
    page = reveal_all(open_job(desktop, f"job.html#{slug}"))
    row = by_slug_v2[slug]
    for modifier, key in CARDS:
        expect(page.locator(f".score-card--{modifier} .score-value")).to_have_text(
            f"{row[key]:.1f}")


@pytest.mark.parametrize("slug", SLUGS)
def test_the_job_page_shows_the_published_shares(desktop, open_job, by_slug_v2, slug):
    page = reveal_all(open_job(desktop, f"job.html#{slug}"))
    row = by_slug_v2[slug]
    percents = share_percents(row["sh"])
    legend = page.locator("#job-shares .shares-legend li")
    expect(legend).to_have_count(len(SKILL_CLASS_ORDER))
    for position, code in enumerate(SKILL_CLASS_ORDER):
        text = legend.nth(position).inner_text()
        assert SKILL_CLASS_NAMES[code] in text
        assert f"{percents[position]}%" in text


@pytest.mark.parametrize("slug", SLUGS)
def test_the_job_page_shows_the_published_type(desktop, open_job, by_slug_v2, slug):
    page = open_job(desktop, f"job.html#{slug}")
    badge = page.locator("#job-meta .type-badge")
    expect(badge.locator(".quadrant-badge-name")).to_have_text(
        TYPE_SHORT[by_slug_v2[slug]["q"]])
    assert badge.get_attribute("data-type") == by_slug_v2[slug]["q"]


@pytest.mark.parametrize("slug", SLUGS)
def test_the_shard_and_the_index_agree_about_the_job(unit_data, units_json, by_slug_v2, slug):
    """A guard on the two files rather than on the browser: if they ever drift
    apart the tests above would pass while the site contradicted itself."""
    shard = unit_data(units_json[slug])
    occupation = next(row for row in shard["occupations"] if row["s"] == slug)
    row = by_slug_v2[slug]
    assert (occupation["ar"], occupation["ap"], occupation["ak"]) == (
        row["a"], row["m"], row["k"])
    assert occupation["sh"] == row["sh"]
    assert occupation["q"] == row["q"]
