"""No page but the job page may download the 13 MB score files.

The brief puts the whole point of the rebuild here: the landing page, the
group overview and the article are built from small index files, and a visitor
who never opens a job never pays for one.
"""

import pytest

#: The two files the rebuild moved off every page but job.html.
BIG_FILES = {"portfolio_data.json", "data.json"}

PAGES = [
    ("index.html", "#chip-row a"),
    ("groups.html", "#tablist [role='tab']"),
    ("groups.html#g=major:2&view=table", "#panel tbody tr"),
    ("insights.html", "#group-table tbody tr"),
]


@pytest.mark.parametrize("path,ready", PAGES, ids=[path for path, _ in PAGES])
def test_the_page_never_asks_for_the_big_score_files(desktop, open_ready, path, ready):
    open_ready(desktop, path, ready)
    asked = sorted(BIG_FILES.intersection(desktop.requests))
    assert not asked, f"{path} downloaded {asked}"
