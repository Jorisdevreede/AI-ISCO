"""What each page costs, and when.

No page downloads a whole score set any more (test_pages.py holds that line for
every page at once). What is left to prove is the second half of the rebuild:
the big per-skill and per-job files are fetched when something on screen needs
them, and not before.
"""

import pytest

from .conftest import reveal_all

SLUG = "software-developer"

#: The unit group software developer sits in. Its shard is the only job data
#: either the job page or the tree may download.
UNIT = "2512"
SHARD = f"{UNIT}_v2.json"

#: Files a page has to earn: a megabyte or more each.
SEARCH_INDEX = "search_index_v2.json"
SKILL_INDEX = "skill_index_v2.json"
SKILL_OCCUPATIONS = "skill_occupations_v2.json"

PAGES = [
    ("index.html", "#chip-row a"),
    ("groups.html", "#tablist [role='tab']"),
    ("groups.html#g=major:2&view=table", "#panel tbody tr"),
    ("insights.html", "#group-table tbody tr"),
    ("tree.html", "[role='tree'] [role='treeitem']"),
]


@pytest.mark.parametrize("path,ready", PAGES, ids=[path for path, _ in PAGES])
def test_a_page_that_shows_no_job_asks_for_no_job_file(desktop, open_ready, path, ready):
    open_ready(desktop, path, ready)
    shards = sorted(name for name in desktop.fetched if name.endswith("_v2.json")
                    and name[0].isdigit())
    assert not shards, f"{path} downloaded {shards}"


def test_the_job_page_reads_one_shard_and_defers_the_search_index(desktop, open_job):
    """The finder is the only thing on a job page that needs every title, and a
    visitor who came to read one job never types in it."""
    page = open_job(desktop, f"job.html#{SLUG}")
    assert SHARD in desktop.fetched
    assert SEARCH_INDEX not in desktop.fetched

    page.locator("#job-find").focus()
    page.wait_for_function(
        "() => performance.getEntriesByType('resource')"
        ".some(entry => entry.name.includes('search_index'))")
    assert desktop.unexpected_errors() == []


def test_the_tree_loads_one_shard_for_the_job_it_was_given(desktop, open_ready):
    page = open_ready(desktop, f"tree.html#job={SLUG}", ".skill-table")
    assert desktop.fetched.count(SHARD) == 1
    assert not [name for name in desktop.fetched
                if name.endswith("_v2.json") and name[0].isdigit() and name != SHARD]
    assert page.locator(".skill-table tbody tr").count() > 0


def test_the_skill_page_reads_two_indexes_until_a_skill_is_opened(
        desktop, open_ready, skill_index_v2):
    """The empty state is a table and an explainer: the occupation lists behind
    a skill are three megabytes nobody has asked for yet."""
    skill = next(row for row in skill_index_v2 if row["ne"] > 0)
    open_ready(desktop, "skill.html", "#skill-table .list-table tbody tr")
    assert SKILL_INDEX in desktop.fetched
    assert SKILL_OCCUPATIONS not in desktop.fetched
    assert SEARCH_INDEX not in desktop.fetched

    page = open_ready(desktop, f"skill.html#{skill['id']}", "#skill-detail .score-row")
    reveal_all(page)
    page.locator("#skill-detail .occ-link").first.wait_for(state="visible", timeout=15_000)
    assert SKILL_OCCUPATIONS in desktop.fetched
    assert desktop.unexpected_errors() == []


def test_the_group_page_waits_for_the_skills_section_to_come_into_view(desktop, open_group):
    """skill_index.json is a megabyte spent on two lists of five names."""
    page = open_group(desktop, "unit:2512", "ranked")
    assert SKILL_INDEX not in desktop.fetched

    page.locator("#skills-section").scroll_into_view_if_needed()
    page.locator("#skill-columns a").first.wait_for()
    assert SKILL_INDEX in desktop.fetched
    assert desktop.unexpected_errors() == []
