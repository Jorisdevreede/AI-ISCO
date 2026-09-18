"""What every page owes every visitor, page by page.

One list drives five checks: no horizontal scroll on a 390 px phone, a clean
console at desktop width, the accessibility basics a screen reader needs, the
social-card tags a shared link needs, and the promise that no page downloads
the portfolio file any more.

Every page is opened as a visitor gets it — no `?scorer=`, so the v2 set and
the shares scheme. test_quadrants.py covers `?scorer=gemini`.
"""

from typing import NamedTuple

import pytest
from playwright.sync_api import expect

from .conftest import SITE

#: "manage budgets" — a published skill id, the same one site/js/pages/skill.js
#: offers as an example. test_the_sample_skill_id_is_still_published guards it.
SKILL_ID = "7677a630"

#: The whole-portfolio files. No page asks for one at all any more, so the
#: check is on every request, HEAD probes included.
PORTFOLIO = "portfolio_data"

#: The other whole score set. site/scorer.js HEAD-probes `data_v2.json` on
#: every page to learn which sets are deployed, so only a real GET counts.
WHOLE_SET = "data"


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
    Page("tree.html", "Job tree", "[role='tree'] [role='treeitem']"),
    Page("tree.html#job=software-developer", "Job tree", "[role='treeitem'][aria-selected='true']"),
    Page("skill.html", "Skill scores", "#skill-examples a"),
    # The occupation lists behind a skill are fetched when they scroll into
    # view, so the scores are what proves this page rendered.
    Page(f"skill.html#{SKILL_ID}", "Skill scores", "#skill-detail .score-row"),
    # The four boxes belong to the other set: under the default the method page
    # shows the seven type rules and the four skill classes instead.
    Page("method.html", "How sure is this?", "#type-table tbody tr"),
    Page("insights.html", "What we found", "#group-table tbody tr"),
]

every_page = pytest.mark.parametrize("page_spec", PAGES, ids=[spec.path for spec in PAGES])


def test_the_sample_skill_id_is_still_published(skill_index_v2):
    assert any(row["id"] == SKILL_ID for row in skill_index_v2)


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


@every_page
def test_a_shared_link_carries_a_social_card(desktop, open_ready, page_spec):
    """Every page names itself to whatever unfurls the link, and points at a
    card that is really published."""
    page = open_ready(desktop, page_spec.path, page_spec.ready)
    tags = {tag["property"]: tag["content"] for tag in page.eval_on_selector_all(
        "meta[property^='og:']",
        "nodes => nodes.map(n => ({property: n.getAttribute('property'),"
        " content: n.getAttribute('content')}))")}
    for wanted in ("og:title", "og:description", "og:url", "og:image", "og:type"):
        assert tags.get(wanted, "").strip(), f"{page_spec.path} has no {wanted}"
    assert tags["og:image"].endswith("/card.png")
    assert (SITE / "card.png").is_file()
    assert page.get_attribute("meta[name='twitter:card']", "content") == "summary_large_image"


#: WCAG 2.2 target size, and the `--tap` token in css/app.css.
TAP = 24

#: How many links of a list are measured. A page of 869 job links is 869
#: targets; one rule sets them all, so a sample proves the rule is on.
SAMPLE = 5

TAP_TARGETS = [
    ("job.html#software-developer", "#job-article", "#skill-columns li a"),
    ("groups.html#g=major:2&view=table", "#panel tbody tr", "#panel tbody td a"),
    ("tree.html#q=programmer", "#tree-results-list .result-link",
     "#tree-results-list .result-link"),
]


@pytest.mark.parametrize("path,ready,links", TAP_TARGETS,
                         ids=[path for path, _, _ in TAP_TARGETS])
def test_a_list_link_is_big_enough_to_tap(phone, open_ready, path, ready, links):
    page = open_ready(phone, path, ready)
    targets = page.locator(links)
    assert targets.count() > 0, f"{path} showed no list links to measure"
    for position in range(min(SAMPLE, targets.count())):
        box = targets.nth(position).bounding_box()
        assert box, f"{links} #{position} on {path} has no box"
        assert box["height"] >= TAP, f"{links} #{position} on {path} is {box['height']} px"


@every_page
def test_the_page_never_downloads_a_whole_score_set(desktop, open_ready, page_spec):
    """No page reads portfolio_data or data any more: the job page and the tree
    take one unit group's own shard, and everything else reads an index."""
    open_ready(desktop, page_spec.path, page_spec.ready)
    asked = sorted({name for name in desktop.requests if name.startswith(PORTFOLIO)})
    assert not asked, f"{page_spec.path} asked for {asked}"
    downloaded = sorted({name for name in desktop.fetched
                         if name.startswith(WHOLE_SET) and name.endswith(".json")})
    assert not downloaded, f"{page_spec.path} downloaded {downloaded}"
