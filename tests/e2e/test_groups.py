"""Browse sectors: the four views, the table alternative, and the trail back.

The group used throughout is unit:2512, Software developers, because it is
small enough that its table can be counted against groups_v2.json. The ranked
list and the scatter need a crowd, so those use its major group.

Under the default set a group splits over the seven types, leads with the mean
make-up of a job in it, and draws the type rules across the scatter instead of
one cut-off. Every figure below comes out of groups_v2.json.
"""

import re

import pytest
from playwright.sync_api import expect

from .conftest import TYPE_NAMES, percent_text, share_percents, whole_percents

GROUP = "unit:2512"
LABEL = "Software developers"
BIG = "major:2"
VIEWS = ["ranked", "scatter", "treemap", "table"]
VIEW_LABELS = {"ranked": "Ranked", "scatter": "Scatter", "treemap": "Treemap", "table": "Table"}

#: site/js/pages/groups-model.js: ranked rows drawn before the visitor asks.
RANKED_PAGE = 50

BACK_URL = re.compile(rf"/groups\.html#g={re.escape(GROUP)}$")


def viewed(view):
    return re.compile(rf"view={view}$")


def type_order(stats_v2):
    return stats_v2["types"]["order"]


# --- how a group splits ----------------------------------------------------


def test_the_split_names_all_seven_types_with_their_counts(
        desktop, open_group, groups_v2, stats_v2):
    page = open_group(desktop, GROUP, "table")
    group = groups_v2[GROUP]
    order = type_order(stats_v2)
    counts = [group["q"].get(code, 0) for code in order]
    total = sum(counts)

    assert page.locator("#mix-heading").inner_text() == "How this group splits by type"
    rows = page.locator("#mix-bars li")
    expect(rows).to_have_count(len(order))
    assert rows.locator(".mix-name").all_inner_texts() == [TYPE_NAMES[code] for code in order]
    for position, (code, count, percent) in enumerate(
            zip(order, counts, whole_percents(counts))):
        assert rows.nth(position).locator(".mix-figure").inner_text() == (
            f"{count:,} of {total:,} jobs · {percent_text(percent, count)}")
        assert rows.nth(position).get_attribute("data-type") == code


def test_a_class_worth_less_than_a_percent_is_never_printed_as_zero(
        desktop, open_group, groups_v2, stats_v2):
    """"0%" is a lie about a group that has one job in that class."""
    order = type_order(stats_v2)
    for key, group in groups_v2.items():
        counts = [group.get("q", {}).get(code, 0) for code in order]
        percents = whole_percents(counts)
        rounded_away = [code for code, count, percent in zip(order, counts, percents)
                        if count > 0 and percent == 0]
        if rounded_away:
            break
    else:
        pytest.skip("no published group rounds a class below half a percent")

    page = open_group(desktop, key, "table")
    row = page.locator(f"#mix-bars li[data-type='{rounded_away[0]}']")
    assert row.locator(".mix-figure").inner_text().endswith("· <1%")


def test_the_group_leads_with_what_its_average_job_is_made_of(
        desktop, open_group, groups_v2):
    page = open_group(desktop, GROUP, "table")
    group = groups_v2[GROUP]
    percents = share_percents(group["sh"])
    line = page.locator("#mean-shares .mean-shares-line").inner_text()

    assert line.startswith(f"Across the {group['n']:,} jobs in this group, on average")
    for label, percent in zip(["AI can take over", "AI assists", "machines", "stays human"],
                              percents):
        assert f"{label} {percent}%" in line


def test_the_caveat_counts_this_group_rather_than_the_whole_site(
        desktop, open_group, groups_v2):
    """The site-wide near-the-line figure reads as a fact about the group, and
    it is usually the wrong one."""
    page = open_group(desktop, GROUP, "table")
    group = groups_v2[GROUP]
    caveat = page.locator("#mix-caveat")
    assert caveat.is_visible()
    share = round(100 * group["near"] / group["n"])
    assert caveat.inner_text().startswith(
        f"{group['near']:,} of {group['n']:,} jobs ({share}%) in this group sit near a cut-off")


def test_the_two_lists_are_named_for_the_share_they_rank(desktop, open_group):
    page = open_group(desktop, GROUP, "table")
    assert page.locator("#top-heading").inner_text() == "Most of the work AI can take over"
    assert page.locator("#bottom-heading").inner_text() == "Least of the work AI can take over"
    assert "sorted into classes" in page.locator("#skills-note").inner_text()


# --- the table -------------------------------------------------------------


def test_the_table_has_one_row_per_job_in_the_group(desktop, open_group, groups_v2):
    page = open_group(desktop, GROUP, "table")
    expect(page.locator("#panel tbody tr")).to_have_count(groups_v2[GROUP]["n"])
    assert page.locator("#panel table").get_attribute("class") == "table--wide"


def test_the_table_shows_all_four_shares_of_every_job(
        desktop, open_group, by_slug_v2, groups_v2):
    page = open_group(desktop, GROUP, "table")
    headers = page.locator("#panel thead th button").all_inner_texts()
    assert headers[:6] == ["Job", "ISCO", "AI can take over", "AI assists",
                           "Machines", "Stays human"]
    first = page.locator("#panel tbody tr").first
    slug = first.locator("a").get_attribute("href").split("#", 1)[1].split("&", 1)[0]
    cells = first.locator("td").all_inner_texts()
    assert cells[2:6] == [f"{percent}%" for percent in share_percents(by_slug_v2[slug]["sh"])]


def test_a_table_sort_is_in_the_url_and_survives_a_reload(desktop, open_group):
    """The page has a Copy link button one section above the table, so the
    sorted view has to be in the link."""
    page = open_group(desktop, GROUP, "table")
    header = page.locator('#panel th[data-key="t"]')
    header.locator("button").click()
    page.wait_for_url(re.compile(r"&sort=[a-z0-9]+&dir=asc$"))
    first = page.locator("#panel tbody tr").first.inner_text()

    page.reload(wait_until="networkidle")
    page.locator("#panel tbody tr").first.wait_for()
    assert page.locator('#panel th[data-key="t"]').get_attribute("aria-sort") == "ascending"
    assert page.locator("#panel tbody tr").first.inner_text() == first

    # A tab is a change of view, not of order, so it never carries the sort on.
    page.get_by_role("tab", name="Ranked").click()
    page.wait_for_url(viewed("ranked"))


def test_a_job_link_carries_the_group_it_came_from(desktop, open_group):
    page = open_group(desktop, GROUP, "table")
    first = page.locator("#panel tbody td a").first
    slug = first.get_attribute("href").split("#", 1)[1].split("&", 1)[0]
    first.click()
    page.wait_for_url(re.compile(rf"/job\.html#{slug}&from={re.escape(GROUP)}$"), timeout=15_000)
    expect(page.locator("#job-back a")).to_contain_text(f"Back to {LABEL}", timeout=15_000)


def test_the_back_link_returns_to_the_group(desktop, open_group):
    page = open_group(desktop, GROUP, "table")
    page.locator("#panel tbody td a").first.click()
    back = page.locator("#job-back a")
    back.wait_for(state="visible", timeout=15_000)
    back.click()
    page.wait_for_url(BACK_URL)
    expect(page.locator("h1")).to_have_text(LABEL)


# --- the ranked list -------------------------------------------------------


def test_a_long_ranked_list_starts_at_one_screenful_and_offers_the_rest(
        desktop, open_group, groups_v2):
    """869 rows is 114,000 px of phone: a list nobody can reach the end of."""
    page = open_group(desktop, BIG, "ranked")
    total = groups_v2[BIG]["n"]
    rows = page.locator("#panel ol.ranked li")
    expect(rows).to_have_count(RANKED_PAGE)
    footer = page.locator(".ranked-more")
    assert footer.locator("button").all_inner_texts() == [
        f"Show {RANKED_PAGE} more", f"Show all {total:,}"]

    footer.get_by_role("button", name=f"Show {RANKED_PAGE} more").click()
    expect(rows).to_have_count(2 * RANKED_PAGE)
    assert page.locator("#chart-live").inner_text() == (
        f"Showing {2 * RANKED_PAGE} of {total:,} jobs.")

    page.get_by_role("button", name=f"Show all {total:,}").click()
    expect(rows).to_have_count(total)
    expect(page.locator(".ranked-more")).to_have_count(0)
    assert desktop.unexpected_errors() == []


# --- the scatter -----------------------------------------------------------


def test_the_scatter_plots_the_two_shares_and_draws_the_type_rules(
        desktop, open_group, groups_v2, search_index_v2):
    page = open_group(desktop, BIG, "scatter")
    scored = [row for row in search_index_v2
              if row["c"].startswith("2") and isinstance(row.get("sh"), list)]
    dots = page.locator("#panel svg.scatter use.dot-point")
    expect(dots).to_have_count(len(scored))
    assert dots.first.get_attribute("data-type")

    # SVG text has no inner_text; textContent is what it says.
    axes = page.locator("#panel svg.scatter text.axis-text").all_text_contents()
    assert "AI can take over" in axes and "AI assists" in axes
    assert "0%" in axes and "100%" in axes

    rules = page.locator("#panel svg.scatter line.rule-line")
    assert rules.count() > 0
    labels = page.locator("#panel svg.scatter text.rule-text").all_text_contents()
    assert labels and all(label.endswith("%") for label in labels)
    # The note says in words what the lines are, so the chart is never read as
    # "these two shares decide the type".
    note = page.locator("#panel .chart-note").inner_text()
    assert "lines" in note and "cut" in note and "%" in note
    assert groups_v2[BIG]["n"] >= len(scored)
    assert desktop.unexpected_errors() == []


# --- the tabs --------------------------------------------------------------


def test_the_back_button_walks_tab_changes(desktop, open_group):
    """Tabs push history entries, so Back undoes a tab change (not a reload)."""
    page = open_group(desktop, GROUP, "ranked")
    page.get_by_role("tab", name="Table").click()
    page.wait_for_url(viewed("table"))
    page.go_back()
    page.wait_for_url(viewed("ranked"))
    expect(page.get_by_role("tab", name="Ranked")).to_have_attribute("aria-selected", "true")


@pytest.mark.parametrize("view", VIEWS)
def test_a_view_can_be_chosen_with_the_mouse(desktop, open_group, view):
    page = open_group(desktop, GROUP, "ranked")
    page.get_by_role("tab", name=VIEW_LABELS[view]).click()
    page.wait_for_url(viewed(view))
    expect(page.get_by_role("tab", name=VIEW_LABELS[view])).to_have_attribute(
        "aria-selected", "true")


@pytest.mark.parametrize("steps,view", list(enumerate(VIEWS[1:] + VIEWS[:1], start=1)))
def test_a_view_can_be_chosen_with_the_keyboard(desktop, open_group, steps, view):
    """Arrow keys rove the tablist; Enter on the focused tab selects it."""
    page = open_group(desktop, GROUP, "ranked")
    page.locator("#tab-ranked").focus()
    for _ in range(steps):
        page.keyboard.press("ArrowRight")
    assert page.evaluate("() => document.activeElement.id") == f"tab-{view}"
    page.keyboard.press("Enter")
    page.wait_for_url(viewed(view))
