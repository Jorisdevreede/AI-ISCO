"""The four shares, wherever they are drawn, and the badge that explains them.

Rounding each of four shares on its own gives 99 or 101 often enough that a
visitor notices, so the site gives the remainder to the largest fractions. That
promise is only worth anything if it holds on every figure: the job page, the
tree's detail pane, a group's mean make-up and the compact bar inside a list
row all have to add up.

The badge beside a job is the only place the site explains a type, and it may
explain it only from that job's own data: the rule it met, its own four
percentages, and whether it sits near a cut-off.
"""

import re

import pytest
from playwright.sync_api import expect

from .conftest import TYPE_NAMES, reveal_all, share_percents

GROUP = "unit:2512"
SLUG = "software-developer"

PERCENT = re.compile(r"(\d+)%")


def open_named_job(open_job, session, row):
    """Open one job and wait for that job, not the one before it.

    Moving from `job.html#a` to `job.html#b` is a fragment navigation: the
    document stays, `#job-article` never goes away, and the new job's file is
    still on its way. The heading is what proves the swap happened.
    """
    page = open_job(session, f"job.html#{row['s']}")
    expect(page.locator("#job-title")).to_have_text(row["t"])
    return reveal_all(page)


def legend_percents(bar):
    """The four numbers a full shares legend prints."""
    return [int(text.rstrip("%"))
            for text in bar.locator(".shares-legend .shares-value").all_inner_texts()]


def sentence_percents(bar):
    """The same four numbers, as a compact bar says them in one line."""
    return [int(value) for value in PERCENT.findall(bar.locator(".shares-line").inner_text())]


@pytest.fixture(scope="module")
def one_job_per_type(search_index_v2, stats_v2):
    """One published job of each of the first few types, so the checks below
    cover more than one shape of split without opening thousands of pages."""
    picked = {}
    for row in search_index_v2:
        picked.setdefault(row["q"], row)
    return [picked[code] for code in stats_v2["types"]["order"][:3] if code in picked]


@pytest.fixture(scope="module")
def near_and_far(search_index_v2):
    near = next(row for row in search_index_v2 if row.get("nl"))
    far = next(row for row in search_index_v2 if not row.get("nl"))
    return near, far


# --- the bar adds up -------------------------------------------------------


def test_the_job_page_bar_adds_up(desktop, open_job, one_job_per_type):
    assert one_job_per_type, "no types in the published stats file"
    for row in one_job_per_type:
        page = open_named_job(open_job, desktop, row)
        shown = legend_percents(page.locator("#job-shares .shares").first)
        assert shown == share_percents(row["sh"])
        assert sum(shown) == 100, f"{row['s']} adds up to {sum(shown)}"


def test_the_tree_pane_bar_adds_up(desktop, by_slug_v2):
    page = desktop.open(f"tree.html#job={SLUG}")
    page.wait_for_selector(".shares-block .shares-legend")
    shown = legend_percents(page.locator(".shares-block .shares").first)
    assert shown == share_percents(by_slug_v2[SLUG]["sh"])
    assert sum(shown) == 100


def test_the_group_bars_add_up(desktop, open_group, groups_v2):
    page = open_group(desktop, GROUP, "ranked")
    mean = legend_percents(page.locator("#mean-shares .shares").first)
    assert mean == share_percents(groups_v2[GROUP]["sh"])
    assert sum(mean) == 100

    rows = page.locator("#panel ol.ranked li .shares--compact")
    assert rows.count() > 0
    for position in range(min(3, rows.count())):
        assert sum(sentence_percents(rows.nth(position))) == 100


# --- the job's skills, grouped by the class the model gave each one --------


@pytest.fixture(scope="module")
def developer_classes(unit_data, units_json):
    """How the sample job's own skills fall into the four classes."""
    shard = unit_data(units_json[SLUG])
    job = next(row for row in shard["occupations"] if row["s"] == SLUG)
    counts = {"S": 0, "A": 0, "M": 0, "I": 0, "total": 0}
    for key in job.get("se", []) + job.get("so", []):
        skill = shard["skills"].get(key)
        if not skill:
            continue
        counts["total"] += 1
        if skill.get("c") in counts:
            counts[skill["c"]] += 1
    return counts


def test_the_job_lists_one_section_per_class_that_has_members(
        desktop, open_job, developer_classes):
    """Under the shares scheme a skill is classified, not cut at a score, so
    the page has one list per class rather than two lists cut at 6."""
    page = open_job(desktop, f"job.html#{SLUG}")
    for code in ("S", "A", "M"):
        rows = page.locator(f"#class-{code}-list li")
        expect(rows).to_have_count(developer_classes[code])
        if developer_classes[code]:
            assert page.locator(f"#class-{code}-heading").is_visible()
    assert page.locator("#skill-columns .skill-list").count() == sum(
        1 for code in ("S", "A", "M") if developer_classes[code])


def test_the_stays_human_note_counts_the_skills_no_list_covers(
        desktop, open_job, developer_classes):
    page = open_job(desktop, f"job.html#{SLUG}")
    note = page.locator("#stays-human")
    if not developer_classes["I"]:
        expect(note).to_be_hidden()
        return
    text = note.inner_text()
    assert f"{developer_classes['I']:,}" in text
    assert f"{developer_classes['total']:,}" in text
    assert "stay human" in text


# --- the badge explains the type from the job's own data -------------------


def badge_text(page):
    badge = page.locator("#job-meta .quadrant-badge")
    badge.click()
    popover = page.locator("#job-meta .badge-popover")
    expect(popover).to_be_visible()
    return badge, popover


def test_the_badge_names_the_rule_the_job_met(desktop, open_job, by_slug_v2):
    page = reveal_all(open_job(desktop, f"job.html#{SLUG}"))
    row = by_slug_v2[SLUG]
    _, popover = badge_text(page)
    text = popover.inner_text()

    assert popover.locator("h3").inner_text() == f'Why "{TYPE_NAMES[row["q"]]}"?'
    assert f'the rule for "{TYPE_NAMES[row["q"]]}"' in text
    for percent in share_percents(row["sh"]):
        assert f"{percent}%" in text
    assert "model estimates" in text
    assert popover.locator("a").get_attribute("href").startswith("method.html")


def test_the_badge_never_says_the_skills_agreed(desktop, open_job, one_job_per_type):
    """An earlier wording said a job's skills "agree", which is a claim about
    consistency the run never measured."""
    for row in one_job_per_type:
        page = open_named_job(open_job, desktop, row)
        _, popover = badge_text(page)
        text = popover.inner_text().lower()
        assert "agree" not in text, f"{row['s']}: {text}"
        assert "at risk" not in text


def test_escape_closes_the_badge_and_gives_the_focus_back(desktop, open_job):
    page = open_job(desktop, f"job.html#{SLUG}")
    badge, popover = badge_text(page)
    page.keyboard.press("Escape")
    expect(popover).to_be_hidden()
    assert badge.get_attribute("aria-expanded") == "false"
    assert page.evaluate("() => document.activeElement.className").startswith(
        "quadrant-badge")


# --- near a cut-off, and not ----------------------------------------------


def test_a_job_near_a_cut_off_says_so_and_one_that_is_not_says_that(
        desktop, open_job, near_and_far):
    near, far = near_and_far

    page = open_named_job(open_job, desktop, near)
    marker = page.locator("#job-meta .quadrant-badge-near")
    expect(marker).to_have_count(1)
    assert marker.inner_text() == "near the line"
    _, popover = badge_text(page)
    assert "sits near a cut-off" in popover.inner_text()

    page = open_named_job(open_job, desktop, far)
    expect(page.locator("#job-meta .quadrant-badge-near")).to_have_count(0)
    _, popover = badge_text(page)
    assert "does not hang on a rounding" in popover.inner_text()


def test_mixed_is_explained_as_a_residual_class(desktop, open_job, search_index_v2):
    """Mixed is what is left when no rule matched, not a finding that AI will
    leave the job alone."""
    row = next((row for row in search_index_v2 if row["q"] == "MIXED"), None)
    if row is None:
        pytest.skip("no occupation came out Mixed in this run")
    page = open_named_job(open_job, desktop, row)
    _, popover = badge_text(page)
    text = popover.inner_text()
    assert "No rule matched" in text
    assert "residual class" in text
