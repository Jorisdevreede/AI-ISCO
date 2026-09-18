""""How sure is this?" under the default set.

The page's prose comes in two versions: an element carrying `data-scheme`
belongs to one of them, everything else to both. Under the default the shares
half is on screen, the four-box half is not, and every figure in it is read out
of stats_v2.json as the page loads — so a rebuild of the data rewrites the page
instead of contradicting it.

`?scorer=gemini`'s half of this lives in test_quadrants.py, and the two-run
comparison in test_second_scorer.py.
"""

import pytest
from playwright.sync_api import expect

from .conftest import (
    JOB_TIMEOUT,
    SKILL_CLASS_NAMES,
    SKILL_CLASS_ORDER,
    TYPE_NAMES,
    reveal_section,
)

READY = "#type-table tbody tr"


def stat(page, name):
    return page.locator(f"[data-stat='{name}']").first.inner_text()


def test_the_shares_prose_is_shown_and_the_four_box_prose_is_not(desktop, open_ready):
    page = open_ready(desktop, "method.html", READY)
    for node in page.locator("[data-scheme='shares']").all():
        assert node.is_visible(), node.inner_text()[:60]
    for node in page.locator("[data-scheme='quadrants']").all():
        assert not node.is_visible(), node.inner_text()[:60]
    assert page.locator("#types-heading").is_visible()
    assert not page.locator("#boxes-heading").is_visible()
    assert desktop.unexpected_errors() == []


def test_the_type_table_gives_every_type_its_rule_and_its_count(
        desktop, open_ready, stats_v2):
    page = open_ready(desktop, "method.html", READY)
    order = stats_v2["types"]["order"]
    rows = page.locator("#type-table tbody tr")
    expect(rows).to_have_count(len(order))
    for position, code in enumerate(order):
        row = rows.nth(position)
        assert row.locator("th").inner_text() == TYPE_NAMES[code]
        assert row.locator("td").nth(0).inner_text().strip(), f"{code} has no rule"
        assert row.locator("td").nth(1).inner_text() == (
            f"{stats_v2['types']['counts'][code]:,}")
        assert row.locator("td").nth(2).inner_text() == (
            f"{round(stats_v2['types']['shares'][code] * 100)}%")


def test_the_class_table_gives_every_class_its_rule_and_its_count(
        desktop, open_ready, stats_v2):
    page = open_ready(desktop, "method.html", READY)
    rows = page.locator("#class-table tbody tr")
    expect(rows).to_have_count(len(SKILL_CLASS_ORDER))
    for position, code in enumerate(SKILL_CLASS_ORDER):
        row = rows.nth(position)
        assert row.locator("th").inner_text() == SKILL_CLASS_NAMES[code]
        assert row.locator("td").nth(0).inner_text().strip(), f"{code} has no rule"
        assert row.locator("td").nth(1).inner_text() == (
            f"{stats_v2['skill_classes']['counts'][code]:,}")


def test_every_figure_in_the_prose_comes_out_of_the_stats_file(
        desktop, open_ready, stats_v2):
    page = open_ready(desktop, "method.html", READY)
    assert stat(page, "skills") == f"{stats_v2['skills_scored']:,}"
    assert stat(page, "occupations") == f"{stats_v2['occupations']:,}"
    assert stat(page, "model") == stats_v2["model"]
    assert stat(page, "built") == stats_v2["built"]
    assert stat(page, "mixedCount") == f"{stats_v2['types']['counts']['MIXED']:,}"
    assert stat(page, "mixedShare") == (
        f"{round(stats_v2['types']['shares']['MIXED'] * 100)}%")
    assert stat(page, "nearLineCount") == f"{stats_v2['near_line']['count']:,}"
    assert stat(page, "nearLineShare") == (
        f"{round(stats_v2['near_line']['share'] * 100)}%")


def test_a_class_is_cut_on_a_probability_not_on_the_older_scheme_s_score(
        desktop, open_ready, stats_v2, stats_json):
    """A class is read off a model probability, so the cut is the one that set
    published — never the 1-10 cut-off the four boxes use."""
    assert stats_v2["threshold"] != stats_json["threshold"]
    page = open_ready(desktop, "method.html", READY)
    assert stat(page, "classCut") == str(stats_v2["threshold"])
    assert str(stats_json["threshold"]) not in stat(page, "classCut")


def test_the_agreement_note_is_one_computed_slot(desktop, open_ready, stats_v2):
    """The old n / total / moved spans said the same thing three times and
    disagreed with the count two sections above."""
    page = open_ready(desktop, "method.html", READY)
    reveal_section(page, "#agreement")
    # The body stays hidden until both index files are in, so waiting on it is
    # what stops this reading an empty slot and calling it a pass.
    page.locator("#agreement-body").wait_for(state="visible", timeout=JOB_TIMEOUT)
    note = page.locator("[data-agree='note']")
    expect(note).to_have_count(1)
    expect(note).not_to_have_text("—")
    assert f"{stats_v2['occupations']:,}" in note.inner_text()


def test_no_page_error_on_a_phone(phone, open_ready):
    open_ready(phone, "method.html", READY)
    assert not phone.overflows()
    assert phone.unexpected_errors() == []


@pytest.mark.parametrize("heading", ["asked-heading", "class-heading", "types-heading",
                                     "choices-heading", "near-heading"])
def test_the_shares_sections_are_all_there(desktop, open_ready, heading):
    page = open_ready(desktop, "method.html", READY)
    assert page.locator(f"#{heading}").is_visible()
