"""The second score set: the switch, the borrowed rationales, the agreement.

The published site carries two score sets. The default is Gemini's; the
"Scores from" switch (or ?scorer=typesafe) shows the same rubric scored by a
model that returns numbers only, so the texts on screen are still Gemini's and
say so. A checkout with only the default set must behave as a one-scorer site.
"""

import json
import re

import pytest

from .conftest import JOB_TIMEOUT, SITE

SLUG = "software-developer"
QUADRANTS = ("TRANSFORM", "STABLE", "EVOLVE", "SHRINK")


@pytest.fixture(scope="module")
def second_index():
    text = (SITE / "search_index_typesafe.json").read_text(encoding="utf-8")
    return {row["s"]: row for row in json.loads(text)}


def open_first_card(page):
    card = page.locator("details.skill-card").first
    card.wait_for(state="visible", timeout=JOB_TIMEOUT)
    card.locator("summary").click()
    return card


def test_the_switch_is_offered_and_gemini_is_the_default(desktop, open_ready):
    page = open_ready(desktop, "index.html", ".scorer-toggle")
    pressed = page.locator(".scorer-toggle button[aria-pressed='true']")
    assert pressed.inner_text() == "Gemini"
    assert page.locator(".scorer-note").count() == 0


def test_switching_reloads_with_the_other_scores_and_a_banner(
        desktop, open_job, by_slug, second_index):
    page = open_job(desktop, f"job.html#{SLUG}")
    page.locator(".scorer-toggle button", has_text="TypeSafe").click()
    page.wait_for_url(re.compile(rf"\?scorer=typesafe#{SLUG}$"))
    page.locator("#job-article").wait_for(state="visible", timeout=JOB_TIMEOUT)

    note = page.locator(".scorer-note").inner_text()
    assert "numbers only" in note and "Gemini wrote" in note
    shown = page.locator("#job-article").inner_text()
    assert f"{second_index[SLUG]['a']:.1f}" in shown
    assert second_index[SLUG]["a"] != by_slug[SLUG]["a"]
    assert desktop.unexpected_errors() == []


def test_a_borrowed_rationale_carries_an_i_that_names_its_writer(desktop, open_job):
    page = open_job(desktop, f"job.html?scorer=typesafe#{SLUG}")
    card = open_first_card(page)
    button = card.locator("button.info-button")
    note = card.locator(".info-note-text")

    assert "written by Gemini" in button.get_attribute("aria-label")
    assert not note.is_visible()
    button.focus()
    page.keyboard.press("Enter")
    assert note.is_visible()
    assert button.get_attribute("aria-expanded") == "true"
    assert note.inner_text().startswith("Gemini wrote this explanation for its own scores")
    assert "written by Gemini" in page.locator("#cards-writer").inner_text()


def test_the_default_scores_show_their_own_rationale_without_an_i(desktop, open_job):
    page = open_job(desktop, f"job.html#{SLUG}")
    open_first_card(page)

    assert page.locator("button.info-button").count() == 0
    assert "what the model wrote" in page.locator("#cards-writer").inner_text()


def test_the_skill_page_says_who_wrote_a_borrowed_explanation(desktop, open_ready, skill_index):
    skill = next(row for row in skill_index if row["ne"] > 0)
    page = open_ready(desktop, f"skill.html?scorer=typesafe#{skill['id']}",
                      "blockquote.rationale")
    quote = page.locator("blockquote.rationale")

    assert "Written by Gemini" in quote.inner_text()
    quote.locator("button.info-button").click()
    assert quote.locator(".info-note-text").is_visible()


def same_quadrant_share(first, second):
    shared = [slug for slug in first if slug in second]
    same = sum(1 for slug in shared if first[slug]["q"] == second[slug]["q"])
    return same, len(shared)


def test_the_method_page_computes_the_agreement_from_both_sets(
        desktop, open_ready, by_slug, second_index):
    page = open_ready(desktop, "method.html", "#agreement")
    same, total = same_quadrant_share(by_slug, second_index)
    text = page.locator("#agreement").inner_text()

    assert f"{round(100 * same / total)}%" in text
    assert f"{total:,}" in text and f"{total - same:,}" in text
    assert page.locator("#agreement-table tbody td").count() == len(QUADRANTS) ** 2
    diagonal = page.locator("#agreement-table td.agrees").all_inner_texts()
    assert sum(int(cell.replace(",", "")) for cell in diagonal) == same
    assert desktop.unexpected_errors() == []


def test_the_agreement_section_fits_a_phone(phone, open_ready):
    open_ready(phone, "method.html", "#agreement")
    assert not phone.overflows()


def test_a_site_with_one_score_set_offers_no_switch_and_no_comparison(single_set, open_ready):
    page = open_ready(single_set, "method.html", "#quadrant-table tbody tr")

    assert page.locator(".scorer-toggle").count() == 0
    assert not page.locator("#agreement").is_visible()
    assert single_set.unexpected_errors() == []


def test_asking_for_a_missing_second_set_falls_back_to_the_default(
        single_set, open_job, by_slug):
    page = open_job(single_set, f"job.html?scorer=typesafe#{SLUG}")

    assert f"{by_slug[SLUG]['a']:.1f}" in page.locator("#job-article").inner_text()
    assert page.locator(".scorer-note").count() == 0
    assert single_set.unexpected_errors() == []
