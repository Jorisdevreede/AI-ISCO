"""The Skill scores page: the table of every skill, and one skill in full.

With nothing selected the page is a filterable, sortable, paged table of every
scored ESCO skill plus "How a skill is scored", which quotes rubric_v2.json.
Open a skill and it adds "How the model answered": one block per question, a
probability on every answer the model was offered, and the one it settled on
marked. Both of those belong to the shares scheme; under `?scorer=gemini` there
is no rubric to quote and no stored answers, and the page says nothing instead
of inventing something.
"""

import re
import unicodedata

import pytest
from playwright.sync_api import expect

from .conftest import GEMINI, JOB_TIMEOUT, SKILL_CLASS_NAMES, reveal_all

SKILL_TITLE = "manage budgets"
SKILL_URL = re.compile(r"/skill\.html#[0-9a-f]{8}$")

#: site/js/pages/skill-model.js: rows per page of the all-skills table.
PAGE_SIZE = 50

#: A query that matches many skills but nowhere near all of them.
QUERY = "budget"

LIST_READY = "#skill-table .list-table tbody tr"


def fold(text):
    """site/js/search.js's `fold`: the folding the table's name filter uses."""
    stripped = "".join(part for part in unicodedata.normalize("NFD", str(text))
                       if not unicodedata.combining(part))
    return " ".join(stripped.lower().split())


def used_in(row):
    return int(row.get("ne") or 0) + int(row.get("no") or 0)


def summary_text(first, last, total):
    noun = "skill" if total == 1 else "skills"
    return f"Showing {first:,}–{last:,} of {total:,} {noun}"


def open_needed_by(open_ready, session, skill_id):
    """Open a skill and scroll to its occupation lists.

    `skill_occupations.json` is three megabytes, so it is fetched when the
    "Needed by" section comes into view rather than on every skill.
    """
    page = open_ready(session, f"skill.html#{skill_id}", "#skill-detail .score-row")
    reveal_all(page)
    page.locator("#skill-detail .occ-link").first.wait_for(
        state="visible", timeout=JOB_TIMEOUT)
    return page


def wait_for_summary(page, text):
    page.wait_for_function(
        "wanted => document.querySelector('.list-summary').textContent === wanted",
        arg=text)


@pytest.fixture(scope="module")
def skill(skill_index_v2):
    """A published skill several occupations need, looked up by its ESCO name."""
    return next(row for row in skill_index_v2
                if row["t"] == SKILL_TITLE and row["ne"] > 0)


# --- one skill -------------------------------------------------------------


def test_searching_by_name_lands_on_the_skill_with_its_three_scores(desktop, skill):
    page = desktop.open("skill.html")
    page.get_by_label("Search ESCO skills by name").fill(skill["t"])
    options = page.locator("#skill-listbox li")
    options.first.wait_for()
    options.first.click()
    page.wait_for_url(SKILL_URL)
    assert page.url.endswith(f"#{skill['id']}")
    expect(page.locator("h1")).to_have_text(skill["t"])
    rows = page.locator(".score-row").all_inner_texts()
    assert len(rows) == 3
    for text, key in zip(rows, ["a", "m", "k"]):
        assert f"{skill[key]:.1f} out of 10" in text
    assert page.locator("#lookup-heading").inner_text() == "Look up another skill"


def test_the_skill_names_its_class_and_the_probabilities_behind_it(desktop, open_ready, skill):
    page = open_ready(desktop, f"skill.html#{skill['id']}", ".skill-class")
    chip = page.locator(".skill-class .class-chip").first
    assert chip.get_attribute("data-class") == skill["c"]
    assert chip.inner_text().startswith(SKILL_CLASS_NAMES[skill["c"]])
    probabilities = page.locator(".skill-class-probs").inner_text()
    for value in skill["p"]:
        assert f"{round(value * 100)}%" in probabilities


def test_the_first_occupation_that_needs_it_opens_a_job_page(desktop, open_ready, skill):
    page = open_needed_by(open_ready, desktop, skill["id"])
    first = page.locator("#skill-detail .occ-link").first
    slug = first.get_attribute("href").split("#", 1)[1]
    first.click()
    page.wait_for_url(re.compile(rf"/job\.html#{re.escape(slug)}$"), timeout=15_000)
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    assert page.locator("h1").inner_text().strip()


# --- how the model answered ------------------------------------------------


def test_how_the_model_answered_covers_every_question_it_was_asked(
        desktop, open_ready, skill, rubric_v2, skill_answers):
    page = open_ready(desktop, f"skill.html#{skill['id']}", "#answers-heading")
    stored = skill_answers(skill["id"])
    assert stored, "the example skill has no stored answers to show"

    blocks = page.locator("#skill-detail .answer")
    expect(blocks).to_have_count(len(rubric_v2["questions"]))
    assert blocks.locator("h3").all_inner_texts() == [
        question["label"] for question in rubric_v2["questions"]]


def test_every_question_carries_a_probability_on_every_answer(desktop, open_ready, skill):
    """The model answers with a probability for each level rather than a pick,
    so each question's bars are a distribution and have to add up."""
    page = open_ready(desktop, f"skill.html#{skill['id']}", "#answers-heading")
    blocks = page.locator("#skill-detail .answer")
    for position in range(blocks.count()):
        block = blocks.nth(position)
        shown = [int(text.rstrip("%"))
                 for text in block.locator(".answer-value").all_inner_texts()]
        assert shown, "a question with no answers should not be shown at all"
        # Each bar is rounded on its own, so the total may miss 100 by the
        # number of bars; it may never be a different distribution.
        assert abs(sum(shown) - 100) <= len(shown)


def test_the_answer_the_model_settled_on_is_the_one_marked(desktop, open_ready, skill):
    page = open_ready(desktop, f"skill.html#{skill['id']}", "#answers-heading")
    blocks = page.locator("#skill-detail .answer")
    for position in range(blocks.count()):
        block = blocks.nth(position)
        marked = block.locator("li.answer-bar.is-modal")
        expect(marked).to_have_count(1)
        shown = [int(text.rstrip("%"))
                 for text in block.locator(".answer-value").all_inner_texts()]
        assert int(marked.locator(".answer-value").inner_text().rstrip("%")) == max(shown)
        assert "the model’s answer" in marked.inner_text()


def test_the_answers_are_also_a_real_table(desktop, open_ready, skill):
    page = open_ready(desktop, f"skill.html#{skill['id']}", "#answers-heading")
    wrap = page.locator("#answers-table")
    expect(wrap).to_be_hidden()
    toggle = page.get_by_role("button", name="View as table")
    toggle.click()
    expect(wrap).to_be_visible()
    bars = page.locator("#skill-detail .answer .answer-bar").count()
    expect(wrap.locator("tbody tr")).to_have_count(bars)


# --- how a skill is scored -------------------------------------------------


def test_the_explainer_lists_every_question_the_rubric_asks(desktop, open_ready, rubric_v2):
    page = open_ready(desktop, "skill.html", "#scoring-heading")
    questions = page.locator("#skill-scoring .rubric-question")
    expect(questions).to_have_count(len(rubric_v2["questions"]))
    assert questions.locator("summary").all_inner_texts() == [
        question["label"] for question in rubric_v2["questions"]]
    assert rubric_v2["preamble"].strip() in page.locator(".rubric-preamble").inner_text()
    assert rubric_v2["model"] in page.locator("#skill-scoring .section-note").inner_text()


def test_the_class_rules_are_in_words_not_in_the_rubric_s_notation(desktop, open_ready):
    """The rubric states each rule as "SUB >= 0.5". That is the right form for
    the repository and the wrong one for a reader."""
    page = open_ready(desktop, "skill.html", "#scoring-heading")
    chips = page.locator(".rubric-class-list .class-chip")
    expect(chips).to_have_count(len(SKILL_CLASS_NAMES))
    assert chips.all_inner_texts() == list(SKILL_CLASS_NAMES.values())
    for rule in page.locator(".rubric-rule").all_inner_texts():
        assert rule.strip()
        assert not re.search(r"[<>]=?|\b[A-Z]{3,}\b|\d", rule), rule


# --- the table of every skill ----------------------------------------------


def test_the_table_opens_on_the_first_page_of_every_skill(desktop, open_ready, skill_index_v2):
    page = open_ready(desktop, "skill.html", LIST_READY)
    total = len(skill_index_v2)
    assert page.locator(".list-summary").inner_text() == summary_text(1, PAGE_SIZE, total)
    expect(page.locator(f"{LIST_READY}")).to_have_count(PAGE_SIZE)
    pages = -(-total // PAGE_SIZE)
    assert page.locator(".list-page").inner_text() == f"Page 1 of {pages:,}"


def test_filtering_by_name_and_by_class_narrows_the_table(desktop, open_ready, skill_index_v2):
    page = open_ready(desktop, "skill.html", LIST_READY)
    needle = fold(QUERY)

    page.fill("#list-query", QUERY)
    by_name = [row for row in skill_index_v2 if needle in fold(row["t"])]
    wait_for_summary(page, summary_text(1, min(PAGE_SIZE, len(by_name)), len(by_name)))
    assert 0 < len(by_name) < len(skill_index_v2)

    page.select_option("#list-class", "S")
    both = [row for row in by_name if row["c"] == "S"]
    wait_for_summary(page, summary_text(1, min(PAGE_SIZE, len(both)), len(both)))
    assert page.url.endswith("#list&q=budget&class=S")
    codes = page.locator(f"{LIST_READY} .class-chip").evaluate_all(
        "nodes => nodes.map(node => node.dataset.class)")
    assert set(codes) == {"S"}
    shown = page.locator(f"{LIST_READY} .class-chip").all_inner_texts()
    assert all(text.startswith(SKILL_CLASS_NAMES["S"]) for text in shown)
    assert desktop.unexpected_errors() == []


def test_a_column_sorts_both_ways_and_says_so(desktop, open_ready, skill_index_v2):
    page = open_ready(desktop, "skill.html", LIST_READY)
    header = page.locator('#skill-table th[data-key="n"]')
    counts = [used_in(row) for row in skill_index_v2]

    header.locator("button").click()
    wait_for_summary(page, summary_text(1, PAGE_SIZE, len(skill_index_v2)))
    assert header.get_attribute("aria-sort") == "descending"
    first_row = page.locator(f"{LIST_READY}").first
    assert first_row.locator("td").last.inner_text() == f"{max(counts):,}"

    header.locator("button").click()
    page.wait_for_function(
        "() => document.querySelector('#skill-table th[data-key=\"n\"]')"
        ".getAttribute('aria-sort') === 'ascending'")
    assert page.locator(f"{LIST_READY}").first.locator("td").last.inner_text() == (
        f"{min(counts):,}")


def test_the_hash_carries_the_whole_state_of_the_table(desktop, open_ready, skill_index_v2):
    """A filtered, sorted, paged view is something to share, so it is in the URL."""
    page = open_ready(desktop, "skill.html#list&class=S&sort=a&page=3", LIST_READY)
    rows = [row for row in skill_index_v2 if row["c"] == "S"]
    first = 2 * PAGE_SIZE + 1
    assert page.locator(".list-summary").inner_text() == summary_text(
        first, min(first + PAGE_SIZE - 1, len(rows)), len(rows))
    assert page.locator("#list-class").input_value() == "S"
    assert page.locator('#skill-table th[data-key="a"]').get_attribute("aria-sort") == (
        "descending")


def test_back_walks_the_filter_and_the_page(desktop, open_ready, skill_index_v2):
    page = open_ready(desktop, "skill.html", LIST_READY)
    everything = summary_text(1, PAGE_SIZE, len(skill_index_v2))

    page.select_option("#list-class", "M")
    mechanised = [row for row in skill_index_v2 if row["c"] == "M"]
    first_page = summary_text(1, min(PAGE_SIZE, len(mechanised)), len(mechanised))
    wait_for_summary(page, first_page)

    page.get_by_role("button", name="Next →").click()
    wait_for_summary(page, summary_text(
        PAGE_SIZE + 1, min(2 * PAGE_SIZE, len(mechanised)), len(mechanised)))
    assert page.url.endswith("#list&class=M&page=2")

    page.go_back()
    wait_for_summary(page, first_page)
    page.go_back()
    wait_for_summary(page, everything)
    assert desktop.unexpected_errors() == []


def test_a_filter_that_matches_nothing_says_so(desktop, open_ready):
    page = open_ready(desktop, "skill.html", LIST_READY)
    page.fill("#list-query", "zzzzqqqq")
    page.wait_for_function(
        "() => document.querySelector('.list-summary').textContent"
        " === 'No skill matches these filters.'")
    expect(page.locator(LIST_READY)).to_have_count(0)


# --- the same page under the older score set -------------------------------


def test_the_gemini_set_has_no_rubric_and_no_answers_to_show(desktop, open_ready, skill_index):
    """Those two sections are the shares scheme explaining itself. The older set
    was scored on two axes with no stored probabilities, so it shows neither —
    and the table drops the columns it has nothing to fill."""
    page = open_ready(desktop, f"skill.html{GEMINI}", LIST_READY)
    expect(page.locator("#scoring-heading")).to_have_count(0)
    expect(page.locator("#list-class")).to_have_count(0)
    assert page.locator('#skill-table th[data-key="a"] button').inner_text() == (
        "Automation risk")

    skill = next(row for row in skill_index if row["ne"] > 0)
    page = open_ready(desktop, f"skill.html{GEMINI}#{skill['id']}", ".score-row")
    expect(page.locator("#answers-heading")).to_have_count(0)
    expect(page.locator(".skill-class")).to_have_count(0)
    expect(page.locator(".score-row")).to_have_count(2)
    assert desktop.unexpected_errors() == []
