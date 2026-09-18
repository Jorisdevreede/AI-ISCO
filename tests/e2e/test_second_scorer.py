"""The score-set switch: which set is on, what it says, and what it borrows.

Three sets are published. The default is `v2` — TypeSafe's jev model on the
current rubric, four shares and seven types — and the switch offers Gemini's
older two-score set beside it. `?scorer=typesafe` is an alias kept for links
shared before v2 existed, and `_typesafe` itself is not offered: it exists only
for the method page's same-rubric comparison.

The jev model returns numbers and writes no text, so under the default every
explanation on screen is Gemini's, borrowed. Each one names the score it was
written for, and where that score is far from the one shown the sentence is
left out rather than stretched to fit. A checkout with only the Gemini set must
behave as a one-scorer site.
"""

import re

import pytest
from playwright.sync_api import expect

from .conftest import GEMINI, JOB_TIMEOUT, read_site_json, reveal_all, reveal_section

SLUG = "software-developer"

TYPESAFE_BUTTON = "TypeSafe · four shares"
GEMINI_BUTTON = "Gemini · two scores"

#: A borrowed rationale is only printed where the two runs are close enough for
#: it to be about the same skill. Well inside whatever the page's own cut is.
CLOSE_ENOUGH = 1.0


def open_first_card(page):
    card = page.locator("details.skill-card").first
    card.wait_for(state="visible", timeout=JOB_TIMEOUT)
    card.locator("summary").click()
    return card


def pressed(page):
    return page.locator(".scorer-toggle button[aria-pressed='true']")


@pytest.fixture(scope="module")
def borrowed_skill(skill_index_v2, skill_note):
    """A skill several occupations need whose borrowed rationale is printed:
    it has one, and the two runs scored it closely enough to show it."""
    for row in skill_index_v2:
        if row["ne"] <= 0:
            continue
        note = skill_note(row["id"])
        source = note and note.get("rf")
        if not note or not note.get("r") or not source:
            continue
        if abs((source.get("a") or 0) - row["a"]) <= CLOSE_ENOUGH:
            return row
    pytest.skip("no borrowed rationale close enough to be printed")
    return None


@pytest.fixture(scope="module")
def developer_rationales(unit_data, units_json):
    """The essential skills of the sample job, keyed by the title on screen."""
    shard = unit_data(units_json[SLUG])
    job = next(row for row in shard["occupations"] if row["s"] == SLUG)
    return {shard["skills"][key]["t"]: shard["skills"][key]
            for key in job.get("se", []) if key in shard["skills"]}


@pytest.fixture(scope="module")
def widest_gap(unit_data, units_json):
    """The skill in the sample unit group whose borrowed explanation was
    written for the score furthest from the one on screen, and a job that
    lists it. If anything is left out, this is."""
    shard = unit_data(units_json[SLUG])
    found = None
    for occupation in shard["occupations"]:
        for key in occupation.get("se", []):
            skill = shard["skills"].get(key)
            if not skill or not skill.get("r") or not skill.get("rf"):
                continue
            gap = abs(skill["rf"]["a"] - skill["a"])
            if found is None or gap > found[0]:
                found = (gap, occupation["s"], skill)
    if found is None:
        pytest.skip("no borrowed explanation in this unit group")
    return found[1], found[2]


# --- which set is on -------------------------------------------------------


def test_typesafe_is_the_default_and_the_switch_offers_both(desktop, open_ready):
    page = open_ready(desktop, "index.html", ".scorer-toggle")
    assert pressed(page).inner_text() == TYPESAFE_BUTTON
    labels = page.locator(".scorer-toggle button").all_inner_texts()
    assert labels == [TYPESAFE_BUTTON, GEMINI_BUTTON]
    assert "scorer=" not in page.url


def test_the_old_typesafe_link_is_an_alias_for_the_default_set(desktop, open_ready):
    """A link shared before v2 existed said ?scorer=typesafe and meant "the jev
    scores"; it lands on the jev scores of the current rubric."""
    page = open_ready(desktop, "index.html?scorer=typesafe", ".scorer-toggle")
    assert pressed(page).inner_text() == TYPESAFE_BUTTON
    assert "four shares" in page.locator("#scorer-note").inner_text()
    assert "seven types" in page.locator("#subtitle").inner_text()


@pytest.mark.parametrize("query,phrase", [("", "four shares"), (GEMINI, "original scores")])
def test_a_note_explains_whichever_set_is_on(desktop, open_ready, query, phrase):
    """Flipping the switch changes the whole way a job is described, not only
    the numbers, so both settings carry a note saying so."""
    page = open_ready(desktop, f"index.html{query}", ".scorer-toggle")
    note = page.locator("#scorer-note")
    assert note.is_visible()
    assert phrase in note.inner_text()
    assert page.locator(".scorer-toggle").get_attribute("aria-describedby") == "scorer-note"


def test_switching_to_gemini_adds_the_query_and_switching_back_removes_it(desktop, open_ready):
    page = open_ready(desktop, "index.html", ".scorer-toggle")
    page.locator(".scorer-toggle button", has_text="Gemini").click()
    page.wait_for_url(re.compile(r"/index\.html\?scorer=gemini$"))
    page.locator(".scorer-toggle").first.wait_for()
    assert pressed(page).inner_text() == GEMINI_BUTTON

    page.locator(".scorer-toggle button", has_text="TypeSafe").click()
    page.wait_for_url(re.compile(r"/index\.html$"))
    page.locator(".scorer-toggle").first.wait_for()
    assert pressed(page).inner_text() == TYPESAFE_BUTTON


def test_switching_reloads_the_job_with_the_other_scores(
        desktop, open_job, by_slug, by_slug_v2):
    page = reveal_all(open_job(desktop, f"job.html#{SLUG}"))
    expect(page.locator(".score-card--sub .score-value")).to_have_text(
        f"{by_slug_v2[SLUG]['a']:.1f}")

    page.locator(".scorer-toggle button", has_text="Gemini").click()
    page.wait_for_url(re.compile(rf"\?scorer=gemini#{SLUG}$"))
    page.locator("#job-article").wait_for(state="visible", timeout=JOB_TIMEOUT)
    reveal_all(page)

    assert by_slug[SLUG]["a"] != by_slug_v2[SLUG]["a"]
    expect(page.locator(".score-card--auto .score-value")).to_have_text(
        f"{by_slug[SLUG]['a']:.1f}")
    assert "original scores" in page.locator("#scorer-note").inner_text()
    assert desktop.unexpected_errors() == []


# --- the borrowed rationales ----------------------------------------------


#: "Gemini scored this 8.0 for AI substitution and wrote:"
LEAD = re.compile(r"^(\w+) scored this (\d+\.\d) for AI substitution and wrote:$")


def test_no_borrowed_explanation_is_printed_without_the_score_it_was_for(
        desktop, open_job, developer_rationales):
    """A sentence written for a very different score argues with the class
    beside it, so the page either names both scores or says why it is leaving
    the sentence out. It never just prints it."""
    page = reveal_all(open_job(desktop, f"job.html#{SLUG}"))
    cards = page.locator("#cards-list details.skill-card")
    expect(cards.first).to_be_visible(timeout=JOB_TIMEOUT)
    assert cards.count() == len(developer_rationales)

    printed = 0
    for position in range(cards.count()):
        card = cards.nth(position)
        skill = developer_rationales[card.locator(".sc-name").inner_text()]
        lead = card.locator(".rationale-lead")
        if lead.count():
            printed += 1
            writer, wrote_for = LEAD.match(lead.inner_text()).groups()
            assert writer == "Gemini"
            assert float(wrote_for) == skill["rf"]["a"]
            assert card.locator(".rationale-after").inner_text() == (
                f"This page shows {skill['a']:.1f}.")
        else:
            missing = card.locator(".rationale-missing").inner_text()
            assert f"{skill['a']:.1f}" in missing or "no rationale" in missing
    assert printed, "no borrowed explanation was printed at all"
    assert "Gemini" in page.locator("#cards-writer").inner_text()


def test_the_explanation_furthest_from_its_score_is_left_out_and_says_so(
        desktop, open_job, widest_gap):
    """A disclaimer does not stop a reader taking the sentence as the
    explanation of the label, so past some gap the sentence is not shown."""
    slug, skill = widest_gap
    page = reveal_all(open_job(desktop, f"job.html#{slug}"))
    card = page.locator("#cards-list details.skill-card").filter(
        has=page.locator(".sc-name", has_text=skill["t"])).first
    card.wait_for(state="visible", timeout=JOB_TIMEOUT)

    body = card.inner_text()
    assert skill["r"] not in body, "the furthest explanation was printed anyway"
    missing = card.locator(".rationale-missing").inner_text()
    assert f"{skill['rf']['a']:.1f}" in missing and f"{skill['a']:.1f}" in missing
    assert "Gemini" in missing


def test_the_skill_page_says_which_score_a_borrowed_explanation_was_written_for(
        desktop, open_ready, borrowed_skill, skill_note):
    page = open_ready(desktop, f"skill.html#{borrowed_skill['id']}", "blockquote.rationale")
    quote = page.locator("blockquote.rationale")
    source = skill_note(borrowed_skill["id"])["rf"]

    writer, wrote_for = LEAD.match(quote.locator(".rationale-lead").inner_text()).groups()
    assert writer == "Gemini"
    assert float(wrote_for) == source["a"]
    assert f"This page shows {borrowed_skill['a']:.1f}." in (
        quote.locator(".rationale-source").inner_text())


def test_the_gemini_set_shows_its_own_rationale_with_nothing_borrowed(desktop, open_job):
    page = open_job(desktop, f"job.html{GEMINI}#{SLUG}")
    open_first_card(page)

    assert page.locator(".rationale-lead").count() == 0
    assert page.locator(".rationale-after").count() == 0
    assert "what the model wrote" in page.locator("#cards-writer").inner_text()


# --- the same-rubric comparison on the method page -------------------------


def same_box_share(first, second):
    """How far the two scorings of the ORIGINAL rubric agree, as the page counts."""
    scored = {row["s"]: row for row in first
              if isinstance(row.get("a"), (int, float))
              and isinstance(row.get("m"), (int, float))}
    other = {row["s"]: row for row in second
             if isinstance(row.get("a"), (int, float))
             and isinstance(row.get("m"), (int, float))}
    shared = [slug for slug in scored if slug in other]
    same = sum(1 for slug in shared if scored[slug]["q"] == other[slug]["q"])
    return same, len(shared)


@pytest.fixture(scope="module")
def typesafe_index():
    return read_site_json("search_index_typesafe")


def open_agreement(session, open_ready):
    """The comparison costs two more index files, so it is fetched when the
    section it explains comes into view."""
    page = open_ready(session, "method.html", "h1")
    reveal_section(page, "#agreement")
    page.locator("#agreement-body").wait_for(state="visible", timeout=JOB_TIMEOUT)
    page.locator("#agreement-table td").first.wait_for(
        state="visible", timeout=JOB_TIMEOUT)
    return page


def test_the_method_page_computes_the_agreement_from_both_sets(
        desktop, open_ready, search_index, typesafe_index, stats_v2):
    """The comparison is about the older rubric scored twice, so it stands
    whichever set the switch is on."""
    page = open_agreement(desktop, open_ready)
    same, n = same_box_share(search_index, typesafe_index)
    text = page.locator("#agreement").inner_text()

    assert f"{round(100 * same / n)}%" in text
    assert f"{stats_v2['occupations']:,}" in text
    assert f"{n - same:,}" in text
    assert page.locator("#agreement-table tbody td").count() == len(
        page.locator("#agreement-table thead th").all()) ** 2
    diagonal = page.locator("#agreement-table td.agrees").all_inner_texts()
    assert sum(int(cell.replace(",", "")) for cell in diagonal) == same
    assert desktop.unexpected_errors() == []


def test_the_agreement_section_fits_a_phone(phone, open_ready):
    open_agreement(phone, open_ready)
    assert not phone.overflows()


# --- a checkout that never ran the jev scores ------------------------------


def test_a_site_with_one_score_set_offers_no_switch_and_no_comparison(single_set, open_ready):
    page = open_ready(single_set, "method.html", "#quadrant-table tbody tr")

    assert page.locator(".scorer-toggle").count() == 0
    assert page.locator("#scorer-note").count() == 0
    assert not page.locator("#agreement").is_visible()
    assert single_set.unexpected_errors() == []


def test_a_site_with_one_score_set_renders_the_gemini_scheme(
        single_set, open_job, by_slug, stats_json):
    page = reveal_all(open_job(single_set, f"job.html#{SLUG}"))

    assert page.locator(".score-card--auto .score-value").inner_text() == (
        f"{by_slug[SLUG]['a']:.1f}")
    assert page.locator("#job-meta .quadrant-badge").get_attribute("data-quadrant") == (
        by_slug[SLUG]["q"])
    assert not page.locator("#job-shares").is_visible()
    assert f"cut-off of {stats_json['threshold']}" in (
        page.locator("#scatter-threshold").inner_text())
    assert single_set.unexpected_errors() == []


def test_asking_for_a_missing_second_set_falls_back_to_the_default(
        single_set, open_job, by_slug):
    page = reveal_all(open_job(single_set, f"job.html?scorer=typesafe#{SLUG}"))

    assert f"{by_slug[SLUG]['a']:.1f}" in page.locator("#job-article").inner_text()
    assert page.locator("#scorer-note").count() == 0
    assert single_set.unexpected_errors() == []
