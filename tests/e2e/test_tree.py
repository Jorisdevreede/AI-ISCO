"""End-to-end checks for site/tree.html — the ISCO-08 tree and its detail pane.

The default set splits a job into four shares and gives it one of seven types,
so the detail pane leads with a shares bar and every mix bar has seven
segments. Nothing here fetches a whole score set: a job's skills come from its
own unit group's shard.

Run:  AIISCO_E2E_CHANNEL=chrome uv run pytest tests/e2e/test_tree.py -q
"""

from itertools import pairwise

import pytest

from .conftest import SKILL_CLASS_NAMES, TYPE_NAMES, TYPE_SHORT, share_percents

CHAIN = ["major:2", "sub:25", "minor:251", "unit:2512"]
UNIT = "2512"
SLUG = "software-developer"

#: How many matching jobs tree-model.js draws before it asks for a narrower word.
FILTER_LIMIT = 300

BASE_TITLE = "Job tree — AI-ISCO"


@pytest.fixture(scope="module")
def developer_skills(unit_data):
    """The skills software developer really has in its unit group's shard."""
    shard = unit_data(UNIT)
    job = next(row for row in shard["occupations"] if row["s"] == SLUG)
    known = shard["skills"]
    return {
        "essential": [key for key in job.get("se", []) if key in known],
        "optional": [key for key in job.get("so", []) if key in known],
        "rows": [known[key] for key in job.get("se", []) + job.get("so", []) if key in known],
    }


def item(page, node_id):
    return page.locator(f'[data-id="{node_id}"]')


def drill(page, keys):
    """Click each group in turn, waiting for it to open."""
    for key in keys:
        item(page, key).click()
        item(page, key).and_(page.locator('[aria-expanded="true"]')).wait_for()


def jobs_in_unit(index, code):
    titles = [row["t"] for row in index if row["c"] == code]
    return sorted(titles, key=str.casefold)


def announced_count(page):
    """The number in "1,234 jobs match", whichever way it is worded."""
    text = page.locator("#tree-count").inner_text()
    return int(text.split()[0].replace(",", ""))


def test_the_tree_opens_on_the_ten_major_groups(desktop):
    page = desktop.open("tree.html")
    items = page.locator('[role="treeitem"]')
    items.first.wait_for()
    assert items.count() == 10
    assert page.locator(".tree-item--job").count() == 0
    assert page.locator("#tree").get_attribute("role") == "tree"
    assert item(page, "major:2").get_attribute("aria-expanded") == "false"
    assert item(page, "major:2").get_attribute("aria-level") == "1"
    assert item(page, "major:2").get_attribute("aria-setsize") == "10"
    assert page.title() == BASE_TITLE
    assert desktop.unexpected_errors() == []


def test_the_key_says_what_the_figure_beside_a_job_is(desktop):
    """Sighted visitors had only the screen-reader label to go on."""
    page = desktop.open("tree.html")
    page.locator('[role="treeitem"]').first.wait_for()
    assert "AI can take over" in page.locator("#tree-key").inner_text()


def test_a_mouse_drill_reaches_a_job_and_its_skills(desktop, developer_skills):
    page = desktop.open("tree.html")
    drill(page, CHAIN)
    item(page, f"job:{SLUG}").click()
    page.wait_for_selector(".skill-table")
    assert page.locator(".skill-table tbody tr").count() == len(developer_skills["rows"])
    assert page.locator("#detail-title").inner_text() == SLUG.replace("-", " ")
    assert page.title() == f"{SLUG.replace('-', ' ')} — {BASE_TITLE}"
    assert desktop.unexpected_errors() == []


def test_a_keyboard_drill_reaches_a_job(desktop, groups_v2, search_index_v2):
    page = desktop.open("tree.html")
    item(page, "major:2").first.wait_for()
    item(page, "major:2").focus()
    for parent, child in pairwise(CHAIN):
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        for _ in range(groups_v2[parent]["children"].index(child)):
            page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    for _ in range(jobs_in_unit(search_index_v2, UNIT).index(SLUG.replace("-", " "))):
        page.keyboard.press("ArrowDown")
    assert page.evaluate("document.activeElement.dataset.id") == f"job:{SLUG}"
    page.keyboard.press("Enter")
    page.wait_for_function(f"() => location.hash === '#job={SLUG}'")
    assert page.locator("#detail-title").inner_text() == SLUG.replace("-", " ")
    assert desktop.unexpected_errors() == []


def test_a_deep_link_opens_with_the_job_selected(desktop):
    page = desktop.open(f"tree.html#job={SLUG}")
    leaf = item(page, f"job:{SLUG}")
    leaf.wait_for()
    assert leaf.get_attribute("aria-selected") == "true"
    assert leaf.is_visible()
    assert leaf.get_attribute("aria-level") == "5"
    for key in CHAIN:
        assert item(page, key).get_attribute("aria-expanded") == "true"
    assert page.locator("#detail-title").inner_text() == SLUG.replace("-", " ")
    assert desktop.unexpected_errors() == []


def test_a_leaf_names_its_type_and_the_share_ai_can_take_over(desktop, by_slug_v2):
    page = desktop.open(f"tree.html#job={SLUG}")
    leaf = item(page, f"job:{SLUG}")
    leaf.wait_for()
    row = by_slug_v2[SLUG]
    assert leaf.locator(".tree-quad").get_attribute("data-type") == row["q"]
    assert leaf.locator(".tree-quad").inner_text() == TYPE_SHORT[row["q"]]
    assert leaf.locator(".tree-scores").inner_text() == (
        f"{share_percents(row['sh'])[0]}% AI can take over")


def test_back_walks_the_selections(desktop):
    page = desktop.open("tree.html")
    item(page, "major:2").first.wait_for()
    item(page, "major:2").click()
    item(page, "sub:25").click()
    assert page.evaluate("location.hash") == "#g=sub:25"
    page.go_back()
    page.wait_for_function("() => location.hash === '#g=major:2'")
    assert item(page, "major:2").get_attribute("aria-selected") == "true"
    assert page.locator("#detail-title").inner_text() == groups_label(page)
    assert desktop.unexpected_errors() == []


def groups_label(page):
    return page.locator('[data-id="major:2"] .tree-label').first.inner_text()


def test_the_filter_answers_with_a_list_and_announces_a_count(desktop):
    """A filter is a question about jobs, so it is answered with jobs; the
    hierarchy that holds them waits behind a disclosure."""
    page = desktop.open("tree.html")
    item(page, "major:2").first.wait_for()
    summary = page.locator("#tree-hierarchy-summary")
    assert not summary.is_visible()

    page.fill("#tree-q", "programmer")
    page.wait_for_function(
        "() => document.getElementById('tree-count').textContent.includes('match')"
    )
    total = announced_count(page)
    results = page.locator("#tree-results-list li")
    assert results.count() == min(total, FILTER_LIMIT)
    assert SLUG.replace("-", " ") in [
        text.strip() for text in results.locator(".result-title").all_inner_texts()]
    assert page.evaluate("location.hash") == "#q=programmer"
    assert summary.is_visible()
    assert page.locator("#tree-hierarchy").get_attribute("open") is None

    page.fill("#tree-q", "")
    page.wait_for_function("() => document.getElementById('tree-count').textContent === ''")
    assert page.locator('[role="treeitem"]').count() == 10
    assert not page.locator("#tree-results").is_visible()
    assert not summary.is_visible()
    assert desktop.unexpected_errors() == []


def test_a_result_says_why_a_job_without_the_word_is_in_the_list(desktop):
    page = desktop.open("tree.html#q=programmer")
    page.locator("#tree-results-list .result-link").first.wait_for()
    also = page.locator("#tree-results-list .result-also").first
    assert "programmer" in also.inner_text()


def test_a_group_shows_the_seven_types_from_groups_json(desktop, groups_v2, stats_v2):
    page = desktop.open(f"tree.html#g=unit:{UNIT}")
    page.wait_for_selector("#detail-body .mix-bars")
    group = groups_v2[f"unit:{UNIT}"]
    order = stats_v2["types"]["order"]
    total = sum(group["q"].get(code, 0) for code in order)
    assert f"{group['n']} jobs" in page.locator("#detail-body p").first.inner_text()
    names = page.locator("#detail-body .mix-name").all_inner_texts()
    assert names == [TYPE_NAMES[code] for code in order]
    figures = page.locator("#detail-body .mix-figure").all_inner_texts()
    for text, code in zip(figures, order):
        assert text.startswith(f"{group['q'].get(code, 0)} of {total} jobs")
    assert desktop.unexpected_errors() == []


def test_the_empty_pane_splits_every_job_and_every_skill(desktop, stats_v2):
    page = desktop.open("tree.html")
    page.wait_for_selector("#detail-body .mix-bars")
    headings = page.locator("#detail-body h3").all_inner_texts()
    assert f"How all {stats_v2['occupations']:,} jobs split" in headings
    assert f"How all {stats_v2['skills_scored']:,} skills split" in headings
    classes = page.locator("#detail-body .mix-bars").nth(1)
    assert classes.locator(".mix-name").all_inner_texts() == list(SKILL_CLASS_NAMES.values())
    assert desktop.unexpected_errors() == []


def test_a_job_pane_says_what_stays_human_is_made_of(desktop):
    page = desktop.open(f"tree.html#job={SLUG}")
    page.wait_for_selector(".skill-table")
    why = page.locator("#why-line")
    assert why.is_visible()
    assert "stays human" in why.inner_text()


def test_every_link_out_of_the_pane_is_real(desktop):
    page = desktop.open(f"tree.html#job={SLUG}")
    page.wait_for_selector(".skill-table")
    full = page.get_by_role("link", name="Open the full job page")
    assert full.get_attribute("href") == f"job.html#{SLUG}&from=unit:{UNIT}"
    assert page.locator("#detail-body a").first.get_attribute("href") == f"tree.html#g=unit:{UNIT}"
    skill = page.locator(".skill-table tbody a").first.get_attribute("href")
    assert skill.startswith("skill.html#") and len(skill) > len("skill.html#")
    page.goto(f"{desktop.base_url}/tree.html#g=unit:{UNIT}", wait_until="networkidle")
    sector = page.get_by_role("link", name="Open this sector")
    assert sector.get_attribute("href") == f"groups.html#g=unit:{UNIT}"
    assert desktop.unexpected_errors() == []


def test_the_skills_table_sorts_and_filters(desktop, developer_skills):
    page = desktop.open(f"tree.html#job={SLUG}")
    page.wait_for_selector(".skill-table")
    rows = page.locator(".skill-table tbody tr")
    assert rows.count() == len(developer_skills["rows"])
    page.get_by_role("button", name="Essential", exact=True).click()
    assert rows.count() == len(developer_skills["essential"])
    assert page.locator(".skill-table tbody td").nth(1).inner_text() == "Essential"

    page.get_by_role("button", name="All", exact=True).click()
    header = page.locator('.skill-table th[data-key="auto"]')
    assert header.locator("button").inner_text() == "AI substitution"
    header.locator("button").click()
    assert header.get_attribute("aria-sort") == "descending"
    top = max(skill["a"] for skill in developer_skills["rows"])
    assert page.locator(".skill-table tbody td.numeric").first.inner_text() == f"{top:.1f}"
    assert desktop.unexpected_errors() == []


def test_a_skill_row_names_the_class_the_model_gave_it(desktop, developer_skills):
    """A cell may add a "near the line" note after the name; it may never show
    a different class from the one the published shard gave the skill."""
    page = desktop.open(f"tree.html#job={SLUG}")
    page.wait_for_selector(".skill-table")
    shown = page.locator(".skill-table tbody td.class-cell").evaluate_all(
        "nodes => nodes.map((node) =>"
        " ({code: node.dataset.class, text: node.textContent}))")
    assert sorted(cell["code"] for cell in shown) == sorted(
        skill.get("c") or "" for skill in developer_skills["rows"])
    for cell in shown:
        assert cell["text"].startswith(SKILL_CLASS_NAMES[cell["code"]]), cell


def test_the_tree_works_without_a_job_shard(desktop, groups_v2, by_slug_v2):
    """Losing one job's skills must not take the tree, the filter or the scores
    with it: the shard is the only thing that was missing."""
    desktop.page.route(f"**/jobs/{UNIT}_v2.json", lambda route: route.abort())
    page = desktop.open("tree.html")
    assert page.locator('[role="treeitem"]').count() == 10
    drill(page, CHAIN)
    assert (f"{groups_v2[f'unit:{UNIT}']['n']} jobs"
            in page.locator("#detail-body p").first.inner_text())
    page.fill("#tree-q", "programmer")
    page.wait_for_function("() => document.querySelectorAll('.result-link').length > 0")
    page.fill("#tree-q", "")
    item(page, f"job:{SLUG}").click()
    page.wait_for_selector("#skills-block .error-block")
    assert (f"{by_slug_v2[SLUG]['a']:.1f}"
            in page.locator(".score-card--auto .score-value").inner_text())
    assert page.get_by_role("button", name="Retry").is_visible()
    assert page.get_by_role("link", name="Open the full job page").is_visible()


def test_no_sideways_scroll_on_a_phone(phone):
    page = phone.open("tree.html")
    assert not phone.overflows()
    page.goto(f"{phone.base_url}/tree.html#job={SLUG}", wait_until="networkidle")
    page.wait_for_selector(".skill-table")
    assert not phone.overflows()
    back = page.get_by_role("button", name="Back to the tree")
    assert back.is_visible()
    back.click()
    page.wait_for_function(
        f"() => document.activeElement.dataset.id === 'job:{SLUG}'"
    )
    assert phone.unexpected_errors() == []
