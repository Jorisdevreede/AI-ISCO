"""End-to-end checks for site/tree.html — the ISCO-08 tree and its detail pane.

Run:  AIISCO_E2E_CHANNEL=chrome uv run pytest tests/e2e/test_tree.py -q
"""

import json
from itertools import pairwise
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[2] / "site"
CHAIN = ["major:2", "sub:25", "minor:251", "unit:2512"]
QUADRANTS = ["TRANSFORM", "STABLE", "EVOLVE", "SHRINK"]

@pytest.fixture(scope="module")
def groups():
    return json.loads((SITE / "groups.json").read_text())


@pytest.fixture(scope="module")
def search_index():
    return json.loads((SITE / "search_index.json").read_text())


@pytest.fixture(scope="module")
def developer_skill_ids():
    """The skill ids software developer really has in portfolio_data.json."""
    data = json.loads((SITE / "portfolio_data.json").read_text())
    skills = data["skills"]
    job = next(row for row in data["occupations"] if row["s"] == "software-developer")
    wanted = list(job.get("se", [])) + list(job.get("so", []))
    return [skill_id for skill_id in wanted if skill_id in skills]


def item(page, node_id):
    return page.locator(f'[data-id="{node_id}"]')


def drill(page, keys):
    """Click each group in turn, waiting for it to open."""
    for key in keys:
        item(page, key).click()
        item(page, key).and_(page.locator('[aria-expanded="true"]')).wait_for()


def jobs_in_unit(search_index, code):
    titles = [row["t"] for row in search_index if row["c"] == code]
    return sorted(titles, key=str.casefold)


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
    assert desktop.unexpected_errors() == []


def test_a_mouse_drill_reaches_a_job_and_its_skills(desktop, developer_skill_ids):
    page = desktop.open("tree.html")
    drill(page, CHAIN)
    item(page, "job:software-developer").click()
    page.wait_for_selector(".skill-table")
    assert page.locator(".skill-table tbody tr").count() == len(developer_skill_ids)
    assert page.locator("#detail-title").inner_text() == "software developer"
    assert desktop.unexpected_errors() == []


def test_a_keyboard_drill_reaches_a_job(desktop, groups, search_index):
    page = desktop.open("tree.html")
    item(page, "major:2").first.wait_for()
    item(page, "major:2").focus()
    for parent, child in pairwise(CHAIN):
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        for _ in range(groups[parent]["children"].index(child)):
            page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    for _ in range(jobs_in_unit(search_index, "2512").index("software developer")):
        page.keyboard.press("ArrowDown")
    assert page.evaluate("document.activeElement.dataset.id") == "job:software-developer"
    page.keyboard.press("Enter")
    page.wait_for_function("() => location.hash === '#job=software-developer'")
    assert page.locator("#detail-title").inner_text() == "software developer"
    assert desktop.unexpected_errors() == []


def test_a_deep_link_opens_with_the_job_selected(desktop):
    page = desktop.open("tree.html#job=software-developer")
    leaf = item(page, "job:software-developer")
    leaf.wait_for()
    assert leaf.get_attribute("aria-selected") == "true"
    assert leaf.is_visible()
    assert leaf.get_attribute("aria-level") == "5"
    for key in CHAIN:
        assert item(page, key).get_attribute("aria-expanded") == "true"
    assert page.locator("#detail-title").inner_text() == "software developer"
    assert desktop.unexpected_errors() == []


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


def test_the_filter_narrows_the_tree_and_announces_a_count(desktop):
    page = desktop.open("tree.html")
    item(page, "major:2").first.wait_for()
    page.fill("#tree-q", "programmer")
    page.wait_for_function(
        "() => document.getElementById('tree-count').textContent.includes('match')"
    )
    announced = page.locator("#tree-count").inner_text()
    assert "jobs match" in announced
    assert item(page, "job:software-developer").is_visible()
    assert page.locator(".tree-item--job").count() == int(announced.split()[0])
    assert page.evaluate("location.hash") == "#q=programmer"
    page.fill("#tree-q", "")
    page.wait_for_function("() => document.getElementById('tree-count').textContent === ''")
    assert page.locator('[role="treeitem"]').count() == 10
    assert desktop.unexpected_errors() == []


def test_a_group_shows_the_counts_from_groups_json(desktop, groups):
    page = desktop.open("tree.html#g=unit:2512")
    page.wait_for_selector("#detail-body .mix-bars")
    group = groups["unit:2512"]
    total = sum(group["q"].values())
    assert f"{group['n']} jobs" in page.locator("#detail-body p").first.inner_text()
    figures = page.locator("#detail-body .mix-figure").all_inner_texts()
    assert len(figures) == len(QUADRANTS)
    for text, code in zip(figures, QUADRANTS):
        assert text.startswith(f"{group['q'].get(code, 0)} of {total} jobs")
    assert desktop.unexpected_errors() == []


def test_the_small_files_are_fetched_before_the_big_one(desktop):
    page = desktop.open("tree.html")
    page.locator('[role="treeitem"]').first.wait_for()
    asked = desktop.requests
    for small in ["groups.json", "search_index.json", "stats.json"]:
        assert asked.index(small) < asked.index("portfolio_data.json")
    assert desktop.unexpected_errors() == []


def test_the_tree_works_without_the_big_file(desktop, groups):
    desktop.page.route("**/portfolio_data.json", lambda route: route.abort())
    page = desktop.open("tree.html")
    assert page.locator('[role="treeitem"]').count() == 10
    drill(page, CHAIN)
    assert f"{groups['unit:2512']['n']} jobs" in page.locator("#detail-body p").first.inner_text()
    page.fill("#tree-q", "programmer")
    page.wait_for_function("() => document.querySelectorAll('.tree-item--job').length > 0")
    page.fill("#tree-q", "")
    item(page, "job:software-developer").click()
    page.wait_for_selector("#skills-block .error-block")
    assert "6.4" in page.locator(".score-card--auto .score-value").inner_text()
    assert page.get_by_role("button", name="Retry").is_visible()
    assert page.get_by_role("link", name="Open the full job page").is_visible()


def test_every_link_out_of_the_pane_is_real(desktop):
    page = desktop.open("tree.html#job=software-developer")
    page.wait_for_selector(".skill-table")
    full = page.get_by_role("link", name="Open the full job page")
    assert full.get_attribute("href") == "job.html#software-developer&from=unit:2512"
    assert page.locator("#detail-body a").first.get_attribute("href") == "tree.html#g=unit:2512"
    skill = page.locator(".skill-table tbody a").first.get_attribute("href")
    assert skill.startswith("skill.html#") and len(skill) > len("skill.html#")
    page.goto(f"{desktop.base_url}/tree.html#g=unit:2512", wait_until="networkidle")
    sector = page.get_by_role("link", name="Open this sector")
    assert sector.get_attribute("href") == "groups.html#g=unit:2512"
    assert desktop.unexpected_errors() == []


def test_the_skills_table_sorts_and_filters(desktop):
    page = desktop.open("tree.html#job=software-developer")
    page.wait_for_selector(".skill-table")
    every = page.locator(".skill-table tbody tr").count()
    page.get_by_role("button", name="Essential", exact=True).click()
    essential = page.locator(".skill-table tbody tr").count()
    assert 0 < essential < every
    assert page.locator(".skill-table tbody td").nth(1).inner_text() == "Essential"
    page.get_by_role("button", name="All", exact=True).click()
    header = page.locator('.skill-table th[data-key="auto"]')
    header.locator("button").click()
    assert header.get_attribute("aria-sort") == "descending"
    first = page.locator(".skill-table tbody td.numeric").first.inner_text()
    assert float(first) >= 8.0
    assert desktop.unexpected_errors() == []


def test_no_sideways_scroll_on_a_phone(phone):
    page = phone.open("tree.html")
    assert not phone.overflows()
    page.goto(f"{phone.base_url}/tree.html#job=software-developer", wait_until="networkidle")
    page.wait_for_selector(".skill-table")
    assert not phone.overflows()
    back = page.get_by_role("button", name="Back to the tree")
    assert back.is_visible()
    back.click()
    page.wait_for_function(
        "() => document.activeElement.dataset.id === 'job:software-developer'"
    )
    assert phone.unexpected_errors() == []
