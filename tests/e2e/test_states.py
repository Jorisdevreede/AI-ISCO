"""A slug that does not exist, and a data file that never arrives.

Neither may leave a blank page or an uncaught error: the brief asks every
failure to name itself and offer a way on.
"""

from playwright.sync_api import expect

UNKNOWN = "job.html#no-such-job"
KNOWN = "job.html#software-developer"


def test_an_unknown_slug_explains_itself_and_offers_a_way_on(desktop):
    page = desktop.open(UNKNOWN)
    expect(page.locator("#job-title")).to_contain_text("job called", timeout=15_000)
    expect(page.locator("#job-intro")).to_contain_text("Search with the box")
    expect(page.locator("#job-find")).to_be_visible()
    expect(page.get_by_role("link", name="Browse sectors →")).to_be_visible()
    assert desktop.unexpected_errors() == []


def test_a_data_file_that_never_arrives_offers_a_retry_and_a_link(desktop):
    desktop.page.route("**/portfolio_data.json", lambda route: route.abort())
    page = desktop.open(KNOWN)
    block = page.locator("#job-error .error-block")
    expect(block).to_be_visible(timeout=15_000)
    expect(block).to_contain_text("portfolio_data.json")
    expect(block.get_by_role("button", name="Try again")).to_be_visible()
    expect(block.get_by_role("link", name="Browse sectors instead →")).to_be_visible()
    expect(page.locator("#job-article")).to_be_hidden()
