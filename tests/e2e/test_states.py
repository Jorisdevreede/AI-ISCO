"""A slug that does not exist, and a data file that never arrives.

Neither may leave a blank page or an uncaught error: the brief asks every
failure to name itself and offer a way on. The job page reads one unit group's
own shard now, so the file that can go missing is that shard — and losing it
must not cost the visitor the rest of the page.
"""

from playwright.sync_api import expect

from .conftest import reveal_all

UNKNOWN = "job.html#no-such-job"
KNOWN = "job.html#software-developer"

#: The shard software developer's skills live in, and the only job file the
#: page may ask for.
SHARD = "**/jobs/2512_v2.json"


def test_an_unknown_slug_explains_itself_and_offers_a_way_on(desktop):
    page = desktop.open(UNKNOWN)
    expect(page.locator("#job-title")).to_contain_text("job called", timeout=15_000)
    expect(page.locator("#job-intro")).to_contain_text("Search with the box")
    expect(page.locator("#job-find")).to_be_visible()
    expect(page.get_by_role("link", name="Browse sectors →")).to_be_visible()
    assert desktop.unexpected_errors() == []


def test_an_unknown_slug_costs_nothing_but_the_slug_map(desktop):
    """jobs/units.json already knows every slug there is, so a bad link never
    downloads a job file to find out it is bad."""
    page = desktop.open(UNKNOWN)
    expect(page.locator("#job-title")).to_contain_text("job called", timeout=15_000)
    shards = [name for name in desktop.fetched
              if name.endswith("_v2.json") and name[0].isdigit()]
    assert not shards, f"an unknown slug downloaded {shards}"


def test_a_data_file_that_never_arrives_offers_a_retry_and_a_link(desktop):
    desktop.page.route(SHARD, lambda route: route.abort())
    page = desktop.open(KNOWN)
    block = page.locator("#job-error .error-block")
    expect(block).to_be_visible(timeout=15_000)
    expect(block).to_contain_text("jobs/2512")
    expect(block.get_by_role("button", name="Try again")).to_be_visible()
    expect(block.get_by_role("link", name="Browse sectors instead →")).to_be_visible()
    expect(page.locator("#job-article")).to_be_hidden()


def test_retrying_after_the_shard_comes_back_renders_the_job(desktop, by_slug_v2):
    """A retry has to really retry: the failed load is evicted from the cache."""
    failures = {"left": 1}

    def maybe_abort(route):
        if failures["left"]:
            failures["left"] -= 1
            route.abort()
        else:
            route.continue_()

    desktop.page.route(SHARD, maybe_abort)
    page = desktop.open(KNOWN)
    page.locator("#job-error .error-block").wait_for(state="visible", timeout=15_000)
    page.get_by_role("button", name="Try again").click()
    page.locator("#job-article").wait_for(state="visible", timeout=15_000)
    reveal_all(page)
    expect(page.locator(".score-card--sub .score-value")).to_have_text(
        f"{by_slug_v2['software-developer']['a']:.1f}")
