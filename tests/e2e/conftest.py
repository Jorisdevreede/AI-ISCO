"""End-to-end tests drive the real static site in a real browser.

A throwaway HTTP server serves a COPY of site/, so the tests see exactly what
GitHub Pages would publish, every score set included. A second copy without the
files in EXTRA_SETS stands in for a checkout that has only the Gemini set
(the `single_set_*` fixtures). The browser is Chromium by default (CI installs
it); set AIISCO_E2E_CHANNEL=chrome to use an installed Chrome instead.

The published default is the v2 set — TypeSafe's jev scores, four shares per
job and seven types — so every module here reads the `*_v2` files unless it
says otherwise. The older Gemini set is what `?scorer=gemini` selects, and
test_quadrants.py is the compact proof that it still renders.

Run:  AIISCO_E2E_CHANNEL=chrome uv run pytest tests/e2e -q
"""

import functools
import http.server
import json
import os
import shutil
import socket
import threading
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[2] / "site"
DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}

#: job.html downloads and parses a 13 MB file; every wait on it gets this long.
JOB_TIMEOUT = 15_000

#: What Chromium logs for any resource that came back 4xx. The line names no
#: file, so :meth:`Session.unexpected_errors` counts them against the responses.
RESOURCE_404 = "Failed to load resource: the server responded with a status of 404"

#: Everything the two jev score sets add to site/. A fork that never ran them
#: has none of it, and site/scorer.js then behaves as a single-scorer site.
EXTRA_SETS = ("*_v2.json", "*_typesafe.json", "skill_answers_v2")

#: The files site/scorer.js HEAD-probes to learn which sets are deployed.
PROBED = ("data_v2.json", "data_typesafe.json")

#: The query that puts the older Gemini set back on screen. Choosing the
#: default removes the parameter instead, so there is no `?scorer=v2`.
GEMINI = "?scorer=gemini"

#: What the v2 files are called. Everything the default reads carries it.
V2 = "_v2"

#: The seven type names site/js/scheme.js writes, full and short. These are
#: copy, not data: a test may name them, never count them.
TYPE_NAMES = {
    "AUTOMATION_HEAVY": "Automation-heavy",
    "TRANSFORMING": "Transforming",
    "AUGMENTED": "Augmented",
    "MECHANISABLE": "Mechanisable",
    "INSULATED_PHYSICAL": "Insulated by physical work",
    "INSULATED_PEOPLE": "Insulated by work with people",
    "MIXED": "Mixed",
}
TYPE_SHORT = {**TYPE_NAMES,
              "INSULATED_PHYSICAL": "Physical work",
              "INSULATED_PEOPLE": "People work"}

#: What the site calls the four skill classes, in the order `sh` stores them.
SKILL_CLASS_ORDER = ("S", "A", "M", "I")
SKILL_CLASS_NAMES = {"S": "AI can take over", "A": "AI assists",
                     "M": "Machines can do", "I": "Stays human"}

#: The four quadrant names, for the tests that follow ?scorer=gemini.
QUADRANT_NAMES = {"TRANSFORM": "Transform", "SHRINK": "Shrink",
                  "EVOLVE": "Evolve", "STABLE": "Stable"}


def _largest_remainder(exact):
    """Whole percentages from exact ones, the remainder to the largest fractions."""
    parts = [{"index": index, "floor": int(value), "fraction": value - int(value)}
             for index, value in enumerate(exact)]
    left = 100 - sum(part["floor"] for part in parts)
    for part in sorted(parts, key=lambda p: (-p["fraction"], p["index"]))[:left]:
        part["floor"] += 1
    return [part["floor"] for part in parts]


def share_percents(shares):
    """The four shares as whole percentages that add up to 100.

    site/js/shares.js gives the rounding remainder to the largest fractions, so
    a test that rounds each share on its own disagrees with the page about one
    job in three. This is that rule, and nothing else in the suite may round.
    """
    return _largest_remainder([value * 100 for value in shares])


def whole_percents(values):
    """A set of counts as whole percentages that add up to 100.

    site/js/format.js's `largestRemainder`, which every printed and spoken
    split on the site goes through.
    """
    total = sum(values)
    if not total:
        return [0] * len(values)
    return _largest_remainder([value / total * 100 for value in values])


def percent_text(percent, count):
    """What the site prints for one part: "0%" is a lie about a part with one
    thing in it."""
    return "<1%" if count > 0 and percent == 0 else f"{percent}%"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def copy_site(target, *ignored):
    """Copy site/ to `target`, leaving out files that match `ignored`."""
    shutil.copytree(SITE, target, ignore=shutil.ignore_patterns(*ignored))
    return target


def serve(directory):
    """Serve `directory` on a free local port; yields the base URL."""
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="session")
def published_site(tmp_path_factory):
    """A copy of site/ as it is published, with every score set."""
    return copy_site(tmp_path_factory.mktemp("published") / "site")


@pytest.fixture(scope="session")
def base_url(published_site):
    yield from serve(published_site)


@pytest.fixture(scope="session")
def single_set_url(tmp_path_factory):
    """A site with only the Gemini score set, as a fork without the jev runs has."""
    site = copy_site(tmp_path_factory.mktemp("single") / "site", *EXTRA_SETS)
    yield from serve(site)


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        channel = os.environ.get("AIISCO_E2E_CHANNEL")
        try:
            instance = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception as error:  # noqa: BLE001 - any launch failure means "no browser here"
            pytest.skip(f"no browser available for end-to-end tests: {error}")
        yield instance
        instance.close()


class Session:
    """A page plus everything it logged, so a test can assert on errors."""

    def __init__(self, context, base_url):
        self.base_url = base_url
        self.page = context.new_page()
        self.errors = []
        self.requests = []
        self.fetched = []
        self.refused = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("console", self._on_console)
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)

    def _on_console(self, message):
        if message.type == "error":
            self.errors.append(f"console: {message.text}")

    def _on_request(self, request):
        """Every request by file name, and separately the ones that download.

        site/scorer.js HEAD-probes the files in PROBED to learn which sets are
        deployed, so `data_v2.json` is asked for on every page without a byte
        of it arriving. `fetched` leaves those out, which is what a test about
        what a page costs has to measure.
        """
        name = request.url.rsplit("/", 1)[-1].split("?")[0]
        self.requests.append(name)
        if request.method != "HEAD":
            self.fetched.append(name)

    def _on_response(self, response):
        if response.status >= 400:
            self.refused.append(response.url.rsplit("/", 1)[-1].split("?")[0])

    def open(self, path):
        self.page.goto(f"{self.base_url}/{path}", wait_until="networkidle")
        return self.page

    def overflows(self):
        return self.page.evaluate("document.documentElement.scrollWidth > window.innerWidth")

    def unexpected_errors(self):
        """Errors the published site would also log.

        site/scorer.js HEAD-probes the files in PROBED to decide what its score
        switch offers. On the single-set copy they are absent, so the browser
        logs one anonymous 404 line per probe. Drop exactly that many, never
        more, so a real 404 still fails the test. On the published copy the
        probes succeed and nothing is dropped.
        """
        spare = sum(1 for name in self.refused if name in PROBED)
        kept = []
        for error in self.errors:
            if spare and RESOURCE_404 in error:
                spare -= 1
            else:
                kept.append(error)
        return kept


# Every session fixture below is per-test and builds its own browser context.
# That is load-bearing, not tidiness: site/scorer.js remembers the chosen set in
# localStorage, so a context that once saw `?scorer=gemini` would stay on Gemini
# for every bare URL afterwards. A fresh context starts on the default, and a
# test that wants the other set says so on every navigation.


def make_session(browser, base_url, viewport):
    context = browser.new_context(viewport=viewport)
    return context, Session(context, base_url)


@pytest.fixture
def desktop(browser, base_url):
    context, session = make_session(browser, base_url, DESKTOP)
    yield session
    context.close()


@pytest.fixture
def phone(browser, base_url):
    context, session = make_session(browser, base_url, PHONE)
    yield session
    context.close()


@pytest.fixture
def single_set(browser, single_set_url):
    """A desktop session on the site that has only the Gemini score set."""
    context, session = make_session(browser, single_set_url, DESKTOP)
    yield session
    context.close()


# --- opening a page and knowing it finished --------------------------------
#
# `wait_until="networkidle"` says the bytes arrived, not that the page module
# painted with them. Every helper below also waits for one element that only
# exists once the page rendered, so no test needs a sleep.


def open_and_wait(session, path, ready):
    """Open `path` in `session` and wait for the element that proves it rendered."""
    page = session.open(path)
    page.locator(ready).first.wait_for(state="visible", timeout=JOB_TIMEOUT)
    return page


def reveal_all(page):
    """Scroll to the foot of the page and open every closed `<details>` in it.

    Two things on this site are deliberately not on the first screen: a section
    that waits to be scrolled to before it fetches anything, and detail a
    visitor has to ask for. A test about what a page *says*, rather than about
    what it says first, asks for both and then waits for what it came for.
    """
    page.evaluate(
        "() => { window.scrollTo(0, document.body.scrollHeight);"
        " document.querySelectorAll('main details:not([open])')"
        ".forEach((node) => { node.open = true; }); }")
    return page


def reveal_section(page, selector):
    """Wait for a section that appears late, then scroll it into view.

    A section that only fetches once it is about to be read has to be scrolled
    to before it will. `scroll_into_view_if_needed` wants a box, and a section
    that is still a heading and a sentinel may not have one, so this scrolls
    the element itself.
    """
    page.locator(selector).first.wait_for(state="visible", timeout=JOB_TIMEOUT)
    page.eval_on_selector(selector, "node => node.scrollIntoView({block: 'end'})")
    return page


@pytest.fixture(scope="session")
def open_ready():
    """`open_ready(session, path, ready_selector)` -> the rendered page."""
    return open_and_wait


@pytest.fixture(scope="session")
def open_job():
    """`open_job(session, path)` -> job.html with its big data file in."""
    return functools.partial(open_and_wait, ready="#job-article")


@pytest.fixture(scope="session")
def open_group():
    """`open_group(session, key, view)` -> groups.html showing one view."""

    def open_it(session, key, view):
        return open_and_wait(session, f"groups.html#g={key}&view={view}",
                             "#tablist [role='tab']")

    return open_it


# --- the published data the pages are built from ---------------------------


def read_site_json(name):
    """One published JSON file, read straight from site/."""
    return json.loads((SITE / f"{name}.json").read_text(encoding="utf-8"))


# The Gemini set: what `?scorer=gemini` puts on screen, and all a single-set
# checkout has. Only the quadrant tests and the agreement figures read these.


@pytest.fixture(scope="session")
def search_index():
    return read_site_json("search_index")


@pytest.fixture(scope="session")
def by_slug(search_index):
    """search_index.json rows keyed by occupation slug."""
    return {row["s"]: row for row in search_index}


@pytest.fixture(scope="session")
def groups_json():
    return read_site_json("groups")


@pytest.fixture(scope="session")
def skill_index():
    return read_site_json("skill_index")


@pytest.fixture(scope="session")
def stats_json():
    return read_site_json("stats")


# The v2 set: the default. Everything a visitor sees without a `?scorer=` is
# computed from these, never typed in.


@pytest.fixture(scope="session")
def stats_v2():
    return read_site_json(f"stats{V2}")


@pytest.fixture(scope="session")
def search_index_v2():
    return read_site_json(f"search_index{V2}")


@pytest.fixture(scope="session")
def by_slug_v2(search_index_v2):
    """search_index_v2.json rows keyed by occupation slug."""
    return {row["s"]: row for row in search_index_v2}


@pytest.fixture(scope="session")
def groups_v2():
    return read_site_json(f"groups{V2}")


@pytest.fixture(scope="session")
def skill_index_v2():
    return read_site_json(f"skill_index{V2}")


@pytest.fixture(scope="session")
def rubric_v2():
    return read_site_json(f"rubric{V2}")


@pytest.fixture(scope="session")
def units_json():
    """jobs/units.json: every slug's unit group, the one file both sets share."""
    return read_site_json("jobs/units")


@pytest.fixture(scope="session")
def unit_data():
    """`unit_data("2512")` -> the v2 shard job.html and tree.html read."""
    return functools.lru_cache(maxsize=None)(
        lambda unit: read_site_json(f"jobs/{unit}{V2}"))


@pytest.fixture(scope="session")
def skill_note():
    """`skill_note(skill_id)` -> the written rationale skill.html shows, or None.

    A jev score set writes no text, so the build lends each skill the rationale
    Gemini wrote and marks it `rf`; that mark is what the "i" beside it is about.
    """
    shard = functools.lru_cache(maxsize=None)(
        lambda name: read_site_json(f"skill_notes/{name}{V2}"))
    return lambda skill_id: shard(skill_id[:2].lower()).get(skill_id)


@pytest.fixture(scope="session")
def skill_answers():
    """`skill_answers(skill_id)` -> what the model answered, or None."""
    shard = functools.lru_cache(maxsize=None)(
        lambda name: read_site_json(f"skill_answers{V2}/{name}"))
    return lambda skill_id: shard(skill_id[:2].lower()).get(skill_id)
