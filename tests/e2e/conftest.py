"""End-to-end tests drive the real static site in a real browser.

A throwaway HTTP server serves a COPY of site/, so the tests see exactly what
GitHub Pages would publish, both score sets included. A second copy without the
*_typesafe.json files stands in for a checkout that has only the default set
(the `single_set_*` fixtures). The browser is Chromium by default (CI installs
it); set AIISCO_E2E_CHANNEL=chrome to use an installed Chrome instead.

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
    """A copy of site/ as it is published: the default score set and the second one."""
    return copy_site(tmp_path_factory.mktemp("published") / "site")


@pytest.fixture(scope="session")
def base_url(published_site):
    yield from serve(published_site)


@pytest.fixture(scope="session")
def single_set_url(tmp_path_factory):
    """A site with only the default score set, as a fork without the second run has."""
    site = copy_site(tmp_path_factory.mktemp("single") / "site", "*_typesafe.json")
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
        self.refused = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("console", self._on_console)
        self.page.on("request", lambda r: self.requests.append(r.url.rsplit("/", 1)[-1].split("?")[0]))
        self.page.on("response", self._on_response)

    def _on_console(self, message):
        if message.type == "error":
            self.errors.append(f"console: {message.text}")

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

        site/scorer.js HEAD-probes ``data_typesafe.json`` to decide whether to
        offer its score switch. On the single-set copy that file is absent, so
        the browser logs one anonymous 404 line per probe. Drop exactly that
        many, never more, so a real 404 still fails the test. On the published
        copy the probe succeeds and nothing is dropped.
        """
        spare = sum(1 for name in self.refused if name.endswith("_typesafe.json"))
        kept = []
        for error in self.errors:
            if spare and RESOURCE_404 in error:
                spare -= 1
            else:
                kept.append(error)
        return kept


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
    """A desktop session on the site that has only the default score set."""
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
