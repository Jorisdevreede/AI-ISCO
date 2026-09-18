"""Every page the site publishes has a head a link preview can read.

A page shared into a chat window is often the first thing someone sees of the
site; without these tags it arrives as a bare URL. Redirect stubs are exempt:
they exist to be left immediately and have nothing to preview.
"""

import os
import re

import pytest

SITE = os.path.join(os.path.dirname(__file__), "..", "site")

BASE_URL = "https://jorisdevreede.github.io/AI-ISCO/"

#: Every Open Graph tag a page must carry, and how to find it.
OPEN_GRAPH = ("og:site_name", "og:type", "og:title", "og:description", "og:url",
              "og:image")

REDIRECT = re.compile(r"location\.replace|http-equiv=[\"']refresh[\"']",
                      re.IGNORECASE)
TITLE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
DESCRIPTION = re.compile(
    r"""<meta\s+name=["']description["']\s+content=["'](.*?)["']""", re.DOTALL)


def pages():
    """Every published page that is not a redirect stub."""
    found = []
    for name in sorted(os.listdir(SITE)):
        if not name.endswith(".html"):
            continue
        text = read(name)
        if not REDIRECT.search(text):
            found.append(name)
    return found


def read(name):
    """The source of one page."""
    with open(os.path.join(SITE, name), encoding="utf-8") as handle:
        return handle.read()


def meta_property(text, name):
    """The content of one ``<meta property=...>`` tag, or None."""
    pattern = re.compile(
        rf"""<meta\s+property=["']{re.escape(name)}["']\s+content=["'](.*?)["']""",
        re.DOTALL)
    found = pattern.search(text)
    return found.group(1) if found else None


def expected_url(name):
    """The canonical URL of one page; the landing page is the bare site."""
    return BASE_URL if name == "index.html" else BASE_URL + name


def test_the_site_publishes_pages_to_check():
    assert pages(), "no non-redirect pages found under site/"


@pytest.mark.parametrize("name", pages())
def test_a_page_carries_every_open_graph_tag(name):
    text = read(name)
    missing = [tag for tag in OPEN_GRAPH if meta_property(text, tag) is None]
    assert not missing, f"{name} is missing {', '.join(missing)}"


@pytest.mark.parametrize("name", pages())
def test_a_page_names_the_site_and_asks_for_a_large_card(name):
    text = read(name)
    assert meta_property(text, "og:site_name") == "AI-ISCO"
    assert meta_property(text, "og:type") == "website"
    assert 'name="twitter:card" content="summary_large_image"' in text


@pytest.mark.parametrize("name", pages())
def test_a_page_points_at_itself_and_at_the_card_image(name):
    text = read(name)
    assert meta_property(text, "og:url") == expected_url(name)
    assert meta_property(text, "og:image") == BASE_URL + "card.png"


@pytest.mark.parametrize("name", pages())
def test_a_page_has_a_description_that_the_preview_repeats(name):
    text = read(name)
    described = DESCRIPTION.search(text)
    assert described, f"{name} has no meta description"
    assert meta_property(text, "og:description") == described.group(1)


@pytest.mark.parametrize("name", pages())
def test_a_page_has_one_title_that_says_what_it_is(name):
    text = read(name)
    titles = TITLE.findall(text)
    assert len(titles) == 1, f"{name} has {len(titles)} <title> elements"
    assert titles[0].strip() != "Job details", (
        f"{name} still carries the placeholder title")
    assert meta_property(text, "og:title") == titles[0].strip()
