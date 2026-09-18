"""The published search indexes must carry the words people actually type.

A regression test against the real files, not a fixture: the failure it guards
was a budget that silently dropped twelve of thirteen ESCO synonyms, so a test
over synthetic data would not have caught it. "software engineer" returning
nothing is the case the audit found; "lorry driver" is the same failure in a
different register.
"""

import json
import os

import pytest

from aiisco.site_indexes import MAX_ALT_LABELS

SITE = os.path.join(os.path.dirname(__file__), "..", "site")

#: The query people type, and the slug it has to be able to reach.
SYNONYMS = [("software engineer", "software-developer"),
            ("lorry driver", "cargo-vehicle-driver")]

PUBLISHED = ["search_index.json", "search_index_v2.json"]


def index(name):
    """One published search index, or a skip when that scorer is not built."""
    path = os.path.join(SITE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not published in this tree")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def row_for(rows, slug):
    """The row of one occupation."""
    return next((row for row in rows if row["s"] == slug), None)


@pytest.mark.parametrize("name", PUBLISHED)
@pytest.mark.parametrize(("query", "slug"), SYNONYMS)
def test_a_published_index_lets_a_common_synonym_reach_its_job(name, query, slug):
    row = row_for(index(name), slug)
    assert row is not None, f"{slug} is missing from {name}"
    assert query in row["alt"], f"{name}: {slug} does not answer to {query!r}"


@pytest.mark.parametrize("name", PUBLISHED)
def test_a_published_index_carries_more_than_one_synonym_per_job(name):
    rows = index(name)
    labels = sum(len(row["alt"]) for row in rows)
    assert labels / len(rows) > 2.0, (
        f"{name} averages {labels / len(rows):.2f} alternative labels per job; "
        "the budget is dropping the words people search with")


@pytest.mark.parametrize("name", PUBLISHED)
def test_no_job_spends_more_than_its_share_of_the_index(name):
    assert all(len(row["alt"]) <= MAX_ALT_LABELS for row in index(name))


def test_both_published_indexes_carry_the_same_labels():
    """The schemes differ in what they score, never in what they can be found by."""
    labels = [{row["s"]: row["alt"] for row in index(name)} for name in PUBLISHED]
    assert labels[0] == labels[1]
