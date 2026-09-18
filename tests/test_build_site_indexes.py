"""Tests for the build_site_indexes CLI against the synthetic fixtures."""

import json
import os
import shutil

import pytest

import build_site_indexes as cli

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "site_indexes")


@pytest.fixture
def workspace(tmp_path):
    """A site/ and data/esco/ pair holding the fixtures."""
    site = tmp_path / "site"
    esco = tmp_path / "data" / "esco"
    site.mkdir(parents=True)
    esco.mkdir(parents=True)
    shutil.copy(os.path.join(FIXTURES, "portfolio_data.json"), site / "portfolio_data.json")
    for name in ("occupations_en.csv", "ISCOGroups_en.csv"):
        shutil.copy(os.path.join(FIXTURES, name), esco / name)
    return tmp_path


def run(workspace, *extra):
    return cli.main([
        "--site-dir", str(workspace / "site"),
        "--esco-dir", str(workspace / "data" / "esco"),
        *extra,
    ])


def read(workspace, name):
    with open(workspace / "site" / name, encoding="utf-8") as fh:
        return json.load(fh)


# --- readers --------------------------------------------------------------

def test_read_alt_labels_keeps_every_row_for_a_repeated_title():
    rows = cli.read_alt_labels(os.path.join(FIXTURES, "occupations_en.csv"))
    assert sorted(code for code, _ in rows["bookkeeper"]) == ["3313", "4311"]
    assert "programmer" in rows["software developer"][0][1]


def test_read_isco_labels_covers_all_four_levels():
    labels = cli.read_isco_labels(os.path.join(FIXTURES, "ISCOGroups_en.csv"))
    assert labels["2"] == "Professionals"
    assert labels["2512"] == "Software developers"
    assert "7223" not in labels


def test_input_paths_applies_the_scorer_suffix():
    paths = cli.input_paths("site", "esco", "_typesafe")
    assert paths[0].endswith(os.path.join("site", "portfolio_data_typesafe.json"))
    assert paths[1].endswith("occupations_en.csv")
    assert paths[2].endswith("ISCOGroups_en.csv")


def test_build_date_prefers_the_given_date():
    assert cli.build_date("2020-01-01", []) == "2020-01-01"


def test_build_date_falls_back_to_the_newest_input(workspace):
    paths = cli.input_paths(str(workspace / "site"), str(workspace / "data" / "esco"), "")
    os.utime(paths[0], (1_600_000_000, 1_600_000_000))
    assert cli.build_date(None, paths) == cli.build_date(None, paths)
    assert len(cli.build_date(None, paths)) == len("2020-09-13")


# --- the build ------------------------------------------------------------

def test_build_writes_five_files_and_succeeds(workspace, capsys):
    assert run(workspace, "--date", "2026-09-18") == 0
    names = {"search_index", "groups", "stats", "skill_index", "skill_occupations"}
    for name in names:
        assert (workspace / "site" / f"{name}.json").exists()
    printed = capsys.readouterr().out
    assert all(name in printed for name in names)
    assert "OVER BUDGET" not in printed


def test_search_index_carries_alt_labels_and_scores(workspace):
    run(workspace, "--date", "2026-09-18")
    rows = read(workspace, "search_index.json")
    assert [r["s"] for r in rows] == sorted(r["s"] for r in rows)
    developer = next(r for r in rows if r["s"] == "software-developer")
    assert developer["alt"] == ["coder", "programmer", "software engineer"]
    assert (developer["a"], developer["m"], developer["q"]) == (6.9, 7.0, "TRANSFORM")
    assert developer["mg"] == "Professionals"
    assert next(r for r in rows if r["s"] == "lathe-operator")["alt"] == []


def test_groups_use_the_isco_code_not_the_stale_data_file(workspace):
    run(workspace, "--date", "2026-09-18")
    groups = read(workspace, "groups.json")
    assert groups["unit:2512"]["label"] == "Software developers"
    assert groups["unit:7223"]["label"] == "7223"       # missing from ISCOGroups
    assert groups["minor:251"]["n"] == 2
    assert groups["all"]["children"] == ["major:2", "major:3", "major:7"]


def test_stats_is_indented_and_quotes_the_threshold(workspace):
    run(workspace, "--date", "2026-09-18")
    text = (workspace / "site" / "stats.json").read_text(encoding="utf-8")
    assert text.startswith("{\n  ")
    stats = json.loads(text)
    assert stats == {
        "built": "2026-09-18", "threshold": 6, "occupations": 6, "skills_scored": 12,
        "quadrants": {
            "counts": {"EVOLVE": 1, "SHRINK": 2, "STABLE": 1, "TRANSFORM": 2},
            "shares": {"EVOLVE": 0.1667, "SHRINK": 0.3333, "STABLE": 0.1667,
                       "TRANSFORM": 0.3333},
        },
        "near_line": {"count": 3, "share": 0.5},
    }


def test_skill_files_invert_the_occupation_lists(workspace):
    run(workspace, "--date", "2026-09-18")
    rows = read(workspace, "skill_index.json")
    assert {r["id"] for r in rows} == {f"aa00000{c}" for c in "123456789abc"}
    notes = next(r for r in rows if r["id"] == "aa000006")
    assert (notes["ne"], notes["no"]) == (3, 0)
    assert read(workspace, "skill_occupations.json")["aa000001"]["e"] == ["software-developer"]


def test_two_runs_produce_byte_identical_files(workspace):
    run(workspace, "--date", "2026-09-18")
    first = {p.name: p.read_bytes() for p in (workspace / "site").glob("*.json")}
    run(workspace, "--date", "2026-09-18")
    second = {p.name: p.read_bytes() for p in (workspace / "site").glob("*.json")}
    assert first == second


def test_scorer_typesafe_reads_and_writes_suffixed_files(workspace):
    source = workspace / "site" / "portfolio_data.json"
    shutil.copy(source, workspace / "site" / "portfolio_data_typesafe.json")
    assert run(workspace, "--scorer", "typesafe", "--date", "2026-09-18") == 0
    for name in ("search_index", "groups", "stats", "skill_index", "skill_occupations"):
        assert (workspace / "site" / f"{name}_typesafe.json").exists()
        assert not (workspace / "site" / f"{name}.json").exists()


def test_report_fails_when_a_budget_is_missed(capsys, monkeypatch):
    monkeypatch.setitem(cli.ix.BUDGET_GZ_KB, "groups", 0)
    assert cli.report([("groups", 2048, 1024), ("skill_occupations", 10, 10)]) is False
    printed = capsys.readouterr().out
    assert "OVER BUDGET" in printed
    assert "lazy, no budget" in printed


def test_main_exits_non_zero_when_over_budget(workspace, monkeypatch):
    monkeypatch.setitem(cli.ix.BUDGET_GZ_KB, "skill_index", 0)
    assert run(workspace, "--date", "2026-09-18") == 1


def test_parse_args_defaults_to_the_published_scorer():
    args = cli.parse_args([])
    assert (args.scorer, args.site_dir, args.date) == ("gemini", "site", None)
    assert args.esco_dir == os.path.join("data", "esco")
