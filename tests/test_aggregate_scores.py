"""Tests for aggregate_scores.py.

The first tests are characterisation tests: they run main() over a small
synthetic dataset in a tmp directory and compare the whole output files and the
whole printed summary with golden copies, so any change in key order, rounding
or wording shows up as a failure.
"""

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

import aggregate_scores

FIXTURES = Path(__file__).parent / "fixtures" / "aggregation"
EXPECTED = FIXTURES / "expected"
INPUTS = ("esco_occupations.json", "esco_skills.json", "skill_scores.json",
          "skill_scores_typesafe.json")


def run_main(tmp_path, monkeypatch, argv):
    """Run aggregate_scores.main() over the fixture data inside tmp_path."""
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aggregate_scores.py", *argv])
    aggregate_scores.main()


def golden(name):
    """Text of a golden file captured from the behaviour we must keep."""
    return (EXPECTED / name).read_text(encoding="utf-8")


def test_gemini_run_writes_the_expected_files(tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, [])
    printed = capsys.readouterr().out

    assert (tmp_path / "data/occupation_scores.json").read_text(encoding="utf-8") == \
        golden("occupation_scores.json")
    assert (tmp_path / "data/site_data.json").read_text(encoding="utf-8") == \
        golden("site_data.json")
    assert printed == golden("aggregate_stdout.txt")


def test_gemini_run_copies_the_site_data_into_the_site_directory(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    published = tmp_path / "site/data.json"
    assert published.read_text(encoding="utf-8") == \
        (tmp_path / "data/site_data.json").read_text(encoding="utf-8")


def test_typesafe_run_writes_suffixed_files_and_leaves_the_gemini_ones_alone(
        tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, ["--scorer", "typesafe"])
    printed = capsys.readouterr().out

    assert (tmp_path / "data/occupation_scores_typesafe.json").read_text(encoding="utf-8") == \
        golden("occupation_scores_typesafe.json")
    assert (tmp_path / "data/site_data_typesafe.json").read_text(encoding="utf-8") == \
        golden("site_data_typesafe.json")
    assert (tmp_path / "site/data_typesafe.json").read_text(encoding="utf-8") == \
        (tmp_path / "data/site_data_typesafe.json").read_text(encoding="utf-8")
    assert printed == golden("aggregate_typesafe_stdout.txt")
    assert not (tmp_path / "data/occupation_scores.json").exists()
    assert not (tmp_path / "site/data.json").exists()


def test_scores_are_unicode_and_rounded_to_one_decimal(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    raw = (tmp_path / "data/site_data.json").read_text(encoding="utf-8")
    assert "Café manager" in raw
    site_data = json.loads(raw)
    by_title = {occ["title"]: occ for occ in site_data}
    assert by_title["Café manager"]["automation_risk"] == 3.6
    assert by_title["Café manager"]["quadrant"] == "EVOLVE"
    assert by_title["Café manager"]["slug"] == "caf-manager"


def test_occupations_without_a_single_scored_skill_are_dropped(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    titles = [occ["title"] for occ in
              json.loads((tmp_path / "data/occupation_scores.json").read_text(encoding="utf-8"))]
    assert "Unscored helper" not in titles
    assert "Ghost worker" not in titles
    assert "Data entry clerk" in titles


def test_site_data_merges_both_top_lists_to_at_most_ten_skills(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    site_data = json.loads((tmp_path / "data/site_data.json").read_text(encoding="utf-8"))
    clerk = next(occ for occ in site_data if occ["title"] == "Data entry clerk")
    assert len(clerk["top_skills"]) == 10
    assert [s["title"] for s in clerk["top_skills"][:3]] == \
        ["type documents", "scan invoices", "file records"]


def test_an_occupation_with_only_optional_skills_crashes_the_site_data(tmp_path, monkeypatch):
    # BUG: building the site data reads occ["essential_skills"] directly while every
    # other field is read with .get(), so an occupation that lists only optional
    # skills raises KeyError instead of counting zero essential skills. The real
    # ESCO export always carries both keys, so this never fires in production; the
    # current behaviour is pinned here rather than fixed, since a fix would change
    # which occupations reach the output.
    (tmp_path / "data").mkdir()
    skill = {"uri": "u1", "title": "a skill"}
    (tmp_path / "data/esco_occupations.json").write_text(json.dumps(
        [{"uri": "o1", "title": "Optional only", "isco_code": "1",
          "hierarchy": [], "optional_skills": [skill]}]), encoding="utf-8")
    (tmp_path / "data/esco_skills.json").write_text(json.dumps([skill]), encoding="utf-8")
    (tmp_path / "data/skill_scores.json").write_text(json.dumps(
        [{"uri": "u1", "title": "a skill", "automation_risk": 5.0,
          "amplification_potential": 5.0}]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aggregate_scores.py"])

    with pytest.raises(KeyError, match="essential_skills"):
        aggregate_scores.main()


def test_the_weights_and_the_quadrant_rule_stay_importable_from_here():
    """compare_skill_scores.py used to import these four names from this module."""
    assert aggregate_scores.ESSENTIAL_WEIGHT == 2.0
    assert aggregate_scores.OPTIONAL_WEIGHT == 1.0
    assert aggregate_scores.QUADRANT_THRESHOLD == 6
    assert aggregate_scores.assign_quadrant(9.0, 9.0) == "TRANSFORM"


def test_running_the_module_as_a_script_builds_the_same_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aggregate_scores.py"])

    runpy.run_module("aggregate_scores", run_name="__main__")

    assert capsys.readouterr().out == golden("aggregate_stdout.txt")
    assert (tmp_path / "site/data.json").exists()


def test_a_run_without_any_scored_skill_still_writes_empty_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    (tmp_path / "data/skill_scores.json").write_text("[]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aggregate_scores.py"])

    aggregate_scores.main()

    printed = capsys.readouterr().out
    assert json.loads((tmp_path / "data/occupation_scores.json").read_text()) == []
    assert "Aggregated scores for 0 occupations" in printed
    assert "  TRANSFORM   :    0 (  0.0%) " in printed
