"""Tests for compare_skill_scores.py.

All numbers here are synthetic: the fixtures are made-up skills with made-up
scores, so a test never depends on the real run (its results are in the README).

The first tests are characterisation tests: they run main() over that dataset in
a tmp directory and compare the whole printed report and the joined output file
with golden copies.
"""

import json
import runpy
import shutil
import sys
from pathlib import Path

import compare_skill_scores

FIXTURES = Path(__file__).parent / "fixtures" / "aggregation"
EXPECTED = FIXTURES / "expected"
DATA_INPUTS = ("esco_occupations.json", "esco_skills.json", "skill_scores_typesafe.json")


def workspace(tmp_path, monkeypatch, argv):
    """Lay out the files compare_skill_scores.py reads, relative to tmp_path."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "site").mkdir(exist_ok=True)
    for name in DATA_INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    shutil.copy(EXPECTED / "portfolio_data.json", tmp_path / "site/portfolio_data.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare_skill_scores.py", *argv])


def golden(name):
    """Text of a golden file captured from the behaviour we must keep."""
    return (EXPECTED / name).read_text(encoding="utf-8")


def test_default_run_prints_the_expected_report(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, [])

    compare_skill_scores.main()

    assert capsys.readouterr().out == golden("compare_stdout.txt")


def test_occupation_roll_up_and_joined_output(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch,
              ["--occupations", "--top", "3", "--out", "data/joined.json"])

    compare_skill_scores.main()

    assert capsys.readouterr().out == golden("compare_occupations_stdout.txt")
    assert (tmp_path / "data/joined.json").read_text(encoding="utf-8") == \
        golden("compare_joined.json")


def test_skills_are_matched_by_short_id_and_then_by_title(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, ["--out", "data/joined.json"])

    compare_skill_scores.main()
    capsys.readouterr()

    joined = json.loads((tmp_path / "data/joined.json").read_text(encoding="utf-8"))
    by_title = {row["title"]: row for row in joined}
    assert by_title["file records"]["gemini_automation"] == 8.0
    assert len([row for row in joined if row["title"] == "file records"]) == 2
    assert "plan budgets" in by_title
    assert "ghost skill" not in by_title
    assert "unscored task" not in by_title


def test_run_without_a_single_match_stops_after_the_header(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, [])
    (tmp_path / "data/skill_scores_typesafe.json").write_text(
        json.dumps([{"uri": "http://example.org/esco/skill/nope", "title": "nothing here",
                     "automation_risk": 5.0, "amplification_potential": 5.0}]),
        encoding="utf-8")

    compare_skill_scores.main()

    printed = capsys.readouterr().out
    assert printed == "TypeSafe skills: 1 | matched to published Gemini scores: 0\n"


def test_running_the_module_as_a_script_prints_the_same_report(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, [])

    runpy.run_module("compare_skill_scores", run_name="__main__")

    assert capsys.readouterr().out == golden("compare_stdout.txt")


def test_joined_output_keeps_the_gemini_rationale(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, ["--out", "data/joined.json"])

    compare_skill_scores.main()
    capsys.readouterr()

    joined = json.loads((tmp_path / "data/joined.json").read_text(encoding="utf-8"))
    typed = next(row for row in joined if row["title"] == "type documents")
    assert typed["gemini_rationale"] == "Templates and dictation cover most of it."
    assert typed["automation_confidence"] == 0.91
